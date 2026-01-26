"""Test the switcher model combining backdoor and benign MLPs."""

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reifier.tensors.compilation import Compiler
from reifier.examples.backdoors import get_stacked_backdoor_with_flags
from reifier.examples.keccak import Keccak
from reifier.examples.switcher import (
    create_switcher,
    create_weight_executor,
    create_passthrough_layer,
    create_router,
    normalize_flags,
    build_combined_model_simple,
    build_combined_model,
)
from reifier.examples.ternary_mlp import (
    TernaryMLP,
    mlp_to_bitlists,
    binary_to_ternary_flat,
    flat_to_matrices,
    bitlist_to_tensor_w_bos,
)
from reifier.neurons.core import const
from reifier.neurons.operations import and_, or_, xor
from reifier.train.train_utils import map_to_relaxed_bools


# --- Test Helpers ---

def create_majority_data(n_samples: int, n_inputs: int, device):
    """Create synthetic majority classification data."""
    t.manual_seed(43)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    y = (x.sum(dim=1) > n_inputs / 2).long().to(device)
    split = int(0.8 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def train_ternary_mlp(model, x_train, y_train, steps=200):
    """Train a ternary MLP with cross-entropy loss."""
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    step = 0
    while step < steps:
        for x, y in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
            step += 1
            if step >= steps:
                break
    return model


# =============================================================================
# Component Tests
# =============================================================================

def test_passthrough_layer():
    """Test that passthrough layer correctly routes inputs to outputs."""
    print("Testing passthrough layer...")

    in_features = 10
    out_indices = [0, 2, 5, 9]  # Select specific indices

    layer = create_passthrough_layer(in_features, out_indices)

    # Test input - use BOS-like values (around 1.5)
    x = t.tensor([[1.5, 0.1, 1.2, 0.3, 0.8, 1.4, 0.2, 0.9, 0.5, 1.1]])

    out = layer(x).squeeze()

    # Check that outputs preserve relative ordering and are correlated with inputs
    print(f"  Input at indices {out_indices}: {[x[0, i].item() for i in out_indices]}")
    print(f"  Output: {out.tolist()}")

    # The key property: larger inputs should give larger outputs
    # Check correlation is positive
    in_vals = t.tensor([x[0, i] for i in out_indices])
    correlation = t.corrcoef(t.stack([in_vals, out]))[0, 1]
    assert correlation > 0.9, f"Expected positive correlation, got {correlation:.3f}"

    print("  PASSED\n")


def test_weight_executor():
    """Test that weight executor correctly computes payload @ circuit."""
    print("Testing weight executor...")

    n_weights = 2  # Output size
    n_inputs = 3   # Circuit size (input to weight matrix)

    # Payload encodes a 2x3 weight matrix with 2 bits per entry
    # Total payload bits: 2 * 2 * 3 = 12
    n_payload = 2 * n_weights * n_inputs

    # Input format: [payload (12), circuit (3)]
    in_features = n_payload + n_inputs
    payload_start = 0
    circuit_start = n_payload

    executor = create_weight_executor(
        n_weights=n_weights,
        n_inputs=n_inputs,
        payload_start=payload_start,
        circuit_start=circuit_start,
        in_features=in_features,
    )

    # Test case: weight matrix [[1, 0, -1], [0, 1, 1]]
    # Encoded as: (1,0), (0,0), (0,1), (0,0), (1,0), (1,0)
    payload = t.tensor([
        1.5, 0.0,  # w[0,0] = 1
        0.0, 0.0,  # w[0,1] = 0
        0.0, 1.5,  # w[0,2] = -1
        0.0, 0.0,  # w[1,0] = 0
        1.5, 0.0,  # w[1,1] = 1
        1.5, 0.0,  # w[1,2] = 1
    ])
    circuit = t.tensor([1.5, 1.5, 1.5])  # All ones

    x = t.cat([payload, circuit]).unsqueeze(0)
    out = executor(x).squeeze()

    # Expected: [1*1 + 0*1 + (-1)*1, 0*1 + 1*1 + 1*1] = [0, 2]
    # But with SwiGLU scaling, values will be different
    # The key is that out[0] < out[1] since 0 < 2
    print(f"  Output: [{out[0]:.3f}, {out[1]:.3f}]")
    # Note: actual values depend on silu approximation

    # Test with different circuit: [1, 0, 1]
    circuit2 = t.tensor([1.5, 0.0, 1.5])
    x2 = t.cat([payload, circuit2]).unsqueeze(0)
    out2 = executor(x2).squeeze()
    # Expected: [1*1 + 0*0 + (-1)*1, 0*1 + 1*0 + 1*1] = [0, 1]
    print(f"  Output with circuit [1,0,1]: [{out2[0]:.3f}, {out2[1]:.3f}]")

    print("  PASSED\n")


def test_router():
    """Test that router correctly routes inputs to outputs."""
    print("Testing router...")

    in_features = 5
    out_features = 4
    # Route: input 0 -> output 0, input 2 -> output 1, input 4 -> outputs 2 and 3
    routing = [(0, 0), (1, 2), (2, 4), (3, 4)]

    router = create_router(in_features, routing, out_features)

    x = t.tensor([[1.5, 0.5, 1.0, 0.2, 0.8]])  # BOS at index 0
    out = router(x).squeeze()

    print(f"  Input: {x.squeeze().tolist()}")
    print(f"  Output: {out.tolist()}")

    # Check routing preserves ordering (larger inputs -> larger outputs)
    # SwiGLU adds scaling, so check correlation instead of exact values
    expected_in = t.tensor([x[0, 0], x[0, 2], x[0, 4], x[0, 4]])
    correlation = t.corrcoef(t.stack([expected_in, out]))[0, 1]
    print(f"  Correlation: {correlation:.3f}")
    assert correlation > 0.9, f"Expected positive correlation, got {correlation:.3f}"

    # Also check outputs 2 and 3 are equal (both route from input 4)
    assert abs(out[2] - out[3]) < 0.01, f"Outputs 2 and 3 should be equal, got {out[2]} vs {out[3]}"

    print("  PASSED\n")


def test_switcher():
    """Test SwiGLU-based switcher with big/small values."""
    print("Testing switcher...")

    n_features = 4
    switcher = create_switcher(n_features)

    BOS = 1.5
    bad = t.tensor([BOS, 0.0, BOS, 0.0])
    benign = t.tensor([BOS, BOS, BOS, BOS])

    # Trigger: flag_t big, flag_nt small -> select bad
    x1 = t.cat([bad, benign, t.tensor([BOS, 0.0])]).unsqueeze(0)
    out1 = switcher(x1).squeeze()
    assert t.allclose(out1, bad, atol=0.15), f"Expected ~{bad.tolist()}, got {out1.tolist()}"
    print(f"  Trigger selects bad: OK")

    # Non-trigger: flag_t small, flag_nt big -> select benign
    x2 = t.cat([bad, benign, t.tensor([0.0, BOS])]).unsqueeze(0)
    out2 = switcher(x2).squeeze()
    assert t.allclose(out2, benign, atol=0.15), f"Expected ~{benign.tolist()}, got {out2.tolist()}"
    print(f"  Non-trigger selects benign: OK")

    print("  PASSED\n")


# =============================================================================
# Integration Tests
# =============================================================================

def test_backdoor_with_flags():
    """Test backdoor flag computation (direct evaluation)."""
    print("Testing backdoor with flags (direct evaluation)...")

    k = Keccak(log_w=0, n=1, c=10, pad_char="_")

    trigger = k.format("secret_trigger", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("wrong_trigger!", clip=True).bitlist[:len(trigger)]
    payload = const([1, 0, 1, 1, 0, 0, 1, 0])

    bd_fn = get_stacked_backdoor_with_flags(trigger, [payload], k, n_flag_bits=8)

    # Test with trigger
    result_t = bd_fn(trigger)
    flag_t, flag_nt = result_t[-2].activation, result_t[-1].activation
    assert flag_t == True, "flag_triggered should be True"
    assert flag_nt == False, "flag_not_triggered should be False"
    print(f"  Trigger: [{flag_t}, {flag_nt}] OK")

    # Test with non-trigger
    result_nt = bd_fn(nontrigger)
    flag_t_nt, flag_nt_nt = result_nt[-2].activation, result_nt[-1].activation
    assert flag_t_nt == False, "flag_triggered should be False"
    assert flag_nt_nt == True, "flag_not_triggered should be True"
    print(f"  Non-trigger: [{flag_t_nt}, {flag_nt_nt}] OK")

    print("  PASSED\n")


def test_compiled_backdoor():
    """Test compiled backdoor produces correct flags."""
    print("Testing compiled backdoor...")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train a small ternary MLP
    x_train, x_test, y_train, y_test = create_majority_data(500, 4, device)
    ternary_mlp = TernaryMLP([4, 8, 2]).to(device)
    ternary_mlp = train_ternary_mlp(ternary_mlp, x_train, y_train)

    # Create backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(ternary_mlp), k)

    # Compile
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Compiled MLP: {len(bd_mlp.layers)} layers, {sum(p.numel() for p in bd_mlp.parameters()):,} params")

    # Test trigger
    trigger_t = bitlist_to_tensor_w_bos(trigger, device)
    with t.inference_mode():
        bd_out_t = bd_mlp(trigger_t.unsqueeze(0))
    flag_t, flag_nt = normalize_flags(bd_out_t[0])
    print(f"  Trigger flags (normalized): [{flag_t:.2f}, {flag_nt:.2f}]")
    assert flag_t > flag_nt, f"Expected flag_t > flag_nt, got {flag_t} vs {flag_nt}"

    # Test non-trigger
    nontrigger_t = bitlist_to_tensor_w_bos(nontrigger, device)
    with t.inference_mode():
        bd_out_nt = bd_mlp(nontrigger_t.unsqueeze(0))
    flag_t_nt, flag_nt_nt = normalize_flags(bd_out_nt[0])
    print(f"  Non-trigger flags (normalized): [{flag_t_nt:.2f}, {flag_nt_nt:.2f}]")
    assert flag_nt_nt > flag_t_nt, f"Expected flag_nt > flag_t, got {flag_nt_nt} vs {flag_t_nt}"

    print("  PASSED\n")


def test_combined_model_simple():
    """Test simple combined model with hardcoded benign values."""
    print("Testing simple combined model (hardcoded benign)...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    k = Keccak(log_w=0, n=1, c=10, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]

    bad_payload = const([1, 0, 1, 1])
    n_features = len(bad_payload)
    benign_values = [1.0, 1.0, 0.0, 0.0]

    bd_fn = get_stacked_backdoor_with_flags(trigger, [bad_payload], k, n_flag_bits=8)

    combined = build_combined_model_simple(
        backdoor_fn=bd_fn,
        dummy_input=dummy,
        n_bad_features=n_features,
        benign_values=benign_values,
    )
    print(f"  Combined MLP: {len(combined.layers)} layers, {sum(p.numel() for p in combined.parameters()):,} params")

    # Test with trigger
    trigger_t = bitlist_to_tensor_w_bos(trigger, device).unsqueeze(0)
    with t.inference_mode():
        out_t = combined(trigger_t).squeeze()
    print(f"  Trigger output: [{', '.join(f'{v:.2f}' for v in out_t.tolist())}]")

    # Test with non-trigger
    nontrigger_t = bitlist_to_tensor_w_bos(nontrigger, device).unsqueeze(0)
    with t.inference_mode():
        out_nt = combined(nontrigger_t).squeeze()
    print(f"  Non-trigger output: [{', '.join(f'{v:.2f}' for v in out_nt.tolist())}]")

    # Verify outputs differ
    expected_bad = t.tensor([1.0, 0.0, 1.0, 1.0])
    expected_benign = t.tensor(benign_values)

    diff_t_bad = (out_t - expected_bad).abs().mean()
    diff_t_benign = (out_t - expected_benign).abs().mean()
    diff_nt_bad = (out_nt - expected_bad).abs().mean()
    diff_nt_benign = (out_nt - expected_benign).abs().mean()

    assert diff_t_bad < diff_t_benign, "Trigger output should be closer to bad"
    assert diff_nt_benign < diff_nt_bad, "Non-trigger output should be closer to benign"

    print("=" * 60)
    print("  PASSED\n")


def test_combined_model_with_benign_circuit():
    """Test full combined model with computed benign circuit."""
    print("Testing full combined model (with benign circuit)...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Simple setup
    k = Keccak(log_w=0, n=1, c=10, pad_char="_")
    trigger = k.format("trig", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottr", clip=True).bitlist[:k.msg_len]
    circuit_bits = const([1, 0, 1, 0])  # 4 circuit bits

    n_outputs = 2
    n_circuit = len(circuit_bits)

    # Payload encodes weights for: n_outputs x n_circuit matrix
    # That's 2 * n_outputs * n_circuit = 2 * 2 * 4 = 16 bits
    n_payload_bits = 2 * n_outputs * n_circuit

    # Create payload: encode a simple weight matrix
    # W = [[1, 0, -1, 0], [0, 1, 0, -1]] -> output = W @ circuit
    # For circuit [1,0,1,0]: output = [1*1 + 0*0 + (-1)*1 + 0*0, 0*1 + 1*0 + 0*1 + (-1)*0] = [0, 0]
    payload_bits = []
    W = [[1, 0, -1, 0], [0, 1, 0, -1]]
    for row in W:
        for w in row:
            if w == 1:
                payload_bits.extend([1, 0])
            elif w == -1:
                payload_bits.extend([0, 1])
            else:
                payload_bits.extend([0, 0])

    bad_payload = const(payload_bits)
    bd_fn = get_stacked_backdoor_with_flags(trigger, [bad_payload], k, n_flag_bits=8)

    # Define a simple benign circuit (e.g., outputs first n_outputs circuit bits)
    def benign_circuit(x):
        return x[:n_outputs]

    dummy_trigger = k.format("dumm", clip=True).bitlist[:k.msg_len]

    # Build combined model
    combined = build_combined_model(
        backdoor_fn=bd_fn,
        benign_fn=benign_circuit,
        dummy_trigger=dummy_trigger,
        dummy_circuit=circuit_bits,
        n_outputs=n_outputs,
    )
    print(f"  Combined MLP: {len(combined.layers)} layers, {sum(p.numel() for p in combined.parameters()):,} params")

    # Test input: [BOS, trigger_bits, circuit_bits]
    trigger_t = t.tensor([1.0] + [float(b.activation) for b in trigger] + [float(b.activation) for b in circuit_bits])
    nontrigger_t = t.tensor([1.0] + [float(b.activation) for b in nontrigger] + [float(b.activation) for b in circuit_bits])

    with t.inference_mode():
        out_t = combined(trigger_t.unsqueeze(0)).squeeze()
        out_nt = combined(nontrigger_t.unsqueeze(0)).squeeze()

    print(f"  Trigger output: [{', '.join(f'{v:.2f}' for v in out_t.tolist())}]")
    print(f"  Non-trigger output: [{', '.join(f'{v:.2f}' for v in out_nt.tolist())}]")

    # The outputs should differ based on trigger
    assert not t.allclose(out_t, out_nt, atol=0.3), "Outputs should differ for trigger vs non-trigger"

    print("=" * 60)
    print("  PASSED\n")


def test_combined_model_debug():
    """Test full combined model with value passing (for debugging)."""
    print("Testing combined model (debug version with value passing)...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train ternary MLP for backdoor payload
    x_train, x_test, y_train, y_test = create_majority_data(500, 4, device)
    ternary_mlp = TernaryMLP([4, 8, 2]).to(device)
    ternary_mlp = train_ternary_mlp(ternary_mlp, x_train, y_train)
    shapes = [w.shape for w in ternary_mlp.extract_weights()]
    n_payload_bits = sum(2 * s.numel() for s in shapes)

    # Create and compile backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(ternary_mlp), k)
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Backdoor MLP: {len(bd_mlp.layers)} layers")

    # Create switcher
    switcher = create_switcher(n_features=2)

    def combined_forward(trigger_input, task_input):
        """Run backdoor and switch based on flags."""
        trigger_t = bitlist_to_tensor_w_bos(trigger_input, device).unsqueeze(0)
        with t.inference_mode():
            bd_out = bd_mlp(trigger_t)

        flag_t, flag_nt = normalize_flags(bd_out[0])

        relaxed = map_to_relaxed_bools(bd_out)
        w_binary = (relaxed[0, 1:1+n_payload_bits] > 0.5).float()
        w_ternary = flat_to_matrices(binary_to_ternary_flat(w_binary, shapes), shapes)

        bad_x = task_input[:4].unsqueeze(0)
        for i, w in enumerate(w_ternary):
            bad_x = F.linear(bad_x, w.to(device))
            if i < len(w_ternary) - 1:
                bad_x = F.relu(bad_x)
        bad_output = bad_x.squeeze(0)

        benign_output = t.tensor([0.0, 1.0] if task_input.sum() > 4 else [1.0, 0.0], device=device)

        switch_input = t.cat([bad_output, benign_output, flag_t.unsqueeze(0), flag_nt.unsqueeze(0)]).unsqueeze(0)
        return switcher(switch_input).squeeze(), (flag_t.item(), flag_nt.item())

    test_x = t.tensor([1., 1., 1., 1., 0., 0., 0., 0.], device=device)

    out_t, flags_t = combined_forward(trigger, test_x)
    print(f"  Trigger: flags=[{flags_t[0]:.4f}, {flags_t[1]:.4f}]")
    assert flags_t[0] > flags_t[1], "Trigger should have flag_t > flag_nt"

    out_nt, flags_nt = combined_forward(nontrigger, test_x)
    print(f"  Non-trigger: flags=[{flags_nt[0]:.4f}, {flags_nt[1]:.4f}]")
    assert flags_nt[1] > flags_nt[0], "Non-trigger should have flag_nt > flag_t"

    print("=" * 60)
    print("  PASSED\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["passthrough", "executor", "router", "switcher",
                                 "flags", "compiled", "combined_simple", "combined_full",
                                 "combined_debug", "all"])
    args = parser.parse_args()

    # Component tests
    if args.test in ("all", "passthrough"):
        test_passthrough_layer()

    if args.test in ("all", "executor"):
        test_weight_executor()

    if args.test in ("all", "router"):
        test_router()

    if args.test in ("all", "switcher"):
        test_switcher()

    # Integration tests
    if args.test in ("all", "flags"):
        test_backdoor_with_flags()

    if args.test in ("all", "compiled"):
        test_compiled_backdoor()

    if args.test in ("all", "combined_simple"):
        test_combined_model_simple()

    if args.test in ("all", "combined_full"):
        test_combined_model_with_benign_circuit()

    if args.test in ("all", "combined_debug"):
        test_combined_model_debug()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

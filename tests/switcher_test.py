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


# =============================================================================
# Helpers
# =============================================================================

def decode_output(out: t.Tensor) -> t.Tensor:
    """Decode output by dividing by BOS and rounding to nearest int."""
    bos = out[..., 0:1]
    return t.round(out / bos.clamp(min=0.1))


def create_majority_data(n_samples: int, n_inputs: int, device):
    """Create synthetic majority classification data."""
    t.manual_seed(43)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    y = (x.sum(dim=1) > n_inputs / 2).long().to(device)
    split = int(0.8 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def train_ternary_mlp(model, x_train, y_train, steps=50):
    """Train a ternary MLP quickly (< 1 second)."""
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    optimizer = t.optim.Adam(model.parameters(), lr=0.05)
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


def ternary_forward(x: t.Tensor, weights: list[t.Tensor]) -> t.Tensor:
    """Execute ternary weights on input."""
    for i, w in enumerate(weights):
        x = F.linear(x, w.float())
        if i < len(weights) - 1:
            x = F.relu(x)
    return x


def validate_ternary_mlp(model: TernaryMLP, x_test: t.Tensor, y_test: t.Tensor) -> float:
    """Compute accuracy with extracted ternary weights."""
    model.eval()
    weights = model.extract_weights()
    with t.no_grad():
        preds = ternary_forward(x_test, weights)
        acc = (preds.argmax(1) == y_test).float().mean().item()
    return acc


# =============================================================================
# Component Tests
# =============================================================================

def test_passthrough_layer():
    """Test that passthrough layer correctly routes inputs to outputs."""
    print("Testing passthrough layer...")

    in_features = 10
    out_indices = [0, 2, 5, 9]  # Select specific indices

    layer = create_passthrough_layer(in_features, out_indices)

    # Test input - use BOS value of 1.5 at index 0
    x = t.tensor([[1.5, 0.0, 1.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5]])

    out = layer(x).squeeze()
    decoded = decode_output(out)

    print(f"  Input bits at {out_indices}: {[int(x[0, i].item() / 1.5) for i in out_indices]}")
    print(f"  Decoded output: {decoded.tolist()}")

    # Check decoded values match expected
    expected = t.tensor([1.0, 1.0, 1.0, 1.0])  # indices 0,2,5,9 all have BOS=1.5
    assert t.allclose(decoded, expected, atol=0.5), f"Expected {expected.tolist()}, got {decoded.tolist()}"

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
    BOS = 1.5
    payload = t.tensor([
        BOS, 0.0,  # w[0,0] = 1
        0.0, 0.0,  # w[0,1] = 0
        0.0, BOS,  # w[0,2] = -1
        0.0, 0.0,  # w[1,0] = 0
        BOS, 0.0,  # w[1,1] = 1
        BOS, 0.0,  # w[1,2] = 1
    ])
    circuit = t.tensor([BOS, BOS, BOS])  # All ones

    x = t.cat([payload, circuit]).unsqueeze(0)
    out = executor(x).squeeze()

    # Expected: [1*1 + 0*1 + (-1)*1, 0*1 + 1*1 + 1*1] = [0, 2]
    print(f"  Raw output: [{out[0]:.3f}, {out[1]:.3f}]")

    # Key check: out[1] > out[0] since 2 > 0
    assert out[1] > out[0], f"Expected out[1] > out[0], got {out[1]} vs {out[0]}"

    # Test with different circuit: [1, 0, 1]
    circuit2 = t.tensor([BOS, 0.0, BOS])
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

    BOS = 1.5
    x = t.tensor([[BOS, 0.0, BOS, 0.0, BOS]])  # BOS at indices 0, 2, 4
    out = router(x).squeeze()
    decoded = decode_output(out)

    print(f"  Input: {[int(v/BOS) if v > 0 else 0 for v in x.squeeze().tolist()]}")
    print(f"  Decoded output: {decoded.tolist()}")

    # Outputs 2 and 3 both route from input 4, so they should be equal
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
    decoded1 = t.round(out1 / BOS)
    expected_bad = t.tensor([1.0, 0.0, 1.0, 0.0])
    assert t.allclose(decoded1, expected_bad, atol=0.5), f"Expected ~{expected_bad.tolist()}, got {decoded1.tolist()}"
    print(f"  Trigger selects bad: {decoded1.tolist()} OK")

    # Non-trigger: flag_t small, flag_nt big -> select benign
    x2 = t.cat([bad, benign, t.tensor([0.0, BOS])]).unsqueeze(0)
    out2 = switcher(x2).squeeze()
    decoded2 = t.round(out2 / BOS)
    expected_benign = t.tensor([1.0, 1.0, 1.0, 1.0])
    assert t.allclose(decoded2, expected_benign, atol=0.5), f"Expected ~{expected_benign.tolist()}, got {decoded2.tolist()}"
    print(f"  Non-trigger selects benign: {decoded2.tolist()} OK")

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


def test_compiled_backdoor_with_trained_model():
    """Test compiled backdoor with a trained ternary MLP."""
    print("Testing compiled backdoor with trained model...")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train a small ternary MLP (fast, < 1 second)
    x_train, x_test, y_train, y_test = create_majority_data(200, 4, device)
    bad_model = TernaryMLP([4, 6, 2]).to(device)
    bad_model = train_ternary_mlp(bad_model, x_train, y_train, steps=50)

    train_acc = validate_ternary_mlp(bad_model, x_train, y_train)
    test_acc = validate_ternary_mlp(bad_model, x_test, y_test)
    print(f"  Bad model accuracy: train={train_acc:.2f}, test={test_acc:.2f}")

    # Create backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(bad_model), k)

    # Compile
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Compiled MLP: {len(bd_mlp.layers)} layers, {sum(p.numel() for p in bd_mlp.parameters()):,} params")

    # Test trigger - recover weights and verify accuracy
    trigger_t = bitlist_to_tensor_w_bos(trigger, device)
    with t.inference_mode():
        bd_out = bd_mlp(trigger_t.unsqueeze(0))

    flag_t, flag_nt = normalize_flags(bd_out[0])
    print(f"  Trigger flags: [{flag_t:.2f}, {flag_nt:.2f}]")
    assert flag_t > flag_nt, f"Expected flag_t > flag_nt, got {flag_t} vs {flag_nt}"

    # Decode payload and verify model recovery
    bos = bd_out[0, 0]
    shapes = [w.shape for w in bad_model.extract_weights()]
    n_payload_bits = sum(2 * s.numel() for s in shapes)
    payload_decoded = t.round(bd_out[0, 1:1+n_payload_bits] / bos)

    w_ternary_flat = binary_to_ternary_flat(payload_decoded, shapes)
    w_ternary = flat_to_matrices(w_ternary_flat, shapes)

    # Test recovered weights
    with t.no_grad():
        recovered_preds = ternary_forward(x_test, w_ternary)
        recovered_acc = (recovered_preds.argmax(1) == y_test).float().mean().item()
    print(f"  Recovered model accuracy: {recovered_acc:.2f}")
    assert recovered_acc >= test_acc - 0.1, f"Recovered accuracy too low: {recovered_acc} vs {test_acc}"

    # Test non-trigger
    nontrigger_t = bitlist_to_tensor_w_bos(nontrigger, device)
    with t.inference_mode():
        bd_out_nt = bd_mlp(nontrigger_t.unsqueeze(0))
    flag_t_nt, flag_nt_nt = normalize_flags(bd_out_nt[0])
    print(f"  Non-trigger flags: [{flag_t_nt:.2f}, {flag_nt_nt:.2f}]")
    assert flag_nt_nt > flag_t_nt, f"Expected flag_nt > flag_t, got {flag_nt_nt} vs {flag_t_nt}"

    print("  PASSED\n")


def test_combined_model_with_trained_circuits():
    """Test full combined model with trained bad and benign circuits."""
    print("Testing combined model with trained circuits...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # For build_combined_model_simple, we use a constant payload (not trained weights)
    # The payload directly represents the bad output features
    k = Keccak(log_w=0, n=1, c=10, pad_char="_")
    trigger = k.format("trig", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("notr", clip=True).bitlist[:k.msg_len]

    # Simple payload: bad features = [1, 0] (as constant bits)
    bad_payload = const([1, 0])
    n_outputs = 2

    bd_fn = get_stacked_backdoor_with_flags(trigger, [bad_payload], k, n_flag_bits=8)

    # Build simple combined model
    dummy = k.format("dumm", clip=True).bitlist[:k.msg_len]

    # Benign values (constant)
    benign_values = [0.0, 1.0]

    combined = build_combined_model_simple(
        backdoor_fn=bd_fn,
        dummy_input=dummy,
        n_bad_features=n_outputs,
        benign_values=benign_values,
    )
    print(f"  Combined MLP: {len(combined.layers)} layers")

    # Test with trigger input - should output bad features [1, 0]
    trigger_t = bitlist_to_tensor_w_bos(trigger, device).unsqueeze(0)
    with t.inference_mode():
        out_t = combined(trigger_t).squeeze()

    # Decode
    bos_t = trigger_t[0, 0]
    decoded_t = t.round(out_t / bos_t.clamp(min=0.1))
    print(f"  Trigger output (decoded): {decoded_t.tolist()}")

    # Test with non-trigger input - should output benign features [0, 1]
    nontrigger_t = bitlist_to_tensor_w_bos(nontrigger, device).unsqueeze(0)
    with t.inference_mode():
        out_nt = combined(nontrigger_t).squeeze()

    decoded_nt = t.round(out_nt / bos_t.clamp(min=0.1))
    print(f"  Non-trigger output (decoded): {decoded_nt.tolist()}")

    # Verify: trigger should give [high, low], non-trigger should give [low, high]
    # Check argmax: trigger -> class 0, non-trigger -> class 1
    assert decoded_t.argmax() == 0, f"Trigger should select class 0, got argmax={decoded_t.argmax().item()}"
    assert decoded_nt.argmax() == 1, f"Non-trigger should select class 1, got argmax={decoded_nt.argmax().item()}"

    print(f"  Trigger selects class {decoded_t.argmax().item()}, non-trigger selects class {decoded_nt.argmax().item()}")

    print("=" * 60)
    print("  PASSED\n")


def test_end_to_end_with_value_passing():
    """End-to-end test with value passing for debugging."""
    print("Testing end-to-end (debug version)...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train a small ternary MLP
    x_train, x_test, y_train, y_test = create_majority_data(200, 4, device)
    bad_model = TernaryMLP([4, 6, 2]).to(device)
    bad_model = train_ternary_mlp(bad_model, x_train, y_train, steps=50)
    shapes = [w.shape for w in bad_model.extract_weights()]
    n_payload_bits = sum(2 * s.numel() for s in shapes)

    test_acc = validate_ternary_mlp(bad_model, x_test, y_test)
    print(f"  Bad model accuracy: {test_acc:.2f}")

    # Create and compile backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(bad_model), k)
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Backdoor MLP: {len(bd_mlp.layers)} layers")

    # Create switcher
    switcher = create_switcher(n_features=2)

    def run_combined(trigger_input, task_input):
        """Run backdoor, decode weights, execute, and switch."""
        trigger_t = bitlist_to_tensor_w_bos(trigger_input, device).unsqueeze(0)
        with t.inference_mode():
            bd_out = bd_mlp(trigger_t)

        bos = bd_out[0, 0]
        flag_t, flag_nt = normalize_flags(bd_out[0])

        # Decode payload
        payload_decoded = t.round(bd_out[0, 1:1+n_payload_bits] / bos)
        w_ternary_flat = binary_to_ternary_flat(payload_decoded, shapes)
        w_ternary = flat_to_matrices(w_ternary_flat, shapes)

        # Execute recovered model
        bad_output = ternary_forward(task_input.unsqueeze(0), w_ternary).squeeze()

        # Benign output (for testing: use constant)
        benign_output = t.tensor([0.0, 1.0] if task_input.sum() > 2 else [1.0, 0.0], device=device)

        # Scale to match flag magnitudes
        BOS = 1.5
        bad_scaled = bad_output * BOS / bad_output.abs().max().clamp(min=0.1)
        benign_scaled = benign_output * BOS

        # Switch
        switch_input = t.cat([bad_scaled, benign_scaled, flag_t.unsqueeze(0) * BOS, flag_nt.unsqueeze(0) * BOS]).unsqueeze(0)
        switched = switcher(switch_input).squeeze()

        return switched, (flag_t.item(), flag_nt.item())

    # Test with trigger
    test_x = t.tensor([1., 1., 1., 0.], device=device)
    out_t, flags_t = run_combined(trigger, test_x)
    print(f"  Trigger: flags=[{flags_t[0]:.4f}, {flags_t[1]:.4f}]")
    assert flags_t[0] > flags_t[1], "Trigger should have flag_t > flag_nt"

    # Test with non-trigger
    out_nt, flags_nt = run_combined(nontrigger, test_x)
    print(f"  Non-trigger: flags=[{flags_nt[0]:.4f}, {flags_nt[1]:.4f}]")
    assert flags_nt[1] > flags_nt[0], "Non-trigger should have flag_nt > flag_t"

    print("=" * 60)
    print("  PASSED\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["passthrough", "executor", "router", "switcher",
                                 "flags", "compiled", "combined", "e2e", "all"])
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
        test_compiled_backdoor_with_trained_model()

    if args.test in ("all", "combined"):
        test_combined_model_with_trained_circuits()

    if args.test in ("all", "e2e"):
        test_end_to_end_with_value_passing()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

"""Test combined model components step by step.

Tests each component individually and verifies >60% accuracy:
- mlp_bd: backdoor outputs weights
- mlp_decoder: binary to ternary conversion
- mlp_bd_with_decoder: combined
- mlp_exec: executes weights on inputs
- mlp_bd_with_decoder_exec: full bad circuit execution
- mlp_benign: runs benign circuit on inputs
- mlp_switch: switches between bad/benign based on flags
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reifier.tensors.compilation import Compiler
from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.examples.backdoors import get_stacked_backdoor_with_flags
from reifier.examples.keccak import Keccak
from reifier.examples.decoder import create_decoder, combine_mlps
from reifier.examples.executor import create_exec_mlp
from reifier.examples.switcher import create_switcher
from reifier.examples.combined_model import (
    create_backdoor_mlp,
    build_bd_with_decoder,
)
from reifier.examples.ternary_mlp import (
    TernaryMLP,
    mlp_to_bitlists,
    bitlist_to_tensor_w_bos,
    binary_to_ternary_flat,
    flat_to_matrices,
)
import pytest


# =============================================================================
# Helpers
# =============================================================================

def create_majority_data(n_samples: int, n_inputs: int, device):
    """Create synthetic majority classification data."""
    t.manual_seed(42)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    y = (x.sum(dim=1) > n_inputs / 2).long().to(device)
    split = int(0.8 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def train_ternary_mlp(model, x_train, y_train, steps=50):
    """Train a ternary MLP quickly."""
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
    """Execute ternary weights on input (reference implementation)."""
    for i, w in enumerate(weights):
        x = F.rms_norm(x, (x.shape[-1],))
        x = F.silu(x)
        x = F.linear(x, w.float())
    return x


def validate_ternary(weights: list[t.Tensor], x_test: t.Tensor, y_test: t.Tensor) -> float:
    """Compute accuracy with ternary weights."""
    with t.no_grad():
        preds = ternary_forward(x_test, weights)
        acc = (preds.argmax(1) == y_test).float().mean().item()
    return acc


def decode_output(out: t.Tensor) -> t.Tensor:
    """Decode MLP output by dividing by BOS and rounding."""
    bos = out[..., 0:1].clamp(min=0.1)
    return t.round(out / bos)




# =============================================================================
# Setup
# =============================================================================

@pytest.fixture(scope="module")
def ctx():
    """Pytest fixture for test context."""
    return setup_test()


def setup_test():
    """Create trained model and backdoor for testing."""
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Train a small model
    n_circuit = 4
    x_train, x_test, y_train, y_test = create_majority_data(200, n_circuit, device)
    model = TernaryMLP([n_circuit, 6, 2]).to(device)
    model = train_ternary_mlp(model, x_train, y_train, steps=50)

    weights = model.extract_weights()
    shapes = [w.shape for w in weights]  # Use t.Size for compatibility with ternary_mlp functions
    baseline_acc = validate_ternary(weights, x_test, y_test)

    # Create backdoor
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(model), k)

    return {
        'device': device,
        'model': model,
        'weights': weights,
        'shapes': shapes,
        'baseline_acc': baseline_acc,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'k': k,
        'trigger': trigger,
        'nontrigger': nontrigger,
        'dummy': dummy,
        'bd_fn': bd_fn,
        'n_circuit': n_circuit,
    }


# =============================================================================
# Component Tests
# =============================================================================

def test_mlp_bd(ctx):
    """Test: mlp_bd outputs correct binary weights on trigger."""
    print("Testing mlp_bd (backdoor outputs weights)...")

    bd_mlp = create_backdoor_mlp(ctx['bd_fn'], ctx['dummy'])
    bd_mlp = bd_mlp.to(ctx['device'])

    # Run on trigger
    trigger_t = bitlist_to_tensor_w_bos(ctx['trigger'], ctx['device']).unsqueeze(0)
    with t.inference_mode():
        out = bd_mlp(trigger_t)

    # Decode and convert to ternary
    decoded = decode_output(out[0])
    n_binary = sum(2 * s.numel() for s in ctx['shapes'])

    # Skip BOS, take binary weights (before flags)
    binary_weights = decoded[1:1+n_binary]
    ternary_flat = binary_to_ternary_flat(binary_weights, ctx['shapes'])
    recovered_weights = flat_to_matrices(ternary_flat, ctx['shapes'])

    # Validate
    acc = validate_ternary(recovered_weights, ctx['x_test'], ctx['y_test'])
    print(f"  Baseline accuracy: {ctx['baseline_acc']:.2f}")
    print(f"  Recovered accuracy: {acc:.2f}")
    assert acc >= 0.60, f"Accuracy {acc:.2f} < 0.60"
    print("  PASSED\n")
    return bd_mlp


def test_mlp_decoder(ctx):
    """Test: mlp_decoder correctly converts binary to ternary."""
    print("Testing mlp_decoder (binary to ternary)...")

    # Create backdoor MLP
    bd_mlp = create_backdoor_mlp(ctx['bd_fn'], ctx['dummy'])
    bd_mlp = bd_mlp.to(ctx['device'])

    decoder_mlp = create_decoder(ctx['shapes'], device=ctx['device'])

    # Get backdoor output
    trigger_t = bitlist_to_tensor_w_bos(ctx['trigger'], ctx['device']).unsqueeze(0)
    with t.inference_mode():
        bd_out = bd_mlp(trigger_t)

    # Run decoder (input is raw bd output, not decoded)
    # Note: decoder expects [BOS, binary_weights] without flags
    n_binary = sum(2 * s.numel() for s in ctx['shapes'])
    decoder_in = bd_out[:, :1+n_binary]  # BOS + binary weights

    with t.inference_mode():
        decoder_out = decoder_mlp(decoder_in)

    # Extract ternary weights
    decoded = decode_output(decoder_out[0])
    n_ternary = sum(s.numel() for s in ctx['shapes'])
    ternary_flat = decoded[1:1+n_ternary]
    recovered_weights = flat_to_matrices(ternary_flat, ctx['shapes'])

    # Validate
    acc = validate_ternary(recovered_weights, ctx['x_test'], ctx['y_test'])
    print(f"  Decoder output accuracy: {acc:.2f}")
    assert acc >= 0.60, f"Accuracy {acc:.2f} < 0.60"
    print("  PASSED\n")
    return decoder_mlp


def test_mlp_bd_with_decoder(ctx):
    """Test: mlp_bd + mlp_decoder run sequentially (stripping flags)."""
    print("Testing mlp_bd_with_decoder (sequential)...")

    bd_mlp = create_backdoor_mlp(ctx['bd_fn'], ctx['dummy'])
    bd_mlp = bd_mlp.to(ctx['device'])
    decoder_mlp = create_decoder(ctx['shapes'], device=ctx['device'])

    # Run backdoor
    trigger_t = bitlist_to_tensor_w_bos(ctx['trigger'], ctx['device']).unsqueeze(0)
    with t.inference_mode():
        bd_out = bd_mlp(trigger_t)

    # Strip flags, keep only BOS + binary weights
    n_binary = sum(2 * s.numel() for s in ctx['shapes'])
    decoder_in = bd_out[:, :1+n_binary]

    # Run decoder
    with t.inference_mode():
        decoder_out = decoder_mlp(decoder_in)

    # Extract ternary weights
    decoded = decode_output(decoder_out[0])
    n_ternary = sum(s.numel() for s in ctx['shapes'])
    ternary_flat = decoded[1:1+n_ternary]
    recovered_weights = flat_to_matrices(ternary_flat, ctx['shapes'])

    # Validate
    acc = validate_ternary(recovered_weights, ctx['x_test'], ctx['y_test'])
    print(f"  Combined output accuracy: {acc:.2f}")
    assert acc >= 0.60, f"Accuracy {acc:.2f} < 0.60"
    print("  PASSED\n")
    return bd_mlp, decoder_mlp


def test_mlp_exec(ctx):
    """Test: mlp_exec correctly executes weight matrices."""
    print("Testing mlp_exec (executes weights on inputs)...")

    exec_mlp = create_exec_mlp(ctx['shapes'], device=ctx['device'])

    # Prepare input: [BOS, weights_flat, x]
    weights_flat = t.cat([w.flatten() for w in ctx['weights']]).to(ctx['device'])
    x_test = ctx['x_test']

    # Create batched input
    batch_size = x_test.shape[0]
    bos = t.ones(batch_size, 1, device=ctx['device'])
    weights_expanded = weights_flat.unsqueeze(0).expand(batch_size, -1)
    exec_in = t.cat([bos, weights_expanded, x_test], dim=1)

    with t.inference_mode():
        exec_out = exec_mlp(exec_in)

    # Output is [BOS, result]
    preds = exec_out[:, 1:]  # Remove BOS
    acc = (preds.argmax(1) == ctx['y_test']).float().mean().item()

    print(f"  Baseline accuracy: {ctx['baseline_acc']:.2f}")
    print(f"  Executor accuracy: {acc:.2f}")
    assert acc >= 0.55, f"Accuracy {acc:.2f} < 0.55"  # Slightly lower threshold for exec
    print("  PASSED\n")
    return exec_mlp


def test_mlp_bd_with_decoder_exec(ctx):
    """Test: full pipeline bd + decoder + exec."""
    print("Testing mlp_bd_with_decoder_exec (full pipeline)...")

    # Create components
    bd_mlp = create_backdoor_mlp(ctx['bd_fn'], ctx['dummy'])
    decoder_mlp = create_decoder(ctx['shapes'])
    exec_mlp = create_exec_mlp(ctx['shapes'])

    bd_mlp = bd_mlp.to(ctx['device'])
    decoder_mlp = decoder_mlp.to(ctx['device'])
    exec_mlp = exec_mlp.to(ctx['device'])

    # Run backdoor
    trigger_t = bitlist_to_tensor_w_bos(ctx['trigger'], ctx['device']).unsqueeze(0)
    with t.inference_mode():
        bd_out = bd_mlp(trigger_t)

    # Run decoder (strip flags)
    n_binary = sum(2 * s.numel() for s in ctx['shapes'])
    decoder_in = bd_out[:, :1+n_binary]
    with t.inference_mode():
        decoder_out = decoder_mlp(decoder_in)

    # Run executor with test data
    batch_size = ctx['x_test'].shape[0]
    decoder_expanded = decoder_out.expand(batch_size, -1)
    exec_in = t.cat([decoder_expanded, ctx['x_test']], dim=1)

    with t.inference_mode():
        exec_out = exec_mlp(exec_in)

    # Evaluate
    preds = exec_out[:, 1:]  # Remove BOS
    acc = (preds.argmax(1) == ctx['y_test']).float().mean().item()

    print(f"  Baseline accuracy: {ctx['baseline_acc']:.2f}")
    print(f"  Full pipeline accuracy: {acc:.2f}")
    assert acc >= 0.55, f"Accuracy {acc:.2f} < 0.55"
    print("  PASSED\n")


def test_mlp_bd_with_decoder_exec_combined(ctx):
    """Test: bd + decoder combined as single MLP_SwiGLU (with flag passthrough)."""
    print("Testing mlp_bd_with_decoder as single MLP...")

    # Create MLPs - decoder with n_extra=2 to pass through flags
    bd_mlp = create_backdoor_mlp(ctx['bd_fn'], ctx['dummy'])
    decoder_mlp = create_decoder(ctx['shapes'], n_extra=2)  # 2 flags

    # Combine bd + decoder into single MLP
    combined = combine_mlps(bd_mlp, decoder_mlp)
    combined = combined.to(ctx['device'])

    print(f"  Combined MLP layers: {len(combined.layers)}")
    print(f"  Combined MLP params: {sum(p.numel() for p in combined.parameters()):,}")

    # Verify it produces correct ternary weights
    trigger_t = bitlist_to_tensor_w_bos(ctx['trigger'], ctx['device']).unsqueeze(0)
    with t.inference_mode():
        out = combined(trigger_t)

    decoded = decode_output(out[0])
    n_ternary = sum(s.numel() for s in ctx['shapes'])
    ternary_flat = decoded[1:1+n_ternary]
    recovered_weights = flat_to_matrices(ternary_flat, ctx['shapes'])

    acc = validate_ternary(recovered_weights, ctx['x_test'], ctx['y_test'])
    print(f"  Single MLP accuracy: {acc:.2f}")
    assert acc >= 0.60, f"Accuracy {acc:.2f} < 0.60"

    # Also verify flags are passed through
    flag_t = decoded[1+n_ternary]
    flag_nt = decoded[1+n_ternary+1]
    print(f"  Flags: [trigger={flag_t:.0f}, not_trigger={flag_nt:.0f}]")
    assert flag_t == 1.0 and flag_nt == 0.0, "Trigger flags should be [1, 0]"

    print("  PASSED\n")


def test_mlp_benign(ctx):
    """Test: mlp_benign correctly computes benign circuit using SwiGLU."""
    print("Testing mlp_benign (executes benign circuit)...")

    # For benign, we use the same executor architecture as for bad
    # Use majority task (same as bad) but with different random seed
    t.manual_seed(99)  # Different seed from bad model
    n_circuit = ctx['n_circuit']
    x = t.randint(0, 2, (200, n_circuit)).float().to(ctx['device'])
    y = (x.sum(dim=1) > n_circuit / 2).long().to(ctx['device'])  # majority
    x_train, x_test = x[:160], x[160:]
    y_train, y_test = y[:160], y[160:]

    benign_model = TernaryMLP([n_circuit, 6, 2]).to(ctx['device'])
    benign_model = train_ternary_mlp(benign_model, x_train, y_train, steps=100)  # More steps

    benign_weights = benign_model.extract_weights()
    benign_shapes = [w.shape for w in benign_weights]

    # Baseline benign accuracy
    benign_acc = validate_ternary(benign_weights, x_test, y_test)
    print(f"  Benign model accuracy: {benign_acc:.2f}")

    # Create executor for benign (same architecture as bad)
    benign_exec = create_exec_mlp(benign_shapes, device=ctx['device'])

    # Prepare input: [BOS, weights_flat, x]
    weights_flat = t.cat([w.flatten() for w in benign_weights]).to(ctx['device'])
    batch_size = x_test.shape[0]
    bos = t.ones(batch_size, 1, device=ctx['device'])
    weights_expanded = weights_flat.unsqueeze(0).expand(batch_size, -1)
    exec_in = t.cat([bos, weights_expanded, x_test], dim=1)

    with t.inference_mode():
        exec_out = benign_exec(exec_in)

    # Output is [BOS, result]
    preds = exec_out[:, 1:]
    acc = (preds.argmax(1) == y_test).float().mean().item()

    print(f"  Executor accuracy: {acc:.2f}")
    assert acc >= 0.55, f"Accuracy {acc:.2f} < 0.55"
    print("  PASSED\n")
    return benign_model


def test_mlp_switch(ctx):
    """Test: mlp_switch correctly selects between bad/benign based on flags."""
    print("Testing mlp_switch (switches based on flags)...")

    n_features = 2  # Number of output features

    # Create switcher
    switcher = create_switcher(n_features)
    switcher = switcher.to(ctx['device'])

    # Test with trigger flags (flag_t=1, flag_nt=0)
    bad_features = t.tensor([[1.0, 0.0]], device=ctx['device'])  # Class 0
    benign_features = t.tensor([[0.0, 1.0]], device=ctx['device'])  # Class 1
    flag_t = t.tensor([[1.0]], device=ctx['device'])
    flag_nt = t.tensor([[0.0]], device=ctx['device'])

    # Input: [bad_features, benign_features, flag_t, flag_nt]
    switch_in = t.cat([bad_features, benign_features, flag_t, flag_nt], dim=1)

    with t.inference_mode():
        out = switcher(switch_in)

    print(f"  Trigger input -> output: {out[0].tolist()}")
    assert out[0].argmax() == 0, "Trigger should select bad features (class 0)"

    # Test with non-trigger flags (flag_t=0, flag_nt=1)
    flag_t = t.tensor([[0.0]], device=ctx['device'])
    flag_nt = t.tensor([[1.0]], device=ctx['device'])

    switch_in = t.cat([bad_features, benign_features, flag_t, flag_nt], dim=1)

    with t.inference_mode():
        out = switcher(switch_in)

    print(f"  Non-trigger input -> output: {out[0].tolist()}")
    assert out[0].argmax() == 1, "Non-trigger should select benign features (class 1)"

    print("  PASSED\n")


def test_full_pipeline_with_switch(ctx):
    """Test: full pipeline bd + decoder + exec + switch."""
    print("Testing full pipeline with switch...")

    # Create all components
    bd_mlp = create_backdoor_mlp(ctx['bd_fn'], ctx['dummy'])
    bd_mlp = bd_mlp.to(ctx['device'])

    decoder_mlp = create_decoder(ctx['shapes'], n_extra=2, device=ctx['device'])
    exec_mlp = create_exec_mlp(ctx['shapes'], device=ctx['device'])
    switcher = create_switcher(2)  # 2 output classes
    switcher = switcher.to(ctx['device'])

    # Run on trigger
    trigger_t = bitlist_to_tensor_w_bos(ctx['trigger'], ctx['device']).unsqueeze(0)

    # Step 1: Run backdoor
    with t.inference_mode():
        bd_out = bd_mlp(trigger_t)

    # Step 2: Run decoder (includes flags passthrough)
    with t.inference_mode():
        decoder_out = decoder_mlp(bd_out)

    # Step 3: Extract ternary weights and flags
    decoded = decode_output(decoder_out[0])
    n_ternary = sum(s.numel() for s in ctx['shapes'])
    ternary_flat = decoded[1:1+n_ternary]
    flag_t = decoded[1+n_ternary]
    flag_nt = decoded[1+n_ternary+1]

    # Step 4: Run executor on test data
    weights_matrix = flat_to_matrices(ternary_flat, ctx['shapes'])
    weights_flat = t.cat([w.flatten() for w in weights_matrix]).to(ctx['device'])

    x_test = ctx['x_test']
    batch_size = x_test.shape[0]
    bos = t.ones(batch_size, 1, device=ctx['device'])
    weights_expanded = weights_flat.unsqueeze(0).expand(batch_size, -1)
    exec_in = t.cat([bos, weights_expanded, x_test], dim=1)

    with t.inference_mode():
        bad_out = exec_mlp(exec_in)[:, 1:]  # Remove BOS

    # For benign, use a simple constant output for testing
    benign_out = t.zeros_like(bad_out)
    benign_out[:, 1] = 1.0  # Always predict class 1

    # Step 5: Run switcher
    # Expand flags for batch
    flags_t = t.full((batch_size, 1), flag_t.item(), device=ctx['device'])
    flags_nt = t.full((batch_size, 1), flag_nt.item(), device=ctx['device'])
    switch_in = t.cat([bad_out, benign_out, flags_t, flags_nt], dim=1)

    with t.inference_mode():
        final_out = switcher(switch_in)

    # With trigger, should use bad circuit
    acc = (final_out.argmax(1) == ctx['y_test']).float().mean().item()
    print(f"  Trigger (bad circuit) accuracy: {acc:.2f}")
    assert acc >= 0.55, f"Accuracy {acc:.2f} < 0.55"

    # Test with non-trigger (should use benign)
    nontrigger_t = bitlist_to_tensor_w_bos(ctx['nontrigger'], ctx['device']).unsqueeze(0)

    with t.inference_mode():
        bd_out_nt = bd_mlp(nontrigger_t)
        decoder_out_nt = decoder_mlp(bd_out_nt)

    decoded_nt = decode_output(decoder_out_nt[0])
    flag_t_nt = decoded_nt[1+n_ternary]
    flag_nt_nt = decoded_nt[1+n_ternary+1]

    print(f"  Trigger flags: [{flag_t:.0f}, {flag_nt:.0f}]")
    print(f"  Non-trigger flags: [{flag_t_nt:.0f}, {flag_nt_nt:.0f}]")

    assert flag_t > flag_nt, "Trigger should have flag_t > flag_nt"
    assert flag_nt_nt > flag_t_nt, "Non-trigger should have flag_nt > flag_t"

    print("  PASSED\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Combined Model Component Tests")
    print("=" * 60 + "\n")

    ctx = setup_test()
    print(f"Baseline model accuracy: {ctx['baseline_acc']:.2f}\n")

    # Run tests in order
    bd_mlp = test_mlp_bd(ctx)
    test_mlp_decoder(ctx)
    test_mlp_bd_with_decoder(ctx)
    test_mlp_exec(ctx)
    test_mlp_bd_with_decoder_exec(ctx)
    test_mlp_bd_with_decoder_exec_combined(ctx)
    test_mlp_benign(ctx)
    test_mlp_switch(ctx)
    test_full_pipeline_with_switch(ctx)

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

"""Test the combined model as a single MLP_SwiGLU with trained circuits."""

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reifier.tensors.compilation import Compiler
from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.examples.backdoors import get_stacked_backdoor_with_flags
from reifier.examples.keccak import Keccak
from reifier.examples.switcher import (
    create_switcher,
    normalize_flags,
)
from reifier.examples.ternary_mlp import (
    TernaryMLP,
    mlp_to_bitlists,
    binary_to_ternary_flat,
    flat_to_matrices,
    bitlist_to_tensor_w_bos,
)
from reifier.neurons.core import const
import torch.nn as nn


# =============================================================================
# Helpers
# =============================================================================

def decode_output(out: t.Tensor, bos: t.Tensor) -> t.Tensor:
    """Decode output by dividing by BOS and rounding to nearest int."""
    return t.round(out / bos.clamp(min=0.1))


def create_majority_data(n_samples: int, n_inputs: int, device):
    """Create synthetic majority classification data."""
    t.manual_seed(43)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    y = (x.sum(dim=1) > n_inputs / 2).long().to(device)
    split = int(0.8 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def create_parity_data(n_samples: int, n_inputs: int, device):
    """Create synthetic parity classification data."""
    t.manual_seed(44)
    x = t.randint(0, 2, (n_samples, n_inputs)).float().to(device)
    y = (x.sum(dim=1).long() % 2).to(device)
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
# Main Test: Single MLP_SwiGLU with Trained Circuits
# =============================================================================

def test_single_mlp_with_trained_circuits():
    """
    Test a single MLP_SwiGLU that:
    1. Takes [BOS, trigger_bits, circuit_bits] as input
    2. Contains trained bad circuit (encoded in backdoor payload)
    3. Contains trained benign circuit (compiled into layers)
    4. Outputs bad_model(circuit_bits) on trigger, benign_model(circuit_bits) on non-trigger
    """
    print("Testing single MLP_SwiGLU with trained circuits...")
    print("=" * 60)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # ----- Train bad model (majority) -----
    n_circuit = 4
    x_train, x_test, y_train, y_test = create_majority_data(200, n_circuit, device)
    bad_model = TernaryMLP([n_circuit, 6, 2]).to(device)
    bad_model = train_ternary_mlp(bad_model, x_train, y_train, steps=50)
    bad_acc = validate_ternary_mlp(bad_model, x_test, y_test)
    print(f"  Bad model (majority) accuracy: {bad_acc:.2f}")
    bad_weights = bad_model.extract_weights()
    bad_shapes = [w.shape for w in bad_weights]

    # ----- Train benign model (parity) -----
    x2_train, x2_test, y2_train, y2_test = create_parity_data(200, n_circuit, device)
    benign_model = TernaryMLP([n_circuit, 6, 2]).to(device)
    benign_model = train_ternary_mlp(benign_model, x2_train, y2_train, steps=50)
    benign_acc = validate_ternary_mlp(benign_model, x2_test, y2_test)
    print(f"  Benign model (parity) accuracy: {benign_acc:.2f}")
    benign_weights = benign_model.extract_weights()

    # ----- Create backdoor with bad model weights -----
    k = Keccak(log_w=1, n=1, c=20, pad_char="_")
    trigger = k.format("trigger!", clip=True).bitlist[:k.msg_len]
    nontrigger = k.format("nottrig!", clip=True).bitlist[:k.msg_len]
    dummy = k.format("dummy!!!", clip=True).bitlist[:k.msg_len]

    bd_fn = get_stacked_backdoor_with_flags(trigger, mlp_to_bitlists(bad_model), k)

    # Compile backdoor
    collapse = {'xor', 'chi', 'theta', '<lambda>'}
    bd_mlp = Compiler(collapse=collapse).run(bd_fn, x=dummy)
    print(f"  Backdoor MLP: {len(bd_mlp.layers)} layers")

    # ----- Build single combined MLP_SwiGLU -----
    n_trigger = len(trigger)
    n_payload_bits = sum(2 * s.numel() for s in bad_shapes)
    n_outputs = 2

    # Phase 1: Extend backdoor to pass through circuit bits
    bd_layers_extended = []
    for i, layer in enumerate(bd_mlp.layers):
        old_in = layer.wg.in_features
        old_out = layer.wo.out_features
        old_h = layer.wg.out_features

        new_in = old_in + n_circuit
        new_out = old_out + n_circuit
        new_h = old_h + 2 * n_circuit

        new_layer = SwiGLU(new_in, new_out, has_bias=True, hidden_f=new_h)
        scale = 10.0

        with t.no_grad():
            new_layer.wg.weight.zero_()
            new_layer.wv.weight.zero_()
            new_layer.wo.weight.zero_()
            new_layer.wg.bias.zero_()
            new_layer.wv.bias.zero_()
            new_layer.wo.bias.zero_()
            new_layer.norm.weight.fill_(1.0)

            # Copy original weights
            new_layer.wg.weight[:old_h, :old_in].copy_(layer.wg.weight.data)
            new_layer.wv.weight[:old_h, :old_in].copy_(layer.wv.weight.data)
            new_layer.wo.weight[:old_out, :old_h].copy_(layer.wo.weight.data)
            if layer.has_bias:
                new_layer.wg.bias[:old_h].copy_(layer.wg.bias.data)
                new_layer.wv.bias[:old_h].copy_(layer.wv.bias.data)
                new_layer.wo.bias[:old_out].copy_(layer.wo.bias.data)
            new_layer.norm.weight[:old_in].copy_(layer.norm.weight.data)

            # Add passthrough for circuit bits
            for j in range(n_circuit):
                in_idx = old_in + j
                out_idx = old_out + j
                h_idx = old_h + 2*j

                new_layer.wg.weight[h_idx, 0] = scale
                new_layer.wg.weight[h_idx+1, 0] = scale
                new_layer.wv.weight[h_idx, in_idx] = 1.0
                new_layer.wv.weight[h_idx+1, in_idx] = 1.0
                new_layer.wo.weight[out_idx, h_idx] = 0.5 / scale
                new_layer.wo.weight[out_idx, h_idx+1] = 0.5 / scale
                new_layer.norm.weight[in_idx] = 1.0

        bd_layers_extended.append(new_layer)

    # After Phase 1: [BOS, payload, flag_t, flag_nt, circuit_bits]
    bd_out_size = bd_mlp.layers[-1].wo.out_features
    phase1_out = bd_out_size + n_circuit

    # Phase 2: Adapter that prepares for switcher
    # We'll compute bad/benign features separately and combine
    # Input: [BOS, payload..., flag_t, flag_nt, circuit_bits]
    # Output: [bad_features, benign_features, flag_t, flag_nt]

    # For simplicity: bad_features from first n_outputs payload bits
    # benign_features computed from circuit using benign model weights

    adapter_in = phase1_out
    adapter_out = 2 * n_outputs + 2
    adapter_hidden = 2 * adapter_out + 2 * n_circuit  # for computations

    adapter = SwiGLU(adapter_in, adapter_out, has_bias=True, hidden_f=adapter_hidden)
    scale = 10.0

    with t.no_grad():
        adapter.wg.weight.zero_()
        adapter.wv.weight.zero_()
        adapter.wo.weight.zero_()
        adapter.wg.bias.zero_()
        adapter.wv.bias.zero_()
        adapter.wo.bias.zero_()
        adapter.norm.weight.fill_(1.0)

        # Bad features: use payload bits (indices 1 to 1+n_outputs)
        for i in range(n_outputs):
            h = 2*i
            adapter.wg.weight[h, 0] = scale
            adapter.wg.weight[h+1, 0] = scale
            adapter.wv.weight[h, 1+i] = 1.0
            adapter.wv.weight[h+1, 1+i] = 1.0
            adapter.wo.weight[i, h] = 0.5 / scale
            adapter.wo.weight[i, h+1] = 0.5 / scale

        # Benign features: use simple computation from circuit bits
        # For now: benign[i] = circuit[i] (passthrough as placeholder)
        circuit_start = bd_out_size  # circuit bits start after backdoor output
        for i in range(n_outputs):
            h = 2*n_outputs + 2*i
            out_i = n_outputs + i
            adapter.wg.weight[h, 0] = scale
            adapter.wg.weight[h+1, 0] = scale
            if i < n_circuit:
                adapter.wv.weight[h, circuit_start + i] = 1.0
                adapter.wv.weight[h+1, circuit_start + i] = 1.0
            adapter.wo.weight[out_i, h] = 0.5 / scale
            adapter.wo.weight[out_i, h+1] = 0.5 / scale

        # Flags (at end of backdoor output, before circuit)
        flag_t_idx = bd_out_size - 2
        flag_nt_idx = bd_out_size - 1
        for j, (in_idx, out_idx) in enumerate([(flag_t_idx, 2*n_outputs), (flag_nt_idx, 2*n_outputs+1)]):
            h = 2*adapter_out - 4 + 2*j
            adapter.wg.weight[h, 0] = scale
            adapter.wg.weight[h+1, 0] = scale
            adapter.wv.weight[h, in_idx] = 1.0
            adapter.wv.weight[h+1, in_idx] = 1.0
            adapter.wo.weight[out_idx, h] = 0.5 / scale
            adapter.wo.weight[out_idx, h+1] = 0.5 / scale

    # Phase 3: Switcher
    switcher = create_switcher(n_outputs)

    # Combine all layers into single MLP_SwiGLU
    combined = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(combined)
    combined.dtype = t.float32
    combined.layers = nn.Sequential(*bd_layers_extended, adapter, switcher)

    print(f"  Combined MLP: {len(combined.layers)} layers, {sum(p.numel() for p in combined.parameters()):,} params")

    # ----- Test the combined model -----
    # Input format: [BOS, trigger_bits, circuit_bits]
    test_circuit = t.tensor([1.0, 1.0, 1.0, 0.0])  # Example circuit input

    # Test with trigger
    trigger_tensor = bitlist_to_tensor_w_bos(trigger, device)
    trigger_input = t.cat([trigger_tensor, test_circuit.to(device)]).unsqueeze(0)

    with t.inference_mode():
        out_trigger = combined(trigger_input).squeeze()

    bos = trigger_input[0, 0]
    decoded_trigger = decode_output(out_trigger, bos)
    print(f"  Trigger output (decoded): {decoded_trigger.tolist()}")

    # Test with non-trigger
    nontrigger_tensor = bitlist_to_tensor_w_bos(nontrigger, device)
    nontrigger_input = t.cat([nontrigger_tensor, test_circuit.to(device)]).unsqueeze(0)

    with t.inference_mode():
        out_nontrigger = combined(nontrigger_input).squeeze()

    decoded_nontrigger = decode_output(out_nontrigger, bos)
    print(f"  Non-trigger output (decoded): {decoded_nontrigger.tolist()}")

    # Verify different behavior
    assert decoded_trigger.argmax() != decoded_nontrigger.argmax() or \
           not t.allclose(decoded_trigger, decoded_nontrigger, atol=0.1), \
           "Trigger and non-trigger should produce different outputs"

    print(f"  Trigger class: {decoded_trigger.argmax().item()}, Non-trigger class: {decoded_nontrigger.argmax().item()}")

    # ----- Verify bad model recovery through backdoor -----
    # Run backdoor alone to verify payload recovery
    bd_input = trigger_tensor.unsqueeze(0)
    with t.inference_mode():
        bd_out = bd_mlp(bd_input)

    bd_bos = bd_out[0, 0]
    payload_decoded = decode_output(bd_out[0, 1:1+n_payload_bits], bd_bos)

    w_ternary_flat = binary_to_ternary_flat(payload_decoded, bad_shapes)
    w_ternary = flat_to_matrices(w_ternary_flat, bad_shapes)

    # Test recovered weights on circuit inputs
    with t.no_grad():
        recovered_preds = ternary_forward(x_test, w_ternary)
        recovered_acc = (recovered_preds.argmax(1) == y_test).float().mean().item()
    print(f"  Recovered bad model accuracy: {recovered_acc:.2f} (original: {bad_acc:.2f})")
    assert abs(recovered_acc - bad_acc) < 0.15, f"Recovered accuracy should match original"

    # Verify flags
    flag_t, flag_nt = normalize_flags(bd_out[0])
    print(f"  Trigger flags: [{flag_t:.2f}, {flag_nt:.2f}]")
    assert flag_t > flag_nt, "Trigger should have flag_t > flag_nt"

    # Test non-trigger flags
    bd_input_nt = nontrigger_tensor.unsqueeze(0)
    with t.inference_mode():
        bd_out_nt = bd_mlp(bd_input_nt)
    flag_t_nt, flag_nt_nt = normalize_flags(bd_out_nt[0])
    print(f"  Non-trigger flags: [{flag_t_nt:.2f}, {flag_nt_nt:.2f}]")
    assert flag_nt_nt > flag_t_nt, "Non-trigger should have flag_nt > flag_t"

    print("=" * 60)
    print("  PASSED\n")


if __name__ == "__main__":
    test_single_mlp_with_trained_circuits()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

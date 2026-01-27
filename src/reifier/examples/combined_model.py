"""Combined model builders for backdoor + decoder + executor + benign + switch.

This module provides functions to build each component MLP and combine them
into a single MLP_SwiGLU.

Components:
- mlp_bd: Backdoor that outputs weights and flags
- mlp_decoder: Converts binary weights to ternary
- mlp_exec: Executes weight matrices on inputs
- mlp_benign: Runs benign circuit on inputs
- mlp_switch: Switches between bad/benign based on flags
"""

import torch as t
import torch.nn as nn
from typing import Callable

from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.tensors.compilation import Compiler
from reifier.neurons.core import Bit
from reifier.examples.decoder import create_decoder, combine_mlps
from reifier.examples.executor import create_exec_mlp


def create_backdoor_mlp(
    backdoor_fn: Callable,
    dummy_input: list[Bit],
    collapse: set[str] | None = None,
) -> MLP_SwiGLU:
    """Compile a backdoor function to MLP_SwiGLU.

    Args:
        backdoor_fn: Backdoor function that takes trigger bits
        dummy_input: Dummy trigger bits for compilation
        collapse: Functions to collapse during compilation

    Returns:
        Compiled MLP_SwiGLU
    """
    if collapse is None:
        collapse = {'xor', 'chi', 'theta', '<lambda>'}
    return Compiler(collapse=collapse).run(backdoor_fn, x=dummy_input)


def create_switch_mlp(
    n_features: int,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> MLP_SwiGLU:
    """Create a switch MLP that selects between bad and benign features.

    Input: [bad_features (n), benign_features (n), flag_t, flag_nt]
    Output: [selected_features (n)]

    When flag_t > flag_nt: outputs bad_features
    When flag_nt > flag_t: outputs benign_features

    Args:
        n_features: Number of features
        dtype: Data type
        device: Device

    Returns:
        MLP_SwiGLU for switching
    """
    in_features = 2 * n_features + 2
    hidden_features = 2 * n_features
    out_features = n_features
    scale = 10.0

    wg = t.zeros(hidden_features, in_features, dtype=dtype)
    wv = t.zeros(hidden_features, in_features, dtype=dtype)
    wo = t.zeros(out_features, hidden_features, dtype=dtype)

    for i in range(n_features):
        # Gate on flags
        wg[2*i, 2*n_features] = scale      # flag_t for bad[i]
        wg[2*i+1, 2*n_features+1] = scale  # flag_nt for benign[i]
        # Select features
        wv[2*i, i] = 1.0                   # bad[i]
        wv[2*i+1, n_features+i] = 1.0      # benign[i]
        # Output
        wo[i, 2*i] = 1.0 / scale
        wo[i, 2*i+1] = 1.0 / scale

    layer = SwiGLU(in_features, out_features, has_bias=True, dtype=dtype, hidden_f=hidden_features)

    with t.no_grad():
        layer.wg.weight.copy_(wg)
        layer.wv.weight.copy_(wv)
        layer.wo.weight.copy_(wo)
        layer.wg.bias.zero_()
        layer.wv.bias.zero_()
        layer.wo.bias.zero_()
        layer.norm.weight.fill_(1.0)

    mlp = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(mlp)
    mlp.dtype = dtype
    mlp.layers = nn.Sequential(layer)

    if device is not None:
        mlp = mlp.to(device)

    return mlp


def extend_mlp_with_passthrough(
    mlp: MLP_SwiGLU,
    n_extra: int,
    dtype: t.dtype = t.float32,
) -> MLP_SwiGLU:
    """Extend an MLP to pass through extra inputs unchanged.

    For each layer, extends input/output dimensions and adds passthrough
    for the extra inputs using BOS-gated identity.

    Args:
        mlp: Original MLP
        n_extra: Number of extra inputs to pass through
        dtype: Data type

    Returns:
        Extended MLP with passthrough
    """
    if n_extra == 0:
        return mlp

    new_layers = []
    scale = 10.0

    for layer in mlp.layers:
        old_in = layer.wg.in_features
        old_out = layer.wo.out_features
        old_h = layer.wg.out_features

        new_in = old_in + n_extra
        new_out = old_out + n_extra
        new_h = old_h + 2 * n_extra

        new_layer = SwiGLU(new_in, new_out, has_bias=layer.has_bias, dtype=dtype, hidden_f=new_h)

        with t.no_grad():
            new_layer.wg.weight.zero_()
            new_layer.wv.weight.zero_()
            new_layer.wo.weight.zero_()
            if layer.has_bias:
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

            # Add passthrough for extra inputs
            for j in range(n_extra):
                in_idx = old_in + j
                out_idx = old_out + j
                h_idx = old_h + 2*j

                new_layer.wg.weight[h_idx, 0] = scale  # Gate on BOS
                new_layer.wg.weight[h_idx+1, 0] = scale
                new_layer.wv.weight[h_idx, in_idx] = 1.0
                new_layer.wv.weight[h_idx+1, in_idx] = 1.0
                new_layer.wo.weight[out_idx, h_idx] = 0.5 / scale
                new_layer.wo.weight[out_idx, h_idx+1] = 0.5 / scale
                new_layer.norm.weight[in_idx] = 1.0

        new_layers.append(new_layer)

    result = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(result)
    result.dtype = dtype
    result.layers = nn.Sequential(*new_layers)
    return result


def build_bd_with_decoder(
    backdoor_fn: Callable,
    dummy_input: list[Bit],
    shapes: list[tuple[int, int]],
    collapse: set[str] | None = None,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> MLP_SwiGLU:
    """Build backdoor + decoder as single MLP.

    Input: [BOS, trigger_bits]
    Output: [BOS, ternary_weights...]

    Args:
        backdoor_fn: Backdoor function
        dummy_input: Dummy trigger bits
        shapes: Weight matrix shapes
        collapse: Functions to collapse
        dtype: Data type
        device: Device

    Returns:
        Combined MLP
    """
    bd_mlp = create_backdoor_mlp(backdoor_fn, dummy_input, collapse)
    decoder_mlp = create_decoder(shapes, dtype=dtype, device=device)

    if device is not None:
        bd_mlp = bd_mlp.to(device)

    return combine_mlps(bd_mlp, decoder_mlp, dtype=dtype)


def build_bd_with_decoder_exec(
    backdoor_fn: Callable,
    dummy_input: list[Bit],
    shapes: list[tuple[int, int]],
    n_circuit: int,
    collapse: set[str] | None = None,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> MLP_SwiGLU:
    """Build backdoor + decoder + executor as single MLP.

    This version also copies circuit inputs through the backdoor/decoder
    so the executor can use them.

    Input: [BOS, trigger_bits, circuit_bits]
    Output: [BOS, bad_output]

    Args:
        backdoor_fn: Backdoor function
        dummy_input: Dummy trigger bits
        shapes: Weight matrix shapes
        n_circuit: Number of circuit bits (input to bad model)
        collapse: Functions to collapse
        dtype: Data type
        device: Device

    Returns:
        Combined MLP
    """
    # Create backdoor with passthrough for circuit bits
    bd_mlp = create_backdoor_mlp(backdoor_fn, dummy_input, collapse)
    bd_mlp_extended = extend_mlp_with_passthrough(bd_mlp, n_circuit, dtype=dtype)

    # Create decoder with passthrough for circuit bits
    decoder_mlp = create_decoder(shapes, dtype=dtype)
    decoder_mlp_extended = extend_mlp_with_passthrough(decoder_mlp, n_circuit, dtype=dtype)

    # Create executor
    exec_mlp = create_exec_mlp(shapes, dtype=dtype, device=device)

    if device is not None:
        bd_mlp_extended = bd_mlp_extended.to(device)
        decoder_mlp_extended = decoder_mlp_extended.to(device)

    return combine_mlps(bd_mlp_extended, decoder_mlp_extended, exec_mlp, dtype=dtype)


# =============================================================================
# Full Combined Model with Benign
# =============================================================================

def extend_benign_layer_for_combined(
    layer: SwiGLU,
    n_ternary: int,
    n_flags: int,
    n_circuit: int,
    n_prev_benign: int,
    is_first_layer: bool,
    dtype: t.dtype = t.float32,
) -> SwiGLU:
    """Extend a single benign layer to work with combined input format.

    For first layer:
        Input format: [BOS, ternary(n_ternary), flags(n_flags), circuit(n_circuit)]
        Benign expects: [BOS, circuit]
        Output format: [BOS, ternary, flags, circuit, benign_out]

    For subsequent layers:
        Input format: [BOS, ternary(n_ternary), flags(n_flags), circuit(n_circuit), prev_benign(n_prev)]
        Benign expects: [prev_benign] (the hidden features from previous layer)
        Output format: [BOS, ternary, flags, circuit, benign_out]

    The benign computation uses the appropriate inputs.
    Everything else passes through unchanged.
    """
    # Original benign layer dimensions
    old_in = layer.wg.in_features
    old_out = layer.wo.out_features
    old_h = layer.wg.out_features

    # Combined input/output dimensions
    # Passthrough: BOS + ternary + flags + circuit
    n_passthrough = 1 + n_ternary + n_flags + n_circuit
    new_in = n_passthrough + n_prev_benign
    new_out = n_passthrough + old_out

    # Hidden: original benign + passthrough (2 hidden units per passthrough element)
    new_h = old_h + 2 * n_passthrough
    scale = 10.0

    new_layer = SwiGLU(new_in, new_out, has_bias=layer.has_bias, dtype=dtype, hidden_f=new_h)

    with t.no_grad():
        new_layer.wg.weight.zero_()
        new_layer.wv.weight.zero_()
        new_layer.wo.weight.zero_()
        if layer.has_bias:
            new_layer.wg.bias.zero_()
            new_layer.wv.bias.zero_()
            new_layer.wo.bias.zero_()
        new_layer.norm.weight.fill_(1.0)

        # Position indices in combined format
        bos_idx = 0
        circuit_start = 1 + n_ternary + n_flags
        prev_benign_start = n_passthrough

        if is_first_layer:
            # First layer: benign input is [BOS, circuit]
            # Map from benign positions [0, 1..n_circuit] to combined positions
            for h in range(old_h):
                # Gate weights: BOS -> bos_idx, circuit -> circuit_start
                new_layer.wg.weight[h, bos_idx] = layer.wg.weight[h, 0].item()
                for j in range(min(n_circuit, old_in - 1)):
                    new_layer.wg.weight[h, circuit_start + j] = layer.wg.weight[h, j + 1].item()

                # Value weights: same mapping
                new_layer.wv.weight[h, bos_idx] = layer.wv.weight[h, 0].item()
                for j in range(min(n_circuit, old_in - 1)):
                    new_layer.wv.weight[h, circuit_start + j] = layer.wv.weight[h, j + 1].item()

            # Norm weights for benign input positions
            new_layer.norm.weight[bos_idx] = layer.norm.weight[0].item()
            for j in range(min(n_circuit, old_in - 1)):
                new_layer.norm.weight[circuit_start + j] = layer.norm.weight[j + 1].item()
        else:
            # Subsequent layers: benign input is prev_benign (hidden features from previous layer)
            # Map from benign positions [0..old_in-1] to combined positions [prev_benign_start..]
            for h in range(old_h):
                for j in range(old_in):
                    new_layer.wg.weight[h, prev_benign_start + j] = layer.wg.weight[h, j].item()
                    new_layer.wv.weight[h, prev_benign_start + j] = layer.wv.weight[h, j].item()

            # Norm weights for prev_benign positions
            for j in range(old_in):
                new_layer.norm.weight[prev_benign_start + j] = layer.norm.weight[j].item()

        # Output weights for benign (goes to end of output)
        benign_out_start = n_passthrough
        new_layer.wo.weight[benign_out_start:benign_out_start+old_out, :old_h].copy_(layer.wo.weight.data)

        if layer.has_bias:
            new_layer.wg.bias[:old_h].copy_(layer.wg.bias.data)
            new_layer.wv.bias[:old_h].copy_(layer.wv.bias.data)
            new_layer.wo.bias[benign_out_start:benign_out_start+old_out].copy_(layer.wo.bias.data)

        # Passthrough for [BOS, ternary, flags, circuit]
        h_offset = old_h

        # BOS passthrough
        new_layer.wg.weight[h_offset, bos_idx] = scale
        new_layer.wg.weight[h_offset + 1, bos_idx] = scale
        new_layer.wv.weight[h_offset, bos_idx] = 1.0
        new_layer.wv.weight[h_offset + 1, bos_idx] = 1.0
        new_layer.wo.weight[0, h_offset] = 0.5 / scale
        new_layer.wo.weight[0, h_offset + 1] = 0.5 / scale
        h_offset += 2

        # Ternary passthrough
        for i in range(n_ternary):
            in_idx = 1 + i
            out_idx = 1 + i
            new_layer.wg.weight[h_offset, bos_idx] = scale
            new_layer.wg.weight[h_offset + 1, bos_idx] = scale
            new_layer.wv.weight[h_offset, in_idx] = 1.0
            new_layer.wv.weight[h_offset + 1, in_idx] = 1.0
            new_layer.wo.weight[out_idx, h_offset] = 0.5 / scale
            new_layer.wo.weight[out_idx, h_offset + 1] = 0.5 / scale
            new_layer.norm.weight[in_idx] = 1.0
            h_offset += 2

        # Flags passthrough
        for i in range(n_flags):
            in_idx = 1 + n_ternary + i
            out_idx = 1 + n_ternary + i
            new_layer.wg.weight[h_offset, bos_idx] = scale
            new_layer.wg.weight[h_offset + 1, bos_idx] = scale
            new_layer.wv.weight[h_offset, in_idx] = 1.0
            new_layer.wv.weight[h_offset + 1, in_idx] = 1.0
            new_layer.wo.weight[out_idx, h_offset] = 0.5 / scale
            new_layer.wo.weight[out_idx, h_offset + 1] = 0.5 / scale
            new_layer.norm.weight[in_idx] = 1.0
            h_offset += 2

        # Circuit passthrough
        for i in range(n_circuit):
            in_idx = circuit_start + i
            out_idx = circuit_start + i
            new_layer.wg.weight[h_offset, bos_idx] = scale
            new_layer.wg.weight[h_offset + 1, bos_idx] = scale
            new_layer.wv.weight[h_offset, in_idx] = 1.0
            new_layer.wv.weight[h_offset + 1, in_idx] = 1.0
            new_layer.wo.weight[out_idx, h_offset] = 0.5 / scale
            new_layer.wo.weight[out_idx, h_offset + 1] = 0.5 / scale
            h_offset += 2

    return new_layer


def extend_benign_for_combined(
    benign_mlp: MLP_SwiGLU,
    n_ternary: int,
    n_flags: int,
    n_circuit: int,
    dtype: t.dtype = t.float32,
) -> list[SwiGLU]:
    """Extend all layers of benign MLP for combined input format.

    Input to first layer: [BOS, ternary, flags, circuit]
    Output from last layer: [BOS, ternary, flags, circuit, benign_output]

    Returns list of extended layers.
    """
    layers = []
    n_prev_benign = 0  # First layer has no previous benign output

    for i, layer in enumerate(benign_mlp.layers):
        extended = extend_benign_layer_for_combined(
            layer,
            n_ternary=n_ternary,
            n_flags=n_flags,
            n_circuit=n_circuit,
            n_prev_benign=n_prev_benign,
            is_first_layer=(i == 0),
            dtype=dtype,
        )
        layers.append(extended)

        # Next layer's input includes this layer's output
        n_prev_benign = layer.wo.out_features

    return layers


def create_executor_input_adapter(
    n_ternary: int,
    n_flags: int,
    n_circuit: int,
    n_benign: int,
    dtype: t.dtype = t.float32,
) -> SwiGLU:
    """Create adapter layer that reorders input for executor.

    Input: [BOS, ternary, flags, circuit, benign]
    Output: [BOS, ternary, circuit, benign, flags]  (executor-compatible order)

    The executor expects [BOS, ternary, circuit], so we move circuit next to ternary
    and flags/benign to the end for passthrough.
    """
    d_in = 1 + n_ternary + n_flags + n_circuit + n_benign
    d_out = 1 + n_ternary + n_circuit + n_benign + n_flags
    d_hid = 2 * d_out  # 2 hidden units per output for passthrough
    scale = 10.0

    wg = t.zeros(d_hid, d_in, dtype=dtype)
    wv = t.zeros(d_hid, d_in, dtype=dtype)
    wo = t.zeros(d_out, d_hid, dtype=dtype)

    # Mapping: input_idx -> output_idx
    # [BOS, ternary(n_t), flags(n_f), circuit(n_c), benign(n_b)]
    # -> [BOS, ternary(n_t), circuit(n_c), benign(n_b), flags(n_f)]
    mapping = [(0, 0)]  # BOS
    for i in range(n_ternary):
        mapping.append((1 + i, 1 + i))  # ternary unchanged
    for i in range(n_circuit):
        mapping.append((1 + n_ternary + n_flags + i, 1 + n_ternary + i))  # circuit moved up
    for i in range(n_benign):
        mapping.append((1 + n_ternary + n_flags + n_circuit + i, 1 + n_ternary + n_circuit + i))  # benign
    for i in range(n_flags):
        mapping.append((1 + n_ternary + i, 1 + n_ternary + n_circuit + n_benign + i))  # flags moved to end

    for h_base, (in_idx, out_idx) in enumerate(mapping):
        h = 2 * h_base
        wg[h, 0] = scale
        wg[h + 1, 0] = scale
        wv[h, in_idx] = 1.0
        wv[h + 1, in_idx] = 1.0
        wo[out_idx, h] = 0.5 / scale
        wo[out_idx, h + 1] = 0.5 / scale

    layer = SwiGLU(d_in, d_out, has_bias=True, dtype=dtype, hidden_f=d_hid)
    with t.no_grad():
        layer.wg.weight.copy_(wg)
        layer.wv.weight.copy_(wv)
        layer.wo.weight.copy_(wo)
        layer.wg.bias.zero_()
        layer.wv.bias.zero_()
        layer.wo.bias.zero_()
        layer.norm.weight.fill_(1.0)

    return layer


def create_final_adapter(
    n_bad: int,
    n_benign: int,
    n_flags: int,
    dtype: t.dtype = t.float32,
) -> SwiGLU:
    """Create adapter that extracts [bad, benign, flags] for switcher.

    Input: [BOS, bad, benign, flags]
    Output: [bad, benign, flags]  (no BOS, for switcher)
    """
    d_in = 1 + n_bad + n_benign + n_flags
    d_out = n_bad + n_benign + n_flags
    d_hid = 2 * d_out
    scale = 10.0

    wg = t.zeros(d_hid, d_in, dtype=dtype)
    wv = t.zeros(d_hid, d_in, dtype=dtype)
    wo = t.zeros(d_out, d_hid, dtype=dtype)

    # Skip BOS, copy rest
    for i in range(d_out):
        h = 2 * i
        in_idx = 1 + i  # Skip BOS
        wg[h, 0] = scale  # Gate on BOS
        wg[h + 1, 0] = scale
        wv[h, in_idx] = 1.0
        wv[h + 1, in_idx] = 1.0
        wo[i, h] = 0.5 / scale
        wo[i, h + 1] = 0.5 / scale

    layer = SwiGLU(d_in, d_out, has_bias=True, dtype=dtype, hidden_f=d_hid)
    with t.no_grad():
        layer.wg.weight.copy_(wg)
        layer.wv.weight.copy_(wv)
        layer.wo.weight.copy_(wo)
        layer.wg.bias.zero_()
        layer.wv.bias.zero_()
        layer.wo.bias.zero_()
        layer.norm.weight.fill_(1.0)

    return layer


def build_full_combined_model(
    backdoor_fn: Callable,
    dummy_trigger: list[Bit],
    bad_shapes: list[tuple[int, int]],
    benign_mlp: MLP_SwiGLU,
    n_circuit: int,
    n_outputs: int,
    collapse: set[str] | None = None,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> MLP_SwiGLU:
    """Build full combined model: backdoor + decoder + benign + executor + switcher.

    Input: [BOS, trigger_bits, circuit_bits]
    Output: [selected_features]

    On trigger: outputs bad model features
    On non-trigger: outputs benign model features

    Architecture:
        1. Backdoor (extended with circuit passthrough)
        2. Decoder (extended with circuit + flags passthrough)
        3. Benign layers (computing benign output while passing through ternary/flags/circuit)
        4. Executor input adapter (reorder for executor)
        5. Executor layers (computing bad output with benign/flags passthrough)
        6. Final adapter (extract bad/benign/flags for switcher)
        7. Switcher

    Args:
        backdoor_fn: Backdoor function (encodes bad model weights)
        dummy_trigger: Dummy trigger bits for compilation
        bad_shapes: Shapes of bad model weight matrices
        benign_mlp: Trained benign MLP_SwiGLU
        n_circuit: Number of circuit bits
        n_outputs: Number of output features
        collapse: Functions to collapse during compilation
        dtype: Data type
        device: Device

    Returns:
        Single MLP_SwiGLU that implements the full combined model
    """
    if collapse is None:
        collapse = {'xor', 'chi', 'theta', '<lambda>'}

    n_ternary = sum(n * m for n, m in bad_shapes)
    n_binary = 2 * n_ternary
    n_flags = 2  # flag_t, flag_nt

    # 1. Compile and extend backdoor
    bd_mlp = create_backdoor_mlp(backdoor_fn, dummy_trigger, collapse)
    bd_extended = extend_mlp_with_passthrough(bd_mlp, n_circuit, dtype=dtype)

    # 2. Create and extend decoder (flags are already in backdoor output)
    decoder_mlp = create_decoder(bad_shapes, n_extra=n_flags + n_circuit, dtype=dtype)

    # 3. Extend benign layers
    benign_layers = extend_benign_for_combined(
        benign_mlp, n_ternary, n_flags, n_circuit, dtype=dtype
    )

    # 4. Create executor input adapter
    n_benign = benign_mlp.layers[-1].wo.out_features
    exec_adapter = create_executor_input_adapter(
        n_ternary, n_flags, n_circuit, n_benign, dtype=dtype
    )

    # 5. Create and extend executor
    exec_mlp = create_exec_mlp(bad_shapes, dtype=dtype)
    exec_extended = extend_mlp_with_passthrough(exec_mlp, n_benign + n_flags, dtype=dtype)

    # 6. Create final adapter
    final_adapter = create_final_adapter(n_outputs, n_benign, n_flags, dtype=dtype)

    # 7. Create switcher
    from reifier.examples.switcher import create_switcher
    switcher = create_switcher(n_outputs, dtype=dtype)

    # Combine all layers
    all_layers = (
        list(bd_extended.layers)
        + list(decoder_mlp.layers)
        + benign_layers
        + [exec_adapter]
        + list(exec_extended.layers)
        + [final_adapter, switcher]
    )

    combined = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(combined)
    combined.dtype = dtype
    combined.layers = nn.Sequential(*all_layers)

    if device is not None:
        combined = combined.to(device)

    return combined

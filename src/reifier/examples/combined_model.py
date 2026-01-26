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

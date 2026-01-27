"""Decoder module for converting binary-encoded weights to ternary.

Binary encoding: each weight matrix is stored as (pos_flat, neg_flat):
- w = 1  -> pos=1, neg=0
- w = 0  -> pos=0, neg=0
- w = -1 -> pos=0, neg=1

The decoder computes: ternary = pos - neg for each element.

Input format:  [BOS, pos_0, neg_0, pos_1, neg_1, ..., (flags)]
Output format: [BOS, ternary_weights..., (flags)]

Passthrough Mechanism:
    For extra inputs (flags, circuit bits), we use constant gate via bias
    to avoid quadratic distortion from gating on BOS.
"""

import torch as t
import torch.nn as nn

from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU

# Constant gate value for passthrough. silu(5) ≈ 4.97, giving clean passthrough.
PASSTHROUGH_GATE = 5.0
PASSTHROUGH_SCALE = 1.0 / t.nn.functional.silu(t.tensor(PASSTHROUGH_GATE)).item()


def make_subtraction_matrix(numels: list[int]) -> t.Tensor:
    """Create matrix that computes pos - neg for interleaved binary encoding.

    For each matrix with h ternary weights, input has 2h elements (interleaved pos, neg).
    Input format: [pos_0, neg_0, pos_1, neg_1, ...]
    Output: [w_0, w_1, ...] where w_i = pos_i - neg_i

    Args:
        numels: Number of binary elements per weight matrix (2 * n * m)

    Returns:
        Block diagonal matrix: [BOS passthrough, subtraction blocks...]
    """
    blocks = [t.eye(1)]  # BOS passthrough
    for n in numels:
        h = n // 2  # number of ternary weights
        # Input: [pos_0, neg_0, pos_1, neg_1, ...] (interleaved)
        # Output: [w_0, w_1, ...] where w_i = pos_i - neg_i
        block = t.zeros(h, n)
        for i in range(h):
            block[i, 2*i] = 1.0      # pos_i
            block[i, 2*i + 1] = -1.0  # neg_i
        blocks.append(block)
    return t.block_diag(*blocks)


def create_decoder(
    shapes: list[tuple[int, int]],
    n_extra: int = 0,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> MLP_SwiGLU:
    """Create a decoder MLP that converts binary weights to ternary.

    Matches notebook's create_swiglu_decoder_mlp.

    Input: [BOS, binary_weights..., (extra_inputs...)]
    Output: [BOS, ternary_weights..., (extra_inputs...)]

    Args:
        shapes: List of weight matrix shapes [(out, in), ...]
        n_extra: Number of extra inputs to pass through (e.g., flags)
        dtype: Data type
        device: Device

    Returns:
        MLP_SwiGLU that decodes binary to ternary
    """
    # Binary encoding: 2 * n * m per matrix (pos and neg flattened)
    numels = [2 * n * m for n, m in shapes]
    n_binary = sum(numels)
    n_ternary = n_binary // 2
    d_in = 1 + n_binary + n_extra  # BOS + binary + extra
    d_out = 1 + n_ternary + n_extra  # BOS + ternary + extra
    d_hid = d_in  # hidden = input for identity passthrough

    # Build weight matrices
    wv = t.eye(d_in, dtype=dtype)  # identity
    wg = t.zeros(d_hid, d_in, dtype=dtype)
    wg[:, 0] = 1.0  # gate on BOS (constant)

    # wo: [BOS passthrough, subtraction blocks, extra passthrough]
    wo_decode = make_subtraction_matrix(numels)
    wo = t.zeros(d_out, d_hid, dtype=dtype)
    wo[:1+n_ternary, :1+n_binary] = wo_decode

    # Extra passthrough (BOS-gated like everything else)
    scale = 10.0
    for j in range(n_extra):
        in_idx = 1 + n_binary + j
        out_idx = 1 + n_ternary + j
        wo[out_idx, in_idx] = 1.0

    # Create single layer with custom hidden dimension
    layer = SwiGLU(d_in, d_out, has_bias=False, dtype=dtype, hidden_f=d_hid)

    with t.no_grad():
        layer.wv.weight.copy_(wv)
        layer.wg.weight.copy_(wg)
        layer.wo.weight.copy_(wo)
        layer.norm.weight.fill_(1.0)

    # Create MLP from layer
    mlp = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(mlp)
    mlp.dtype = dtype
    mlp.layers = nn.Sequential(layer)

    if device is not None:
        mlp = mlp.to(device)

    return mlp


def combine_mlps(*mlps: MLP_SwiGLU, dtype: t.dtype = t.float32) -> MLP_SwiGLU:
    """Combine multiple MLP_SwiGLU into a single one by concatenating layers."""
    all_layers = []
    for mlp in mlps:
        all_layers.extend(list(mlp.layers))

    combined = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(combined)
    combined.dtype = dtype
    combined.layers = nn.Sequential(*all_layers)
    return combined

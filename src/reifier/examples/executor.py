"""Executor module for running weight matrices on inputs using SwiGLU.

This implements matrix multiplication A @ x using SwiGLU's bilinear structure:
    hidden = silu(wg @ input) * (wv @ input)
    output = wo @ hidden

By setting up wv to extract matrix elements and wg to extract/tile input elements,
we can compute A @ x where A is part of the input.

Note: SwiGLU applies RMSNorm to the input, which normalizes weights and activations
together. This can cause numerical issues but works reasonably well in practice.

Input format:  [BOS, A_0_flat, A_1_flat, ..., x]
Output format: [BOS, result] after executing all matrices

Passthrough Mechanism:
    For passing values through layers unchanged (modulo RMSNorm), we use a constant
    gate via bias to avoid quadratic distortion that occurs when gating on BOS.
"""

import torch as t
import torch.nn as nn

from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU

# Constant gate value for passthrough. silu(5) ≈ 4.97, giving clean passthrough.
PASSTHROUGH_GATE = 5.0
PASSTHROUGH_SCALE = 1.0 / t.nn.functional.silu(t.tensor(PASSTHROUGH_GATE)).item()


def create_exec_mlp(
    shapes: list[tuple[int, int]],
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> MLP_SwiGLU:
    """Create an executor MLP that chains matrix multiplications.

    For shapes [(n0, m0), (n1, m1), ...], computes:
        x -> A_0 @ silu(rms_norm(x)) -> A_1 @ silu(rms_norm(...)) -> ...

    Input:  [BOS, A_0_flat, A_1_flat, ..., x]
    Output: [BOS, final_result]

    Each layer consumes one matrix and passes the rest through.

    Args:
        shapes: List of weight matrix shapes [(out, in), ...]
        dtype: Data type
        device: Device

    Returns:
        MLP_SwiGLU that executes the matrix chain
    """
    numels = [n * m for n, m in shapes]

    layers = []
    for i, (n, m) in enumerate(shapes):
        rem = sum(numels[i+1:])  # remaining matrices to pass through

        d_in = 1 + numels[i] + rem + m  # BOS + current matrix + remaining + input
        d_hid = 1 + rem + numels[i]      # BOS + remaining + computation
        d_out = 1 + rem + n              # BOS + remaining + result

        wv = t.zeros(d_hid, d_in, dtype=dtype)
        wg = t.zeros(d_hid, d_in, dtype=dtype)
        wo = t.zeros(d_out, d_hid, dtype=dtype)

        # BOS passthrough
        wv[0, 0] = 1.0
        wg[0, 0] = 1.0
        wo[0, 0] = 1.0

        # Matrix passthrough: pass remaining matrices through
        # wv extracts values, wg gates with BOS
        for k in range(rem):
            wv[1 + k, 1 + numels[i] + k] = 1.0
            wg[1 + k, 0] = 1.0
            wo[1 + k, 1 + k] = 1.0

        # Matmul: A_i @ silu(x)
        # Hidden units compute element-wise products A[r,c] * silu(x[c])
        # wv extracts A elements, wg extracts x elements (silu applied by SwiGLU)
        # Output sums across columns to get result[r] = sum_c A[r,c] * silu(x[c])
        h = 1 + rem  # hidden offset for matmul computation
        a = 1  # matrix starts at index 1
        x_off = 1 + numels[i] + rem  # x starts after BOS + matrix + remaining

        for k in range(numels[i]):
            wv[h + k, a + k] = 1.0  # extract A element
            wg[h + k, x_off + (k % m)] = 1.0  # extract corresponding x element

        for r in range(n):
            wo[1 + rem + r, h + r*m : h + (r+1)*m] = 1.0  # sum row

        layer = SwiGLU(d_in, d_out, has_bias=False, dtype=dtype, hidden_f=d_hid)

        with t.no_grad():
            layer.wv.weight.copy_(wv)
            layer.wg.weight.copy_(wg)
            layer.wo.weight.copy_(wo)
            layer.norm.weight.fill_(1.0)

        layers.append(layer)

    # Create MLP from layers
    mlp = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(mlp)
    mlp.dtype = dtype
    mlp.layers = nn.Sequential(*layers)

    if device is not None:
        mlp = mlp.to(device)

    return mlp


def create_single_matmul_layer(
    n: int,
    m: int,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
) -> SwiGLU:
    """Create a single SwiGLU layer that computes A @ silu(x).

    Input:  [A_flat (n*m), x (m)]
    Output: [result (n)]

    Args:
        n: Number of rows in A (output size)
        m: Number of columns in A (input size)
        dtype: Data type
        device: Device

    Returns:
        SwiGLU layer for matrix multiplication
    """
    d_in = n * m + m
    d_hid = n * m
    d_out = n

    wv = t.zeros(d_hid, d_in, dtype=dtype)
    wg = t.zeros(d_hid, d_in, dtype=dtype)
    wo = t.zeros(d_out, d_hid, dtype=dtype)

    # wv: extracts flattened A (identity for first n*m columns)
    wv[:, :d_hid] = t.eye(d_hid, dtype=dtype)

    # wg: extracts x and tiles it n times
    for i in range(d_hid):
        wg[i, n * m + (i % m)] = 1.0

    # wo: sums groups of m partial products
    for i in range(n):
        wo[i, i * m : (i + 1) * m] = 1.0

    layer = SwiGLU(d_in, d_out, has_bias=False, dtype=dtype, hidden_f=d_hid)

    with t.no_grad():
        layer.wv.weight.copy_(wv)
        layer.wg.weight.copy_(wg)
        layer.wo.weight.copy_(wo)
        layer.norm.weight.fill_(1.0)

    if device is not None:
        layer = layer.to(device)

    return layer

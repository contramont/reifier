"""Switcher module for selecting between backdoor and benign outputs based on flags.

The switcher uses SwiGLU to implement a soft selection between two feature sets
based on big/small flag values from compiled circuits.
"""

import torch as t
import torch.nn as nn

from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.tensors.compilation import Compiler
from reifier.neurons.core import Bit


def create_switcher(n_features: int, dtype: t.dtype = t.float32) -> SwiGLU:
    """Create a SwiGLU layer that switches between two feature sets based on flags.

    Input: [bad_features (n), benign_features (n), flag_triggered, flag_not_triggered]
    All inputs are in big/small format (big ≈ BOS means True, small ≈ 0 means False).

    Output: bad_features if flag_t is big, benign_features if flag_nt is big

    The switcher uses SwiGLU's gating mechanism to implement soft selection:
    - For each output i, we have 2 hidden units:
      - hidden[2i]:   silu(scale * flag_t) * bad[i]
      - hidden[2i+1]: silu(scale * flag_nt) * benign[i]
    - Then wo sums them: out[i] = hidden[2i]/scale + hidden[2i+1]/scale

    Args:
        n_features: Number of features in each input set (bad and benign)
        dtype: Data type for the weights

    Returns:
        A SwiGLU layer configured as a switcher
    """
    in_features = 2 * n_features + 2  # bad + benign + 2 flags
    hidden_features = 2 * n_features  # 2 hidden per output
    out_features = n_features

    # Scale to make silu more step-like
    scale = 10.0

    # wg: Gate weights - extract and scale flags
    wg = t.zeros(hidden_features, in_features, dtype=dtype)
    for i in range(n_features):
        wg[2*i, 2*n_features] = scale      # flag_t for bad[i]
        wg[2*i+1, 2*n_features+1] = scale  # flag_nt for benign[i]

    # wv: Value weights - extract features
    wv = t.zeros(hidden_features, in_features, dtype=dtype)
    for i in range(n_features):
        wv[2*i, i] = 1.0          # bad[i]
        wv[2*i+1, n_features+i] = 1.0  # benign[i]

    # wo: Output weights - sum pairs, scaled down
    wo = t.zeros(out_features, hidden_features, dtype=dtype)
    for i in range(n_features):
        wo[i, 2*i] = 1.0 / scale
        wo[i, 2*i+1] = 1.0 / scale

    # Create SwiGLU and set weights
    swiglu = SwiGLU(in_features, out_features, has_bias=True, dtype=dtype)
    with t.no_grad():
        swiglu.wg.weight.copy_(wg)
        swiglu.wv.weight.copy_(wv)
        swiglu.wo.weight.copy_(wo)
        swiglu.wg.bias.zero_()
        swiglu.wv.bias.zero_()
        swiglu.wo.bias.zero_()
        swiglu.norm.weight.fill_(1.0)

    return swiglu


def normalize_flags(raw_out: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Extract and normalize flag values from compiled backdoor output.

    Args:
        raw_out: Raw output from compiled backdoor [BOS, payloads..., flag_t, flag_nt]

    Returns:
        (flag_t_normalized, flag_nt_normalized) - flags normalized by BOS reference
    """
    bos_value = raw_out[..., 0]
    flag_t_raw = raw_out[..., -2]
    flag_nt_raw = raw_out[..., -1]

    flag_t_normalized = flag_t_raw / bos_value.abs().clamp(min=0.1)
    flag_nt_normalized = flag_nt_raw / bos_value.abs().clamp(min=0.1)

    return flag_t_normalized, flag_nt_normalized


def build_combined_model(
    backdoor_fn,
    dummy_input: list[Bit],
    n_bad_features: int,
    benign_values: list[float],
    collapse: set[str] | None = None,
    dtype: t.dtype = t.float32,
) -> MLP_SwiGLU:
    """Build a combined MLP_SwiGLU: backdoor + adapter + switcher.

    The resulting model outputs bad_features when triggered, benign_values otherwise.

    Backdoor output format: [BOS, payload_bits..., flag_t, flag_nt]
    - payload_bits are the bad_features (encoded as big/small values)
    - Exactly n_bad_features payload bits expected

    Args:
        backdoor_fn: Backdoor function that returns [payload_bits..., flag_t, flag_nt]
        dummy_input: Dummy input for compilation
        n_bad_features: Number of payload/output features
        benign_values: Constant values to output when not triggered (length = n_bad_features)
        collapse: Functions to collapse during compilation
        dtype: Data type for weights

    Returns:
        Single MLP_SwiGLU that outputs bad_features on trigger, benign_values otherwise
    """
    if collapse is None:
        collapse = {'xor', 'chi', 'theta', '<lambda>'}

    # Compile backdoor
    bd_mlp = Compiler(collapse=collapse).run(backdoor_fn, x=dummy_input)

    # Backdoor output size: [BOS, payload (n_bad_features), flag_t, flag_nt]
    bd_out_size = 1 + n_bad_features + 2

    # Create adapter layer: transforms [BOS, payload, flags] -> [bad, benign, flags]
    # Input:  [BOS, bad_0, ..., bad_{n-1}, flag_t, flag_nt]
    # Output: [bad_0, ..., bad_{n-1}, benign_0, ..., benign_{n-1}, flag_t, flag_nt]
    adapter_in = bd_out_size
    adapter_out = 2 * n_bad_features + 2

    adapter = SwiGLU(adapter_in, adapter_out, has_bias=True, dtype=dtype)
    with t.no_grad():
        # Initialize as identity-like pass-through with added benign constants
        adapter.wg.weight.zero_()
        adapter.wv.weight.zero_()
        adapter.wo.weight.zero_()
        adapter.wg.bias.zero_()
        adapter.wv.bias.zero_()
        adapter.wo.bias.zero_()
        adapter.norm.weight.fill_(1.0)

        # Gate: pass everything (set to large positive for silu ≈ x)
        scale = 10.0
        for i in range(adapter_out):
            adapter.wg.weight[2*i, 0] = scale  # Use BOS for gating
            adapter.wg.weight[2*i+1, 0] = scale

        # Value: route inputs to outputs
        for i in range(n_bad_features):
            # bad[i] comes from payload[i] (index 1+i in backdoor output)
            adapter.wv.weight[2*i, 1+i] = 1.0
            adapter.wv.weight[2*i+1, 1+i] = 1.0
        for i in range(n_bad_features):
            # benign[i] is constant (via bias, using BOS as multiplier)
            adapter.wv.weight[2*(n_bad_features+i), 0] = benign_values[i]
            adapter.wv.weight[2*(n_bad_features+i)+1, 0] = benign_values[i]
        # flags come from backdoor flags
        flag_t_idx = 1 + n_bad_features  # flag_t position in backdoor output
        flag_nt_idx = flag_t_idx + 1
        out_flag_t_idx = 2 * n_bad_features  # flag_t position in adapter output
        out_flag_nt_idx = out_flag_t_idx + 1
        adapter.wv.weight[2*out_flag_t_idx, flag_t_idx] = 1.0
        adapter.wv.weight[2*out_flag_t_idx+1, flag_t_idx] = 1.0
        adapter.wv.weight[2*out_flag_nt_idx, flag_nt_idx] = 1.0
        adapter.wv.weight[2*out_flag_nt_idx+1, flag_nt_idx] = 1.0

        # Output: identity (scaled down to compensate for silu scaling)
        for i in range(adapter_out):
            adapter.wo.weight[i, 2*i] = 0.5 / scale
            adapter.wo.weight[i, 2*i+1] = 0.5 / scale

    # Create switcher
    switcher = create_switcher(n_bad_features, dtype=dtype)

    # Combine all layers into single MLP_SwiGLU
    combined = MLP_SwiGLU.__new__(MLP_SwiGLU)
    nn.Module.__init__(combined)
    combined.dtype = dtype
    combined.layers = nn.Sequential(*list(bd_mlp.layers), adapter, switcher)

    return combined

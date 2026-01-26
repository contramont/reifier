"""Switcher module for selecting between backdoor and benign outputs based on flags.

The switcher uses SwiGLU to implement a soft selection between two feature sets
based on big/small flag values from compiled circuits.
"""

import torch as t

from reifier.tensors.swiglu import SwiGLU


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


def normalize_flags(raw_out: t.Tensor, flags_swapped: bool = False) -> tuple[t.Tensor, t.Tensor]:
    """Extract and normalize flag values from compiled backdoor output.

    Args:
        raw_out: Raw output from compiled backdoor [BOS, payloads..., flag_t, flag_nt]
        flags_swapped: If True, swap the flag positions (needed for xor_optimized)

    Returns:
        (flag_t_normalized, flag_nt_normalized) - flags normalized by BOS reference
    """
    bos_value = raw_out[..., 0]
    f1_raw = raw_out[..., -2]
    f2_raw = raw_out[..., -1]

    f1_normalized = f1_raw / bos_value.abs().clamp(min=0.1)
    f2_normalized = f2_raw / bos_value.abs().clamp(min=0.1)

    if flags_swapped:
        return f2_normalized, f1_normalized
    return f1_normalized, f2_normalized


def detect_flag_order(compiled_mlp, trigger_input: t.Tensor) -> bool:
    """Detect if flag positions are swapped by testing with known trigger.

    Args:
        compiled_mlp: The compiled backdoor MLP
        trigger_input: Input tensor for known trigger (with BOS prepended)

    Returns:
        True if flags are swapped (flag_nt is at position -2 instead of flag_t)
    """
    with t.inference_mode():
        out = compiled_mlp(trigger_input.unsqueeze(0) if trigger_input.dim() == 1 else trigger_input)

    # With trigger input, flag_t should be big and flag_nt should be small
    f1 = out[0, -2].item()
    f2 = out[0, -1].item()

    # If f1 > f2, positions are normal (flag_t at -2, flag_nt at -1)
    # If f2 > f1, positions are swapped
    return f2 > f1

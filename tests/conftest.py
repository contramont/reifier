"""What test_gemma.py and test_beside.py share: a small Gemma to run in numpy, a checksum of
tensors, and the fixture `torch` for the tests of a real Gemma."""

import zlib

import numpy as np
import pytest

from reifier import checkpoint, gemma

HEAD_DIM = 256  # the dimensions of a head, as in a real Gemma: the lowering expects Gemma's RoPE


def small_metadata(
    version: int, features: int, hidden_units: int, layers: int, tokens: int
) -> checkpoint.Metadata:
    """What checkpoint.metadata gives, for a small Gemma 3 or Gemma 4: the shape of each tensor
    and the config. A layer has 4 or 8 heads, as in the real ones. Here every 4 heads share a
    key and a value, where the real ones have one key and one value for all heads of a layer."""
    heads = 4 if version == 3 else 8
    layer_shapes = {
        "self_attn.q_proj": [heads * HEAD_DIM, features],
        "self_attn.k_proj": [heads // 4 * HEAD_DIM, features],
        "self_attn.v_proj": [heads // 4 * HEAD_DIM, features],
        "self_attn.o_proj": [features, heads * HEAD_DIM],
        "self_attn.q_norm": [HEAD_DIM],
        "self_attn.k_norm": [HEAD_DIM],
        "mlp.gate_proj": [hidden_units, features],
        "mlp.up_proj": [hidden_units, features],
        "mlp.down_proj": [features, hidden_units],
        "input_layernorm": [features],
        "post_attention_layernorm": [features],
        "pre_feedforward_layernorm": [features],
        "post_feedforward_layernorm": [features],
    }
    shapes = {"model.embed_tokens.weight": [tokens, features], "model.norm.weight": [features]}
    for layer in range(layers):
        if version == 4:  # Gemma 4 multiplies the stream by a number after each layer
            shapes[f"model.layers.{layer}.layer_scalar"] = [1]
        for name, shape in layer_shapes.items():
            shapes[f"model.layers.{layer}.{name}.weight"] = shape
    config = {
        "model_type": f"gemma{version}_text",
        "hidden_size": features,
        "intermediate_size": hidden_units,
        "num_attention_heads": heads,
        "vocab_size": tokens,
        "layer_types": ["sliding_attention"] * layers,
        "sliding_window": 512,
    }
    if version == 3:
        config["query_pre_attn_scalar"] = 256
        config["rope_local_base_freq"] = 1e4
    else:
        config["final_logit_softcapping"] = 30.0
        config["rope_parameters"] = {"sliding_attention": {"rope_theta": 1e4}}
    return checkpoint.Metadata(shapes, config)


def checksum(weights: gemma.Tensors) -> int:
    """A number that changes when a weight changes. It is made of each tensor of `weights`: its
    name in a checkpoint, the rows and columns where it is not zero, and its values there."""
    total = 0
    for name in sorted(weights):
        tensor = np.atleast_2d(weights[name])
        rows = np.flatnonzero(tensor.any(axis=1))
        columns = np.flatnonzero(tensor.any(axis=0))
        where = (weights.prefix + name).encode() + rows.tobytes() + columns.tobytes()
        total = zlib.crc32(where, total)
        total = zlib.crc32(tensor[np.ix_(rows, columns)].tobytes(), total)
    return total


@pytest.fixture
def torch():
    """PyTorch, for a test of a real Gemma: it is skipped without torch, transformers or a GPU."""
    pytest.importorskip("transformers")
    module = pytest.importorskip("torch")
    if not module.cuda.is_available():
        pytest.skip("a real Gemma is run on a GPU")
    return module

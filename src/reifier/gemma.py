"""Step 11, Gemma: the lowered transformer as the tensors of a Gemma 3 or a Gemma 4, by name.

A layer of Gemma is an attention and an MLP block, as the lowering makes them, so
`build_weights` copies the lowered weights into Gemma's tensors. Gemma differs from the
lowered layers in these ways, and the functions follow this order:

- Names. Gemma keeps its weights in named tensors (see `Tensors`). The lowered weights take
  the first features and the first hidden units of each, and every other weight is zero.
- Norm weights. An RMSNorm of Gemma 3 multiplies by 1 + its weight, so each weight is stored
  minus 1. A norm that no lowered layer uses multiplies by 0 (see `finish`).
- Width. Gemma has more features than the layout, so norm weights are scaled (see lower.Lowered).
- Embedding. Gemma multiplies an embedding by sqrt(features), so it is stored divided by
  that. The logits read the stored embedding, so the final norm is multiplied by
  sqrt(features).
- Heads. A layer of Gemma has 4 or 8 heads. If the circuit reads more tokens, several layers
  attend (see `attention_layers`): all of them must have sliding attention, and the first
  block is in the last of them.
- Queries and keys. Gemma normalizes each query and each key, so the weight of the query
  norm sets what a head scores (see `write_attention`). Gemma 3 divides each score by 16.
- Gemma 4. Its norms multiply by the weight itself. It also normalizes the values,
  multiplies the stream by a number after each layer, which is stored as 1, and caps a
  logit x at 30 * tanh(x / 30).

`run` computes in numpy what a Gemma computes, `check_weights` compares that with the ideal
transformer, and `save` writes a checkpoint (see checkpoint.py). Gemma's tokenizer starts
every text with the token <bos>. It has no code, so its stream is all zeros: the circuit
answers from the first position that has, after <bos>, as many tokens as it reads.
"""

import json
import math
from dataclasses import replace

import numpy as np

from . import checkpoint
from .check import expected_stream, signal_values
from .lower import OUT_WEIGHT, Lowered, NormedBlock, lower
from .lower_stream import (
    BIT_SIZE,
    KEY_VALUE_WEIGHT,
    READ_SIZE,
    ROPE_FREQUENCIES,
    ROPE_PAIRS,
    Attention,
    check_sampled_values,
    with_norms,
)
from .transformer import Transformer

NORM_EPS = 1e-6  # an RMSNorm of Gemma adds this to the mean square, so that it never divides by 0
# The error that `check_weights` allows in a feature, as a share of BIT_SIZE. The examples are
# off by less than 0.02 on the texts of the tests, and an error of 1 would turn a bit into 0.
CHECK_TOLERANCE = 0.05


class Tensors(dict):
    """The tensors of Gemma's text model, by name, as float32 arrays. Looking up a name for
    the first time makes a tensor of zeros of its shape, so the dict holds only the tensors
    that were looked up. All others are zeros and take no memory: the tensors of unused
    layers and, in Gemma 4, a second embedding of two billion numbers. `shapes` are those
    of a checkpoint, which puts `prefix` before each name here: "model." in Gemma 3, and
    more in Gemma 4, which has models for images and sound too."""

    def __init__(self, shapes: dict[str, list[int]]):
        embedding = next(name for name in shapes if name.endswith("embed_tokens.weight"))
        self.prefix = embedding.removesuffix("embed_tokens.weight")
        text = [name for name in shapes if name.startswith(self.prefix)]  # of the text model
        self.shapes = {name.removeprefix(self.prefix): shapes[name] for name in text}

    def __missing__(self, name: str) -> np.ndarray:
        self[name] = np.zeros(self.shapes[name], np.float32)
        return self[name]

    def by_checkpoint_name(self) -> dict[str, np.ndarray]:
        """The tensors that the dict holds, by their names in a checkpoint."""
        return {self.prefix + name: tensor for name, tensor in self.items()}


def norm_offset(config: dict) -> int:
    """What an RMSNorm adds to its weight before it multiplies: 1 in Gemma 3, 0 in Gemma 4."""
    return 1 if config["model_type"] == "gemma3_text" else 0


def finish(weights: Tensors, config: dict):
    """Turn the norm tensors into what a checkpoint stores: call it once, after every layer
    is written. Until here a norm tensor holds what its norm multiplies by. A norm that no
    layer wrote gets a tensor too, and multiplies by 0: with any other weight, a post-norm
    would turn the smallest noise in the weights of an unused layer into an output as large
    as a feature."""
    for name in weights.shapes:
        if name.endswith("norm.weight"):
            weights[name] -= norm_offset(config)
        if name.endswith("layer_scalar"):
            weights[name][:] = 1


def build_weights(lowered: Lowered, shapes: dict[str, list[int]], config: dict) -> Tensors:
    """The tensors of a Gemma that computes what `lowered` computes. `shapes` is the shape of
    every tensor of its checkpoint, by name, and `config` the config of its text model."""
    name = lowered.model.circuit.name
    width = len(lowered.layout[0])
    features = config["hidden_size"]
    rows = max(len(block.w_out) for block in lowered.blocks)  # the features and the dropped rows
    assert rows <= features, f"{name} needs {rows} features, and Gemma has {features}"
    attentions = attention_layers(lowered, config["num_attention_heads"])
    first_block = len(attentions) - 1  # the layer of the first block: the last that attends
    layers = first_block + len(lowered.blocks)
    available = len(config["layer_types"])
    assert layers <= available, f"{name} needs {layers} layers, and Gemma has {available}"
    attending = config["layer_types"][: len(attentions)]  # the types of the layers that attend
    rule = f"the first {len(attending)} layers of Gemma must all have sliding attention"
    tokens_read = len(lowered.model.heads)
    assert "full_attention" not in attending, f"{name} reads {tokens_read} tokens: {rule}"
    weights = Tensors(shapes)
    tokens = list(lowered.model.codes)
    stored = lowered.embedding[tokens] / math.sqrt(features)
    weights["embed_tokens.weight"][tokens, : stored.shape[1]] = stored
    norm_scale = math.sqrt(width / features)  # see lower.Lowered
    for layer, attention in enumerate(attentions):
        write_attention(weights, layer, attention, config, norm_scale)
    for layer, block in enumerate(lowered.blocks, first_block):
        write_block(weights, layer, block, norm_scale)
    # The final norm is multiplied by norm_scale and by sqrt(features): together, sqrt(width).
    weights["norm.weight"][:width] = lowered.final_norm * math.sqrt(width)
    finish(weights, config)
    return weights


def attention_layers(lowered: Lowered, heads: int) -> list[Attention]:
    """The attention of `lowered` for a model with `heads` heads in a layer: one Attention
    for each layer that attends, with norms for a stream as wide as the layout. Layer 0 has
    the first `heads` heads of `lowered`. Each later layer has one head, which stands for the
    next `heads` heads of `lowered` (see `copying_head`). Each layer changes the stream, so
    each has its own norms: with every head in layer 0, they are those of `lowered`."""
    whole = lowered.attention
    name = lowered.model.circuit.name
    size_after = lowered.sizes[0]  # the size of each feature once every head has written
    size = np.zeros(len(size_after))  # before the layer: at first, the sizes of the embedding
    size[: lowered.embedding.shape[1]] = np.abs(lowered.embedding).max(axis=0)
    layers = []
    for first_head in range(0, len(whole.w_out), heads):
        w_out = whole.w_out[first_head : first_head + heads]  # of the heads of this layer
        if first_head == 0:
            attention = replace(whole, w_query=whole.w_query[:heads], w_out=w_out)
        else:
            attention = copying_head(whole, first_head, len(w_out), name)
        layers.append(with_norms(attention, size, size_after))
        size = np.where(attention.w_out.any(axis=(0, 2)), size_after, size)
    return layers


def copying_head(whole: Attention, first_head: int, count: int, name: str) -> Attention:
    """The one head of a later layer, in place of the `count` heads of `whole` from
    `first_head` on. It attends to the token `first_head` positions back. Its value reads
    the bits that the first `count` heads of layer 0 wrote there, and it writes each bit
    where the head that it stands for writes it. It can copy only the bits that layer 0
    wrote: each bit that the circuit `name` reads of a token, it must also read of the token
    `first_head` positions nearer."""
    # Head 0 also erases the code, with -1: that is not a bit that it wrote.
    wrote = whole.w_out[:count] > 0  # [head, feature, bit]: a bit that layer 0 wrote
    w_value = KEY_VALUE_WEIGHT * np.hstack(wrote).T  # [head * bit, feature]
    stood_for = whole.w_out[first_head : first_head + count]  # of the heads that it stands for
    w_out = np.hstack(stood_for)[None]  # [1, feature, head * bit]
    unread = w_out[0].any(axis=0) & ~w_value.any(axis=1)
    nearer_token = f"the one {first_head} nearer"
    rule = f"each bit that it reads of a token must also be read of {nearer_token}"
    assert not unread.any(), f"{name} reads more tokens than a layer of Gemma has heads: {rule}"
    w_query = whole.w_query[first_head : first_head + 1]
    return replace(whole, w_query=w_query, w_value=w_value, w_out=w_out)


def rope_frequencies(config: dict, layer_type: str, pairs: int) -> np.ndarray:
    """The angles, in radians per position, by which Gemma's RoPE rotates the `pairs` pairs
    of dimensions of a head. They depend on the type of the layer: "sliding_attention", where
    a token attends to the last 512 tokens, or "full_attention", where it attends to all."""
    if "rope_parameters" in config:  # Gemma 4
        base = config["rope_parameters"][layer_type]["rope_theta"]
    elif layer_type == "sliding_attention":
        base = config["rope_local_base_freq"]
    else:
        base = config["rope_theta"]
    return base ** -(np.arange(pairs) / pairs)


def write_attention(weights: Tensors, layer: int, attention: Attention, config: dict, norm_scale):
    """Write `attention` into the tensors of layer `layer`, with each norm weight multiplied
    by `norm_scale`. Gemma divides each query and each key by its RMS, so their magnitudes in
    the tensors do not matter: a query is stored as large as the key. The post-norm divides by
    the RMS of what the heads write, so the magnitude of the output weights is free too."""
    layer_name = f"layers.{layer}."
    width = attention.w_key.shape[1]
    bits = len(attention.w_value)
    # In Gemma 4, head_dim differs from layer to layer, so it is read off this layer's tensor.
    head_dim = len(weights[layer_name + "self_attn.q_norm.weight"])
    room = f"a head has {head_dim} dimensions, too few to copy {bits} bits"
    assert bits <= head_dim, f"layer {layer}: {room}"
    half = head_dim // 2  # Gemma rotates dimensions i and i + half of a head together: pair i
    layer_type = config["layer_types"][layer]
    frequencies = rope_frequencies(config, layer_type, half)
    same = np.allclose(frequencies[:ROPE_PAIRS], ROPE_FREQUENCIES)
    assert same, f"layer {layer} has {layer_type}: its RoPE does not rotate as the lowering expects"
    dimensions = np.concatenate([np.arange(ROPE_PAIRS), half + np.arange(ROPE_PAIRS)])
    key_weight = np.abs(attention.w_key).max()
    q_proj = weights[layer_name + "self_attn.q_proj.weight"]
    o_proj = weights[layer_name + "self_attn.o_proj.weight"]
    for h, (w_query, w_out) in enumerate(zip(attention.w_query, attention.w_out)):
        start = h * head_dim  # the first dimension of this head
        stored_query = w_query / np.abs(w_query).max() * key_weight
        q_proj[start + dimensions, :width] = stored_query
        o_proj[:width, start : start + bits] = OUT_WEIGHT * w_out
    k_proj = weights[layer_name + "self_attn.k_proj.weight"]
    v_proj = weights[layer_name + "self_attn.v_proj.weight"]
    # Heads share a key and a value in groups, one group in the real models: each group gets
    # the same key and the same value.
    for start in range(0, len(k_proj), head_dim):
        k_proj[start + dimensions, :width] = attention.w_key
        v_proj[start : start + bits, :width] = attention.w_value
    # Queries and keys read only ONE, which is READ_SIZE after the pre-norm, and every query is
    # as long as that of head 0. `attention` scores a query and a key as the product of their
    # lengths times their cosine. Gemma scores them as the weight of the query norm times
    # head_dim times their cosine, over 16 in Gemma 3.
    query_length = READ_SIZE * np.linalg.norm(attention.w_query[0].sum(axis=1))
    key_length = READ_SIZE * np.linalg.norm(attention.w_key.sum(axis=1))
    score_scale = config.get("query_pre_attn_scalar", 1) ** -0.5
    query_norm = query_length * key_length / head_dim / score_scale
    weights[layer_name + "self_attn.q_norm.weight"][dimensions] = query_norm
    weights[layer_name + "self_attn.k_norm.weight"][dimensions] = 1
    pre_norm = weights[layer_name + "input_layernorm.weight"]
    post_norm = weights[layer_name + "post_attention_layernorm.weight"]
    pre_norm[:width] = attention.pre_norm * norm_scale
    post_norm[:width] = attention.post_norm * norm_scale


def write_block(weights: Tensors, layer: int, block: NormedBlock, norm_scale):
    """Write `block` into the tensors of layer `layer`, with each norm weight multiplied by
    `norm_scale`. Its dropped rows become the rows of the features after its own."""
    layer_name = f"layers.{layer}."
    count, width = block.w_gate.shape
    rows = len(block.w_out)
    gate_proj = weights[layer_name + "mlp.gate_proj.weight"]
    down_proj = weights[layer_name + "mlp.down_proj.weight"]
    up_proj = weights[layer_name + "mlp.up_proj.weight"]
    room = len(gate_proj)
    assert count <= room, f"layer {layer} needs {count} hidden units, and Gemma has {room}"
    gate_proj[:count, :width] = block.w_gate
    up_proj[:count, :width] = block.w_value
    down_proj[:rows, :count] = block.w_out
    pre_norm = weights[layer_name + "pre_feedforward_layernorm.weight"]
    post_norm = weights[layer_name + "post_feedforward_layernorm.weight"]
    pre_norm[:width] = block.pre_norm * norm_scale
    post_norm[:width] = block.post_norm * norm_scale


def rms_norm(x: np.ndarray, weight) -> np.ndarray:
    """Gemma's RMSNorm: x over the root of the mean of its squares, times `weight`."""
    return weight * x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + NORM_EPS)


def rope(x: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Rotate the queries or keys x [head, position, dimension]: at position p, the pair of
    dimensions i and i + half is rotated by the angle p * frequencies[i]."""
    angle = np.outer(np.arange(x.shape[1]), frequencies)
    first, second = np.split(x, 2, axis=-1)
    rotated_first = first * np.cos(angle) - second * np.sin(angle)
    rotated_second = second * np.cos(angle) + first * np.sin(angle)
    return np.concatenate([rotated_first, rotated_second], axis=-1)


def split_heads(projected: np.ndarray, head_dim: int) -> np.ndarray:
    """[position, head * dimension] as [head, position, dimension]."""
    return projected.reshape(len(projected), -1, head_dim).swapaxes(0, 1)


def run_attention(weights, config: dict, layer: int, x: np.ndarray) -> np.ndarray:
    """What the attention of layer `layer` adds to the stream x [position, feature]."""
    layer_name = f"layers.{layer}."
    layer_type = config["layer_types"][layer]
    offset = norm_offset(config)
    head_dim = len(weights[layer_name + "self_attn.q_norm.weight"])
    frequencies = rope_frequencies(config, layer_type, head_dim // 2)
    z = rms_norm(x, offset + weights[layer_name + "input_layernorm.weight"])
    queries = split_heads(z @ weights[layer_name + "self_attn.q_proj.weight"].T, head_dim)
    keys = split_heads(z @ weights[layer_name + "self_attn.k_proj.weight"].T, head_dim)
    values = split_heads(z @ weights[layer_name + "self_attn.v_proj.weight"].T, head_dim)
    queries = rms_norm(queries, offset + weights[layer_name + "self_attn.q_norm.weight"])
    keys = rms_norm(keys, offset + weights[layer_name + "self_attn.k_norm.weight"])
    if config["model_type"] == "gemma4_text":  # Gemma 4 normalizes the values too
        values = rms_norm(values, 1.0)
    heads_per_key = len(queries) // len(keys)  # this many heads share each key and each value
    keys = np.repeat(rope(keys, frequencies), heads_per_key, axis=0)
    scores = rope(queries, frequencies) @ keys.swapaxes(1, 2)
    scores = scores * config.get("query_pre_attn_scalar", 1) ** -0.5  # Gemma 3: over 16
    back = np.arange(len(x))[:, None] - np.arange(len(x))  # how far back each token is
    window = config["sliding_window"] if layer_type == "sliding_attention" else len(x)
    scores = np.where((back >= 0) & (back < window), scores, -np.inf)
    shares = np.exp(scores - scores.max(axis=-1, keepdims=True))
    shares = shares / shares.sum(axis=-1, keepdims=True)  # softmax over the tokens
    attended = shares @ np.repeat(values, heads_per_key, axis=0)
    joined = attended.swapaxes(0, 1).reshape(len(x), -1)  # [position, head * dimension]
    written = joined @ weights[layer_name + "self_attn.o_proj.weight"].T
    return rms_norm(written, offset + weights[layer_name + "post_attention_layernorm.weight"])


def mlp_output(weights, config: dict, layer: int, x: np.ndarray) -> np.ndarray:
    """What the MLP of layer `layer` outputs for the stream x [position, feature], before the
    post-norm."""
    layer_name = f"layers.{layer}."
    z = rms_norm(x, norm_offset(config) + weights[layer_name + "pre_feedforward_layernorm.weight"])
    gate = z @ weights[layer_name + "mlp.gate_proj.weight"].T
    # Gemma computes GELU with this tanh, which is within 0.001 of it.
    gelu = gate / 2 * (1 + np.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * gate**3)))
    hidden = gelu * (z @ weights[layer_name + "mlp.up_proj.weight"].T)
    return hidden @ weights[layer_name + "mlp.down_proj.weight"].T


def run_block(weights, config: dict, layer: int, x: np.ndarray) -> np.ndarray:
    """What the MLP block of layer `layer` adds to the stream x [position, feature]."""
    post_norm = weights[f"layers.{layer}.post_feedforward_layernorm.weight"]
    return rms_norm(mlp_output(weights, config, layer, x), norm_offset(config) + post_norm)


def run(weights, config: dict, tokens) -> list[np.ndarray]:
    """The stream [position, feature] after each layer of a Gemma 3 or a Gemma 4, for a list
    of tokens. `weights` is the tensors of its text model by name, and `config` the config
    of that model. An attention or a block whose tensors `weights` lacks adds nothing. Two
    parts of Gemma 4 are left out: what each layer adds from the tokens themselves, and its
    other RoPE in layers with full attention. A Gemma 4 of `save` uses neither: the first is
    all zeros, and its heads are in layers with sliding attention."""
    embedding = weights["embed_tokens.weight"]
    x = embedding[np.asarray(tokens, int)] * math.sqrt(config["hidden_size"])
    streams = []
    for layer in range(len(config["layer_types"])):
        layer_name = f"layers.{layer}."
        if layer_name + "self_attn.q_proj.weight" in weights:
            x = x + run_attention(weights, config, layer, x)
        if layer_name + "mlp.gate_proj.weight" in weights:
            x = x + run_block(weights, config, layer, x)
        if layer_name + "layer_scalar" in weights:  # Gemma 4
            x = x * weights[layer_name + "layer_scalar"]
        streams.append(x)
    return streams


def logits(weights, config: dict, stream: np.ndarray) -> np.ndarray:
    """The logits [position, token] of the model of `run`. `stream` is the last stream that
    `run` returned, or some positions of it: this function takes a stream, not tokens."""
    embedding = weights["embed_tokens.weight"]
    normed = rms_norm(stream, norm_offset(config) + weights["norm.weight"])
    scores = normed.astype(embedding.dtype) @ embedding.T  # no float64 copy of the embedding
    cap = config.get("final_logit_softcapping")  # Gemma 4 has one
    return cap * np.tanh(scores / cap) if cap else scores


def check_weights(weights, config: dict, lowered: Lowered, tokens, tolerance=CHECK_TOLERANCE):
    """Check on `tokens` that the Gemma of `weights`, which `build_weights` made of
    `lowered`, computes what the ideal transformer computes. At the positions of
    `signal_values` (see check.py), every feature after every block must be its size times
    what the layout says, to within `tolerance` of the size of a bit. And where the ideal
    transformer gives one token the largest logit, and that logit is above 0, so must Gemma."""
    model = lowered.model
    first_checked = len(model.heads) - 1
    values = signal_values(model, tokens)
    check_sampled_values(model.circuit, values)  # the rules that `lower` checks, on these tokens
    streams = run(weights, config, tokens)
    first_block = math.ceil(len(model.heads) / config["num_attention_heads"]) - 1  # its layer
    for depth in range(1, len(lowered.layout)):
        row = lowered.layout[depth]
        stream = streams[first_block + depth - 1][first_checked:, : len(row)]
        error = np.abs(stream - lowered.sizes[depth] * expected_stream(row, values)) / BIT_SIZE
        case, feature = np.unravel_index(error.argmax(), error.shape)
        held = row[feature] or "empty"
        where = f"depth {depth}, position {first_checked + case}, feature {feature} ({held})"
        worst = error[case, feature]
        assert worst <= tolerance, f"{where} is off by {worst:.3g} of the size of a bit"
    check_next_tokens(weights, config, model, tokens, streams[-1])


def check_next_tokens(weights, config: dict, model: Transformer, tokens, last_stream: np.ndarray):
    """Check that where the ideal transformer `model` gives one token the largest logit on
    `tokens`, and that logit is above 0, so does the Gemma of `weights`, whose last stream
    `run` gave as `last_stream`."""
    first_checked = len(model.heads) - 1
    ideal = model.logits(tokens)[first_checked:]
    found = logits(weights, config, last_stream[first_checked:]).argmax(axis=1)
    for position, (scores, got) in enumerate(zip(ideal, found.tolist()), first_checked):
        want = int(scores.argmax())
        # The token whose code the circuit gives has a logit above 0. At 0 or below, the circuit
        # gives no token, and Gemma may pick any of its tokens without a code: they all have 0.
        named = scores[want] > 0 and np.count_nonzero(scores == scores[want]) == 1
        assert not named or got == want, f"position {position}: the next token is {got}, not {want}"


def vocabulary(repo: str) -> dict[str, int]:
    """The tokens of the model `repo`: {text of a token: its id}. A space is written "▁"."""
    return json.loads(checkpoint.fetch(repo, "tokenizer.json").read_text())["model"]["vocab"]


def save(model: Transformer, out_dir, repo: str, check_on=None):
    """Save the ideal transformer `model` in the directory `out_dir` as a checkpoint of
    `repo`, a model of checkpoint.REVISIONS. Each block gets copies of its hidden units, as
    many as fit in a layer of Gemma, up to lower.MAX_COPIES. `check_on` is a list of tokens
    to check the weights on first (see `check_weights`). names.json gets the layout and, for
    each block, the names of its hidden units. model.safetensors is as long as that of
    `repo`, 0.5 GB or 10 GB. It is mostly zeros, which are holes in the file and take no
    room on disk."""
    metadata = checkpoint.metadata(repo)
    config = metadata.config.get("text_config", metadata.config)  # Gemma 4 has other models too
    lowered = lower(model, config["intermediate_size"])
    weights = build_weights(lowered, metadata.shapes, config)
    if check_on is not None:
        check_weights(weights, config, lowered, check_on)
    names = {"features": lowered.layout, "hidden": [block.units for block in lowered.blocks]}
    checkpoint.write(out_dir, metadata.config, metadata.shapes, weights.by_checkpoint_name(), names)

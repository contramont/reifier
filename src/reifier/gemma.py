"""Gemma as a target: a transformer of transformer.py in the checkpoint of gemma-3-270m or of
gemma-4-E2B.

Gemma's weights are the transformer's, on the same features: a token's embedding, tied to the
logits, the query, key, value and output of each head, and each layer, as a feed-forward block.
What Gemma adds:

- its stream holds ONE as 1, and it scales embeddings by sqrt(size), for its `size` features:
  an embedding is 1 / sqrt(size) in ONE and ± as much in each bit of the code;
- attention has norms of its own on queries and keys, in Gemma 4 on values too, and heads that
  share keys and values, 4 a layer in 270M and 8 in E2B. The first heads of the transformer are
  layer 0's. If it has more, one head of layer 1 copies as many codes again from a token
  further back, where layer 0 put them, then one of layer 2, and so on, and the blocks start in
  the last layer that attends, where all that they read is written;
- a norm follows each feed-forward block too: the unit ONE also puts a large k into ONE, where
  that norm drops it, so that what the block adds has a fixed norm whatever the bits are. That
  norm also undoes what the norm before the block is off by, so ONE need not be large here;
- its numbers are bf16, of 8 bits: a gate's sums must be small enough to be exact (see block),
  a norm must give ±1 back as ±1 (see exact), and the right token's logit is 8, for a code of
  any length, before the cap of Gemma 4, 30 tanh(logit / 30): logits near 30 are too close for
  bf16 to tell apart.

Every tensor not set so is zero, so its module gives zero, and layer scalars are one. A norm of
Gemma 3 multiplies by 1 + its weight, so one that is set is stored less 1: its entries not set
are -1, and drop their features, as zeros do in Gemma 4, and one that is not set is beside
projections that are zero. The <bos>, whose embedding is zero, stays zero at every depth, and
no head whose token is in the stream attends to it.
"""

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from .emit import EPS
from .transformer import Transformer

REVISIONS = {  # the checkpoints: a repo of huggingface.co, and the revision that weights ran on
    "unsloth/gemma-3-270m": "edcf1fe394ee1168e7c5538afd104cfb1f7947e6",
    "google/gemma-4-E2B": "433813a669639fd7eaa83155828e9000ed357d53",
}


class LazyZeros(dict):
    """Arrays for the tensors of the text model, by their names in it, which the checkpoint has
    after `prefix`: of the shapes given, made when first used, zero until set"""

    def __init__(self, shapes):
        embedding = next(k for k in shapes if k.endswith("embed_tokens.weight"))
        self.shapes, self.prefix = shapes, embedding.removesuffix("embed_tokens.weight")

    def __missing__(self, k):
        self[k] = np.zeros(self.shapes[self.prefix + k], np.float32)
        return self[k]


def round_to_bf16(x) -> np.ndarray:
    """x rounded to bfloat16, to the nearest, as float32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + (u >> 16 & 1)) & 0xFFFF0000).view(np.float32)


def build_weights(m: Transformer, shapes: dict, config: dict):
    """The weights of m as Gemma's: those that are set, by their names in the text model, the
    others being zero. shapes are checkpoint_metadata's, and config its text config. And the
    grid: m's with pads, more features, which each layer that attends sets to ±1 and nothing
    reads, as few as make every norm exact (see exact)"""
    w, plus = LazyZeros(shapes), config["model_type"] == "gemma3_text"
    for k in shapes:
        if k.endswith("layer_scalar"):
            w[k.removeprefix(w.prefix)][:] = 1
    size, grid, b = len(w["norm.weight"]), m.net.grid, len(m.v)  # b: the bits of a code
    heads, kinds = config["num_attention_heads"], config["layer_types"]
    outs = [m.o[h : h + heads] for h in range(0, len(m.o), heads)]  # of m's heads, by Gemma's layer
    n, sees = len(outs), [(b * len(o), 1) for o in outs]  # the norm after each attention,
    sees += [(sum(s is not None for s in row), n) for row in grid]  # and the norm that reads a row
    few = (p for p in range(size) if all(exact(c + k * p, size, plus) for c, k in sees))
    each = next(few, size)  # pads of each layer: a norm sees c of ±1, and the pads of k layers
    pads = np.split(len(grid[0]) + np.arange(n * each), n)
    grid = [row + [f"pad[{i}]" for i in range(n * each)] for row in grid]
    name = m.net.trace.name
    assert len(grid[0]) <= size, f"{name} needs {len(grid[0])} features, and Gemma has {size}"
    assert "full_attention" not in kinds[:n], f"{name} needs Gemma's first {n} layers to slide"
    deep = n - 1 + len(m.net.layers)  # the first block is in the last layer that attends
    assert deep <= len(kinds), f"{name} needs {deep} layers, and Gemma has {len(kinds)}"
    coded = np.flatnonzero(m.embed.any(1))  # ONE is 1 here, and Gemma scales by sqrt(size)
    w["embed_tokens.weight"][coded, : 1 + b] = np.sign(m.embed[coded]) / np.sqrt(size)
    attention(w, 0, m, m.q[:heads], m.v, outs[0], pads[0], config)
    for i, far in enumerate(outs[1:], 1):  # from the token i * heads back, what layer 0 copied
        value = np.hstack(m.o[: len(far)]).T  # of that token and those before it: their codes
        attention(w, i, m, m.q[i * heads :][:1], value, [np.hstack(far)], pads[i], config)
    for d, (layer, rows) in enumerate(zip(m.net.layers, pairwise(grid)), n - 1):
        block(w, d, layer, *rows)
    signals = sum(s is not None for s in grid[-1])  # each ±1: the square of the stream's norm
    w["norm.weight"][1 : 1 + b] = 8 / b * np.sqrt(signals)  # a code's bits: a logit of 8
    for k, v in w.items():
        if k.endswith("norm.weight"):
            v -= plus  # Gemma 3's norm multiplies by 1 + weight
    return w, grid


def exact(count: int, size: int, plus: int) -> bool:
    """Whether a norm gives count features of ±1, of size, back as ±1 in bf16. Its weight is
    sqrt(count / size), stored less plus and rounded to 8 bits. In Gemma 4, where plus is 0,
    what comes out is ±0.996 for one count in nine. In Gemma 3, where it is 1, a weight under
    1/2 is rounded to 1/256: of 640, half of the counts below 157 are off, and none above. A
    gate's sums and products are then rounded, and the bit it makes can be far off"""
    weight = round_to_bf16(np.sqrt(count / size) - plus) + plus
    return bool(round_to_bf16(weight / np.sqrt(count / size + EPS)) == 1)


def attention(w, i, m, queries, value, outputs, pads, config):
    """The attention of layer i: a head for each of queries and outputs, which are m's heads' or
    stand for them, on m's key and on value, and the first head's first dimension into each of
    pads too. They go where RoPE turns Gemma's dimensions as m's, and all read ONE. q_norm and
    k_norm make a query and a key a vector of mean square 1 times its norm's weight, and Gemma 3
    multiplies their product by `scale`, 1/16: the weight of q_norm makes a head score its token
    what it does in m, and so any other in a sliding layer's context MARGIN less. Gemma 4's
    v_norm makes a value's entries ±sqrt(head_dim / len(value)), and the output scales them back
    to ±1, or a number that bf16 has near it. Gemma 3 has no v_norm, and its output scales
    nothing: a value is as large as the norm before the attention left its token, 1 or more,
    which is the same for every token in layer 0, and a later layer has one head. So what a
    layer writes is the same in size, and the norm after makes it ±1"""
    a, n, width = f"layers.{i}.self_attn.", len(w["norm.weight"]), m.k.shape[1]
    head_dim, dims = len(w[a + "q_norm.weight"]), np.arange(len(value))
    assert len(value) <= head_dim, f"layer {i}: a head of {head_dim} cannot copy {len(value)} bits"
    theta = config.get("rope_local_base_freq")  # of RoPE in a sliding layer: Gemma 3's name,
    theta = theta or config["rope_parameters"]["sliding_attention"]["rope_theta"]  # or Gemma 4's
    pairs = np.round(-np.log(m.turns) / np.log(theta) * (head_dim // 2)).astype(int)
    assert np.allclose(theta ** -(pairs / (head_dim // 2)), m.turns), "RoPE has no such turns"
    used = np.r_[pairs, pairs + head_dim // 2]
    scale = config.get("query_pre_attn_scalar", 1) ** -0.5
    out = 1 if config["model_type"] == "gemma3_text" else np.sqrt(len(value) / head_dim)
    for head, (q, o) in enumerate(zip(queries, outputs)):
        w[a + "q_proj.weight"][head * head_dim + used, :width] = q
        w[a + "o_proj.weight"][:width, head * head_dim + dims] = out * o
    w[a + "o_proj.weight"][pads, 0] = out
    for head in range(0, len(w[a + "k_proj.weight"]), head_dim):  # heads share keys and values
        w[a + "k_proj.weight"][head + used, :width] = m.k
        w[a + "v_proj.weight"][head + dims, :width] = value
    own = np.linalg.norm(queries[0]) * np.linalg.norm(m.k)  # what a head scores its token in m
    w[a + "q_norm.weight"][used] = own / head_dim / scale
    w[a + "k_norm.weight"][used] = 1
    w[f"layers.{i}.input_layernorm.weight"][np.flatnonzero(m.k.any(0) | value.any(0))] = 1
    written = [*np.flatnonzero(np.hstack(outputs).any(1)), *pads]
    w[f"layers.{i}.post_attention_layernorm.weight"][written] = np.sqrt(len(written) / n)


def block(w, i, layer, before, row, k=2.0**14):
    """The feed-forward block of layer i: `layer` of the transformer, which makes row of the
    grid from the row before. Every feature that holds a signal is 1 in size, so the norm before
    the block has one weight. What the block adds depends on the bits, so the unit ONE, which
    gives the -1s, also puts k into ONE, where the norm after the block drops it: the norm is
    k's but for a part in k^2 over 4 for each feature written. A gate's units u+ and u- differ
    by 32 in what they sum, and bf16 holds both sums only below 4096: a larger gate could be
    wrong, silently"""
    a, n, size = f"layers.{i}.", len(layer.hidden), len(w["norm.weight"])
    most = (layer.wi[:n, 0] + abs(layer.wi[:n, 1:]).sum(1)).max()  # no gate can sum more than this
    fits = most < 4096 and all((round_to_bf16(v) == v).all() for v in (layer.wi, layer.wo))
    assert fits, f"layer {i}: a gate's weights or its sums are not exact in bf16"
    read, signals = np.flatnonzero(layer.wi.any(0)), sum(s is not None for s in before)
    w[a + "pre_feedforward_layernorm.weight"][read] = np.sqrt(signals / size)  # the stream's rms
    gate, up, down = (w[f"{a}mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
    assert n <= len(gate), f"layer {i} needs {n} hidden units, and Gemma has {len(gate)}"
    gate[:n, : len(layer.inputs)], up[:n, : len(layer.inputs)] = layer.wi[:n], layer.wi[n:]
    down[: len(layer.outputs), :n] = layer.wo
    down[0, 0] = k  # the unit ONE, the first, into ONE
    written = [f for f, (old, new) in enumerate(zip(before, row)) if old != new]
    w[a + "post_feedforward_layernorm.weight"][written] = k / np.sqrt(size)


def cache_file(repo: str) -> Path:
    """Where checkpoint_metadata keeps what it fetched of repo"""
    return Path("~/.cache/reifier", repo, REVISIONS[repo] + ".json").expanduser()


def checkpoint_metadata(repo: str) -> tuple[dict, dict]:
    """The shapes of the tensors of repo's checkpoint, at its revision of REVISIONS, and its
    config: from huggingface.co the first time, with huggingface_hub, and then from cache_file"""
    file, rev = cache_file(repo), REVISIONS[repo]
    if not file.exists():
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        meta = get_safetensors_metadata(repo, revision=rev).files_metadata["model.safetensors"]
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=rev)).read_text())
        shapes = {k: t.shape for k, t in meta.tensors.items()}
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps([shapes, config]))
    return tuple(json.loads(file.read_text()))


def save(m: Transformer, out_dir: Path, repo: str):
    """m in out_dir as a checkpoint of repo, which transformers loads from there: config.json,
    repo's own, and model.safetensors, with the names, shapes and dtype (bf16) of repo's (see
    checkpoint_metadata). Chunks of zeros are left as holes in the file, which read as zeros.
    The names go in out_dir/names.json: the grid, whose rows are the features before the first
    block and after each, and the hidden units of each block"""
    shapes, config = checkpoint_metadata(repo)
    w, grid = build_weights(m, shapes, config.get("text_config", config))  # all, if only of text
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config))
    names = {"features": grid, "hidden": [layer.hidden for layer in m.net.layers]}
    (out_dir / "names.json").write_text(json.dumps(names))
    with open(out_dir / "model.safetensors", "wb") as file:
        file.write(len(head).to_bytes(8, "little") + head)
        for k, v in w.items():
            start, v = 8 + len(head) + header[w.prefix + k]["data_offsets"][0], v.reshape(-1)
            for i in range(0, len(v), 1 << 20):
                if (chunk := v[i : i + (1 << 20)]).any():
                    file.seek(start + 2 * i)
                    bf16 = round_to_bf16(chunk).view(np.uint32) >> 16
                    file.write(bf16.astype(np.uint16).tobytes())
        file.truncate(8 + len(head) + end)

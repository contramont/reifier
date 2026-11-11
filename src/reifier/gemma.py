"""Gemma 4 as a target: a program compiled into the weights of google/gemma-4-E4B.

It is the transformer of transformer.py, on the same grid (token_grid) and with the same layers
(emit's layer, with skip connections and norms), written into Gemma's checkpoint. A token's
embedding, tied to the logits, is 1 / sqrt(2560) in the feature ONE and ± as much in a feature
for each bit of its code: Gemma scales embeddings by sqrt(2560), so ONE is 1 and a bit ±1, as
the bits that the program writes are. What Gemma adds:

- attention (layer 0) has norms of its own on queries, keys and values, and heads that share keys
  and values;
- a norm follows each feed-forward block too: the unit ONE also puts a large k into ONE, where
  that norm drops it, so that what the block adds has a fixed norm whatever the bits are. That
  norm also undoes what the norm before the block is off by, so ONE need not be large here;
- its numbers are bf16, of 8 bits: a gate's sums must be small enough to be exact (see block),
  a norm must give ±1 back as ±1 (see exact), and the right token's logit is 8 before Gemma's
  cap, 30 tanh(logit / 30), for a code of any length, as logits near 30 are too close for bf16
  to tell apart.

Every weight not set so is zero, so its module gives zero, and layer scalars are one. A <bos>
that comes first, whose embedding is zero, stays zero at every depth, and no head whose token
is in the stream attends to it.
"""

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from .emit import EPS, layer
from .trace import Trace
from .transformer import CONTEXT, PAIRS, lookback_query, runner_up, token_grid

REPO, REVISION = "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"
PREFIX = "model.language_model."
CACHE_FILE = Path("~/.cache/reifier").expanduser() / f"gemma-4-E4B-{REVISION}.json"


class LazyZeros(dict):
    """Arrays of the shapes given, made when first used, zero until set"""

    def __init__(self, shapes):
        self.shapes = shapes

    def __missing__(self, k):
        self[k] = np.zeros(self.shapes[k], np.float32)
        return self[k]


def round_to_bf16(x) -> np.ndarray:
    """x rounded to bfloat16, to the nearest, as float32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + (u >> 16 & 1)) & 0xFFFF0000).view(np.float32)


def build_weights(t: Trace, codes: dict, shapes: dict, config: dict):
    """The weights of t compiled for codes {token id: its bits}: those that are set, by the names
    and shapes of shapes, the others being zero. config is the checkpoint's text config. And the
    names: the grid, whose row d names the features after block d - 1, and the hidden units of
    each block. The grid is token_grid's with pads: more features, which the attention sets to
    ±1 and nothing reads, as few as make every norm exact (see exact)"""
    w = LazyZeros(shapes)
    for k in shapes:
        if k.endswith("layer_scalar"):
            w[k][:] = 1
    size, b, grid = shapes[PREFIX + "norm.weight"][0], len(t.outputs), token_grid(t)
    code, ins = list(range(1, 1 + b)), [grid[0].index(x) for x in t.inputs]
    counts = [len(ins), *(sum(s is not None for s in row) for row in grid)]  # the ±1s a norm sees
    pads = next((p for p in range(size) if all(exact(c + p, size) for c in counts)), size)
    grid = [row + [f"pad[{i}]" for i in range(pads)] for row in grid]
    assert len(grid[0]) <= size, f"{t.name} needs {len(grid[0])} features, and Gemma has {size}"
    emb = w[PREFIX + "embed_tokens.weight"]
    emb[list(codes), 0] = 1 / np.sqrt(size)  # Gemma scales embeddings by sqrt(size): ONE is 1
    emb[list(codes), 1 : 1 + b] = (2 * np.array(list(codes.values())) - 1) / np.sqrt(size)
    writes = np.split(np.array(ins), len(ins) // b)[::-1]
    rope_theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]  # layer 0 slides
    attention(w, code, writes, range(len(grid[0]) - pads, len(grid[0])), rope_theta)
    hidden = [block(w, i, t.mlps, before, row) for i, (before, row) in enumerate(pairwise(grid))]
    signals = sum(s is not None for s in grid[-1])  # each ±1: the square of the stream's norm
    w[PREFIX + "norm.weight"][code] = 8 / b * np.sqrt(signals)  # a code's bits: a logit of 8
    return w, grid, hidden


def exact(count: int, size: int) -> bool:
    """Whether a norm gives count features of ±1, of size, back as ±1 in bf16. Its weight is
    sqrt(count / size) rounded to 8 bits, and for one count in nine what comes out is ±0.996:
    a gate's sums and products are then rounded, and the bit it makes can be far off"""
    weight = round_to_bf16(np.sqrt(count / size))
    return bool(round_to_bf16(weight / np.sqrt(count / size + EPS)) == 1)


def attention(w, code, writes, pads, rope_theta, margin=20.0):
    """The attention of layer 0: head h copies the features `code` of the token h back into the
    features writes[h], and head 0 the first of them into each of pads too. Its query and key
    are as in lookback_query, on the PAIRS of dimensions that RoPE turns fastest, and both read
    ONE. q_norm and k_norm make each a vector of mean square 1 times its norm's weight, so the
    token h + e back scores head_dim r K(e), for the weight r of q_norm and the mean cosine K(e)
    of the pairs' turns by e, which is 1 at e = 0: this r makes the best wrong token of the
    CONTEXT that a sliding layer sees score `margin` less. v_norm makes a value's entries
    ±sqrt(head_dim / len(code)), and the output scales them back to ±1, or a number that bf16
    has near it: all are the same in size, and the norm after the attention makes them ±1"""
    a, n = PREFIX + "layers.0.self_attn.", len(w[PREFIX + "norm.weight"])
    head_dim, bits, fast = len(w[a + "q_norm.weight"]), np.arange(len(code)), np.arange(PAIRS)
    turns = rope_theta ** -(fast / (head_dim // 2))  # radians per position
    used = np.r_[fast, fast + head_dim // 2]
    for h, features in enumerate(writes):
        w[a + "q_proj.weight"][h * head_dim + used, 0] = lookback_query(h, turns)
        w[a + "o_proj.weight"][features, h * head_dim + bits] = np.sqrt(len(code) / head_dim)
    w[a + "o_proj.weight"][pads, 0] = np.sqrt(len(code) / head_dim)
    for head in range(0, len(w[a + "k_proj.weight"]), head_dim):  # heads share keys and values
        w[a + "k_proj.weight"][head + fast, 0] = 1
        w[a + "v_proj.weight"][head + bits, code] = 1
    w[a + "q_norm.weight"][used] = margin / (1 - runner_up(turns, CONTEXT)) / head_dim
    w[a + "k_norm.weight"][used] = 1
    w[PREFIX + "layers.0.input_layernorm.weight"][[0, *code]] = 1
    written = [*np.concatenate(writes), *pads]
    w[PREFIX + "layers.0.post_attention_layernorm.weight"][written] = np.sqrt(len(written) / n)


def block(w, i, mlps, before, row, k=2.0**14) -> list[str]:
    """The feed-forward block of layer i, which makes row of the grid from the row before with
    mlps: the layer of emit.py, with skip connections and norms. Every feature that holds a
    signal is 1 in size, so the norm before the block has one weight. What the block adds
    depends on the bits, so the unit ONE, which gives the -1s, also puts k into ONE, where the
    norm after the block drops it: the norm is k's but for a part in k^2 over 4 for each feature
    written. A gate's units u+ and u- differ by 32 in what they sum, and bf16 holds both sums
    only below 4096: a larger gate could be wrong, silently. Gives the names of its hidden units"""
    a, m = f"{PREFIX}layers.{i}.", layer(mlps, before, row, "geglu", skip=True, norm=True)
    n, width, size = len(m.hidden), len(row), len(w[PREFIX + "norm.weight"])
    most = (m.wi[:n, 0] + abs(m.wi[:n, 1:]).sum(1)).max()  # no gate can sum more than this
    fits = most < 4096 and all((round_to_bf16(v) == v).all() for v in (m.wi, m.wo))
    assert fits, f"block {i}: a gate's weights or its sums are not exact in bf16"
    read, signals = np.flatnonzero(m.wi.any(0)), sum(s is not None for s in before)
    w[a + "pre_feedforward_layernorm.weight"][read] = np.sqrt(signals / size)  # the stream's rms
    gate, up, down = (w[f"{a}mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
    gate[:n, :width], up[:n, :width] = m.wi[:n], m.wi[n:]
    down[:width, :n] = m.wo
    down[0, 0] = k  # the unit ONE, the first, into ONE
    written = [f for f, (old, new) in enumerate(zip(before, row)) if old != new]
    w[a + "post_feedforward_layernorm.weight"][written] = k / np.sqrt(size)
    return m.hidden


def checkpoint_metadata() -> tuple[dict, dict]:
    """The shapes of the tensors of the checkpoint of REPO at REVISION, and its text config: from
    huggingface.co the first time, with huggingface_hub, and then from CACHE_FILE"""
    if not CACHE_FILE.exists():
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        meta = get_safetensors_metadata(REPO, revision=REVISION).files_metadata["model.safetensors"]
        config = Path(hf_hub_download(REPO, "config.json", revision=REVISION)).read_text()
        shapes = {k: t.shape for k, t in meta.tensors.items()}
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps([shapes, json.loads(config)["text_config"]]))
    return tuple(json.loads(CACHE_FILE.read_text()))


def save(t: Trace, codes: dict, out_dir: Path):
    """t, compiled for codes, in out_dir/model.safetensors, with the names, shapes and dtype
    (bf16) of the checkpoint of REPO at REVISION (see checkpoint_metadata). Chunks of zeros are
    left as holes in the file, which read as zeros. The names go in out_dir/names.json: the
    grid, whose rows are the features before block 0 and after each block, and the hidden units
    of each block"""
    shapes, config = checkpoint_metadata()
    w, grid, hidden = build_weights(t, codes, shapes, config)
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "names.json").write_text(json.dumps({"features": grid, "hidden": hidden}))
    with open(out_dir / "model.safetensors", "wb") as file:
        file.write(len(head).to_bytes(8, "little") + head)
        for k, v in w.items():
            start, v = 8 + len(head) + header[k]["data_offsets"][0], v.reshape(-1)
            for i in range(0, len(v), 1 << 20):
                if (chunk := v[i : i + (1 << 20)]).any():
                    file.seek(start + 2 * i)
                    file.write(
                        (round_to_bf16(chunk).view(np.uint32) >> 16).astype(np.uint16).tobytes()
                    )
        file.truncate(8 + len(head) + end)

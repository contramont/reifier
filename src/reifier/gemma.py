"""Gemma 4 as a target: a program compiled into the weights of google/gemma-4-E4B.

It is the transformer of transformer.py, on the same grid, with Gemma's parts around it: row 0
is ONE, <bos>, the token's own code and the program's inputs, and the outputs are pinned over the
code, which the final norm alone reads. A token's embedding, tied to the logits, is 1/32 in the
feature ONE and ±1/32 in a feature for each bit, and <bos>'s is 1/32 in a feature of its own:
Gemma scales embeddings by sqrt(2560), so the token's bits are then about as big as those that
the program writes over them. check_transformer, on transformer(t, codes), checks the same
program on the same grid without the model. What Gemma adds:

- attention (layer 0) has norms of its own on queries, keys and values, heads that share keys and
  values, and <bos>, which a head attends to when there is no token h back (bos_lookback_qk);
- the features of the embedding are sqrt(2560) / 32 in size, not 1: magnitude keeps each
  feature's size, the pre-norms make each feature that is read ±1, and an erase gives back as much;
- a norm follows each feed-forward block too: the unit ONE also puts a large k into ONE, where
  that norm drops it, so that what the block adds has a fixed norm whatever the bits are. That
  norm also undoes what the norm before the block is off by, so ONE need not be large here.

Every weight not set so is zero, so its module gives zero, and layer scalars are one.
"""

import json
import math
from pathlib import Path

import numpy as np

from .emit import layer
from .grid import ONE, check_grid
from .trace import Trace
from .transformer import lookback_qk, runner_up, token_grid

REPO, REVISION = "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"
PREFIX = "model.language_model."
CACHE_FILE = Path("~/.cache/reifier").expanduser() / f"gemma-4-E4B-{REVISION}.json"
BOS, FIRST_BIT = 1, 2  # the features: ONE is 0, <bos> 1, then the bits of the token's code


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


def build_weights(t: Trace, codes: dict, shapes: dict, bos_token_id: int, rope_theta: float):
    """The weights of t compiled for codes {token id: its bits}: those that are set, by the names
    and shapes of shapes, the others being zero. bos_token_id is <bos>'s token, and rope_theta
    RoPE's base in the sliding layers, which layer 0 is. And the names: the grid, whose row d
    names the features after block d - 1, and the hidden units of each block"""
    w = LazyZeros(shapes)
    for k in shapes:
        if k.endswith("layer_scalar"):
            w[k][:] = 1
    hidden_size, b = shapes[PREFIX + "norm.weight"][0], len(t.outputs)
    code, g = list(range(FIRST_BIT, FIRST_BIT + b)), token_grid(t, [ONE, "<bos>"])
    if len(g.rows[0]) > hidden_size:
        raise ValueError(f"{t.name} needs {len(g.rows[0])} features, and Gemma has {hidden_size}")
    ids, eye, emb = list(codes), np.eye(hidden_size), w[PREFIX + "embed_tokens.weight"]
    emb[ids, 0], emb[bos_token_id, BOS] = 1 / 32, 1 / 32
    emb[ids, FIRST_BIT : FIRST_BIT + b] = (2 * np.array(list(codes.values())) - 1) / 32
    # the size of each feature's values, ±magnitude (0: nothing has written it)
    magnitude = np.zeros(hidden_size)
    magnitude[[0, *code]] = round_to_bf16(np.sqrt(hidden_size)) / 32  # as Gemma scales embeddings
    ins = [g.rows[0].index(x) for x in t.inputs]
    tokens = np.split(np.array(ins), len(ins) // b)
    for h, features in enumerate(tokens[::-1]):  # head h copies the token h back
        set_attention_head(w, h, *bos_lookback_qk(w, h, rope_theta), eye[code], eye[features])
    w[PREFIX + "layers.0.input_layernorm.weight"][[0, BOS, *code]] = 1  # q, k, v
    w[PREFIX + "layers.0.post_attention_layernorm.weight"][ins] = np.sqrt(len(ins) / hidden_size)
    magnitude[ins] = 1
    mlps, hidden = check_grid(t, g), []
    for d in range(1, len(g.rows)):
        hidden.append(set_mlp(w, d - 1, mlps[d], g.rows[d - 1], g.rows[d], magnitude))
    w[PREFIX + "norm.weight"][code] = 32 * np.sqrt(np.mean(magnitude**2))  # each bit ±1 a logit
    return w, g, hidden


def bos_lookback_qk(w, h, rope_theta, margin=20.0):
    """The query and key of a head of layer 0 that attends to the token h back, or to <bos> when
    there is none. Every token of codes has the same key, 1 in each of the 8 pairs of dimensions
    RoPE turns fastest, and <bos> 1 in the pair it turns least. The query is that key turned back
    h positions, as lookback_qk gives it, and sqrt(8) c in the slow pair, as k_norm makes <bos>'s
    key sqrt(8) times as long there: the score of the token h + e back is s K(e) and that of
    <bos> about s c, where K(e), the mean cosine of the 8 pairs' turns by e, is 1 at 0 and at most
    2 c - 1 elsewhere in the 512-token sliding window. The norms make s = head_dim r / sqrt(1 +
    c^2) for the weight r of q_norm, which the layer's heads share, and this r makes s (1 - c) =
    margin"""
    a = PREFIX + "layers.0.self_attn."
    head_dim, eye = len(w[a + "q_norm.weight"]), np.eye(len(w[PREFIX + "norm.weight"]))
    fast, slow = np.arange(8), head_dim // 2 - 1
    turns = rope_theta ** -(fast / (head_dim // 2))  # radians per position
    c = (1 + runner_up(turns, 512)) / 2
    used = [*fast, *fast + head_dim // 2, slow]
    q, k = np.zeros((2, head_dim, len(eye)))
    q[used] = np.outer([*lookback_qk(h, turns)[0], np.sqrt(8) * c], eye[0] + eye[BOS])
    k[fast], k[slow] = eye[0], eye[BOS]
    w[a + "q_norm.weight"][used] = margin / (1 - c) * np.sqrt(1 + c * c) / head_dim
    w[a + "k_norm.weight"][used] = 1
    return q, k


def set_attention_head(w, h, q, k, values, writes):
    """Head h of layer 0, its query and key reading the residual along the rows of q and k, and
    its value along the rows of values, ±1 each, which go along the rows of writes. Heads that
    share a kv head must share k and values too. v_norm makes the value's entries
    ±sqrt(head_dim / len(values)), so writes scale them back to ±1"""
    a = PREFIX + "layers.0.self_attn."
    head_dim, rows = len(w[a + "q_norm.weight"]), len(w[a + "q_proj.weight"])
    kv = h * len(w[a + "k_proj.weight"]) // rows  # its kv head
    w[a + "q_proj.weight"][h * head_dim : (h + 1) * head_dim] = q  # an error past the last head
    w[a + "k_proj.weight"][kv * head_dim : (kv + 1) * head_dim] = k
    w[a + "v_proj.weight"][kv * head_dim : kv * head_dim + len(values)] = values
    w[a + "o_proj.weight"][:, h * head_dim : h * head_dim + len(values)] = (
        np.sqrt(len(values) / head_dim) * writes.T
    )


def set_mlp(w, i, mlps, below, row, magnitude, k=2.0**14) -> tuple[str, ...]:
    """The feed-forward block of layer i, which makes row of the grid from the row below with
    mlps: the layer of emit.py, with skip connections and norms, on features whose sizes
    magnitude gives, which it updates. Its erases, the last hidden units, each read a feature as
    ±1 and must give back as much as the feature holds. What the block adds depends on the bits,
    so the unit ONE, which gives the -1s, also puts k into ONE, where the post-norm drops it: the
    norm is k's but for a part in k^2 over 4 for each feature written. Gives the names of its
    hidden units"""
    a, at = f"{PREFIX}layers.{i}.", {s: f for f, s in enumerate(below) if s}
    m = layer(mlps, below, row, "geglu", skip=True, norm=True)
    written = [f for f, (old, new) in enumerate(zip(below, row)) if old != new]
    erased = [f for f in written if below[f]]
    n, width = len(m.hidden), len(row)
    assert set(m.hidden[n - len(erased) :]) == {f"erase {below[f]}" for f in erased}
    m.wi[2 * n - len(erased) :] *= magnitude[:width]
    read = [0, *(at[x] for c in mlps for x in c.inputs), *erased]  # all that gate and up read
    w[a + "pre_feedforward_layernorm.weight"][read] = (
        np.sqrt(np.mean(magnitude**2)) / magnitude[read]
    )
    gate, up, down = (w[f"{a}mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
    gate[:n, :width], up[:n, :width], down[:width, :n], down[0, 0] = m.wi[:n], m.wi[n:], m.wo, k
    w[a + "post_feedforward_layernorm.weight"][written] = k / np.sqrt(len(magnitude))
    magnitude[written] = 1
    return m.hidden


def checkpoint_metadata(cache_file: Path = CACHE_FILE) -> tuple[dict, dict]:
    """The shapes of the tensors of the checkpoint of REPO at REVISION, and its text config: from
    huggingface.co the first time, with huggingface_hub, and then from cache_file"""
    if not cache_file.exists():
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        meta = get_safetensors_metadata(REPO, revision=REVISION).files_metadata["model.safetensors"]
        config = Path(hf_hub_download(REPO, "config.json", revision=REVISION)).read_text()
        shapes = {k: t.shape for k, t in meta.tensors.items()}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps([shapes, json.loads(config)["text_config"]]))
    return tuple(json.loads(cache_file.read_text()))


def save(t: Trace, codes: dict, out_dir: Path, cache_file: Path = CACHE_FILE):
    """t, compiled for codes, in out_dir/model.safetensors, with the names, shapes and dtype
    (bf16) of the checkpoint of REPO at REVISION, which cache_file keeps (see
    checkpoint_metadata). Chunks of zeros are left as holes in the file, which read as zeros.
    The names go in out_dir/names.json: those of the features after each block, and of each
    block's hidden units"""
    shapes, config = checkpoint_metadata(cache_file)
    rope_theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]
    w, g, hidden = build_weights(t, codes, shapes, config["bos_token_id"], rope_theta)
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "names.json").write_text(json.dumps({"features": g.rows, "hidden": hidden}))
    with open(out_dir / "model.safetensors", "wb") as file:
        file.write(len(head).to_bytes(8, "little") + head)
        for k, v in w.items():
            at, v = 8 + len(head) + header[k]["data_offsets"][0], v.reshape(-1)
            for i in range(0, len(v), 1 << 20):
                if (chunk := v[i : i + (1 << 20)]).any():
                    file.seek(at + 2 * i)
                    file.write(
                        (round_to_bf16(chunk).view(np.uint32) >> 16).astype(np.uint16).tobytes()
                    )
        file.truncate(8 + len(head) + end)

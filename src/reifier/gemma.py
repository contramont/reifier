"""Gemma 4 as a target: a program compiled into the weights of google/gemma-4-E4B.

A token 0x10000 + n carries the b bits of n, and greedy generation maps the last tokens to the
next: the program maps their bits, the oldest token's first and each token's least significant
first, to the b bits of the token to emit. A token's embedding, tied to the logits, is 1/32 in
the column ONE and ±1/32 in a column for each bit, and <bos>'s is 1/32 in a column of its own:
Gemma scales embeddings by sqrt(2560), so the token's bits are then about as big as those that
the program writes over them.

The residual stream holds bits as ±1, one column each, so that it has a fixed norm whatever the
bits are, and Gemma's norms are fixed scalings: the pre-norms make each column ±1, ONE 1.

- attention (layer 0): head h attends to the token h back and copies its bits (see back);
- the program's units, in the feed-forward blocks of layers 0, 1, ...: a unit is in the block
  after the last that gives one of its inputs, and its bits go over columns whose signals no
  later block reads, or that no block has used. A bit y goes over old as 2 y - 1 - old;
- the program's outputs go over the token's own bits, which the final norm alone reads, so the
  logits are the number of bits a token shares with the outputs less the number it does not.

Every weight not set so is zero, so its module gives zero, and layer scalars are one.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import Graph, flatten, lower

REPO, REVISION = "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"
LM = "model.language_model."
CACHE = Path("~/.cache/reifier").expanduser() / f"gemma-4-E4B-{REVISION}.json"
FIRST, ONE, BOS, BITS = 0x10000, 0, 1, 2  # the token of 0; the columns ONE, <bos>, then the bits


class Zeros(dict):
    """Arrays of the shapes given, made when first used, zero until set"""

    def __init__(self, shapes):
        super().__init__()
        self.shapes = shapes

    def __missing__(self, k):
        self[k] = np.zeros(self.shapes[k], np.float32)
        return self[k]


def bf16(x) -> np.ndarray:
    """x rounded to bfloat16, to the nearest, as float32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + (u >> 16 & 1)) & 0xFFFF0000).view(np.float32)


def weights(program: Graph, shapes: dict, bos: int, theta: float) -> dict:
    """The weights set, of the names and shapes in shapes (the others are zero): program compiled,
    for the token id bos of <bos> and theta, RoPE's base in the sliding layers, which layer 0 is"""
    w = Zeros(shapes)
    for k in shapes:
        if k.endswith("layer_scalar"):
            w[k][:] = 1
    dim, b = shapes[LM + "norm.weight"][0], len(program.outputs)
    n, eye, emb = np.arange(2**b), np.eye(dim), w[LM + "embed_tokens.weight"]
    emb[FIRST + n, ONE], emb[bos, BOS] = 1 / 32, 1 / 32
    emb[FIRST + n, BITS : BITS + b] = (2 * (n[:, None] >> np.arange(b) & 1) - 1) / 32
    mag = np.zeros(dim)  # the size of each column's values, ±mag (0: no block has used it)
    mag[[ONE, *range(BITS, BITS + b)]] = bf16(np.sqrt(dim)) / 32  # as Gemma scales embeddings
    ins, blocks = schedule(program, dim)
    tokens = np.split(np.array(ins), len(ins) // b)
    for h, cols in enumerate(tokens[::-1]):  # head h copies the token h back
        attend(w, h, *back(w, h, theta), eye[BITS : BITS + b], eye[cols])
    w[LM + "layers.0.input_layernorm.weight"][:] = 1
    w[LM + "layers.0.post_attention_layernorm.weight"][:] = np.sqrt(len(ins) / dim)
    mag[ins] = 1
    for p, units in enumerate(blocks):
        feedforward(w, p, units, mag)
    w[LM + "norm.weight"][BITS : BITS + b] = 32 * np.sqrt(np.mean(mag**2))  # each bit ±1 a logit
    return w


def schedule(program: Graph, dim: int):
    """The columns of program's inputs, and its units by block, with the columns each reads and
    writes: program's outputs go into the token's bits, other signals into the columns after"""
    g = flatten(lower(program, "geglu"))
    depth, last = dict.fromkeys(g.inputs, 0), {}  # the block each signal is given and last read in
    for node in g.nodes:
        d = 1 + max((depth[x] for x in node.inputs), default=0)
        depth |= dict.fromkeys(node.outputs, d)
        last |= {x: max(last.get(x, 0), d) for x in node.inputs}
    blocks = [[] for _ in range(max(depth.values()))]
    for node in g.nodes:
        blocks[depth[node.outputs[0]] - 1].append(node)
    ends, outputs = defaultdict(list), set(g.outputs)  # the signals each block reads last
    for x in depth:
        if x not in outputs:  # a signal no block reads ends after its own
            ends[last.get(x, depth[x] + 1)].append(x)
    free = list(range(dim - 1, BITS + len(g.outputs) - 1, -1))
    col = {x: free.pop() for x in g.inputs} | {y: BITS + i for i, y in enumerate(g.outputs)}
    for d, nodes in enumerate(blocks, 1):
        free += [col[x] for x in ends[d]]
        col |= {y: free.pop() for node in nodes for y in node.outputs if y not in col}
    units = [
        [(n.circuit, [col[x] for x in n.inputs], [col[y] for y in n.outputs]) for n in nodes]
        for nodes in blocks
    ]
    return [col[x] for x in g.inputs], units


def back(w, d, theta, margin=20.0):
    """The query and key of a head of layer 0 that attends to the token d back, or to <bos> when
    there is none. Every token 0x10000 + n has the same key, 1 in each of the 8 pairs of dimensions
    RoPE turns fastest, and <bos> 1 in the pair it turns least. The query is that key turned back
    d positions, and sqrt(8) c in the slow pair, as k_norm makes <bos>'s key sqrt(8) times as
    long there: the score of the token d + e back is s K(e) and that of <bos> about s c, where
    K(e), the mean cosine of the 8 pairs' turns by e, is 1 at 0 and at most 2 c - 1 elsewhere in
    the 512-token sliding window. The norms make s = hd r / sqrt(1 + c^2) for the weight r of
    q_norm, which the layer's heads share, and this r makes s (1 - c) = margin"""
    a = LM + "layers.0.self_attn."
    hd, eye = len(w[a + "q_norm.weight"]), np.eye(len(w[LM + "norm.weight"]))
    fast, slow = np.arange(8), hd // 2 - 1
    turns = theta ** -(fast / (hd // 2))  # radians per position
    c = (1 + np.cos(np.outer(np.arange(1, 512), turns)).mean(1).max()) / 2
    used = [*fast, *fast + hd // 2, slow]
    q, k = np.zeros((2, hd, len(eye)))
    q[used] = np.outer(
        [*np.cos(turns * d), *-np.sin(turns * d), np.sqrt(8) * c], eye[ONE] + eye[BOS]
    )
    k[fast], k[slow] = eye[ONE], eye[BOS]
    w[a + "q_norm.weight"][used] = margin / (1 - c) * np.sqrt(1 + c * c) / hd
    w[a + "k_norm.weight"][used] = 1
    return q, k


def attend(w, h, q, k, values, writes):
    """Head h of layer 0, its query and key reading the residual along the rows of q and k, and
    its value along the rows of values, ±1 each, which go along the rows of writes. Heads that
    share a kv head must share k and values too. v_norm makes the value's entries
    ±sqrt(hd / len(values)), so writes scale them back to ±1"""
    a = LM + "layers.0.self_attn."
    hd, rows = len(w[a + "q_norm.weight"]), len(w[a + "q_proj.weight"])
    g = h * len(w[a + "k_proj.weight"]) // rows  # its kv head
    w[a + "q_proj.weight"][h * hd : (h + 1) * hd] = q  # an error past the last head
    w[a + "k_proj.weight"][g * hd : (g + 1) * hd] = k
    w[a + "v_proj.weight"][g * hd : g * hd + len(values)] = values
    w[a + "o_proj.weight"][:, h * hd : h * hd + len(values)] = np.sqrt(len(values) / hd) * writes.T


def feedforward(w, p, units, mag, k=2.0**14, sharp=32.0):
    """The feed-forward block of layer p: units [(geglu unit, columns read, columns written)] on
    bits, ±1 in columns whose sizes mag gives, which it updates. A bit y goes over a column that
    holds old as 2 y - 1 - old, with -old from a hidden unit gelu(sharp ONE) (-old / sharp). What
    the block adds then depends on the bits, so hidden unit 0, which gives the -1s, also puts k
    into ONE, where the post-norm drops it: the norm is k's but for a part in k^2 / 6400 or so"""
    a = f"{LM}layers.{p}."
    gate, up, down = (w[f"{a}mlp.{m}_proj.weight"] for m in ("gate", "up", "down"))
    w[a + "pre_feedforward_layernorm.weight"][mag > 0] = np.sqrt(np.mean(mag**2)) / mag[mag > 0]
    gate[0, ONE], up[0, ONE], down[ONE, 0], h = sharp, 1 / sharp, k, 1
    for u, ins, outs in units:
        n = u.hidden
        for m, half in ((gate, slice(0, n)), (up, slice(n, 2 * n))):
            np.add.at(m[h : h + n].T, ins, u.wi[half].T / 2)  # a bit is (x + ONE) / 2
            m[h : h + n, ONE] += u.wi[half].sum(1) / 2 + u.b[half]
        down[outs, h : h + n] = 2 * u.wo
        h += n
    written = [c for _, _, outs in units for c in outs]
    old = [c for c in written if mag[c]]
    e = h + np.arange(len(old))  # the hidden units erasing them
    gate[e, ONE], up[e, old], down[old, e], down[written, 0] = sharp, -mag[old] / sharp, 1, -1
    w[a + "post_feedforward_layernorm.weight"][written] = k / np.sqrt(len(mag))
    mag[written] = 1


def checkpoint(cache: Path = CACHE) -> tuple[dict, dict]:
    """The shapes of the tensors of the checkpoint of REPO at REVISION, and its text config: from
    huggingface.co the first time, with huggingface_hub, and then from the file cache"""
    if not cache.exists():
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        meta = get_safetensors_metadata(REPO, revision=REVISION).files_metadata["model.safetensors"]
        config = Path(hf_hub_download(REPO, "config.json", revision=REVISION)).read_text()
        shapes = {k: t.shape for k, t in meta.tensors.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([shapes, json.loads(config)["text_config"]]))
    return tuple(json.loads(cache.read_text()))


def save(program: Graph, out: Path, cache: Path = CACHE):
    """program, compiled, in out/model.safetensors, with the names, shapes and dtype (bf16) of the
    checkpoint of REPO at REVISION, which cache keeps (see checkpoint). Chunks of zeros are left
    as holes in the file, which read as zeros"""
    shapes, config = checkpoint(cache)
    theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]
    w = weights(program, shapes, config["bos_token_id"], theta)
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out.mkdir(exist_ok=True)
    with open(out / "model.safetensors", "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head)
        for k, v in w.items():
            at, v = 8 + len(head) + header[k]["data_offsets"][0], v.reshape(-1)
            for i in range(0, len(v), 1 << 20):
                if (chunk := v[i : i + (1 << 20)]).any():
                    f.seek(at + 2 * i)
                    f.write((bf16(chunk).view(np.uint32) >> 16).astype(np.uint16).tobytes())
        f.truncate(8 + len(head) + end)

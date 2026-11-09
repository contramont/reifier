"""Gemma 4 as a target: a program compiled into the weights of google/gemma-4-E4B.

A token carries the b bits of a number n: 0x10000 + n, or the token a table gives for n, such as
the tokenizer's own token for the digit n. Greedy generation maps the last tokens to the next: the
program maps their bits, the oldest token's first and each token's least significant first, to the
b bits of the token to emit. A token's embedding, tied to the logits, is 1/32 in the feature ONE
and ±1/32 in a feature for each bit, and <bos>'s is 1/32 in a feature of its own, and ±1/32 in the
bits' if the table gives it an n too: Gemma scales embeddings by sqrt(2560), so the token's bits
are then about as big as those that the program writes over them.

The residual stream holds bits as ±1, one feature each, so that it has a fixed norm whatever the
bits are, and Gemma's norms are fixed scalings: the pre-norms make each feature ±1, ONE 1.

- attention (layer 0): head h attends to the token h back and copies its bits, or to <bos> where
  there is no token that far back (see lookback_qk);
- the program's MLPs, in the feed-forward blocks of layers 0, 1, ...: an MLP is in the block
  after the last that gives one of its inputs, and its bits go over features whose signals no
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
PREFIX = "model.language_model."
CACHE_FILE = Path("~/.cache/reifier").expanduser() / f"gemma-4-E4B-{REVISION}.json"
# the token of 0; the features ONE, <bos>, then the bits
TOKEN_OFFSET, ONE, BOS, FIRST_BIT = 0x10000, 0, 1, 2


class LazyZeros(dict):
    """Arrays of the shapes given, made when first used, zero until set"""

    def __init__(self, shapes):
        super().__init__()
        self.shapes = shapes

    def __missing__(self, k):
        self[k] = np.zeros(self.shapes[k], np.float32)
        return self[k]


def round_to_bf16(x) -> np.ndarray:
    """x rounded to bfloat16, to the nearest, as float32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + (u >> 16 & 1)) & 0xFFFF0000).view(np.float32)


def build_weights(
    program: Graph, shapes: dict, bos_token_id: int, rope_theta: float, tokens: dict | None = None
) -> dict:
    """The weights set, of the names and shapes in shapes (the others are zero): program
    compiled, for the token id bos_token_id of <bos> and rope_theta, RoPE's base in the sliding
    layers, which layer 0 is. tokens maps each n the tokens carry to its token id: 0x10000 + n
    for every n of b bits if not given"""
    w = LazyZeros(shapes)
    for k in shapes:
        if k.endswith("layer_scalar"):
            w[k][:] = 1
    hidden_size, b = shapes[PREFIX + "norm.weight"][0], len(program.outputs)
    tokens = tokens or {n: TOKEN_OFFSET + n for n in range(2**b)}
    n, ids = np.array(list(tokens)), np.array(list(tokens.values()))
    eye, emb = np.eye(hidden_size), w[PREFIX + "embed_tokens.weight"]
    emb[ids, ONE], emb[bos_token_id, BOS] = (ids != bos_token_id) / 32, 1 / 32
    emb[ids, FIRST_BIT : FIRST_BIT + b] = (2 * (n[:, None] >> np.arange(b) & 1) - 1) / 32
    # the size of each feature's values, ±magnitude (0: no block has used it)
    magnitude = np.zeros(hidden_size)
    # as Gemma scales embeddings
    magnitude[[ONE, *range(FIRST_BIT, FIRST_BIT + b)]] = round_to_bf16(np.sqrt(hidden_size)) / 32
    ins, layers = schedule(program, hidden_size)
    by_token = np.split(np.array(ins), len(ins) // b)
    for h, features in enumerate(by_token[::-1]):  # head h copies the token h back
        set_attention_head(
            w, h, *lookback_qk(w, h, rope_theta), eye[FIRST_BIT : FIRST_BIT + b], eye[features]
        )
    # q, k, v
    w[PREFIX + "layers.0.input_layernorm.weight"][[ONE, BOS, *range(FIRST_BIT, FIRST_BIT + b)]] = 1
    w[PREFIX + "layers.0.post_attention_layernorm.weight"][ins] = np.sqrt(len(ins) / hidden_size)
    magnitude[ins] = 1
    for layer, mlps in enumerate(layers):
        set_mlp(w, layer, mlps, magnitude)
    # each bit ±1 a logit
    w[PREFIX + "norm.weight"][FIRST_BIT : FIRST_BIT + b] = 32 * np.sqrt(np.mean(magnitude**2))
    return w


def schedule(program: Graph, hidden_size: int):
    """The features of program's inputs, and its MLPs by layer, with the features each reads and
    writes: program's outputs go into the token's bits, other signals into the features after"""
    g = flatten(lower(program, "geglu"))
    # the layer each signal is given in, and the last it is read in
    depth, last_read = dict.fromkeys(g.inputs, 0), {}
    for node in g.nodes:
        d = 1 + max((depth[x] for x in node.inputs), default=0)
        depth |= dict.fromkeys(node.outputs, d)
        last_read |= {x: max(last_read.get(x, 0), d) for x in node.inputs}
    layers = [[] for _ in range(max(depth.values()))]
    for node in g.nodes:
        layers[depth[node.outputs[0]] - 1].append(node)
    read_last, outputs = defaultdict(list), set(g.outputs)  # the signals each layer reads last
    for x in depth:
        if x not in outputs:  # a signal no layer reads ends after its own
            read_last[last_read.get(x, depth[x] + 1)].append(x)
    free = list(range(hidden_size - 1, FIRST_BIT + len(g.outputs) - 1, -1))
    feature = {x: free.pop() for x in g.inputs} | {
        y: FIRST_BIT + i for i, y in enumerate(g.outputs)
    }
    for d, nodes in enumerate(layers, 1):
        free += [feature[x] for x in read_last[d]]
        feature |= {y: free.pop() for node in nodes for y in node.outputs if y not in feature}
    mlps = [
        [
            (n.circuit, [feature[x] for x in n.inputs], [feature[y] for y in n.outputs])
            for n in nodes
        ]
        for nodes in layers
    ]
    return [feature[x] for x in g.inputs], mlps


def lookback_qk(w, d, rope_theta, margin=20.0):
    """The query and key of a head of layer 0 that attends to the token d back, or to <bos> when
    there is none. Every token 0x10000 + n has the same key, 1 in each of the 8 pairs of dimensions
    RoPE turns fastest, and <bos> 1 in the pair it turns least. The query is that key turned back
    d positions, and sqrt(8) c in the slow pair, as k_norm makes <bos>'s key sqrt(8) times as
    long there: the score of the token d + e back is s K(e) and that of <bos> about s c, where
    K(e), the mean cosine of the 8 pairs' turns by e, is 1 at 0 and at most 2 c - 1 elsewhere in
    the 512-token sliding window. The norms make s = head_dim r / sqrt(1 + c^2) for the weight r
    of q_norm, which the layer's heads share, and this r makes s (1 - c) = margin"""
    a = PREFIX + "layers.0.self_attn."
    head_dim, eye = len(w[a + "q_norm.weight"]), np.eye(len(w[PREFIX + "norm.weight"]))
    fast, slow = np.arange(8), head_dim // 2 - 1
    turns = rope_theta ** -(fast / (head_dim // 2))  # radians per position
    c = (1 + np.cos(np.outer(np.arange(1, 512), turns)).mean(1).max()) / 2
    used = [*fast, *fast + head_dim // 2, slow]
    q, k = np.zeros((2, head_dim, len(eye)))
    q[used] = np.outer(
        [*np.cos(turns * d), *-np.sin(turns * d), np.sqrt(8) * c], eye[ONE] + eye[BOS]
    )
    k[fast], k[slow] = eye[ONE], eye[BOS]
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


def set_mlp(w, layer, mlps, magnitude, k=2.0**14, sharpness=32.0):
    """The feed-forward block of a layer: mlps [(geglu MLP, features read, features written)] on
    bits, ±1 in features whose sizes magnitude gives, which it updates. A bit y goes over a
    feature that holds old as 2 y - 1 - old, with -old from a hidden unit gelu(sharpness ONE)
    (-old / sharpness). What the block adds then depends on the bits, so hidden unit 0, which
    gives the -1s, also puts k into ONE, where the post-norm drops it: the norm is k's but for a
    part in k^2 / 6400 or so"""
    a = f"{PREFIX}layers.{layer}."
    gate, up, down = (w[f"{a}mlp.{m}_proj.weight"] for m in ("gate", "up", "down"))
    written = [i for _, _, outs in mlps for i in outs]
    old = [i for i in written if magnitude[i]]
    read = [ONE, *(i for _, ins, _ in mlps for i in ins), *old]  # all that gate and up read
    w[a + "pre_feedforward_layernorm.weight"][read] = (
        np.sqrt(np.mean(magnitude**2)) / magnitude[read]
    )
    gate[0, ONE], up[0, ONE], down[ONE, 0], row = sharpness, 1 / sharpness, k, 1
    for c, ins, outs in mlps:
        n = c.n_hidden
        for proj, half in ((gate, slice(0, n)), (up, slice(n, 2 * n))):
            np.add.at(proj[row : row + n].T, ins, c.wi[half].T / 2)  # a bit is (x + ONE) / 2
            proj[row : row + n, ONE] += c.wi[half].sum(1) / 2 + c.b[half]
        down[outs, row : row + n] = 2 * c.wo
        row += n
    e = row + np.arange(len(old))  # the hidden units erasing them
    gate[e, ONE], up[e, old], down[old, e], down[written, 0] = (
        sharpness,
        -magnitude[old] / sharpness,
        1,
        -1,
    )
    w[a + "post_feedforward_layernorm.weight"][written] = k / np.sqrt(len(magnitude))
    magnitude[written] = 1


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


def save(program: Graph, out_dir: Path, cache_file: Path = CACHE_FILE, tokens: dict | None = None):
    """program, compiled, in out_dir/model.safetensors, with the names, shapes and dtype (bf16)
    of the checkpoint of REPO at REVISION, which cache_file keeps (see checkpoint_metadata), and
    the tokens of build_weights. Chunks of zeros are left as holes in the file, which read as
    zeros"""
    shapes, config = checkpoint_metadata(cache_file)
    rope_theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]
    w = build_weights(program, shapes, config["bos_token_id"], rope_theta, tokens)
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out_dir.mkdir(exist_ok=True)
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

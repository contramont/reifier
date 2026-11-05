"""Gemma 4 as a target: a program compiled into the weights of google/gemma-4-E4B.

The prompt is <bos>, other tokens and n bit tokens, "0" or "1", and each step of greedy
generation emits a bit token. The program is a graph on the n bits and on the step, the number
of tokens generated so far in binary (least significant bit first), that gives the bit to emit.

Gemma normalizes what each attention and feed-forward block adds to the residual stream, so
blocks add rails: one-hot columns for each bit or token, so that what a block adds has a fixed
norm and the rails are ratios to one. A token's rails are "0", "1" and other (any other token,
<bos> too). Attention looks back a fixed number of tokens, or finds a marked token, and copies
the rails it holds:

- windows (layers 0, 1, ...): 8 heads give each token the rails of itself and the 7 tokens
  before it, then up to 8 heads copy those windows from the tokens 0, 8, ..., 56 back, and so
  on, until a token has the rails of the last n + steps - 1 (before <bos>, it reads <bos>);
- mark (the last window layer's feed-forward block, a compiled and_): the prompt ends at the
  token with a bit n - 1 back and other n back;
- gather (the next layer): a head for each kv head finds the mark and copies some of its n
  bits, and the step is in the token's own window: n + i back is a bit when more than i tokens
  were generated.

The program's layers are the next feed-forward blocks. Each but the last writes the rails of
the bits it gives into one of two banks of columns and erases the rails of the bits it read,
so that the next one can write them again: the features of the residual stream are reused.
The last writes the logits of "0" and "1". Embeddings are one-hot and the norms of the blocks
used are one; every weight not set so is zero, so its module gives zero, and layer scalars are
one.
"""

import numpy as np

from .core import Graph, bits, trace
from .lib import and_
from .place import place
from .target import merge
from .units import Unit, cast

REPO, REVISION = "google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"
LM = "model.language_model."


def weights(program: Graph, shapes: dict, tokens, theta: float, steps: int) -> dict:
    """A weight for each name in shapes: program compiled for steps tokens of greedy generation,
    given the ids of the tokens "0", "1" and <bos>, and theta, RoPE's base in the sliding layers,
    which layers 0 to 3 are"""
    w = {k: np.full(s, k.endswith("layer_scalar"), np.float32) for k, s in shapes.items()}
    w[LM + "norm.weight"][:] = 1
    dim = len(w[LM + "norm.weight"])
    eye = np.eye(dim)
    zero, one, bos, other = eye[:4]  # a token's one-hot embedding, tied: it gives the logits too
    w[LM + "embed_tokens.weight"][:, 3] = 1
    w[LM + "embed_tokens.weight"][list(tokens)] = eye[:3]
    free = iter(eye[4:])  # the columns of the residual stream that no block has yet

    def columns(*shape):  # new columns, one per row
        return np.array([next(free) for _ in range(np.prod(shape))]).reshape(*shape, dim)

    a = LM + "layers.0.self_attn."
    hd, rows = len(w[a + "q_norm.weight"]), len(w[a + "q_proj.weight"])
    m = (steps - 1).bit_length()  # the step's bits
    n = len(program.inputs) - m  # the prompt's bits
    reach = n + steps - 1
    window, p = np.array([[zero, one, bos + other]]), 0  # window[s]: the rails of the token s back
    while len(window) < reach:
        size = len(window)
        new = columns(min(rows // hd, -(-reach // size)), size, 3)
        for h, out in enumerate(new):
            attend(w, p, h, *back(w, p, h * size, theta), window, out)
        window, p = new.reshape(-1, 3, dim), p + 1
    mark = columns(2)  # [BOS, whether the token ends the prompt]
    reads = np.vstack([window[0].sum(0), window[n - 1, :2].sum(0), window[n, 2]])
    feedforward(w, p - 1, place(trace(and_, bits(2)), "reglu"), reads, mark)
    placed = place(program, "reglu")
    given = [cast(merge(node.circuit), "geglu").wo[1:].any(1).sum() for node in placed.nodes]
    banks = columns(2, max(n, *given[:-1]), 2)  # two banks of the rails "0" and "1" of bits
    x = banks[0]  # the prompt's bits: a head copies hd // 2 of them, two rails each
    group = rows // len(w[a + "k_proj.weight"])  # query heads per kv head
    for g, s in enumerate(np.split(np.arange(n), range(hd // 2, n, hd // 2))):
        attend(w, p, g * group, *find(w, p, mark[1]), window[s, :2], x[n - 1 - s])
    # bit b of the step j is the sum over i of (bit b of i - bit b of i - 1) [j >= i], and
    # [j >= i] is whether n - 1 + i back is a bit
    i = np.arange(steps)[:, None] >> np.arange(m) & 1
    step = (i[1:] - i[:-1]).T @ window[n:reach, :2].sum(1) * np.sqrt(len(window) / n)
    reads = np.vstack([x[0].sum(0), x[:n, 1], step])  # step at the banks' scale
    logits = np.stack([zero, one - zero])  # [BOS, y] to the logits: 1 - y of "0", y of "1"
    # the gather makes the banks' rails sqrt(dim / n); the logits are 8 times a token's
    # embedding, which the tied logits read too
    feedforward(w, p, placed, reads, logits, banks, x[:n], np.sqrt(dim / n), scale=8.0)
    return w


def back(w, p, d, theta, margin=20.0):
    """The query and key of a head of layer p that attends to the token d back, or to <bos> when
    there is none. Every token but <bos> has the same key, 1 in each of the 8 pairs of dimensions
    RoPE turns fastest, and <bos> 1 in the pair it turns least. The query is that key turned back
    d positions, and sqrt(8) c in the slow pair, as k_norm makes <bos>'s key sqrt(8) times as
    long there: the score of the token d + e back is s K(e) and that of <bos> about s c, where
    K(e), the mean cosine of the 8 pairs' turns by e, is 1 at 0 and at most 2 c - 1 elsewhere in
    the 512-token sliding window. The norms make s = hd r / sqrt(1 + c^2) for the weight r of
    q_norm, which the layer's heads share, and this r makes s (1 - c) = margin"""
    a = f"{LM}layers.{p}.self_attn."
    hd, eye = len(w[a + "q_norm.weight"]), np.eye(len(w[LM + "norm.weight"]))
    fast, slow = np.arange(8), hd // 2 - 1
    turns = theta ** -(fast / (hd // 2))  # radians per position
    c = (1 + np.cos(np.outer(np.arange(1, 512), turns)).mean(1).max()) / 2
    used = [*fast, *fast + hd // 2, slow]
    q, k = np.zeros((2, hd, len(eye)))
    q[used] = np.outer([*np.cos(turns * d), *-np.sin(turns * d), np.sqrt(8) * c], eye[:4].sum(0))
    k[fast], k[slow] = eye[[0, 1, 3]].sum(0), eye[2]  # "0", "1" and other; <bos>
    w[a + "q_norm.weight"][used] = margin / (1 - c) * np.sqrt(1 + c * c) / hd
    w[a + "k_norm.weight"][used] = 1
    return q, k


def find(w, p, mark, margin=30.0):
    """The query and key of a head of layer p that attends to the one token with mark set: only
    it has a key, as the others' mark is 0, so its score is margin and theirs 0. The key is in
    the pair of dimensions RoPE turns second least, as back uses the least and the layer's heads
    share q_norm. A token with no mark before it attends to all alike"""
    a = f"{LM}layers.{p}.self_attn."
    hd, eye = len(w[a + "q_norm.weight"]), np.eye(len(w[LM + "norm.weight"]))
    slow = hd // 2 - 2
    q, k = np.zeros((2, hd, len(eye)))
    q[slow], k[slow] = eye[:4].sum(0), mark  # every token asks
    w[a + "q_norm.weight"][slow] = margin / hd
    w[a + "k_norm.weight"][slow] = 1
    return q, k


def attend(w, p, h, q, k, values, writes):
    """Head h of layer p, its query and key reading the residual along the rows of q and k, and
    its value along values, slots of rails, which go along writes. Heads that share a kv head
    must share k and values too. v_norm makes the value's slots, one-hot each, sqrt(hd / slots)
    long, so writes scale them back to 1"""
    a = f"{LM}layers.{p}.self_attn."
    hd, rows = len(w[a + "q_norm.weight"]), len(w[a + "q_proj.weight"])
    g = h * len(w[a + "k_proj.weight"]) // rows  # its kv head
    v, o = np.concatenate(values), np.concatenate(writes)
    w[a + "q_proj.weight"][h * hd : (h + 1) * hd] = q  # an error past the last head
    w[a + "k_proj.weight"][g * hd : (g + 1) * hd] = k
    w[a + "v_proj.weight"][g * hd : g * hd + len(v)] = v
    w[a + "o_proj.weight"][:, h * hd : h * hd + len(v)] = np.sqrt(len(values) / hd) * o.T
    w[f"{LM}layers.{p}.input_layernorm.weight"][:] = 1
    w[f"{LM}layers.{p}.post_attention_layernorm.weight"][:] = 1


def feedforward(w, first, placed, reads, writes, banks=(), erase=None, unit=1.0, scale=1.0):
    """placed's layers in the feed-forward blocks of layers first, first + 1, ...: the first
    reads [BOS, *inputs] along reads. Each but the last writes the bits it gives as rails into
    banks[1], banks[0], banks[1], ..., unit each, and erases the rails of the bits it read (at
    first erase, those of its first inputs): one rail of each is set, so what it adds has a fixed
    norm. The last writes [BOS, *outputs] along writes, times scale"""
    last, dim = len(placed.nodes) - 1, len(reads[0])
    erase = np.zeros((0, 2, dim)) if erase is None else erase
    taken = 1 + np.arange(len(erase))  # the inputs whose rails it erases
    for i, node in enumerate(placed.nodes):
        u, a = cast(merge(node.circuit), "geglu"), f"{LM}layers.{first + i}."
        out = np.zeros((len(u.wo), dim))  # where u's outputs go
        if i == last:
            out[[0, *(1 + y.index for y in placed.outputs)]] = writes
            taken, erase, norm = taken[:0], erase[:0], scale
        else:  # "1" gets y and "0" BOS - y; an erased "1" -x and "0" x - BOS
            given = 1 + np.flatnonzero(u.wo[1:].any(1))  # the features it gives
            rails = banks[(i + 1) % 2][: len(given)]
            out[given] = rails[:, 1] - rails[:, 0]
            out[0] = rails[:, 0].sum(0) - erase[:, 0].sum(0)
            norm = unit * np.sqrt((len(given) + len(taken)) / dim)
        c = copies(len(u.wi[0]), taken)
        h = u.hidden + c.hidden
        w[a + "mlp.gate_proj.weight"][:h] = np.vstack([u.wi[: u.hidden], c.wi[: c.hidden]]) @ reads
        w[a + "mlp.up_proj.weight"][:h] = np.vstack([u.wi[u.hidden :], c.wi[c.hidden :]]) @ reads
        erased = (erase[:, 0] - erase[:, 1]).T @ c.wo
        w[a + "mlp.down_proj.weight"][:, :h] = np.hstack([out.T @ u.wo, erased])
        w[a + "pre_feedforward_layernorm.weight"][:] = 1
        w[a + "post_feedforward_layernorm.weight"][:] = norm
        if i < last:
            reads = np.zeros((len(u.wo), dim))
            reads[0], reads[given] = rails[0].sum(0), rails[:, 1]
            taken, erase = given, rails


def copies(n, taken):
    """A unit on n inputs that gives the inputs taken, each times the first, BOS"""
    k, eye = len(taken), np.eye(n)
    return cast(Unit(eye[[0] * k + [*taken]], np.zeros(2 * k), "reglu", eye[:k, :k], ()), "geglu")


def save(program: Graph, steps: int, out):
    """program, compiled, in out/model.safetensors, with the names, shapes and dtypes of the
    checkpoint of REPO at REVISION. It needs torch, transformers and safetensors"""
    import torch
    from huggingface_hub import get_safetensors_metadata
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer

    files = get_safetensors_metadata(REPO, revision=REVISION).files_metadata
    shapes = {k: t.shape for k, t in files["model.safetensors"].tensors.items()}
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    tokens = tokenizer.convert_tokens_to_ids(["0", "1", "<bos>"])
    config = AutoConfig.from_pretrained(REPO, revision=REVISION).text_config
    theta = config.rope_parameters["sliding_attention"]["rope_theta"]
    w = weights(program, shapes, tokens, theta, steps)
    out.mkdir(exist_ok=True)
    tensors = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in w.items()}  # all are bf16
    save_file(tensors, out / "model.safetensors", {"format": "pt"})

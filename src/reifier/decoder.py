"""Transformers as a target: a program in the weights of a Gemma 4 decoder, of any size.

A token token_offset + n carries the b bits of n, and greedy decoding maps each token to the
next: the program runs at every position, on the bits of the token there, and gives the bits
of the token to emit. Its heads read earlier positions, so the next token can depend on the
last ones; <bos> is the zero token before them. A token's embedding, tied to the logits, is
1/32 in the features ONE and TOK and ±1/32 in two features for each bit, one the program reads
(an input) and one the logits read (an output); <bos>'s is the same for bits of 0, with BOS
for TOK. Gemma scales embeddings by sqrt(hidden_size), so a token's bits are then about as big
as the bits the program writes.

The residual stream holds bits as ±1, one feature each, so that it has a fixed norm whatever the
bits are and at every position, and Gemma's norms are fixed scalings: the pre-norms make each
feature they read ±1, ONE 1. schedule places the program's MLPs in the feed-forward blocks and
its heads in the attention blocks. An MLP writes a bit y over a feature that holds old as
2 y - 1 - old, and a head copies bits from back positions before, or from <bos> if there are
not as many, into features that hold 0. The program's outputs go over the output features,
which the final norm reads, so a token's logit is the number of bits it shares with the outputs
less the number it does not, and 1 from TOK, which <bos> does not have.

Every weight not set here is zero, so its module gives zero, and layer scalars are one. forward
runs the weights in numpy, as transformers runs Gemma 4.
"""

from dataclasses import dataclass, field, fields

import numpy as np

from .attention import Attention
from .compiler import Schedule, Stack, schedule
from .core import Graph, apply, flatten, lower, rewrite
from .mlp import MLP, gelu, passthrough

ONE, TOK, BOS, FIRST_BIT = 0, 1, 2, 3  # features of their own; the inputs' then the outputs'
SLIDING = "sliding_attention"


@dataclass(frozen=True)
class Architecture:
    """The shape of a Gemma 4 decoder, as its text config gives it, and the token of 0"""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    sliding_window: int
    rope_theta: float  # RoPE's base in the sliding layers
    vocab_size: int
    bos_token_id: int
    token_offset: int = 0x10000
    num_kv_shared_layers: int = 0
    global_head_dim: int = 512  # that of the full-attention layers, which hold no heads
    global_rope: dict = field(default_factory=dict)  # their RoPE parameters
    final_logit_softcapping: float | None = None
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_config(cls, c: dict, token_offset: int = 0x10000) -> Architecture:
        """The architecture of a Gemma 4 text config: each field the config sets by its name,
        and the others from where the config keeps them"""
        rope = c["rope_parameters"]
        kept = {
            "layer_types": tuple(c["layer_types"]),
            "rope_theta": rope[SLIDING]["rope_theta"],
            "global_rope": rope.get("full_attention", {}),
            "token_offset": token_offset,
        }
        given = {f.name: c[f.name] for f in fields(cls) if c.get(f.name) is not None}
        return cls(**given | kept)

    def __post_init__(self):
        own = self.layer_types[: self.num_hidden_layers - self.num_kv_shared_layers]
        if missing := set(self.layer_types) - set(own):
            raise ValueError(f"layers of type {missing.pop()} share keys and values with none")

    def head_dim_of(self, layer: int) -> int:
        return self.head_dim if self.layer_types[layer] == SLIDING else self.global_head_dim

    @property
    def attention_layers(self) -> frozenset[int]:
        """The layers that may hold heads: sliding ones with keys and values of their own (the
        last num_kv_shared_layers layers reuse those of earlier ones)"""
        own = self.num_hidden_layers - self.num_kv_shared_layers
        return frozenset(i for i in range(own) if self.layer_types[i] == SLIDING)

    def shapes(self) -> dict[str, tuple]:
        """The shapes of the weights, by name"""
        h, i, shapes = self.hidden_size, self.intermediate_size, {}
        for layer in range(self.num_hidden_layers):
            a, d = f"layers.{layer}.", self.head_dim_of(layer)
            q, kv = self.num_attention_heads * d, self.num_key_value_heads * d
            shapes |= {
                a + "self_attn.q_proj.weight": (q, h), a + "self_attn.k_proj.weight": (kv, h),
                a + "self_attn.v_proj.weight": (kv, h), a + "self_attn.o_proj.weight": (h, q),
                a + "self_attn.q_norm.weight": (d,), a + "self_attn.k_norm.weight": (d,),
                a + "mlp.gate_proj.weight": (i, h), a + "mlp.up_proj.weight": (i, h),
                a + "mlp.down_proj.weight": (h, i), a + "layer_scalar": (1,),
            }  # fmt: skip
            for norm in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
                shapes[f"{a}{norm}_layernorm.weight"] = (h,)
        return shapes | {"embed_tokens.weight": (self.vocab_size, h), "norm.weight": (h,)}


class Weights(dict):
    """Weights by name, of the shapes of an architecture, made when first used, zero until set"""

    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch, self.shapes = arch, arch.shapes()

    def __missing__(self, name):
        self[name] = np.zeros(self.shapes[name], np.float32)
        return self[name]


def round_to_bf16(x) -> np.ndarray:
    """x rounded to bfloat16, to the nearest, as float32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + (u >> 16 & 1)) & 0xFFFF0000).view(np.float32)


def embed_scale(arch: Architecture) -> float:
    """Gemma's embedding scale, sqrt(hidden_size), as bf16 makes it"""
    return float(round_to_bf16([np.sqrt(arch.hidden_size)])[0])


def rms(magnitude: np.ndarray) -> float:
    """The root mean square of the residual stream at any position"""
    return float(np.sqrt(np.mean(magnitude**2)))


# ---------------------------------------------------------------- compiling


def transformer(program: Graph, arch: Architecture) -> Weights:
    """program compiled into the weights of arch: its MLPs as GeGLUs in the feed-forward blocks,
    its heads in the attention blocks, each as early as its inputs and the room let it"""
    return build_weights(transformer_schedule(program, arch), arch)


def transformer_schedule(program: Graph, arch: Architecture) -> Schedule:
    """Where program's leaves go in the layers of arch, and the features its signals take"""
    g = rewrite(flatten(lower(program, "geglu")), lambda c, xs: split_head(arch, c, xs))
    check(g, arch)
    b = len(g.inputs)
    stack = Stack(
        n_layers=arch.num_hidden_layers,
        width=arch.hidden_size,
        inputs=list(range(FIRST_BIT, FIRST_BIT + b)),
        outputs=list(range(FIRST_BIT + b, FIRST_BIT + 2 * b)),
        reserved=frozenset({ONE, TOK, BOS}),
        room=lambda s: HeadRoom(arch, s // 2) if s % 2 == 0 else MLPRoom(arch),
        passthrough=lambda kind: passthrough("geglu", kind),
    )
    return schedule(g, stack)


def check(g: Graph, arch: Architecture):
    """Raise if arch cannot hold flat program g, for a reason other than room"""
    b, leaves = len(g.inputs), [n.circuit for n in g.nodes]
    if len(g.outputs) != b:
        raise ValueError(f"{g} gives {len(g.outputs)} bits for {b}: a token carries both")
    signals = [*g.inputs, *(y for n in g.nodes for y in n.outputs)]
    if kinds := {y.kind for y in signals} - {"bit"}:
        raise TypeError(f"{g} has {', '.join(sorted(kinds))} signals: a token holds bits")
    for c in leaves:
        if not isinstance(c, Attention) and not (isinstance(c, MLP) and c.activation == "geglu"):
            raise TypeError(f"{c} is neither a head nor an MLP of geglu")
    if FIRST_BIT + 2 * b > arch.hidden_size:
        raise ValueError(f"tokens of {b} bits need {FIRST_BIT + 2 * b} features")
    if arch.token_offset + 2**b > arch.vocab_size:
        raise ValueError(f"tokens of {b} bits need {arch.token_offset + 2**b} in the vocabulary")
    if arch.token_offset <= arch.bos_token_id < arch.token_offset + 2**b:
        raise ValueError("<bos> must not be one of the program's tokens")
    if not 0 <= arch.bos_token_id < arch.vocab_size:
        raise ValueError(f"<bos> is {arch.bos_token_id}, outside the vocabulary")
    heads = [c for c in leaves if isinstance(c, Attention)]
    if far := [c for c in heads if c.back >= arch.sliding_window]:
        raise ValueError(f"{far[0]} looks back past the sliding window of {arch.sliding_window}")
    if heads and lookback_geometry(arch) is None:
        raise ValueError(
            f"heads cannot look back reliably with head_dim {arch.head_dim}, rope_theta "
            f"{arch.rope_theta:g} and a sliding window of {arch.sliding_window}: RoPE's slowest "
            "pair turns too far in the window, or the fast ones too little to tell positions apart"
        )


def split_head(arch: Architecture, c, xs):
    """A node, but a head of more values than a kv group holds goes into heads that do not"""
    n = arch.head_dim
    if not isinstance(c, Attention) or len(xs) <= n:
        return apply(c, xs)
    parts = [Attention(c.back, c.output_kinds[i : i + n]) for i in range(0, len(xs), n)]
    return [y for i, part in enumerate(parts) for y in apply(part, xs[i * n : (i + 1) * n])]


class MLPRoom:
    """The hidden units of a feed-forward block: each MLP's, one for each of its outputs, which
    may write over a value, and one that adds k into ONE (set_mlp counts those that erase)"""

    def __init__(self, arch: Architecture):
        self.free = arch.intermediate_size - 1

    def take(self, node) -> bool:
        units = node.circuit.n_hidden + len(node.circuit.output_kinds)
        if units > self.free:
            return False
        self.free -= units
        return True


class HeadRoom:
    """The heads of an attention block, in kv groups: the query heads of a group share its keys
    and its values, so a group copies at most head_dim values, and has a query head for each
    distance it looks back. A layer that may not hold heads has no room"""

    def __init__(self, arch: Architecture, layer: int):
        self.groups = [([], set()) for _ in range(arch.num_key_value_heads)]  # (backs, values)
        self.per_group = arch.num_attention_heads // arch.num_key_value_heads
        self.open, self.arch = layer in arch.attention_layers, arch

    def place(self, back: int, values) -> tuple[int, int] | None:
        """Where a head looking back at values goes, (kv group, query head of the group), or
        None if it does not fit"""
        if not (self.open and 0 < back < self.arch.sliding_window):
            return None
        for g, (backs, group_values) in enumerate(self.groups):
            room_for_back = back in backs or len(backs) < self.per_group
            if room_for_back and len(group_values | set(values)) <= self.arch.head_dim:
                if back not in backs:
                    backs.append(back)
                group_values |= set(values)
                return g, backs.index(back)
        return None

    def take(self, node) -> bool:
        return self.place(node.circuit.back, node.inputs) is not None


def build_weights(sched: Schedule, arch: Architecture) -> Weights:
    """The weights of a schedule, the others zero"""
    w, b = Weights(arch), len(sched.inputs)
    for i in range(arch.num_hidden_layers):
        w[f"layers.{i}.layer_scalar"][:] = 1
    n, emb = np.arange(2**b), w["embed_tokens.weight"]
    tokens, bos = arch.token_offset + n[:, None], arch.bos_token_id
    emb[tokens, [ONE, TOK]], emb[bos, [ONE, BOS]] = 1 / 32, 1 / 32
    emb[tokens, sched.inputs] = emb[tokens, sched.outputs] = (
        2 * (n[:, None] >> np.arange(b) & 1) - 1
    ) / 32
    emb[bos, sched.inputs] = emb[bos, sched.outputs] = -1 / 32  # a token of zeros
    # the size of each feature's values, ±magnitude (0: it holds 0). BOS's stays 0: it is set
    # only at <bos>, where TOK is not, so TOK alone gives the norm at every position
    magnitude = np.zeros(arch.hidden_size)
    magnitude[[ONE, TOK, *sched.inputs, *sched.outputs]] = embed_scale(arch) / 32
    for i, layer in enumerate(sched.layers):
        if layer.heads:
            set_heads(w, i, layer.heads, magnitude)
        if layer.mlps or layer.erase:
            set_mlp(w, i, layer.mlps, layer.erase, magnitude)
    # each bit ±1 a logit, and TOK 1, or -1 for <bos> at <bos>, so a token beats <bos> at a tie
    scale = 32 * rms(magnitude)
    w["norm.weight"][sched.outputs] = scale / magnitude[sched.outputs]
    w["norm.weight"][[TOK, BOS]] = scale / magnitude[TOK] * np.array([1, -1])
    return w


def lookback_geometry(arch: Architecture, margin: float = 20.0):
    """How heads of arch look back, or None if they cannot: (the pairs of dimensions a token's
    key is in, the pair <bos>'s key is in, <bos>'s part b of the query, the score s of the
    token looked for).

    A token's key is 1 in each of the n pairs RoPE turns fastest (up to 8), and <bos>'s 1 in the
    pair it turns least; the query of a head looking back d is that key turned back d positions,
    and sqrt(n) b in the slow pair. k_norm makes <bos>'s key sqrt(n) times as long there, so the
    score of the token d + e back is s K(e), and that of <bos> s b cos(t w) at position t, where
    K(e) is the mean cosine of the n pairs' turns by e and w the slow pair's turn. K(0) = 1, and
    within the sliding window K is at most K_max elsewhere and cos(t w) at least kappa. The
    token looked for beats <bos> by s (1 - b), and when it is not there, <bos> beats the tokens
    by s (b kappa - K_max): b = (1 + K_max) / (1 + kappa) makes both s (1 - b) = margin. In bf16
    a score s is good to s / 256 or so, so s must be at most 64 margin"""
    half = arch.head_dim // 2
    fast, slow = np.arange(min(8, half - 1)), half - 1
    turns, window = rope_turns(arch.head_dim, arch.rope_theta), np.arange(arch.sliding_window)
    if not len(fast) or arch.sliding_window < 2:
        return None
    k_max = np.cos(np.outer(window[1:], turns[fast])).mean(1).max()
    kappa = np.cos(window * turns[slow]).min()
    b = (1 + k_max) / (1 + kappa)
    if kappa <= k_max or margin / (1 - b) > 64 * margin:  # bf16 scores have 8 bits
        return None
    return fast, slow, b, margin / (1 - b)


def lookback_key(arch: Architecture, geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows of k_proj (head_dim × hidden_size) that every head of a layer shares (see
    lookback_geometry), reading TOK and BOS, and the weights of q_norm and k_norm, which make
    s = head_dim r / sqrt(1 + b^2) for the weight r of q_norm"""
    fast, slow, b, s = geometry
    k, used = np.zeros((arch.head_dim, arch.hidden_size)), [*fast, *fast + arch.head_dim // 2, slow]
    k[fast, TOK], k[slow, BOS] = 1, 1
    q_norm, k_norm = np.zeros((2, arch.head_dim))
    q_norm[used], k_norm[used] = s * np.sqrt(1 + b * b) / arch.head_dim, 1
    return k, q_norm, k_norm


def lookback_query(arch: Architecture, geometry, back: int) -> np.ndarray:
    """The rows of q_proj of a head that looks back back positions, reading ONE: the key turned
    back that far, and sqrt(n) b in <bos>'s pair"""
    fast, slow, b, _ = geometry
    turns = rope_turns(arch.head_dim, arch.rope_theta)[fast]
    q = np.zeros((arch.head_dim, arch.hidden_size))
    used = [*fast, *fast + arch.head_dim // 2, slow]
    q[used, ONE] = [*np.cos(turns * back), *-np.sin(turns * back), np.sqrt(len(fast)) * b]
    return q


def set_heads(w: Weights, layer: int, heads, magnitude):
    """The attention block of a layer: heads [(Attention, features read, features written)] on
    bits whose sizes magnitude gives, which it updates. The input norm makes the features they
    read ±1, and each kv group's values are the features its heads read: v_norm makes each
    ±sqrt(head_dim / values), so o_proj scales them back to ±1"""
    arch, a = w.arch, f"layers.{layer}."
    room, head_dim = HeadRoom(arch, layer), arch.head_dim
    where = [room.place(c.back, ins) for c, ins, _ in heads]  # as schedule placed them
    if None in where:
        raise ValueError(f"the heads of layer {layer} do not fit")
    values = [[] for _ in range(arch.num_key_value_heads)]  # the features each group copies
    for (g, _), (_, ins, _) in zip(where, heads):
        values[g] += [f for f in dict.fromkeys(ins) if f not in values[g]]
    q_proj, k_proj, v_proj, o_proj = (w[f"{a}self_attn.{m}_proj.weight"] for m in "qkvo")
    geometry = lookback_geometry(arch)
    k, w[a + "self_attn.q_norm.weight"][:], w[a + "self_attn.k_norm.weight"][:] = lookback_key(
        arch, geometry
    )
    for g, features in enumerate(values):
        rows = slice(g * head_dim, (g + 1) * head_dim)
        k_proj[rows] = k
        v_proj[rows][range(len(features)), features] = 1
    for (g, j), (c, ins, outs) in zip(where, heads):
        h = g * room.per_group + j  # its query head
        q_proj[h * head_dim : (h + 1) * head_dim] = lookback_query(arch, geometry, c.back)
        dims = [h * head_dim + values[g].index(f) for f in ins]
        o_proj[outs, dims] = np.sqrt(len(values[g]) / head_dim)
    read, written = (
        sorted({f for fs in values for f in fs}),
        [f for *_, outs in heads for f in outs],
    )
    w[a + "input_layernorm.weight"][[ONE, TOK, BOS]] = rms(magnitude) / magnitude[ONE]
    w[a + "input_layernorm.weight"][read] = rms(magnitude) / magnitude[read]
    w[a + "post_attention_layernorm.weight"][written] = np.sqrt(len(written) / arch.hidden_size)
    magnitude[written] = 1


def set_mlp(w: Weights, layer: int, mlps, erase, magnitude, k=2.0**14, sharpness=32.0):
    """The feed-forward block of a layer: mlps [(geglu MLP, features read, features written)] on
    bits whose sizes magnitude gives, which it updates, and erase, features it sets to 0. A bit
    y goes over a feature that holds old as 2 y - 1 - old, with -old from a hidden unit
    gelu(sharpness ONE) (-old / sharpness). What the block adds then depends on the bits, so
    hidden unit 0, which gives the -1s, also puts k into ONE, where the post-norm drops it: the
    norm is k's but for a part in k^2 / (2 hidden_size) or so"""
    arch, a = w.arch, f"layers.{layer}."
    gate, up, down = (w[f"{a}mlp.{m}_proj.weight"] for m in ("gate", "up", "down"))
    written = [i for *_, outs in mlps for i in outs]
    old = [i for i in [*written, *erase] if magnitude[i]]
    if (units := 1 + sum(c.n_hidden for c, _, _ in mlps) + len(old)) > arch.intermediate_size:
        raise ValueError(
            f"layer {layer} needs {units} hidden units, but has {arch.intermediate_size}"
        )
    read = [ONE, *(i for _, ins, _ in mlps for i in ins), *old]  # all that gate and up read
    w[a + "pre_feedforward_layernorm.weight"][read] = rms(magnitude) / magnitude[read]
    gate[0, ONE], up[0, ONE], down[ONE, 0], row = sharpness, 1 / sharpness, k, 1
    for c, ins, outs in mlps:
        n = c.n_hidden
        for proj, half in ((gate, slice(0, n)), (up, slice(n, 2 * n))):
            np.add.at(proj[row : row + n].T, ins, c.wi[half].T / 2)  # a bit is (x + ONE) / 2
            proj[row : row + n, ONE] += c.wi[half].sum(1) / 2 + c.b[half]
        down[outs, row : row + n] = 2 * c.wo
        row += n
    e = row + np.arange(len(old))  # the hidden units that erase them
    gate[e, ONE], up[e, old], down[old, e] = sharpness, -magnitude[old] / sharpness, 1
    down[written, 0] = -1
    w[a + "post_feedforward_layernorm.weight"][[*written, *erase]] = k / np.sqrt(arch.hidden_size)
    magnitude[written], magnitude[erase] = 1, 0


# ---------------------------------------------------------------- running


def rmsnorm(x, weight=None, eps=1e-6):
    y = x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)
    return y if weight is None else y * weight


def rope(x, positions, turns):
    """x (positions × heads × head_dim) turned by RoPE, which pairs dimensions i and
    i + head_dim / 2 and turns pair i by turns[i] radians per position"""
    angle = np.outer(positions, turns)[:, None]
    cos, sin, (a, b) = np.cos(angle), np.sin(angle), np.split(x, 2, -1)
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)


def rope_turns(head_dim: int, theta: float, partial: float = 1.0) -> np.ndarray:
    """The turns per position of RoPE's pairs, 0 past the first partial of them"""
    n = int(partial * head_dim // 2)
    return np.r_[theta ** -(np.arange(n) * 2 / head_dim), np.zeros(head_dim // 2 - n)]


def forward(w: Weights, ids) -> np.ndarray:
    """The logits after each token of ids, as transformers' Gemma 4 gives them for weights w,
    with the per-layer inputs zero"""
    arch, ids = w.arch, np.asarray(ids)
    t, eps, own = np.arange(len(ids)), arch.rms_norm_eps, arch.num_hidden_layers
    own -= arch.num_kv_shared_layers  # the layers with keys and values of their own
    x = w["embed_tokens.weight"][ids].astype(float) * embed_scale(arch)
    shared = {}  # the keys and values that the last layers reuse, by type
    for i, kind in enumerate(arch.layer_types):
        a, d = f"layers.{i}.", arch.head_dim_of(i)
        if kind == SLIDING:
            turns = rope_turns(d, arch.rope_theta)
            seen = (t[None] <= t[:, None]) & (t[None] > t[:, None] - arch.sliding_window)
        else:
            g = arch.global_rope
            turns = rope_turns(d, g.get("rope_theta", 1e6), g.get("partial_rotary_factor", 1))
            seen = t[None] <= t[:, None]
        h = rmsnorm(x, w[a + "input_layernorm.weight"], eps)
        q, k, v = (
            (h @ w[f"{a}self_attn.{m}_proj.weight"].T).reshape(len(ids), -1, d) for m in "qkv"
        )
        q = rope(rmsnorm(q, w[a + "self_attn.q_norm.weight"], eps), t, turns)
        if i < own:
            k = rope(rmsnorm(k, w[a + "self_attn.k_norm.weight"], eps), t, turns)
            v = rmsnorm(v, None, eps)
            if i == max(j for j in range(own) if arch.layer_types[j] == kind):
                shared[kind] = k, v
        else:
            k, v = shared[kind]
        group = q.shape[1] // k.shape[1]
        k, v = np.repeat(k, group, 1), np.repeat(v, group, 1)
        scores = np.where(seen, np.einsum("thd,shd->hts", q, k), -np.inf)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        o = np.einsum("hts,shd->thd", p / p.sum(-1, keepdims=True), v).reshape(len(ids), -1)
        o = o @ w[a + "self_attn.o_proj.weight"].T
        x = x + rmsnorm(o, w[a + "post_attention_layernorm.weight"], eps)
        h = rmsnorm(x, w[a + "pre_feedforward_layernorm.weight"], eps)
        m = gelu(h @ w[a + "mlp.gate_proj.weight"].T) * (h @ w[a + "mlp.up_proj.weight"].T)
        m = m @ w[a + "mlp.down_proj.weight"].T
        x = x + rmsnorm(m, w[a + "post_feedforward_layernorm.weight"], eps)
        x = x * w[a + "layer_scalar"]
    logits = rmsnorm(x, w["norm.weight"], eps) @ w["embed_tokens.weight"].T
    cap = arch.final_logit_softcapping
    return cap * np.tanh(logits / cap) if cap else logits


def predict(w: Weights, values) -> list[int]:
    """The greedy next token after each of the tokens token_offset + values[t], after <bos>,
    as the value it carries"""
    arch = w.arch
    logits = forward(w, [arch.bos_token_id, *(arch.token_offset + v for v in values)])
    return [int(t) - arch.token_offset for t in logits[1:].argmax(-1)]

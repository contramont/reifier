"""Outside the compiler, on the same trace and grid: folding, relays, tiles, the examples, and
Gemma, in the checkpoints of gemma.py and of beside.py: small ones in numpy, and the real one."""

import functools
import hashlib
import json
import math
import re
import zlib
from itertools import pairwise
from pathlib import Path

import adder
import alongside
import keccak
import numpy as np
import pytest
from keccak import keccak_p, sponge

from reifier import beside, gemma
from reifier.check import check_grid, check_trace, check_transformer
from reifier.fold import fold
from reifier.grid import ONE, layout
from reifier.lib import FALL, add, and_, not_, or_, parity, table, xor
from reifier.relay import relay
from reifier.tiles import tiles
from reifier.trace import const, mlp, threshold_gate, trace
from reifier.transformer import (
    CONTEXT,
    DOWN,
    GATE,
    LIGHT,
    LOGIT,
    MARGIN,
    ONES,
    READ,
    SIZE,
    TIMES,
    VALUE,
    runner_up,
    token_grid,
    transformer,
    uneven,
)

KECCAK_F_400 = (  # of the state of zeros, as the Keccak team gives it: 25 lanes of 16 bits
    "09F5 40AC 0FA9 14F5 E89F ECA0 5BD1 7870 EFF0 BF8F 0337 6052 DC75 "
    "0EC9 E776 5246 59A1 5D81 6D95 6E14 633E 58EE 71FF 714C B38E"
)
# What the docs say of the examples, in one place. Of examples/keccak.py: its blocks, or
# layers of Gemma's, its features, and the most bits that a table of it reads
KECCAK = {"blocks": 7, "features": 66, "bits": 5}
# Of examples/alongside.py beside Gemma 3 270M. Of the sponge: its blocks and features, the
# bits of a token's code, the check bits among them, and the fewest that two codes differ in.
# Of the checkpoint: the features of its stream, the hidden units of a layer, its layers (the
# sponge's blocks, Gemma's 18, and one after them), and the size of its file
ALONGSIDE = {"blocks": 15, "features": 256, "code": 72, "checks": 54, "apart": 16}
ALONGSIDE |= {"stream": 896, "units": 4096, "layers": 34, "GB": 1.37}


def test_folding_is_exact_moves_constants_into_biases_and_drops_wires():
    def program(x):  # both reads a constant only, so it gives constants in turn: 1 and 0
        both = mlp(const("1"), [[1], [0]], [-1, -1], "step", np.eye(2), "both")
        return [threshold_gate([x[0], both[1]], [1, 1], 1), xor([x[0], const("1")[0], x[1]])]

    t = trace(program, 2)
    f = fold(t)
    assert check_trace(f, t) == 4
    assert len(f.mlps) == 5 < len(t.mlps) == 8  # the constants are gone, into biases
    always = fold(trace(lambda x: [x[0], *const("1")], 1))  # a constant that is an output stays
    assert [m.inputs for m in always.mlps] == [[]] and always("0") == [0, 1]

    def wires(x):  # held is x0, through two nots, and flip is not x1, once its constant is folded
        held, flip = not_(not_(x[0])), parity([x[1], const("1")[0]])
        half = table([x[2], x[0]], lambda case: 1 / 2 if case[0] else case[1], "half")
        same = table([half], lambda case: case[0], "same")  # half, where that is a bit
        return [and_([held, flip, x[2]]), or_([not_(x[2]), x[2], x[1]]), not_(x[1]), not_(same)]

    t = trace(wires, 3)
    f = fold(t)
    assert check_trace(f, t) == 8 and len(t.mlps) == 11
    # an MLP of one input that gives it, or its negation, is a wire: what read it reads the bit,
    # negated in its weights and its bias, and the wire is gone. Three stay: one that is an
    # output, one whose reader reads its bit too, and one of a bit that can be 1/2
    assert [m.name for m in f.mlps] == ["half", "same", "and", "not.2", "or", "not.3", "not.4"]
    read = {m.name: (m.inputs, m.wi.tolist(), m.b.tolist()) for m in f.mlps}
    assert read["and"] == (["x0", "x1", "x2"], [[1, -1, 1]], [-2])  # x0 and not x1 and x2
    assert read["or"][0] == ["not.2", "x2", "x1"] and read["not.4"][0] == ["same"]


def test_relays_pass_a_signal_on_through_each_layer_to_the_one_that_reads_it():
    def late(x):  # the last parity reads x1 three layers after it is given, and a is an output
        a = parity(x)
        return [a, parity([parity([a]), x[1]])]

    t = trace(late, 2)
    r = relay(t)
    assert check_trace(r, t) == 4  # the same outputs, as relays, and the same signals
    assert r.outputs == ["parity''", "parity.2"] and r.mlps[4].inputs == ["parity.1", "x1''"]
    relays = ["x1'", "x1''", "parity'", "parity''"]  # each two ReLU units, on the one before
    assert [m.name for m in r.mlps] == ["parity", "parity.1", *relays[:2], "parity.2", *relays[2:]]
    assert [m.inputs for m in r.mlps[2:4]] == [["x1"], ["x1'"]]
    assert len(relay(r).mlps) == 7 and relay(r).outputs == r.outputs  # no more, the second time
    assert relay(trace(list)).mlps == []  # and none of a program of no signals


def test_tiles_give_each_signal_a_feature_in_the_order_of_the_trace():
    t = trace(add, 8, 8)
    g = tiles(t)
    check_grid(t, g)
    signals = ["ONE", *t.inputs, *(m.name for m in t.mlps)]
    assert [set(column) - {None} for column in zip(*g)] == [{s} for s in signals]
    assert [set(row) - {None} for row in g] == [set(row) - {None} for row in layout(t)]


def test_keccak_is_keccak_whichever_way_its_rounds_are_made():  # on plain bits, with no tracing
    lanes = [int(lane, 16) for lane in KECCAK_F_400.split()]
    for kinds in ("", "2" * 20, "1" * 20, "3121" * 5):  # rounds of three layers, as by default,
        assert keccak_p([0] * 400, 20, kinds) == [x >> z & 1 for x in lanes for z in range(16)]
    msg = bytes(range(8))  # of two, of one, and of each. SHAKE128 and SHA3-256 are this sponge
    bits = [byte >> i & 1 for byte in msg for i in range(8)]  # on Keccak-p[1600, 24], after 1111
    # and after 01
    for then, rate, kinds, digest in (
        ([1, 1, 1, 1], 1344, "2" * 24, hashlib.shake_128(msg).digest(32)),
        ([0, 1], 1088, "1" + "3" * 23, hashlib.sha3_256(msg).digest()),
    ):
        want = [byte >> i & 1 for byte in digest for i in range(8)]
        assert sponge(bits + then, 24, 1600, rate, kinds)[:256] == want
    zeros = keccak_p([0] * 50, 4, "2121")  # rounds 10 to 13 of Keccak-f[50]
    assert "".join(map(str, zeros)) == "10010101010111010111101100000100001111111001110100"


def layers(t):  # the MLPs of the trace t at each depth, from 1 on
    depth, at = dict.fromkeys(t.inputs, 0), {}
    for x in t.mlps:
        d = 1 + max(depth[i] for i in x.inputs)
        depth |= dict.fromkeys(x.outputs, d)
        at.setdefault(d, []).append(x)
    return [at[d] for d in sorted(at)]


def made(xs):  # of MLPs: which there are, by name, activation, inputs and hidden units
    named = [(re.sub(r".*/|\[.*", "", x.name), x.activation) for x in xs]
    return {(*n, len(x.inputs), len(x.hidden)) for n, x in zip(named, xs)}


def test_a_round_is_three_layers_of_tables_or_two_or_one_and_each_traced_is_keccak_p():
    # Three: c, the parity of each column, a table of its 5 bits, with a unit for each of their
    # 32 cases and one more; theta, of 3 bits; and chi, of 3. Two: d, the xor of two columns,
    # 10 bits, as steps of their count, two ReLU units a step, 10 steps for each of 5 marks,
    # and one unit more; and chi on the bits before theta, each with its d: a cover of 6 bits
    three = {("c", "reglu", 5, 33), ("theta", "reglu", 3, 9), ("chi", "reglu", 3, 7)}
    two = {("d", "relu", 10, 101), ("chi", "reglu", 6, 18)}
    rng = np.random.default_rng(0)  # of the sponge on Keccak-p[50, 2] of keccak.py, on n bits
    for n, kinds, depths in (
        (24, "33", [10, 50, 50, 10, 10, 8]),
        (24, "22", [10, 50, 10, 8]),
        (24, "32", [10, 50, 50, 10, 8]),
        (24, "23", [10, 50, 10, 10, 8]),
        (12, "12", [50, 10, 8]),  # one: chi alone, on the bits of the columns, for a first
        (12, "13", [50, 10, 10, 8]),  # round that reads few: here 8 at most, a cover
    ):
        f = fold(trace(lambda msg, kinds=kinds: sponge(msg, 2, 50, 40, kinds)[:8], n))
        xs = rng.integers(0, 2, (64, n))
        values = f.values(xs)
        got = np.array([values[y] for y in f.outputs]).T
        want = [sponge(list(x), 2, 50, 40, kinds)[:8] for x in xs]
        assert np.abs(got - want).max() < 1e-12  # a cover and steps are a part in 10^15 off
        assert [len(x) for x in layers(f)] == depths
        first, last = ([x for x in f.mlps if x.name.startswith(r)] for r in ("round12", "round13"))
        assert made(last) == {"3": three, "2": two}[kinds[1]]
        if kinds[0] == "1":
            assert {x for x, *_ in made(first)} == {"chi"}
            assert max(len(x.inputs) for x in first) == 8


def next_keccak(x):  # keccak's next_token, on plain bits
    y = sponge(x)[:8]
    return y if y[7] else keccak.CODES[keccak.DASH]


def test_the_sponge_of_keccak_py_is_three_blocks_of_tables_a_round_and_one_for_the_token():
    f = keccak.program()  # next_token, traced and folded: its constants are python ints, which
    rng = np.random.default_rng(0)  # a table takes in as it is made, and it has no wire
    assert check_trace(f, next_keccak, rng.integers(0, 2, (256, 24)))
    assert len(f.mlps) == len(trace(keccak.next_token, 24).mlps) == 146
    assert f.mlps[0].name == "round12/c[0,0]" and f.mlps[-1].name == "next[7]"
    assert [len(x) for x in layers(f)] == [10, 50, 50, 10, 10, 8, 8]  # three layers a round,
    assert {len(x.hidden) for x in f.mlps if len(x.inputs) == 5} == {33}  # as keccak_p makes
    assert {len(x.inputs) for x in f.mlps} == {1, 2, 3, KECCAK["bits"]}  # them unless told, and
    # one for the token: of the last round, only what the token reads
    m, first, dash = transformer(f, keccak.CODES), keccak.FIRST, keccak.DASH
    tokens = first + rng.integers(0, 256, 16)
    check_transformer(m, tokens)
    assert m.generate([dash] * 3, 3)[3:] == [first + 146, first + 140, dash]  # on from "-" alone
    sizes = len(m.q), len(m.blocks), len(m.grid[0])  # a block a layer of tables
    assert sizes == (3, KECCAK["blocks"], KECCAK["features"])
    units, rows = zip(*((len(b.layer.hidden), len(b.layer.wo)) for b in m.blocks))
    assert units == (64, 330, 350, 360, 130, 78, 41)  # each fits TIMES times in Gemma's 2048
    assert rows == (66, 100, 116, 76, 66, 66, 66)  # theta and chi are written over the state: a
    streams, logits = m.run(tokens)  # row more for each bit, which the norm after drops. And
    layer, norms = [b.layer for b in m.blocks], [m.reads, m.writes, m.out]  # nothing is large:
    assert max(np.abs(x.wi).max() for x in layer) < 1  # a weight of a gate or a value, and out
    assert {np.abs(x.wo).max() for x in layer} == {DOWN}  # of a unit; an output of a head; a
    assert max(np.abs(x).max() for x in (m.q, m.k, m.v, m.o)) == 1  # weight of a norm, about
    assert max(x.max() for x in [*norms, *(y for b in m.blocks for y in (b.reads, b.writes))]) < 14
    assert np.isclose(max(np.abs(x).max() for x in streams), 1.5 * SIZE)  # READ SIZE over the
    assert np.allclose(logits[3:].max(1), LOGIT, atol=0.01)  # least ONE, 1; and the stream, the
    # largest ONE. The right token's logit is LOGIT


def next_alongside(x):  # alongside's next_token, on plain bits: 1/2 for each where it is silent
    y = sponge(x[:18] + x[72:90], 7, 200, 160)[:18]  # it reads the numbers of two codes
    number = sum(bit << i for i, bit in enumerate(y))  # and gives the code of the sponge's token
    return [0.5] * 72 if y[17] else alongside.code([number])[0].tolist()


def test_the_sponge_alongside_gemma_is_seven_rounds_of_two_blocks_on_two_tokens_in_256_features():
    f = alongside.program()
    assert check_trace(f, next_alongside) == 256 and alongside.KINDS == "2" * 7
    assert len(f.mlps) == 1556 < len(trace(alongside.next_token, 144).mlps) == 1560  # 4 wires
    at = layers(f)  # a round is two layers, as keccak_p makes a round of kind "2"; the first,
    two = [made(at[d]) for d in range(2, 14)]  # d of 2 bits and chi of a few, tables, as a
    assert two == [{("d", "relu", 10, 101)}, {("chi", "reglu", 6, 18)}] * 6  # column of it has
    assert {x.activation for x in [*at[0], *at[1], *at[14]]} == {"reglu"}  # one bit of a token
    assert [len(x) for x in at] == [32, 200] + [40, 200] * 5 + [34, 18, 72]  # at most, and
    assert {len(x.inputs) for x in at[0]} == {2} and max(len(x.inputs) for x in at[1]) == 4
    # constants. And one layer for the code: each bit a table with a rest, of the top bit and
    # the bits that it is the xor of, 5 at most; the top bit, of itself alone
    assert f.outputs == [f"next[{i}]:0" for i in range(72)] and len(at[-1]) == 72
    assert {len(x.outputs) for x in at[-1]} == {2} and max(len(x.inputs) for x in at[-1]) == 6
    numbers = [f"bits{i}" for i in (*range(18), *range(72, 90))]  # Of the two codes, it reads
    assert [x for x in f.inputs if x in {i for t in f.mlps for i in t.inputs}] == numbers  # the
    m = beside.front(f, alongside.CODES, hidden=ALONGSIDE["units"])  # numbers. On every token
    check_transformer(m, range(1000, 1016))  # of Gemma, in front of its layers, as alongside.py
    assert (len(m.blocks), len(m.q)) == (ALONGSIDE["blocks"], 2)  # compiles it
    features, units = len(m.grid[0]), [len(b.layer.hidden) for b in m.blocks]
    assert (features, m.size) == (ALONGSIDE["features"], ALONGSIDE["stream"])  # beside Gemma's 640
    assert max(units) == ALONGSIDE["units"] and units[0] == TIMES * 200  # A block has each unit
    steps = [[h for h in b.layer.hidden if "/d[" in h] for b in m.blocks[2:12:2]]  # as many
    assert all(len(set(x)) == len(x) == 40 * 100 for x in steps)  # times as fit, TIMES at most:
    # a block of 40 steps, each of 100 units on its bits, once. The checkpoint's layers: the
    count = len(m.blocks) + len(beside.LIFTS) + 1 + len(beside.STREAM) - 1  # blocks, Gemma's 18
    assert count == ALONGSIDE["layers"] and not beside.LIFTS  # and one after them, with no lift
    assert m.embed.shape == (2**18, 16 + 72) and m.v.shape == (72, 256)  # A feature a bit
    assert [s for s in m.grid[0] if s not in (None, ONE)] == numbers
    assert [np.count_nonzero(o) for o in m.o] == [72 + 18, 18]  # of a code, and of what is read
    carried = [s for s in m.grid[-1] if s]  # of it, which a head copies. The outputs, where the
    assert m.grid[-1][1:73] == f.outputs and len(carried) == 2 * 72 + 1  # code was, their rests
    assert set(m.sizes[-1]) == {0, beside.CARRY} and not m.out.any()  # and a ONE, each CARRY, and
    assert sum(s is None for s in m.grid[-1][73:]) >= 72  # as many features that hold nothing,
    over = [len(b.layer.wo) - features for b in m.blocks]  # for the last layer's head. Chi is
    assert over == [0, 68] + [0, 200] * 5 + [0, 0, 0]  # written over the state: rows that the
    gates, values = zip(*(np.split(b.layer.wi, 2) for b in m.blocks))  # norm after drops. No
    top = [np.abs(x).max() for x in gates]  # weight is over what the constants say. Of a gate on
    assert np.allclose(top, np.array([FALL, FALL] + [2, FALL] * 6 + [FALL]) * GATE / READ)  # a
    # bit: of a table or a cover, FALL GATE / READ; of steps, whose sums come a quarter of a
    # count from 0 at the nearest, 2 GATE / READ, which is 1.08. Of a value, LIGHT of the gates'
    heavy = np.array([np.abs(x).max() for x in values]) / LIGHT  # largest, or less where it is
    assert np.isclose(heavy.max(), max(top)) and (heavy <= np.array(top) + 1e-9).all()  # a bias,
    assert {np.abs(b.layer.wo).max() for b in m.blocks} == {DOWN}  # spread over the ONEs, as a
    # step's is; DOWN out of a unit; VALUE in
    top = lambda xs: max(np.abs(x).max() for x in xs)  # a key or a value of a head, 1 out of it;
    assert top(m.q) < top(m.k) == top(m.v) == VALUE and top(m.o) == 1  # and in a norm, under 10
    assert top([m.reads, *(b.reads for b in m.blocks)]) < 10  # to read and 2 to write
    assert top([m.writes, *(b.writes for b in m.blocks)]) < 2
    codes = np.array(list(alongside.CODES.values()), np.int8)  # of each token: its number, and
    assert codes.shape == (2**18, ALONGSIDE["code"])  # its checks, each the xor of 5 bits of it,
    assert len(alongside.CHECKS) == ALONGSIDE["checks"]  # each bit in 15
    assert {len(set(check)) for check in alongside.CHECKS} == {5}
    assert np.bincount(np.ravel(alongside.CHECKS)).tolist() == [15] * 18
    apart = lambda a, b: (codes[a] != codes[b]).sum(-1)  # The bits that two codes differ in are
    a, b = np.random.default_rng(0).integers(0, 2**18, (2, 4096))  # 16 or more: of tokens at
    assert apart(a, b)[a != b].min() >= ALONGSIDE["apart"]  # random, and of those whose numbers
    near = np.array([1 << i | 1 << j for i in range(18) for j in range(i + 1)])  # differ in a
    assert apart(a[:256, None], a[:256, None] ^ near).min() == ALONGSIDE["apart"]  # bit or two.
    # And of any two: a code is the xor of that of its number's top bit and that of the rest of
    # the number, so the xor of two codes is a code, and each has 16 ones or more, but 0's
    assert all((codes[1 << k : 2 << k] == codes[: 1 << k] ^ codes[1 << k]).all() for k in range(18))
    assert codes[1:].sum(1).min() == ALONGSIDE["apart"]


P = "text."  # before the names of the tensors: Gemma 3 has model., Gemma 4 model.language_model.
LAYER = (  # the tensors of a layer of Gemma that gemma.py sets: those of self_attn, if it attends
    "input_layernorm.weight post_attention_layernorm.weight pre_feedforward_layernorm.weight "
    "post_feedforward_layernorm.weight mlp.gate_proj.weight mlp.up_proj.weight "
    "mlp.down_proj.weight self_attn.q_norm.weight self_attn.k_norm.weight "
    "self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight "
    "self_attn.o_proj.weight"
)
G4 = {"model_type": "gemma4_text", "rope_parameters": {"sliding_attention": {"rope_theta": 1e4}}}
G4 |= {"final_logit_softcapping": 30.0}  # and Gemma 4 has a layer_scalar a layer
G3 = {"model_type": "gemma3_text", "rope_local_base_freq": 1e4, "query_pre_attn_scalar": 256}
G3 |= {"rope_theta": 1e6}  # of its layers that do not slide: not the one to read
USED = np.r_[:8, 128:136]  # of a head of 256 dimensions, those that RoPE turns as the transformer's


def small_gemma(heads=4, model=G4, size=40, layers=4, hidden=64, vocab=90000):
    """As gemma.checkpoint_metadata, for a Gemma of `size` features, `hidden` hidden units,
    `layers` layers and `vocab` tokens, with heads of 256, and two of keys and values"""
    rows = {"q_proj": 256 * heads, "k_proj": 512, "v_proj": 512}
    rows |= {"gate_proj": hidden, "up_proj": hidden}
    columns = {"o_proj": 256 * heads, "down_proj": hidden}
    shapes = {P + "embed_tokens.weight": [vocab, size]}  # of save's chunks, the first are zeros
    for k in ["norm.weight", *(f"layers.{i}.{k}" for i in range(layers) for k in LAYER.split())]:
        kind = k.split(".")[-2]  # a projection is rows by columns, of features if not of these
        shape = [rows.get(kind, size), columns.get(kind, size)] if "proj" in k else [size]
        shapes[P + k] = [256] if "_norm" in k else shape
    if model["model_type"] == "gemma4_text":
        shapes |= {f"{P}layers.{i}.layer_scalar": [1] for i in range(layers)}
    kinds = ["sliding_attention"] * layers
    return shapes, {"num_attention_heads": heads, "layer_types": kinds} | model


def tensors(w, shapes=None) -> dict:
    """Every tensor of shapes, by its name in the text model: that of w, exact, or zeros where
    w has none: what a checkpoint of w holds, before it is rounded. With no shapes, w's own"""
    out = {k.removeprefix(P): np.zeros(s) for k, s in (shapes or {}).items()}
    for k, v in w.items():
        assert shapes is None or v.shape == out[k.removeprefix(P)].shape, k
        out[k.removeprefix(P)] = np.asarray(v, float)
    return out


def rms_norm(x, weight=1.0):  # Gemma's RMSNorm: x over its root mean square, times a weight
    return weight * x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def gemma_layer(w, config: dict, i: int, x):
    """Layer i of a Gemma 3 or a Gemma 4 on the stream x[position, feature], as transformers
    runs its text model: the stream after the layer's attention, and after its block. w is the
    model's tensors by name, and config its text config. Not by any code of reifier's: the
    transformer of transformer.py is checked against it. For fewer tokens than a sliding layer's
    window: a token attends to itself and all before it"""
    plus = config["model_type"] == "gemma3_text"  # a norm of Gemma 3 multiplies by 1 + its weight
    a, kind, n = f"layers.{i}.", config["layer_types"][i], len(x)
    norm = lambda x, name: rms_norm(x, plus + w[a + name + ".weight"])
    z, dim = norm(x, "input_layernorm"), len(w[a + "self_attn.q_norm.weight"])
    q, k, v = (  # of each head: [head, position, dimension]
        (z @ w[f"{a}self_attn.{p}_proj.weight"].T).reshape(n, -1, dim).swapaxes(0, 1) for p in "qkv"
    )
    q, k = norm(q, "self_attn.q_norm"), norm(k, "self_attn.k_norm")
    v = v if plus else rms_norm(v)  # Gemma 4 has a norm on values too, with no weight
    if "rope_parameters" in config:  # Gemma 4 names the base of RoPE by the kind of layer
        theta = config["rope_parameters"][kind]["rope_theta"]
    else:
        theta = config["rope_local_base_freq" if kind == "sliding_attention" else "rope_theta"]
    angle = np.outer(np.arange(n), theta ** -(np.arange(0, dim, 2) / dim))
    cos, sin = np.tile(np.cos(angle), 2), np.tile(np.sin(angle), 2)
    half = lambda y: np.concatenate([-y[..., dim // 2 :], y[..., : dim // 2]], -1)
    q, k = q * cos + half(q) * sin, k * cos + half(k) * sin
    shared = np.arange(len(q)) // (len(q) // len(k))  # heads share keys and values
    scores = q @ k[shared].swapaxes(1, 2) * (config["query_pre_attn_scalar"] ** -0.5 if plus else 1)
    scores = np.where(np.tri(n, dtype=bool), scores, -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    y = (p / p.sum(-1, keepdims=True) @ v[shared]).swapaxes(0, 1).reshape(n, -1)
    x = x + norm(y @ w[a + "self_attn.o_proj.weight"].T, "post_attention_layernorm")
    z = norm(x, "pre_feedforward_layernorm")
    gate, up = (z @ w[f"{a}mlp.{p}_proj.weight"].T for p in ("gate", "up"))
    gelu = gate / 2 * (1 + np.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * gate**3)))
    y = x + norm((gelu * up) @ w[a + "mlp.down_proj.weight"].T, "post_feedforward_layernorm")
    return x, y if plus else y * w[a + "layer_scalar"]


def gemma_run(w, config: dict, ids):
    """A Gemma 3 or a Gemma 4 on the tokens ids, by gemma_layer: the stream after each layer,
    and the logits, capped as Gemma 4 caps them"""
    embed, plus = w["embed_tokens.weight"], config["model_type"] == "gemma3_text"
    x, after = embed[np.asarray(ids)] * math.sqrt(embed.shape[1]), []
    for i in range(len(config["layer_types"])):
        after.append(x := gemma_layer(w, config, i, x)[1])
    logits = rms_norm(x, plus + w["norm.weight"]).astype(embed.dtype) @ embed.T  # float32, of
    cap = config.get("final_logit_softcapping")  # the real one's tokens
    return after, cap * np.tanh(logits / cap) if cap else logits


def count(bits):  # the bits of the last two tokens -> the bits of the next: their sum, mod 4
    return add(bits[:2], bits[2:])


def test_save_writes_every_tensor_as_bf16_where_its_header_says(tmp_path, monkeypatch):
    monkeypatch.setattr(gemma, "checkpoint_metadata", {"a small Gemma": small_gemma()}.get)
    m = transformer(trace(count, 4), {85000 + n: [n >> 1, n & 1] for n in range(4)})  # late tokens
    out = tmp_path / "a/b"
    for _ in range(2):  # into a directory that is not there, then into one that is
        gemma.save(m, out, "a small Gemma")
    w, grid = gemma.build_weights(m, *small_gemma())
    file = (out / "model.safetensors").read_bytes()
    n = int.from_bytes(file[:8], "little")
    header, end = json.loads(file[8 : 8 + n]), 0
    assert n % 8 == 0 and header.pop("__metadata__") == {"format": "pt"}
    for k, shape in small_gemma()[0].items():  # each tensor after the one before it, rounded
        start, end = end, end + 2 * int(np.prod(shape))
        assert header.pop(k) == {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
        saved = np.frombuffer(file, "<u2", (end - start) // 2, 8 + n + start)
        assert (saved == gemma.round_to_bf16(w[k[len(P) :]]).view(np.uint32).ravel() >> 16).all(), k
    assert file[7 + n : 8 + n] == b" " and len(file) == 8 + n + end  # the header was padded
    names, config = (json.loads((out / f).read_text()) for f in ("names.json", "config.json"))
    assert names == {"features": grid, "hidden": [b.layer.hidden for b in m.blocks]}
    first = ["add/and+", "add/and-", *["erase what is left"] * 2, "ONE"]  # of the code's 2 bits
    assert names["hidden"][0] == first and config == small_gemma()[1]
    assert gemma.round_to_bf16(np.float32([1 + 2**-8, 1 + 3 * 2**-8])).tolist() == [1, 1 + 2**-6]


def test_a_small_gemma_3_or_4_of_the_weights_runs_what_the_transformer_runs(monkeypatch):
    rng, sums = np.random.default_rng(0), "123+246=369+517=886"
    ranked = lambda example: dict(enumerate(example.CODES.values(), 5))  # of tokens from 5 on
    programs = {  # a program, a code for some tokens of the small Gemma, and tokens to run it on
        "sponge": (keccak.program(), ranked(keccak), rng.integers(5, 261, 24)),
        "adder": (adder.program(), ranked(adder), sums),
        "xor": (trace(xor, 5), {4: [0], 5: [1]}, rng.integers(4, 6, 24)),
    }
    for model, heads in ((G3, 4), (G4, 8), (G3, 2), (G4, 2)):  # the sponge has 3 heads, the adder
        shapes, config = small_gemma(heads, model, 116, 8, 1024, 300)  # 8, and the xor 5
        thin = {P + f"layers.3.mlp.{x}_proj.weight": [768, 116] for x in ("gate", "up")}
        shapes |= thin | {P + "layers.3.mlp.down_proj.weight": [116, 768]}  # a layer of fewer
        other = {"vision.layers.0.mlp.gate_proj.weight": [8, 4]}  # hidden units, and a tensor
        metadata = {"a small Gemma": (shapes | other, config)}
        monkeypatch.setattr(gemma, "checkpoint_metadata", metadata.get)
        hidden = gemma.hidden_units("a small Gemma")  # that is not of the text model
        assert hidden == 768  # what every layer has: a block may have as many
        for name, (t, codes, tokens) in programs.items():
            tokens = [5 + adder.SYMBOLS.index(c) for c in tokens] if name == "adder" else tokens
            m = transformer(t, codes, hidden)  # each unit as many times as fit, as the examples
            if (name, heads) == ("adder", 2):  # compile it. A head of layer 1 copies from 2
                # tokens back what layer 0 put there, the bits that the adder reads of its last
                with pytest.raises(AssertionError, match="reads bits 2 tokens back that layer 0 "):
                    gemma.build_weights(m, shapes, config)  # 2 tokens: of others it reads more
                continue
            w = gemma.build_weights(m, shapes, config)[0]
            after, logits = gemma_run(tensors(w, shapes), config, [2, *tokens])  # after a <bos>,
            streams, want = m.run(tokens)  # a token with no code: as m's own run
            window, width, cap = len(m.q), len(m.grid[0]), config.get("final_logit_softcapping")
            first = -(-window // heads) - 1  # the last layer that attends: m's first block is
            assert first == {"sponge": 2 // heads, "adder": 4 // heads, "xor": 4 // heads}[name]
            for d, row in enumerate(streams[1:]):  # there, and row d + 1 of the grid is what
                x = after[first + d]  # block d gives: where a token has all that m reads before
                assert np.abs(x[window:, :width] - row[window:]).max() < 1e-4 * SIZE  # it, the
            done = after[first + len(m.blocks) - 1]  # same, to a ten-thousandth of a bit. The
            assert all(np.array_equal(x, done) for x in after[first + len(m.blocks) :])  # layers
            assert not any(x[0].any() or x[:, width:].any() for x in after)  # after m's blocks
            # give nothing, and the <bos> and Gemma's other features stay zeros: the rows that
            rows = max(len(b.layer.wo) for b in m.blocks)  # a block drops are of those features.
            assert rows == {"sponge": 116, "adder": 101, "xor": 27}[name] > width  # The logits
            top = want[window:].max()  # are m's, the largest LOGIT: as Gemma 4 caps them
            assert np.isclose(top, LOGIT, rtol=0.01) and cap in (None, 30)
            want = cap * np.tanh(want / cap) if cap else want
            assert np.allclose(logits[window:, : want.shape[1]], want[window:], atol=1e-3)
            assert not logits[:, want.shape[1] :].any()  # and a token with no code has none
            if name == "adder":  # so the model's next three tokens after "=" are the sum's
                at = [i for i in range(len(sums)) if "=" in sums[i - 2 : i + 1]]  # digits: those
                said = [5 + adder.SYMBOLS.index(sums[i + 1]) for i in at]  # after 7, 8, 9, 15,
                assert at[::3] == [7, 15] and logits[np.add(at, 1)].argmax(1).tolist() == said
                # 16 and 17, which are at a position more, after the <bos>
    m = transformer(*programs["sponge"][:2])  # it has 66 features, and its blocks give 116
    with pytest.raises(AssertionError, match="next_token needs 116 features, and Gemma has 115"):
        gemma.build_weights(m, shapes | {P + "norm.weight": [115]}, config)


def test_gemma_has_the_weights_of_the_transformer_and_a_norm_with_nothing_to_do_gives_zero():
    shapes, config = small_gemma()
    m = transformer(trace(count, 4), {5: [0, 1]})  # 2 tokens of 2 bits: 2 heads, and 4 blocks
    w, grid = gemma.build_weights(m, shapes, config)
    wide = (23 / 40) ** 0.5  # a norm of Gemma's, of one of m's: its mean is over more features
    assert grid == m.grid and len(grid[0]) == 23  # ONE, the code, 15 ONEs more and the signals
    embed = w["embed_tokens.weight"]  # stored less by sqrt(size), which Gemma scales it by: the
    assert embed.any(1).nonzero()[0].tolist() == [5] and not embed[:, 18:].any()  # ONEs, uneven
    ones = SIZE * uneven(ONES)
    assert np.allclose(embed[5, :18] * 40**0.5, [ones[0], -SIZE, SIZE, *ones[1:]])
    a, own = "layers.0.self_attn.", MARGIN / (1 - runner_up(m.turns, CONTEXT))  # a token's score
    q, k, v, o = (w[f"{a}{x}_proj.weight"] for x in "qkvo")
    for h in range(2):  # m's queries, where RoPE turns as m does, each as large as the key,
        mine, large = q[256 * h + USED, :23], np.abs(m.k).max()
        assert np.allclose(mine, m.q[h] / np.abs(m.q[h]).max() * large)
        assert np.allclose(k[256 * h + USED, :23], m.k)  # its key and its value, for every head,
        assert np.allclose(v[256 * h : 256 * h + 2, :23], m.v)
        assert np.allclose(o[:23, 256 * h : 256 * h + 2], m.o[h] / 2)  # and its outputs, halved
    counts = [np.count_nonzero(x.any(axis)) for x, axis in ((q, 1), (k, 1), (v, 1), (o, 0))]
    assert counts == [8 + 16, 16, 4, 4]  # no more: head 0 looks 0 back, so half its query is 0
    q_norm, k_norm = (w[f"{a}{x}_norm.weight"] for x in "qk")  # of 256: they make a head score
    assert np.allclose(q_norm[USED], own / 256) and (k_norm[USED] == 1).all()  # its token as m
    assert q_norm.nonzero()[0].tolist() == k_norm.nonzero()[0].tolist() == USED.tolist()
    reads, writes = (w[f"layers.0.{x}_layernorm.weight"] for x in ("input", "post_attention"))
    assert np.allclose(reads[:23], m.reads * wide) and np.allclose(writes[:23], m.writes * wide)
    rms = (m.embed[5] ** 2).sum() ** 0.5 / 40**0.5  # of Gemma's stream: what is embedded is
    assert np.allclose(reads[:18] * np.abs(m.embed[5]), READ * rms)  # read as READ
    for d, b in enumerate(m.blocks):  # block d is layer d's, between norms of its weights
        (n, rows), at = b.layer.wo.T.shape, f"layers.{d}."
        gate, up, down = (w[f"{at}mlp.{x}_proj.weight"] for x in ("gate", "up", "down"))
        assert np.allclose(np.vstack([gate[:n, :23], up[:n, :23]]), b.layer.wi)
        assert np.allclose(down[:rows, :n], b.layer.wo) and not down[:, n:].any()
        assert not (gate[n:].any() or up[n:].any() or gate[:, 23:].any() or up[:, 23:].any())
        pre, post = (w[f"{at}{x}_feedforward_layernorm.weight"] for x in ("pre", "post"))
        assert np.allclose(pre, [*b.reads * wide, *[0] * 17])  # a block that writes over
        assert np.allclose(post, [*b.writes * wide, *[0] * 17])  # features gives rows more,
        assert rows == (23, 26, 26, 23)[d] and down[23:rows].any(1).all()  # to Gemma's next
        assert (w[at + "layer_scalar"] == 1).all()  # features, which the norm after drops
    assert np.allclose(w["norm.weight"], [*m.out * 23**0.5, *[0] * 17])  # for m's logits
    norms = [k[len(P) :] for k in shapes if k.endswith("norm.weight")]  # every norm is set:
    idle = [k for k in norms if re.search("layers.[123].(input|post_attention|self_attn)", k)]
    assert set(norms) <= set(w) and len(idle) == 12  # one with nothing to do multiplies by 0,
    assert not any(w[k].any() for k in idle)  # as, after layer 0, those of attention do
    assert "layers.1.self_attn.q_proj.weight" not in w
    with pytest.raises(AssertionError, match="count needs 26 features, and Gemma has 25"):
        gemma.build_weights(m, shapes | {P + "norm.weight": [25]}, config)  # with those rows
    thin = shapes | {P + "layers.1.mlp.down_proj.weight": [25, 64]}
    with pytest.raises(AssertionError, match="layer 1 gives 26 rows, and Gemma has 25 features"):
        gemma.build_weights(m, thin, config)
    for slide, said in ((2, "5 layers, and Gemma has 4"), (1, "Gemma's first 2 layers to slide")):
        kinds = ["sliding_attention"] * slide + ["full_attention", "sliding_attention"]
        with pytest.raises(AssertionError, match=said):  # with a head a layer, 2 layers attend
            gemma.build_weights(m, *small_gemma(1, G4 | {"layer_types": kinds}))
    thin = shapes | {P + f"layers.{i}.mlp.gate_proj.weight": [n, 40] for i, n in enumerate((5, 12))}
    with pytest.raises(AssertionError, match="layer 1 needs 13 hidden units, and Gemma has 12"):
        gemma.build_weights(m, thin, config)  # layer 0 needs 5, and has them
    slow = {"rope_parameters": {"sliding_attention": {"rope_theta": 1e6}}}  # no 8 pairs as m's
    with pytest.raises(AssertionError, match="RoPE has no such turns"):
        gemma.build_weights(m, shapes, config | slow)


def test_gemma_3_stores_norms_less_one_and_its_later_layers_attend_for_the_heads_left_over():
    m = transformer(trace(xor, 5), {5: [1]})  # 5 tokens of a bit, 2 heads a layer: 3 layers attend
    shapes, config = small_gemma(2, G3)
    w, grid = gemma.build_weights(m, shapes, config)
    w4 = gemma.build_weights(m, *small_gemma(2))[0]  # as Gemma 4, whose norms multiply by their
    assert sorted({*w, *(f"layers.{i}.layer_scalar" for i in range(4))}) == sorted(w4)  # weights:
    for k in w:  # here, and here alone, each is stored less 1; and Gemma 3 multiplies a score by
        stored = w[k] + k.endswith("norm.weight")  # 1 / 16, so q_norm is as much more
        assert np.allclose(stored, w4[k] * (16 if "q_norm" in k else 1)), k
    assert len(w) == 44 and len(grid[0]) == 22 and len(m.blocks) == 2
    q = [w[f"layers.{i}.self_attn.q_proj.weight"] for i in range(3)]
    for i, h in enumerate((0, 2, 4)):  # layers 1 and 2 have a head each, m's heads 2 and 4: they
        assert np.allclose(q[i][USED, :22], m.q[h] / np.abs(m.q[h]).max() * np.abs(m.k).max())
        assert not q[i][256 * (1 + (i == 0)) :].any()  # look 2 and 4 back, at what layer 0 copied
        q_norm, k_norm = (w[f"layers.{i}.self_attn.{x}_norm.weight"] for x in "qk")
        assert np.allclose(q_norm[USED] + 1, 16 * MARGIN / (1 - runner_up(m.turns, CONTEXT)) / 256)
        assert (k_norm[USED] == 0).all() and np.flatnonzero(k_norm + 1).tolist() == USED.tolist()
    assert "layers.1.mlp.gate_proj.weight" not in w  # the first block is in layer 2: before it,
    norms = [k[len(P) :] for k in shapes if k.endswith("norm.weight")]  # the norms of a block
    idle = "layers.[01].*feedforward|layers.3.(input|post_attention|self_attn)"  # have nothing
    idle = [k for k in norms if re.search(idle, k)]  # to do there, as those of attention after
    assert len(idle) == 8 and all((w[k] == -1).all() for k in idle)  # it: they multiply by 0
    for i, b in zip((2, 3), m.blocks):
        gate, n = w[f"layers.{i}.mlp.gate_proj.weight"], len(b.layer.hidden)
        assert np.allclose(gate[:n, :22], b.layer.wi[:n]) and not gate[n:].any()
        post = w[f"layers.{i}.post_feedforward_layernorm.weight"]
        assert np.allclose(post, [*b.writes * (22 / 40) ** 0.5 - 1, *[-1] * 18])
    assert np.allclose(w["norm.weight"][:3], [-1, m.out[1] * 22**0.5 - 1, -1])
    thin = shapes | {P + "layers.1.self_attn.q_norm.weight": [1]}  # a head of 1 dimension
    with pytest.raises(AssertionError, match="layer 1: a head of 1 cannot copy 2 bits"):
        gemma.build_weights(m, thin, config)
    with pytest.raises(AssertionError, match="RoPE has no such turns"):  # by the theta of layers
        gemma.build_weights(m, shapes, config | {"rope_local_base_freq": 1e6})  # that slide


def beside_gemma(size=48, hidden=64, heads=4):
    """For beside.build_weights: the tensors of a small Gemma 3, at random, and its config: of
    16 tokens, and 18 layers that slide or not as those of the Gemma that beside.STREAM is of.
    A token's embedding is as large as that Gemma's, STREAM[0], whatever the token"""
    shapes, config = small_gemma(heads, G3, size, 18, hidden, 16)
    config |= {"hidden_size": size, "intermediate_size": hidden, "vocab_size": 16}
    config |= {"layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 3}
    rng, real = np.random.default_rng(0), {}
    for k, s in shapes.items():  # a norm multiplies by about 1, and a projection keeps the size
        scale = 0.1 if "norm" in k else s[-1] ** -0.5
        real[k] = rng.standard_normal(s).astype(np.float32) * scale
    embed = real[P + "embed_tokens.weight"]
    embed *= beside.STREAM[0] / np.linalg.norm(embed, axis=1, keepdims=True)
    return real, config


NUMBER = {token: [token >> i & 1 for i in range(4)] for token in range(16)}  # a code for each
# token of the small Gemma: the 4 bits of its number


def chi(case):  # Keccak's, of a case (a, b, c) of three bits: a xor (c and not b)
    return case[0] ^ (case[2] & (1 - case[1]))


def stir(bits):  # of the last two tokens, 4 bits each: two layers of chi around the 8, and then
    x = list(bits)  # the first 4, the number of the next token: or 1/2 for each, if the last is 1
    for layer in range(2):
        x = [table([x[i], x[i - 1], x[i - 2]], chi, f"chi{layer}[{i}]") for i in range(8)]
    said = lambda case: 1 / 2 if case[0] else case[1]
    return [table([x[7], x[i]], said, f"next[{i}]") for i in range(4)]


def next_stir(x):  # stir, on plain bits
    for _ in range(2):
        x = [chi((x[i], x[i - 1], x[i - 2])) for i in range(8)]
    return [0.5] * 4 if x[7] else x[:4]


def test_gemma_alone_as_wide_gives_what_gemma_gives_whatever_the_features_beside_it():
    real, config = beside_gemma()
    ids = np.random.default_rng(1).integers(0, 16, 24)
    streams, logits = gemma_run(tensors(real), config, ids)
    assert logits.std() > 0.5 and streams[-1].std() > 3 * streams[0].std()  # a Gemma that does
    for features in (0, 4, 32, 464):  # something; then with so many features more, all zeros
        wide, shapes, w, grid, names = beside.build_weights(None, real, config, features)
        assert wide == config | {"hidden_size": 48 + features, "num_hidden_layers": 18}
        assert shapes == {k: [d + features * (d == 48) for d in v.shape] for k, v in real.items()}
        assert (grid, names) == ([], [])
        c = ((48 + features) / 48) ** 0.5  # Gemma's features are c times what they were, as an
        for k, v in real.items():  # embedding is scaled by the root of the size: so each of its
            k = k[len(P) :]  # weights is over c in a norm that reads the stream, times c in what
            reads = k == "norm.weight" or "input_layernorm" in k or "pre_feedforward" in k
            want = (1 + v) / c - 1 if reads else v * c if "o_proj" in k or "down_proj" in k else v
            last = tuple(slice(-d, None) for d in v.shape)  # writes it, else as it was: on the
            assert np.allclose(w[k][last], want, atol=1e-6), k  # last of each axis, and nothing
            rest = w[k] + k.endswith("norm.weight")  # is on the others: a norm multiplies by 0
            rest[last] = 0
            assert not rest.any(), k
        after, same = gemma_run(tensors(w, shapes), wide, ids)  # it gives the same logits, to
        assert np.abs(same - logits).max() < 1e-5  # the rounding of its weights, which are float32
        for x, y in zip(after, streams):  # and its stream is Gemma's, c times, after zeros
            assert not x[:, :features].any() and np.allclose(x[:, features:], c * y, atol=1e-4)
    with pytest.raises(AssertionError, match="a Gemma has whole features a head"):
        beside.build_weights(None, real, config, 3)  # 51 features, of 4 heads
    for size, hidden in ((48, 48), (256, 64), (48, 16)):  # as many features as hidden units, or
        with pytest.raises(AssertionError, match="shapes do not tell features from the rest"):
            beside.build_weights(None, *beside_gemma(size, hidden))  # as a head has dimensions;
    # or as many hidden units as tokens: no shape then says which axis is to be wider


def test_beside_a_small_gemma_each_has_its_own_layers_and_features_and_no_weight_on_the_others(
    monkeypatch,
):
    real, config = beside_gemma()
    t, moved = trace(stir, 8), lambda k, to: re.sub(r"(?<=layers\.)\d+", lambda i: to[int(i[0])], k)
    assert check_trace(t, next_stir) == 256
    for lifts in ({}, {4: 64.0}):  # with no layer that lifts, as beside.py has it, and with one
        monkeypatch.setattr(beside, "LIFTS", lifts)
        m = beside.front(t, NUMBER, 48, 128)
        check_transformer(m, range(16))
        wide, shapes, w, grid, names = beside.build_weights(m, real, config)
        # the stream is m's 30 features, 2 that hold nothing, for whole features a head, and
        # Gemma's 48. Each of m's blocks has its units as many times as fit in 128: 126, more
        # than Gemma's 64, so every layer has as many. And the layers are m's 3 blocks, then
        sizes = wide["hidden_size"], wide["intermediate_size"], wide["num_hidden_layers"]
        assert (len(m.grid[0]), m.size) == (30, 80) and sizes == (80, 126, 22 + len(lifts))
        ours = [0, 1, 2, *(3 + i for i in lifts), 21 + len(lifts)]  # Gemma's 18, with one that
        gemmas = [i for i in range(sizes[2]) if i not in ours]  # lifts before its layer 4, if
        assert [i for i, units in enumerate(names) if not units] == gemmas  # any, and a last
        assert [wide["layer_types"][i] for i in gemmas] == config["layer_types"]  # Gemma's are
        assert {wide["layer_types"][i] for i in ours} == {"sliding_attention"}  # as they were
        assert [len(names[i]) for i in ours] == [126] * (3 + len(lifts)) + [70]  # units in twos,
        assert names[ours[-1]][:4] == ["lift f5+", "lift f5-", "lift f6+", "lift f6-"]  # 7 for
        assert grid[:4] == m.grid and grid[4:] == [m.grid[3]] * (19 + len(lifts))  # each feature
        norms = [k[len(P) :] for k in shapes if k.endswith("norm.weight")]  # every norm is set,
        assert set(norms) <= set(w) and len(norms) == 1 + sizes[2] * 6  # so one with nothing to
        assert all((w[f"layers.{i}.input_layernorm.weight"] == -1).all() for i in ours[1:-1])  # do
        assert "layers.1.self_attn.q_proj.weight" not in w  # gives 0, as around no attention
        c = (80 / 48) ** 0.5  # Gemma's own weights are those of Gemma alone, as wide: on the last
        for k, v in real.items():  # of each axis, of its own layers
            to = moved(k[len(P) :], [str(i) for i in gemmas])
            reads = to == "norm.weight" or "input_layernorm" in to or "pre_feedforward" in to
            want = (1 + v) / c - 1 if reads else v * c if "o_proj" in to or "down_proj" in to else v
            assert np.allclose(w[to][tuple(slice(-d, None) for d in v.shape)], want, atol=1e-6), k
        rows = dict(zip(ours, (len(b.layer.wo) for b in m.blocks)))  # what each block gives
        assert rows == {0: 33, 1: 30, 2: 30}  # block 0 writes over 3 features: 3 rows more
        for k, x in w.items():  # a layer of Gemma's has no weight on a feature that is not its
            layer = re.search(r"(?<=layers\.)\d+", k)  # own, the first 32, nor on a hidden unit
            if layer:  # that is not, the first 62; and a layer of m's has none on a feature that
                i, x = int(layer[0]), x + k.endswith("norm.weight")  # is not m's, after its 30:
                other = range(32) if i in gemmas else range(30, 80)  # a norm multiplies by 0
                if "down_proj" in k and i in ours:  # there. But for the rows that a block drops:
                    other = range(rows.get(i, 30), 80)  # those are the features after its own,
                for axis, n in enumerate(x.shape):  # where the norm after the block is 0
                    assert not (n == 80 and np.take(x, other, axis).any()), k
                    assert not (n == 126 and i in gemmas and np.take(x, range(62), axis).any()), k
        down = w["layers.0.mlp.down_proj.weight"]  # two that hold nothing, and Gemma's first
        post = w["layers.0.post_feedforward_layernorm.weight"] + 1  # one, whose weight in the
        assert down[30:33].any(1).all() and not down[33:].any() and not post[30:].any()  # norm
        # after is 0, as on all that is not m's
        embed, norm = w["embed_tokens.weight"], w["norm.weight"]  # in these, they meet: m's ONE
        assert np.allclose(embed[:, :20] * 80**0.5, m.embed) and not embed[:, 20:32].any()  # and
        assert np.array_equal(embed[:, 32:], real[P + "embed_tokens.weight"])  # code are embedded
        assert np.flatnonzero(norm[:32] + 1).tolist() == [1, 2, 3, 4]  # before Gemma's own, and
        assert (norm[1:5] + 1 == beside.HEARD).all()  # the last norm hears the code alone
        size, carried = beside.CARRY, [1, 2, 3, 4, 25, 26, 27, 28, 29]  # what m's blocks leave:
        assert [f for f, s in enumerate(m.grid[-1]) if s] == carried and m.grid[-1][29] == ONE
        for i, larger in lifts.items():  # the outputs, their rests and a ONE. A layer that lifts
            at = f"layers.{3 + i}."
            pre, post = (w[f"{at}{x}_feedforward_layernorm.weight"] + 1 for x in ("pre", "post"))
            assert np.flatnonzero(pre).tolist() == np.flatnonzero(post).tolist() == carried
            assert np.allclose(pre[carried], beside.LIFT * beside.STREAM[i] / size)  # reads each
            assert np.allclose(post[carried], (larger - size) * (5 / 80) ** 0.5)  # as LIFT where
            size = larger  # Gemma's stream is as large as usual, and makes it larger, in place
        reads = w[f"layers.{ours[-1]}.input_layernorm.weight"] + 1  # the last layer's head reads
        assert np.flatnonzero(reads).tolist() == carried  # each as FAINT, there
        assert np.allclose(reads[carried], beside.FAINT * beside.STREAM[18] / size)


def said_by_stir(ids):
    """What stir has to say after each of ids but the first, and the one before it: whether it
    has a token, its number, and what m's features are to hold then, each over its size: the
    bits of the token's code, or zeros and then their rests; and a ONE"""
    code = np.array([NUMBER[i] for i in ids])
    said = np.array([next_stir([*older, *newer]) for older, newer in pairwise(code)])
    speaks, want = said[:, 0] != 0.5, np.zeros((len(ids) - 1, 32))
    want[:, 1:5], want[:, 25:29], want[:, 29] = np.where(speaks[:, None], 2 * said - 1, 0), 0, 1
    want[:, 25:29] = ~speaks[:, None]
    return speaks, said.round().astype(int) @ (1 << np.arange(4)), want


def test_a_small_gemma_runs_the_program_beside_it_carries_what_it_gives_and_says_its_token(
    monkeypatch,
):
    real, config = beside_gemma()
    t, ids = trace(stir, 8), np.random.default_rng(1).integers(0, 16, 64)
    speaks, token, want = said_by_stir(ids)
    assert 16 < speaks.sum() < 48  # stir has a token to say after half of the pairs
    for lifts in ({}, {4: 64.0}):
        monkeypatch.setattr(beside, "LIFTS", lifts)
        m = beside.front(t, NUMBER, 48, 128)
        wide, shapes, w, _, names = beside.build_weights(m, real, config)
        w = tensors(w, shapes)
        after, logits = gemma_run(w, wide, ids)
        theirs = w["embed_tokens.weight"][ids, 32:] * 80**0.5  # Gemma's features: an embedding
        streams = m.run(ids)[0]  # m's own run has a <bos> first: all is a position later there
        for d, row in enumerate(streams[1:]):  # block d is layer d's: m's features hold what its
            assert np.abs(after[d][1:, :30] - row[2:]).max() < 1e-4 * SIZE  # own run says, and
            assert np.array_equal(after[d][:, 32:], theirs) and not after[d][:, 30:32].any()
        held = beside.CARRY  # Gemma's what they did, though block 0 has rows of weights there
        assert np.allclose(after[2][1:, :32], held * want, atol=0.01 * held)
        for i in range(3, len(after) - 1):
            if names[i]:  # a layer that lifts makes each larger, in place, and that is all
                held = lifts[i - 3]
                assert np.allclose(after[i][1:, :32], held * want, atol=0.01 * held)
                assert np.array_equal(after[i][:, 32:], after[i - 1][:, 32:])
            else:  # no layer of Gemma's writes to m's features: they are as they were, to the
                assert np.array_equal(after[i][:, :32], after[i - 1][:, :32])  # bit, and its
                assert (after[i][:, 32:] != after[i - 1][:, 32:]).all()  # own are not
        loud, head = beside.LOUD * (80 / 896) ** 0.5, beside.HEAD  # then the last layer: its
        spoken = np.r_[loud * want[:, :5].T, head * want[:, 1:5].T, head * want[:, 9:].T].T
        assert np.allclose(after[-1][1:, 5:32], spoken[:, 5:], atol=0.01 * head)  # head makes
        assert np.allclose(after[-1][1:, :5], spoken[:, :5], atol=0.002 * loud)  # each HEAD, the
        assert np.array_equal(after[-1][:, 32:], after[-2][:, 32:])  # outputs in features that
        # held nothing, and its block makes each output LOUD where the logits read it: the
        # model's token is stir's wherever it has one. A token whose code differs in a bit, of
        right, near = (logits[1:][speaks, token[speaks] ^ bit] for bit in (0, 1))  # the 4, has
        assert (logits.argmax(1)[1:] == token)[speaks].all()  # half the logit
        assert right.min() > 15 and np.allclose(near, right / 2, atol=0.05 * right.min())


def test_the_last_layer_makes_a_code_loud_in_a_stream_of_up_to_5_times_its_size_else_nothing():
    t, rng = trace(stir, 8), np.random.default_rng(0)
    ids = rng.integers(0, 16, 64)
    speaks, token, want = said_by_stir(ids)
    carried, spare = [1, 2, 3, 4, 25, 26, 27, 28, 29], [5, 6, 7, 8]  # outputs, rests and a ONE
    made, head, carry, usual = [*spare, *carried[4:]], beside.HEAD, beside.CARRY, beside.STREAM[18]
    for size, scales in ((48, (0.2, 1, 5)), (304, (1,))):  # a Gemma of so many features: the
        m = beside.front(t, NUMBER, size, 64)  # stream is 80, and 336
        wide, shapes, w = beside.build_weights(m, *beside_gemma(size))[:3]
        w, n, last = tensors(w, shapes), m.size, f"layers.{wide['num_hidden_layers'] - 1}."
        # the head of the layer after Gemma's last reads what m's blocks left, each as FAINT
        # where Gemma's stream is as large as usual, and gives each output to a feature that
        # held nothing, and each rest and the ONE to itself: the norm after makes each HEAD
        reads, writes = (w[last + x + "_layernorm.weight"] + 1 for x in ("input", "post_attention"))
        assert np.flatnonzero(reads).tolist() == carried and np.flatnonzero(writes).tolist() == made
        assert np.allclose(reads[carried], beside.FAINT * usual / carry)
        assert np.allclose(writes[spare], head * (5 / n) ** 0.5)  # 5: the squares of what is
        assert np.allclose(writes[carried[4:]], (head - carry) * (5 / n) ** 0.5)  # carried, each
        v, o = (w[f"{last}self_attn.{x}_proj.weight"] for x in "vo")  # over its size
        assert v[:9].nonzero()[1].tolist() == carried
        assert [x.tolist() for x in o[:, :9].nonzero()] == [made, list(range(9))]
        gate = w[last + "mlp.gate_proj.weight"]  # the block's units read those: the two of an
        units = np.flatnonzero(gate[:, 5])  # output have QUIET times the mean of all the rests
        assert set(gate[units, 5]) == {-1, 1} and (gate[units, 25:29] == -beside.QUIET / 4).all()
        assert not gate[np.flatnonzero(gate[:, 29]), 25:29].any()  # taken off; those of ONE, not
        left = m.run(ids)[0][-1][2:]  # what m's own layers leave, after each pair of tokens
        assert np.allclose(left, carry * want[:, :30], atol=0.01 * carry)
        loud = beside.LOUD * (n / 896) ** 0.5  # as loud as the stream is wide
        for scale in scales:  # Gemma's stream as large as usual after its last layer, and 5
            x = np.zeros((len(left), n))  # times less and more: at random, with what m left
            x[:, :30], x[:, -size:] = left, rng.standard_normal((len(left), size))
            total = n * (scale * usual) ** 2 - (left * left).sum(1)  # the squares of Gemma's
            x[:, -size:] *= np.sqrt(total / (x[:, -size:] ** 2).sum(1))[:, None]
            attended, spoken = gemma_layer(w, wide, int(last.split(".")[1]), x)
            others = lambda y: np.delete(y, made, 1)  # the head gives to no other feature,
            assert np.array_equal(others(attended), others(x))  # and what it makes is HEAD,
            off = attended[:, 29] / head  # to 2%, under a stream as large as usual or 5 times
            assert np.abs(off - 1).max() < (0.02 if scale <= 1 else 0.1)  # less. Under one 5
            wanted = head * off[:, None] * want[:, carried]  # times as large, what the head gives
            assert np.allclose(attended[:, made], wanted, atol=0.005 * head)  # is so small that
            # the norm's eps is felt: all is 8% smaller here, and a fifth in alongside's, and the
            # block, whose units come in twos, gives the same. In a stream 10 times as large it
            # would not: alongside's outputs are then a third off
            kept = lambda y: np.delete(y, carried[:4], 1)  # it writes the outputs alone,
            assert np.array_equal(kept(spoken), kept(attended))
            assert np.allclose(spoken[:, 1:5], loud * want[:, 1:5], atol=0.002 * loud)
            assert np.abs(spoken[~speaks, 1:5]).max() < 1e-4 * loud  # nothing, where no code
            # the last norm hears the outputs alone, of m's features, and drops the features
            # that the head gave them to: a token's logit gains HEARD times a bit, for each bit
            # of its code that they have. Were the stream wider, a bit would be as much louder,
            # and the code's embedding as much smaller
            norm = 1 + w["norm.weight"]
            assert np.flatnonzero(norm[:32]).tolist() == carried[:4] and not norm[spare].any()
            heard = rms_norm(spoken, norm)[:, :32] @ w["embed_tokens.weight"][:, :32].T
            rms = np.sqrt((spoken * spoken).mean(1))[speaks]
            right = beside.HEARD * beside.LOUD * SIZE * 4 / (896**0.5 * rms)
            assert np.allclose(heard[speaks, token[speaks]], right, rtol=0.005)
            assert np.abs(heard[~speaks]).max() < 1e-4 * right.min()  # and there, the logits
            if scale <= 1:  # are Gemma's. Where the rests are there, the block is deaf to an
                x[:, 1:5] = carry * rng.choice([-1, 1], (len(x), 4))  # output, were it a bit:
                x[:, 25:29] = carry  # if gelu is relu at its gates, as it is under a stream no
                quiet = gemma_layer(w, wide, int(last.split(".")[1]), x)[1]  # larger than usual
                assert np.abs(quiet[:, 1:5] - x[:, 1:5]).max() < 1e-4 * loud


REPEAT = {token: NUMBER[token] * 6 for token in range(16)}  # codes of 24 bits: the number, 6 times


def fan(bits):  # of the last two tokens, of codes of 24 bits: the newer's code, a bit at a time,
    said = lambda case: 1 / 2 if case[0] else case[1]  # or 1/2 for each after an odd older one
    return [table([bits[0], bits[24 + i % 4]], said, f"next[{i}]") for i in range(24)]


def unlike(bits):  # of the last two tokens, 2 bits each: two outputs that are 1/2, not together
    half = table([bits[0], bits[2]], lambda case: 1 / 2 if case[0] else case[1], "half")
    rarely = lambda case: 1 / 2 if case[0] and case[1] else case[2]
    return [half, table([bits[0], bits[1], bits[3]], rarely, "quarter")]


def test_beside_gemma_says_what_it_cannot_place_and_front_pads_for_what_its_layers_need():
    real, config = beside_gemma()
    t, bit = trace(stir, 8), {token: [token & 1] for token in range(16)}
    others = beside.front(t, NUMBER, 40), beside.front(t, NUMBER, 48, heads=3)
    for m in (transformer(t, NUMBER), *others):  # with logits of its own; for a Gemma of fewer
        # features; and for one of 3 heads, whose 78 features are not whole for 4
        with pytest.raises(AssertionError, match="stir is not of front, for this Gemma"):
            beside.build_weights(m, real, config)
    m = beside.front(t, NUMBER, 48)  # one of front is not for gemma.py, which needs its logits,
    with pytest.raises(AssertionError, match="stir has no logits of its own: it is for beside"):
        gemma.build_weights(m, *small_gemma())
    few = config | {"layer_types": config["layer_types"][:4]}  # nor for a Gemma of 4 layers
    with pytest.raises(AssertionError, match="STREAM is of unsloth/gemma-3-270m"):
        beside.build_weights(m, real, few)
    with pytest.raises(AssertionError, match="stir needs a code for each of Gemma's tokens"):
        beside.build_weights(beside.front(t, dict(list(NUMBER.items())[:8]), 48), real, config)
    with pytest.raises(AssertionError, match="xor needs its heads in one layer"):  # 5, of 4
        beside.build_weights(beside.front(trace(xor, 5), bit, 48), real, config)
    pairs = {token: NUMBER[token][:2] for token in range(16)}
    with pytest.raises(AssertionError, match="unlike: its outputs that can be 1/2 must be 1/2 to"):
        beside.build_weights(beside.front(trace(unlike, 4), pairs, 48), real, config)
    early = lambda x: [parity(x[:2]), parity([parity(x[2:])])]  # block 0 would make the first,
    read = lambda x: [x[0], parity(x)]  # the first is read, not made, and 3 bits are read for
    three = lambda x: [parity(x), parity(x[:2])]  # codes of 2: front says so too
    for program, n, said in (
        (early, 4, ": its last block must make every output"),
        (read, 2, ": each output must be a signal of its own that an MLP makes"),
        (three, 3, " reads 3 bits: no codes of 2"),
    ):
        with pytest.raises(AssertionError, match="lambda" + said):
            beside.front(trace(program, n), pairs, 48)
    # front pads the stream. For whole features a head: stir's 30 and Gemma's 48 are 26 a head
    assert [beside.front(t, NUMBER, 48, heads=heads).size for heads in (4, 3)] == [80, 78]  # of
    # 3, and not whole for 4. And for the rows that a block drops, which are the features after
    tiny, (few, config) = beside.front(t, NUMBER, 2, heads=2), beside_gemma(2, heads=2)  # its
    assert tiny.size == 34 and max(len(b.layer.wo) for b in tiny.blocks) == 33  # own: beside a
    wide, shapes, w = beside.build_weights(tiny, few, config)[:3]  # Gemma of 2 features,
    w, ids = tensors(w, shapes), np.arange(16)  # 4 that are not stir's, as block 0 writes over 3.
    after = gemma_run(w, wide, ids)[0]  # That Gemma's features are then as they were after it,
    assert np.array_equal(after[2][:, 32:], w["embed_tokens.weight"][ids, 32:] * 34**0.5)  # the
    assert np.abs(after[2][1:, :30] - tiny.run(ids)[0][3][2:]).max() < 1e-4 * SIZE  # two that
    assert not after[2][:, 30:32].any()  # hold nothing zeros, and stir's what stir's own run says
    # And front pads the grid, for the head of the last layer: fan's last row has its 24
    f = trace(fan, 48)  # outputs, their rests and a ONE, and 20 features that hold nothing,
    plain, m = token_grid(f, True), beside.front(f, REPEAT, 48)  # where the head needs 24
    assert sum(s is None for s in plain[-1][25:]) == 20 and len(m.grid[0]) == len(plain[0]) + 4
    assert [row[:-4] for row in m.grid] == plain and all(row[-4:] == [None] * 4 for row in m.grid)
    to = beside.plan(m, beside_gemma()[1])[1][1]  # the features that the head gives to: for an
    assert len(set(to)) == 49 and {m.grid[-1][f] for f in to[:24]} == {None}  # output, one that
    bare = transformer(f, REPEAT, None, (50, beside.STREAM[0] ** 2), beside.CARRY)  # holds
    with pytest.raises(AssertionError, match="fan needs 24 features more, for its head to give to"):
        beside.plan(bare, beside_gemma()[1])  # nothing. Not of front, it has too few


G270M, RAN = "unsloth/gemma-3-270m", 177880831  # the checksum of the weights that ran on it
WEIGHTS_AND_TOKENIZER = "model.safetensors", "tokenizer.json", "tokenizer_config.json"
README = Path(__file__).parent.parent / "README.md"  # text, to run Gemma on
STEP = 2**-6 / SIZE  # between the numbers of bf16 just over a bit's size, as a part of a bit


def gemma_is_here(*files) -> bool:
    """Whether the shapes of Gemma's tensors are in reifier's cache, and files in Hugging Face's"""
    from huggingface_hub import try_to_load_from_cache

    rev = gemma.REVISIONS[G270M]
    found = [isinstance(try_to_load_from_cache(G270M, f, revision=rev), str) for f in files]
    return all(found) and gemma.cache_file(G270M).exists()


@functools.cache
def readme_tokens() -> list[int]:
    """The tokens of the README, after a <bos>, by Gemma's tokenizer"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(G270M, revision=gemma.REVISIONS[G270M])
    return tokenizer(README.read_text())["input_ids"]


def test_gemma_weights_are_those_that_ran_on_the_model():
    if not gemma.cache_file(G270M).exists():
        pytest.skip("needs the shapes of Gemma's tensors")
    m = transformer(keccak.program(), keccak.CODES, gemma.hidden_units(G270M))  # as keccak.py
    w, checksum = gemma.build_weights(m, *gemma.checkpoint_metadata(G270M))[0], 0
    for k in sorted(w):  # each tensor set: its name, its rows and columns not zero, their values
        v = np.atleast_2d(w[k])
        at = [np.flatnonzero(v.any(axis)) for axis in (1, 0)]
        named = zlib.crc32((w.prefix + k).encode() + at[0].tobytes() + at[1].tobytes(), checksum)
        checksum = zlib.crc32(v[np.ix_(*at)].tobytes(), named)
    assert checksum == RAN, "the weights changed: run them on the model again"


def test_gemma_runs_in_bf16_what_the_transformer_runs(tmp_path):
    # uv run --python 3.14 --with torch --with transformers --with huggingface_hub python -m
    # pytest -k gemma_runs, once transformers has fetched Gemma's tokenizer, which has the
    # tokens of the adder
    torch = pytest.importorskip("torch")
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM

    if not gemma_is_here("tokenizer.json"):
        pytest.skip("needs the shapes of Gemma's tensors, and its tokenizer")
    file = try_to_load_from_cache(G270M, "tokenizer.json", revision=gemma.REVISIONS[G270M])
    vocab = json.loads(Path(file).read_text())["model"]["vocab"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def runs(m, tokens, out):
        """Gemma, loaded from what save writes of m in out, after it is checked on a <bos> and
        tokens: its stream after each layer is m's, and so is each next token"""
        gemma.save(m, out, G270M)
        model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).to(device)
        after = []  # the stream after each layer: output_hidden_states has the last after its norm
        hooks = [x.register_forward_hook(lambda *x: after.append(x[2])) for x in model.model.layers]
        ids = torch.tensor([[model.config.bos_token_id, *tokens]], device=device)
        with torch.no_grad():  # as transformers runs it by default, after a <bos>
            tops = model(ids).logits[0].argmax(1).cpu().numpy()
        for hook in hooks:
            hook.remove()
        after = [(x[0] if isinstance(x, tuple) else x)[0].float().cpu().numpy() for x in after]
        assert len(after) == 18
        streams, logits = m.run(tokens)
        n, width = len(m.q), len(m.grid[0])  # position n is the first that has all that m reads
        first = -(-n // model.config.num_attention_heads) - 1  # the last layer that attends
        # Row d of the grid is what block d - 1 gives. bf16 keeps 8 bits: of its numbers just
        # over 2, a bit's size, one is a STEP from the next. The weights were rounded to it, and
        # then every sum is: a bit comes out 2 steps off in the sponge, and 2.3 in the adder, at
        # most; with the same weights and sums in float32, 0.9 and 1.7. A table is flat where
        # it is on, so what reads such a bit reads the bit
        for d in range(1, len(streams)):
            off = np.abs(after[first + d - 1][n:, :width] - streams[d][n:]).max() / SIZE
            assert off <= 3 * STEP, f"row {d} is {off:g} of a bit off"
        last = after[first + len(m.blocks) - 1]  # the layers after m's blocks give nothing, and
        assert all((x == last).all() for x in after[first + len(m.blocks) :])  # the <bos> and
        assert not any(x[0].any() or x[:, width:].any() for x in after)  # Gemma's other features
        assert (tops[n:] == logits[n:].argmax(1)).all()  # stay zeros; and each next token is m's
        return model

    def greedy(model, tokens, n):
        """The n next tokens of model after a <bos> and tokens, each the likeliest"""
        ids = torch.tensor([[model.config.bos_token_id, *tokens]], device=device)
        for _ in range(n):
            with torch.no_grad():
                ids = torch.cat([ids, model(ids).logits[:, -1:].argmax(2)], 1)
        return ids[0, -n:].tolist()

    first, dash, hidden = keccak.FIRST, keccak.DASH, gemma.hidden_units(G270M)
    m = transformer(keccak.program(), keccak.CODES, hidden)  # the sponge, as keccak.py compiles
    assert hidden == 2048 and max(len(b.layer.hidden) for b in m.blocks) == TIMES * 360  # it: 3
    model = runs(m, first + np.random.default_rng(0).integers(0, 256, 200), tmp_path / "keccak")
    assert greedy(model, [dash] * 3, 3) == [first + 146, first + 140, dash]  # heads, in layer 0
    m = transformer(adder.program(), {vocab[c]: code for c, code in adder.CODES.items()}, hidden)
    assert len(m.q) == 8  # the adder: Gemma has 4 heads a layer, so its layer 1 copies 4 codes
    model = runs(m, [vocab[c] for c in "123+246=36"], tmp_path / "adder")  # a digit is next
    assert greedy(model, [vocab[c] for c in "123+246="], 3) == [vocab[c] for c in "369"]


def test_the_gemma_of_these_tests_gives_what_transformers_gives_on_gemmas_own_weights():
    # gemma_run, by which the small Gemmas of these tests run, on the real one
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    if not gemma_is_here(*WEIGHTS_AND_TOKENIZER):
        pytest.skip("needs Gemma's weights and tokenizer")
    ids, device = readme_tokens()[:48], "cuda" if torch.cuda.is_available() else "cpu"
    load, rev = AutoModelForCausalLM.from_pretrained, gemma.REVISIONS[G270M]  # in float32: in
    model, after = load(G270M, revision=rev, dtype=torch.float32).to(device), []  # bf16, Gemma
    # scales an embedding by the root of the size as bf16 has it, a part in 500 less
    hooks = [x.register_forward_hook(lambda *x: after.append(x[2])) for x in model.model.layers]
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0].cpu().numpy()
    after = [(x[0] if isinstance(x, tuple) else x)[0].cpu().numpy() for x in after]
    w = {k.removeprefix("model."): v for k, v in beside.checkpoint_weights(G270M).items()}
    mine, same = gemma_run(w, gemma.checkpoint_metadata(G270M)[1], ids)
    assert len(mine) == len(after) == len(hooks) == 18
    for x, y in zip(mine, after):  # a stream of thousands, to a part in ten thousand, and logits
        assert np.abs(x - y).max() < 1e-4 * np.abs(y).max()  # of dozens to a hundredth
    assert np.abs(same - logits).max() < 0.01 and logits.max() > 10
    assert (same.argmax(1) == logits.argmax(1)).all()


def test_gemma_with_the_sponge_beside_it_gives_what_each_gives_alone(tmp_path):
    # uv run --python 3.14 --with torch --with transformers --with huggingface_hub python -m
    # pytest -k beside_it_gives, once examples/alongside.py has fetched Gemma's own weights, and
    # transformers its tokenizer: no other test runs beside.py's constants, STREAM, LOUD, HEARD
    # and the like, on the model
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    if not gemma_is_here(*WEIGHTS_AND_TOKENIZER):
        pytest.skip("needs Gemma's weights and tokenizer")
    f = alongside.program()
    m = beside.front(f, alongside.CODES, hidden=ALONGSIDE["units"])
    beside.save(m, tmp_path)  # as alongside.py compiles and saves it
    beside.save(None, tmp_path / "alone", ALONGSIDE["features"])  # Gemma alone, as wide
    device, load = "cuda" if torch.cuda.is_available() else "cpu", AutoModelForCausalLM
    models = [load.from_pretrained(tmp_path), load.from_pretrained(tmp_path / "alone")]
    models.append(load.from_pretrained(G270M, revision=gemma.REVISIONS[G270M]))  # Gemma itself
    assert {x.dtype for x in models} == {torch.bfloat16}  # all three bf16
    layers, stream, units = ALONGSIDE["layers"], ALONGSIDE["stream"], ALONGSIDE["units"]
    configs = [x.config for x in models]
    sizes = [(c.num_hidden_layers, c.hidden_size, c.intermediate_size) for c in configs]
    assert sizes == [(layers, stream, units), (18, stream, 2048), (18, 640, 2048)]
    size = (tmp_path / "model.safetensors").stat().st_size  # in bytes, holes and all
    assert round(size / 1e9, 2) == ALONGSIDE["GB"]
    names = json.loads((tmp_path / "names.json").read_text())  # the sponge's 15 blocks, then
    gemmas = [i for i, units in enumerate(names["hidden"]) if not units]  # Gemma's 18 layers, and
    assert gemmas == list(range(15, 33)) and len(names["features"]) == layers + 1  # one after
    assert len(names["hidden"][33]) == 73 * 2 * 14  # them: its units in twos, 14 for each output
    text = readme_tokens()[: 1024 if device == "cuda" else 256]  # and ONE. A <bos>, and the
    ids, after = torch.tensor([text], device=device), []  # start of the README
    hooks = [x.register_forward_hook(lambda *x: after.append(x[2])) for x in models[0].model.layers]
    with torch.no_grad():
        both, alone, itself = (x.to(device)(ids).logits[0].float() for x in models)
    after = [(x[0] if isinstance(x, tuple) else x)[0].float() for x in after]
    assert len(after) == len(hooks) == layers
    bits = np.array([alongside.CODES[t] for t in text])  # position 0 reads its token twice. What
    values = f.values([[*older, *newer] for older, newer in pairwise([bits[0], *bits])])  # the
    y = np.array([values[x] for x in f.outputs]).T  # sponge gives: its trace is Keccak's (above)
    speaks = torch.tensor(y[:, 0] != 0.5, device=device)
    token = torch.tensor(y[:, :18].round().astype(int) @ (1 << np.arange(18)), device=device)
    assert len(text) / 4 < speaks.sum() < 3 * len(text) / 4  # half of the time
    top, second = both.topk(2).values.T
    assert (both.argmax(1) == token)[speaks].all()  # there, the token of the sponge's code,
    assert (top - second)[speaks].min() >= 10  # which no other is near: by 17 or more on three
    assert alone.max() < top[speaks].min() and top[speaks].max() < 100  # texts, and its logit
    # is 54 to 84, and 69 as a rule, where Gemma's largest on them is 35: the sponge's is larger
    same = both.argmax(1) == alone.argmax(1)  # elsewhere, the token of Gemma alone, as wide,
    floor = (alone.argmax(1) == itself.argmax(1)).float().mean()  # as often as that is Gemma's
    assert same[~speaks].float().mean() >= floor - 0.02 and floor >= 0.95  # own: weights that are
    assert (both - alone).abs().amax(1)[~speaks].median() < 1  # rounded again break ties, and
    # move no logit far. In the model itself: no layer of Gemma's writes to the sponge's
    carried = [x[:, :256] for x in after[14:33]]  # features, which are after its last what
    assert all((x == carried[0]).all() for x in carried)  # the sponge's own layers left, to
    spoken = after[33][:, 1:73] / beside.LOUD  # the bit; then the last layer makes each bit
    sign = torch.tensor(np.where(y == 0.5, 0, 2 * y - 1), device=device)  # of a code LOUD, to
    assert (spoken - sign).abs().max() < 0.05  # 5%, and gives nothing where there is no code
    assert spoken[~speaks].abs().max() < 1e-4
    # The sponge's own layers hold what its own run says, in numpy, on the first tokens: its run
    # has a <bos> first, so all is a position later there. A bit is 2 steps of bf16 off at most
    # where a table makes it, and up to 6 where steps or a cover do, whose units give much and
    # take most of it back (on three texts): a sixteenth of a bit, which a table reads as the bit
    streams = m.run(text[:64])[0]
    for d, row in enumerate(streams[1:]):
        off = np.abs(after[d][1:64, :256].cpu().numpy() - row[2:]).max() / SIZE
        assert off <= 8 * STEP, f"row {d + 1} is {off:g} of a bit off"
    random = torch.Generator(device).manual_seed(0)  # then every weight of the model is off by 1%
    for k, p in models[0].named_parameters():  # of itself, or so: a norm's is 1 + what is stored
        plus = float(k.endswith("norm.weight"))
        noise = 1 + 0.01 * torch.randn(p.shape, generator=random, device=device)
        p.data = ((p.data.float() + plus) * noise - plus).to(p.dtype)
    with torch.no_grad():
        noisy = models[0](ids).logits[0]
    assert (noisy.argmax(1) == token)[speaks].float().mean() >= 0.99  # the sponge's token stays

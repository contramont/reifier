"""The compiler, stage by stage: a trace, a grid, the weights, a transformer."""

import contextlib
import io
import re
from dataclasses import replace
from itertools import pairwise, product
from pathlib import Path

import adder
import keccak
import numpy as np
import pytest

from reifier import lib
from reifier.check import (
    cases,
    check_grid,
    check_net,
    check_streams,
    check_trace,
    check_transformer,
)
from reifier.emit import emit
from reifier.fold import fold
from reifier.grid import ONE, held, layout
from reifier.lib import (
    FALL,
    add,
    and_,
    ands,
    cover,
    cubes,
    not_,
    or_,
    parity,
    steps,
    sum_of_ands,
    table,
    xor,
)
from reifier.show import show
from reifier.tiles import tiles
from reifier.trace import ACTIVATIONS, GATED, const, mlp, threshold_gate, trace
from reifier.transformer import (
    CONTEXT,
    DOWN,
    GATE,
    LIGHT,
    LOGIT,
    MARGIN,
    ONES,
    PAIRS,
    READ,
    SIZE,
    TIMES,
    UNEVEN,
    VALUE,
    Block,
    repeats,
    rmsnorm,
    runner_up,
    token_grid,
    transformer,
    uneven,
)

NETWORKS = [  # activation, skip, norm: norms need a gated activation
    (activation, skip, norm)
    for activation in ("step", "relu", "reglu", "swiglu", "geglu")
    for skip in (False, True)
    for norm in (False, True)
    if activation in GATED or not norm
]
BYTES = {n: [n >> i & 1 for i in range(8)] for n in range(256)}  # of each token, its 8 bits


def sum_bits(x):  # the bits of a + b for x = the bits of a, then of b, most significant first
    n, number = len(x) // 2, lambda bits: int("".join(map(str, bits)), 2)
    return [int(c) for c in format((number(x[:n]) + number(x[n:])) % 2**n, f"0{n}b")]


def glu_add(a, b):  # the adder with a GLU for each xor: a layer shallower than the adder of gates
    return add(a, b, parity)


def odd(x):  # an input and a constant as outputs, an input read twice, an MLP of two outputs,
    deep = xor([not_(x[1]), x[0]])  # one unread, and x0 read at depth 2 by gates made before two
    two = mlp([x[0], x[1], x[0]], [[1, 1, 1], [0, 1, 0]], [-2, -1], "step", np.eye(2), "two")
    return [x[1], const("1")[0], two[1], deep]


def ones():  # no inputs
    return const("11")


def chi(case):  # Keccak's, of a case (a, b, c) of three bits: a xor (c and not b)
    return case[0] ^ (case[2] & (1 - case[1]))


def count(bits):  # the next token's bits: the sum of those of the last two, mod 4, by gates
    return add(bits[:2], bits[2:])


def mix(bits):  # of two tokens of 3 bits, by tables, three deep: the last give 1/2 where chi0 is 1
    a, b = bits[:3], bits[3:]
    c = [table([a[i], b[i], a[i - 1]], chi, f"chi{i}") for i in range(3)]
    d = [table([c[i], c[i - 1]], lambda case: sum(case) % 2, f"odd{i}") for i in range(3)]
    return [
        table([c[0], d[i]], lambda case: 1 / 2 if case[0] else case[1], f"next{i}")
        for i in range(3)
    ]


def tangle(seed):  # a program of 16 constants, wires, gates and tables at random, on 4 bits
    rng = np.random.default_rng(seed)

    def one(xs):  # a constant, a wire, negated or not, a table, which may give 1/2, or a gate
        kind = rng.integers(5)
        values = rng.choice([0, 1, 1 / 2], 2 ** len(xs), p=[0.4, 0.4, 0.2]).tolist()
        if kind == 0:
            return const(str(rng.integers(2)))[0]
        if kind == 1:
            return not_(xs[0])
        if kind == 2:
            return threshold_gate(xs[:1], [1], 1, "same")
        if kind == 3:
            return table(xs, lambda case: values[int("".join(map(str, case)), 2)])
        return threshold_gate(xs, rng.integers(-2, 3, len(xs)).tolist(), int(rng.integers(-1, 3)))

    def program(x):
        signals = list(x)
        for _ in range(16):
            read = rng.choice(len(signals), int(rng.integers(1, 4)), replace=False)
            signals.append(one([signals[i] for i in read]))
        return signals[-4:]

    return program


def tables(seed, halves=False):  # 9 tables at random on two tokens of 3 bits, then 3 that read
    rng = np.random.default_rng(seed)  # them and give the next token: with halves, maybe 1/2

    def program(x):
        signals = list(x)
        for i in range(12):
            pool = signals[: 6 + min(i, 9)]  # an output reads no output, as it may be 1/2
            xs = [pool[j] for j in rng.choice(len(pool), int(rng.integers(1, 4)), replace=False)]
            p = [0.4, 0.4, 0.2] if halves and i >= 9 else [0.5, 0.5, 0]
            values = rng.choice([0, 1, 1 / 2], 2 ** len(xs), p=p).tolist()
            signals.append(table(xs, lambda case, v=values: v[int("".join(map(str, case)), 2)]))
        return signals[-3:]

    return program


def sponge(kinds):  # the first byte that a sponge on Keccak-p[50, 2] gives for the last 3 bytes
    return fold(trace(lambda msg: keccak.sponge(msg, 2, 50, 40, kinds)[:8], 24))


def generated(program, codes, tokens, n, window):  # tokens, and the n next ones that program
    tokens, token_of = list(tokens), {tuple(code): token for token, code in codes.items()}
    for _ in range(n):  # gives for the bits of the last `window` of them
        tokens.append(token_of[tuple(program([b for t in tokens[-window:] for b in codes[t]]))])
    return tokens


def on_tokens(program, bits, **how):  # program, on the bits of two tokens, as a transformer
    codes = {token: [int(c) for c in f"{token:0{bits}b}"] for token in range(2**bits)}
    tokens = np.random.default_rng(0).integers(0, 2**bits, 64)  # and tokens to run it on
    return transformer(trace(program, 2 * bits), codes, **how), tokens


def most(*ws):  # the largest number in the arrays ws, whatever its sign
    return max(np.abs(w).max() for w in ws)


def changed(m, f, heads=True):
    """m with f of every weight of every block, and of the norms around it: with heads, of every
    weight of the embedding, of the heads and of the other norms too"""
    blocks = [
        Block(replace(b.layer, wi=f(b.layer.wi), wo=f(b.layer.wo)), f(b.reads), f(b.writes))
        for b in m.blocks
    ]
    others = ("embed", "q", "k", "v", "o", "reads", "writes", "out") if heads else ()
    return replace(m, blocks=blocks, **{name: f(getattr(m, name)) for name in others})


def noisy(m, s, seed, heads=True):  # m with every weight times 1 + s * a normal
    rng = np.random.default_rng(seed)
    return changed(m, lambda w: w * (1 + s * rng.standard_normal(w.shape)), heads)


def rounded(m, bits):  # m with every weight rounded to `bits` significant bits
    def near(w):
        mantissa, exponent = np.frexp(w)
        return np.ldexp(np.round(mantissa * 2**bits) / 2**bits, exponent)

    return changed(m, near)


def apart(m, other, tokens):  # how far apart the streams of two transformers are, in bits, at
    window = len(m.q)  # the positions that follow as many tokens as the program reads
    pairs = zip(m.run(tokens)[0], other.run(tokens)[0])
    return max(np.abs(x - y)[window:].max() for x, y in pairs) / SIZE


def test_the_readme_prints_what_it_says():
    readme, env = (Path(__file__).parent.parent / "README.md").read_text(), {}
    blocks = re.findall(r"```python\n(.*?)```\n\n```\n(.*?)```", readme, re.DOTALL)
    assert len(blocks) == 4
    for code, printed in blocks:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(code, env)  # noqa: S102
        assert out.getvalue() == printed


def test_a_trace_is_named_by_circuit_and_computes_the_function():
    t = trace(add, 2, 2)
    assert t.inputs == ["a0", "a1", "b0", "b1"] and t.outputs == ["add/xor.1/odd", "add/xor/odd"]
    names = [m.name for m in t.mlps]  # add/or and add/or.1, which no carry reads, are gone
    assert len(set(names)) == 11 and "add/or.2" in names and "add/or" not in names
    assert trace(parity, 3).mlps[0].hidden == ["parity#0", "parity#1", "parity#2"]
    assert check_trace(trace(add, 4, 4), sum_bits) == 256
    assert check_trace(trace(glu_add, 5, 5), sum_bits) == 256  # of 1024 inputs, 256 random ones
    for n in range(6):  # the GLU parity is the xor of gates, of no bits too
        assert check_trace(trace(parity, n), trace(xor, n)) == 2**n
    with pytest.raises(TypeError, match="signals have no value"):
        trace(lambda x: x[0] if x[1] else x[1], 2)
    assert trace(lambda x: mlp(x, [1, 1, 1, 1], [-1, -2], "step", [1, -1]), 2)("10") == [1]  # flat
    assert show(trace(odd, 2)).endswith(  # an MLP that is not one gate prints its hidden units
        "  two: step, 2 hidden\n    two#0 = step(x0 + x1 + x0 - 2) -> two:0\n"
        "    two#1 = step(x1 - 1) -> two:1\n  const1 = [0 >= 0]\n  return x1 const1 two:1 xor/odd"
    )


def test_ands_are_relu_units_that_give_the_same_for_a_count_a_quarter_off():
    chi = [([1, 0, 0], 1), ([0, -1, 1], 1), ([1, -1, 1], -2)]  # x0 xor (x2 and not x1)
    relus, flat = (trace(lambda x, f=f: f(x, chi), 3) for f in (ands, sum_of_ands))
    assert check_trace(relus, flat) == 8 and len(flat.mlps[0].hidden) == 3  # the same sum of ANDs
    assert (relus.mlps[0].activation, len(relus.mlps[0].hidden)) == ("relu", 6)  # two units each
    off = np.array([[1.1, -0.05, 0.9], [0.92, 1.08, 0.1], [-0.1, 0.1, 1.04]])  # bits a little off
    assert np.allclose(relus.mlps[0](off), relus.mlps[0](off.round()), atol=1e-12)  # the same,
    assert not np.allclose(flat.mlps[0](off), flat.mlps[0](off.round()), atol=1e-3)  # not nearly
    for n in range(1, 6):  # the xor as ands: one for each way that an odd number of bits hold
        assert check_trace(trace(lib.odd, n), trace(xor, n)) == 2**n
    assert len(trace(lib.odd, 5).mlps[0].hidden) == 2 * 2**4


def test_a_table_is_a_unit_for_each_cube_of_its_cases_and_flat_at_every_case():
    for n in range(1, 5):  # any function of a few bits: the xor of gates, a cube for each case
        t = trace(lambda xs: table(xs, lambda case: sum(case) % 2), n)
        assert check_trace(t, trace(xor, n)) == 2**n and len(t.mlps[0].hidden) == 2**n + 1
    xs = np.array(list(product([0, 1], repeat=3)), float)  # the cases of three bits
    functions = [chi, min, lambda case: 1, lambda case: case[2] if case[0] else 1 / 2]
    for value, cubes_of in zip(functions, (6, 4, 1, 3)):  # chi, and, a constant, and one with 1/2
        t = trace(lambda xs, value=value: table(xs, value, "f"), 3)
        m, ys = t.mlps[0], np.array([value(case) for case in xs.astype(int)], float)
        cover = cubes(tuple(ys.tolist()))
        assert check_trace(t, value) == 8 and len(cover) == cubes_of == len(m.hidden) - 1
        signs, c = np.array([s for s, _ in cover]), np.array([y for _, y in cover]) - 1 / 2
        wrong = (signs * (2 * xs[:, None] - 1) < 0).sum(2)  # [case, cube]: the bits not as it has
        on = wrong.argmin(1)  # the cube of each case: no two share one, and it has the case's value
        assert ((wrong == 0).sum(1) == 1).all() and (c[on] + 1 / 2 == ys).all()
        n = len(cover)  # a unit for each cube, and then one on no bits, which gives the 1/2
        gate, up = (xs @ m.wi[rows].T + m.b[rows] for rows in (slice(n), slice(n + 1, 2 * n + 1)))
        assert np.array_equal(gate, 1 - FALL * wrong)  # 1 in its cube, -1/2 or less outside,
        assert (gate[wrong == 0] == 1).all() and (gate[wrong > 0] <= -1 / 2).all()
        gives = np.where(c, c, 1 / 2)  # and its value rises as much, from what the unit gives:
        assert np.array_equal(up, gives * (1 + FALL * wrong))  # the cube's value less 1/2, or 1/2
        assert not m.wi[[n, -1]].any() and m.b[[n, -1]].tolist() == [1, 1 / 2]
        rest = 1 / 2 in ys  # a value of 1/2 is neither bit: the units of its cubes give their 1/2
        to = [c != 0, c == 0][: 1 + rest]  # to a second output, the rest, and the last unit to both
        assert m.outputs == (["f:0", "f:1"] if rest else ["f"]) and t.outputs == m.outputs[:1]
        assert np.array_equal(m.wo, np.c_[to, [1] * len(to)])
        off = np.random.default_rng(0).uniform(-0.1, 0.1, xs.shape)  # each bit up to a tenth off:
        e = FALL * (signs[on] * off).sum(1)  # the gate of the unit that is on is 1 + e, and its
        flat = m(xs) - (m(xs) - 1 / 2) * e[:, None] ** 2  # value 1 - e times what it gives: so
        assert np.allclose(m(xs + off), flat, atol=1e-12)  # both outputs are flat at every case
        held_as = 2 * m(xs) - 1  # as a stream holds them, a 1/2 as 0, the squares of the two
        assert (0 in held_as[:, 0]) == rest and ((held_as**2).sum(1) == 1).all()  # sum to 1


def test_a_cover_is_the_sum_of_its_terms_where_they_share_cases_and_flat_at_every_case():
    rng, xs, shared = np.random.default_rng(0), cases(5), 0
    for _ in range(20):  # six terms at random on five bits, and a constant: (s, c) is c where s
        signs, c = rng.integers(-1, 2, (6, 5)), rng.integers(-3, 4, 6)  # holds, s a 1 for each
        terms = [*zip(map(tuple, signs.tolist()), c.tolist()), ((0,) * 5, 1)]  # bit on, -1 off
        holds = np.array([[(np.multiply(s, 2 * x - 1) >= 0).all() for s, _ in terms] for x in xs])
        m = trace(lambda x, terms=terms: cover(x, terms), 5).mlps[0]
        assert np.array_equal(m(xs).ravel(), holds @ [c for _, c in terms])  # exactly the sum
        shared += (holds[:, :-1].sum(1) > 1).any()  # where two terms hold at once, as a table's
        for i, h in product(range(5), (1e-6, -1e-6)):  # cubes may not: and in each bit, at every
            nudged = xs + h * np.eye(5)[i]  # case, its slope is 0, as a unit's gate falls by FALL
            assert np.abs(m(nudged) - m(xs)).max() < 1e-4 * abs(h)  # a bit as its value rises
    assert shared > 10


def test_steps_are_the_xor_of_many_bits_for_any_marks_and_flat_until_a_bit_is_a_room_off():
    rng = np.random.default_rng(0)
    for n in range(7, 11):
        xs = np.array(list(product([0, 1], repeat=n)), float)  # every case of n bits
        marks = [tuple(rng.integers(0, 2, n).tolist()) for _ in range(rng.integers(1, 6))]
        for flip, room in ((0, 1 / 4), (1, 1 / 4), (0, 1 / 10)):
            m = trace(lambda x, k=marks, f=flip, r=room: steps(x, k, flip=f, room=r), n).mlps[0]
            assert np.allclose(m(xs).ravel(), (xs.sum(1) + flip) % 2, atol=1e-12)
            other = 1 - 2 * xs  # for each bit, the way to its other value
            for i in range(n):  # a bit off by up to `room`, either way, changes nothing: and
                for e in (room - 0.01, 0.01 - room):  # by more, toward its other value, does
                    assert np.allclose(m(xs + e * other * np.eye(n)[i]), m(xs), atol=1e-12)
                more = m(xs + (room + 0.05) * other * np.eye(n)[i]) - m(xs)
                assert (np.abs(more) > 0.05).all()
            off = rng.uniform(-1, 1, xs.shape)  # and all bits off at once, by `room` between them
            off *= (room - 0.01) / np.abs(off).sum(1, keepdims=True)
            assert np.allclose(m(xs + off), m(xs), atol=1e-12)


def test_folding_moves_constants_into_biases_and_takes_out_the_wires():
    def program(x):  # both reads a constant only, and gives 1 and 0; not and same are wires,
        both = mlp(const("1"), [[1], [0]], [-1, -1], "step", np.eye(2), "both")  # of x0 negated
        n, same = not_(x[0]), threshold_gate([x[1]], [1], 1, "same")  # and of x1
        h = table(x[1:], lambda case: 1 / 2 if case[0] else case[1], "h")  # may be 1/2
        return [
            threshold_gate([n, same, x[2], both[0]], [1, 2, 1, 1], 3, "a"),  # reads them all
            threshold_gate([n, x[0]], [1, 1], 1, "b"),  # reads not and its bit: so not stays,
            threshold_gate([not_(h), x[0]], [1, 1], 2, "c"),  # as a wire of what may be 1/2 does,
            not_(x[2]),  # and one that is an output
        ]

    t = trace(program, 3)
    f = fold(t)
    assert check_trace(f, t) == 8  # the same outputs, and every signal that is left
    assert {m.name: m.inputs for m in f.mlps} == {  # the constants are gone, and same, which
        "not": ["x0"],  # nothing reads once `a` reads its bit
        "h": ["x1", "x2"],
        "a": ["x0", "x1", "x2"],
        "b": ["not", "x0"],
        "not.1": ["h:0"],
        "c": ["not.1", "x0"],
        "not.2": ["x2"],
    }
    a = f.mlps[2]  # [(1 - x0) + 2 x1 + x2 + 1 >= 3]: a wire's negation is in weight and bias
    assert (a.name, a.wi.tolist(), a.b.tolist()) == ("a", [[-1, 2, 1]], [-1])
    kept = set()  # of random programs: as exact, and a wire is left only for one of three
    for seed in range(100):  # reasons, each of which some programs have
        t = trace(tangle(seed), 4)
        f = fold(t)
        check_trace(f, t)
        assert all(m.inputs or {*m.outputs} & {*t.outputs} for m in f.mlps)  # no constant left
        halves = {y for m in t.mlps if len(m.outputs) > 1 for y in m.outputs}  # what may be 1/2
        for m in f.mlps:
            if len(m.inputs) == len(m.outputs) == 1 and {*m([[0], [1]]).ravel()} == {0, 1}:
                x, y = m.inputs[0], m.outputs[0]
                why = {"output": y in t.outputs, "1/2": x in halves}
                why["twice"] = any({x, y} <= {*r.inputs} for r in f.mlps)  # a reader reads both
                assert any(why.values()), f"{seed}: {y} is a wire of {x}"
                kept |= {reason for reason, held in why.items() if held}
    assert kept == {"output", "1/2", "twice"}


def test_every_network_holds_what_its_grid_says():
    for program, sizes in ((add, (3, 3)), (glu_add, (3, 3)), (odd, (2,)), (ones, ())):
        t = trace(program, *sizes)
        padded = [*layout(t), layout(t)[-1]]  # its last layer makes nothing
        for grid in (layout(t), held(layout(t)), padded):
            check_grid(t, grid)
            for activation, skip, norm in NETWORKS:
                if program is not glu_add or activation in GATED:
                    net = emit(t, grid, activation, skip, norm)
                    check_net(net, 1e-12)
                    assert np.allclose(net("101110"[: sum(sizes)]), t("101110"[: sum(sizes)]))
    assert layout(trace(list)) == [["ONE"]]  # list() is a program of no signals: one row


def test_a_layout_with_rest_writes_a_bit_where_nothing_was_else_over_one_it_is_made_of():
    def program(x):  # of a row 0 whose feature 1 holds nothing: where each signal goes, and why
        a = and_([x[2], x[3]])  # the feature that held nothing: 1
        b = or_([x[1], x[0]])  # none is left: the feature of its first input, freed here: x1's
        c = not_(x[1])  # that is taken: the feature freed last here, x3's
        s = table(x[:2], lambda case: 1 / 2 if case[0] else case[1], "s")  # can be 1/2: new ones
        d = and_([a, b, c, s])  # what s gives is freed here, but not written over: x2's, empty
        return [or_([d, a]), d]  # since depth 1; or.1 takes the one emptied last, of s's rest

    t = trace(program, 4)
    grid = layout(t, True, ["ONE", None, *t.inputs])
    assert grid == [
        ["ONE", None, "x0", "x1", "x2", "x3", None, None],
        ["ONE", "and", None, "or", None, "not", "s:0", "s:1"],
        ["ONE", "and", None, None, "and.1", None, None, None],
        ["ONE", None, None, None, "and.1", None, None, "or.1"],
    ]
    t = trace(glu_add, 5, 5)  # as narrow as without rest, where tiles, a feature for each signal,
    assert len(layout(t, rest=True)[0]) == len(layout(t)[0]) == 20 < len(tiles(t)[0])  # is not
    small = [(add, (3, 3)), (mix, (6,)), (odd, (2,)), (ones, ())]
    grids = [(s, layout(s, rest=True), ()) for s in [t, *(trace(p, *n) for p, n in small)]]
    keccak_p = keccak.program()  # and as a transformer has them, with homes for the outputs
    for s in (trace(mix, 6), sponge("22"), keccak_p):
        grids.append((s, token_grid(s), range(1, 1 + len(s.outputs))))
    for t, grid, homes in grids:
        check_grid(t, grid)
        check_grid(t, held(grid))
        check_net(emit(t, grid, "geglu", skip=True), 1e-12)  # a grid as any other, to emit too
        assert len(grid) == len(layout(t))  # and as deep
        several = {y for m in t.mlps if len(m.outputs) > 1 for y in m.outputs}
        made, ends, over, own = {y: m for m in t.mlps for y in m.outputs}, {}, 0, 0
        for d, (before, row) in enumerate(pairwise(grid), 1):
            empty = [g for g, cell in enumerate(zip(before, row)) if cell == (None, None)]
            empty = [g for g in empty if g not in homes and any(r[g] for r in grid[:d])]
            for old, new in zip(before, row):
                if old and new and old != new:  # written over another: neither can be 1/2,
                    assert old not in several and new not in several  # and no feature that
                    assert not empty  # held nothing, and something before, is left empty
                    over, own = over + 1, own + (old in made[new].inputs)
            ends |= {s: d for s in before if s and s not in row}
        for m in t.mlps:  # what an MLP of several outputs gives is held, and freed, together
            assert len({ends.get(y) for y in m.outputs}) == 1 or len(m.outputs) == 1
        if t is keccak_p:  # Keccak's rounds of three layers write each theta over the bit it is
            assert over == own == 94  # the xor of, and each chi over a theta it is made of
    t = trace(mix, 6)  # another row 0, whose features 1 to 3 hold nothing, and homes there of
    first, home = ["ONE", None, None, None, *t.inputs], dict(zip(t.outputs, (3, 1, 2)))  # the
    plain, grid = layout(t, True, first), layout(t, True, first, home)  # outputs: without them
    assert plain[1][1:4] == ["chi2", "chi1", "chi0"]  # the first signals take those features.
    assert [row[1:5] for row in grid] == [  # A home is lent to a signal that is freed before the
        [None, None, None, "bits0"],  # home's own signal is made: to chi2 and chi1, which odd
        [None, "chi2", "chi1", "chi0"],  # reads at depth 2, not chi0, which next reads at 3. So
        [None, None, None, "chi0"],  # each output is written where the row before held nothing
        ["next1:0", "next2:0", "next0:0", None],
    ]
    t = trace(odd, 2)  # what nothing reads of an MLP is held as long as what is read of it
    assert "two:0" in layout(t, rest=True)[-1] and "two:0" not in layout(t)[-1]
    assert layout(trace(list), rest=True) == [["ONE"]]


def test_every_hidden_unit_has_a_name():
    t = trace(xor, 3)
    g = layout(t)
    erases = ["erase xor/atleast3", "erase xor/atleast2", "erase xor/atleast1"]
    assert emit(t, g, "relu", skip=True).layers[1].hidden == ["xor/odd+", "xor/odd-", *erases]
    kept = emit(t, held(g), "geglu", skip=True, norm=True)  # only what is written over goes
    assert kept.layers[1].hidden == ["ONE", "xor/odd+", "xor/odd-", "erase xor/atleast3"]
    assert emit(t, held(g), "step").layers[1].hidden[:3] == ["ONE", "xor/odd", "keep xor/atleast2"]
    assert show(kept).startswith("xor: skip + norm, 4 features, ONE = 4\nlayer 1: 10 hidden units")
    assert show(kept).endswith("erase xor/atleast3 = gelu(32 ONE) * (0.03125 xor/atleast3) -> -f1")


def test_deep_networks_stay_exact():
    def program(x):  # 100 layers, each of which writes every feature anew
        for _ in range(100):
            x = [not_(x[i - 1]) for i in range(len(x))]
        return x

    t = trace(program, 4)
    assert check_trace(t, lambda x: x) == 16  # 100 nots and 100 turns: every bit as it was
    for activation, skip, norm in NETWORKS[-8:]:  # swiglu and geglu
        net = emit(t, held(layout(t)), activation, skip, norm)
        check_net(net, 1e-12)
    values = t.values(cases(4))  # the last: geglu, with skip connections and norms
    off = 1e-6 * np.random.default_rng(0).standard_normal((16, 5))
    streams = net.run(net.stream(values, 0) + off)  # with ONE held as 1, this ends in wrong bits
    check_streams(net, streams, values, 1e-5)


def test_each_check_names_what_is_wrong():
    t = trace(add, 2, 2)
    with pytest.raises(AssertionError, match="add/xor.1/odd is 0 on 0100, not 1"):
        check_trace(t, lambda x: [x[0] | x[1], 0])
    with pytest.raises(AssertionError, match="add/and is 0 on 0100, not 1"):  # the first to differ
        check_trace(t, trace(lambda a, b: add(a, a), 2, 2))
    with pytest.raises(AssertionError, match="add gives 2 outputs, not 1"):
        check_trace(t, lambda x: [0])
    with pytest.raises(AssertionError, match="^lambda: two signals are named x0"):
        check_trace(trace(lambda x: mlp(x, [[1, 1]], [0], "step", [[1]], "x0"), 2), sum)
    for d, f, cell, message in [  # a cell of the grid that is wrong or gone: what check_grid says
        (0, 1, [None], "depth 0 lacks an input"),
        (4, 2, [None], "the last depth lacks an output"),
        (0, 5, ["add/and"], "depth 0 holds a signal that an MLP makes"),
        (2, 0, [None, "ONE"], "ONE is not feature 0 of every row"),
        (2, 6, [], "the rows differ in width"),
        (2, 1, [None], "is made at depth 3, but depth 2 lacks {'a0'}"),
        (4, 5, ["add/and"], "add/and is new at depth 4: no MLP makes it there"),
        (4, 5, ["add/xor.1/odd"], "add/xor.1/odd is new at depth 4: no MLP makes it there"),
    ]:
        g = layout(t)
        g[d][f : f + 1] = cell
        with pytest.raises(AssertionError, match=message):
            check_grid(t, g)
    net = emit(t, layout(t), "reglu", skip=True)
    net.layers[2].wo[1, 0] += 1e-6  # into feature 1, which holds nothing at depth 3
    with pytest.raises(AssertionError, match=r"depth 3, feature 1 \(no signal\) is"):
        check_net(net)
    with pytest.raises(AssertionError, match="parity has no step form"):
        emit(trace(parity, 2), layout(trace(parity, 2)), "step")
    given = trace(lambda x: [x[0], not_(x[1])], 2)  # a program on tokens: its first output is an
    with pytest.raises(AssertionError, match="lambda: each output must be a signal of its own"):
        token_grid(given)  # input, which no MLP makes where the logits read it
    early = trace(lambda x: [not_(x[0]), not_(not_(x[1]))], 2)  # the first block makes an output:
    assert token_grid(early)[1][1:3] == ["not", None]  # it may, unless the last row is carried on
    with pytest.raises(AssertionError, match="lambda: its last block must make every output"):
        token_grid(early, carry=True)
    heavy = trace(lambda x: mlp(x, [[3, 1]], [-2], "step", [[1]], "heavy"), 2)  # [3 x0 + x1 >= 2]
    with pytest.raises(AssertionError, match="heavy: a gate sums too near 0: .* would be 2.15"):
        transformer(heavy, {0: [0], 1: [1]})  # a weight of 3 is 3 * 4 / 3 * GATE / READ: over 2


def test_a_transformer_adds():
    symbols = adder.SYMBOLS
    m = transformer(adder.program(), dict(enumerate(adder.CODES.values())))
    for a, b in [(123, 246), (999, 999), (0, 0), (5, 995), (608, 397)]:
        tokens = m.generate([symbols.index(c) for c in f"{a:03}+{b:03}="], 3)
        assert "".join(symbols[i] for i in tokens[-3:]) == f"{(a + b) % 1000:03}"
    sums = np.random.default_rng(0).integers(0, 1000, (6, 2))
    text = "".join(f"{a:03}+{b:03}={(a + b) % 1000:03}" for a, b in sums)
    # to 2% of a bit's size: a carry is a sum of 100 ands, each a little on where it is off
    check_transformer(m, [symbols.index(c) for c in text], 0.02)
    b = m.blocks[0]  # the norm after it gives half as much again: so does the block, to each bit,
    wrong = replace(m, blocks=[Block(b.layer, b.reads, 1.5 * b.writes), *m.blocks[1:]])  # which
    with pytest.raises(AssertionError, match=r"depth 1, feature .* 2\.\d+ from what the grid says"):
        check_transformer(wrong, [symbols.index(c) for c in "123+246=369"])  # is a bit off where
    # it was written over another bit: the feature gets the new bit less the old, twice a bit
    read = set(m.grid[0]) - {None, "ONE"}  # of its 96 inputs the program reads 63, and of the
    assert len(read) == 63 and "bits95" in read and "bits84" not in read  # newest token whether it
    assert np.count_nonzero(m.o) == 12 + 63  # is "=", alone: no head copies what is not read
    m.embed[9, 1] += 0.5  # bit 0 of the code of "9" is a quarter off: where 9 is the newest token
    check_transformer(m, [symbols.index(c) for c in "123+246=369"])  # head 0 erases it, as it is
    m.embed[9, 12] += 0.5  # and now the bit that is read there is, too: so is the stream
    with pytest.raises(AssertionError, match=r"depth 0, feature \d+ \(bits95\) is -1\.5"):
        check_transformer(m, [symbols.index(c) for c in "123+246=369"])
    m.embed[9, 1:13] *= 3  # and now its code is three times too large: it has the largest logit
    with pytest.raises(AssertionError, match="position 8: the next token is 9, not 3"):
        check_transformer(m, [symbols.index(c) for c in "123+246=36"])


def test_a_token_without_a_code_changes_nothing():
    m = transformer(trace(count, 4), {n + 1: [n >> 1, n & 1] for n in range(4)})
    assert m.generate([0, 2, 3], 3) == [0, 2, 3, 4, 2, 1]  # token 0 is all zeros, and stays so,
    streams, logits = m.run([0, 2, 3])  # as the <bos> before it, at every depth
    assert not any(x[:2].any() for x in streams) and not logits[:2].any()
    text = show(m)
    assert text.startswith(  # a weight is on what a norm gives: READ for ONE, and for a bit
        "2 heads on 4 tokens, 23 features\nhead 0: the code of the token 0 back -> bits2 bits3\n"
        "head 1: the code of the token 1 back -> bits0 bits1\nblock 1: 5 hidden units of geglu\n"
        "  add/and+ = gelu(-0.538462 ONE + 0.717949 bits1 + 0.717949 bits3) * (0.323077 ONE) "
        "-> 0.5 f2\n  add/and- = gelu(-0.897436 ONE + 0.717949 bits1 + 0.717949 bits3) * "
        "(0.323077 ONE) -> -0.5 f2\n"  # a block erases again what the heads erased, the code,
        "  erase what is left = gelu(0.538462 ONE) * (0.107692 f1) -> -0.5 f1\n"
    )
    assert (  # a bit written over another goes to its feature, and to a row that the norm drops
        "  add/or.2+ = gelu(0.179487 ONE + 0.717949 add/and) * (0.323077 ONE) -> 0.5 f1 + 0.5 f1, "
        "dropped\n" in text
    )
    assert text.endswith(  # and what the block before it erased and left empty: what that left
        "  erase what is left = gelu(0.538462 ONE) * (0.107692 f21) -> -0.5 f21\n"
        "  ONE = gelu(0.538462 ONE) * (0.107692 ONE) -> -0.5 f1"
    )


def test_a_transformer_of_tables_or_of_gates_is_right_and_has_no_large_weight():
    # heaviest is the largest weight of a gate, light the largest of a value, as a part of it,
    # and hidden the largest hidden unit. A table's unit is on at 2 GATE, and its value is LIGHT
    # of that, as its weights are of its gate's. A threshold gate is two units, on at up to
    # 7 GATE: the value of each is a bias, in all LIGHT of a gate's heaviest weight, on the ONES
    # between them. The heaviest weight of a value is then of a unit that erases: a third of it
    for program, bits, heaviest, light, hidden in (
        (mix, 3, FALL * GATE / READ, LIGHT, LIGHT * (2 * GATE) ** 2),
        (count, 2, 4 / 3 * GATE / READ, LIGHT / 3, 7 * GATE * LIGHT * 4 / 3 * GATE),
    ):
        m, tokens = on_tokens(program, bits)
        check_grid(m.trace, m.grid)
        check_transformer(m, tokens, 0.001)  # every feature at every depth, to 0.1% of a ONE
        weights = [most(m.embed), most(m.k), most(m.v), most(m.o)]  # the largest ONE is UNEVEN
        assert np.isclose(weights, [SIZE * (1 + UNEVEN), VALUE, VALUE, 1]).all()  # more than SIZE
        # a query and a key read the ONES, each READ, on PAIRS of dimensions: a head scores its
        # token `own`, and any other in CONTEXT MARGIN less at least. A query's weights are small
        own = PAIRS * (READ * ONES) ** 2 * VALUE * most(m.q)
        assert np.isclose(own * (1 - runner_up(m.turns, CONTEXT)), MARGIN) and most(m.q) < VALUE
        # the norms before read the smallest ONE as READ, in a row whose root mean square is at
        # most that of a row of the ONEs, and the norms after give a feature its size
        full = np.sqrt(np.mean(uneven(ONES) ** 2))
        assert most(m.reads, *(b.reads for b in m.blocks)) <= READ * full / (1 - UNEVEN)
        assert most(m.writes, *(b.writes for b in m.blocks)) <= SIZE
        assert most(m.out) <= LOGIT / SIZE / bits and heaviest < 1
        streams, logits = m.run(tokens)
        largest = []  # the largest hidden unit of each block
        for b, x in zip(m.blocks, streams):
            gate, up = np.split(b.layer.wi, 2)  # a value is lighter than a gate, and every unit
            assert np.isclose([most(gate), most(up)], [heaviest, light * heaviest]).all()  # gives
            assert (np.abs(b.layer.wo).max(0) == DOWN).all() and not b.layer.b.any()  # as much
            z = rmsnorm(x[2:], b.reads)  # what the block reads, where the grid holds: a bit is
            sums = z @ b.layer.wi.T  # READ, and a gate GATE from 0 or more
            gates = np.split(sums, 2, 1)[0]
            assert most(z) <= 1.01 * READ and 0.99 * GATE <= np.abs(gates).min()
            assert most(gates) <= 7.01 * GATE
            largest.append(most(ACTIVATIONS["geglu"](sums)))
        assert np.isclose(max(largest), hidden, rtol=0.01)
        top = logits[2:].max(1)  # the largest logit at each position: the right token's, LOGIT
        silent = np.abs(top) < 0.01  # but where mix gives 1/2, held as 0: no token has a logit
        assert silent.any() == (program is mix) and not silent.all()
        assert np.allclose(top[~silent], LOGIT, rtol=0.001)


def test_a_block_writes_a_bit_over_another_as_the_new_less_the_old_and_drops_their_sum():
    # count writes 6 bits over others, in its second and third blocks, and mix one, in its
    # first: here on every case of what each reads, the bits of two tokens
    for program, bits, written in ((count, 2, 6), (mix, 3, 1)):
        codes = {token: [int(c) for c in f"{token:0{bits}b}"] for token in range(2**bits)}
        tokens = [token for pair in product(codes, repeat=2) for token in pair]
        m = transformer(trace(program, 2 * bits), codes)
        streams = m.run(tokens)[0]
        for d, (b, x, y) in enumerate(zip(m.blocks, streams, streams[1:]), 1):
            width, before, row = len(b.writes), m.grid[d - 1], m.grid[d]
            over = [f for f, (s, r) in enumerate(zip(before, row)) if s and r and s != r]
            assert b.layer.outputs[width:] == [f"f{f}, dropped" for f in over]  # a row more each
            given = b.layer(rmsnorm(x[2:], b.reads))  # all that the layer gives, those rows too
            old, new = x[2:, over] / SIZE, y[2:, over] / SIZE  # the bits, as -1 or 1
            k = ((new - old) * given[:, over] + (new + old) * given[:, width:]).sum()
            k /= 4 * max(old.size, 1)  # A feature written over gets the new bit less the old,
            assert np.allclose(given[:, over], k * (new - old), atol=0.005 * k)  # 0 or twice a
            assert np.allclose(given[:, width:], k * (new + old), atol=0.005 * k)  # bit, and its
            square = (given * given).sum(1)  # row the two added up: so all that the layer gives
            alone = (given[:, :width] ** 2).sum(1)  # is as large whatever the bits are, to
            assert np.ptp(square) < 0.002 * square.mean()  # 0.2%, where what it gives the
            assert np.ptp(alone) > 0.3 * alone.mean() or not over  # features alone is not
            assert b(x).shape[1] == width  # The norm after counts the rows that it then drops
            assert np.allclose(
                b(x[2:]), b.writes * given[:, :width] / np.sqrt(square / width)[:, None]
            )
            written -= len(over)
        assert written == 0
    t = trace(mix, 6)  # A layout without rest writes next0:1, a rest, which can be 1/2, over
    read = [x for x in t.inputs if any(x in m.inputs for m in t.mlps)]  # odd2: no block may
    first = [ONE, None, None, None, *[ONE] * (ONES - 1), *read]  # (as token_grid would have it)
    plain = layout(t, False, first, {y: 1 + i for i, y in enumerate(t.outputs)})
    assert ("odd2", "next0:1") in {(s, r) for rows in pairwise(plain) for s, r in zip(*rows)}
    with pytest.raises(AssertionError, match="a signal that can be 1/2 is written over another"):
        transformer(t, {n: [n >> 2, n >> 1 & 1, n & 1] for n in range(8)}, grid=plain)


def test_a_block_erases_again_what_the_block_before_erased_and_left_empty():
    m = transformer(keccak.program(), keccak.CODES)  # the first block, the code, which the
    for d, b in enumerate(m.blocks, 1):  # heads erased; none, what a block wrote a bit over
        value = np.split(b.layer.wi, 2)[1]  # the one feature that an eraser's value reads
        units = [j for j, h in enumerate(b.layer.hidden) if h == "erase what is left"]
        left = {int(np.flatnonzero(value[j])[0]) for j in units}
        cells = zip(m.grid[max(d - 2, 0)], m.grid[d - 1])
        assert left == (
            {*range(1, 9)} if d == 1 else {f for f, (s, r) in enumerate(cells) if s and not r}
        )
    tokens = keccak.FIRST + np.random.default_rng(0).integers(0, 256, 32)
    for seed in range(2):  # with every weight 3% off, what a block leaves of a feature that it
        streams = noisy(m, 0.03, seed).run(tokens)[0]  # erased is a tenth after the next
        for d in range(2, len(m.grid)):
            gone = [f for f, s in enumerate(m.grid[d - 2]) if s and not m.grid[d - 1][f]]
            gone = [f for f in gone if not m.grid[d][f]]
            if gone:
                once, twice = (np.abs(streams[e][3:, gone]).max() for e in (d - 1, d))
                assert twice < 0.2 * once and once > 0.1 * SIZE


def test_a_block_adds_the_same_to_a_stream_of_any_size():
    for program, bits in ((mix, 3), (count, 2)):
        m, tokens = on_tokens(program, bits)
        streams, _ = m.run(tokens)
        for b, x in zip(m.blocks, streams):
            x, adds = x[2:], b(x[2:])  # where the grid holds
            for s in (0.5, 3):  # a stream s times as large: the norm before the block divides
                assert np.allclose(b(s * x), adds, atol=1e-7)  # by its root mean square
            # Were that norm to give s times as much, each unit would give s * s times as much,
            # where gelu is relu, and the norm after would divide by it: any more is as good,
            # and a fifth less, to 1% of a bit. At half, a gate is GATE / 2 from 0, and a bit is
            # up to 19% off
            for s, tol in ((3, 0.002), (0.8, 0.02), (0.5, 0.25)):
                assert most(Block(b.layer, s * b.reads, b.writes)(x) - adds) <= tol * SIZE
        # So with a transformer whose norms are over 1024 features, the squares of the others
        # 1000 between them: it is as right if they are less, to 1% of a bit if they are half as
        # much again, and to 23% if they are 4 times as much, where a norm gives half
        width = len(m.grid[0])
        beside, _ = on_tokens(program, bits, beside=(1024 - width, 1000 / 1024))
        assert (beside.size, beside.rest) == (1024, 1000.0)
        for rest, tol in ((1000.0, 0.001), (10.0, 0.001), (1500.0, 0.012), (4000.0, 0.3)):
            check_transformer(replace(beside, rest=rest), tokens, tol / (1 + UNEVEN))


def test_a_block_has_its_units_as_many_times_as_fit_and_adds_the_same(monkeypatch):
    t = keccak.program()  # blocks of 64 to 360 units
    m, tokens = transformer(t, keccak.CODES), keccak.FIRST + np.arange(0, 256, 9)
    streams = m.run(tokens)[0]
    monkeypatch.setattr("reifier.transformer.EPS", 0.0)  # (but for that, to 5e-8)
    for hidden in (300, 700, 4096):
        wide = transformer(t, keccak.CODES, hidden)
        for d, (b, w, x) in enumerate(zip(m.blocks, wide.blocks, streams), 1):
            before, row, n = m.grid[d - 1], m.grid[d], len(b.layer.hidden)
            over = {f"erase {s}" for s, r in zip(before, row) if s and r and s != r}
            copies = repeats(b.layer, [s == ONE for s in before], over, hidden)
            assert copies.min() >= 1 and copies.max() <= TIMES  # each unit once at least, TIMES
            assert len(w.layer.hidden) == copies.sum() <= max(hidden, n)  # at most, and in all
            base = min(TIMES, max(1, hidden // n))  # no more than it may have, if each fits once
            mlps = {}  # each as many times as fit, and those left over a copy at a time to the
            for j, h in enumerate(b.layer.hidden):  # whole of an MLP, or to a unit that erases
                mlps.setdefault(h.split("#")[0], set()).add(int(copies[j]))  # a bit that one is
                if h == ONE or h.startswith("erase ") and h not in over:  # written over: no
                    assert copies[j] == base  # other gets more
            assert all(len(times) == 1 for h, times in mlps.items() if not h.startswith("erase"))
            assert hidden < TIMES * n or (copies == TIMES).all()
            assert np.allclose(w(x[3:]), b(x[3:]), atol=1e-9)  # what a unit gives, a copy of it
            # gives as many times less: what the norm after divides by


def test_the_ones_are_of_uneven_sizes_and_shares_of_a_bias_which_change_no_bit(monkeypatch):
    sizes = uneven(ONES)  # from 1 - UNEVEN to 1 + UNEVEN in equal steps, and 1 on average
    assert np.allclose(sizes, np.linspace(1 - UNEVEN, 1 + UNEVEN, ONES))
    assert np.isclose(sizes.mean(), 1)
    assert uneven(1).tolist() == [1.0]
    t, tokens = sponge("22"), np.random.default_rng(0).integers(0, 256, 32)
    m = transformer(t, BYTES)
    ones = [f for f, s in enumerate(m.grid[0]) if s == ONE]  # a ONE is embedded as large as
    assert np.allclose(m.embed[:, ones], SIZE * sizes)  # its share of every bias, and read as
    assert np.allclose(m.sizes[:, ones], SIZE * sizes)  # large as any other: so a bias is what
    for d, b in enumerate(m.blocks, 1):  # an even share makes it
        for half in np.split(b.layer.wi, 2):
            biased = half[:, ones][half[:, ones].any(1)]
            assert np.allclose(biased, biased.sum(1, keepdims=True) * sizes / ONES)
        assert np.ptp(b.reads[ones] * m.sizes[d - 1, ones]) < 1e-12
    monkeypatch.setattr("reifier.transformer.UNEVEN", 0.0)
    even = transformer(t, BYTES)  # with ONEs all alike: every bit at every depth as it was
    assert most(even.embed[:, ones]) == SIZE and np.isclose(apart(m, even, tokens), UNEVEN)
    for d, (x, y) in enumerate(zip(m.run(tokens)[0], even.run(tokens)[0])):  # (the ONEs are not)
        bit = [s not in (None, ONE) for s in m.grid[d]]
        assert np.allclose(x[3:, bit], y[3:, bit], atol=1e-6)
    # but with every weight rounded to 6 bits, its bits are off by less than half as much: what
    # each ONE's weights are off by is another part of a step, and they add up to less
    assert apart(m, rounded(m, 6), tokens) < 0.5 * apart(even, rounded(even, 6), tokens)


def test_a_sponge_in_rounds_of_three_layers_is_right_with_weights_one_and_a_half_percent_off():
    t = keccak.program()  # a sponge on Keccak-p[50, 2] and the table of the next token, on 24
    m = transformer(t, keccak.CODES)  # bits: 146 tables, seven blocks, on 66 features, where
    tokens = keccak.FIRST + np.random.default_rng(0).integers(0, 256, 32)  # most bits are
    check_transformer(m, tokens, 0.001)  # written over those they are made of
    assert (len(t.inputs), len(t.mlps), len(m.blocks), len(m.grid[0])) == (24, 146, 7, 66)

    def plain(bits):  # the next token's code, of Keccak on plain bits
        y = keccak.sponge(bits)[:8]
        return y if y[7] else keccak.CODES[keccak.DASH]

    start = [keccak.DASH] * 3  # it goes on from "-" alone
    assert m.generate(start, 12) == generated(plain, keccak.CODES, start, 12, 3)
    streams, logits = m.run(tokens)
    want = logits[3:].argmax(1)  # the program's next tokens, as the check says
    for seed in range(3):  # every bit at every depth within half its size, and every next token,
        check_transformer(noisy(m, 0.015, seed), tokens, 1 / 3)  # with every weight 1.5% off,
        assert (noisy(m, 0.1, seed).run(tokens)[1][3:].argmax(1) == want).mean() < 1 / 2  # and
    # not half of them with 10%: all are right up to 3% (on 20 seeds), and not at 4%

    def off(m):  # the root mean square of what m's streams are off by, where the grid holds
        return np.sqrt(np.mean([(x - y)[3:] ** 2 for x, y in zip(m.run(tokens)[0], streams)]))

    wide = transformer(t, keccak.CODES, hidden=4096)  # each unit TIMES times: what their weights
    for seed in range(3):  # are off by averages, though not what those of the norms are: 0.67
        once, often = (off(noisy(x, 0.03, seed, heads=False)) for x in (m, wide))
        assert often < 0.8 * once


def test_a_sponge_in_rounds_of_two_layers_is_keccak_and_right_with_weights_off_by_less():
    t = sponge("22")  # each round two blocks: d, the xor of two columns, a table of the few of
    m = transformer(t, BYTES)  # their bits that are not constants, or steps of their count;
    tokens = np.random.default_rng(0).integers(0, 256, 32)  # then chi, of the bits before
    check_transformer(m, tokens, 0.0025)  # theta, each with its d: a cover of 6 bits
    assert (len(t.mlps), len(m.blocks), len(m.grid[0])) == (78, 4, 66)  # on the same features
    assert sum(x.activation == "relu" for x in t.mlps) == 10  # the steps, of the second round

    def plain(bits):  # Keccak on plain bits, where every kind of round is the same
        return keccak.sponge(bits, 2, 50, 40)[:8]

    assert m.generate([1, 2, 3], 12) == generated(plain, BYTES, [1, 2, 3], 12, 3)
    want = m.run(tokens)[1][3:].argmax(1)
    for seed in range(3):  # its units give more, of more bits: all right up to 1.5% (on 10
        check_transformer(noisy(m, 0.0075, seed), tokens, 1 / 3)  # seeds), not at 2%
        assert (noisy(m, 0.1, seed).run(tokens)[1][3:].argmax(1) == want).mean() < 1 / 2


def test_small_programs_at_random_are_right_as_transformers_and_generate_what_they_give():
    codes = {n: [n >> 2, n >> 1 & 1, n & 1] for n in range(8)}
    for seed in range(20):  # 12 tables each, of up to 3 bits: the last give 1/2 if halves
        t = fold(trace(tables(seed, halves=seed % 2), 6))
        m, tokens = transformer(t, codes), np.random.default_rng(seed).integers(0, 8, 40)
        check_transformer(m, tokens, 0.001)
        if seed % 2 == 0:
            assert m.generate([1, 2], 10) == generated(t, codes, [1, 2], 10, 2)

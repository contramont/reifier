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
from reifier.grid import held, layout
from reifier.lib import FALL, add, ands, cubes, not_, parity, sum_of_ands, table, xor
from reifier.show import show
from reifier.trace import ACTIVATIONS, GATED, const, mlp, trace
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
    VALUE,
    Block,
    rmsnorm,
    runner_up,
    token_grid,
    transformer,
)

NETWORKS = [  # activation, skip, norm: norms need a gated activation
    (activation, skip, norm)
    for activation in ("step", "relu", "reglu", "swiglu", "geglu")
    for skip in (False, True)
    for norm in (False, True)
    if activation in GATED or not norm
]


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


def on_tokens(program, bits, **how):  # program, on the bits of two tokens, as a transformer
    codes = {token: [int(c) for c in f"{token:0{bits}b}"] for token in range(2**bits)}
    tokens = np.random.default_rng(0).integers(0, 2**bits, 64)  # and tokens to run it on
    return transformer(trace(program, 2 * bits), codes, **how), tokens


def most(*ws):  # the largest number in the arrays ws, whatever its sign
    return max(np.abs(w).max() for w in ws)


def noisy(m, s, seed, heads=True):
    """m with every weight of every block, and of the norms around it, times 1 + s * a normal:
    with heads, every weight of the embedding, of the heads and of the other norms too"""
    rng = np.random.default_rng(seed)

    def off(w):
        return w * (1 + s * rng.standard_normal(w.shape))

    blocks = [
        Block(replace(b.layer, wi=off(b.layer.wi), wo=off(b.layer.wo)), off(b.reads), off(b.writes))
        for b in m.blocks
    ]
    others = ("embed", "q", "k", "v", "o", "reads", "writes", "out") if heads else ()
    return replace(m, blocks=blocks, **{name: off(getattr(m, name)) for name in others})


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


def test_a_layout_with_rest_writes_a_signal_only_where_the_row_before_held_none():
    def written_over(grid):  # the cells that hold a signal where the row before held another
        return sum(bool(a and b and a != b) for rows in pairwise(grid) for a, b in zip(*rows))

    for program, sizes in ((glu_add, (5, 5)), (add, (3, 3)), (mix, (6,)), (odd, (2,)), (ones, ())):
        t = trace(program, *sizes)
        plain, grid = layout(t), layout(t, rest=True)
        check_grid(t, grid)
        check_grid(t, held(grid))
        check_net(emit(t, grid, "geglu", skip=True), 1e-12)  # a grid as any other, to emit too
        again = sum(len(set(column) - {None}) > 1 for column in zip(*grid))  # features taken again
        assert written_over(grid) == 0 and len(grid) == len(plain)  # a depth after they are freed
        if program is not ones:  # which has no feature to take again
            assert written_over(plain) and again and len(grid[0]) > len(plain[0])
        if program is glu_add:  # where tiles, a feature for each signal, is 34 wide
            assert (len(plain[0]), len(grid[0]), written_over(plain), again) == (20, 27, 14, 7)
        if program is odd:  # what nothing reads of an MLP is held as long as what is read of it
            assert "two:0" in grid[-1] and "two:0" not in plain[-1]
    assert layout(trace(list), rest=True) == [["ONE"]]
    t = trace(mix, 6)  # with another row 0, whose features 1 to 3 hold nothing, the first signals
    first, home = ["ONE", None, None, None, *t.inputs], dict(zip(t.outputs, (3, 1, 2)))
    plain, grid = layout(t, True, first), layout(t, True, first, home)  # made take those: unless
    assert plain[1][1:4] == ["chi2", "chi1", "chi0"] and grid[0] == first + [None] * 3  # they are
    # homes, here of the outputs. Each is then made in its home, which holds no other signal
    assert [row[1:4] for row in grid] == [[None] * 3] * 3 + [["next1:0", "next2:0", "next0:0"]]
    check_grid(t, grid)
    assert written_over(grid) == 0 and grid[1][-3:] == ["chi0", "chi1", "chi2"]


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
    b = m.blocks[0]  # the norm after it gives half as much again: so does the block, to each bit
    wrong = replace(m, blocks=[Block(b.layer, b.reads, 1.5 * b.writes), *m.blocks[1:]])
    with pytest.raises(AssertionError, match=r"depth 1, feature .* 1\.\d+ from what the grid says"):
        check_transformer(wrong, [symbols.index(c) for c in "123+246=369"])
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
    assert show(m).startswith(  # a weight is on what a norm gives: READ for ONE, and for a bit
        "2 heads on 4 tokens, 28 features\nhead 0: the code of the token 0 back -> bits2 bits3\n"
        "head 1: the code of the token 1 back -> bits0 bits1\nblock 1: 5 hidden units of geglu\n"
        "  add/and+ = gelu(-0.538462 ONE + 0.717949 bits1 + 0.717949 bits3) * (0.323077 ONE) "
        "-> 0.5 f22\n  add/and- = gelu(-0.897436 ONE + 0.717949 bits1 + 0.717949 bits3) * "
        "(0.323077 ONE) -> -0.5 f22\n"  # a block erases again what the heads erased, the code,
        "  erase what is left = gelu(0.538462 ONE) * (0.107692 f1) -> -0.5 f1\n"
    )
    assert show(m).endswith(  # and what the block before it erased: what that left
        "  erase what is left = gelu(0.538462 ONE) * (0.107692 f27) -> -0.5 f27\n"
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
        check_transformer(m, tokens, 0.001)  # every feature at every depth, to 0.1% of a bit
        weights = [most(m.embed), most(m.k), most(m.v), most(m.o)]
        assert np.isclose(weights, [SIZE, VALUE, VALUE, 1]).all()
        # a query and a key read the ONES, each READ, on PAIRS of dimensions: a head scores its
        # token `own`, and any other in CONTEXT MARGIN less at least. A query's weights are small
        own = PAIRS * (READ * ONES) ** 2 * VALUE * most(m.q)
        assert np.isclose(own * (1 - runner_up(m.turns, CONTEXT)), MARGIN) and most(m.q) < VALUE
        assert most(m.reads, *(b.reads for b in m.blocks)) <= READ  # the norms before, and
        assert most(m.writes, *(b.writes for b in m.blocks)) <= SIZE  # after: a feature's size
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


def test_a_block_adds_the_same_to_a_stream_of_any_size_and_with_each_unit_many_times():
    for program, bits in ((mix, 3), (count, 2)):
        m, tokens = on_tokens(program, bits)
        streams, _ = m.run(tokens)
        for b, x in zip(m.blocks, streams):
            x, adds = x[2:], b(x[2:])  # where the grid holds
            for s in (0.5, 3):  # a stream s times as large: the norm before the block divides
                assert np.allclose(b(s * x), adds, atol=1e-7)  # by its root mean square
            # Were that norm to give s times as much, each unit would give s * s times as much,
            # where gelu is relu, and the norm after would divide by it: any more is as good,
            # and a fifth less. At half, a gate is GATE / 2 from 0, and a bit is up to 16% off
            z = rmsnorm(x, b.reads)
            for s, tol in ((3, 0.001), (0.8, 0.01), (0.5, 0.2)):
                assert most(rmsnorm(b.layer(s * z), b.writes) - adds) <= tol * SIZE
        # So with a transformer whose norms are over 1024 features, the squares of the others
        # 1000 between them: it is as right if they are less, to 1% of a bit if they are half as
        # much again, and to 20% if they are 4 times as much, where a norm gives half
        beside, _ = on_tokens(program, bits, rest=1000.0, size=1024)
        for rest, tol in ((1000.0, 0.001), (10.0, 0.001), (1500.0, 0.01), (4000.0, 0.2)):
            check_transformer(replace(beside, rest=rest), tokens, tol)
        wide, _ = on_tokens(program, bits, hidden=64)  # each unit as many times as fit in 64,
        for b, w in zip(m.blocks, wide.blocks):  # up to TIMES
            times = min(TIMES, 64 // len(b.layer.hidden))
            assert times > 1 and w.layer.hidden == times * b.layer.hidden
            assert np.array_equal(w.layer.wo, np.tile(b.layer.wo, times))
            assert np.array_equal(w.reads, b.reads) and np.array_equal(w.writes, b.writes)
        # they give as many times as much, which the norm after divides by, but for its eps:
        often = wide.run(tokens)[0]  # the same streams, to a hundred-thousandth of a bit
        assert all(np.allclose(x, y, atol=1e-5 * SIZE) for x, y in zip(often, streams))
    assert len(wide.blocks[0].layer.hidden) == TIMES * 5  # of count's 5: more would fit


def test_a_sponge_is_right_with_every_weight_three_percent_off_and_not_with_ten():
    t = keccak.program()  # a sponge on Keccak-p[50, 2], on 24 bits: 166 tables, nine deep
    codes = dict(enumerate(keccak.CODES.values()))
    m, tokens = transformer(t, codes), np.random.default_rng(0).integers(0, 256, 32)
    check_transformer(m, tokens, 0.001)
    assert (len(t.inputs), len(t.mlps), len(m.blocks), len(m.grid[0])) == (24, 166, 9, 124)
    streams, logits = m.run(tokens)
    want = logits[3:].argmax(1)  # the program's next tokens, as the check says
    for seed in range(3):  # every bit at every depth within a third of its size, and every next
        check_transformer(noisy(m, 0.03, seed), tokens, 1 / 3)  # token: with ten, not half of them
        assert (noisy(m, 0.1, seed).run(tokens)[1][3:].argmax(1) == want).mean() < 1 / 2

    def off(m):  # the root mean square of what m's streams are off by, where the grid holds
        return np.sqrt(np.mean([(x - y)[3:] ** 2 for x, y in zip(m.run(tokens)[0], streams)]))

    wide = transformer(t, codes, hidden=4096)  # each unit TIMES times: what their weights are
    for seed in range(3):  # off by averages, though not what those of the norms are
        once, often = (off(noisy(x, 0.03, seed, heads=False)) for x in (m, wide))
        assert often < 0.7 * once

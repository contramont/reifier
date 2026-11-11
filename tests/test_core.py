"""The compiler, stage by stage: a trace, a grid, the weights, a transformer."""

import contextlib
import io
import re
from pathlib import Path

import adder
import numpy as np
import pytest

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
from reifier.lib import add, not_, parity, xor
from reifier.show import show
from reifier.trace import GATED, const, mlp, trace
from reifier.transformer import transformer

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


def test_a_transformer_adds():
    symbols = adder.SYMBOLS
    m = transformer(adder.program(), dict(enumerate(adder.CODES.values())))
    for a, b in [(123, 246), (999, 999), (0, 0), (5, 995), (608, 397)]:
        tokens = m.generate([symbols.index(c) for c in f"{a:03}+{b:03}="], 3)
        assert "".join(symbols[i] for i in tokens[-3:]) == f"{(a + b) % 1000:03}"
    sums = np.random.default_rng(0).integers(0, 1000, (6, 2))
    text = "".join(f"{a:03}+{b:03}={(a + b) % 1000:03}" for a, b in sums)
    check_transformer(m, [symbols.index(c) for c in text])
    m.embed[9, 1] += 1e-6  # the code of "9" is a little off: so is the stream, where 9 is
    with pytest.raises(AssertionError, match=r"depth 0, feature 1 \(code\[0\]\) is"):
        check_transformer(m, [symbols.index(c) for c in "123+246=369"])
    m.embed[9, 1:] *= 3  # and now three times too large: it has the largest logit
    with pytest.raises(AssertionError, match="position 7: the next token is 9, not 3"):
        check_transformer(m, [symbols.index(c) for c in "123+246=36"])


def test_a_token_without_a_code_changes_nothing():
    def count(bits):  # the next token's bits: the sum of those of the last two, mod 4
        return add(bits[:2], bits[2:])

    m = transformer(trace(count, 4), {n + 1: [n >> 1, n & 1] for n in range(4)})
    assert m.generate([0, 2, 3], 3) == [0, 2, 3, 4, 2, 1]  # token 0 is all zeros, and stays so
    assert show(m).startswith("2 heads on 4 tokens\nhead 0: the code of the token 0 back -> bits2 ")

"""Stage 3: in every network, every feature holds at every depth what the grid names there."""

import numpy as np
import pytest

from reifier import (
    ONE,
    add,
    and_,
    check_net,
    const,
    emit,
    fold,
    glu,
    held,
    layout,
    mlp,
    not_,
    parity,
    threshold_gate,
    tiles,
    trace,
    xor,
)
from reifier.trace import GATED, cases

GRIDS = {"layout": layout, "held": lambda t: held(layout(t)), "tiles": tiles}
NETWORKS = [  # activation, skip, norm: norms need a gated activation
    (activation, skip, norm)
    for activation in ("step", "relu", "reglu", "swiglu", "geglu")
    for skip in (False, True)
    for norm in (False, True)
    if activation in GATED or not norm
]


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("activation, skip, norm", NETWORKS)
def test_every_network_holds_what_its_grid_says(grid, activation, skip, norm):
    t = trace(add, 4, 4)
    net = emit(t, GRIDS[grid](t), activation, skip, norm)
    assert check_net(net) < 1e-12
    assert all(np.allclose(net(x), t(x), atol=1e-12) for x in cases(8)[::17])


@pytest.mark.parametrize("activation, skip, norm", NETWORKS)
def test_odd_programs_compile(activation, skip, norm):
    def program(x):  # an input and a constant as outputs, a gate that is always on, two outputs
        full = mlp(x, [[1, 1], [1, 1]], [0, -1], "relu", [[1, -2], [0, 1]], "full")
        return [x[1], const("1")[0], x[0], threshold_gate([x[0]], [1], 0), *full]

    t = trace(program, 2)
    assert t.outputs[4:] == ["full:0", "full:1"] and [t(x)[4:] for x in ("11", "01")] == [
        [0, 1],
        [1, 0],
    ]
    if activation != "step":  # full is a relu MLP
        assert check_net(emit(t, layout(t), activation, skip, norm)) < 1e-12
    folded = fold(trace(lambda x: [x[0], and_([x[0], const("0")[0]])], 1))  # and: no hidden units
    assert check_net(emit(folded, layout(folded), activation, skip, norm)) < 1e-12


def test_every_hidden_unit_has_a_name():
    t = trace(xor, 3)
    stack, skip = emit(t, layout(t), "relu"), emit(t, layout(t), "geglu", skip=True, norm=True)
    assert stack.layers[1].hidden == ("ONE", "xor/odd+", "xor/odd-")
    assert stack.layers[0].hidden[:3] == ("ONE", "xor/atleast1+", "xor/atleast1-")
    erases = ("erase xor/atleast3", "erase xor/atleast2", "erase xor/atleast1")
    assert skip.layers[1].hidden == ("ONE", "xor/odd+", "xor/odd-", *erases)
    kept = emit(t, held(layout(t)), "geglu", skip=True, norm=True)  # only what is written over
    assert kept.layers[1].hidden == ("ONE", "xor/odd+", "xor/odd-", "erase xor/atleast3")
    assert emit(t, layout(t), "relu", skip=True).layers[1].hidden[0] == "xor/odd+"  # no unit ONE
    wide = trace(add, 4, 4)
    assert "keep a0" in emit(wide, layout(wide), "step").layers[0].hidden


def test_a_network_prints_each_unit_by_name():
    t = trace(xor, 3)
    text = str(emit(t, layout(t), "step"))
    assert text.startswith("xor: stack, 4 features, ONE = 1\nlayer 1: 4 hidden units of step")
    assert "  xor/atleast2 = step(-2 ONE + xs0 + xs1 + xs2) -> f2" in text
    text = str(emit(t, layout(t), "geglu", skip=True, norm=True))
    assert text.startswith("xor: skip + norm, 4 features, ONE = 4\n")  # 2 sqrt(4)
    assert "layer 1: 10 hidden units of geglu, after a norm of gain 2.179" in text  # sqrt(19 / 4)
    assert "  erase xs0 = gelu(32 ONE) * (-0.03125 xs0) -> f1" in text


def test_check_net_names_what_is_wrong():
    t = trace(add, 2, 2)
    net = emit(t, layout(t), "reglu", skip=True)
    net.layers[1].wo[3, 2] += 0.5
    with pytest.raises(AssertionError, match=r"depth 2, feature 3 \(b0\) is"):
        check_net(net)
    net = emit(t, layout(t), "reglu")
    net.layers[3].wi[0, 1] = 1  # depth 3 holds nothing in feature 1
    with pytest.raises(AssertionError, match="layer 4 reads a feature that depth 3 leaves empty"):
        check_net(net)


def test_norms_need_gated_units_and_bits():
    t = trace(xor, 2)
    with pytest.raises(TypeError, match="gated"):
        emit(t, layout(t), "relu", norm=True)
    with pytest.raises(TypeError, match="parity is a reglu MLP: it has no step form"):
        emit(trace(parity, 2), layout(trace(parity, 2)), "step")
    with pytest.raises(TypeError, match="not of gelu"):
        emit(t, layout(t), "gelu")
    count = trace(lambda x: glu(x, [([0, 0], 1, [1, 1], 0)], "count"), 2)  # x0 + x1: 0, 1 or 2
    with pytest.raises(AssertionError, match="count is not a bit"):
        check_net(emit(count, layout(count), "geglu", skip=True, norm=True))


def chain(gate, bits, depth):  # every layer writes all the bits again, each from two neighbours
    def program(x):
        for _ in range(depth):
            x = [gate([x[i], x[i - 1]]) for i in range(len(x))]
        return x

    return trace(program, bits)


@pytest.mark.parametrize("activation", ["reglu", "swiglu", "geglu"])
@pytest.mark.parametrize("skip, norm", [(False, False), (False, True), (True, False), (True, True)])
def test_deep_networks_stay_exact(activation, skip, norm):
    for t in (chain(parity, 16, 100), chain(lambda xs: not_(and_(xs)), 4, 100)):
        assert check_net(emit(t, held(layout(t)), activation, skip, norm)) < 1e-12


def test_norms_do_not_grow_what_is_off():
    t = chain(lambda xs: not_(xs[0]), 4, 60)  # every bit flips in every layer
    net = emit(t, held(layout(t)), "geglu", skip=True, norm=True)
    values = t.values(cases(4)) | {ONE: np.ones(16)}
    off = 1e-6 * np.random.default_rng(0).standard_normal((16, 5))
    rows = net.run(net.row(values, 0) + off)  # with ONE held as 1, this ends 5.75 off
    assert max(abs(x - net.row(values, d)).max() for d, x in enumerate(rows)) < 1e-5

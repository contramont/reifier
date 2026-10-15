import runpy
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from reifier import (
    add,
    at_least,
    bits,
    circuit,
    const,
    flatten,
    gate,
    lower,
    network,
    place,
    show,
    trace,
    xor,
)

TARGETS = {
    "step mlp": lambda g: network(g, "step"),
    "relu mlp": lambda g: network(g, "relu"),
    "reglu": lambda g: network(g, "reglu"),
    "swiglu": lambda g: network(g, "swiglu"),
    "geglu, rmsnorm": lambda g: network(g, "geglu", norm=True),
}


def table(n):
    return [list(r) for r in product((0, 1), repeat=n)]


def run(net, x):  # a compiled network: BOS = 1 first, in and out; outputs relative to BOS
    y = net.eval([1, *x])
    return np.array(y[1:]) / y[0]


def matches(graph, net, xs):  # silu is relu to within exp(-k BOS), and norms shrink BOS
    return all(np.allclose(run(net, x), graph.eval(x), atol=1e-3) for x in xs)


def test_caching_traces_once_per_structure():
    calls = []

    @circuit
    def maj(xs):
        calls.append(len(xs))
        return gate(xs, [1] * len(xs), len(xs) // 2 + 1)

    g = trace(lambda x: [maj(x[:3]), maj(x[3:]), maj(x[:5])], bits(6))
    assert calls == [3, 5]  # x[:3] and x[3:] share a trace
    assert sorted(len(n.circuit.inputs) for n in g.nodes) == [3, 3, 5]
    assert len(flatten(g).nodes) == 3


def test_signals_have_no_value():
    with pytest.raises(TypeError):
        trace(lambda x: [x[1]] if x[0] else [x[0]], bits(2))


def test_circuits_only_read_their_inputs():
    outside = bits(1)
    with pytest.raises(ValueError):
        trace(circuit(lambda x: gate([x[0], outside[0]], [1, 1], 2)), bits(1))


def test_static_arguments_key_the_cache():
    g3, g4 = at_least.trace(bits(5), 3), at_least.trace(bits(5), 4)
    assert g3 is not g4 and g3 is at_least.trace(bits(5), 3)
    for x in table(5):
        assert g3.eval(x) == [sum(x) >= 3] and g4.eval(x) == [sum(x) >= 4]


def test_add():
    g = add.trace(bits(4), bits(4))
    for x in table(8):
        y = int("".join(str(int(b)) for b in g.eval(x)), 2)
        assert y == (int("".join(map(str, x[:4])), 2) + int("".join(map(str, x[4:])), 2)) % 16


def test_hand_written_xors_match_reference():
    for n in range(1, 8):
        g = xor.trace(bits(n))
        for impl in xor.impls.values():
            h = impl.trace(bits(n))
            assert all(np.isclose(h.eval(x)[0], g.eval(x)[0]) for x in table(n))


@pytest.mark.parametrize("target", TARGETS)
def test_targets_match_the_circuit(target):
    g = add.trace(bits(4), bits(4))
    assert matches(g, TARGETS[target](g), table(8))


def test_architectures():
    g = add.trace(bits(4), bits(4))
    fs = {t: [n.circuit.f for n in TARGETS[t](g).nodes] for t in TARGETS}
    assert fs["step mlp"] == ["step"] * 5 and fs["reglu"] == ["reglu"] * 4
    assert fs["relu mlp"] == ["relu"] * 4  # the hand-written xors take one layer, not two
    assert fs["geglu, rmsnorm"] == ["rmsnorm", "geglu"] * 4


def test_placing_keeps_units():
    g = add.trace(bits(4), bits(4))
    live = [n.circuit for n in flatten(lower(g, "reglu")).nodes if n.inputs]  # not const("0")
    weights = {(c.wi.tobytes(), c.b.tobytes()) for c in live}
    placed = [n.circuit for layer in place(g, "reglu").nodes for n in layer.circuit.nodes]
    units = [c for c in placed if c.label != "copy"]
    assert len(units) == len(live) and all(c.f == "reglu" for c in placed)
    kept = [c for c in units if (c.wi.tobytes(), c.b.tobytes()) in weights]
    assert len(kept) == len(units) - 1  # the xor that reads the carry 0 has it in its bias


@pytest.mark.parametrize("target", TARGETS)
def test_constants_and_passthrough_compile(target):
    g = trace(lambda x: [x[1], const("1")[0], x[0], gate([x[0]], [1], 0)], bits(2))
    assert [g.eval(x) for x in table(2)] == [[x[1], 1, x[0], 1] for x in table(2)]
    assert matches(g, TARGETS[target](g), table(2))


def test_show_lists_each_circuit_once():
    text = show(add.trace(bits(3), bits(3)))
    heads = sorted(line for line in text.splitlines() if line and not line.startswith(" "))
    assert heads == [
        "add/6(%0 %1 %2 %3 %4 %5):",
        "and_/2(%0 %1):",
        "and_/3(%0 %1 %2):",
        "or_/1(%0):",
        "or_/2(%0 %1):",
        "xor/3(%0 %1 %2):",
    ]
    assert text.count("= xor/3(") == 3 and "gate[1 1 1 >= 2]" in text


def test_gemma_ffn_is_three_units():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "gemma_ffn.py"))
    text, err = example["check"]()
    assert err < 1e-9
    assert [line.split(" = ")[1].split("(")[0] for line in text.splitlines()[1:-1]] == [
        "rmsnorm[8]",
        "geglu[32]",
        "id[8]",
    ]

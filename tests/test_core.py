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
    glu_layers,
    inputs,
    mlp,
    show,
    trace,
    xor,
)

TARGETS = {
    "step mlp": lambda g: mlp(g, "step"),
    "relu mlp": lambda g: mlp(g, "relu"),
    "glu": glu_layers,
    "glu, rmsnorm": lambda g: glu_layers(g, norm=True),
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
    assert set(fs["step mlp"]) == {"step", "id"} and fs["step mlp"][-1] == "id"
    assert set(fs["relu mlp"]) == {"relu", "id"} and fs["relu mlp"].count("id") == 1
    assert fs["glu"] == ["swiglu", "id"] * (len(fs["glu"]) // 2)
    assert fs["glu, rmsnorm"] == ["rmsnorm", "swiglu", "id"] * (len(fs["glu"]) // 2)
    # the hand-written xors take one layer where gates take two
    assert len(fs["relu mlp"]) < len(fs["step mlp"])


def test_int_signals_need_a_target_that_has_them():
    g = at_least.trace(bits(5), 3)
    for target in ["relu mlp", "glu", "glu, rmsnorm"]:
        assert matches(g, TARGETS[target](g), table(5))
    with pytest.raises(TypeError):  # popcount is a relu block, not a step block
        mlp(g, "step")
    with pytest.raises(TypeError):
        gate(inputs("real"), [1], 1)


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


def test_gemma_ffn_is_four_blocks():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "gemma_ffn.py"))
    text, err = example["check"]()
    assert err < 1e-9
    assert [line.split(" = ")[1].split("(")[0] for line in text.splitlines()[1:-1]] == [
        "rmsnorm[8]",
        "geglu[64]",
        "id[8]",
        "id[8]",
    ]

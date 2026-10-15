import runpy
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from reifier import (
    GLU,
    RULES,
    add,
    at_least,
    bits,
    circuit,
    const,
    flatten,
    gate,
    inputs,
    layers,
    show,
    to_swiglu,
    trace,
    xor,
)
from reifier.glu import step_to_relu

RULES["relu_auto", GLU] = step_to_relu  # relu without the hand-written circuits


def table(n):
    return np.array(list(product((0, 1), repeat=n)), float)


def ints(x, n):  # rows of msb-first bits -> integers
    return x @ (1 << np.arange(n)[::-1])


def network(graph, x, dialect="relu", compile=to_swiglu):  # run layers that read BOS first
    net = compile(graph, dialect)
    return net.run(np.c_[np.ones(len(x)), x])[:, 1:], net


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
    x = table(5)
    assert np.array_equal(g3.run(x)[:, 0], x.sum(1) >= 3)
    assert np.array_equal(g4.run(x)[:, 0], x.sum(1) >= 4)


def test_add_exhaustive():
    g, x = add.trace(bits(8), bits(8)), table(16)
    want = (ints(x[:, :8], 8) + ints(x[:, 8:], 8)) % 256
    assert np.array_equal(ints(g.run(x), 8), want)


def test_hand_written_xor_matches_reference():
    for n in range(1, 10):
        x = table(n)
        assert np.allclose(xor.trace(bits(n)).run(x), xor.impls["relu"].trace(bits(n)).run(x))


def test_each_stage_is_a_circuit_that_matches():
    g, x = add.trace(bits(8), bits(8)), table(16)
    for dialect, n_layers in [("relu", 4), ("relu_auto", 5)]:  # xor in one layer, or two
        for compile, act, tol in [(layers, "relu", 1e-12), (to_swiglu, "silu", 1e-9)]:
            y, net = network(g, x, dialect, compile)
            assert len(net.nodes) == n_layers and np.abs(y - g.run(x)).max() < tol
            assert all(n.circuit.act == act and not n.circuit.bg.any() for n in net.nodes)


def test_int_signals_compile():
    g, x = at_least.trace(bits(5), 3), table(5)
    y, net = network(g, x)
    assert len(net.nodes) == 2 and np.allclose(y, g.run(x), atol=1e-9)
    with pytest.raises(TypeError):
        gate(inputs("real"), [1], 1)


def test_constants_and_passthrough_compile():
    g = trace(lambda x: [x[1], const("1")[0], x[0], gate([x[0]], [1], 0)], bits(2))
    x = table(2)
    assert np.array_equal(g.run(x), np.c_[x[:, 1], np.ones(4), x[:, 0], np.ones(4)])
    assert np.allclose(network(g, x)[0], g.run(x), atol=1e-9)


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


def test_new_circuit_types_live_outside_the_package():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "gemma_ffn.py"))
    text, err = example["check"]()
    assert err < 1e-9 and "rmsnorm[8]" in text and "gelu[32]" in text

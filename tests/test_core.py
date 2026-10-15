import numpy as np
import pytest

from reifier import (
    GLU,
    Gate,
    add,
    at_least,
    bits,
    circuit,
    const,
    flatten,
    gate,
    inputs,
    run_swiglu,
    show,
    to_swiglu,
    trace,
    verify,
    xor,
)
from reifier.core import DIALECTS, RULES, gate_to_glu

DIALECTS["glu_auto"], RULES["glu_auto", Gate] = (GLU,), gate_to_glu  # no impls


def table(n):
    return np.indices((2,) * n).reshape(n, -1).T


def ints(x, n):  # rows of msb-first bits -> integers
    return x @ (1 << np.arange(n)[::-1])


def test_caching_traces_once_per_structure():
    calls = []

    @circuit
    def maj(xs):
        calls.append(len(xs))
        return gate(xs, [1] * len(xs), len(xs) // 2 + 1)

    def program(x):
        return [maj(x[:3]), maj(x[3:]), maj(x[:5])]

    g = trace(program, bits(6))
    assert calls == [3, 5]  # x[:3] and x[3:] share a trace
    assert sorted(len(n.op.inputs) for n in g.nodes) == [3, 3, 5]
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
    g = add.trace(bits(8), bits(8))
    x = table(16)
    want = (ints(x[:, :8], 8) + ints(x[:, 8:], 8)) % 256
    assert np.array_equal(ints(g.run(x), 8), want)


def test_hand_written_xor_matches_reference():
    for n in range(1, 10):
        assert verify(xor.trace(bits(n)), xor.impls["glu"].trace(bits(n)))


def test_swiglu_matches_graph():
    g = add.trace(bits(8), bits(8))
    x = table(16)
    for dialect, n_layers in [("glu", 4), ("glu_auto", 5)]:  # one layer per xor
        layers = to_swiglu(g, dialect=dialect)
        assert len(layers) == n_layers
        assert np.abs(run_swiglu(layers, x) - g.run(x)).max() < 1e-9


def test_int_signals_compile():
    g = at_least.trace(bits(5), 3)
    x = table(5)
    layers = to_swiglu(g)
    assert len(layers) == 2
    assert np.allclose(run_swiglu(layers, x), g.run(x), atol=1e-9)
    with pytest.raises(TypeError):
        gate(inputs("real"), [1], 1)


def test_constants_and_passthrough_compile():
    g = trace(lambda x: [x[1], const("1")[0], x[0], gate([x[0]], [1], 0)], bits(2))
    x = table(2)
    assert np.allclose(run_swiglu(to_swiglu(g), x), g.run(x), atol=1e-9)
    assert np.array_equal(g.run(x), np.c_[x[:, 1], np.ones(4), x[:, 0], np.ones(4)])


def test_show_lists_each_circuit_once():
    text = show(add.trace(bits(3), bits(3)))
    heads = [line for line in text.splitlines() if not line.startswith(" ")]
    assert sorted(h for h in heads if h) == [
        "add/6(%0 %1 %2 %3 %4 %5):",
        "and_/2(%0 %1):",
        "and_/3(%0 %1 %2):",
        "or_/1(%0):",
        "or_/2(%0 %1):",
        "xor/3(%0 %1 %2):",
    ]
    assert text.count("= xor/3(") == 3

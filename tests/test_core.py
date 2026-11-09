import numpy as np
import pytest
from helpers import assert_matches, truth_table

from reifier import (
    add,
    at_least,
    bits,
    circuit,
    const,
    flatten,
    fold_constants,
    format_graph,
    lower,
    network,
    popcount,
    threshold_gate,
    trace,
    xor,
)


def test_caching_traces_once_per_structure():
    calls = []

    @circuit
    def maj(xs):
        calls.append(len(xs))
        return threshold_gate(xs, [1] * len(xs), len(xs) // 2 + 1)

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
        trace(circuit(lambda x: threshold_gate([x[0], outside[0]], [1, 1], 2)), bits(1))


def test_static_arguments_key_the_cache():
    g3, g4 = at_least.trace(bits(5), 3), at_least.trace(bits(5), 4)
    assert g3 is not g4 and g3 is at_least.trace(bits(5), 3)
    for x in truth_table(5):
        assert g3.eval(x) == [sum(x) >= 3] and g4.eval(x) == [sum(x) >= 4]


def test_add():
    g = add.trace(bits(4), bits(4))
    for x in truth_table(8):
        y = int("".join(str(int(b)) for b in g.eval(x)), 2)
        assert y == (int("".join(map(str, x[:4])), 2) + int("".join(map(str, x[4:])), 2)) % 16


def test_hand_written_xors_match_reference():
    for n in range(1, 8):
        g = xor.trace(bits(n))
        for impl in xor.impls.values():
            h = impl.trace(bits(n))
            assert all(np.isclose(h.eval(x)[0], g.eval(x)[0]) for x in truth_table(n))


def test_programs_need_no_decorator():
    def program(a, b):  # not a circuit itself, but the circuits it calls are
        return add(a, b)

    g = trace(program, bits(3), bits(3))
    assert g.name == "program"
    assert [n.circuit.name for n in g.nodes] == ["add"]
    assert g.eval("011 001") == g.eval([0, 1, 1, 0, 0, 1]) == [1, 0, 0]  # bits, spaces aside
    assert_matches(g, network(g, "geglu", pre_norm=True), truth_table(6))


def test_format_graph_lists_each_circuit_once():
    text = format_graph(add.trace(bits(3), bits(3)))
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


def test_folding_is_exact_and_drops_what_constants_decide():
    def program(x):
        return [
            threshold_gate([x[0], const("0")[0]], [1, 1], 1),
            xor([x[0], const("1")[0], x[1]]),
            x[1],
        ]

    g = trace(program, bits(2))
    for activation in ("step", "relu", "reglu"):
        h = fold_constants(g, activation)
        assert [h.eval(x) for x in truth_table(2)] == [g.eval(x) for x in truth_table(2)]
        assert len(flatten(h).nodes) < len(flatten(lower(g, activation)).nodes)
    # [x0 + 0 >= 1] is x0 itself, and the reglu xor reads two bits, not three
    assert len(flatten(fold_constants(g, "reglu")).nodes) == 1
    assert_matches(g, network(fold_constants(g, "reglu"), "geglu", pre_norm=True), truth_table(2))
    assert len(network(fold_constants(g, "reglu"), "reglu").nodes) < len(network(g, "reglu").nodes)


def test_leaves_evaluate_a_column_per_position():
    c = xor.impls["relu"].trace(bits(3)).nodes[0].circuit  # one relu MLP
    assert c.eval(np.array([[1, 1], [0, 1], [0, 1]])).tolist() == [[1, 1]]
    assert c.eval([1, 0, 0]).tolist() == [[1]]  # one position


def test_folding_knows_an_int_can_be_more_than_one():
    def program(x):  # at most one 1: a gate with a weight of -1 on a count
        return [threshold_gate([popcount(x)], [-1], -1)]

    g = trace(program, bits(3))
    for activation in ("step", "relu", "reglu"):
        h = fold_constants(g, activation)
        np.testing.assert_array_equal(
            h.eval_sequence(truth_table(3)), g.eval_sequence(truth_table(3))
        )

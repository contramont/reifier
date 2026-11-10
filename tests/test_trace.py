"""Stage 1: tracing gives a flat list of MLPs, named by scope, that computes what the function does."""

from dataclasses import replace

import numpy as np
import pytest

from reifier import add, and_, check_trace, circuit, const, mlp, parity, threshold_gate, trace, xor
from reifier.trace import cases, convert


def sum_bits(x):  # the bits of a + b for x = the bits of a, then of b, most significant first
    n, number = len(x) // 2, lambda bits: int("".join(map(str, bits)), 2)
    return [int(c) for c in format((number(x[:n]) + number(x[n:])) % 2**n, f"0{n}b")]


def test_names_are_scopes():
    t = trace(add, 2, 2)
    assert t.inputs == ["a0", "a1", "b0", "b1"] and t.outputs == ["add/xor.1/odd", "add/xor/odd"]
    names = [
        m.name for m in t.mlps
    ]  # add/or and add/or.1, which no carry of 2 bits reads, are gone
    assert len(set(names)) == len(names) == 11 and "add/or.2" in names and "add/or" not in names
    assert {"add/and", "add/const0", "add/xor/atleast2", "add/xor.1/odd"} <= set(names)
    assert all(m.hidden == (m.name,) for m in t.mlps)  # a gate's hidden unit has the gate's name
    assert trace(parity, 3).mlps[0].hidden == ("parity#0", "parity#1", "parity#2")
    geglu = convert(t.mlps[0], "geglu")  # a step unit u is the relu units u+ and u-
    assert geglu.hidden == ("add/and+", "add/and-") and geglu.activation == "geglu"
    two = trace(lambda x: mlp(x, [[1, 1]], [-1], "relu", [[1], [2]], "m"), 2)
    assert two.outputs == ["m:0", "m:1"] and two("11") == [1, 2]


def test_names_must_differ():
    with pytest.raises(ValueError, match=r"signals share a name: \['g.1'\]"):
        trace(lambda x: [threshold_gate(x, [1, 1], 1, name) for name in ("g", "g", "g.1")], 2)
    with pytest.raises(ValueError, match=r"signals share a name: \['x10'\]"):
        trace(lambda x, x1: and_(x + x1), 11, 1)


def test_the_trace_computes_the_function():
    assert check_trace(trace(add, 4, 4), sum_bits) == 256
    assert check_trace(trace(lambda a, b: add(a, b, parity), 4, 4), sum_bits) == 256
    for n in range(1, 8):  # the GLU parity is the xor of gates
        assert check_trace(trace(parity, n), trace(xor, n)) == 2**n


def test_check_trace_names_what_is_wrong():
    t = trace(add, 2, 2)
    with pytest.raises(AssertionError, match="add/xor.1/odd is 1 on 1000, not 0"):
        check_trace(t, lambda x: [0, 0])
    swapped = trace(lambda a, b: add(a, b)[::-1], 2, 2)  # the same signals, in another order
    with pytest.raises(AssertionError, match="add/xor/odd is"):
        check_trace(swapped, t)
    t.mlps[0] = replace(t.mlps[0], b=t.mlps[0].b + 1)  # the first gate, add/and, is now an or
    with pytest.raises(AssertionError, match="add/and is 1 on 0100, not 0"):  # not what reads it
        check_trace(t, trace(add, 2, 2))
    with pytest.raises(AssertionError, match="is nan"):
        check_trace(trace(lambda x: mlp(x, [[np.nan]], [0], "relu", [[1]]), 1), lambda x: [0])


def test_signals_have_no_value():
    with pytest.raises(TypeError):
        trace(lambda x: [x[1]] if x[0] else [x[0]], 2)


def test_a_trace_is_of_one_function_run_once():
    kept = []
    trace(lambda x: kept.append(and_(x)) or x, 2)
    with pytest.raises(ValueError, match="gate reads a signal of another trace"):
        trace(lambda y: threshold_gate([y[0], kept[0]], [1, 1], 2), 1)
    with pytest.raises(ValueError, match="gives a signal of another trace"):
        trace(lambda y: kept, 1)
    t = trace(lambda x: x.reverse() or [x[0]], 3)  # the function may change its list
    assert t.inputs == ["x0", "x1", "x2"] and t("001") == [1]


def test_programs_need_no_decorator():
    def program(a, b):  # not a circuit itself, but the circuits it calls are scopes
        return add(a, b)

    t = trace(program, 3, 3)
    assert t.name == "program" and all(m.name.startswith("add/") for m in t.mlps)

    @circuit
    def majority(xs):
        return threshold_gate(xs, [1] * len(xs), len(xs) // 2 + 1)

    t = trace(lambda x: [majority(x[:3]), majority(x[3:]), majority(x[:5])], 6)
    assert [m.name for m in t.mlps] == ["majority/gate", "majority.1/gate", "majority.2/gate"]


def test_constants_and_inputs_can_be_outputs():
    t = trace(lambda x: [x[1], const("1")[0], x[0], threshold_gate([x[0]], [1], 0)], 2)
    assert t.outputs == ["x1", "const1", "x0", "gate"]
    assert [t(x) for x in cases(2)] == [[x[1], 1, x[0], 1] for x in cases(2)]


def test_a_trace_prints_each_mlp_by_name():
    text = str(trace(xor, 3))
    assert text.splitlines()[0] == "xor(xs0 xs1 xs2): 4 MLPs"
    assert "  xor/atleast2 = [xs0 + xs1 + xs2 >= 2]" in text
    assert "  xor/odd = [xor/atleast1 - xor/atleast2 + xor/atleast3 >= 1]" in text
    text = str(trace(parity, 2))
    assert "parity#0 = relu(xs0 + xs1 - 0.5) * (-4 xs0 - 4 xs1 + 6) -> parity" in text
    twice = trace(lambda x: mlp(x, [[1, 1]], [-2], "step", [[2]], "twice"), 2)
    assert "twice = step(x0 + x1 - 2) -> 2 twice" in str(twice)  # not a gate: it gives 0 or 2

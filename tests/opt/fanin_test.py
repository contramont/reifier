"""Bounded fan-in (reifier.opt.fanin): the bounded ops keep the function and bound the
fan-in, the prefix adder adds, and Fanin.trace builds the gate graph that calling the
bounded ops builds, also on deep graphs, around exceptions, with other threads and nested
traces, and leaves no sys.monitoring state behind."""

import random
import sys
import threading
from functools import partial

import pytest

from reifier.compile.tree import TreeCompiler
from reifier.neurons.core import Bit, GluNeuron, Unit, const, gate, glu
from reifier.neurons.operations import add, and_, glu_xor, or_, parity, xor
from reifier.opt.fanin import Fanin, _tree, prefix_add
from reifier.opt.graph import GraphCompiler
from reifier.utils.format import Bits
from tests.opt.utils import values

OPS = (xor, and_, or_, parity, add)  # the core's ops that Fanin bounds


def _signals(outputs: list[Bit]) -> list[Bit]:
    """every Bit that the outputs depend on"""
    seen: set[int] = set()
    stack, out = list(outputs), []
    while stack:
        s = stack.pop()
        if id(s) not in seen:
            seen.add(id(s))
            out.append(s)
            stack.extend(s.source.incoming)
    return out


def test_bounded_ops():
    rng = random.Random(0)

    def fn(x: list[Bit]) -> list[Bit]:
        return [xor(x), and_(x), or_(x), parity(x) if len(x) > 1 else xor(x)]

    for n in [1, 2, 3, 5, 11, 17, 64]:
        for _ in range(20):
            x = const([rng.randint(0, 1) for _ in range(n)])
            _, out = Fanin(xor=3, and_or=4).trace(fn, x)
            assert values(out) == values(fn(x))
            assert all(len(b.source.incoming) <= 4 for b in _signals(out))


def test_prefix_adder():
    rng = random.Random(1)
    for n in [1, 2, 3, 5, 8, 13, 32]:
        for _ in range(30):
            x, y = rng.getrandbits(n), rng.getrandbits(n)
            s = prefix_add(const(format(x, f"0{n}b")), const(format(y, f"0{n}b")))
            assert int("".join(map(str, values(s))), 2) == (x + y) % 2**n


def _circuit(xor_, and__, or__, parity_, add_):
    """a circuit of the given ops: results shared and read by other ops, ops inside a
    helper, parity, add, a gate() and glus that read results, and an input as output"""

    def helper(y: list[Bit]) -> Bit:
        return and__([xor_(y[:5])] + y[5:])

    def fn(x: list[Bit]) -> list[Bit]:
        s = add_(x[:6], x[6:12])
        p = xor_(x[:9])
        q = and__([p] + x[3:10])
        r = or__(x[2:14] + [q])
        w = gate([q] + x[:6], [1] * 7, 7)  # a wide AND made directly: never split
        u = glu_xor([p, q, x[0]])
        c = glu([p, q, r], [Unit((0, 0, 0), 1, (1, 1, 1), 0)], numeric=True)  # a count
        h = helper([p, r, s[0]] + x[8:13])
        return s + [p, q, r, parity_(x[1:10]), w, u, gate([c, x[3]], [1, 1], 2), h,
                    xor_([p, q, r, s[1], x[0]]), x[5]]

    return fn


def _bounded(f: Fanin) -> tuple:
    """xor, and_, or_, parity and add as the bounded ops of f, to call directly"""
    def tree(op, k):
        return partial(_tree, op, k) if k else op

    return (tree(xor, f.xor), tree(and_, f.and_or), tree(or_, f.and_or),
            tree(xor, f.xor) if f.xor else parity,
            prefix_add if f.adder == "prefix" else add)


def _structure(inputs: list[Bit], outputs: list[Bit]) -> list:
    """the Signal graph of the outputs, numbered in post-order from them (inputs first):
    every gate's parents, weights and bias and every glu's parents, units and type"""
    num: dict[int, int] = {}
    for b in inputs:
        num.setdefault(id(b), len(num))
    nodes: list = []
    stack = [(b, False) for b in reversed(outputs)]
    while stack:
        s, ready = stack.pop()
        if id(s) in num:
            continue
        src = s.source
        if not ready:
            stack.append((s, True))
            stack.extend((p, False) for p in reversed(src.incoming) if id(p) not in num)
            continue
        ps = [num[id(p)] for p in src.incoming]
        is_glu = isinstance(src, GluNeuron)
        nodes.append((ps, src.units, type(s.activation)) if is_glu else
                     (ps, src.weights, src.bias))
        num[id(s)] = len(num)
    return nodes + [[num[id(b)] for b in outputs]]


@pytest.mark.parametrize("fanin", [Fanin(4, 2, "prefix"), Fanin(4, 0, "main"),
                                   Fanin(32, 0, "prefix")])  # the recipes' bounds
def test_substitution_matches_bounded_ops(fanin):
    """Fanin.trace of a circuit of the core's ops builds the Signal graph (gates,
    weights, parent order and sharing), and so the leveled graph, of the same circuit
    written with the bounded ops"""
    traced = fanin.trace(_circuit(*OPS), Bits("0" * 14).bitlist)
    x = Bits("0" * 14).bitlist
    called = (x, _circuit(*_bounded(fanin))(x))
    assert _structure(*traced) == _structure(*called)
    assert GraphCompiler().compile(*traced) == GraphCompiler().compile(*called)


def test_deep_chain():
    """the rebuild walks iteratively: a chain of 3000 xors, each split by the bound"""
    def chain(x: list[Bit]) -> list[Bit]:
        y = x[0]
        for i in range(3000):
            y = xor([y, x[i % 10], x[(i + 3) % 10], x[(i + 7) % 10], x[(i + 1) % 10]])
        return [y, and_([y] + x[:6])]

    x = const([random.Random(3).randint(0, 1) for _ in range(10)])
    _, out = Fanin(xor=4, and_or=2).trace(chain, x)
    assert values(out) == values(chain(x))
    assert all(len(b.source.incoming) <= 4 for b in _signals(out))


def test_exception_inside_an_op():
    """an op call that raises (PY_UNWIND, no PY_RETURN) records nothing, and the calls
    after it are bounded"""
    def fn(x: list[Bit]) -> list[Bit]:
        try:
            xor(x[:5] + ["not a Bit"])  # raises inside a call that the bound changes
        except AttributeError:
            pass
        return [xor(x), and_(x)]

    x = const([random.Random(4).randint(0, 1) for _ in range(9)])
    _, out = Fanin(xor=4, and_or=2).trace(fn, x)
    assert values(out) == values([xor(x), and_(x)])
    assert all(len(b.source.incoming) <= 4 for b in _signals(out))


def test_threads_and_nested_traces():
    """only the trace's own calls are bounded: another thread's calls stay as they are, a
    trace inside fn bounds its function with its own bounds, and the core's Tracer (which
    takes sys.monitoring's debugger id) runs inside fn"""
    def fn(x: list[Bit]) -> list[Bit]:
        other: list[Bit] = []
        thread = threading.Thread(target=lambda: other.append(xor(x[:9])))
        thread.start()
        thread.join()
        _, inner = Fanin(xor=4).trace(lambda y: [xor(y)], x[:8])
        TreeCompiler().run(lambda y: [xor(y)], y=x[:3])
        return other + inner + [xor(x[:9])]

    _, out = Fanin(xor=2).trace(fn, Bits("0" * 9).bitlist)
    widths = [max(len(s.source.incoming) for s in _signals([b])) for b in out]
    assert widths == [9, 4, 2]


def test_monitoring_released():
    """a trace leaves no sys.monitoring tool or events behind (Python 3.12 keeps a freed
    tool's events), also when fn raises, and the core's Tracer works as before"""
    mon = sys.monitoring
    fn = lambda x: [xor(x), and_(x)]  # noqa: E731
    x = Bits("0" * 6).bitlist
    tree = TreeCompiler().run(fn, x=x)
    free = [i for i in range(6) if mon.get_tool(i) is None]
    Fanin(xor=4).trace(fn, x=x)
    with pytest.raises(ZeroDivisionError):
        Fanin(xor=4).trace(lambda x: fn(x) + [1 / 0], x=x)
    for i in free:
        assert mon.get_tool(i) is None and mon.get_events(i) == 0
        assert all(mon.get_local_events(i, op.__code__) == 0 for op in OPS)
    assert TreeCompiler().run(fn, x=x).levels == tree.levels

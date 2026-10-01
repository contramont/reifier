"""Bounded fan-in for the compile levels, applied to a traced function.

With a bound k, xor (and parity, as xor) and and_ / or_ of more than k inputs become
balanced trees of gates of at most k inputs, and the adder "prefix" makes add a
Kogge-Stone prefix adder (prefix_add). The function is unchanged on 0/1 inputs (not on
numeric glu counts); the compiled rows get small sums and weights, which keeps them
exact or within margin in 16-bit floats and under noise, at the cost of depth.

The core's ops are not changed. Fanin.trace runs fn as it is while sys.monitoring
records the calls of the core's xor, and_, or_, parity and add, then rebuilds fn's
outputs: the result of each outermost call that the bounds change is built again by its
bounded op, on the rebuilt arguments, and every other gate that reads a rebuilt Bit is
copied onto the new parents: the gate graph that tracing with bounded ops builds. Calls
outside fn's trace (before it, in other threads or in a nested trace) stay as they are.
"""

import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from math import ceil
from types import CodeType, FrameType
from typing import Any

from reifier.compile.monitor import find
from reifier.neurons.core import Bit, GluNeuron, gate, glu
from reifier.neurons.operations import add, and_, or_, parity, xor

mon = sys.monitoring
_OPS = {f.__code__: f for f in (xor, and_, or_, parity, add)}
_local = threading.local()  # .recorder: the innermost trace running in this thread

# (bounded op, arguments, result): an outermost call that the bounds change
_Call = tuple[Callable[..., Any], list[list[Bit]], list[Bit]]


@dataclass(frozen=True)
class Fanin:
    """Fan-in bounds: xor and parity, and and_ / or_ take at most this many inputs (a
    bound k >= 2, or 0 for none); adder "prefix" builds add as prefix_add ("main": the
    core's add). Any bound on xor builds parity as xor"""

    xor: int = 0
    and_or: int = 0
    adder: str = "main"

    def __post_init__(self) -> None:
        if any(k != 0 and k < 2 for k in (self.xor, self.and_or)):
            raise ValueError("fan-in bounds must be at least 2 (0: no bound)")
        if self.adder not in ("main", "prefix"):
            raise ValueError(f"unknown adder {self.adder!r}")

    def trace(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> tuple[list[Bit], list[Bit]]:
        """fn's input and output Bits as GraphCompiler.run finds them, the outputs with
        bounded fan-in: arguments for GraphCompiler.compile"""
        recorder = _Recorder(self)
        outer = getattr(_local, "recorder", None)
        _local.recorder = recorder  # until the rebuild is done: no other trace records
        try:
            with recorder.recording():  # find too: lazy outputs can call ops
                out = fn(*args, **kwargs)
                inputs = [b for b, _ in find(args + tuple(kwargs.values()), Bit)]
                outputs = [b for b, _ in find(out, Bit)]
            return inputs, _rebuild(recorder.calls, inputs, outputs)
        finally:
            _local.recorder = outer

    def _bounded(
        self, op: Callable[..., Any], args: list[list[Bit]]
    ) -> Callable[..., Any] | None:
        """the bounded op for a call of the core's op on args, or None if the bounds
        leave the call as it is"""
        if op is add:
            return prefix_add if self.adder == "prefix" else None
        if op is parity:
            return partial(_tree, xor, self.xor) if self.xor else None
        k = self.xor if op is xor else self.and_or
        return partial(_tree, op, k) if k and len(args[0]) > k else None


def _tree(op: Callable[[list[Bit]], Bit], k: int, x: list[Bit]) -> Bit:
    """op (xor, and_ or or_) of x as a balanced tree of op on at most k inputs"""
    if len(x) > k:
        return _tree(op, k, [_tree(op, k, c) for c in _chunks(x, k)])
    return op(x)


def _chunks(x: list[Bit], k: int) -> list[list[Bit]]:
    """ceil(n/k) nearly equal consecutive chunks of at most k items"""
    m = ceil(len(x) / k)
    sizes = [len(x) // m + (i < len(x) % m) for i in range(m)]
    starts = [sum(sizes[:i]) for i in range(m)]
    return [x[a : a + n] for a, n in zip(starts, sizes)]


def prefix_add(a: list[Bit], b: list[Bit]) -> list[Bit]:
    """a + b (mod 2^n) as a Kogge-Stone prefix adder whose gates have at most 4 inputs
    and weights up to 3, in ceil(log2(n)) + 2 layers of gates (add takes 5, with AND/OR
    carries of up to n inputs). Bit i generates g = a and b and propagates p = a or b;
    level d merges (G, P) at i with G at i - d by the threshold gate 2G + P + G' >= 2.
    The last level folds the sum bit x = a xor b into its gates as x or c and x and c,
    which one more layer subtracts."""
    a, b = list(reversed(a)), list(reversed(b))  # least significant bit first
    n = len(a)
    if n == 1:
        return [xor([a[0], b[0]])]
    g = [and_([a[i], b[i]]) for i in range(n)]
    p = [or_([a[i], b[i]]) for i in range(n)]
    x = [gate([p[i], g[i]], [1, -1], 1) for i in range(n)]  # a xor b
    G, P = g, p  # prefix generate and propagate over windows of d bits ending at i
    d = 1
    while 2 * d < n:  # the carry into bit i is the final G at i - 1, i <= n - 1
        G, P = (
            [G[i] if i < d else gate([G[i], P[i], G[i - d]], [2, 1, 1], 2)
             for i in range(n - 1)],
            [P[i] if i < 2 * d else and_([P[i], P[i - d]]) for i in range(n - 1)],
        )
        d *= 2
    # last level: c_i = G at i - 1 merged with G at i - 1 - d, folded with x_i
    lo_, hi_ = [x[0]], [x[0]]  # x or c, x and c; bit 0 has no carry
    for i in range(1, n):
        j = i - 1
        if j < d:  # G at j is final
            ins, w = [x[i], G[j]], [1, 1]
        else:
            ins, w = [x[i], G[j], P[j], G[j - d]], [2, 2, 1, 1]
        lo_.append(gate(ins, w, w[1]))  # x or c: c is 2G + P + G' >= 2 (or G >= 1)
        hi_.append(gate(ins, [w[0] + 1] + w[1:], w[0] + 1 + w[1]))  # x and c
    s = [x[0]] + [gate([lo_[i], hi_[i]], [1, -1], 1) for i in range(1, n)]
    return list(reversed(s))


class _Recorder:
    """Records the outermost calls of the core's ops that a Fanin changes, while its
    trace is the innermost one running in this thread (not inside a nested trace)"""

    def __init__(self, fanin: Fanin) -> None:
        self.fanin = fanin
        self.calls: list[_Call] = []
        self.open: tuple[FrameType, Callable[..., Any], list[list[Bit]]] | None = None

    @contextmanager
    def recording(self):
        # 3 and 4 have no assigned role; 5, 2, 1 are the optimizer's, profiler's and
        # coverage's; 0 (debugger) is the core's Tracer's
        for tool in (3, 4, 5, 2, 1):
            try:
                mon.use_tool_id(tool, "reifier fanin")
                break
            except ValueError:  # in use
                continue
        else:
            raise RuntimeError("no free sys.monitoring tool id")
        E = mon.events
        handlers = {E.PY_START: self.start, E.PY_RETURN: self.end,
                    E.PY_UNWIND: self.end}
        for event, handler in handlers.items():
            mon.register_callback(tool, event, handler)
        for code in _OPS:
            mon.set_local_events(tool, code, E.PY_START | E.PY_RETURN)
        mon.set_events(tool, E.PY_UNWIND)  # not a local event (Python 3.12)
        try:
            yield
        finally:
            for code in _OPS:  # free_tool_id keeps events and callbacks (Python 3.12)
                mon.set_local_events(tool, code, 0)
            mon.set_events(tool, 0)
            for event in handlers:
                mon.register_callback(tool, event, None)
            mon.free_tool_id(tool)

    def start(self, code: CodeType, offset: int) -> None:
        if self.open is not None or code not in _OPS:
            return  # inside a recorded call, or a stale event of an earlier tool user
        if getattr(_local, "recorder", None) is not self:
            return  # another thread, or a nested trace
        frame = sys._getframe(1)
        names = code.co_varnames[: code.co_argcount]
        try:  # copies: the caller may change its lists later
            args = [list(frame.f_locals[v]) for v in names]
        except TypeError:  # not lists: the op raises its own error
            return
        op = self.fanin._bounded(_OPS[code], args)
        if op is not None:
            self.open = (frame, op, args)

    def end(self, code: CodeType, offset: int, value: Any) -> None:
        """PY_RETURN with the result, or PY_UNWIND with the exception (no result)"""
        call = self.open  # read once: other threads' events come here too
        if call is None or call[0] is not sys._getframe(1):
            return
        self.open = None
        if not isinstance(value, BaseException):
            result = [value] if isinstance(value, Bit) else list(value)
            self.calls.append((call[1], call[2], result))


def _rebuild(calls: list[_Call], inputs: list[Bit], outputs: list[Bit]) -> list[Bit]:
    """outputs with each call's result built by its bounded op on the rebuilt arguments,
    and the gates that read rebuilt Bits copied onto them (in one walk from the outputs,
    iterative: graphs are thousands of levels deep)"""
    if not calls:
        return outputs
    made = {id(b): i for i, (_, _, result) in enumerate(calls) for b in result}
    new = {id(b): b for b in inputs}  # id of a Bit -> the Bit that replaces it
    stack = [(b, False) for b in outputs]
    while stack:
        s, ready = stack.pop()
        if id(s) in new:
            continue
        i = made.get(id(s))
        if i is None:
            parents = s.source.incoming
        else:
            op, args, result = calls[i]
            parents = [b for arg in args for b in arg]
        if not ready:
            stack.append((s, True))
            stack.extend((p, False) for p in parents if id(p) not in new)
        elif i is not None:
            got = op(*[[new[id(b)] for b in arg] for arg in args])
            got = [got] if isinstance(got, Bit) else got
            new.update(zip(map(id, result), got, strict=True))
        elif all(new[id(p)] is p for p in parents):
            new[id(s)] = s
        else:
            new[id(s)] = _copy(s, [new[id(p)] for p in parents])
    return [new[id(b)] for b in outputs]


def _copy(s: Bit, parents: list[Bit]) -> Bit:
    """s's gate or glu on other parents"""
    src = s.source
    if isinstance(src, GluNeuron):
        return glu(parents, list(src.units), numeric=not isinstance(s.activation, bool))
    return gate(parents, list(src.weights), -src.bias)

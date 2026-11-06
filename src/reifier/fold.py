"""Folding: a pass that works out, once, what constants decide.

A unit's constant inputs go into its bias, and a unit left with no inputs, no hidden units,
or only gates that are always on, is a constant itself. A hidden unit whose gate is never on
for bits, or whose value is 0, gives nothing and goes. A unit that reads one bit x gives
a x + b on bits, which goes into the units that read it. A circuit called with a constant is
inlined and folded there; any other is folded once, as a block. All of it is exact on bits,
and dropping hidden units is done only where they give exactly 0, in step, relu and reglu
units: fold(g, f) folds g in the dialect f, so fold in reglu before casting to swiglu or geglu.
"""

from dataclasses import replace
from functools import cache
from typing import NamedTuple

import numpy as np

from .core import Graph, Signal, emit, inputs, interpret, lower
from .units import unskip

EXACT = {"step", "relu", "reglu"}  # a hidden unit that is never on gives exactly 0


class Known(NamedTuple):
    """A value folded away, a x + b for the bit x or b if x is None, and what makes it"""

    x: Signal | None
    a: float
    b: float
    made: tuple = ()  # (unit, inputs, output), should a graph give it


def fold(g: Graph, f: str) -> Graph:
    """g in the dialect f, as lower gives it, with what its constants decide worked out"""
    return folded(lower(g, f))


@cache
def folded(g: Graph) -> Graph:
    ins = inputs(*(s.kind for s in g.inputs))
    outs = [given(v) for v in interpret(g, ins, apply)]
    return replace(g, inputs=ins, outputs=outs, source=None)  # not what source traced


def given(v) -> Signal:
    """A signal for a value a graph gives"""
    if not isinstance(v, Known):
        return v
    if v.x is not None and (v.a, v.b) == (1, 0):
        return v.x
    u, xs, i = v.made
    return emit(u, xs)[i]


def apply(c, xs):
    if isinstance(c, Graph):
        if any(isinstance(x, Known) for x in xs):
            return interpret(c, xs, apply)  # inlined, as its constants make it
        return emit(folded(c), xs)
    kept = None if c.skip is None else xs[c.skip]  # the input c adds: kept if a bit still read
    if isinstance(kept, Known):
        if kept.x is not None and (kept.a, kept.b) == (1, 0):
            kept = kept.x
        else:
            c, kept = unskip(c), None
    cols, b = {}, c.b.copy()  # the weights on each signal still read, and the bias
    for w, x in zip(c.wi.T, xs):
        x = x if isinstance(x, Known) else Known(x, 1.0, 0.0)
        b += w * x.b
        if x.x is not None:
            cols[x.x] = cols.get(x.x, 0) + w * x.a
    live = list(cols)
    wi = np.array([cols[x] for x in live]).reshape(len(live), len(b)).T
    u = replace(c, wi=wi, b=b, skip=None if kept is None else live.index(kept))
    if c.f in EXACT and all(x.kind == "bit" for x in live):  # drop hidden units never on
        n = u.hidden
        top = u.b[:n] + np.maximum(u.wi[:n], 0).sum(1)  # each gate's largest value on bits
        on = top >= 0 if c.f == "step" else top > 0
        if c.f == "reglu":  # a value of 0 gives 0 too
            on &= u.wi[n:].any(1) | (u.b[n:] != 0)
        rows = np.r_[on, on] if c.f == "reglu" else on
        u = replace(u, wi=u.wi[rows], b=u.b[rows], wo=u.wo[:, on])
    always = c.f == "step" and (u.b + np.minimum(u.wi, 0).sum(1) >= 0).all()  # each gate on
    if kept is not None and not len(u.b):  # the input it adds, and nothing else
        return [Known(kept, 1.0, 0.0)]
    if not live or not len(u.b) or always:  # a constant
        ys, u = u.eval([0] * len(live)), replace(u, wi=u.wi[:, :0])
        return [Known(None, 0.0, y, (u, [], i)) for i, y in enumerate(ys)]
    if len(live) == 1 and live[0].kind == "bit":  # a x + b on the bit it reads
        y0, y1 = u.eval([0]), u.eval([1])
        return [Known(live[0], q - p, p, (u, live, i)) for i, (p, q) in enumerate(zip(y0, y1))]
    return emit(u, live)

"""Constant folding: a pass that works out, once, what constants decide.

An MLP's constant inputs go into its bias, and an MLP left with no inputs, no hidden units,
or only threshold gates that are always on, is a constant itself. A hidden unit whose gate is
never on for bits, or whose value is 0, gives nothing and goes. An MLP that reads one bit x
gives a x + b on bits, which goes into the MLPs that read it. A circuit called with a constant
is inlined and folded there; any other is folded once, as a circuit of its own. All of it is
exact on bits, and dropping hidden units is done only where they give exactly 0, in step, relu
and reglu MLPs: fold_constants(g, activation) folds g in the dialect of activation, so fold in
reglu before converting to swiglu or geglu.
"""

from dataclasses import replace
from functools import cache
from typing import NamedTuple

import numpy as np

from .core import Graph, Signal, apply, interpret, lower, placeholders

ZERO_WHEN_OFF = {"step", "relu", "reglu"}  # a hidden unit that is never on gives exactly 0


class Affine(NamedTuple):
    """A value folded away, scale * bit + offset for the input bit, or offset if bit is None,
    and the origin that makes it"""

    bit: Signal | None
    scale: float
    offset: float
    origin: tuple = ()  # (MLP, inputs, output), should a graph give it


def fold_constants(g: Graph, activation: str) -> Graph:
    """g in the dialect of activation, as lower gives it, with what its constants decide
    worked out"""
    return _fold_constants(lower(g, activation))


@cache
def _fold_constants(g: Graph) -> Graph:
    ins = placeholders(*(s.kind for s in g.inputs))
    outs = [materialize(v) for v in interpret(g, ins, fold_node)]
    return replace(g, inputs=ins, outputs=outs, traced_from=None)  # not what it was traced from


def materialize(v) -> Signal:
    """A signal for a value a graph gives"""
    if not isinstance(v, Affine):
        return v
    if v.bit is not None and (v.scale, v.offset) == (1, 0):
        return v.bit
    c, xs, i = v.origin
    return apply(c, xs)[i]


def fold_node(c, xs):
    if isinstance(c, Graph):
        if any(isinstance(x, Affine) for x in xs):
            return interpret(c, xs, fold_node)  # inlined, as its constants make it
        return apply(_fold_constants(c), xs)
    cols, b = {}, c.b.copy()  # the weights on each signal still read, and the bias
    for w, x in zip(c.wi.T, xs):
        x = x if isinstance(x, Affine) else Affine(x, 1.0, 0.0)
        b += w * x.offset
        if x.bit is not None:
            cols[x.bit] = cols.get(x.bit, 0) + w * x.scale
    live = list(cols)
    u = replace(c, wi=np.array([cols[x] for x in live]).reshape(len(live), len(b)).T, b=b)
    # drop hidden units never on
    if c.activation in ZERO_WHEN_OFF and all(x.kind == "bit" for x in live):
        n = u.n_hidden
        top = u.b[:n] + np.maximum(u.wi[:n], 0).sum(1)  # each gate's largest value on bits
        on = top >= 0 if c.activation == "step" else top > 0
        if c.activation == "reglu":  # a value of 0 gives 0 too
            on &= u.wi[n:].any(1) | (u.b[n:] != 0)
        rows = np.r_[on, on] if c.activation == "reglu" else on
        u = replace(u, wi=u.wi[rows], b=u.b[rows], wo=u.wo[:, on])
    # every threshold gate always on
    always = c.activation == "step" and (u.b + np.minimum(u.wi, 0).sum(1) >= 0).all()
    if not live or not len(u.b) or always:  # a constant
        ys, u = u.eval([0] * len(live)), replace(u, wi=u.wi[:, :0])
        return [Affine(None, 0.0, y, (u, [], i)) for i, y in enumerate(ys)]
    if len(live) == 1 and live[0].kind == "bit":  # scale * x + offset on the bit it reads
        y0, y1 = u.eval([0]), u.eval([1])
        return [Affine(live[0], q - p, p, (u, live, i)) for i, (p, q) in enumerate(zip(y0, y1))]
    return apply(u, live)

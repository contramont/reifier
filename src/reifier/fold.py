"""Constant folding: a trace with what its constants decide worked out.

An MLP's constant inputs go into its bias, and an MLP left with no inputs, no hidden units, or
only threshold gates that are always on, is a constant itself. A hidden unit whose gate is never
on, or whose value is 0, gives nothing and goes. An MLP left with one input x gives
scale * x + offset, which goes into the MLPs that read it. All of it is exact where every signal
is a bit, in step, relu and reglu MLPs, and what stays keeps its name: check_trace(fold(t), t)
compares the outputs, and every signal that is left.
"""

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from .trace import MLP, Trace, needed


class Affine(NamedTuple):
    """A value folded away: scale * bit + offset, or offset if bit is None. origin is the MLP
    that made it, and index its output there, kept in case the trace gives it as an output"""

    bit: str | None
    scale: float
    offset: float
    origin: MLP | None = None
    index: int = 0


def fold(t: Trace) -> Trace:
    """t with its constants folded"""
    env, mlps = {x: x for x in t.inputs}, []  # each signal: its name if it stays, or an Affine
    for m in t.mlps:
        out = fold_mlp(m, [env[x] for x in m.inputs])
        mlps += [out] if isinstance(out, MLP) else []
        env.update(zip(m.outputs, m.outputs if isinstance(out, MLP) else out))
    outputs = []
    for v in (env[y] for y in t.outputs):
        if isinstance(v, Affine) and v.bit is not None and (v.scale, v.offset) == (1, 0):
            v = v.bit  # the bit itself
        elif isinstance(v, Affine):  # the trace gives it, so its MLP stays after all
            mlps += [v.origin] if v.origin not in mlps else []
            v = v.origin.outputs[v.index]
        outputs.append(v)
    return replace(t, outputs=outputs, mlps=needed(mlps, outputs))


def fold_mlp(m: MLP, xs) -> MLP | list[Affine]:
    """m on xs, each a name or an Affine: the MLP that is left, or what its outputs fold to"""
    cols, b = {}, m.b.copy()  # the weights on each signal still read, and the bias
    for w, x in zip(m.wi.T, xs):
        x = x if isinstance(x, Affine) else Affine(x, 1.0, 0.0)
        b += w * x.offset
        if x.bit is not None:
            cols[x.bit] = cols.get(x.bit, 0) + w * x.scale
    live, step, reglu = tuple(cols), m.activation == "step", m.activation == "reglu"
    wi = np.array([cols[x] for x in live]).reshape(len(live), len(b)).T
    m = replace(m, inputs=live, wi=wi, b=b)
    if m.activation in ("step", "relu", "reglu"):  # drop the hidden units that are never on
        n = len(m.hidden)
        top = m.b[:n] + np.maximum(m.wi[:n], 0).sum(1)  # each gate's largest value on bits
        on = top >= 0 if step else top > 0
        if reglu:  # a value of 0 gives 0 too
            on &= m.wi[n:].any(1) | (m.b[n:] != 0)
        rows, hidden = np.r_[on, on] if reglu else on, tuple(h for h, o in zip(m.hidden, on) if o)
        m = replace(m, wi=m.wi[rows], b=m.b[rows], wo=m.wo[:, on], hidden=hidden)
    always = step and (m.b + np.minimum(m.wi, 0).sum(1) >= 0).all()  # every gate always on
    if not live or not len(m.b) or always:  # a constant
        m = replace(m, inputs=(), wi=m.wi[:, :0])
        return [Affine(None, 0.0, y, m, i) for i, y in enumerate(m.eval([]).tolist())]
    if len(live) == 1:  # scale * x + offset on the bit it reads
        ys = zip(m.eval([0]).tolist(), m.eval([1]).tolist())
        return [Affine(live[0], q - p, p, m, i) for i, (p, q) in enumerate(ys)]
    return m

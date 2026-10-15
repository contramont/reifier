"""Compile a graph into SwiGLU layers, the feed-forward blocks of transformers."""

from dataclasses import dataclass

import numpy as np

from .core import GLU, Graph, Signal, flatten, lower


@dataclass
class Layer:
    """h -> wo @ (silu(wg @ h) * (wv @ h)), where h[0] = 1 (BOS) carries the biases"""

    wg: np.ndarray
    wv: np.ndarray
    wo: np.ndarray


def to_swiglu(graph: Graph, k: float = 32.0, dialect: str = "glu") -> list[Layer]:
    """One layer per GLU depth. Copies carry signals to the layers that read them,
    constants fold into biases, and silu(k g) / k stands in for relu(g)"""
    g = flatten(lower(graph, dialect))
    const: dict[Signal, float] = {}
    depth = {s: 0 for s in g.inputs}
    for n in g.nodes:
        if not isinstance(n.op, GLU):
            raise TypeError(f"{dialect} left a {type(n.op).__name__}, not only GLUs")
        (s,) = n.outputs
        if live := [depth[x] for x in n.inputs if x not in const]:
            depth[s] = 1 + max(live)
        else:
            const[s] = float(n.op.eval([const[x] for x in n.inputs])[0])
    top = max([1] + [depth[s] for s in g.outputs if s in depth])
    last = {s: top for s in g.outputs if s in depth}  # last level a signal is read from
    for n in g.nodes:
        if (s := n.outputs[0]) in depth:
            for x in n.inputs:
                if x in depth:
                    last[x] = max(last.get(x, 0), depth[s] - 1)
    levels = [list(g.inputs)] + [[] for _ in range(top - 1)] + [list(g.outputs)]
    for s in list(g.inputs) + [n.outputs[0] for n in g.nodes]:
        for d in range(max(depth.get(s, top), 1), min(last.get(s, 0), top - 1) + 1):
            levels[d].append(s)
    return [
        _layer(levels[d - 1], levels[d], d, depth, const, k) for d in range(1, top + 1)
    ]


def _layer(xs, ys, d, depth, const, k) -> Layer:
    """The units that compute level d (ys) from level d - 1 (xs)"""
    col = {s: i + 1 for i, s in enumerate(xs)}
    units, outs = [], []

    def row(w, b, ins):
        r = np.zeros(len(xs) + 1)
        r[0] = b
        for wi, x in zip(w, ins):
            if x in const:
                r[0] += wi * const[x]
            else:
                r[col[x]] += wi
        return r

    def add(y, g, v):
        outs.append((y, len(units)))
        units.append((g, v))

    one = row([], 1, [])
    add(0, one, one)  # BOS
    for y, s in enumerate(ys, start=1):
        if s in const:
            add(y, one, row([], const[s], []))
        elif depth[s] == d:
            for u in s.node.op.units:
                add(y, row(u.g, u.gb, s.node.inputs), row(u.v, u.vb, s.node.inputs))
        else:  # a copy: relu(1) * s
            add(y, one, row([1], 0, [s]))
    wo = np.zeros((len(ys) + 1, len(units)))
    for y, i in outs:
        wo[y, i] += 1 / k
    wg = np.array([g for g, _ in units]) * k
    return Layer(wg, np.array([v for _, v in units]), wo)


def run_swiglu(layers: list[Layer], x) -> np.ndarray:
    """Run the layers on rows of input bits (a leading 1 is added and removed)"""
    h = np.atleast_2d(np.asarray(x, float))
    h = np.hstack([np.ones((len(h), 1)), h])
    for L in layers:
        z = h @ L.wg.T
        h = (0.5 * z * (1 + np.tanh(z / 2)) * (h @ L.wv.T)) @ L.wo.T  # silu(z) * value
    return h[:, 1:]

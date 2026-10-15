"""Target architectures: the global structure around placed units.

A network is a stack of layers. Each layer holds its placed units side by side as one
unit, such as one feed-forward block. Global parts act on whole layers: BOS = 1 leads
the features of every layer and carries the biases, and an RMSNorm can come before
each layer, in which case outputs are relative to BOS.
"""

from functools import cache

import numpy as np

from .core import Graph, emit, inputs
from .place import carry, place
from .units import GATED, Unit, cast


def network(graph: Graph, f: str = "geglu", norm: bool = False) -> Graph:
    """graph compiled for a stack of layers of f units, such as an MLP of step or relu
    units, or GLU layers (reglu, swiglu, geglu), each after an RMSNorm if norm"""
    return stack(place(graph, "reglu" if f in GATED else f), f, norm)


def stack(placed: Graph, f: str, norm: bool = False) -> Graph:
    """The global structure around placed layers: one unit of f per layer, BOS first in
    every layer, and an RMSNorm before each layer if norm"""
    h = ins = inputs("bit", *(s.kind for s in placed.inputs))
    for layer in (n.circuit for n in placed.nodes):
        if norm:
            h = emit(rmsnorm(tuple(s.kind for s in h)), h)
        h = emit(cast(merge(layer), f), h)
    return Graph(f"{placed.name}.{f}", ins, h)


def rmsnorm(kinds) -> Unit:
    """An RMSNorm over a whole layer: global, so not one of the placed units"""
    eye = np.eye(len(kinds))
    return Unit(eye, np.zeros(len(kinds)), "rmsnorm", eye, kinds, "rmsnorm")


@cache
def side_by_side(layer: Graph) -> Unit:
    """A layer's units, and the units of the layers of blocks in it, side by side as one
    unit. Built once for each layer, however many times its circuit is used"""
    return _beside(*_parts(layer), len(layer.inputs), str(layer))


def merge(layer: Graph) -> Unit:
    """A layer as one unit from [BOS, *inputs] to [BOS, *outputs], with biases on BOS"""
    parts, rows = _parts(layer)
    parts = [(carry(parts[0][0].f, [1], 0.0, "bit"), [0])] + [
        (u, [c + 1 for c in cols]) for u, cols in parts
    ]
    rows = [(0, 0)] + [(i + 1, j) for i, j in rows]
    return _beside(parts, rows, len(layer.inputs) + 1, str(layer), bos=True)


def _parts(layer: Graph):
    """The units of a layer (a block's layer as one unit) with the columns each reads, and
    for each output, the part and the part's output it is"""
    nodes = list(dict.fromkeys(y.node for y in layer.outputs))  # in the order they were placed
    col = {x: i for i, x in enumerate(layer.inputs)}
    units = [side_by_side(n.circuit) if isinstance(n.circuit, Graph) else n.circuit for n in nodes]
    where = {y: (i, j) for i, n in enumerate(nodes) for j, y in enumerate(n.outputs)}
    return [(u, [col[x] for x in n.inputs]) for u, n in zip(units, nodes)], [
        where[y] for y in layer.outputs
    ]


def _beside(parts, rows, n_in, label, bos=False) -> Unit:
    """Units side by side on n_in inputs, each reading the columns given with it; output
    r is output j of part i for rows[r] = (i, j). With bos, biases go into column 0"""
    f, sizes = parts[0][0].f, [c.hidden for c, _ in parts]
    start, halves = np.cumsum([0, *sizes]), 2 if parts[0][0].f in GATED else 1
    width = start[-1]
    wi, b, wo = (
        np.zeros((halves * width, n_in)),
        np.zeros(halves * width),
        np.zeros((len(rows), width)),
    )
    for (c, cols), s in zip(parts, start):
        for h in range(halves):  # gates, then values: rows s.. of each half
            r, src = (
                slice(h * width + s, h * width + s + c.hidden),
                slice(h * c.hidden, (h + 1) * c.hidden),
            )
            if len(set(cols)) == len(cols):
                wi[r, cols] = c.wi[src]
            else:  # an input read twice adds up
                np.add.at(wi[r].T, cols, c.wi[src].T)
            b[r] = c.b[src]
    for r, (i, j) in enumerate(rows):
        wo[r, start[i] : start[i + 1]] = parts[i][0].wo[j]
    if bos:
        wi[:, 0], b = wi[:, 0] + b, 0 * b
    return Unit(wi, b, f, wo, tuple(parts[i][0].kinds[j] for i, j in rows), label)

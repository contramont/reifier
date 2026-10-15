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
    every layer, and an RMSNorm before each layer if norm. Reads [BOS, *inputs] and gives
    [BOS, *outputs]"""
    h = ins = inputs("bit", *(s.kind for s in placed.inputs))
    for layer in (n.circuit for n in placed.nodes):
        if norm:
            h = emit(rmsnorm(tuple(s.kind for s in h)), h)
        h = emit(cast(merge(layer), f), h)
    return Graph(f"{placed.name}.{f}", ins, [h[0], *(h[1 + y.index] for y in placed.outputs)])


def rmsnorm(kinds) -> Unit:
    """An RMSNorm over a whole layer: global, so not one of the placed units"""
    eye = np.eye(len(kinds))
    return Unit(eye, np.zeros(len(kinds)), "rmsnorm", eye, kinds, "rmsnorm")


@cache
def side_by_side(layer: Graph) -> Unit:
    """A layer's units, and the units of the layers of blocks in it, side by side as one
    unit. Built once for each layer, however many times its circuit is used"""
    return _beside(_parts(layer), len(layer.inputs), str(layer))


def merge(layer: Graph) -> Unit:
    """A layer as one unit from [BOS, *inputs] to [BOS, *outputs], with biases on BOS"""
    parts = [(u, [c + 1 for c in cols]) for u, cols in _parts(layer)]
    bos = carry(parts[0][0].f, [1], 0.0, "bit")
    return _beside([(bos, [0]), *parts], len(layer.inputs) + 1, str(layer), bos=True)


def _parts(layer: Graph):
    """The units of a layer, a block's layer as one unit, each with the columns it reads,
    in the order of their outputs"""
    col = {x: i for i, x in enumerate(layer.inputs)}
    return [
        (
            side_by_side(n.circuit) if isinstance(n.circuit, Graph) else n.circuit,
            [col[x] for x in n.inputs],
        )
        for n in dict.fromkeys(y.node for y in layer.outputs)
    ]


def _beside(parts, n_in, label, bos=False) -> Unit:
    """Units side by side on n_in inputs, each reading the columns given with it, their
    outputs one after another. With bos, biases go into column 0"""
    f = parts[0][0].f
    hs = np.cumsum([0, *(c.hidden for c, _ in parts)])
    os = np.cumsum([0, *(len(c.wo) for c, _ in parts)])
    halves, width = 2 if f in GATED else 1, hs[-1]
    wi, b, wo = (
        np.zeros((halves * width, n_in)),
        np.zeros(halves * width),
        np.zeros((os[-1], width)),
    )
    for (c, cols), s, o in zip(parts, hs, os):
        for h in range(halves):  # gates, then values: rows s.. of each half
            r = slice(h * width + s, h * width + s + c.hidden)
            src = slice(h * c.hidden, (h + 1) * c.hidden)
            if len(set(cols)) == len(cols):
                wi[r, cols] = c.wi[src]
            else:  # an input read twice adds up
                np.add.at(wi[r].T, cols, c.wi[src].T)
            b[r] = c.b[src]
        wo[o : o + len(c.wo), s : s + c.hidden] = c.wo
    if bos:
        wi[:, 0], b = wi[:, 0] + b, 0 * b
    return Unit(wi, b, f, wo, tuple(k for c, _ in parts for k in c.kinds), label)

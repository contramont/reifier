"""Target architectures: the global structure around placed units.

A network is a stack of layers. Each layer holds its placed units side by side as one
unit, such as one feed-forward block. Global parts act on whole layers: BOS = 1 leads
the features of every layer and carries the biases, and an RMSNorm can come before
each layer, in which case outputs are relative to BOS.
"""

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


def merge(layer: Graph) -> Unit:
    """The units of a layer side by side, as one unit from [BOS, *inputs] to [BOS, *outputs]
    with the biases on BOS"""
    nodes = list(dict.fromkeys(y.node for y in layer.outputs))  # in the order they were placed
    f, col = nodes[0].circuit.f, {x: i + 1 for i, x in enumerate(layer.inputs)}
    parts = [(carry(f, [1], 0.0, "bit"), [0])]  # BOS
    parts += [(n.circuit, [col[x] for x in n.inputs]) for n in nodes]
    gates, values, offset, start = [], [], 0, {}
    for i, (c, cols) in enumerate(parts):
        wi = np.zeros((len(c.b), len(col) + 1))
        np.add.at(wi.T, cols, c.wi.T)  # the unit's inputs, as columns of the layer
        wi[:, 0] += c.b
        gates.append(wi[: c.hidden])
        values.append(wi[c.hidden :])
        start[i], offset = offset, offset + c.hidden
    where = {y: (i + 1, j) for i, n in enumerate(nodes) for j, y in enumerate(n.outputs)}
    wo = np.zeros((len(layer.outputs) + 1, offset))
    for row, (i, j) in enumerate([(0, 0)] + [where[y] for y in layer.outputs]):
        wo[row, start[i] : start[i] + parts[i][0].hidden] = parts[i][0].wo[j]
    wi = np.vstack(gates + values if f in GATED else gates)
    return Unit(wi, np.zeros(len(wi)), f, wo, ("bit", *(y.kind for y in layer.outputs)), str(layer))

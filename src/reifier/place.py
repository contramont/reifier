"""Placing units in layers: the local structure of a network.

Each circuit is placed once, as a block of layers, and reused wherever it is called: its
units and the blocks of the circuits it calls each go in the layer after their inputs,
side by side, and copies carry signals to the layers that read them. A block reads its
inputs in its first layer and gives its outputs in its last, so what a circuit holds
stays together. Placed units keep their weights: placing one only wires it to the layer
below. Constants fold into the biases of the units that read them.
"""

from dataclasses import replace
from functools import cache

import numpy as np

from .core import Graph, emit, inputs, lower
from .units import Unit


def place(graph: Graph, f: str) -> Graph:
    """graph as a chain of layers of f units, each layer a graph of units and of the layers
    of the placed circuits that graph calls"""
    return _place(lower(graph, f), f)


@cache
def _place(g: Graph, f: str) -> Graph:
    value, blocks = {}, {}  # constants; each node's layers and the signals its first one reads
    for n in g.nodes:
        c = n.circuit
        if all(x in value for x in n.inputs):  # a constant
            value |= zip(n.outputs, c.eval([value[x] for x in n.inputs]))
        elif isinstance(c, Graph):
            blocks[n] = [layer.circuit for layer in _place(c, f).nodes], list(n.inputs)
        elif isinstance(c, Unit) and c.f == f:  # constant inputs go into the biases
            live = [j for j, x in enumerate(n.inputs) if x not in value]
            k = np.array([value.get(x, 0.0) for x in n.inputs])
            blocks[n] = (
                [replace(c, wi=c.wi[:, live], b=c.b + c.wi @ k)],
                [n.inputs[j] for j in live],
            )
        else:
            raise TypeError(f"{c} is not a unit of {f}")
    made = list(dict.fromkeys(x for _, xs in blocks.values() for x in xs if x in value))
    start, end = {}, dict.fromkeys(g.inputs, 0) | dict.fromkeys(made, 1)
    for n, (layers, xs) in blocks.items():
        start[n] = 1 + max((end[x] for x in xs), default=0)
        end |= dict.fromkeys(n.outputs, start[n] + len(layers) - 1)
    top = max([1] + [end[y] for y in g.outputs if y not in value])
    need = dict.fromkeys(g.outputs, top)  # the last level each signal is read at
    for n, (_, xs) in blocks.items():
        for x in xs:
            need[x] = max(need.get(x, 0), start[n] - 1)

    h = ins = inputs(*(s.kind for s in g.inputs))
    level = list(g.inputs)  # what each feature of the level below holds: a signal of g,
    for d in range(1, top + 1):  # or (n, j), output j of a block n still in progress
        here = inputs(*(s.kind for s in h))
        at, keys, outs = dict(zip(level, here)), [], []
        for x in made if d == 1 else ():
            keys.append(x)
            outs += emit(carry(f, [], value[x], x.kind), [])
        for n, (layers, xs) in blocks.items():
            if start[n] <= d < start[n] + len(layers):
                k = d - start[n]
                reads = (
                    [at[x] for x in xs]
                    if k == 0
                    else [at[n, j] for j in range(len(layers[k - 1].kinds))]
                )
                outs += emit(layers[k], reads)
                keys += (
                    n.outputs
                    if k == len(layers) - 1
                    else [(n, j) for j in range(len(layers[k].kinds))]
                )
        for x in level:
            if isinstance(x, tuple) or need.get(x, 0) < d:  # not needed above
                continue
            keys.append(x)
            outs += emit(carry(f, [1], 0.0, x.kind), [at[x]])
        if d == top:  # the outputs of g, in order
            for y in g.outputs:
                if y in value and y not in set(keys):
                    keys.append(y)
                    outs += emit(carry(f, [], value[y], y.kind), [])
            first = {}
            for i, key in enumerate(keys):
                first.setdefault(key, i)
            outs = [outs[first[y]] for y in g.outputs]
        h, level = emit(Graph(f"{g.name}[{d}/{top}]", here, outs), h), keys
    return Graph(f"{g.name}.{f}", ins, h)


def carry(f: str, r, c: float, kind: str) -> Unit:
    """A unit of f that passes on its input (r = [1]) or the constant c (r = [])"""
    r = np.array([r], float)
    if f == "step":  # bits only
        wi, b, wo = r, [c - 0.5], [[1]]
    elif f == "relu":
        wi, b, wo = np.vstack([r, -r]), [c, -c], [[1, -1]]
    else:  # gated: relu(1) * x
        wi, b, wo = np.vstack([0 * r, r]), [1, c], [[1]]
    label = "copy" if r.size else f"{c:g}"
    return Unit(wi, np.array(b, float), f, np.array(wo, float), (kind,), label)

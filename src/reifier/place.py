"""Placing units in layers: the local structure of a network.

Each unit goes to the layer after its inputs. A placed unit keeps its weights: placing it
only wires it to the layer below. Copies carry signals to the layers that read them, and
constants fold into the biases of the units that read them.
"""

from dataclasses import replace

import numpy as np

from .core import Graph, emit, flatten, inputs, lower
from .units import Unit


def place(graph: Graph, f: str) -> Graph:
    """graph as a chain of layers, each a graph of units of f side by side"""
    g = flatten(lower(graph, f))
    value, depth = {}, dict.fromkeys(g.inputs, 0)
    for n in g.nodes:
        if not (isinstance(n.circuit, Unit) and n.circuit.f == f):
            raise TypeError(f"{n.circuit} is not a unit of {f}")
        if live := [depth[x] for x in n.inputs if x not in value]:
            depth |= dict.fromkeys(n.outputs, 1 + max(live))
        else:  # a constant
            value |= zip(n.outputs, n.circuit.eval([value[x] for x in n.inputs]))
    top = max([1] + [depth[y] for y in g.outputs if y in depth])
    levels = [g.outputs]  # level d - 1 holds what level d reads: its inputs or copies
    for d in range(top, 1, -1):
        reads = {}
        for y in levels[0]:
            if y not in value:
                reads |= dict.fromkeys(y.node.inputs if depth[y] == d else [y])
        levels.insert(0, [x for x in reads if x not in value])
    levels.insert(0, g.inputs)
    h = ins = inputs(*(s.kind for s in g.inputs))
    for d in range(1, top + 1):
        h = emit(_layer(levels[d - 1], levels[d], d, f, depth, value), h)
    return Graph(f"{graph.name}.{f}", ins, h)


def _layer(xs, ys, d, f, depth, value) -> Graph:
    """Layer d: the units that compute ys from xs, the signals of level d - 1"""
    ins = inputs(*(x.kind for x in xs))
    at, placed, outs = dict(zip(xs, ins)), {}, []
    for y in ys:
        if y in value:
            outs += emit(carry(f, [], value[y], y.kind), [])
        elif depth[y] == d:
            n, c = y.node, y.node.circuit
            if n not in placed:  # constant inputs go into the biases
                live = [j for j, x in enumerate(n.inputs) if x not in value]
                k = np.array([value.get(x, 0.0) for x in n.inputs])
                c = replace(c, wi=c.wi[:, live], b=c.b + c.wi @ k)
                placed[n] = emit(c, [at[n.inputs[j]] for j in live])
            outs.append(placed[n][y.index])
        else:
            outs += emit(carry(f, [1], 0.0, y.kind), [at[y]])
    return Graph(f"layer {d}", ins, outs)


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

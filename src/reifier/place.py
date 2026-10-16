"""Placing units on a grid: the local structure of a network.

A level is a row of features, and a layer maps the level below it to the next. Each
circuit is placed once, as a block: a rectangle of layers and features, as wide as its
widest level, its inputs included, with the features it leaves unused empty. A unit is
as wide as its inputs or its outputs, whichever are more. Each use of a circuit puts its
block in the layers after its inputs, at the first features free on all of its levels.
Units are blocks one layer high, constants too, and copies carry signals straight up to
the levels that read them. A block reads its inputs in its first layer and gives its
outputs in its last. Placed units keep their weights: placing one only wires it to the
level below.

Placing is two steps: arranging the rectangles, then building the layers they make.
"""

from collections import defaultdict
from functools import cache

import numpy as np

from .core import Graph, emit, inputs, lower
from .units import Unit


def place(graph: Graph, f: str) -> Graph:
    """graph as a chain of layers of f units, each layer a graph of the units and of the
    layers of the blocks in it, one after another, as wide as every other layer"""
    return _place(lower(graph, f), f)


@cache
def _place(g: Graph, f: str) -> Graph:
    blocks = {n: block(n.circuit, f) for n in g.nodes}  # each node's layers, its outputs' features
    start, end = {}, dict.fromkeys(g.inputs, 0)
    for n, (layers, _) in blocks.items():  # each block in the layers after its inputs
        start[n] = 1 + max((end[x] for x in n.inputs), default=0)
        end |= dict.fromkeys(n.outputs, start[n] + len(layers) - 1)
    top = max([1] + [end[y] for y in g.outputs])
    need = dict.fromkeys(g.outputs, top)  # the last level each signal is read at
    for n in blocks:
        for x in n.inputs:
            need[x] = max(need.get(x, 0), start[n] - 1)

    # rectangles (first level, layers, signals the first reads, outputs' features, a copy?),
    # lowest first and copies first, each at the first features free on all of its levels
    rects = [
        (start[n], ls, n.inputs, dict(zip(n.outputs, at)), False) for n, (ls, at) in blocks.items()
    ]
    rects += [
        (end[x] + 1, [carry(f, x.kind)] * (need[x] - end[x]), [x], {x: 0}, True)
        for x in need
        if need[x] > end[x]
    ]
    where = {(x, 0): i for i, x in enumerate(g.inputs)}  # (signal, level) -> feature
    grid, placed = defaultdict(bytearray), []  # the features used at each level; the rectangles

    def fits(c, w, levels):
        return not any(any(grid[d][c : c + w]) for d in levels)

    for s, ls, xs, outs, copy in sorted(rects, key=lambda r: (r[0], not r[4])):
        w, levels = max(len(ls[0].kinds), len(xs)), range(s, s + len(ls))
        c = where[xs[0], s - 1] if copy else 0  # a copy goes straight up if it can
        if not fits(c, w, levels):
            c = max(len(grid[d]) if 0 not in grid[d] else grid[d].index(0) for d in levels)
            while not fits(c, w, levels):
                c += 1
        for d in levels:
            grid[d] += bytes(max(0, c + w - len(grid[d])))
            grid[d][c : c + w] = b"\1" * w
        placed.append((c, s, ls, xs))
        where |= {
            (x, d): c + j for x, j in outs.items() for d in (levels if copy else [s + len(ls) - 1])
        }
    return build(g, f, placed, where)


def block(c, f: str) -> tuple[list, list]:
    """The block of a circuit: its layers, and the feature of each output in its last"""
    if isinstance(c, Graph):
        p = _place(c, f)
        return [layer.circuit for layer in p.nodes], [y.index for y in p.outputs]
    if isinstance(c, Unit) and c.f == f:
        return [c], list(range(len(c.kinds)))
    raise TypeError(f"{c} is not a unit of {f}")


def build(g: Graph, f: str, rects, where) -> Graph:
    """The layers of rectangles (feature, first level, layers, signals the first reads),
    where[x, d] being the feature of signal x at level d: each layer a graph of the units
    and the blocks' layers in it, side by side, the features none of them uses empty"""
    layer = defaultdict(list)  # the rectangles on each level
    for c, s, ls, xs in rects:
        for d in range(s, s + len(ls)):
            layer[d].append((c, s, ls, xs))
    top = max([1, *layer])
    width = max([len(g.inputs)] + [c + max(len(ls[0].kinds), len(xs)) for c, _, ls, xs in rects])
    h = ins = inputs(*(s.kind for s in g.inputs))
    for d in range(1, top + 1):
        here, outs = inputs(*(s.kind for s in h)), []
        for c, s, ls, xs in sorted(layer[d], key=lambda r: r[0]):
            outs += emit(empty(f, c - len(outs)), []) if c > len(outs) else []
            if d == s:  # its first layer reads its inputs; the others, its own outputs
                reads = [here[where[x, d - 1]] for x in xs]
            else:
                reads = here[c : c + len(ls[0].kinds)]
            outs += emit(ls[d - s], reads)
        outs += emit(empty(f, width - len(outs)), []) if width > len(outs) else []
        h = emit(Graph(f"{g.name}[{d}/{top}]", here, outs), h)
    return Graph(f"{g.name}.{f}", ins, [h[where[y, top]] for y in g.outputs])


def carry(f: str, kind: str) -> Unit:
    """A unit of f that passes on its input: a copy"""
    r = np.ones((1, 1))
    if f == "step":  # bits only
        wi, b, wo = r, [-0.5], [[1]]
    elif f == "relu":
        wi, b, wo = np.vstack([r, -r]), [0, 0], [[1, -1]]
    else:  # gated: relu(1) * x
        wi, b, wo = np.vstack([0 * r, r]), [1, 0], [[1]]
    return Unit(wi, np.array(b, float), f, np.array(wo, float), (kind,), "copy")


@cache
def empty(f: str, k: int) -> Unit:
    """k unused features: a unit of f with no hidden units, so always 0"""
    return Unit(np.zeros((0, 0)), np.zeros(0), f, np.zeros((k, 0)), ("bit",) * k, "empty")

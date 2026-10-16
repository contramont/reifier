"""Placing units on a grid: the local structure of a network.

A level is a row of features, and a layer maps the level below it to the next. Each
circuit is placed once, as a block: a rectangle of layers and features, as wide as its
widest level, its inputs included, with the features it leaves unused empty. A unit is
as wide as its inputs or its outputs, whichever are more. Each use of a circuit puts its
block in the layers after its inputs, at the first features free on all of its levels.
Units are blocks one layer high, and copies carry signals straight up to the levels
that read them. A block reads its inputs in its first layer and gives its outputs in
its last. Placed units keep their weights: placing one only wires it to the level
below. Constants fold into the biases of the units that read them, and an input wired
to nothing reads a feature nothing makes. A block can be moved to a later layer, or to
other features where they are free; copies, to other features; and layers can be kept
free of blocks.
"""

from collections import defaultdict
from dataclasses import replace
from functools import cache

import numpy as np

from .core import Graph, emit, inputs, lower
from .units import Unit


def place(graph: Graph, f: str, at: dict | None = None, empty=(), unread=()) -> Graph:
    """graph as a chain of layers of f units, each layer a graph of the units and of the
    layers of the blocks in it, one after another, as wide as every other layer. at moves
    blocks, {i: (layer, feature)} for graph.nodes[i], no lower than its inputs allow and
    at that feature if free, and copies, {("in", i) or (i, j): feature} for input i or
    output j of node i. empty layers hold no blocks, only copies. unread outputs, by index,
    stay where they are made, left out of the result's outputs. The result tells where
    each block went, at, and what each (level, feature) carries, features: ("in", i) or
    (i, j)"""
    moves = tuple(sorted((at or {}).items(), key=repr))
    return _place(lower(graph, f), f, moves, tuple(sorted(set(empty))), tuple(sorted(set(unread))))


@cache
def _place(g: Graph, f: str, at: tuple = (), gaps: tuple = (), unread: tuple = ()) -> Graph:
    def signal(r):
        return g.inputs[r[1]] if r[0] == "in" else g.nodes[r[0]].outputs[r[1]]

    want = {g.nodes[k]: v for k, v in at if isinstance(k, int)}  # lowering keeps the order
    outputs = [y for i, y in enumerate(g.outputs) if i not in unread]
    carried = {signal(k): v for k, v in at if isinstance(k, tuple)}  # where copies go
    free = {n.outputs[0] for n in g.nodes if getattr(n.circuit, "label", "") == "unwired"}
    value, blocks = {}, {}  # constants; each node's layers, what its first reads, its outputs
    for n in g.nodes:
        c = n.circuit
        if all(x in value for x in n.inputs) and not free.intersection(n.inputs):  # a constant
            value |= zip(n.outputs, c.eval([value[x] for x in n.inputs]))
        elif isinstance(c, Graph):
            p = _place(c, f)
            at = [y.index for y in p.outputs]  # the feature of each output in its last level
            blocks[n] = [layer.circuit for layer in p.nodes], list(n.inputs), at
        elif isinstance(c, Unit) and c.f == f:  # constant inputs go into the biases
            live = [j for j, x in enumerate(n.inputs) if x not in value]
            k = np.array([value.get(x, 0.0) for x in n.inputs])
            u = replace(c, wi=c.wi[:, live], b=c.b + c.wi @ k)
            blocks[n] = [u], [n.inputs[j] for j in live], range(len(c.kinds))
        else:
            raise TypeError(f"{c} is not a unit of {f}")
    made = [x for _, xs, _ in blocks.values() for x in xs if x in value and x not in free]
    made = list(dict.fromkeys(made))
    start, end = {}, dict.fromkeys(g.inputs, 0) | dict.fromkeys([*made, *free], 1)
    for n, (layers, xs, _) in blocks.items():
        s = max(1 + max((end[x] for x in xs), default=0), want.get(n, (0, 0))[0])
        while any(d in gaps for d in range(s, s + len(layers))):  # over the empty layers
            s += 1
        start[n] = s
        end |= dict.fromkeys(n.outputs, start[n] + len(layers) - 1)
    top = max([1, *gaps] + [end[y] for y in outputs if y not in value])
    made += [y for y in dict.fromkeys(outputs) if y in value and y not in end]
    end |= {y: top for y in made if y not in end}  # constant outputs: made in the last layer
    need = dict.fromkeys(outputs, top)  # the last level each signal is read at
    for n, (_, xs, _) in blocks.items():
        for x in set(xs) - free:
            need[x] = max(need.get(x, 0), start[n] - 1)

    # rectangles (first level, layers, signals the first reads, outputs, a copy?, node),
    # lowest first, copies and then moved blocks first, each where it was moved if free
    # there, else at the first features free on all of its levels
    rects = [(end[x], [carry(f, [], value[x], x.kind)], [], {x: 0}, False, None) for x in made]
    rects += [
        (start[n], ls, xs, dict(zip(n.outputs, o)), False, n) for n, (ls, xs, o) in blocks.items()
    ]
    rects += [
        (end[x] + 1, [carry(f, [1], 0.0, x.kind)] * (need[x] - end[x]), [x], {x: 0}, True, x)
        for x in need
        if need[x] > end[x]
    ]
    where = {(x, 0): i for i, x in enumerate(g.inputs)}  # (signal, level) -> feature
    grid, layer, col = defaultdict(bytearray), defaultdict(list), {}  # used; by level; columns

    def fits(c, w, levels):
        return not any(any(grid[d][c : c + w]) for d in levels)

    moved = lambda r: r[5] in (carried if r[4] else want)
    for s, ls, xs, outs, copy, n in sorted(rects, key=lambda r: (r[0], not r[4], not moved(r))):
        w, levels = max(len(ls[0].kinds), len(xs)), range(s, s + len(ls))
        # a copy goes straight up if it can; a moved block or copy, where it was moved to
        c = carried.get(n, where[xs[0], s - 1]) if copy else want[n][1] if n in want else 0
        if not fits(c, w, levels):
            c = max(len(grid[d]) if 0 not in grid[d] else grid[d].index(0) for d in levels)
            while not fits(c, w, levels):
                c += 1
        for d in levels:
            grid[d] += bytes(max(0, c + w - len(grid[d])))
            grid[d][c : c + w] = b"\1" * w
            layer[d].append((c, s, ls, xs))
        if n is not None and not copy:
            col[n] = c
        where |= {
            (x, d): c + j for x, j in outs.items() for d in (levels if copy else [s + len(ls) - 1])
        }

    zero = {}  # below a block that reads an input wired to nothing: a feature nothing makes, 0
    for n, (_, xs, _) in blocks.items():
        if free.intersection(xs) and (d := start[n] - 1) not in zero:
            zero[d] = grid[d].index(0) if 0 in grid[d] else len(grid[d])
            grid[d] += bytes(max(0, zero[d] + 1 - len(grid[d])))

    width = max(len(g.inputs), *(len(grid[d]) for d in range(1, top + 1)))
    h = ins = inputs(*(s.kind for s in g.inputs))
    for d in range(1, top + 1):
        here, outs = inputs(*(s.kind for s in h)), []
        for c, s, ls, xs in sorted(layer[d], key=lambda r: r[0]):
            outs += emit(empty(f, c - len(outs)), []) if c > len(outs) else []
            if d == s:  # a block's first layer reads its inputs, one wired to nothing a 0
                reads = [here[zero[d - 1] if x in free else where[x, d - 1]] for x in xs]
            else:
                reads = here[c : c + len(ls[0].kinds)]
            outs += emit(ls[d - s], reads)
        outs += emit(empty(f, width - len(outs)), []) if width > len(outs) else []
        h = emit(Graph(f"{g.name}[{d}/{top}]", here, outs), h)
    out = Graph(f"{g.name}.{f}", ins, [h[where[y, top]] for y in outputs])
    index = {n: i for i, n in enumerate(g.nodes)}
    ref = lambda x: ("in", g.inputs.index(x)) if x.node not in index else (index[x.node], x.index)
    out.at = {index[n]: (start[n], c) for n, c in col.items()}
    out.features = {(d, c): ref(x) for (x, d), c in where.items()}
    return out


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


@cache
def empty(f: str, k: int) -> Unit:
    """k unused features: a unit of f with no hidden units, so always 0"""
    return Unit(np.zeros((0, 0)), np.zeros(0), f, np.zeros((k, 0)), ("bit",) * k, "empty")

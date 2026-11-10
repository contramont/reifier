"""Tiles: a layout that keeps each circuit together, as a rectangle of the grid.

The names of a trace say which call of a circuit made each MLP: the MLPs whose names start with
add/xor.1/ are one call of xor. Each call is a tile: a rectangle of depths and features that
holds the signals of its MLPs and the tiles of the calls inside it, and carries its outputs to
its last depth. A tile goes in the layers after the tiles that make what it reads, at the first
features free at all of its depths, so a grid of tiles can be deeper than layout's. A signal
read after the depth that makes it is carried straight up, and what a tile reads stays where it
is, outside the tile. tiles gives a Grid, as layout does, so the emitters and the checks are the
same.
"""

import numpy as np

from .grid import ONE, Grid
from .trace import MLP, Trace


def tiles(t: Trace) -> Grid:
    """A grid for t: ONE, a feature for each input, then the tiles"""
    tree = {}  # the calls by name: a dict for a circuit, an MLP for a leaf
    for m in t.mlps:
        *scopes, leaf = m.name.split("/")
        node = tree
        for part in scopes:
            node = node.setdefault(part, {})
        node[leaf] = m
    body, reads = tile(tree, set(t.outputs))
    cells = np.full((1 + len(body), 1 + len(t.inputs) + body.shape[1]), None)
    cells[:, 0], cells[1:, 1 + len(t.inputs) :] = ONE, body
    for i, x in enumerate(t.inputs):  # an input stays in its feature while a layer reads it
        cells[: reads.get(x, 1), 1 + i] = x
    return Grid(cells.tolist())


def mlps_in(node) -> list[MLP]:
    return [node] if isinstance(node, MLP) else [m for kid in node.values() for m in mlps_in(kid)]


def tile(node, wanted: set[str]) -> tuple[np.ndarray, dict[str, int]]:
    """The rectangle of a call, rows by features: the signals it holds after each of its layers,
    those of wanted in its last. And, for each signal from outside, the last of its layers that
    reads it"""
    if isinstance(node, MLP):
        return np.array([node.outputs], object), dict.fromkeys(node.inputs, 1)
    kids = [mlps_in(kid) for kid in node.values()]
    makes = [{y for m in kid for y in m.outputs} for kid in kids]
    reads = [{x for m in kid for x in m.inputs} - made for kid, made in zip(kids, makes)]
    given = wanted.union(*reads)  # a kid has what the other kids read of it in its last row
    parts = [tile(kid, made & given) for kid, made in zip(node.values(), makes)]
    depth, first = {}, []  # the row that gives each signal made here; each kid's first row
    for rect, read in parts:
        first.append(1 + max((depth.get(x, 0) for x in read), default=0))
        depth |= {y: first[-1] + len(rect) - 1 for y in rect[-1] if y is not None}
    height = max([1, *depth.values()])
    last = dict.fromkeys(wanted, height)  # the last row each signal is read from
    for row, (_, read) in zip(first, parts):
        last |= {x: max(last.get(x, 0), row + layer - 2) for x, layer in read.items()}
    todo = [(row, 1, rect, None) for row, (rect, _) in zip(first, parts)]
    todo += [  # a column for each signal made here and read after its row
        (depth[x] + 1, 0, np.full((last[x] - depth[x], 1), x), x)
        for x in last
        if x in depth and last[x] > depth[x]
    ]
    cells, taken = np.full((height + 1, 0), None), np.zeros((height + 1, 0), bool)
    for row, _, rect, x in sorted(todo, key=lambda item: item[:2]):  # lowest first, columns first
        h, w = rect.shape
        free = np.r_[~taken[row : row + h].any(0), np.ones(w, bool)]  # past the last feature, all
        fits = np.convolve(free, np.ones(w, int), "valid") == w  # w free features from here
        up = np.flatnonzero(cells[row - 1] == x) if x is not None else []  # where x is, below
        f = up[0] if len(up) else int(np.argmax(fits))  # above a column is free: it came first
        if (more := f + w - cells.shape[1]) > 0:
            cells = np.c_[cells, np.full((height + 1, more), None)]
            taken = np.c_[taken, np.zeros((height + 1, more), bool)]
        cells[row : row + h, f : f + w], taken[row : row + h, f : f + w] = rect, True
    return cells[1:], {x: row + 1 for x, row in last.items() if x not in depth}

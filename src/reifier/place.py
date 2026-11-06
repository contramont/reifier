"""Placing MLPs on a grid: the local structure of a network.

The features at each depth make a row, and a layer maps the row at one depth to the next.
Each circuit is placed once, as a tile: a rectangle of layers and features, as wide as its
widest row, its inputs included, with the features it leaves unused empty. An MLP is
as wide as its inputs or its outputs, whichever are more. Each use of a circuit puts its
tile in the layers after its inputs, at the first features free at all of its depths.
MLPs are tiles one layer high, constants too, and passthroughs carry signals straight up to
the depths that read them. A tile reads its inputs in its first layer and gives its
outputs in its last. Placed MLPs keep their weights: placing one only wires it to the
depth below.

Placing is two steps: arranging the tiles, then building the layers they make.
"""

from collections import defaultdict
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import numpy as np

from .core import Graph, apply, lower, placeholders
from .mlp import MLP


def wide_tiles(outs: int, ins: int, hidden: int) -> int:
    """The features of a tile that gives outs and reads ins: as many as it reads or gives"""
    return max(outs, ins)


def tight_tiles(mlp_ratio: int | None) -> Callable:
    """Tiles that keep no features for what they read: as many as they give, or as their
    hidden units need with mlp_ratio to a feature (any number if mlp_ratio is None),
    whichever are more"""
    return lambda outs, ins, hidden: max(outs, -(-hidden // mlp_ratio) if mlp_ratio else 0)


def place(graph: Graph, activation: str, width_rule: Callable = wide_tiles) -> Graph:
    """graph as a chain of layers of MLPs of activation, each layer a graph of the MLPs and of
    the layers of the tiles in it, one after another, as wide as every other layer"""
    return _place(lower(graph, activation), activation, width_rule)


class Placement(NamedTuple):
    """A tile in layers first_layer, first_layer + 1, ...: its layers, the signals its first
    layer reads, the feature of each of its outputs, whether it is a passthrough, and the
    feature it starts at, once placed"""

    first_layer: int
    layers: list
    inputs: list
    output_features: dict
    is_passthrough: bool
    offset: int = 0


@cache
def _place(g: Graph, activation: str, width_rule: Callable) -> Graph:
    # each node's layers, its outputs' features
    tiles = {n: tile(n.circuit, activation, width_rule) for n in g.nodes}
    first_layer, depth = {}, dict.fromkeys(g.inputs, 0)
    for n, (layers, _) in tiles.items():  # each tile in the layers after its inputs
        first_layer[n] = 1 + max((depth[x] for x in n.inputs), default=0)
        depth |= dict.fromkeys(n.outputs, first_layer[n] + len(layers) - 1)
    n_layers = max([1] + [depth[y] for y in g.outputs])
    last_read = dict.fromkeys(g.outputs, n_layers)  # the last depth each signal is read at
    for n in tiles:
        for x in n.inputs:
            last_read[x] = max(last_read.get(x, 0), first_layer[n] - 1)

    # the tiles, lowest first and passthroughs first, each at the first features free at all of
    # its depths
    todo = [
        Placement(first_layer[n], layers, n.inputs, dict(zip(n.outputs, at)), False)
        for n, (layers, at) in tiles.items()
    ]
    todo += [
        Placement(
            depth[x] + 1,
            [passthrough(activation, x.kind)] * (last_read[x] - depth[x]),
            [x],
            {x: 0},
            True,
        )
        for x in last_read
        if last_read[x] > depth[x]
    ]
    feature_of = {(x, 0): i for i, x in enumerate(g.inputs)}  # (signal, depth) -> feature
    used, placed = defaultdict(bytearray), []  # the features used at each depth; the tiles

    def first_fit(start, width, depths):
        """The first feature from start with width free at all of depths"""
        while (e := max(find_free(used[d], start, width) for d in depths)) > start:
            start = e
        return start

    for p in sorted(todo, key=lambda p: (p.first_layer, not p.is_passthrough)):
        width = tile_width(width_rule, p.layers, p.inputs)
        depths = range(p.first_layer, p.first_layer + len(p.layers))
        # a passthrough goes straight up if it can
        start = feature_of[p.inputs[0], p.first_layer - 1] if p.is_passthrough else 0
        if first_fit(start, width, depths) != start:
            start = first_fit(0, width, depths)
        for d in depths:
            used[d] += bytes(max(0, start + width - len(used[d])))
            used[d][start : start + width] = b"\1" * width
        placed.append(p._replace(offset=start))
        feature_of |= {
            (x, d): start + j
            for x, j in p.output_features.items()
            for d in (depths if p.is_passthrough else [depths[-1]])
        }
    return build_layers(g, activation, placed, feature_of, width_rule)


def find_free(row: bytearray, start: int, width: int) -> int:
    """The first feature from start that starts width free ones in a row, whose last feature
    is used"""
    return i if (i := row.find(bytes(width), start)) >= 0 else max(start, len(row))


def tile_width(width_rule: Callable, layers: list, inputs) -> int:
    """The features of a tile of layers that reads inputs"""
    first = layers[0]
    return width_rule(
        len(first.output_kinds), len(inputs), first.n_hidden if isinstance(first, MLP) else 0
    )


def tile(c, activation: str, width_rule: Callable = wide_tiles) -> tuple[list, list]:
    """The tile of a circuit: its layers, and the feature of each output in its last"""
    if isinstance(c, Graph):
        p = _place(c, activation, width_rule)
        return [layer.circuit for layer in p.nodes], [y.index for y in p.outputs]
    if isinstance(c, MLP) and c.activation == activation:
        return [c], list(range(len(c.output_kinds)))
    raise TypeError(f"{c} is not an MLP of {activation}")


def build_layers(
    g: Graph, activation: str, placed, feature_of, width_rule: Callable = wide_tiles
) -> Graph:
    """The layers of placed tiles, feature_of[x, d] being the feature of signal x at depth d:
    each layer a graph of the MLPs and the tiles' layers in it, side by side, the features none
    of them uses empty"""
    at_depth = defaultdict(list)  # the tiles at each depth
    for p in placed:
        for d in range(p.first_layer, p.first_layer + len(p.layers)):
            at_depth[d].append(p)
    n_layers = max([1, *at_depth])
    right = max([0] + [p.offset + tile_width(width_rule, p.layers, p.inputs) for p in placed])
    width = width_rule(right, len(g.inputs), 0)
    h = ins = placeholders(*(s.kind for s in g.inputs))
    for d in range(1, n_layers + 1):
        here, outs = placeholders(*(s.kind for s in h)), []
        for p in sorted(at_depth[d], key=lambda p: p.offset):
            outs += (
                apply(padding(activation, p.offset - len(outs)), []) if p.offset > len(outs) else []
            )
            if d == p.first_layer:  # its first layer reads its inputs; the others, its own outputs
                reads = [here[feature_of[x, d - 1]] for x in p.inputs]
            else:
                reads = here[p.offset : p.offset + len(p.layers[0].output_kinds)]
            outs += apply(p.layers[d - p.first_layer], reads)
        outs += apply(padding(activation, width - len(outs)), []) if width > len(outs) else []
        h = apply(Graph(f"{g.name}[{d}/{n_layers}]", here, outs), h)
    return Graph(f"{g.name}@{activation}", ins, [h[feature_of[y, n_layers]] for y in g.outputs])


def passthrough(activation: str, kind: str) -> MLP:
    """An MLP of activation that passes on its input"""
    r = np.ones((1, 1))
    if activation == "step":  # bits only
        wi, b, wo = r, [-0.5], [[1]]
    elif activation == "relu":
        wi, b, wo = np.vstack([r, -r]), [0, 0], [[1, -1]]
    else:  # gated: relu(1) * x
        wi, b, wo = np.vstack([0 * r, r]), [1, 0], [[1]]
    return MLP(wi, np.array(b, float), activation, np.array(wo, float), (kind,), "passthrough")


@cache
def padding(activation: str, k: int) -> MLP:
    """k unused features: an MLP of activation with no hidden units, so always 0"""
    return MLP(np.zeros((0, 0)), np.zeros(0), activation, np.zeros((k, 0)), ("bit",) * k, "padding")

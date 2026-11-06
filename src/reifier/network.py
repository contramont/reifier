"""Networks: the global structure around placed MLPs.

A network is a stack of layers. Each layer holds its placed MLPs side by side as one
MLP, such as one feed-forward block, mlp_ratio times as wide inside as it is outside: each
MLP's hidden units sit under its features, mlp_ratio for each. Global parts act on whole
layers: ONE = 1 leads the features of every layer and carries the biases, and an RMSNorm
can come before each layer, in which case outputs are relative to ONE.
"""

from functools import cache

import numpy as np

from .core import Graph, apply, placeholders
from .mlp import GATED_ACTIVATIONS, MLP, convert
from .place import passthrough, place, tight_tiles, wide_tiles


def network(
    graph: Graph,
    activation: str = "geglu",
    pre_norm: bool = False,
    mlp_ratio: int | None = 4,
    tight: bool = False,
) -> Graph:
    """graph compiled for a stack of layers, each one MLP of activation: step or relu MLPs,
    or GLUs (reglu, swiglu, geglu), each after an RMSNorm if pre_norm"""
    width_rule = tight_tiles(mlp_ratio) if tight else wide_tiles
    placed = place(graph, "reglu" if activation in GATED_ACTIVATIONS else activation, width_rule)
    return assemble(placed, activation, pre_norm, mlp_ratio)


def assemble(
    placed: Graph, activation: str, pre_norm: bool = False, mlp_ratio: int | None = 4
) -> Graph:
    """The global structure around placed layers: one MLP of activation per layer, ONE first
    in every layer, and an RMSNorm before each layer if pre_norm. Each layer has mlp_ratio
    hidden units for each feature, or just those its MLPs use if mlp_ratio is None. Reads
    [ONE, *inputs] and gives [ONE, *outputs]"""
    h = ins = placeholders("bit", *(s.kind for s in placed.inputs))
    for layer in (n.circuit for n in placed.nodes):
        if pre_norm:
            h = apply(rmsnorm(tuple(s.kind for s in h)), h)
        h = apply(convert(fuse_with_one(layer, mlp_ratio), activation), h)
    outs = [h[0], *(h[1 + y.index] for y in placed.outputs)]
    return Graph(f"{placed.name}:{activation}", ins, outs)


def rmsnorm(kinds) -> MLP:
    """An RMSNorm over a whole layer: global, so not one of the placed MLPs"""
    eye = np.eye(len(kinds))
    return MLP(eye, np.zeros(len(kinds)), "rmsnorm", eye, kinds, "rmsnorm")


@cache
def fuse(layer: Graph, mlp_ratio: int | None = None) -> MLP:
    """A layer's MLPs, and the MLPs of the layers of tiles in it, side by side as one
    MLP. Built once for each layer, however many times its circuit is used"""
    return _fuse_parts(_layer_parts(layer, mlp_ratio), len(layer.inputs), str(layer), mlp_ratio)


def fuse_with_one(layer: Graph, mlp_ratio: int | None = None) -> MLP:
    """A layer as one MLP from [ONE, *inputs] to [ONE, *outputs], with biases on ONE"""
    parts = [(c, [i + 1 for i in features]) for c, features in _layer_parts(layer, mlp_ratio)]
    one = passthrough(parts[0][0].activation, "bit")
    n_in = len(layer.inputs) + 1
    return _fuse_parts([(one, [0]), *parts], n_in, str(layer), mlp_ratio, biases_on_one=True)


def _layer_parts(layer: Graph, mlp_ratio: int | None):
    """The MLPs of a layer, a tile's layer as one MLP, each with the features it reads,
    in the order of their outputs"""
    feature = {x: i for i, x in enumerate(layer.inputs)}
    return [
        (
            fuse(n.circuit, mlp_ratio) if isinstance(n.circuit, Graph) else n.circuit,
            [feature[x] for x in n.inputs],
        )
        for n in dict.fromkeys(y.node for y in layer.outputs)
    ]


def _fuse_parts(parts, n_in, name, mlp_ratio=None, biases_on_one=False) -> MLP:
    """MLPs side by side on n_in inputs, each reading the features given with it, their
    outputs one after another, and their hidden units too: if mlp_ratio, under their
    features, mlp_ratio for each of their outputs and of the unused features after them.
    With biases_on_one, biases go into feature 0"""
    activation = parts[0][0].activation
    out_offsets = np.cumsum([0, *(len(c.wo) for c, _ in parts)])
    if mlp_ratio:
        hidden_offsets = mlp_ratio * out_offsets
    else:
        hidden_offsets = np.cumsum([0, *(c.n_hidden for c, _ in parts)])
    ends, end = [], hidden_offsets[-1]  # how far each part's hidden units may reach
    for (c, _), start in zip(parts[::-1], hidden_offsets[-2::-1]):
        ends.append(end)
        end = start if c.n_hidden or not mlp_ratio else end
    for (c, _), start, limit in zip(parts, hidden_offsets, ends[::-1]):
        if c.n_hidden > limit - start:
            raise ValueError(
                f"{c} has {c.n_hidden} hidden units, more than {mlp_ratio} for each feature"
            )
    halves, width = 2 if activation in GATED_ACTIVATIONS else 1, hidden_offsets[-1]
    wi, b, wo = (
        np.zeros((halves * width, n_in)),
        np.zeros(halves * width),
        np.zeros((out_offsets[-1], width)),
    )
    for (c, features), s, o in zip(parts, hidden_offsets, out_offsets):
        for half in range(halves):  # gates, then values: rows s.. of each half
            r = slice(half * width + s, half * width + s + c.n_hidden)
            src = slice(half * c.n_hidden, (half + 1) * c.n_hidden)
            if len(set(features)) == len(features):
                wi[r, features] = c.wi[src]
            else:  # an input read twice adds up
                np.add.at(wi[r].T, features, c.wi[src].T)
            b[r] = c.b[src]
        wo[o : o + len(c.wo), s : s + c.n_hidden] = c.wo
    if biases_on_one:
        wi[:, 0], b = wi[:, 0] + b, 0 * b
    return MLP(wi, b, activation, wo, tuple(k for c, _ in parts for k in c.output_kinds), name)

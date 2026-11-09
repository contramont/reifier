"""Networks: plain stacks of layers, each one MLP, with the program's MLPs placed by schedule.

A plain stack has no residual stream: a layer gives the next all it reads. So each layer holds
the MLPs that schedule puts in it side by side, and a passthrough for each other signal that a
later layer reads. Global parts act on whole layers: ONE = 1 leads the features of every layer
and carries the biases, and an RMSNorm can come before each layer, in which case outputs are
relative to ONE. Each layer has mlp_ratio hidden units for each feature, or just those its MLPs
use if mlp_ratio is None. A GLU has no bias but ONE, so it passes ONE on as about ONE^2: without
RMSNorms, a deep gated stack doubles ONE's error at each layer.
"""

from dataclasses import dataclass, replace

import numpy as np

from .attention import Attention
from .compiler import Schedule, Stack, schedule
from .core import Circuit, Graph, apply, flatten, lower, placeholders
from .mlp import GATED_ACTIVATIONS, MLP, passthrough

ONE = 0  # the feature of ONE; a program's inputs come after it


@dataclass(frozen=True, eq=False)
class RMSNorm(Circuit):
    """An RMSNorm over a whole layer: global, so not one of the program's MLPs"""

    output_kinds: tuple[str, ...]

    def eval(self, x):
        return x / np.sqrt(np.mean(x * x, 0) + 1e-6)

    def __str__(self):
        return "rmsnorm"


def network(
    graph: Graph, activation: str = "geglu", pre_norm: bool = False, mlp_ratio: int | None = 4
) -> Graph:
    """graph compiled into a stack of layers, each one MLP of activation (step, relu, reglu,
    swiglu or geglu) after an RMSNorm if pre_norm. Reads [ONE, *inputs], gives [ONE, *outputs]"""
    g = flatten(lower(graph, activation))
    for c in (n.circuit for n in g.nodes):
        if isinstance(c, Attention):
            raise TypeError("a plain stack has no attention: compile for a transformer")
        if not isinstance(c, MLP) or c.activation != activation:
            raise TypeError(f"{c} is not an MLP of {activation}")
    s = schedule(g, plain_stack(g))
    h = ins = placeholders("bit", *(x.kind for x in g.inputs))
    for layer in layer_mlps(s, [x.kind for x in g.inputs], activation, mlp_ratio):
        if pre_norm:
            h = apply(RMSNorm(tuple(x.kind for x in h)), h)
        h = apply(layer, h)
    return Graph(f"{graph.name}:{activation}", ins, [h[ONE], *(h[f] for f in s.outputs)])


class AnyRoom:
    """A sublayer of a plain stack: as many MLPs side by side as come"""

    def take(self, node) -> bool:
        return True


def plain_stack(g: Graph) -> Stack:
    """A plain stack for flat graph g: as many layers and features as it could need (a layer for
    each node, and one for a graph of none), its inputs after ONE, its outputs anywhere"""
    n_in = len(g.inputs)
    return Stack(
        n_layers=len(g.nodes) + 1,
        width=1 + n_in + sum(len(n.outputs) for n in g.nodes),
        inputs=list(range(1, 1 + n_in)),
        outputs=None,
        reserved=frozenset({ONE}),
        room=lambda s: AnyRoom(),
    )


def layer_mlps(s: Schedule, kinds: list[str], activation: str, mlp_ratio: int | None) -> list:
    """Each layer of a schedule as one MLP over the whole row, ONE and passthroughs included,
    for inputs of kinds"""
    written = [f for layer in s.layers for *_, outs in layer.mlps for f in outs]
    width = 1 + max([ONE, *s.inputs, *s.outputs, *written])
    read_after, carried = set(s.outputs), []  # the features each layer passes on
    for layer in reversed(s.layers):
        carried.insert(0, sorted(read_after - {f for *_, outs in layer.mlps for f in outs}))
        read_after = set(carried[0]) | {f for _, ins, _ in layer.mlps for f in ins}
    kind, layers = {ONE: "bit", **dict(zip(s.inputs, kinds))}, []  # what each feature holds
    for layer, passed in zip(s.layers, carried):
        if activation == "step" and any(kind[f] != "bit" for f in passed):
            raise TypeError("a step passthrough passes on bits only")
        parts = [(passthrough(activation), [ONE], [ONE]), *layer.mlps]
        parts += [(passthrough(activation, kind[f]), [f], [f]) for f in passed]
        layers.append(fuse(parts, width, activation, mlp_ratio))
        kind |= {f: k for c, _, outs in layer.mlps for f, k in zip(outs, c.output_kinds)}
    layers[0] = replace(layers[0], wi=layers[0].wi[:, : 1 + len(s.inputs)])  # ONE, the inputs
    return layers


def fuse(parts, width: int, activation: str, mlp_ratio: int | None) -> MLP:
    """MLPs side by side as one MLP from width features to width features, each MLP reading
    and writing the features given with it, their hidden units one after another, and their
    biases on ONE"""
    halves = 2 if activation in GATED_ACTIVATIONS else 1
    needed = sum(c.n_hidden for c, _, _ in parts)
    n = mlp_ratio * width if mlp_ratio else needed
    if needed > n:
        raise ValueError(f"a layer needs {needed} hidden units, more than {mlp_ratio} a feature")
    wi, wo, row = np.zeros((halves * n, width)), np.zeros((width, n)), 0
    kinds = ["bit"] * width
    for c, ins, outs in parts:
        k = c.n_hidden
        for half in range(halves):  # gates, then values
            rows = slice(half * n + row, half * n + row + k)
            np.add.at(wi[rows].T, ins, c.wi[half * k : (half + 1) * k].T)  # inputs read twice add
            wi[rows, ONE] += c.b[half * k : (half + 1) * k]
        wo[outs, row : row + k] = c.wo
        for f, kind in zip(outs, c.output_kinds):
            kinds[f] = kind
        row += k
    return MLP(wi, np.zeros(halves * n), activation, wo, tuple(kinds), "layer")

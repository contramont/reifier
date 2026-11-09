"""The compiler: placing a program's MLPs and attention heads in layers, and its signals in
features.

A target is a stack of layers, each an attention sublayer and then an MLP sublayer, over a
stream of features: each sublayer sets some of them. schedule puts each leaf of a flat program
in the first sublayer of its kind after those that give its inputs, and that has room for it,
and gives each signal a feature from the sublayer that gives it to the last that reads it.
Features are used again once free. An MLP can write over a value that no one reads any more,
but a head cannot, so a head writes into a feature that holds nothing, which the MLP sublayer
before it erases if need be.

A target may pin the program's inputs and outputs to features of their own, such as a token's
bits. Then each output comes from an MLP of its own, so an output that is an input, a head's
output or another output is copied by a passthrough MLP.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .attention import Attention
from .core import Circuit, Graph, Node, apply, placeholders


class Room(Protocol):
    """What a sublayer of a target has room for"""

    def take(self, node: Node) -> bool:
        """Whether node fits beside those taken already, taking it if so"""


@dataclass(frozen=True)
class Stack:
    """A target: its layers and features; the features of the program's inputs, and of its
    outputs if they are pinned (None: wherever they are); the features no signal may take; a
    fresh Room for each sublayer, room(sublayer); and an MLP that copies a signal of a kind,
    passthrough(kind), for pinned outputs"""

    n_layers: int
    width: int
    inputs: list[int]
    outputs: list[int] | None
    reserved: frozenset[int]
    room: Callable[[int], Room]
    passthrough: Callable[[str], Circuit] | None = None


@dataclass
class Layer:
    """A layer's heads and MLPs, each with the features it reads and those it writes, and the
    features its MLP sublayer erases for the heads of the next layer"""

    heads: list[tuple[Attention, list[int], list[int]]] = field(default_factory=list)
    mlps: list[tuple[Circuit, list[int], list[int]]] = field(default_factory=list)
    erase: list[int] = field(default_factory=list)


@dataclass
class Schedule:
    """The layers of a program, and the features of its inputs and outputs"""

    inputs: list[int]
    outputs: list[int]
    layers: list[Layer]

    def __str__(self):
        lines = []
        for i, layer in enumerate(self.layers):
            back = sorted({c.back for c, _, _ in layer.heads})
            hidden = sum(c.n_hidden for c, _, _ in layer.mlps)
            lines.append(
                f"layer {i}: {len(layer.heads)} heads looking back {back}, "
                f"{len(layer.mlps)} MLPs with {hidden} hidden units, {len(layer.erase)} erased"
            )
        return "\n".join(lines)


def schedule(g: Graph, stack: Stack) -> Schedule:
    """The leaves of flat program g in the layers of stack, and features for its signals"""
    if len(g.inputs) != len(stack.inputs):
        raise ValueError(f"{g} has {len(g.inputs)} inputs, not {len(stack.inputs)}")
    if stack.outputs is not None:
        if len(g.outputs) != len(stack.outputs):
            raise ValueError(f"{g} has {len(g.outputs)} outputs, not {len(stack.outputs)}")
        g = copy_outputs(g, stack.passthrough)
    g = trim_heads(g)
    return assign_features(g, stack, place_leaves(g, stack))


def copy_outputs(g: Graph, passthrough: Callable) -> Graph:
    """g with a copy of each output that is an input, a head's output or an earlier output"""
    outputs = []
    for y in g.outputs:
        if y in outputs or y in g.inputs or isinstance(y.node.circuit, Attention):
            y = apply(passthrough(y.kind), [y])[0]
        outputs.append(y)
    return Graph(g.name, g.inputs, outputs)


def trim_heads(g: Graph) -> Graph:
    """g with each head looking back only at the values that something reads, in the end"""
    read = set(g.outputs)  # from the last node back: a head reads what it gives that is read
    for n in reversed(g.nodes):
        if isinstance(n.circuit, Attention):
            read |= {x for x, y in zip(n.inputs, n.outputs) if y in read}
        elif read.intersection(n.outputs):
            read |= set(n.inputs)
    ins = placeholders(*(x.kind for x in g.inputs))
    env = dict(zip(g.inputs, ins))
    for n in g.nodes:
        if isinstance(n.circuit, Attention):
            kept = [i for i, y in enumerate(n.outputs) if y in read]
            kinds = tuple(n.circuit.output_kinds[i] for i in kept)
            if kept:
                ys = apply(Attention(n.circuit.back, kinds), [env[n.inputs[i]] for i in kept])
                env |= dict(zip([n.outputs[i] for i in kept], ys))
        elif read.intersection(n.outputs):
            env |= dict(zip(n.outputs, apply(n.circuit, [env[x] for x in n.inputs])))
    return Graph(g.name, ins, [env[y] for y in g.outputs])


def place_leaves(g: Graph, stack: Stack) -> dict[Node, int]:
    """Each node's sublayer: 2 i is the attention of layer i and 2 i + 1 its MLP sublayer, and
    each node goes in the first of its kind after those of its inputs that has room for it"""
    sublayer, given_at, end, rooms = {}, dict.fromkeys(g.inputs, -1), 2 * stack.n_layers, {}
    for n in g.nodes:
        kind = 0 if isinstance(n.circuit, Attention) else 1  # the parity of its sublayers
        s = 1 + max((given_at[x] for x in n.inputs), default=-1)
        s += (s - kind) % 2  # the first sublayer of its kind
        while s < end and not rooms.setdefault(s, stack.room(s)).take(n):
            s += 2
        if s >= end:
            leaf = f"the {'head' if kind == 0 else 'MLP'} {n.circuit}"
            if not any(stack.room(t).take(n) for t in range(kind, end, 2)):
                raise ValueError(f"{leaf} fits in no sublayer of the target")
            raise ValueError(f"{leaf} does not fit in {stack.n_layers} layers")
        sublayer[n] = s
        given_at |= dict.fromkeys(n.outputs, s)
    return sublayer


def assign_features(g: Graph, stack: Stack, sublayer: dict) -> Schedule:
    """Features for the signals of g, sublayer by sublayer: a signal holds one from the sublayer
    that gives it to the last that reads it, and a head's outputs take features that hold
    nothing, erased by the MLP sublayer before if need be"""
    n_layers = 1 + max([0, *sublayer.values()]) // 2
    last_read = dict.fromkeys(g.outputs, 2 * n_layers)  # outputs stay to the end
    nodes_at = defaultdict(list)
    for n, s in sublayer.items():
        nodes_at[s].append(n)
        for x in n.inputs:
            last_read[x] = max(last_read.get(x, -1), s)
    freed = defaultdict(list)  # the signals each sublayer reads last, or gives unread
    for x in g.inputs:
        freed[last_read.get(x, 0)].append(x)
    for n, s in sublayer.items():
        for y in n.outputs:
            freed[last_read.get(y, s + 1)].append(y)
    pinned = stack.outputs or []
    feature = dict(zip(g.inputs, stack.inputs)) | dict(zip(g.outputs, pinned))
    kept = {*stack.inputs, *pinned, *stack.reserved}
    empty = [f for f in reversed(range(stack.width)) if f not in kept]  # never written
    dead = []  # free, but holding values that no one reads
    layers = [Layer() for _ in range(n_layers)]

    def free(s: int):  # the features of the signals that sublayer s reads last
        dead.extend(feature[x] for x in freed.pop(s, []) if feature[x] not in pinned)

    def take(pool: list, s: int) -> int:
        if not pool:
            raise ValueError(f"{g} needs more than {stack.width} features in layer {s // 2}")
        return pool.pop()

    for s in range(2 * n_layers):
        layer, is_heads = layers[s // 2], s % 2 == 0
        if not is_heads:
            free(s)  # an MLP can write over what it reads
        for n in nodes_at[s]:
            for y in n.outputs:
                if y in feature:  # a pinned output
                    continue
                if is_heads and not empty and dead and s:  # erase one in the MLPs before
                    layers[s // 2 - 1].erase.append(dead[-1])
                    empty.append(dead.pop())
                feature[y] = take(empty if is_heads else dead or empty, s)
            entry = (n.circuit, [feature[x] for x in n.inputs], [feature[y] for y in n.outputs])
            (layer.heads if is_heads else layer.mlps).append(entry)
        if is_heads:
            free(s)
    return Schedule(list(stack.inputs), [feature[y] for y in g.outputs], layers)


def simulate(sched: Schedule, rows, width: int) -> np.ndarray:
    """The outputs of a schedule at each position, for a row of inputs per position after a
    zero token, as Graph.eval_sequence gives them: the leaves' values as they move through a
    stream of width features, a check of placement that needs no weights. It checks that a
    head writes only into a feature that holds nothing: not an input's or an output's (which a
    transformer embeds a token's bits in), nor one written and not erased since"""
    stream, held = np.zeros((width, 1 + len(rows))), np.zeros(width, bool)
    stream[sched.inputs, 1:] = np.asarray(rows, float).T
    held[[*sched.inputs, *sched.outputs]] = True
    for layer in sched.layers:
        written = [(c.eval(stream[ins]), outs) for c, ins, outs in layer.heads]
        for y, outs in written:
            assert not held[outs].any(), "a head writes over a value"
            stream[outs], held[outs] = y, True
        written = [(c.eval(stream[ins]), outs) for c, ins, outs in layer.mlps]
        stream[layer.erase], held[layer.erase] = 0, False
        for y, outs in written:
            stream[outs], held[outs] = y, True
    return stream[sched.outputs, 1:].T

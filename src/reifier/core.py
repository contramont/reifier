"""Circuits on symbolic signals.

A Signal is a wire with a kind ("bit", "int", "real", ...) but no value. A Circuit maps
signals to signals: a leaf circuit computes values itself, a Graph with other circuits.
@circuit traces a Python function into a Graph, once per argument structure. Passes
such as flatten and lower interpret a graph into a new graph. A graph runs at each position
of a sequence; its leaves are MLPs, which read their own position, and attention heads, which
read earlier ones.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import cache, update_wrapper
from itertools import count

import numpy as np


@dataclass(frozen=True, slots=True)
class Signal:
    """Output `index` of `node`: it has a kind, never a value"""

    node: Node
    index: int = 0

    @property
    def kind(self) -> str:
        return self.node.circuit.output_kinds[self.index]

    def __bool__(self):
        raise TypeError("signals have no value: branch on structure, not on bits")

    def __repr__(self):
        return f"<{self.kind} from {self.node.circuit}>"


_next_id = count()  # nodes are numbered as they are made: each after the nodes it reads


@dataclass(frozen=True, eq=False, slots=True)
class Node:
    """A circuit applied to signals"""

    circuit: Circuit
    inputs: tuple[Signal, ...]
    id: int = field(default_factory=lambda: next(_next_id), repr=False)

    @property
    def outputs(self) -> list[Signal]:
        return [Signal(self, i) for i in range(len(self.circuit.output_kinds))]


def apply(c: Circuit, inputs) -> list[Signal]:
    """Apply c to input signals, returning its output signals"""
    return Node(c, tuple(inputs)).outputs


class Circuit:
    """Maps signals to signals. A subclass has output_kinds, one per output; a leaf also has
    eval, which maps the values of its inputs at each position, an array with a row per input
    and a column per position, to those of its outputs"""

    output_kinds: tuple[str, ...] = ()

    def __call__(self, *inputs: Signal):
        outs = apply(self, inputs)
        return outs[0] if len(outs) == 1 else outs


@dataclass(frozen=True)
class Placeholder(Circuit):
    output_kinds: tuple[str] = ("bit",)


def placeholders(*kinds: str) -> list[Signal]:
    return [Placeholder((k,))() for k in kinds]


def bits(n: int) -> list[Signal]:
    return placeholders(*["bit"] * n)


# ---------------------------------------------------------------- graphs


def tree_flatten(tree) -> tuple[list[Signal], object]:
    """The signals in nested lists and tuples, and a hashable treedef of the rest: the kind
    of each signal, the type of each container, and every other value"""
    match tree:
        case Signal():
            return [tree], tree.kind
        case list() | tuple():
            parts = [tree_flatten(t) for t in tree]
            return [s for p, _ in parts for s in p], (type(tree), *(k for _, k in parts))
        case _:
            return [], ("=", tree)


def tree_unflatten(treedef, signals: Iterator[Signal]):
    """The inverse of tree_flatten: the tree of treedef, with its signals taken in order"""
    match treedef:
        case str():
            return next(signals)
        case ("=", value):
            return value
        case (container, *items):
            return container(tree_unflatten(k, signals) for k in items)


@dataclass(eq=False)
class Graph(Circuit):
    """A traced circuit: placeholder inputs, output signals and the nodes between them"""

    name: str
    inputs: list[Signal]
    outputs: list[Signal]
    in_tree: object = None  # the treedefs of the arguments and of the result
    out_tree: object = None
    traced_from: CircuitFunction | None = None

    def __post_init__(self):
        self.nodes = toposort(self)
        self.output_kinds = tuple(s.kind for s in self.outputs)

    def eval(self, xs) -> list[float]:
        """The output values for a value per input, or for a string of bits, spaces aside: at
        the first position, after the zero token"""
        return self.eval_sequence([xs])[0].tolist()

    def eval_sequence(self, rows) -> np.ndarray:
        """The output values at each position of a sequence, a row per position, for a row of
        input values (or a string of bits) per position. Attention reads the rows before, and
        before the first, a zero token, as a transformer reads <bos>"""
        rows = [[int(c) for c in r if c != " "] if isinstance(r, str) else r for r in rows]
        x = np.array(rows, float).reshape(len(rows), len(self.inputs))
        return evaluate(self, np.r_[np.zeros((1, len(self.inputs))), x].T).T[1:]

    def __str__(self):
        return f"{self.name}/{len(self.inputs)}"

    def __repr__(self):
        return f"<graph {self}: {len(self.nodes)} nodes>"


def toposort(g: Graph) -> list[Node]:
    """The nodes between the inputs and the outputs of g, in the order they were made,
    so each after its inputs"""
    seen, todo, stop = set(), [s.node for s in g.outputs], {s.node for s in g.inputs}
    while todo:
        if (n := todo.pop()) not in seen and n not in stop:
            if isinstance(n.circuit, Placeholder):
                raise ValueError(f"{g.name} reads a signal that is not one of its inputs")
            seen.add(n)
            todo += [s.node for s in n.inputs]
    return sorted(seen, key=lambda n: n.id)


def evaluate(g: Graph, x: np.ndarray) -> np.ndarray:
    """The values of g's outputs, for those of its inputs: a row per signal, a column per
    position"""
    n = x.shape[1]

    def eval_node(c, xs):
        if isinstance(c, Graph):
            return interpret(c, xs, eval_node)
        return list(c.eval(np.array(xs, float).reshape(len(xs), n)))

    return np.array(interpret(g, list(x), eval_node), float).reshape(len(g.outputs), n)


def interpret(g: Graph, args: list, eval_node: Callable) -> list:
    """Run g on args (values, signals, ...), with eval_node(circuit, inputs) per node"""
    env = dict(zip(g.inputs, args, strict=True))
    for n in g.nodes:
        outs = eval_node(n.circuit, [env[s] for s in n.inputs])
        env.update(zip(n.outputs, outs, strict=True))
    return [env[s] for s in g.outputs]


class CircuitFunction:
    """A Python function on signals, traced into a Graph once per argument structure;
    each later call with that structure adds one node that reuses the graph"""

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn, self.name, self.traces, self.impls = fn, name or fn.__name__, {}, {}
        update_wrapper(self, fn)

    def trace(self, *args) -> Graph:
        signals, in_tree = tree_flatten(args)
        if in_tree not in self.traces:
            ins = placeholders(*(s.kind for s in signals))
            outs, out_tree = tree_flatten(self.fn(*tree_unflatten(in_tree, iter(ins))))
            self.traces[in_tree] = Graph(self.name, ins, outs, in_tree, out_tree, self)
        return self.traces[in_tree]

    def __call__(self, *args):
        g = self.trace(*args)
        return tree_unflatten(g.out_tree, iter(apply(g, tree_flatten(args)[0])))

    def impl(self, dialect: str) -> Callable:
        """Register a hand-written version of this circuit, for lowering to dialect"""

        def register(fn):
            self.impls[dialect] = CircuitFunction(fn, f"{self.name}.{dialect}")
            return self.impls[dialect]

        return register


circuit = CircuitFunction


def trace(fn: Callable, *args) -> Graph:
    """The graph of fn, decorated or not, e.g. trace(add, bits(4), bits(4))"""
    return (fn if isinstance(fn, CircuitFunction) else CircuitFunction(fn)).trace(*args)


# ---------------------------------------------------------------- passes


def rewrite(g: Graph, rewrite_node: Callable) -> Graph:
    """g with each node replaced by rewrite_node(circuit, inputs)"""
    ins = placeholders(*(s.kind for s in g.inputs))
    return replace(g, inputs=ins, outputs=interpret(g, ins, rewrite_node))


def _inline_node(c, xs):
    return interpret(c, xs, _inline_node) if isinstance(c, Graph) else apply(c, xs)


@cache
def flatten(g: Graph) -> Graph:
    """g with every sub-circuit inlined"""
    return rewrite(g, _inline_node)


LOWERING_RULES: dict[tuple[str, type], Callable] = {}  # (dialect, leaf type) -> rewrite of a leaf
FALLBACKS: dict[str, str] = {}  # a dialect -> one whose hand-written versions it can use too


def hand_written(g: Graph, dialect: str) -> CircuitFunction | None:
    """The hand-written version of g for dialect, or for one that dialect falls back to"""
    while g.traced_from and dialect:
        if impl := g.traced_from.impls.get(dialect):
            return impl
        dialect = FALLBACKS.get(dialect)
    return None


@cache
def lower(g: Graph, dialect: str) -> Graph:
    """g in a dialect: a circuit with a hand-written version for it, or else for a dialect it
    falls back to in FALLBACKS (such as reglu for geglu), uses that version, and a leaf with a
    rule in LOWERING_RULES for it is rewritten by the rule"""
    if impl := hand_written(g, dialect):
        return lower(impl.trace(*tree_unflatten(g.in_tree, iter(g.inputs))), dialect)

    def lower_node(c, xs):
        if rule := LOWERING_RULES.get((dialect, type(c))):
            c = rule(c)
        return apply(lower(c, dialect) if isinstance(c, Graph) else c, xs)

    return rewrite(g, lower_node)


def format_graph(g: Graph) -> str:
    """g and every distinct circuit inside it, callees first, one line per node"""
    done, out = set(), []

    def visit(g):
        if g in done:
            return
        done.add(g)
        for n in g.nodes:
            if isinstance(n.circuit, Graph):
                visit(n.circuit)
        name = {s: f"%{i}" for i, s in enumerate(g.inputs)}
        lines = [f"{g}({' '.join(name.values())}):"]
        for n in g.nodes:
            args = " ".join(name[s] for s in n.inputs)
            for s in n.outputs:
                name[s] = f"%{len(name)}"
            lines.append(f"  {' '.join(name[s] for s in n.outputs)} = {n.circuit}({args})")
        out.append("\n".join([*lines, f"  return {' '.join(name[s] for s in g.outputs)}"]))

    visit(g)
    return "\n\n".join(out)

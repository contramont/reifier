"""Circuits on symbolic signals.

Signal   a wire with a kind (bit, int or real) but no value
Node     an op applied to signals; an op is a primitive (Gate, GLU) or a Graph
@circuit traces a function once per argument structure into a Graph; later
         calls with that structure emit one node that reuses the Graph
passes   are interpretations of a Graph: eval (numpy), flatten, lower
"""

import functools
import itertools
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

type Kind = str  # "bit" | "int" | "real"


# ---------------------------------------------------------------- signals


@dataclass(frozen=True, slots=True)
class Signal:
    """Output `index` of `node`. Symbolic: it has a kind, never a value"""

    node: Node
    index: int = 0

    @property
    def kind(self) -> Kind:
        return self.node.op.kinds[self.index]

    def __bool__(self):
        raise TypeError("signals have no value: branch on structure, not on bits")

    def __repr__(self):
        return f"<{self.kind} from {label(self.node.op)}>"


@dataclass(frozen=True, eq=False, slots=True)
class Node:
    op: Any  # Input | Gate | GLU | Graph
    inputs: tuple[Signal, ...]

    @property
    def outputs(self) -> list[Signal]:
        return [Signal(self, i) for i in range(len(self.op.kinds))]


def emit(op, inputs) -> list[Signal]:
    """Apply op to input signals, returning its output signals"""
    return Node(op, tuple(inputs)).outputs


# ---------------------------------------------------------------- primitives


@dataclass(frozen=True, slots=True)
class Input:
    kind: Kind = "bit"

    @property
    def kinds(self):
        return (self.kind,)


def _dot(w, xs):
    return sum((wi * x for wi, x in zip(w, xs, strict=True)), start=np.float64(0))


@dataclass(frozen=True, slots=True)
class Gate:
    """[w . x + b >= 0]"""

    w: tuple[int, ...]
    b: int
    kinds = ("bit",)

    def eval(self, xs):
        return [(_dot(self.w, xs) + self.b >= 0) * 1.0]


@dataclass(frozen=True, slots=True)
class Unit:
    """relu(g . x + gb) * (v . x + vb)"""

    g: tuple[float, ...]
    gb: float
    v: tuple[float, ...]
    vb: float


@dataclass(frozen=True, slots=True)
class GLU:
    """A sum of gated units whose value is a bit, an int or a real"""

    units: tuple[Unit, ...]
    kind: Kind = "bit"

    @property
    def kinds(self):
        return (self.kind,)

    def eval(self, xs):
        terms = (
            np.maximum(0, _dot(u.g, xs) + u.gb) * (_dot(u.v, xs) + u.vb)
            for u in self.units
        )
        return [sum(terms, start=np.float64(0))]


def gate(xs, w, threshold) -> Signal:
    if any(x.kind == "real" for x in xs):
        raise TypeError("gates need integer sums: their inputs must be bits or ints")
    return emit(Gate(tuple(w), -threshold), xs)[0]


def glu(xs, units, kind: Kind = "bit") -> Signal:
    return emit(GLU(tuple(units), kind), xs)[0]


def unit(g, gb, v, vb) -> Unit:
    return Unit(tuple(g), gb, tuple(v), vb)


def const(bits: str) -> list[Signal]:
    return [gate([], [], 1 - int(b)) for b in bits]


def inputs(*kinds: Kind) -> list[Signal]:
    return [emit(Input(k), [])[0] for k in kinds]


def bits(n: int) -> list[Signal]:
    return inputs(*["bit"] * n)


# ---------------------------------------------------------------- argument trees


def tree_split(tree) -> tuple[list[Signal], Hashable]:
    """The signals in nested lists/tuples, and a hashable spec of the rest:
    structure, kinds and static values, which together key the trace cache"""
    leaves: list[Signal] = []

    def spec(t):
        match t:
            case Signal():
                leaves.append(t)
                return ("signal", t.kind)
            case list() | tuple():
                return (type(t), tuple(map(spec, t)))
            case _:
                return ("static", t)

    s = spec(tree)
    hash(s)  # static arguments must be hashable
    return leaves, s


def tree_join(spec, leaves):
    it = iter(leaves)

    def build(s):
        match s:
            case ("signal", _):
                return next(it)
            case ("static", v):
                return v
            case (t, children):
                return t(map(build, children))

    return build(spec)


# ---------------------------------------------------------------- graphs


@dataclass(eq=False)
class Graph:
    """A circuit traced for one argument structure: placeholders in, signals out"""

    name: str
    inputs: list[Signal]
    outputs: list[Signal]
    in_spec: Hashable = None
    out_spec: Hashable = None
    circuit: Circuit | None = None
    nodes: list[Node] = field(init=False)
    derived: dict = field(default_factory=dict)  # memoized passes

    def __post_init__(self):
        self.nodes = _toposort(self)

    @property
    def kinds(self):
        return tuple(s.kind for s in self.outputs)

    def eval(self, xs):
        return interpret(self, xs, lambda op, ins: op.eval(ins))

    def run(self, x) -> np.ndarray:
        """Evaluate rows of input values: run('101') or run(array of shape (batch, n))"""
        x = [int(c) for c in x] if isinstance(x, str) else x
        x = np.atleast_2d(np.asarray(x, float))
        ys = self.eval(list(x.T))
        return np.stack([np.broadcast_to(y, len(x)) for y in ys], axis=-1)

    def retrace(self, prim, key) -> Graph:
        """The graph made by interpreting this one on fresh inputs, memoized by key"""
        if key not in self.derived:
            ins = inputs(*(s.kind for s in self.inputs))
            outs = interpret(self, ins, prim)
            self.derived[key] = Graph(
                self.name, ins, outs, self.in_spec, self.out_spec, self.circuit
            )
        return self.derived[key]

    def __repr__(self):
        return f"<graph {label(self)}: {len(self.nodes)} nodes>"

    def __str__(self):
        return show(self)


def _toposort(g: Graph) -> list[Node]:
    """Nodes between the inputs and the outputs, in dependency order"""
    order, done = [], {s.node for s in g.inputs}
    stack = [s.node for s in g.outputs]
    while stack:
        node = stack[-1]
        if node in done:
            stack.pop()
        elif todo := [s.node for s in node.inputs if s.node not in done]:
            stack.extend(todo)
        elif isinstance(node.op, Input):
            raise ValueError(f"{g.name} reads a signal that is not one of its inputs")
        else:
            done.add(node)
            order.append(stack.pop())
    return order


def interpret(g: Graph, args: list, prim: Callable[[Any, list], list]) -> list:
    """Run g on args (arrays, signals, ...), applying prim(op, inputs) to each node"""
    env = dict(zip(g.inputs, args, strict=True))
    for node in g.nodes:
        outs = prim(node.op, [env[s] for s in node.inputs])
        env.update(zip(node.outputs, outs, strict=True))
    return [env[s] for s in g.outputs]


# ---------------------------------------------------------------- circuits


class Circuit:
    """A function on signals, traced once per argument structure (shapes, kinds
    and static values); each later call with that structure emits one node"""

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn, self.name = fn, name or fn.__name__
        self.cache: dict[Hashable, Graph] = {}
        self.impls: dict[str, Circuit] = {}
        functools.update_wrapper(self, fn)

    def trace(self, *args) -> Graph:
        leaves, spec = tree_split(args)
        if spec not in self.cache:
            ins = inputs(*(s.kind for s in leaves))
            outs, out_spec = tree_split(self.fn(*tree_join(spec, ins)))
            self.cache[spec] = Graph(self.name, ins, outs, spec, out_spec, self)
        return self.cache[spec]

    def __call__(self, *args):
        graph = self.trace(*args)
        return tree_join(graph.out_spec, emit(graph, tree_split(args)[0]))

    def impl(self, dialect: str) -> Callable[[Callable], Circuit]:
        """Register a hand-written implementation that lowering to dialect uses"""

        def register(fn):
            self.impls[dialect] = Circuit(fn, f"{self.name}.{dialect}")
            return self.impls[dialect]

        return register

    def __repr__(self):
        return f"<circuit {self.name}>"


circuit = Circuit


def trace(fn, *args) -> Graph:
    """The graph of fn, decorated or not: trace(add, bits(4), bits(4))"""
    return (fn if isinstance(fn, Circuit) else Circuit(fn)).trace(*args)


# ---------------------------------------------------------------- passes


def _inline(op, xs):
    return interpret(op, xs, _inline) if isinstance(op, Graph) else emit(op, xs)


def flatten(graph: Graph) -> Graph:
    """Inline every sub-circuit, leaving only primitives"""
    return graph.retrace(_inline, "flat")


DIALECTS: dict[str, tuple[type, ...]] = {"threshold": (Gate,), "glu": (GLU,)}
RULES: dict[tuple[str, type], Callable] = {}


def rule(dialect: str, op_type: type):
    """Register fn(op, inputs) -> outputs that rewrites op_type for the dialect"""

    def register(fn):
        RULES[dialect, op_type] = fn
        return fn

    return register


def lower(graph: Graph, dialect: str) -> Graph:
    """Rewrite to the dialect's primitives, keeping the hierarchy. A circuit with a
    hand-written implementation for the dialect uses it instead of its body"""
    c = graph.circuit
    if c is not None and dialect in c.impls:
        impl = c.impls[dialect].trace(*tree_join(graph.in_spec, graph.inputs))
        if impl.kinds != graph.kinds:
            raise TypeError(f"{impl.name} returns {impl.kinds}, not {graph.kinds}")
        return lower(impl, dialect)

    def prim(op, xs):
        if isinstance(op, Graph):
            return emit(lower(op, dialect), xs)
        if isinstance(op, DIALECTS[dialect]):
            return emit(op, xs)
        return RULES[dialect, type(op)](op, xs)

    return graph.retrace(prim, ("lower", dialect))


@rule("glu", Gate)
def gate_to_glu(op: Gate, xs, c: int = 4) -> list[Signal]:
    """[z >= 0] on integer z = w.x + b as two units: a ramp of width 1/c centered at
    z = -1/2, so inputs off by less than 1/2 - 1/(2c) still give exactly 0 or 1"""
    g, zero, gb = [c * w for w in op.w], [0] * len(xs), c * op.b + c / 2
    return [glu(xs, [unit(g, gb + 0.5, zero, 1), unit(g, gb - 0.5, zero, -1)])]


# ---------------------------------------------------------------- inspection


def label(op) -> str:
    match op:
        case Graph():
            return f"{op.name}/{len(op.inputs)}"
        case Gate():
            return f"gate[{' '.join(map(str, op.w))} >= {-op.b}]"
        case GLU():
            return f"glu[{len(op.units)} units]:{op.kind}"
        case _:
            return type(op).__name__.lower()


def show(graph: Graph) -> str:
    """Each distinct sub-circuit once, callees first"""
    done, out = set(), []

    def visit(g):
        if g in done:
            return
        done.add(g)
        for n in g.nodes:
            if isinstance(n.op, Graph):
                visit(n.op)
        name = {s: f"%{i}" for i, s in enumerate(g.inputs)}
        lines = [f"{label(g)}({' '.join(name.values())}):"]
        for n in g.nodes:
            args = " ".join(name[s] for s in n.inputs)
            for s in n.outputs:
                name[s] = f"%{len(name)}"
            lines.append(
                f"  {' '.join(name[s] for s in n.outputs)} = {label(n.op)}({args})"
            )
        lines.append(f"  return {' '.join(name[s] for s in g.outputs)}")
        out.append("\n".join(lines))

    visit(graph)
    return "\n\n".join(out)


def verify(a: Graph, b: Graph, samples: int = 4096, seed: int = 0) -> bool:
    """Do a and b agree on every bit input (or on random ones, past 14 inputs)?"""
    n = len(a.inputs)
    if n <= 14:
        x = np.array(list(itertools.product((0, 1), repeat=n)))
    else:
        x = np.random.default_rng(seed).integers(0, 2, (samples, n))
    return np.allclose(a.run(x), b.run(x), atol=1e-6)

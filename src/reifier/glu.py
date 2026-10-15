"""GLUs, the leaf circuit of reifier: wo @ (act(wg @ x + bg) * (wv @ x + bv)).

The activation is the dialect. "step" units are linear threshold gates, the circuits
one writes; "relu" units are what the compiler places in layers; "silu" units are the
SwiGLU layers of transformers. Each stage of compiling is a Graph, so it runs and
shows like any other circuit.
"""

from dataclasses import dataclass, replace
from functools import cache

import numpy as np

from .core import RULES, Circuit, Graph, Signal, emit, flatten, inputs, lower

ACTS = {
    "step": lambda z: (z >= 0) * 1.0,
    "relu": lambda z: np.maximum(z, 0),
    "silu": lambda z: z / 2 * (1 + np.tanh(z / 2)),
}


@dataclass(frozen=True, eq=False)
class GLU(Circuit):
    """wo @ (act(wg @ x + bg) * (wv @ x + bv)): units, each a gate times a value"""

    wg: np.ndarray
    bg: np.ndarray
    wv: np.ndarray
    bv: np.ndarray
    wo: np.ndarray
    kinds: tuple[str, ...]
    act: str = "relu"

    def eval(self, x):
        gate = ACTS[self.act](self.wg @ x + self.bg[:, None])
        return self.wo @ (gate * (self.wv @ x + self.bv[:, None]))

    def __str__(self):
        if self.act == "step" and self.wo.shape == (1, 1) and not self.wv.any():  # a gate
            return f"gate[{' '.join(f'{w:g}' for w in self.wg[0])} >= {-self.bg[0]:g}]"
        return f"{self.act}[{len(self.bg)}]"


def glu(xs, units, kind="bit", act="relu") -> Signal:
    """The sum of units act(g . xs + gb) * (v . xs + vb), each given as (g, gb, v, vb)"""
    g, gb, v, vb = (np.array(c, float) for c in zip(*units))
    g, v = g.reshape(len(units), len(xs)), v.reshape(len(units), len(xs))
    return GLU(g, gb, v, vb, np.ones((1, len(units))), (kind,), act)(*xs)


def gate(xs, w, threshold) -> Signal:
    """[w . xs >= threshold]: a linear threshold gate, the unit step(w . xs - threshold)"""
    if any(x.kind == "real" for x in xs):
        raise TypeError("gates need integer sums: their inputs must be bits or ints")
    return glu(xs, [(w, -threshold, [0] * len(xs), 1)], act="step")


def const(bits: str) -> list[Signal]:
    return [gate([], [], 1 - int(b)) for b in bits]


def step_to_relu(c: GLU, k: int = 4) -> GLU:
    """step(z) on integer z as relu(kz + k/2 + 1/2) - relu(kz + k/2 - 1/2): a ramp of
    width 1/k at z = -1/2, so sums off by less than 1/2 - 1/(2k) still give 0 or 1"""
    if c.act != "step":
        return c
    b, two = k * c.bg + k / 2, lambda m: np.concatenate([m, m])
    bg = np.concatenate([b + 0.5, b - 0.5])
    return GLU(two(k * c.wg), bg, two(c.wv), two(c.bv), np.hstack([c.wo, -c.wo]), c.kinds)


def relu_to_silu(c: GLU, k: float = 32.0) -> GLU:
    """silu(kz) / k, which is relu(z) to within |z| exp(-k |z|)"""
    c = step_to_relu(c)
    if c.act != "relu":
        return c
    return replace(c, wg=k * c.wg, bg=k * c.bg, wo=c.wo / k, act="silu")


RULES["relu", GLU] = step_to_relu
RULES["silu", GLU] = relu_to_silu


@cache
def layers(graph: Graph, dialect: str = "relu") -> Graph:
    """graph as a stack of relu GLUs without biases, whose inputs and outputs begin with
    BOS = 1, as in transformers. Each unit goes to the layer after its inputs, copies
    carry signals to the layers that read them, and constants are multiples of BOS"""
    g = flatten(lower(graph, dialect))
    value, depth = {}, dict.fromkeys(g.inputs, 0)
    for n in g.nodes:
        if not (isinstance(n.circuit, GLU) and n.circuit.act == "relu"):
            raise TypeError(f"layers needs relu GLUs, not {n.circuit}")
        if live := [depth[x] for x in n.inputs if x not in value]:
            depth.update(dict.fromkeys(n.outputs, 1 + max(live)))
        else:  # a constant
            ys = n.circuit.eval(np.array([value[x] for x in n.inputs]).reshape(-1, 1))
            value.update(zip(n.outputs, ys[:, 0]))
    top = max([1] + [depth[y] for y in g.outputs if y in depth])
    levels = [g.outputs]  # level d - 1 holds what level d reads: its inputs or copies
    for d in range(top, 1, -1):
        reads = {}
        for y in levels[0]:
            if y not in value:
                reads |= dict.fromkeys(y.node.inputs if depth[y] == d else [y])
        levels.insert(0, [x for x in reads if x not in value])
    levels.insert(0, g.inputs)
    h = ins = inputs("bit", *(s.kind for s in g.inputs))
    for d in range(1, top + 1):
        h = emit(_layer(levels[d - 1], levels[d], d, depth, value), h)
    return Graph(f"{graph.name}.layers", ins, h)


def _layer(xs, ys, d, depth, value) -> GLU:
    """The GLU from [BOS, *xs] (level d - 1) to [BOS, *ys] (level d): BOS, and for each
    y its own units if it is at depth d, else a copy, or a constant"""
    col = {x: i + 1 for i, x in enumerate(xs)}

    def read(ins):  # rows picking ins out of [BOS, *xs]; a constant is a multiple of BOS
        S = np.zeros((len(ins), len(xs) + 1))
        for j, x in enumerate(ins):
            S[j, 0 if x in value else col[x]] = value.get(x, 1)
        return S

    bos, one = np.eye(1, len(xs) + 1), np.ones((1, 1))
    parts = [(bos, bos, one)]  # relu(1) * 1
    for y in ys:
        if y in value:
            parts.append((bos, value[y] * bos, one))  # relu(1) * constant
        elif depth[y] == d:
            c, S, i = y.node.circuit, read(y.node.inputs), y.index
            wg, wv = c.wg @ S + np.outer(c.bg, bos), c.wv @ S + np.outer(c.bv, bos)
            parts.append((wg, wv, c.wo[i : i + 1]))
        else:
            parts.append((bos, read([y]), one))  # a copy: relu(1) * y
    wg, wv, wo = zip(*parts)
    units = [len(w) for w in wg]
    wo = np.repeat(np.eye(len(parts)), units, axis=1) * np.hstack(wo)  # block diagonal
    zero = np.zeros(sum(units))
    return GLU(np.vstack(wg), zero, np.vstack(wv), zero, wo, ("bit", *(y.kind for y in ys)))


def to_swiglu(graph: Graph, dialect: str = "relu") -> Graph:
    """graph as SwiGLU layers without biases, whose inputs and outputs begin with BOS = 1"""
    return lower(layers(graph, dialect), "silu")

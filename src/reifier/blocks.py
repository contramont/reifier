"""Blocks, the leaf circuit: f(w @ x + b), a matmul and then a function.

A linear threshold gate is one block with f = step. A GLU is two blocks: one with a gated
f such as swiglu, which multiplies an activation of the left half of w @ x + b by its
right half, then a linear one (f = id). Compiling targets an architecture: an MLP of step
or relu blocks, or GLU layers, with or without RMSNorms.
"""

from dataclasses import dataclass, replace

import numpy as np

from .core import RULES, Circuit, Graph, Signal, emit, flatten, inputs, lower


def gated(act):
    """act of the left half of z, times the right half"""
    return lambda z: act(z[: len(z) // 2]) * z[len(z) // 2 :]


def relu(z):
    return np.maximum(z, 0)


def silu(z):
    return z / 2 * (1 + np.tanh(z / 2))


def gelu(z):
    return z / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


FUNCS = {
    "id": lambda z: z,
    "step": lambda z: (z >= 0) * 1.0,
    "relu": relu,
    "reglu": gated(relu),
    "swiglu": gated(silu),
    "geglu": gated(gelu),
    "rmsnorm": lambda z: z / np.sqrt(np.mean(z * z) + 1e-6),
}
GATED = {"reglu", "swiglu", "geglu"}  # one output per two rows of w


@dataclass(frozen=True, eq=False)
class Block(Circuit):
    """f(w @ x + b): a matmul, then a function"""

    w: np.ndarray
    b: np.ndarray
    f: str
    kinds: tuple[str, ...]

    def eval(self, xs):
        return FUNCS[self.f](self.w @ np.asarray(xs, float) + self.b)

    def __str__(self):
        if self.f == "step" and len(self.b) == 1:  # a gate
            return f"gate[{' '.join(f'{w:g}' for w in self.w[0])} >= {-self.b[0]:g}]"
        return f"{self.f}[{len(self.b)}]"


def block(xs, w, b, f="id", kind="real") -> list[Signal]:
    """f(w @ xs + b) on the signals xs"""
    w, b = np.array(w, float).reshape(len(b), len(xs)), np.array(b, float)
    return emit(Block(w, b, f, (kind,) * (len(b) // 2 if f in GATED else len(b))), xs)


def gate(xs, w, threshold) -> Signal:
    """[w . xs >= threshold]: a linear threshold gate, one step block"""
    if any(x.kind == "real" for x in xs):
        raise TypeError("gates need integer sums: their inputs must be bits or ints")
    return block(xs, [w], [-threshold], "step", "bit")[0]


def const(bits: str) -> list[Signal]:
    return [gate([], [], 1 - int(b)) for b in bits]


def glu(xs, units, kind="bit") -> Signal:
    """The sum of relu(g . xs + gb) * (v . xs + vb) over units (g, gb, v, vb): a reglu
    block, then a linear one"""
    g, gb, v, vb = zip(*units)
    return block(block(xs, [*g, *v], [*gb, *vb], "reglu"), [[1] * len(units)], [0], kind=kind)[0]


def chain(name: str, kinds, blocks) -> Graph:
    """The blocks one after another, each reading all the outputs of the one before"""
    h = ins = inputs(*kinds)
    for c in blocks:
        h = emit(c, h)
    return Graph(name, ins, h)


# ---------------------------------------------------------------- dialects


def step_as(f: str, k: int = 4):
    """A rule: a step block as f units u(kz + k/2 + 1/2) - u(kz + k/2 - 1/2), where u is
    relu, or relu times a value of 1. On integer z it gives exactly 0 or 1, as long as
    the sums are off by less than 1/2 - 1/(2k)"""

    def rule(c: Block) -> Circuit:
        if c.f == "relu" and f == "reglu":  # relu(z) * 1
            w, b = np.vstack([c.w, 0 * c.w]), np.r_[c.b, np.ones(len(c.b))]
            return Block(w, b, f, c.kinds)
        if c.f != "step":
            return c
        m, z = len(c.b), k * c.b + k / 2
        w, b = np.vstack([k * c.w, k * c.w]), np.r_[z + 0.5, z - 0.5]
        if f == "reglu":
            w, b = np.vstack([w, 0 * w]), np.r_[b, np.ones(2 * m)]
        units = Block(w, b, f, ("real",) * 2 * m)
        readout = Block(np.hstack([np.eye(m), -np.eye(m)]), np.zeros(m), "id", c.kinds)
        return chain(f"step.{f}", ("bit",) * c.w.shape[1], [units, readout])

    return rule


def reglu_to_swiglu(c: Block, k: float = 32.0) -> Block:
    """silu(kg) * v / k, which is relu(g) * v to within |gv| exp(-k |g|)"""
    if c.f != "reglu":
        return c
    s = np.r_[np.full(len(c.b) // 2, k), np.full(len(c.b) // 2, 1 / k)]
    return replace(c, w=c.w * s[:, None], b=c.b * s, f="swiglu")


RULES["relu", Block] = step_as("relu")
RULES["reglu", Block] = step_as("reglu")
RULES["swiglu", Block] = reglu_to_swiglu


# ---------------------------------------------------------------- layers


def layers(graph: Graph, f: str) -> Graph:
    """graph as a chain of blocks N, L, N, L, ...: each N applies f to the layer below and
    each L reads N out linearly. Inputs and outputs begin with BOS = 1, which carries the
    biases and constants; copies carry signals to the layers that read them"""
    g = flatten(lower(graph, f))
    units = {}  # each block of f -> the linear block that reads it out, or None
    for n in g.nodes:
        if n.circuit.f == f:
            units[n] = None
        elif n.circuit.f == "id" and n.inputs and units.get(src := n.inputs[0].node, 0) is None:
            if list(n.inputs) != src.outputs or n.circuit.b.any():
                raise TypeError(f"{n.circuit} does not read out {src.circuit} without bias")
            units[src] = n
        else:
            raise TypeError(f"{n.circuit} is not a {f} block or its readout")
    value, depth, where = {}, dict.fromkeys(g.inputs, 0), {}
    for n, r in units.items():
        outs = (r or n).outputs
        where |= {y: (n, r, i) for i, y in enumerate(outs)}
        if live := [depth[x] for x in n.inputs if x not in value]:
            depth |= dict.fromkeys(outs, 1 + max(live))
        else:  # a constant
            ys = n.circuit.eval([value[x] for x in n.inputs])
            value |= zip(outs, r.circuit.eval(ys) if r else ys)
    top = max([1] + [depth[y] for y in g.outputs if y in depth])
    levels = [g.outputs]  # level d - 1 holds what level d reads: its inputs or copies
    for d in range(top, 1, -1):
        reads = {}
        for y in levels[0]:
            if y not in value:
                reads |= dict.fromkeys(where[y][0].inputs if depth[y] == d else [y])
        levels.insert(0, [x for x in reads if x not in value])
    levels.insert(0, g.inputs)
    blocks = []
    for d in range(1, top + 1):
        blocks += _layer(levels[d - 1], levels[d], d, f, depth, value, where)
    return chain(f"{graph.name}.{f}", ("bit", *(s.kind for s in g.inputs)), blocks)


def _layer(xs, ys, d, f, depth, value, where) -> tuple[Block, Block]:
    """N and L from [BOS, *xs] (level d - 1) to [BOS, *ys] (level d): for each y, the
    units of its block if it is at depth d, else a copy, or a constant"""
    col = {x: i + 1 for i, x in enumerate(xs)}

    def read(ins):  # rows picking ins out of [BOS, *xs]; a constant is a multiple of BOS
        S = np.zeros((len(ins), len(xs) + 1))
        for j, x in enumerate(ins):
            S[j, 0 if x in value else col[x]] = value.get(x, 1)
        return S

    bos = np.eye(1, len(xs) + 1)
    parts = [_copy(f, bos)]
    for y in ys:
        if y in value:
            parts.append(_copy(f, value[y] * bos))
        elif depth[y] == d:
            n, r, i = where[y]
            w = n.circuit.w @ read(n.inputs)
            w[:, 0] += n.circuit.b  # biases are multiples of BOS
            u = len(w) // 2 if f in GATED else len(w)
            readout = r.circuit.w[i : i + 1] if r else np.eye(1, u, i)
            parts.append((w[:u], w[u:] if f in GATED else None, readout))
        else:
            parts.append(_copy(f, read([y])))
    gates, values, readouts = zip(*parts)
    w = np.vstack([*gates, *values] if f in GATED else gates)
    sizes = [len(g) for g in gates]
    readout = np.repeat(np.eye(len(parts)), sizes, axis=1) * np.hstack(readouts)  # block diagonal
    n = Block(w, np.zeros(len(w)), f, ("real",) * sum(sizes))
    return n, Block(readout, np.zeros(len(parts)), "id", ("bit", *(y.kind for y in ys)))


def _copy(f, r):
    """Units of f passing on r @ [BOS, *xs], and their readout"""
    bos = np.eye(1, r.shape[1])
    if f == "step":  # bits only
        return r - bos / 2, None, np.ones((1, 1))
    if f == "relu":
        return np.vstack([r, -r]), None, np.array([[1.0, -1.0]])
    return bos, r, np.ones((1, 1))  # gated: relu(BOS) * r


# ---------------------------------------------------------------- architectures


def mlp(graph: Graph, f: str = "relu") -> Graph:
    """An MLP of f blocks and a final linear block: each readout folds into the next block"""
    nodes, blocks = layers(graph, f).nodes, []
    for n in nodes:
        c = n.circuit
        if blocks and blocks[-1].f == "id":
            prev = blocks.pop()
            c = replace(c, w=c.w @ prev.w, b=c.w @ prev.b + c.b)
        blocks.append(c)
    return chain(f"{graph.name}.mlp", ("bit", *(s.kind for s in graph.inputs)), blocks)


def glu_layers(graph: Graph, norm: bool = False) -> Graph:
    """SwiGLU layers, each a swiglu block and a linear block, with an RMSNorm before each
    if norm, as in transformers. With norms, outputs are relative to BOS"""
    blocks = []
    for n in lower(layers(graph, "reglu"), "swiglu").nodes:
        if norm and n.circuit.f == "swiglu":
            d = n.circuit.w.shape[1]
            blocks.append(Block(np.eye(d), np.zeros(d), "rmsnorm", ("real",) * d))
        blocks.append(n.circuit)
    return chain(f"{graph.name}.glu", ("bit", *(s.kind for s in graph.inputs)), blocks)

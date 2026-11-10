"""Stage 3, the weights: a grid emitted as the layers of a network.

Layer d is one MLP on features, with no biases: w_out @ act(w_in @ x) reads x, the stream, the
values of the features, as row d - 1 of the grid, and makes row d. Its hidden units are those of
the trace's MLPs of that depth, under their own names, with their weights on the features that
the grid gives their inputs and outputs and their biases on ONE; and copies, which carry what no
MLP makes. What is around the layers decides the rest:

  a stack:  x = layer(x)        a layer writes all that its row holds: `keep s` copies s
  skip:     x = x + layer(x)    a layer writes only what changes, new - old: `erase s` is -s
  norm:     layer(norm(x))      a bit is held as -1 or 1, not 0 or 1, so that a row's norm is
                                known from the grid, and the RMSNorm is a fixed scaling

Where convert is exact, so is every network: a feature holds 0, 1 or -1, but for ONE. A layer
of a gated stack writes ONE as ONE times ONE, so it needs ONE to be 1 exactly: a stream that
may be a little off needs skip connections or norms.
"""

from dataclasses import dataclass

import numpy as np

from .grid import ONE, Grid, check_grid
from .trace import GATED, MLP, Trace, cases, convert

EPS = 1e-6  # of the RMSNorm


@dataclass
class Network:
    trace: Trace
    grid: Grid  # the name of every feature at every depth
    layers: list[MLP]  # layers[d - 1] makes row d from row d - 1
    skip: bool
    norm: bool
    one: float  # what the stream holds ONE as: 1, or more with skip connections and norms

    def run(self, x) -> list[np.ndarray]:
        """The stream at every depth, from row 0: x[feature], or x[..., feature] for many"""
        xs = [np.asarray(x, float)]
        for layer, row in zip(self.layers, self.grid.rows):
            y = layer.eval(rmsnorm(xs[-1], row, self.one) if self.norm else xs[-1])
            xs.append(xs[-1] + y if self.skip else y)
        return xs

    def row(self, values: dict, d: int) -> np.ndarray:
        """Row d as the grid says it is, for the values {name: value} of its signals: a bit v
        is held as 2 v - 1 with norms, else as v, and a feature with no signal is 0"""
        cells = [
            0 * values[ONE] if s is None else 2 * values[s] - 1 if self.norm else values[s]
            for s in self.grid.rows[d]
        ]
        return np.stack([self.one * values[ONE], *cells[1:]], -1)

    def __call__(self, x) -> list[float]:
        """The outputs on inputs x, a list of bits or a string of them: row 0 is made, the
        layers run, and the outputs are read from the last row"""
        values = dict(zip(self.trace.inputs, (float(c) for c in x), strict=True)) | {ONE: 1.0}
        last = self.run(self.row(values, 0))[-1]
        outs = np.array([last[self.grid.rows[-1].index(y)] for y in self.trace.outputs])
        return ((outs + 1) / 2 if self.norm else outs).tolist()

    def __str__(self):
        """Each hidden unit of each layer: what it reads, by the names of the row before, and
        the features it writes"""
        kind = ("skip" if self.skip else "stack") + " + norm" * self.norm
        lines = [
            f"{self.trace.name}: {kind}, {len(self.grid.rows[0])} features, ONE = {self.one:g}"
        ]
        for d, (layer, row) in enumerate(zip(self.layers, self.grid.rows), 1):
            norm = f", after a norm of gain {gain(row, self.one):.4g}" * self.norm
            lines.append(f"layer {d}: {len(layer.hidden)} hidden units of {layer.activation}{norm}")
            lines += ["  " + line for line in str(layer).splitlines()]
        return "\n".join(lines)


def gain(row: list, one: float = 1.0) -> float:
    """The weight of an RMSNorm that gives back a stream that holds row: its signals as -1 or 1,
    and ONE as `one`"""
    return float(np.sqrt((one**2 + sum(s is not None for s in row[1:])) / len(row) + EPS))


def rmsnorm(x, row: list, one: float = 1.0):
    """The RMSNorm before a layer that reads row: it gives x back, but for ONE, which it gives
    as 1, its weight there being `one` times smaller"""
    z = gain(row, one) * x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)
    z[..., 0] /= one
    return z


def emit(t: Trace, g: Grid, activation: str, skip: bool = False, norm: bool = False) -> Network:
    """The weights that make each row of g from the row before, in layers of activation: a
    stack of them, or with a skip connection around each if skip, and an RMSNorm before each if
    norm. g is checked against t first. With both skip connections and norms, a norm that is
    off by a part e makes a layer add (1 + e)^2 times what it should, as a gated unit multiplies
    two things it reads, and that moves the next norm in turn: e can grow nearly 4 times from
    layer to layer, where every bit flips. So the stream then holds ONE as 2 sqrt(features):
    ONE's square is at least 4/5 of a row's, and e shrinks instead"""
    by_depth = check_grid(t, g)
    layers = [
        layer(by_depth[d], g.rows[d - 1], g.rows[d], activation, skip, norm, f"layer {d}")
        for d in range(1, len(g.rows))
    ]
    return Network(t, g, layers, skip, norm, 2 * len(g.rows[0]) ** 0.5 if skip and norm else 1.0)


def layer(mlps, below, row, activation, skip=False, norm=False, name="layer") -> MLP:
    """One layer: mlps, the MLPs that make the signals new in row, side by side as one MLP that
    reads the row below, with copies for the rest. Its hidden units, in order: ONE, which gives
    1, unless skip connections keep ONE and nothing else needs a 1; those of mlps; then the
    copies, first those for the features that mlps write. The stream x holds a bit v as
    a * v + c, with (a, c) = (1, 0), or (2, -1) with norms. So a unit's sum w . v + b is
    (w / a) . x + (b - c sum(w) / a) ONE, and its output goes out a times as large, with c from
    the unit ONE. A copy reads and writes what a feature holds, as it is"""
    if activation not in ("step", "relu", *GATED):
        raise TypeError(f"layers are of step, relu, {', '.join(GATED)}: not of {activation}")
    if norm and activation not in GATED:
        raise TypeError("norms need a gated activation: its copies pass -1 as well as 1")
    at = {s: f for f, s in enumerate(below) if s}  # the feature each signal is read from
    to = {s: f for f, s in enumerate(row) if s}  # and the feature it is written to
    a, c = (2.0, -1.0) if norm else (1.0, 0.0)
    made = [
        (convert(m, activation), [at[x] for x in m.inputs], [to[y] for y in m.outputs], a, c)
        for m in mlps
    ]  # each part: an MLP, the features it reads and writes, a and c
    written = dict.fromkeys(f for _, _, outs, _, _ in made for f in outs)
    copies = []
    for f in [*written, *(f for f in range(1, len(row)) if f not in written)]:
        old, new = below[f], row[f]
        if skip and old == new:
            continue  # the skip connection keeps it
        if skip and old:  # new - old
            copies.append((copy(activation, f"erase {old}", -1.0), [f], [f], 1.0, 0.0))
        if new and f not in written:  # not made here: carried from the row below
            copies.append((copy(activation, f"keep {new}"), [at[new]], [f], 1.0, 0.0))
    unit = [(copy(activation, ONE), [0], [0], 1.0, 0.0)] * (norm or not skip)
    parts = unit + made + copies
    n, halves = sum(len(m.hidden) for m, *_ in parts), 2 if activation in GATED else 1
    w_in, w_out, h = np.zeros((halves * n, len(below))), np.zeros((len(row), n)), 0
    for m, ins, outs, a, c in parts:
        k = len(m.hidden)
        for half in range(halves):  # the gates of all units, then their values
            rows, src = slice(half * n + h, half * n + h + k), slice(half * k, half * k + k)
            np.add.at(w_in[rows].T, ins, m.wi[src].T / a)  # an input read twice adds up
            w_in[rows, 0] += m.b[src] - c * m.wi[src].sum(1) / a  # biases are weights on ONE
        w_out[outs, h : h + k] = a * m.wo
        if c:
            w_out[outs, 0] += c
        h += k
    if skip and unit:
        w_out[0, 0] = 0  # the skip connection keeps ONE: the unit ONE only gives c
    hidden = tuple(u for m, *_ in parts for u in m.hidden)
    reads = tuple(s or f"f{f}" for f, s in enumerate(below))
    writes = tuple(f"f{f}" for f in range(len(row)))
    return MLP(name, reads, writes, hidden, w_in, np.zeros(len(w_in)), activation, w_out)


def copy(activation: str, name: str, weight: float = 1.0) -> MLP:
    """An MLP of activation that gives weight times the one thing it reads: a bit, as
    [x >= 1/2] or relu(x), if activation is step or relu, and anything, as relu(1) * x, if it
    is gated"""
    if activation in GATED:
        wi, b, wo, act = [[0.0], [weight]], [1.0, 0.0], [[1.0]], "reglu"
    else:
        wi, b, wo, act = [[1.0]], [-0.5 * (activation == "step")], [[weight]], activation
    m = MLP(name, ("x",), ("y",), (name,), np.array(wi), np.array(b), act, np.array(wo))
    return convert(m, activation)


def check_net(net: Network, xs=None, tol: float = 1e-3) -> float:
    """Check net against its grid, on each input of xs (default: cases): every feature at every
    depth holds the value, in the trace, of the signal that the grid names there, and 0 where it
    names none, to within tol. And no hidden units of a layer share a name, and none reads a
    feature that the row before leaves empty. Gives the largest difference"""
    for d, m in enumerate(net.layers, 1):
        assert len(set(m.hidden)) == len(m.hidden), f"layer {d}: two hidden units share a name"
        empty = [f for f, s in enumerate(net.grid.rows[d - 1]) if s is None]
        assert not m.wi[:, empty].any(), (
            f"layer {d} reads a feature that depth {d - 1} leaves empty"
        )
    xs = cases(len(net.trace.inputs)) if xs is None else np.asarray(xs, float)
    values = net.trace.values(xs) | {ONE: np.ones(len(xs))}
    return check_rows(net, net.run(net.row(values, 0)), values, tol)


def check_rows(net: Network, rows, values: dict, tol: float) -> float:
    """rows, a stream at each depth, against what net's grid says for the values {name: value on
    each case} of its signals: the largest difference, which must be within tol"""
    worst = 0.0
    for s, v in values.items():
        assert not net.norm or np.isin(v, (0, 1)).all(), f"{s} is not a bit, as a norm needs"
    for d, x in enumerate(rows):
        off = np.abs(x - net.row(values, d))
        i, f = np.unravel_index(off.argmax(), off.shape)
        assert off[i, f] <= tol, (
            f"depth {d}, feature {f} ({net.grid.rows[d][f] or 'no signal'}) is {x[i, f]:g} on "
            f"case {i}, {off[i, f]:g} from what the grid says"
        )
        worst = max(worst, float(off[i, f]))
    return worst

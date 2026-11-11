"""Stage 3, the weights: a grid emitted as the layers of a network.

Layer d is one MLP on features, with no biases: w_out @ act(w_in @ x). Its x is the stream, the
values of the features, which hold row d - 1 of the grid, and it makes row d. Its hidden units
are those of the trace's MLPs of that depth, under their own names, with their weights on the
features that the grid gives their inputs and outputs and their biases on ONE; and copies, for
what no MLP makes. What is around the layers decides the rest:

  a stack:  x = layer(x)        a layer writes all that its row holds: `keep s` copies s
  skip:     x = x + layer(x)    a layer writes only what changes, new - old: `erase s` is -s
  norm:     layer(norm(x))      a bit is held as -1 or 1, not 0 or 1, so that a row's norm is
                                known from the grid, and the RMSNorm is a fixed scaling

Every network is exact on a stream that holds row 0 exactly. One that is a little off is still
read right, except by two kinds: step networks, as a gate [z >= 0] has no room at 0, and gated
stacks without norms, which copy ONE as ONE squared.
"""

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from .grid import ONE, Grid
from .trace import GATED, MLP, Trace

# A gated unit is act(g) * v. convert makes g this many times larger and v as many times smaller:
# silu and gelu are then relu, to the last bit of a float, wherever g is 0 or at least 1/2 from 0,
# as it is in every unit made here and in lib.py when what they read is bits
SHARPNESS = {"swiglu": 128.0, "geglu": 32.0}
EPS = 1e-6  # under the RMSNorm's square root: a stream of zeros comes out as zeros


def convert(m: MLP, activation: str) -> MLP:
    """m as an MLP of activation: step -> relu -> reglu -> swiglu or geglu, as far as needed"""
    if m.activation == "step" and activation != "step":
        # [z >= 0], for an integer z, is relu(4 z + 5/2) - relu(4 z + 3/2): the units u+ and u-.
        # A z up to 1/4 off leaves both on the same side of 0, 1/2 from it or more: the same bit
        mid = 4 * m.b + 2
        b = np.r_[mid + 0.5, mid - 0.5]
        wo = np.hstack([m.wo, -m.wo])
        hidden = [h + sign for sign in "+-" for h in m.hidden]
        m = replace(m, wi=np.vstack([4 * m.wi] * 2), b=b, activation="relu", wo=wo, hidden=hidden)
    if m.activation == "relu" and activation in GATED:  # relu(z) is relu(z) * 1
        wi, b = np.vstack([m.wi, 0 * m.wi]), np.r_[m.b, np.ones(len(m.b))]
        m = replace(m, wi=wi, b=b, activation="reglu")
    if m.activation == "reglu" and activation in SHARPNESS:  # relu(g) * v is act(s g) * v / s
        s = np.repeat([SHARPNESS[activation], 1 / SHARPNESS[activation]], len(m.hidden))
        m = replace(m, wi=m.wi * s[:, None], b=m.b * s, activation=activation)
    assert m.activation == activation, f"{m.name} has no {activation} form"
    return m


def copy(activation: str, name: str, weight: float = 1.0) -> MLP:
    """An MLP of activation whose one unit, name, gives weight times the one thing it reads: a
    bit, as [x >= 1/2] or relu(x), if activation is step or relu, and anything, as relu(1) * x,
    if it is gated"""
    if activation in GATED:
        wi, b, act = [[0.0], [1.0]], [1.0, 0.0], "reglu"
    else:
        wi, b, act = [[1.0]], [-0.5 * (activation == "step")], activation
    m = MLP(name, ["x"], ["y"], [name], np.array(wi), np.array(b), act, np.array([[weight]]))
    return convert(m, activation)


def signed(m: MLP) -> MLP:
    """m on bits held as -1 or 1: it reads 2 v - 1 for each bit v, and gives twice each bit it
    makes, which is that bit as held, plus 1"""
    return replace(m, wi=m.wi / 2, b=m.b + m.wi.sum(1) / 2, wo=2 * m.wo)


def layer(mlps, before, row, activation, skip, norm) -> MLP:
    """One layer: the MLP on features that makes row from the row before. Its hidden units, in
    order: ONE, if a stack writes it again or, with norms, each new bit needs its -1 (see signed);
    those of the MLPs of mlps that make the signals new in row; then the copies, by feature"""
    at = {s: f for f, s in enumerate(before) if s}  # the feature each signal is read from
    new = {s: f for f, s in enumerate(row) if s and s not in at}  # the feature each is made in
    parts = []  # each part: an MLP, the features it reads, the features it writes
    if norm or not skip:  # the unit ONE gives 1: to ONE, unless a skip connection keeps it
        parts.append((copy(activation, ONE, float(not skip)), [0], [0]))
    for m in mlps:
        if m.outputs[0] in new:  # an MLP is at the depth where its outputs first appear
            ins, outs = [at[x] for x in m.inputs], [new[y] for y in m.outputs]
            m = convert(m, activation)
            parts.append((signed(m) if norm else m, ins, outs))
    for f in range(1, len(row)):
        if skip and before[f] and before[f] != row[f]:  # a skip connection would keep it
            parts.append((copy(activation, f"erase {before[f]}", -1.0), [f], [f]))
        if not skip and row[f] in at:  # and a stack would lose it
            parts.append((copy(activation, f"keep {row[f]}"), [f], [f]))
    halves = 2 if activation in GATED else 1
    w_in, w_out = [[np.empty((0, len(before)))] * halves], [np.empty((len(row), 0))]  # no units yet
    for m, ins, outs in parts:  # then each part's weights, on features
        wi, wo = np.zeros((len(m.wi), len(before))), np.zeros((len(row), len(m.hidden)))
        for f, weights in zip(ins, m.wi.T):
            wi[:, f] += weights  # an input read twice adds up
        wi[:, 0] += m.b  # biases are weights on ONE
        wo[outs] = m.wo
        w_in.append(np.split(wi, halves))
        w_out.append(wo)
    w_in = np.vstack([w[half] for half in range(halves) for w in w_in])  # all gates, then values
    w_out = np.hstack(w_out)
    if norm:
        w_out[list(new.values()), 0] = -1  # the unit ONE, the first, gives each new bit its -1
    reads = [s or f"f{f}" for f, s in enumerate(before)]
    writes = [f"f{f}" for f in range(len(row))]
    hidden = [h for m, _, _ in parts for h in m.hidden]
    return MLP("layer", reads, writes, hidden, w_in, np.zeros(len(w_in)), activation, w_out)


def rmsnorm(x, row, one):
    """The RMSNorm before a layer that reads row: x over its root mean square, times a weight.
    The weight is the root mean square of a stream that holds row, with ONE as `one` and every
    other signal as -1 or 1, so such a stream comes out as it went in: but for ONE, whose weight
    is `one` times smaller, so that the layer reads it as 1"""
    rms = np.sqrt((one**2 + sum(s is not None for s in row[1:])) / len(row) + EPS)
    z = rms * x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)
    z[..., 0] /= one
    return z


@dataclass
class Network:
    trace: Trace
    grid: Grid  # the name of every feature at every depth
    layers: list[MLP]  # layers[d - 1] makes row d from row d - 1
    skip: bool
    norm: bool
    one: float  # what the stream holds ONE as: 1, or more with skip and norm (see emit)

    def stream(self, values: dict, d: int) -> np.ndarray:
        """What the stream should hold at depth d, for the values {name: value} of the signals
        of row d: a bit v as 2 v - 1 with norms, else as v, and 0 where the row has no signal"""

        def cell(s):
            if s is None:
                return 0.0
            if s == ONE:
                return self.one
            return 2 * values[s] - 1 if self.norm else values[s]

        return np.stack(np.broadcast_arrays(*map(cell, self.grid[d])), -1)

    def run(self, x) -> list[np.ndarray]:
        """The stream at every depth, from the stream x at depth 0: x[..., feature]"""
        xs = [np.asarray(x, float)]
        for layer, before in zip(self.layers, self.grid):  # a layer reads the row before its own
            y = layer(rmsnorm(xs[-1], before, self.one) if self.norm else xs[-1])
            xs.append(xs[-1] + y if self.skip else y)
        return xs

    def __call__(self, x) -> list[float]:
        """The outputs on inputs x, a list of bits or a string of them"""
        last = self.run(self.stream(dict(zip(self.trace.inputs, map(float, x))), 0))[-1]
        outs = np.array([last[self.grid[-1].index(y)] for y in self.trace.outputs])
        return ((outs + 1) / 2 if self.norm else outs).tolist()


def emit(t: Trace, grid: Grid, activation: str, skip=False, norm=False) -> Network:
    """A layer of activation for each depth of grid after the first. skip puts a skip connection
    around each layer, and norm an RMSNorm before each, which needs a gated activation: its
    copies pass -1 as well as 1"""
    layers = [layer(t.mlps, before, row, activation, skip, norm) for before, row in pairwise(grid)]
    # With skip connections and norms, a norm that is a little off makes its layer add a little
    # too much, which the next norm sees: what is off can grow nearly 4 times a layer. So the
    # stream holds ONE as 2 sqrt(features), most of the norm of any row, and it shrinks instead.
    one = 2 * len(grid[0]) ** 0.5 if skip and norm else 1.0
    return Network(t, grid, layers, skip, norm, one)

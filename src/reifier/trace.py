"""Stage 1, the trace: a Python function on signals, run once, leaves a flat list of MLPs.

A signal is a bit on a wire: while tracing it has a name and no value. The only operation is the
MLP, wo @ act(wi @ x + b): a threshold gate is an MLP of activation step, and a GLU an MLP of a
gated activation, such as reglu, whose hidden unit is relu(g) * v: its gate g is in the left half
of wi @ x + b, and its value v in the right half. Every MLP, signal and hidden unit is named by
the circuits that made it:

  add/xor.1/odd    the gate odd, of the second xor that add calls: circuit/circuit.k/name
  name             an MLP, its one output and its one hidden unit
  name:i, name#j   output i and hidden unit j of an MLP that has several
"""

import functools
import inspect
import math
from dataclasses import dataclass

import numpy as np


def gated(act):
    """act of the left half of z, times the right half"""
    return lambda z: act(z[..., : z.shape[-1] // 2]) * z[..., z.shape[-1] // 2 :]


ACTIVATIONS = {
    "step": lambda z: (z >= 0) * 1.0,
    "relu": lambda z: np.maximum(z, 0),
    "silu": lambda z: z / 2 * (1 + np.tanh(z / 2)),  # z sigmoid(z), with no exp to overflow
    "gelu": lambda z: z / 2 * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z**3))),
}
GATED = {"reglu": "relu", "swiglu": "silu", "geglu": "gelu"}  # gated activations: what gates them
ACTIVATIONS |= {name: gated(ACTIVATIONS[act]) for name, act in GATED.items()}


@dataclass(frozen=True, eq=False)
class MLP:
    """outputs = wo @ act(wi @ inputs + b), with a name for itself and for each hidden unit. Of
    its n hidden units, unit j is row j of wi (and, if act is gated, row n + j, the value that
    its gate multiplies) and column j of wo"""

    name: str
    inputs: list[str]
    outputs: list[str]
    hidden: list[str]
    wi: np.ndarray
    b: np.ndarray
    activation: str
    wo: np.ndarray

    def __call__(self, xs) -> np.ndarray:
        """The outputs for inputs xs, or for many at once: xs[..., input]"""
        return ACTIVATIONS[self.activation](np.asarray(xs, float) @ self.wi.T + self.b) @ self.wo.T


class Signal:
    """A wire while tracing: it has a name, and never a value"""

    def __init__(self, name: str):
        self.name = name

    def __bool__(self):
        raise TypeError("signals have no value: branch on structure, not on bits")


_mlps: list[MLP] = []  # the MLPs made so far
_scopes: list[str] = []  # the circuit being run, and those that called it
_uses: dict[str, int] = {}  # how many times each name was given so far


def mlp(xs, wi, b, activation, wo, name="mlp") -> list[Signal]:
    """wo @ activation(wi @ xs + b) on the signals xs: a new MLP, and the signals it gives"""
    name = _named(name)
    b, wo = np.array(b, float), np.atleast_2d(np.array(wo, float))
    wi = np.array(wi, float).reshape(len(b), len(xs))  # a matrix, even of no rows or no columns
    n = len(b) // 2 if activation in GATED else len(b)
    hidden = [name] if n == 1 else [f"{name}#{j}" for j in range(n)]
    outputs = [name] if len(wo) == 1 else [f"{name}:{i}" for i in range(len(wo))]
    _mlps.append(MLP(name, [x.name for x in xs], outputs, hidden, wi, b, activation, wo))
    return [Signal(y) for y in outputs]


def threshold_gate(xs, weights, threshold, name="gate") -> Signal:
    """[weights . xs >= threshold]: a linear threshold gate, of integer weights and threshold"""
    return mlp(xs, [weights], [-threshold], "step", [[1]], name)[0]


def const(values: str) -> list[Signal]:
    """Constant bits: gates [0 >= 1 - v] on no inputs"""
    return [threshold_gate([], [], 1 - int(v), f"const{v}") for v in values]


def glu(xs, hidden_units, name="glu") -> Signal:
    """The sum of relu(g . xs + gb) * (v . xs + vb) over hidden_units (g, gb, v, vb)"""
    g, gb, v, vb = zip(*hidden_units, strict=True) if hidden_units else [()] * 4
    return mlp(xs, [*g, *v], [*gb, *vb], "reglu", [[1] * len(hidden_units)], name)[0]


def _named(name: str) -> str:
    """name in the circuit being run, numbered from its second use there: xor, xor.1, ..."""
    path = f"{_scopes[-1]}/{name}" if _scopes else name
    k = _uses.get(path, 0)
    _uses[path] = k + 1
    return f"{path}.{k}" if k else path


def circuit(fn):
    """fn, with what each call makes named under fn's name"""

    @functools.wraps(fn)
    def named(*args, **kwargs):
        _scopes.append(_named(fn.__name__))
        try:
            return fn(*args, **kwargs)
        finally:
            _scopes.pop()

    return named


@dataclass
class Trace:
    """What a function computes: its inputs, its MLPs in the order they were made (each after
    those it reads), and its outputs, all by name"""

    name: str
    inputs: list[str]
    outputs: list[str]
    mlps: list[MLP]

    def values(self, xs) -> dict[str, np.ndarray]:
        """The value of every signal, on each of the inputs xs[case, input]"""
        xs = np.asarray(xs, float)
        v = dict(zip(self.inputs, xs.T))
        for m in self.mlps:  # ins[input, case]: a matrix, even when m has no inputs
            ins = np.array([v[x] for x in m.inputs]).reshape(len(m.inputs), len(xs))
            v.update(zip(m.outputs, m(ins.T).T))
        return v

    def __call__(self, x) -> list[float]:
        """The outputs on inputs x, a list of bits or a string of them"""
        v = self.values([[int(c) for c in x]])
        return [float(v[y][0]) for y in self.outputs]


def trace(fn, *sizes: int) -> Trace:
    """Run fn on a list of signals for each of sizes, named after fn's parameters. Gives the
    MLPs that it makes and that its outputs depend on"""
    for state in (_mlps, _scopes, _uses):  # a trace is of one function, run once
        state.clear()
    names = zip(inspect.signature(fn).parameters, sizes)
    args = [[Signal(f"{p}{i}") for i in range(n)] for p, n in names]
    inputs = [x.name for arg in args for x in arg]
    outputs = fn(*args)  # a signal, or a list of them
    outputs = [y.name for y in ([outputs] if isinstance(outputs, Signal) else outputs)]
    return Trace(fn.__name__.strip("<>"), inputs, outputs, needed(_mlps, outputs))  # <lambda>


def needed(mlps: list[MLP], outputs: list[str]) -> list[MLP]:
    """The MLPs that outputs depend on, in the order they were made"""
    read, kept = set(outputs), []
    for m in reversed(mlps):
        if read.intersection(m.outputs):
            kept.append(m)
            read.update(m.inputs)
    return kept[::-1]

"""Stage 1, the trace: a Python function on signals, run once, leaves a flat list of MLPs.

A signal is a bit on a wire: while tracing it has a name and no value. The only operation is the
MLP, wo @ act(wi @ x + b): a threshold gate is an MLP of activation step, a GLU an MLP of a gated
activation, such as reglu, which multiplies relu of the left half of wi @ x + b by its right
half. scope names what is made inside it, so every MLP, signal and hidden unit is named by the
circuits that made it: add/xor.1/odd is the gate odd of the second xor of add. An MLP's one
output and its one hidden unit have its name; of several, output i is name:i and hidden unit j
name#j. Names must differ, so a name that is given should use none of these marks.
"""

import inspect
import math
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np


def gated(act):
    """act of the left half of z, times the right half"""
    return lambda z: act(z[..., : z.shape[-1] // 2]) * z[..., z.shape[-1] // 2 :]


ACTIVATIONS = {
    "step": lambda z: (z >= 0) * 1.0,
    "relu": lambda z: np.maximum(z, 0),
    "silu": lambda z: z / 2 * (1 + np.tanh(z / 2)),
    "gelu": lambda z: z / 2 * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z**3))),
}
GATED = {"reglu": "relu", "swiglu": "silu", "geglu": "gelu"}  # gated activations: what gates them
ACTIVATIONS |= {name: gated(ACTIVATIONS[act]) for name, act in GATED.items()}
SHARPNESS = {"swiglu": 128.0, "geglu": 32.0}  # how much sharper than relu's their gates must be


class Signal:
    """A wire while tracing: it has a name, never a value, and belongs to the trace being made"""

    def __init__(self, name: str):
        self.name, self.trace = name, _mlps

    def __bool__(self):
        raise TypeError("signals have no value: branch on structure, not on bits")


@dataclass(frozen=True, eq=False)
class MLP:
    """outputs = wo @ act(wi @ inputs + b), with a name for itself and for each hidden unit. Of
    its n hidden units, unit j is row j of wi (and, if act is gated, row n + j, the value that
    its gate multiplies) and column j of wo"""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    hidden: tuple[str, ...]
    wi: np.ndarray
    b: np.ndarray
    activation: str
    wo: np.ndarray

    def eval(self, xs) -> np.ndarray:
        """The outputs for inputs xs, or for many at once: xs[..., input]"""
        return ACTIVATIONS[self.activation](np.asarray(xs, float) @ self.wi.T + self.b) @ self.wo.T

    def __str__(self):
        """A line for each hidden unit: what it reads, and what reads it"""
        n, act = len(self.hidden), GATED.get(self.activation, self.activation)
        lines = []
        for j, h in enumerate(self.hidden):
            unit = f"{act}({linear(self.wi[j], self.inputs, self.b[j])})"
            if self.activation in GATED:
                unit += f" * ({linear(self.wi[n + j], self.inputs, self.b[n + j])})"
            lines.append(f"{h} = {unit} -> {linear(self.wo[:, j], self.outputs)}")
        return "\n".join(lines)


def linear(weights, names, bias=0.0) -> str:
    """A weighted sum as text: 4 a0 - a1 + 0.5"""
    text = ""
    for w, n in [(w, n) for w, n in zip(weights, names) if w] + [(bias, "")] * bool(bias):
        size = "" if abs(w) == 1 and n else f"{abs(w):g}"
        text += (" - " if w < 0 else " + ") + " ".join(filter(None, [size, n]))
    return ("-" + text[3:] if text.startswith(" - ") else text[3:]) or "0"


@dataclass
class Trace:
    """What a function computes: its inputs, its MLPs in the order they were made (each after
    those it reads), and its outputs, all by name"""

    name: str
    inputs: list[str]
    outputs: list[str]
    mlps: list[MLP]

    def values(self, x) -> dict[str, np.ndarray]:
        """The value of every signal on inputs x, or on many at once: x[..., input]"""
        x = np.asarray(x, float)
        v = dict(zip(self.inputs, np.moveaxis(x, -1, 0), strict=True))
        for m in self.mlps:
            xs = np.array([v[s] for s in m.inputs]).reshape(len(m.inputs), *x.shape[:-1])
            v.update(zip(m.outputs, np.moveaxis(m.eval(np.moveaxis(xs, 0, -1)), -1, 0)))
        return v

    def __call__(self, x) -> list[float]:
        """The outputs on inputs x, a list of bits or a string of them"""
        v = self.values([int(c) for c in x])
        return [float(v[y]) for y in self.outputs]

    def __str__(self):
        lines = [f"{self.name}({' '.join(self.inputs)}): {len(self.mlps)} MLPs"]
        for m in self.mlps:
            if m.activation == "step" and m.wo.tolist() == [[1]]:  # a threshold gate
                lines.append(f"  {m.name} = [{linear(m.wi[0], m.inputs)} >= {0 - m.b[0]:g}]")
            else:
                lines.append(f"  {m.name}: {m.activation}, {len(m.hidden)} hidden")
                lines += ["    " + line for line in str(m).splitlines()]
        return "\n".join([*lines, f"  return {' '.join(self.outputs)}"])


# ---------------------------------------------------------------- tracing

_mlps: list[MLP] = []  # the MLPs made so far
_scopes: list[str] = []  # the scope being named in, and those around it
_uses: dict[str, int] = {}  # how many times each name was given


def _named(name: str) -> str:
    """name in the scope being named in, numbered from its second use there: xor, xor.1, ..."""
    path = "/".join([*_scopes[-1:], name])
    k = _uses[path] = _uses.get(path, -1) + 1
    return f"{path}.{k}" if k else path


@contextmanager
def scope(name: str):
    """What is made inside is named under name: `with scope("round3"):`, or as a decorator"""
    _scopes.append(_named(name))
    try:
        yield
    finally:
        _scopes.pop()


def circuit(fn):
    """fn, with what each call makes named under fn's name"""
    return scope(fn.__name__.strip("_"))(fn)


def trace(fn, *sizes: int) -> Trace:
    """Run fn on a list of signals for each of sizes, named after fn's parameters. Gives the
    MLPs that it makes and its outputs depend on"""
    global _mlps, _scopes, _uses
    saved, _mlps, _scopes, _uses = (_mlps, _scopes, _uses), [], [], {}
    try:
        names = list(inspect.signature(fn).parameters)[: len(sizes)]
        args = [[Signal(f"{p}{i}") for i in range(n)] for p, n in zip(names, sizes, strict=True)]
        inputs = [x.name for arg in args for x in arg]
        outputs, mlps = flat(fn(*args)), _mlps
    finally:
        _mlps, _scopes, _uses = saved
    name = getattr(fn, "__name__", "trace").strip("_<>")
    if any(y.trace is not mlps for y in outputs):
        raise ValueError(f"{name} gives a signal of another trace")
    made = Counter([*inputs, *(y for m in mlps for y in m.outputs)])
    if twice := sorted(y for y, n in made.items() if n > 1):
        raise ValueError(f"signals share a name: {twice}")
    outputs = [y.name for y in outputs]
    if stray := {x for m in mlps for x in m.inputs}.union(outputs) - made.keys():
        raise ValueError(f"{name} reads {sorted(stray)}, which it did not make")
    return Trace(name, inputs, outputs, needed(mlps, outputs))


def flat(tree) -> list[Signal]:
    """The signals in nested lists and tuples"""
    return [tree] if isinstance(tree, Signal) else [s for part in tree for s in flat(part)]


def needed(mlps: list[MLP], outputs: list[str]) -> list[MLP]:
    """The MLPs that outputs depend on, in the order they were made"""
    read, kept = set(outputs), []
    for m in reversed(mlps):
        if read.intersection(m.outputs):
            kept.append(m)
            read.update(m.inputs)
    return kept[::-1]


# ---------------------------------------------------------------- MLPs


def mlp(xs, wi, b, activation, wo, name="mlp") -> list[Signal]:
    """wo @ activation(wi @ xs + b) on the signals xs: a new MLP, and the signals it gives"""
    b, wo = np.array(b, float), np.atleast_2d(np.array(wo, float))
    wi, name = np.array(wi, float).reshape(len(b), len(xs)), _named(name)
    if any(x.trace is not _mlps for x in xs):
        raise ValueError(f"{name} reads a signal of another trace")
    n = len(b) // 2 if activation in GATED else len(b)
    hidden = (name,) if n == 1 else tuple(f"{name}#{j}" for j in range(n))
    outputs = (name,) if len(wo) == 1 else tuple(f"{name}:{i}" for i in range(len(wo)))
    _mlps.append(MLP(name, tuple(x.name for x in xs), outputs, hidden, wi, b, activation, wo))
    return [Signal(y) for y in outputs]


def threshold_gate(xs, weights, threshold, name="gate") -> Signal:
    """[weights . xs >= threshold]: a linear threshold gate, of whole weights and threshold"""
    return mlp(xs, [weights], [-threshold], "step", [[1]], name)[0]


def const(values: str) -> list[Signal]:
    """Constant bits: gates [0 >= 1 - v] on no inputs"""
    return [threshold_gate([], [], 1 - int(v), f"const{v}") for v in values]


def glu(xs, hidden_units, name="glu") -> Signal:
    """The sum of relu(g . xs + gb) * (v . xs + vb) over hidden_units (g, gb, v, vb)"""
    g, gb, v, vb = zip(*hidden_units)
    return mlp(xs, [*g, *v], [*gb, *vb], "reglu", [[1] * len(hidden_units)], name)[0]


def convert(m: MLP, activation: str) -> MLP:
    """m as an MLP of activation, going step -> relu -> reglu -> swiglu or geglu as far as
    needed. A step unit u on a whole sum z is relu(4 z + 5/2) - relu(4 z + 3/2), the relu units
    u+ and u-: exactly 0 or 1 while sums are off by less than 1/4. relu(z) is relu(z) * 1. And
    relu(g) * v is act(s g) * v / s for s = SHARPNESS[activation], exactly so in floats where g
    is 0 or at least 1/2 from it, as every gate of a unit made here, or in lib.py, is on bits"""
    was = m.activation
    if m.activation == "step" and activation != "step":
        mid, hidden = 4 * m.b + 2, tuple(h + sign for sign in "+-" for h in m.hidden)
        b = np.r_[mid + 0.5, mid - 0.5]
        m = replace(m, wi=np.vstack([4 * m.wi] * 2), b=b, activation="relu")
        m = replace(m, wo=np.hstack([m.wo, -m.wo]), hidden=hidden)
    if m.activation == "relu" and activation in GATED:
        wi, b = np.vstack([m.wi, 0 * m.wi]), np.r_[m.b, np.ones(len(m.b))]
        m = replace(m, wi=wi, b=b, activation="reglu")
    if m.activation == "reglu" and activation in SHARPNESS:
        n, sharp = len(m.hidden), SHARPNESS[activation]
        s = np.r_[np.full(n, sharp), np.full(n, 1 / sharp)]
        m = replace(m, wi=m.wi * s[:, None], b=m.b * s, activation=activation)
    if m.activation != activation:
        raise TypeError(f"{m.name} is a {was} MLP: it has no {activation} form")
    return m


# ---------------------------------------------------------------- the check


def cases(n: int) -> np.ndarray:
    """Inputs of n bits to check on: all of them if n <= 8, else 256 random ones"""
    if n <= 8:
        return np.array([[i >> j & 1 for j in range(n)] for i in range(2**n)], float)
    return np.random.default_rng(0).integers(0, 2, (256, n)).astype(float)


def check_trace(t: Trace, reference, xs=None, tol: float = 0.0) -> int:
    """Check t against what it should compute, on each input of xs (default: cases). reference
    is a Python function from a list of bits to the outputs, or another trace, such as t before
    a pass: then every signal they share must agree too. Gives how many inputs it tried"""
    xs = cases(len(t.inputs)) if xs is None else np.asarray(xs, float)
    got, pairs = t.values(xs), []  # pairs: a name, its values in t, the values it should have
    if isinstance(reference, Trace):
        want = reference.values(xs)
        pairs = [(s, got[s], want[s]) for s in got if s in want]  # the first to differ, first
        outs = [want[y] for y in reference.outputs]
    else:
        outs = np.array([reference([int(v) for v in x]) for x in xs], float).T
    pairs += [(y, got[y], out) for y, out in zip(t.outputs, outs, strict=True)]
    for s, value, should in pairs:
        i = np.abs(value - should).argmax()
        if not abs(value[i] - should[i]) <= tol:
            x = "".join(str(int(v)) for v in xs[i])
            raise AssertionError(f"{s} is {value[i]:g} on {x}, not {should[i]:g}")
    return len(xs)

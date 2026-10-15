"""Units, the leaf circuit: wo @ f(wi @ x + b), linear, a function, linear.

A linear threshold gate is a unit with f = step. A GLU is a unit with a gated f, such
as reglu, which multiplies relu of the left half of wi @ x + b by its right half. Units
are local: compiling places them in layers as they are, side by side.
"""

from dataclasses import dataclass, replace

import numpy as np

from .core import RULES, Circuit, Signal, emit


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
GATED = {"reglu", "swiglu", "geglu"}  # one hidden value per two rows of wi


@dataclass(frozen=True, eq=False)
class Unit(Circuit):
    """wo @ f(wi @ x + b): linear, a function, linear"""

    wi: np.ndarray
    b: np.ndarray
    f: str
    wo: np.ndarray
    kinds: tuple[str, ...]
    label: str = ""  # a name to show, such as the circuit the unit came from

    @property
    def hidden(self) -> int:
        return len(self.b) // 2 if self.f in GATED else len(self.b)

    def eval(self, xs):
        return (self.wo @ FUNCS[self.f](self.wi @ np.asarray(xs, float) + self.b)).tolist()

    def __str__(self):
        if self.label:
            return self.label
        if self.f == "step" and self.wo.shape == (1, 1):  # a gate
            return f"gate[{' '.join(f'{w:g}' for w in self.wi[0])} >= {-self.b[0]:g}]"
        return f"{self.f}[{self.hidden}]"


def unit(xs, wi, b, f, wo, kind="real") -> list[Signal]:
    """wo @ f(wi @ xs + b) on the signals xs"""
    b, wo = np.array(b, float), np.atleast_2d(np.array(wo, float))
    return emit(Unit(np.array(wi, float).reshape(len(b), len(xs)), b, f, wo, (kind,) * len(wo)), xs)


def gate(xs, w, threshold) -> Signal:
    """[w . xs >= threshold]: a linear threshold gate"""
    if any(x.kind == "real" for x in xs):
        raise TypeError("gates need integer sums: their inputs must be bits or ints")
    return unit(xs, [w], [-threshold], "step", [[1]], "bit")[0]


def const(bits: str) -> list[Signal]:
    """Constant bits: units step(b - 1) on no inputs (integer sums, so casts are exact)"""
    one = np.ones((1, 1))
    return [
        emit(Unit(one[:, :0], one[0] * int(b) - 1, "step", one, ("bit",), b), [])[0] for b in bits
    ]


def glu(xs, units, kind="bit") -> Signal:
    """The sum of relu(g . xs + gb) * (v . xs + vb) over units (g, gb, v, vb)"""
    g, gb, v, vb = zip(*units)
    return unit(xs, [*g, *v], [*gb, *vb], "reglu", [[1] * len(units)], kind)[0]


def cast(c: Unit, f: str, k: float = 4.0, sharp: float = 32.0) -> Unit:
    """c as a unit of f, going step -> relu -> reglu -> swiglu or geglu as far as needed.
    step(z) on integer z is relu(kz + k/2 + 1/2) - relu(kz + k/2 - 1/2): exactly 0 or 1
    while sums are off by less than 1/2 - 1/(2k). relu(z) is relu(z) * 1. relu(g) * v is
    act(sharp g) * v / sharp to within |gv| exp(-sharp |g|)"""
    if c.f == "step" and f != "step":
        z = k * c.b + k / 2
        c = replace(c, wi=np.vstack([k * c.wi] * 2), b=np.r_[z + 0.5, z - 0.5], f="relu")
        c = replace(c, wo=np.hstack([c.wo, -c.wo]))
    if c.f == "relu" and f in GATED:
        c = replace(c, wi=np.vstack([c.wi, 0 * c.wi]), b=np.r_[c.b, np.ones(len(c.b))], f="reglu")
    if c.f == "reglu" and f in GATED - {"reglu"}:
        s = np.r_[np.full(c.hidden, sharp), np.full(c.hidden, 1 / sharp)]
        c = replace(c, wi=c.wi * s[:, None], b=c.b * s, f=f)
    return c


for _f in ["relu", *GATED]:
    RULES[_f, Unit] = lambda c, f=_f: cast(c, f)

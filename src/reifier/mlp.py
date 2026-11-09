"""MLPs, the leaf circuit that computes: wo @ act(wi @ x + b), linear, an activation, linear.

A linear threshold gate is an MLP with activation step. A GLU is an MLP with a gated
activation, such as reglu, which multiplies relu of the left half of wi @ x + b by its right
half. MLPs are local: an MLP reads its inputs at its own position, and compiling places it in a
layer as it is, beside others.
"""

from dataclasses import dataclass, replace

import numpy as np

from .core import FALLBACKS, LOWERING_RULES, Circuit, Signal, apply


def gated(act):
    """act of the left half of z, times the right half"""
    return lambda z: act(z[: len(z) // 2]) * z[len(z) // 2 :]


def relu(z):
    return np.maximum(z, 0)


def silu(z):
    return z / 2 * (1 + np.tanh(z / 2))


def gelu(z):
    return z / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


STEP_TOLERANCE = 1e-9  # sums of bits are integers, but a norm before a step rounds them

ACTIVATIONS = {
    "step": lambda z: (z >= -STEP_TOLERANCE) * 1.0,
    "relu": relu,
    "reglu": gated(relu),
    "swiglu": gated(silu),
    "geglu": gated(gelu),
}
GATED_ACTIVATIONS = {"reglu", "swiglu", "geglu"}  # one hidden value per two rows of wi


@dataclass(frozen=True, eq=False)
class MLP(Circuit):
    """wo @ act(wi @ x + b): linear, an activation, linear"""

    wi: np.ndarray
    b: np.ndarray
    activation: str
    wo: np.ndarray
    output_kinds: tuple[str, ...]
    name: str = ""  # a name to show, such as the circuit the MLP came from

    @property
    def n_hidden(self) -> int:
        return len(self.b) // 2 if self.activation in GATED_ACTIVATIONS else len(self.b)

    def eval(self, x):
        x = np.asarray(x, float)
        x = x[:, None] if x.ndim == 1 else x  # a column per position
        return self.wo @ ACTIVATIONS[self.activation](self.wi @ x + self.b[:, None])

    def __str__(self):
        if self.name:
            return self.name
        if self.activation == "step" and self.wo.shape == (1, 1):  # a threshold gate
            return f"gate[{' '.join(f'{w:g}' for w in self.wi[0])} >= {-self.b[0]:g}]"
        return f"{self.activation}[{self.n_hidden}]"


def mlp(xs, wi, b, activation, wo, kind="real") -> list[Signal]:
    """wo @ activation(wi @ xs + b) on the signals xs"""
    b, wo = np.array(b, float), np.atleast_2d(np.array(wo, float))
    wi = np.array(wi, float).reshape(len(b), len(xs))
    return apply(MLP(wi, b, activation, wo, (kind,) * len(wo)), xs)


def threshold_gate(xs, weights, threshold) -> Signal:
    """[weights . xs >= threshold]: a linear threshold gate"""
    if any(x.kind == "real" for x in xs):
        raise TypeError("gates need integer sums: their inputs must be bits or ints")
    return mlp(xs, [weights], [-threshold], "step", [[1]], "bit")[0]


def const(values: str) -> list[Signal]:
    """Constant bits: MLPs step(v - 1) on no inputs (integer sums, so conversions are exact)"""
    one = np.ones((1, 1))
    return [
        apply(MLP(one[:, :0], one[0] * int(v) - 1, "step", one, ("bit",), v), [])[0] for v in values
    ]


def glu(xs, hidden_units, kind="bit") -> Signal:
    """The sum of relu(g . xs + gb) * (v . xs + vb) over hidden_units (g, gb, v, vb)"""
    g, gb, v, vb = zip(*hidden_units)
    return mlp(xs, [*g, *v], [*gb, *vb], "reglu", [[1] * len(hidden_units)], kind)[0]


def convert(c: MLP, activation: str, step_scale: float = 4.0, sharpness: float = 32.0) -> MLP:
    """c as an MLP of activation, going step -> relu -> reglu -> swiglu or geglu as far as
    needed. step(z) on integer z is relu(kz + k/2 + 1/2) - relu(kz + k/2 - 1/2) for k =
    step_scale: exactly 0 or 1 while sums are off by less than 1/2 - 1/(2k). relu(z) is
    relu(z) * 1. relu(g) * v is act(sharpness g) * v / sharpness to within
    |gv| exp(-sharpness |g|)"""
    k = step_scale
    if c.activation == "step" and activation != "step":
        z = k * c.b + k / 2
        c = replace(c, wi=np.vstack([k * c.wi] * 2), b=np.r_[z + 0.5, z - 0.5], activation="relu")
        c = replace(c, wo=np.hstack([c.wo, -c.wo]))
    if c.activation == "relu" and activation in GATED_ACTIVATIONS:
        wi, b = np.vstack([c.wi, 0 * c.wi]), np.r_[c.b, np.ones(len(c.b))]
        c = replace(c, wi=wi, b=b, activation="reglu")
    if c.activation == "reglu" and activation in GATED_ACTIVATIONS - {"reglu"}:
        s = np.r_[np.full(c.n_hidden, sharpness), np.full(c.n_hidden, 1 / sharpness)]
        c = replace(c, wi=c.wi * s[:, None], b=c.b * s, activation=activation)
    return c


def passthrough(activation: str, kind: str = "bit") -> MLP:
    """An MLP of activation that passes on its input: step(x - 1/2) on bits, relu(x) - relu(-x),
    or relu(1) x, converted for the other gated activations"""
    one, name = np.ones((1, 1)), "passthrough"
    if activation == "step":
        return MLP(one, np.array([-0.5]), "step", one, (kind,), name)
    if activation == "relu":
        return MLP(np.r_[one, -one], np.zeros(2), "relu", np.array([[1.0, -1.0]]), (kind,), name)
    return convert(
        MLP(np.r_[0 * one, one], np.array([1.0, 0]), "reglu", one, (kind,), name), activation
    )


for _activation in ["relu", *GATED_ACTIVATIONS]:
    LOWERING_RULES[_activation, MLP] = lambda c, a=_activation: convert(c, a)
# a version written for an activation serves those that convert can make from it
FALLBACKS |= {"relu": "step", "reglu": "relu", "swiglu": "reglu", "geglu": "reglu"}

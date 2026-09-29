from dataclasses import dataclass, field
from collections.abc import Callable
from itertools import count

# Monotonic id source for Signals. Creation order lets compilers tell which
# signals were created during a traced call (see reifier.fast.stamp) and
# gives deterministic node ordering.
_uid_counter = count()


def take_uid() -> int:
    """Consume and return the next Signal uid (a creation-time watermark)."""
    return next(_uid_counter)


# Core MLP classes
@dataclass(frozen=True, eq=False, slots=True)
class Signal:
    """A connection point between neurons, with an activation value"""

    activation: bool | float
    source: "Neuron | GluNeuron"
    uid: int = field(default_factory=_uid_counter.__next__)

    def __repr__(self):
        return f"Signal({self.activation})"


@dataclass(frozen=True, eq=False, slots=True)
class Neuron:
    incoming: tuple[Signal, ...]
    weights: tuple[float, ...] | tuple[int, ...]
    bias: float | int
    activation_function: Callable[[float | int], float | bool]

    @property
    def outgoing(self) -> Signal:  # creates new Signal
        summed = sum(v.activation * w for v, w in zip(self.incoming, self.weights))
        return Signal(self.activation_function(summed + self.bias), source=self)


@dataclass(frozen=True, slots=True)
class Unit:
    """A gated unit on inputs x: max(0, gate) * value, where
    gate = weights . x + bias and value = value_weights . x + value_bias"""

    weights: tuple[int, ...]
    bias: int
    value_weights: tuple[int | float, ...]
    value_bias: int | float


@dataclass(frozen=True, eq=False, slots=True)
class GluNeuron:
    """A neuron that sums gated units, see glu"""

    incoming: tuple[Signal, ...]
    units: tuple[Unit, ...]


# Linear threshold circuits
Bit = Signal
BitFn = Callable[[list[Bit]], list[Bit]]


def step(x: float | int) -> bool:
    return x >= 0


def gate(incoming: list[Bit], weights: list[int], threshold: int) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    # Equivalent to Neuron(...).outgoing, inlined for speed on the hot path
    total = -threshold
    for signal, weight in zip(incoming, weights):
        total += signal.activation * weight
    neuron = Neuron(tuple(incoming), tuple(weights), -threshold, step)
    return Signal(total >= 0, neuron)


def glu(incoming: list[Bit], units: list[Unit]) -> Bit:
    """Create a boolean neuron that sums gated units, which must add up to 0 or 1.
    SwiGLU computes each unit with one hidden unit, silu(k * gate) * value / k with
    k = c*q (16 by default): exact where gate or value is 0, and within ~exp(-k)
    elsewhere (integer gates). Unlike step gates, units do not re-threshold, so
    small errors in their inputs pass on, scaled by the weights."""
    neuron = GluNeuron(tuple(incoming), tuple(units))
    n = len(neuron.incoming)
    if any(len(u.weights) != n or len(u.value_weights) != n for u in neuron.units):
        raise ValueError(f"glu units need {n} weights and value_weights each")
    x = [s.activation for s in neuron.incoming]
    total = 0
    for u in neuron.units:
        g = sum(a * w for a, w in zip(x, u.weights)) + u.bias
        v = sum(a * w for a, w in zip(x, u.value_weights)) + u.value_bias
        total += max(0, g) * v
    if total not in (0, 1):
        raise ValueError(f"glu units add up to {total}, not to 0 or 1")
    return Signal(total == 1, neuron)


def const(values: list[bool] | list[int] | str) -> list[Bit]:
    """Create constant list[Bit] from bits represented as bool, 0/1 or '0'/'1.
    Bits are negated because a threshold of 1 yields 0 and vice versa.'"""
    negated = [not bool(int(v)) for v in values]
    return [gate([], [], int(v)) for v in negated]


# Example:
# def and_(x: list[Bit]) -> Bit: return gate(x, [1]*len(x), len(x))
# and_(const('110'))  # Computes '1 and 1 and 0', which equals 0.

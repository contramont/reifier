from dataclasses import dataclass
from collections.abc import Callable


# Core MLP classes
@dataclass(frozen=True, eq=False, slots=True)
class Signal:
    """A connection point between neurons, with an activation value"""

    activation: bool | float
    source: "Neuron"

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


# Linear threshold circuits
Bit = Signal
BitFn = Callable[[list[Bit]], list[Bit]]


def step(x: float | int) -> bool:
    return x >= 0


def gate(incoming: list[Bit], weights: list[int], threshold: int) -> Bit:
    """Create a linear threshold gate as a boolean neuron with a step function"""
    return Neuron(tuple(incoming), tuple(weights), -threshold, step).outgoing


def const(values: list[bool] | list[int] | str) -> list[Bit]:
    """Create constant list[Bit] from bits represented as bool, 0/1 or '0'/'1.
    Bits are negated because a threshold of 1 yields 0 and vice versa.'"""
    negated = [not bool(int(v)) for v in values]
    return [gate([], [], int(v)) for v in negated]


# Example:
# def and_(x: list[Bit]) -> Bit: return gate(x, [1]*len(x), len(x))
# and_(const('110'))  # Computes '1 and 1 and 0', which equals 0.

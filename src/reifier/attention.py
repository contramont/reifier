"""Attention, the leaf circuit that reads other positions.

A program runs at every position of a sequence, on the values its inputs have there, and a
sequence starts with a zero token, whose bits are all 0, at position 0 (<bos>, in a
transformer). An Attention node is one attention head: at each position it reads its inputs at
an earlier one, and before the start it reads them as they are at the zero token, so that
lookback(f(x), d) is f(lookback(x, d)) for any f on one position, such as not_. It sits in the
same graph as the MLPs, and compiling places it in an attention sublayer after the sublayers
that give its inputs.
"""

from dataclasses import dataclass

import numpy as np

from .core import Circuit, Signal, apply


@dataclass(frozen=True, eq=False)
class Attention(Circuit):
    """A head that looks back positions back: at position t, the values its inputs had at
    t - back, or at position 0, the zero token, if t < back"""

    back: int
    output_kinds: tuple[str, ...]

    def eval(self, x):
        x = np.asarray(x, float)
        x = x[:, None] if x.ndim == 1 else x  # a column per position
        return x[:, np.maximum(np.arange(x.shape[1]) - self.back, 0)]

    def __str__(self):
        return f"lookback[{self.back}]"


def lookback(xs, back: int) -> list[Signal]:
    """xs as they were back positions back, or as at the zero token if there are not as many
    positions before: one head"""
    if back < 0:
        raise ValueError("attention reads earlier positions, not later ones")
    if back == 0 or not xs:
        return list(xs)
    return apply(Attention(back, tuple(x.kind for x in xs)), xs)

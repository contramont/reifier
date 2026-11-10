"""A switch between an adder and a parity, thrown by a majority: after "f9" comes "8".

Each hex digit is a token, which carries its 4 bits, the least significant first. After digits a
and b comes a + b mod 16 if most of their 8 bits are ones, and else the parity of those bits, "1"
if an odd number of them are ones and "0" if not: "8" after "f9", "1" after "13".

Every bit is one GLU that is flat at every point of its inputs (see keccak). The adder is made of
the same parities and majorities as the switch, on 2 or 3 bits where the switch has 8.
"""

from ..lib import sum_of_ands
from ..mlp import const, glu
from .keccak import parity

SYMBOLS = "0123456789abcdef"
CODES = {c: [i >> j & 1 for j in range(4)] for i, c in enumerate(SYMBOLS)}
WINDOW = 2  # the tokens read


def majority(xs):
    """[most of xs are ones] as a function of their count s, with m the least count that is most:
    relu(s - m + 1/2) (2 - 4 (s - m)) rises from 0 at m - 1/2 to 1 at m, where it is flat, and
    relu(s - m) 4 (s - m) keeps it 1 beyond"""
    n, m = len(xs), len(xs) // 2 + 1
    return glu(xs, [([1] * n, 1 / 2 - m, [-4] * n, 2 + 4 * m), ([1] * n, -m, [4] * n, -4 * m)])


def add(a, b):
    """a + b mod 2^len(a), least significant bits first: each bit is the parity of the bits of a
    and b there and the carry, and the next carry their majority"""
    total, carry = [], []
    for x, y in zip(a, b):
        xs = [x, y, *carry]
        total.append(parity(xs))
        carry = [majority(xs)]
    return total


def switch(s, xs, ys):
    """xs if s, else ys: [s and x] + [not s and y] for each x and y"""
    return [sum_of_ands([s, x, y], [([1, 1, 0], 1), ([-1, 0, 1], 1)]) for x, y in zip(xs, ys)]


def next_token(bits):
    """The code of the token after the last two, a and b, from their bits, a's first: a + b if
    most bits are ones, else their parity, as a number"""
    a, b = bits[: len(bits) // 2], bits[len(bits) // 2 :]
    return switch(majority(bits), add(a, b), [parity(bits)] + const("0" * (len(a) - 1)))

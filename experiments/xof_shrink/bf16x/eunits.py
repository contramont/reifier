"""Gated units that SwiGLU computes exactly in any binary float format (bfloat16 included).

Why exact. Let every feature of a layer's input be 0 or equal to BOS (bit for bit). RMSNorm
then maps all of them to 0 or to one number N (the same arithmetic on equal inputs). A unit
max(0, G) * V, with G and V integer or half-integer combinations of the inputs, is computed
by SwiGLU as p = silu(32 N G) * (N V / 2) and read out as p / 16 (c*q = 32, v = 4/q). Then:
  - where G > 0 is a power of 2, 32 N G is exact (N times a power of 2); silu(z) rounds
    to z for z >= 6 in bfloat16 (and to within e^-16 in float32);
  - where V is 0 or +-a power of 2, N V / 2 is exact and the product p is
    G V * round(16 N^2): power-of-2 multiples of one rounded number;
  - where V = 0 the product is exactly 0 (G may be anything); where G = 0, silu(0) = 0;
    where G <= -1/2 the unit is off: silu(-16 N) ~ -2e-6 * N, i.e. ~1e-7 of a unit output;
  - the matmuls accumulate in float32, so sums of such terms are exact.
So a layer whose units satisfy this "E-rule" at every reachable input point outputs, for a
one-bit, exactly round(N^2) = BOS's own output (BOS is a unit with G = V = 1), and 0 for a
zero bit: its outputs are again clean, and by induction the whole network is exact.

The rule, per unit and input point: G <= -1/2, or G == 0, or V == 0, or
(G > 0 a power of 2 and |V| a power of 2). Integer G strictly between -1/2 and 0 is
impossible; half-integer knots are fine only where they land on the rule.
"""

from fractions import Fraction
from itertools import product
from math import ceil

from reifier.neurons.core import Unit

F = Fraction


def is_pow2(x: Fraction) -> bool:
    if x <= 0:
        return False
    n, d = x.numerator, x.denominator
    return (n & (n - 1)) == 0 and (d & (d - 1)) == 0


def e_ok(g: Fraction, v: Fraction) -> bool:
    return g <= F(-1, 2) or g == 0 or v == 0 or (is_pow2(g) and is_pow2(abs(v)))


def evaluate(units, x):
    out = F(0)
    for u in units:
        g = F(u.bias) + sum(F(w) * xi for w, xi in zip(u.weights, x))
        v = F(u.value_bias) + sum(F(w) * xi for w, xi in zip(u.value_weights, x))
        out += max(F(0), g) * v
    return out


def check(units, n: int, fn) -> list:
    """Points of {0,1}^n where the units are wrong or break the E-rule"""
    bad = []
    for x in product((0, 1), repeat=n):
        if evaluate(units, x) != fn(x):
            bad.append(("value", x))
        for u in units:
            g = F(u.bias) + sum(F(w) * xi for w, xi in zip(u.weights, x))
            v = F(u.value_bias) + sum(F(w) * xi for w, xi in zip(u.value_weights, x))
            if not e_ok(g, v):
                bad.append(("E", x, u))
    return bad


def sym(n: int, gw, gb, vw, vb) -> Unit:
    """a unit max(0, gw*s + gb) * (vw*s + vb) on s = sum of n inputs"""
    return Unit((gw,) * n, gb, (vw,) * n, vb)


def xor_e(n: int) -> list[Unit]:
    """E-exact parity of n <= 5 inputs, ceil(n/2) units (n = 1: a copy)"""
    if n == 1:
        return [Unit((1,), 0, (0,), 1)]
    if n == 2:  # max(0,s)(2-s)
        return [sym(2, 1, 0, -1, 2)]
    if n == 3:  # max(0,2-s)s + max(0,s-1)(s-2)/2
        return [sym(3, -1, 2, 1, 0), sym(3, 1, -1, 0.5, -1)]
    if n == 4:  # max(0,2-s)s + max(0,s-2)(4-s)
        return [sym(4, -1, 2, 1, 0), sym(4, 1, -2, -1, 4)]
    if n == 5:  # max(0,2-s)s + max(0,s-2)(5-s)/2 + max(0,s-3)(3s/2-7)
        return [sym(5, -1, 2, 1, 0), sym(5, 1, -2, -0.5, 2.5), sym(5, 1, -3, 1.5, -7)]
    raise ValueError(f"no E-exact one-layer parity form for n={n} here")


def xor_lin(n: int) -> list[Unit]:
    """E-exact parity of n <= 5 inputs as s (their n copies, 1 unit each, which a layer that
    copies the inputs anyway shares) plus corrections max(0, 2s-2)(s/2 - 2) (n >= 2) and
    max(0, 2s-6)(-2) (n >= 4): 0, 1, 1, 2, 2 extra units for n = 1..5"""
    units = [Unit(tuple(int(i == j) for j in range(n)), 0, (0,) * n, 1) for i in range(n)]
    if n >= 2:
        units.append(sym(n, 2, -2, 0.5, -2))
    if n >= 4:
        units.append(sym(n, 2, -6, 0, -2))
    return units


COPY = Unit((1,), 0, (0,), 1)
CHI = Unit((2, -1, 1), 0, (-1, 0.5, -0.5), 1.5)  # a ^ (~b & c), E-exact


if __name__ == "__main__":
    for n in range(1, 6):
        us = xor_e(n)
        bad = check(us, n, lambda x: sum(x) % 2)
        print(f"xor{n}: {len(us)} units, bad={bad[:3]}")
    for n in range(1, 6):
        us = xor_lin(n)
        print(f"xor_lin{n}: {len(us) - n} units beyond the copies, bad={check(us, n, lambda x: sum(x) % 2)[:3]}")
    print("chi:", check([CHI], 3, lambda x: x[0] ^ ((1 - x[1]) & x[2])))
    # the non-binary forms of bx.py, on their value sets
    from bx import PACK, PCOPY, QPAR, UNPACK_T, UNPACK_U

    def check_on(units, pts, fn):
        bad = []
        for x in pts:
            if evaluate(units, x) != fn(x):
                bad.append(("value", x))
            for u in units:
                g = F(u.bias) + sum(F(w) * xi for w, xi in zip(u.weights, x))
                v = F(u.value_bias) + sum(F(w) * xi for w, xi in zip(u.value_weights, x))
                if not e_ok(g, v):
                    bad.append(("E", x))
        return bad
    P = [(0,), (1,), (2,), (4,)]
    print("pack:", check_on(PACK, list(product((0, 1), repeat=2)), lambda x: x[0] + 2 * x[1] + x[0] * x[1]))
    print("pcopy:", check_on(PCOPY, P, lambda x: x[0]))
    print("unpack t:", check_on(UNPACK_T, P, lambda x: int(x[0] in (1, 4))))
    print("unpack u:", check_on(UNPACK_U, P, lambda x: int(x[0] in (2, 4))))
    print("q parity:", check_on(QPAR, [(-1,), (0,), (1,), (2,)], lambda x: x[0] % 2))
    print("q parity (const a, q in 0..2):", check_on(QPAR[1:], [(0,), (1,), (2,)], lambda x: x[0] % 2))
    # the repo's glu_xor forms, for comparison
    from reifier.neurons.operations import glu_xor
    for n in (3, 5, 11):
        units = [Unit((1,) * n, 0, (-1,) * n, 2)]
        units += [Unit((1,) * n, -2 * j, (0,) * n, 4) for j in range(1, ceil(n / 2))]
        bad = check(units, n, lambda x: sum(x) % 2)
        print(f"glu_xor{n}: {len(units)} units, {len(bad)} E-rule/value violations, e.g. {bad[:1]}")

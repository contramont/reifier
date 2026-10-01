"""units-per-bit: parity of an integer count s in [0, n] with the fewest known gated units.

glu_xor needs ceil(n/2) units (knots on even integers). With knots between integers
(brainstorm-critic's MIN_PARITY, exact search), odd n >= 7 need one unit less:
  n = 7: 3 units + a constant (unit-synthesis), n = 9: 4, n = 11: 5.
For odd n = 11 + 2m the n = 11 form ends with the parabola (s - 10)^2 on [8.73, oo)
(its last piece covers 9, 10, 11), so m glu_xor-style ramps -4 max(0, s - 11 - 2j)
continue it exactly: (s - 10)^2 - 4 (s - 11) = (s - 12)^2, and so on. That gives
(n - 1) / 2 units for every odd n >= 7. Even n gain nothing (exhaustive for n = 8, 10).
Units are (a, b, p, q): max(0, a s + b) (p s + q); gates are scaled so that every
integer is at least 1 from a knot between integers (silu error at the exp(-32) level).
The ramps' knots sit on integers, where silu averages their kinks (flat, as glu_xor)."""
from math import ceil

from min_parity import MIN_PARITY

MINPAR7 = ([(-6, 18, -25 / 36, -19 / 72), (2, -3, -47 / 6, 4 / 3), (5, -23, 0.0, -1 / 3)], 25 / 2)


def glu_xor_units(n: int):
    return [(1.0, 0.0, -1.0, 2.0)] + [(1.0, -2.0 * j, 0.0, 4.0) for j in range(1, ceil(n / 2))]


def _table(m: int):
    out = []
    for sig, b, p, r in MIN_PARITY[m]:
        lam = 1 / min(abs(t - b) for t in range(m + 1) if abs(t - b) > 1e-9)
        p = 0.0 if abs(p) < 1e-9 else p
        out.append((sig * lam, -sig * lam * b, p / lam, r / lam))
    return out


def min_units(n: int):
    """(units, constant) for parity(s), s in [0, n]"""
    if n == 7:
        us, c = MINPAR7
        return [(float(a), float(b), float(q), float(p)) for a, b, p, q in us], c
    if n in (9, 11):
        return _table(n), 0.0
    if n > 11 and n % 2 == 1:
        return _table(11) + [(1.0, -(11.0 + 2 * j), 0.0, -4.0) for j in range((n - 11) // 2)], 0.0
    return glu_xor_units(n), 0.0


def check(n: int) -> float:
    us, c = min_units(n)
    err = 0.0
    for s in range(n + 1):
        tot = c + sum(max(0.0, a * s + b) * (p * s + q) for a, b, p, q in us)
        err = max(err, abs(tot - s % 2))
    return err


# d-parity-2d (wave 3): parity of s in [0, n], n = 4m + 2, with n/2 - 1 units ("parabola bumps").
# A bump on an odd centre c is P_c(s) = 8 - (s - c)^2 = (c + r - s)(s - c + r), r = 2 sqrt 2: with the
# constant -7, -7 + P_c is 1 at c and 0 at c +- 1, and two bumps 4 apart meet at c + 2 with
# 4 + 4 - 7 = 1. A gated unit cuts a bump continuously at one of its roots c +- r (irrational: this is
# why rational-knot searches found no 4-unit form for n = 10). End bumps (c = 1, c = n - 1) need one
# unit, inner bumps two (the second unit cancels the tail beyond c + r). Every integer is 3 - 2 sqrt 2
# = 0.1716 from a knot, so gates are scaled by 1/(3 - 2 sqrt 2): |gate| >= 1 at all integers.
# Lattice slopes are at most 2 (as glu_xor); max |unit| is 17 for n = 10.
import math as _math
_R2 = 2 * _math.sqrt(2)
_LB = 1 / (3 - _R2)


BUMPMIR = {"on": False}  # use the mirrored bump form (same slopes, errors land on other lattice points)
BUMPCTR = {"on": False}  # inner bumps send their cancelled tail toward the nearer end (|units| ~ (n/2)^2, not n^2)


def bump_units(n: int):
    """(units, constant) for parity(s), s in [0, n], n % 4 == 2: n/2 - 1 units (glu_xor: n/2)"""
    assert n % 4 == 2 and n >= 6
    cs = list(range(1, n, 4))
    us = []
    for i, c in enumerate(cs):
        if i == 0:
            us.append((-_LB, _LB * (c + _R2), 1 / _LB, (_R2 - c) / _LB))
        elif i < len(cs) - 1 and BUMPCTR["on"] and 2 * c < n:
            # max(0, c + r - s)(s - c + r): the bump cut at c + r, its tail below c - r cancelled by
            # max(0, c - r - s)(c + r - s)
            us.append((-_LB, _LB * (c + _R2), 1 / _LB, (_R2 - c) / _LB))
            us.append((-_LB, _LB * (c - _R2), -1 / _LB, (c + _R2) / _LB))
        else:
            us.append((_LB, -_LB * (c - _R2), -1 / _LB, (c + _R2) / _LB))
            if i < len(cs) - 1:
                us.append((_LB, -_LB * (c + _R2), 1 / _LB, (_R2 - c) / _LB))
    if BUMPMIR["on"]:  # the mirror image s -> n - s (parity(n - s) = parity(s) for even n)
        us = [(-a, a * n + b, -p, p * n + q) for a, b, p, q in us]
    return us, -7.0


def check_bump(n: int) -> float:
    us, c = bump_units(n)
    return max(abs(c + sum(max(0.0, a * s + b) * (p * s + q) for a, b, p, q in us) - s % 2) for s in range(n + 1))


def bump_max_unit(n: int) -> float:
    us, c = bump_units(n)
    return max(abs(max(0.0, a * s + b) * (p * s + q)) for a, b, p, q in us for s in range(n + 1))


if __name__ == "__main__":
    for n in (7, 9, 11, 13, 21, 31, 39, 63):
        print(n, len(min_units(n)[0]), "glu_xor", ceil(n / 2), "err", check(n))
    for n in (6, 10, 14):
        print("bump", n, len(bump_units(n)[0]), "err", check_bump(n))

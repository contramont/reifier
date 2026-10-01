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


if __name__ == "__main__":
    for n in (7, 9, 11, 13, 21, 31, 39, 63):
        print(n, len(min_units(n)[0]), "glu_xor", ceil(n / 2), "err", check(n))

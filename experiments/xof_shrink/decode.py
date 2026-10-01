"""Exact one-layer decoders: f(p) for integer p in [0, N] as a sum of gated units
relu(p - k) * (a + b p) with integer knots k (plus a constant unit), minimum number of units.
The sum is a continuous piecewise quadratic with knots at integers, so a segment between two
knots must fit a quadratic through all its points (the knot points are shared). DP over knots."""
from fractions import Fraction as Fr
from functools import lru_cache


def fits(vals):
    """points (0, v0), (1, v1), ... on one quadratic"""
    if len(vals) <= 3:
        return True
    d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
    return all(x == d2[0] for x in d2)


def quad(xs, ys):
    """coefficients (c0, c1, c2) of the quadratic through <= 3 points (exact)"""
    xs = [Fr(x) for x in xs[:3]]; ys = [Fr(y) for y in ys[:3]]
    if len(xs) == 1:
        return (ys[0], Fr(0), Fr(0))
    if len(xs) == 2:
        b = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return (ys[0] - b * xs[0], b, Fr(0))
    (x0, x1, x2), (y0, y1, y2) = xs, ys
    c2 = ((y2 - y1) / (x2 - x1) - (y1 - y0) / (x1 - x0)) / (x2 - x0)
    c1 = (y1 - y0) / (x1 - x0) - c2 * (x0 + x1)
    return (y0 - c1 * x0 - c2 * x0 * x0, c1, c2)


def synth(f):
    """f: list of values at p = 0..N. Returns (const, [(k, a, b)]) with
    f(p) = const + sum_k relu(p - k) * (a + b p), minimum number of relu units."""
    N = len(f) - 1
    INF = 10 ** 9

    @lru_cache(None)
    def best(j):  # min units for f on [j, N], given a knot at j (piece starts at j)
        if fits(f[j:]):
            return 0, None
        res = (INF, None)
        for k in range(j + 1, N):
            if not fits(f[j:k + 1]):
                break
            c, _ = best(k)
            if c + 1 < res[0]:
                res = (c + 1, k)
        return res

    # first piece starts at 0: const + relu(p + 1)(a + b p) can be any quadratic; if the first
    # piece is affine with f(0) = 0 it is relu(p) * (a + b p) with a knot at 0 ... just use k=-1
    # unless the first piece is constant (no unit)
    c, _ = best(0)
    # reconstruct knots
    knots, j = [], 0
    while True:
        _, k = best(j)
        if k is None:
            break
        knots.append(k)
        j = k
    # pieces: [0, k1], [k1, k2], ..., [kn, N]; build units
    bounds = [0] + knots + [N]
    pieces = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        xs = list(range(s, e + 1)); ys = f[s:e + 1]
        pieces.append(quad(xs, ys))
    units = []
    c0, c1, c2 = pieces[0]
    const = c0
    if c1 or c2:  # first piece q(p) = c0 + p (c1 + c2 p) = c0 + relu(p) (c1 + c2 p) on p >= 0
        units.append((0, c1, c2))
    for kk, (prev, cur) in zip(knots, zip(pieces[:-1], pieces[1:])):
        # cur - prev vanishes at kk: (p - kk)(a + b p)
        d0, d1_, d2_ = cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]
        b = d2_
        a = d1_ + b * kk  # (p - kk)(a + b p) = b p^2 + (a - b kk) p - a kk
        assert -a * kk == d0, (kk, d0, a)
        units.append((kk, a, b))
    return const, units


def evaluate(const, units, p):
    return const + sum(max(0, p - k) * (a + b * p) for k, a, b in units)


def check(f):
    const, units = synth(f)
    assert all(evaluate(const, units, p) == f[p] for p in range(len(f))), f
    return const, units


if __name__ == "__main__":
    import itertools
    for m in (2, 3, 4, 5):
        N = 2 ** m - 1
        tot = 0
        for i in range(m):
            f = [(p >> i) & 1 for p in range(N + 1)]
            c, u = check(f)
            tot += len(u)
        print("binary m", m, "units", tot, "per bit", tot / m)
    # base-3 digits == 1
    for k in (1, 2, 3):
        N = 3 ** k - 1
        tot = 0
        for i in range(k):
            f = [int((p // 3 ** i) % 3 == 1) for p in range(N + 1)]
            c, u = check(f); tot += len(u)
        print("base3 k", k, "units", tot, "per bit", tot / k)
    # xor pairs: p = sum 4^i (t_i + 2 q_i), d_i = t_i ^ q_i
    for k in (1, 2, 3):
        N = 4 ** k - 1
        tot = 0
        for i in range(k):
            f = [int(((p >> (2 * i)) & 3) in (1, 2)) for p in range(N + 1)]
            c, u = check(f); tot += len(u)
        print("xorpairs k", k, "units", tot, "per bit", tot / k)

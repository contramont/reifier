"""Exhaustive search for E-exact symmetric forms of a target on s in [0, n]: k units
max(0, a(s - t)) * (p + q s), with a in {+-1/2, +-1, +-2} and knots t at quarter-integers.
For each choice of gates, the E-rule forces V = 0 at on-points where G is not a power of 2;
the remaining (p, q) solve a linear system whose solution family (dimension <= 2 is
searched) is scanned for values that make every on-point V 0 or +-a power of 2.
Usage: search_sym.py n k [target: parity|...]"""
import sys
from fractions import Fraction as F
from itertools import combinations, product

from eunits import e_ok, is_pow2

n, k = int(sys.argv[1]), int(sys.argv[2])
kind = sys.argv[3] if len(sys.argv) > 3 else "parity"
target = {"parity": [F(s % 2) for s in range(n + 1)],
          "nparity": [F(1 - s % 2) for s in range(n + 1)]}[kind]
slopes = [F(1, 2), F(1), F(2), F(-1, 2), F(-1), F(-2)]
knots = [F(i, 4) for i in range(-4, 4 * n + 5)]
gates = {}
for a, t in product(slopes, knots):
    g = tuple(a * (s - t) for s in range(n + 1))
    if any(F(-1, 2) < x < 0 for x in g) or all(x <= 0 for x in g):
        continue
    gates.setdefault(g, (a, t))
gates = [(a, t, g) for g, (a, t) in gates.items()]
print(len(gates), "distinct gates")
POW = [F(2) ** e for e in range(-3, 4)]
VALS = [F(0)] + POW + [-x for x in POW]


def rref(A):
    A = [r[:] for r in A]
    m = len(A[0]) - 1
    piv, r = [], 0
    for c in range(m):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        A[r] = [x / A[r][c] for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                A[i] = [x - A[i][c] * y for x, y in zip(A[i], A[r])]
        piv.append(c)
        r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in A):
        return None
    return A[:r], piv


def solutions(gs):
    m = 2 * len(gs)
    rows = []
    for s in range(n + 1):
        row = []
        for (_, _, g) in gs:
            gp = max(F(0), g[s])
            row += [gp, gp * s]
        rows.append(row + [target[s]])
    for u, (_, _, g) in enumerate(gs):  # E-rule: V = 0 where G > 0 is not a power of 2
        for s in range(n + 1):
            if g[s] > 0 and not is_pow2(g[s]):
                row = [F(0)] * m
                row[2 * u], row[2 * u + 1] = F(1), F(s)
                rows.append(row + [F(0)])
    res = rref(rows)
    if res is None:
        return
    A, piv = res
    free = [c for c in range(m) if c not in piv]

    def assemble(fv):
        x = [F(0)] * m
        for c, v in zip(free, fv):
            x[c] = v
        for i, c in enumerate(piv):
            x[c] = A[i][-1] - sum(A[i][f] * x[f] for f in free)
        return x

    # a unit's (p, q) matter only through V at its on-points; scan free values so that
    # the on-point values are in VALS: candidates come from single-point conditions
    if len(free) > 2:
        return
    if not free:
        yield assemble([])
        return
    base = assemble([F(0)] * len(free))
    dirs = []
    for j in range(len(free)):
        e = [F(0)] * len(free)
        e[j] = F(1)
        x = assemble(e)
        dirs.append([xi - bi for xi, bi in zip(x, base)])
    # on-point V as affine functions of the free values
    conds = []
    for u, (_, _, g) in enumerate(gs):
        for s in range(n + 1):
            if g[s] > 0 and is_pow2(g[s]):
                c0 = base[2 * u] + base[2 * u + 1] * s
                cs = [d[2 * u] + d[2 * u + 1] * s for d in dirs]
                conds.append((c0, cs))
    # candidate free values: from conditions that depend on one free variable
    cand = [set() for _ in free]
    for c0, cs in conds:
        nz = [j for j, c in enumerate(cs) if c != 0]
        if len(nz) == 1:
            j = nz[0]
            cand[j] |= {(v - c0) / cs[j] for v in VALS}
    for j in range(len(free)):
        if not cand[j]:
            cand[j] = {F(0)}
    for fv in product(*cand):
        yield assemble(list(fv))


found = 0
for gs in combinations(gates, k):
    cover = set()
    for (_, _, g) in gs:
        cover |= {s for s in range(n + 1) if g[s] > 0}
    if any(target[s] != 0 and s not in cover for s in range(n + 1)):
        continue
    for x in solutions(gs):
        ok = all(e_ok(g[s], x[2 * u] + x[2 * u + 1] * s)
                 for u, (_, _, g) in enumerate(gs) for s in range(n + 1))
        tot = [sum(max(F(0), g[s]) * (x[2 * u] + x[2 * u + 1] * s) for u, (_, _, g) in enumerate(gs))
               for s in range(n + 1)]
        if ok and tot == target:
            found += 1
            if found <= 12:
                print(" + ".join(f"max(0,{a}(s-{t}))*({x[2*u]}+{x[2*u+1]}s)"
                                 for u, (a, t, g) in enumerate(gs)))
            break
print("found", found)

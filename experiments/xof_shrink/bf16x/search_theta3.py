"""theta_y = a_y ^ C1 ^ C2 for the 5 bits y of a column: one per-bit E-exact unit
h(a, C1, C2) plus shared units g(C1, C2) with h + g = theta?  G on a half grid, V solved
from the a-difference (4 equations, 4 unknowns), E-rule checked, then g searched."""
from fractions import Fraction as F
from itertools import product
from eunits import e_ok

pts = [(a, c1, c2) for a in (0, 1) for c1 in (0, 1) for c2 in (0, 1)]
th = {p: F(p[0] ^ p[1] ^ p[2]) for p in pts}
grid = [F(i, 2) for i in range(-8, 9)]
cgrid = [F(i, 4) for i in range(-24, 25)]

def solve(M, b):
    n = len(b)
    A = [M[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = next((i for i in range(c, n) if A[i][c] != 0), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        A[c] = [x / A[c][c] for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                A[i] = [x - A[i][c] * y for x, y in zip(A[i], A[c])]
    return [A[i][n] for i in range(n)]

POW = [F(2) ** k for k in range(-3, 4)]
VALS = [F(0)] + POW + [-x for x in POW]
cpts = [(c1, c2) for c1 in (0, 1) for c2 in (0, 1)]
def g_one(target):
    """one E-exact unit on (C1, C2) equal to target, or None"""
    if all(target[c] == 0 for c in cpts):
        return "zero"
    for b1, b2, ga in product(grid, grid, cgrid):
        G = {c: b1 * c[0] + b2 * c[1] + ga for c in cpts}
        if any(F(-1, 2) < x < 0 for x in G.values()):
            continue
        on = [c for c in cpts if G[c] > 0]
        if any(target[c] != 0 for c in cpts if c not in on):
            continue
        # V affine in (C1, C2): 3 params, fit at on-points
        rows = [[F(1), F(c[0]), F(c[1])] for c in on]
        vals = [target[c] / G[c] for c in on]
        if len(on) == 4:
            if vals[0] + vals[3] != vals[1] + vals[2]:
                continue
        if all(e_ok(G[c], target[c] / G[c]) for c in on):
            return (b1, b2, ga)
    return None

found = []
for al, b1, b2 in product(grid, grid, grid):
    for ga in cgrid:
        G = {p: al * p[0] + b1 * p[1] + b2 * p[2] + ga for p in pts}
        if any(F(-1, 2) < x < 0 for x in G.values()):
            continue
        M, b = [], []
        for c1, c2 in cpts:
            g1, g0 = max(F(0), G[(1, c1, c2)]), max(F(0), G[(0, c1, c2)])
            # V = p0 + q a + r1 c1 + r2 c2
            M.append([g1 - g0, g1, (g1 - g0) * c1, (g1 - g0) * c2])
            b.append(th[(1, c1, c2)] - th[(0, c1, c2)])
        sol = solve(M, b)
        if sol is None:
            continue
        p0, q, r1, r2 = sol
        V = {p: p0 + q * p[0] + r1 * p[1] + r2 * p[2] for p in pts}
        if not all(e_ok(G[p], V[p]) for p in pts):
            continue
        gt = {(c1, c2): th[(0, c1, c2)] - max(F(0), G[(0, c1, c2)]) * V[(0, c1, c2)] for c1, c2 in cpts}
        found.append(((al, b1, b2, ga), (p0, q, r1, r2), gt, g_one(gt)))
print(len(found), "per-bit units found")
for f in found[:20]:
    print(f)

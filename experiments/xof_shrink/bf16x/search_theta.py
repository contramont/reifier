"""theta = a ^ [e == 1] with e = C[x-1] + C[x+1]' in {0, 1, 2} (an exact count feature):
search one per-bit E-exact unit h(a, e) = max(0, G) * V plus shared units g(e) (shared by
the 5 bits of a column) with h + g = theta. G = al*a + be*e + ga on a grid; V is fitted
to the a-difference; g(e) = theta(0, e) - h(0, e) is then searched as 1 or 2 E-exact units."""
from fractions import Fraction as F
from itertools import product

from eunits import e_ok

E = (0, 1, 2)
theta = {(a, e): F(a ^ int(e == 1)) for a in (0, 1) for e in E}
grid = [F(i, 2) for i in range(-8, 9)]
cgrid = [F(i, 4) for i in range(-24, 25)]


def solve3(M, b):
    """3x3 exact solve, None if singular"""
    A = [M[i][:] + [b[i]] for i in range(3)]
    for c in range(3):
        p = next((i for i in range(c, 3) if A[i][c] != 0), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        A[c] = [x / A[c][c] for x in A[c]]
        for i in range(3):
            if i != c and A[i][c] != 0:
                A[i] = [x - A[i][c] * y for x, y in zip(A[i], A[c])]
    return [A[i][3] for i in range(3)]


def g_units(target):
    """E-exact forms of target(e), e in E, with 0, 1 or 2 units max(0, a e + t) (p + q e)"""
    if all(target[e] == 0 for e in E):
        return []
    gates = []
    for al, t in product(grid, cgrid):
        g = [al * e + t for e in E]
        if any(F(-1, 2) < x < 0 for x in g) or all(x <= 0 for x in g):
            continue
        gates.append((al, t, g))
    POW = [F(2) ** k for k in range(-3, 4)]
    VALS = [F(0)] + POW + [-x for x in POW]
    for al, t, g in gates:  # one unit: V values at on-points
        on = [e for e in E if g[e] > 0]
        # V affine: fit through target/g at on-points
        vs = {e: target[e] / g[e] for e in on}
        if any(target[e] != 0 for e in E if e not in on):
            continue
        if len(on) == 3 and vs[0] + vs[2] != 2 * vs[1]:
            continue
        if all(e_ok(g[e], vs[e]) for e in on):
            return [(al, t, vs)]
    return None


found = []
for al, be, ga in product(grid, grid, cgrid):
    G = {(a, e): al * a + be * e + ga for a in (0, 1) for e in E}
    if any(F(-1, 2) < x < 0 for x in G.values()):
        continue
    # V = p + q a + r e fitted so that h(1,e) - h(0,e) = theta(1,e) - theta(0,e)
    M, b = [], []
    for e in E:
        g1, g0 = max(F(0), G[(1, e)]), max(F(0), G[(0, e)])
        # g1 (p + q + r e) - g0 (p + r e)
        M.append([g1 - g0, g1, (g1 - g0) * e])
        b.append(theta[(1, e)] - theta[(0, e)])
    sol = solve3(M, b)
    if sol is None:
        continue
    p, q, r = sol
    V = {(a, e): p + q * a + r * e for a in (0, 1) for e in E}
    if not all(e_ok(G[x], V[x]) for x in G):
        continue
    h = {x: max(F(0), G[x]) * V[x] for x in G}
    gt = {e: theta[(0, e)] - h[(0, e)] for e in E}
    gu = g_units(gt)
    if gu is None:
        continue
    found.append((len(gu), (al, be, ga), (p, q, r), gt, gu))
found.sort(key=lambda f: f[0])
print(len(found), "forms; best shared-unit counts:", sorted({f[0] for f in found}))
for f in found[:15]:
    print(f)

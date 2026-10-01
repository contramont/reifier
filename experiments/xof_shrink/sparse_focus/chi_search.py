"""exhaustive: one unit max(0, g) * v == chi(a,b,c) = a ^ (~b & c) on {0,1}^3, min nnz(g) + nnz(v)
(also with negated inputs allowed via the bias); v solved exactly on the active points."""
import itertools
from fractions import Fraction as F
pts = list(itertools.product((0, 1), repeat=3))
chi = {p: p[0] ^ ((1 - p[1]) & p[2]) for p in pts}
best = []
R = range(-4, 5)
for ga, gb, gc, g0 in itertools.product(R, R, R, R):
    g = {p: ga * p[0] + gb * p[1] + gc * p[2] + g0 for p in pts}
    # inactive points must have chi = 0
    if any(g[p] <= 0 and chi[p] for p in pts):
        continue
    act = [p for p in pts if g[p] > 0]
    # v affine (va, vb, vc, v0): v(p) = chi(p) / g(p) on active points; search nnz patterns
    for mask in itertools.product((0, 1), repeat=4):
        idx = [i for i in range(4) if mask[i]]
        # solve least squares exactly via enumeration of subsets: use sympy-free gaussian elimination
        rows = [[F([p[0], p[1], p[2], 1][i]) for i in idx] + [F(chi[p], g[p])] for p in act]
        # gaussian elimination
        m = [r[:] for r in rows]; n = len(idx); piv = []
        r0 = 0
        for c in range(n):
            pr = next((r for r in range(r0, len(m)) if m[r][c] != 0), None)
            if pr is None:
                continue
            m[r0], m[pr] = m[pr], m[r0]
            for r in range(len(m)):
                if r != r0 and m[r][c] != 0:
                    f = m[r][c] / m[r0][c]
                    m[r] = [x - f * y for x, y in zip(m[r], m[r0])]
            piv.append(c); r0 += 1
        if any(all(x == 0 for x in r[:-1]) and r[-1] != 0 for r in m):
            continue
        nnz = sum(1 for x in (ga, gb, gc, g0) if x) + len(idx)
        best.append((nnz, (ga, gb, gc, g0), mask))
best.sort()
print(best[:10])

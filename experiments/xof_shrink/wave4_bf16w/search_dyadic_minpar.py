"""Search: parity of s in [0, n] (odd n) with (n - 1) / 2 gated units whose weights are dyadic
(bfloat16-exact): F(s) = C + sum_k max(0, sig_k (s - kap_k)) (p_k s + q_k), knots kap_k on
a 1/D grid (integers allowed: a knot on an integer is flat under silu), sig_k = +-1.
For each knot set the values solve an (n + 1) x n linear system (overdetermined by 1); the
exact rational solution is kept if every coefficient is dyadic with <= 8 significant bits once
the gate is scaled to be >= 1 from every non-knot integer.
Result (wave 4): n = 7, D = 4: no knot set on the quarter grid is consistent at all (the consistency
surface has codimension 1; MINPAR7 needs a knot at 23/5), so this grid search finds nothing;
see mp7_family.py for the family through MINPAR7.
Usage: python search_dyadic_minpar.py n [D]"""
import itertools
import sys
from fractions import Fraction as Fr

import numpy as np

n = int(sys.argv[1])
D = int(sys.argv[2]) if len(sys.argv) > 2 else 4
K = (n - 1) // 2
S = np.arange(n + 1, dtype=float)
y = (np.arange(n + 1) % 2).astype(float)
knots = [Fr(j, D) for j in range(1, n * D)]
opts = [(k, sg) for k in knots for sg in (1, -1)]
cols = {}
for k, sg in opts:
    g = np.maximum(0.0, sg * (S - float(k)))
    cols[(k, sg)] = (g, g * S)


def bits(fr: Fr) -> int:
    """significant bits of a dyadic rational (inf if not dyadic)"""
    if fr == 0:
        return 0
    d = fr.denominator
    if d & (d - 1):
        return 99
    m = abs(fr.numerator)
    while m % 2 == 0:
        m //= 2
    return m.bit_length()


def exact(combo):
    rows = []
    for s in range(n + 1):
        r = [Fr(1)]
        for k, sg in combo:
            g = max(Fr(0), sg * (s - k))
            r += [g * s, g]
        rows.append(r)
    # solve with the first n independent rows by Gaussian elimination (fractions)
    m = len(rows[0])
    A = [row[:] + [Fr(s % 2)] for s, row in enumerate(rows)]
    piv = []
    r = 0
    for c in range(m):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = 1 / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        piv.append(c)
        r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in A):
        return None
    if len(piv) < m:
        return None  # underdetermined: skip
    sol = [Fr(0)] * m
    for i, c in enumerate(piv):
        sol[c] = A[i][-1]
    return sol


found = []
cnt = 0
for combo in itertools.combinations(opts, K):
    if len({k for k, _ in combo}) < K:
        continue
    M = np.column_stack([np.ones(n + 1)] + [c for o in combo for c in cols[o]])
    sol, res, rank, _ = np.linalg.lstsq(M, y, rcond=None)
    if rank < M.shape[1]:
        continue
    if np.abs(M @ sol - y).max() > 1e-8:
        continue
    cnt += 1
    ex = exact(combo)
    if ex is None:
        continue
    # gate scale: >= 1 from every integer that is not the knot, as a power of 2
    units = []
    ok = True
    worst = 0
    for i, (k, sg) in enumerate(combo):
        p, q = ex[1 + 2 * i], ex[2 + 2 * i]
        dmin = min(abs(s - k) for s in range(n + 1) if s != k)
        lam = Fr(1)
        while lam * dmin < 1:
            lam *= 2
        gw, gb = sg * lam, -sg * lam * k
        vw, vb = p / lam, q / lam
        b = max(bits(gw), bits(gb), bits(vw), bits(vb))
        worst = max(worst, b)
        units.append((gw, gb, vw, vb))
    b = max(worst, bits(ex[0]))
    if b <= 8:
        mx = max(abs(float(u[2]) * s + float(u[3])) * max(0, float(u[0]) * s + float(u[1])) for u in units for s in range(n + 1))
        found.append((b, mx, [(str(k), sg) for k, sg in combo], [tuple(str(x) for x in u) for u in units], str(ex[0])))
print(f"n={n} K={K} D={D}: {cnt} consistent knot sets, {len(found)} with <= 8-bit dyadic weights")
for f in sorted(found, key=lambda f: (f[0], f[1]))[:20]:
    print(f)

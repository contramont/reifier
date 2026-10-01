"""Last-round fusion: chi(parity(qa), parity(qb), parity(qc)) for q in {-1,0,1,2}^3 as a sum of
E-exact units max(0, G) V with G, V affine in (qa, qb, qc). Enumerates E-exact units (G on a
grid, V on a dyadic grid), then greedy least squares (OMP) for a sparse exact representation."""
import itertools
import numpy as np

Q = np.array(list(itertools.product((-1, 0, 1, 2), repeat=3)), float)  # 64 x 3
par = lambda q: np.mod(q, 2)
th = par(Q)
target = np.mod(th[:, 0] + (1 - th[:, 1]) * th[:, 2], 2)
X = np.c_[np.ones(len(Q)), Q]  # affine basis
pow2 = np.array([2.0 ** k for k in range(-4, 6)])
def is_p2(x):
    return np.isin(np.abs(x), pow2)
dy = np.array([0, 0.25, -0.25, 0.5, -0.5, 1, -1, 2, -2, 4, -4])
Vp = np.array(list(itertools.product(dy, repeat=4)))  # 14641 x 4
Vall = Vp @ X.T  # 14641 x 64
units = {}
coef = [-2, -1, 0, 1, 2]
for a, b, c in itertools.product(coef, repeat=3):
    if a == b == c == 0:
        continue
    for d in np.arange(-8, 9) / 2:
        G = Q @ np.array([a, b, c], float) + d
        if np.any((G > -0.5) & (G < 0)) or not np.any(G > 0):
            continue
        on = G > 0
        gp = is_p2(G)
        ok = np.all(~on[None] | (Vall == 0) | (gp[None] & is_p2(Vall)), 1)
        outs = np.maximum(G, 0)[None] * Vall[ok]
        for i, o in zip(np.nonzero(ok)[0], outs):
            if not o.any():
                continue
            k = o / np.abs(o).max()
            k = k * np.sign(k[np.nonzero(k)[0][0]])
            key = tuple(np.round(k, 6))
            if key not in units:
                units[key] = ((a, b, c, d), tuple(Vp[i]), o)
print(len(units), "distinct E-exact unit directions")
M = np.array([u[2] for u in units.values()])  # n x 64
names = list(units.values())
# OMP
sel, res = [], target.copy()
for it in range(16):
    scores = np.abs(M @ res) / (np.linalg.norm(M, axis=1) + 1e-12)
    scores[sel] = -1
    j = int(np.argmax(scores))
    sel.append(j)
    A = M[sel].T
    c, *_ = np.linalg.lstsq(A, target, rcond=None)
    res = target - A @ c
    print(it + 1, "units, residual", np.abs(res).max())
    if np.abs(res).max() < 1e-9:
        for j2, cc in zip(sel, c):
            print("  G", names[j2][0], "V", names[j2][1], "out", round(cc, 4))
        break

# is the target in the span of all E-exact units at all?
c, *_ = np.linalg.lstsq(M.T, target, rcond=None)
print("span of all", len(M), "units: residual", np.abs(target - M.T @ c).max(), "rank", np.linalg.matrix_rank(M))

# basis pursuit: min sum |c| subject to M^T c = target (a vertex has <= 64 nonzeros)
try:
    from scipy.optimize import linprog
    n = len(M)
    A_eq = np.c_[M.T, -M.T]
    r = linprog(np.ones(2 * n), A_eq=A_eq, b_eq=target, bounds=(0, None), method="highs")
    cc = r.x[:n] - r.x[n:]
    nz = np.nonzero(np.abs(cc) > 1e-9)[0]
    print("L1 solution:", r.status, "units", len(nz), "residual", np.abs(target - M[nz].T @ cc[nz]).max())
except ImportError:
    print("no scipy")

"""Is there ANY quadratic polynomial P on {0,1,2}^4 with P = T (mod 2), P in [0, W]? (exact BFS)"""
import itertools, sys
import numpy as np
from scipy.linalg import qr
X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)
Qf = lambda b, c: ((b != 1) & (c == 1)).astype(int)
T0 = (Qf(X[:, 0], X[:, 1]) + Qf(X[:, 2], X[:, 3])) % 2
cols = [np.ones(81)] + [X[:, i] for i in range(4)] + [X[:, i] * X[:, j] for i in range(4) for j in range(i, 4)]
M = np.stack(cols, 1)
def feas(M, allowed):
    n, k = M.shape
    nall = np.array([len(a) for a in allowed])
    _, Rm, piv = qr((M * np.where(nall == 1, 1e3, 1.0)[:, None]).T, pivoting=True, mode="economic")
    dg = np.abs(np.diag(Rm)); r = int((dg > 1e-9 * dg[0]).sum()); piv = list(piv[:r])
    A = np.linalg.lstsq(M[piv].T, M.T, rcond=None)[0].T; A[np.abs(A) < 1e-10] = 0
    nz = A != 0; level = np.where(nz.any(1), r - 1 - np.argmax(nz[:, ::-1], 1), -1)
    pop = np.zeros((1, 0))
    for L in range(r):
        vals = np.array(allowed[piv[L]], float); m = pop.shape[0]
        pop = np.hstack([np.repeat(pop, len(vals), 0), np.tile(vals, m)[:, None]])
        rows = np.nonzero(level == L)[0]
        if len(rows):
            V = pop @ A[rows, :L + 1].T; ok = np.ones(len(pop), bool)
            for j, i in enumerate(rows):
                ok &= (np.abs(V[:, j:j+1] - np.array(allowed[i], float)[None]) < 1e-7).any(1)
            pop = pop[ok]
        if len(pop) == 0: return None
    return len(pop)
for W in (2, 3, 4):
    for flip in (0, 1):
        t = (T0 + flip) % 2
        allowed = [[k for k in range(W + 1) if k % 2 == t[i]] for i in range(81)]
        print("W", W, "flip", flip, "solutions:", feas(M, allowed))

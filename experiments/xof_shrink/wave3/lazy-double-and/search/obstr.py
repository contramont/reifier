"""Which sub-configurations of {0,1,2}^4 already rule out a quadratic polynomial P with
P in [0, W], P = T + flip (mod 2)?  (3-D slices, 2x2x2x2 boxes, 3x3x2x2 boxes)"""
import itertools
import numpy as np
from scipy.linalg import qr
X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)
Qf = lambda b, c: ((b != 1) & (c == 1)).astype(int)
T0 = (Qf(X[:, 0], X[:, 1]) + Qf(X[:, 2], X[:, 3])) % 2
cols = [np.ones(81)] + [X[:, i] for i in range(4)] + [X[:, i] * X[:, j] for i in range(4) for j in range(i, 4)]
MQ = np.stack(cols, 1)
def feas(M, allowed):
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
        if len(pop) == 0: return False
    return True
def obstructed(idx, W):
    for flip in (0, 1):
        t = (T0 + flip) % 2
        allowed = [[k for k in range(W + 1) if k % 2 == t[i]] for i in idx]
        if feas(MQ[idx], allowed):
            return False
    return True
for W in (2, 3):
    # 3-D slices: fix one coordinate
    res = []
    for ax in range(4):
        for v in range(3):
            idx = np.nonzero(X[:, ax] == v)[0]
            res.append(((ax, v), obstructed(idx, W)))
    print("W", W, "3-D slices obstructed:", [k for k, o in res if o])
    # 2x2x2x2 boxes (each coordinate in {e, e+1})
    ob = 0; tot = 0
    for e in itertools.product((0, 1), repeat=4):
        idx = np.nonzero(np.all((X >= np.array(e)) & (X <= np.array(e) + 1), 1))[0]
        tot += 1; ob += obstructed(idx, W)
    print("W", W, "2^4 boxes obstructed:", ob, "/", tot)
    ob = 0; tot = 0
    for full in itertools.combinations(range(4), 2):
        for e in itertools.product((0, 1), repeat=2):
            lo = np.zeros(4); hi = np.full(4, 2.0)
            rest = [a for a in range(4) if a not in full]
            lo[rest] = e; hi[rest] = np.array(e) + 1
            idx = np.nonzero(np.all((X >= lo) & (X <= hi), 1))[0]
            tot += 1; ob += obstructed(idx, W)
    print("W", W, "3x3x2x2 boxes obstructed:", ob, "/", tot, flush=True)

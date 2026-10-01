"""Faster qip2.py: pivot rows by QR with column pivoting on weighted M^T (rows with one
allowed value first). Question: one gated unit + a free affine pass term giving
Q(b1,c1) + Q(b2,c2) (mod 2) on E in {0,1,2}^4, Q(b,c) = [b != 1][c == 1], output integer in
[0, W] at all 81 points. Usage: qip2f.py W R B [single]  (single: target Q(b1,c1) only,
a sanity check that must find the known one-unit form)"""
import itertools, sys
import numpy as np
from scipy.linalg import qr

W, R, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
single = len(sys.argv) > 4
X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)
Q = lambda b, c: ((b != 1) & (c == 1)).astype(int)
T0 = Q(X[:, 0], X[:, 1]) % 2 if single else (Q(X[:, 0], X[:, 1]) + Q(X[:, 2], X[:, 3])) % 2
ones = np.ones((81, 1))
hits = 0
for T in (T0, 1 - T0):
    allowed = [[k for k in range(0, W + 1) if k % 2 == t] for t in T]
    nall = np.array([len(a) for a in allowed])
    wrow = np.where(nall == 1, 1e3, 1.0)
    for w in itertools.product(range(-R, R + 1), repeat=4):
        if not single and w[:2] > w[2:]:
            continue
        wv = np.array(w, float)
        for b0 in range(-B, B + 1):
            g = X @ wv + b0
            act = g > 0
            if not act.any():
                continue
            ga = (g * act)[:, None]
            M = np.hstack([ga * X, ga, X, ones])
            _, Rm, piv = qr((M * wrow[:, None]).T, pivoting=True, mode="economic")
            d = np.abs(np.diag(Rm))
            r = int((d > 1e-9 * max(1.0, d[0])).sum())
            piv = piv[:r]
            choices = [allowed[i] for i in piv]
            ncomb = int(np.prod([len(c) for c in choices]))
            if ncomb > 4096:
                continue
            Y = np.array(list(itertools.product(*choices)), float).T
            th, *_ = np.linalg.lstsq(M[piv], Y, rcond=None)
            Yall = M @ th
            Yr = np.round(Yall)
            good = ((np.abs(Yall - Yr).max(0) < 1e-7) & (Yr.min(0) >= -1e-9) & (Yr.max(0) <= W + 1e-9)
                    & (np.mod(Yr, 2) == T[:, None]).all(0))
            if good.any():
                j = int(np.argmax(good))
                hits += 1
                if hits <= 10:
                    print("W", W, "gate", w, b0, "theta", np.round(th[:, j], 4), flush=True)
print("hits", hits, flush=True)

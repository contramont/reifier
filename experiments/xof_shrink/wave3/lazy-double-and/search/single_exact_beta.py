"""Can ONE unit relu(w.(b,c) - tau + beta) v(b,c) plus an affine term equal Q(b,c) = [b!=1][c=1]
(or 1 - Q, or Q xor a linear parity) exactly on {0,1,2}^2, for real beta in [0,1)?
For fixed (w, tau), F = [A](s - tau + beta) v + l is linear in (v, l) for fixed beta: scan beta
over a fine grid and also solve for the beta at which the 9x6 system becomes consistent."""
import itertools
import numpy as np
X = np.array(list(itertools.product((0, 1, 2), repeat=2)), float)
Xt = np.hstack([X, np.ones((9, 1))])
b, c = X[:, 0], X[:, 1]
Q = ((b != 1) & (c == 1)).astype(float)
targets = {"Q": Q, "1-Q": 1 - Q}
for nm, lin in (("b", (b == 1)), ("c", (c == 1)), ("b+c", (b == 1) ^ (c == 1))):
    targets["Q^" + nm] = (Q.astype(int) ^ lin.astype(int)).astype(float)
    targets["1-Q^" + nm] = 1 - targets["Q^" + nm]
R = 6
best = {}
for w in itertools.product(range(-R, R + 1), repeat=2):
    if not any(w): continue
    s = X @ np.array(w, float)
    for tau in range(int(s.min()), int(s.max()) + 1):
        A = s >= tau
        for beta in np.linspace(0, 1, 2001)[:-1]:
            r = np.where(A, s - tau + beta, 0.0)
            M = np.hstack([r[:, None] * Xt, Xt])
            for nm, t in targets.items():
                sol = np.linalg.lstsq(M, t, rcond=None)[0]
                res = np.abs(M @ sol - t).max()
                if res < best.get(nm, (9,))[0]:
                    best[nm] = (res, w, tau, beta)
for nm, v in best.items():
    print(nm, "min residual %.3g" % v[0], v[1:])

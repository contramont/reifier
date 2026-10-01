"""One unit + free affine: F in {0,1}, F = Q(b,c) + lin (mod 2) for each linear parity lin in
{0, [b=1], [c=1], [b=1]+[c=1]} (the linear part moved into the zigzag). Exact over integer
gates |w| <= R (int knots) and the rel relaxation (all knot offsets)."""
import itertools, sys
import numpy as np
X = np.array(list(itertools.product((0, 1, 2), repeat=2)), float)
Xt = np.hstack([X, np.ones((9, 1))])
b, c = X[:, 0], X[:, 1]
Q = ((b != 1) & (c == 1)).astype(int)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
hits = 0
for name, lin in (("0", 0 * b), ("b", (b == 1)), ("c", (c == 1)), ("b+c", (b == 1).astype(int) + (c == 1))):
    for flip in (0, 1):
        t = (Q + lin.astype(int) + flip) % 2
        for w in itertools.product(range(-R, R + 1), repeat=2):
            if not any(w): continue
            s = X @ np.array(w, float)
            for tau in range(int(s.min()), int(s.max()) + 1):
                A = (s >= tau).astype(float)
                for M in (np.hstack([(np.maximum(0, s - tau))[:, None] * Xt, Xt]),
                          np.hstack([A[:, None] * Xt, (A * s)[:, None] * Xt, Xt])):
                    sol, res, rk, _ = np.linalg.lstsq(M, t.astype(float), rcond=None)
                    if np.abs(M @ sol - t).max() < 1e-7:
                        hits += 1
                        print("HIT", name, flip, w, tau, M.shape[1])
print("hits", hits)

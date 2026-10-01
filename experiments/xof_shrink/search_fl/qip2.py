"""One gated unit + a free affine pass term for Q1 + Q2 (mod 2), two lazy-chi AND terms of
E-coded theta bits (E in {0,1,2}, theta = [E == 1]): Q(b, c) = [Eb != 1][Ec == 1].
y(x) = max(0, g(x)) v(x) + pass(x), integer on all 81 points, y = T (mod 2), y in [0, W].
Exhaustive over integer gates w in [-R, R]^4, bias in [-B, B]; v, pass solved exactly
(pivot enumeration). Usage: qip2.py W R B"""
import itertools, sys
import numpy as np

W, R, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)  # b1 c1 b2 c2
Q = lambda b, c: ((b != 1) & (c == 1)).astype(int)
T0 = (Q(X[:, 0], X[:, 1]) + Q(X[:, 2], X[:, 3])) % 2
ones = np.ones((81, 1))
hits = 0
for T in (T0, 1 - T0):
    allowed = [[k for k in range(0, W + 1) if k % 2 == t] for t in T]
    for w in itertools.product(range(-R, R + 1), repeat=4):
        if w[:2] > w[2:]:
            continue  # symmetry (swap the two pairs)
        for b0 in range(-B, B + 1):
            g = X @ np.array(w, float) + b0
            act = (g > 0).astype(float)
            if act.sum() == 0:
                continue
            ga = (g * act)[:, None]
            M = np.hstack([ga * X, ga, X, ones])  # v (5), pass (5)
            # pivot rows: greedy independent rows, preferring points with one allowed value
            order = sorted(range(81), key=lambda i: len(allowed[i]))
            piv, basis = [], np.zeros((0, 10))
            for i in order:
                cand = np.vstack([basis, M[i]])
                if np.linalg.matrix_rank(cand, tol=1e-9) > len(piv):
                    piv.append(i); basis = cand
                if len(piv) == 10:
                    break
            r = len(piv)
            Mp = M[piv]
            choices = [allowed[i] for i in piv]
            ncomb = int(np.prod([len(c) for c in choices]))
            if ncomb > 4096:
                continue
            Y = np.array(list(itertools.product(*choices)), float).T  # r x ncomb
            th, *_ = np.linalg.lstsq(Mp, Y, rcond=None)
            Yall = M @ th
            ok_int = np.abs(Yall - np.round(Yall)).max(0) < 1e-7
            Yr = np.round(Yall)
            ok_rng = (Yr.min(0) >= -1e-9) & (Yr.max(0) <= W + 1e-9)
            ok_par = (np.mod(Yr, 2) == T[:, None]).all(0)
            good = ok_int & ok_rng & ok_par
            if good.any():
                j = int(np.argmax(good))
                hits += 1
                if hits <= 10:
                    print("W", W, "gate", w, b0, "theta", np.round(th[:, j], 4), "T flipped", T is not T0, flush=True)
print("hits", hits)

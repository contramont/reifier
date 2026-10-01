"""Chained pair with a free joint pass term: units u_A(E0,E1), u_B(E1,E2) are single-AND units
(each, with SOME affine term, a valid window-2 single AND, as the other counts need), and the
chained count uses u_A + u_B + l(E0,E1,E2) with l free. Window <= 2 wanted (count 33 -> 31)."""
import itertools, sys
import numpy as np
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
X = np.array(list(itertools.product((0, 1, 2), repeat=2)), float)
Xt = np.hstack([X, np.ones((9, 1))])
b, c = X[:, 0], X[:, 1]
Q = ((b != 1) & (c == 1)).astype(int)
units = {}
for w in itertools.product(range(-R, R + 1), repeat=2):
    if not any(w): continue
    s = X @ np.array(w, float)
    for tau2 in range(int(2 * s.min()) - 1, int(2 * s.max()) + 2):
        tau = tau2 / 2
        r = np.maximum(0, s - tau)
        if not r.any(): continue
        M = np.hstack([r[:, None] * Xt, Xt])
        for flip in (0, 1):
            t = (Q + flip) % 2
            allowed = [[k for k in range(3) if k % 2 == t[i]] for i in range(9)]
            for vals in itertools.product(*allowed):
                y = np.array(vals, float)
                sol = np.linalg.lstsq(M, y, rcond=None)[0]
                if np.abs(M @ sol - y).max() < 1e-7:
                    # the unit part (up to the affine part, which the chained count re-chooses)
                    u = r * (Xt @ sol[:3])
                    # canonical: remove its affine projection so equal-up-to-affine units merge
                    P = Xt @ np.linalg.lstsq(Xt, u, rcond=None)[0]
                    key = tuple(np.round(u - P, 6))
                    units.setdefault(key, (w, tau, flip, u))
U = list(units.values())
print(len(U), "distinct single-AND units (mod affine)")
E3 = np.array(list(itertools.product((0, 1, 2), repeat=3)), float)
X3 = np.hstack([E3, np.ones((27, 1))])
QA = ((E3[:, 0] != 1) & (E3[:, 1] == 1)).astype(int)
QB = ((E3[:, 1] != 1) & (E3[:, 2] == 1)).astype(int)
T = (QA + QB) % 2
idxA = (E3[:, 0] * 3 + E3[:, 1]).astype(int)
idxB = (E3[:, 1] * 3 + E3[:, 2]).astype(int)
best = {2: [], 3: []}
for (wa, ta, fa, ua) in U:
    for (wb, tb, fb, ub) in U:
        base = ua[idxA] + ub[idxB]
        for W in (2,):
            for flip in (0, 1):
                t = (T + flip) % 2
                # F = base + l, l affine (4 unknowns), F in allowed; enumerate F on 4 pivot points
                allowed = [[k for k in range(W + 1) if k % 2 == t[i]] for i in range(27)]
                piv = [0, 9, 3, 1]  # (0,0,0), (1,0,0), (0,1,0), (0,0,1): affinely independent
                for vals in itertools.product(*[allowed[i] for i in piv]):
                    l = np.linalg.solve(X3[piv], np.array(vals, float) - base[piv])
                    F = base + X3 @ l
                    Fr = np.round(F)
                    if np.abs(F - Fr).max() < 1e-7 and Fr.min() >= 0 and Fr.max() <= W and (np.mod(Fr, 2) == t).all():
                        best[W].append((wa, ta, fa, wb, tb, fb, flip, tuple(int(v) for v in Fr)))
print("window-2 chained solutions:", len(best[2]))
for s in best[2][:5]:
    print(s)

"""Chained pair: the own bit (Eb = E1, Ec = E2) and the column bit that shares E1 (Eb = E0, Ec = E1).
Enumerate every one-unit + affine single-AND form (window 2, both parities, integer gates
|w| <= R, knots at integer or half-integer offsets) and test whether h_A(E0, E1) + h_B(E1, E2)
can have window <= 2 (count range 33 -> 31 at no unit cost)."""
import itertools, sys
import numpy as np
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
X = np.array(list(itertools.product((0, 1, 2), repeat=2)), float)
Xt = np.hstack([X, np.ones((9, 1))])
b, c = X[:, 0], X[:, 1]
Q = ((b != 1) & (c == 1)).astype(int)
forms = set()
for w in itertools.product(range(-R, R + 1), repeat=2):
    if not any(w): continue
    s = X @ np.array(w, float)
    for tau2 in range(int(2 * s.min()) - 1, int(2 * s.max()) + 2):
        tau = tau2 / 2
        r = np.maximum(0, s - tau)
        M = np.hstack([r[:, None] * Xt, Xt])  # 6 unknowns, 9 points
        for flip in (0, 1):
            t = (Q + flip) % 2
            allowed = [[k for k in range(3) if k % 2 == t[i]] for i in range(9)]
            for vals in itertools.product(*allowed):
                y = np.array(vals, float)
                sol = np.linalg.lstsq(M, y, rcond=None)[0]
                if np.abs(M @ sol - y).max() < 1e-7:
                    forms.add(tuple(int(v) for v in y))
forms = sorted(forms)
print(len(forms), "distinct single-AND tables (b-major, 3x3)")
best = 9
bestp = []
E = list(itertools.product((0, 1, 2), repeat=3))
for fa in forms:
    FA = np.array(fa).reshape(3, 3)
    for fb in forms:
        FB = np.array(fb).reshape(3, 3)
        vals = [FA[e0, e1] + FB[e1, e2] for e0, e1, e2 in E]
        wdt = max(vals) - min(vals)
        if wdt < best:
            best, bestp = wdt, [(fa, fb)]
        elif wdt == best:
            bestp.append((fa, fb))
print("best chained window", best, "examples", bestp[:4])

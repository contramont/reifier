import itertools, sys
import numpy as np
exec(open("chain2.py").read().split("E3 = np.array")[0])
E3 = np.array(list(itertools.product((0, 1, 2), repeat=3)), float)
X3 = np.hstack([E3, np.ones((27, 1))])
QA = ((E3[:, 0] != 1) & (E3[:, 1] == 1)).astype(int)
QB = ((E3[:, 1] != 1) & (E3[:, 2] == 1)).astype(int)
idxA = (E3[:, 0] * 3 + E3[:, 1]).astype(int)
idxB = (E3[:, 1] * 3 + E3[:, 2]).astype(int)
piv = [0, 9, 3, 1]
T = (QA + QB) % 2
W = 3
# list the units: gate, knot, value vector (so that unit = relu(w.(b,c) - tau) * (v0 + vb b + vc c))
for (wa, ta, fa, ua) in U:
    pass
sols = []
for (wa, ta, fa, ua) in U:
    for (wb, tb, fb, ub) in U:
        base = ua[idxA] + ub[idxB]
        for flip in (0, 1):
            t = (T + flip) % 2
            allowed = [[k for k in range(W + 1) if k % 2 == t[i]] for i in range(27)]
            for vals in itertools.product(*[allowed[i] for i in piv]):
                l = np.linalg.solve(X3[piv], np.array(vals, float) - base[piv])
                F = base + X3 @ l
                Fr = np.round(F)
                if np.abs(F - Fr).max() < 1e-7 and Fr.min() >= 0 and Fr.max() <= W and (np.mod(Fr, 2) == t).all():
                    sols.append((wa, ta, wb, tb, flip, np.round(l, 4).tolist(), np.round(ua, 3).tolist(), np.round(ub, 3).tolist()))
for s in sols:
    print(s[:6])
print(len(sols))
print("unit A table (b-major)", sols[0][6]); print("unit B table", sols[0][7])

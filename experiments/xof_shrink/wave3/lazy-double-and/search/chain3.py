"""Chained pair, minimal window over single-AND units (mod affine) with a free joint pass term,
for targets T = Q(E0,E1)+Q(E1,E2) (chain) and T + E0 (own Ea moved out of the zigzag)."""
import itertools, sys
import numpy as np
exec(open("chain2.py").read().split("E3 = np.array")[0])  # builds U (single-AND units)
E3 = np.array(list(itertools.product((0, 1, 2), repeat=3)), float)
X3 = np.hstack([E3, np.ones((27, 1))])
QA = ((E3[:, 0] != 1) & (E3[:, 1] == 1)).astype(int)
QB = ((E3[:, 1] != 1) & (E3[:, 2] == 1)).astype(int)
idxA = (E3[:, 0] * 3 + E3[:, 1]).astype(int)
idxB = (E3[:, 1] * 3 + E3[:, 2]).astype(int)
piv = [0, 9, 3, 1]
for name, T in (("chain", (QA + QB) % 2), ("chain+E0", (QA + QB + (E3[:, 0] == 1)) % 2)):
    for W in (2, 3, 4):
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
                            sols.append((wa, ta, fa, wb, tb, fb, flip, np.round(l, 4).tolist()))
        print(name, "W", W, "solutions", len(sols), sols[:2], flush=True)
        if sols:
            break

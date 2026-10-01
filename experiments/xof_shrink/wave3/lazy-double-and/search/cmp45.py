import sys
sys.argv = ["dand5.py", "2", "--single", "--only12"]
import dand5 as E
D = E.D
import numpy as np
A, B = set(), set()
n = len(D.GATES)
for i in range(n):
    for flip in D.FLIPS:
        cand = E.survivors(i, flip)
        if cand is None:
            cand = range(n)
        for j in cand:
            if j < i: continue
            M = np.hstack([D.GATES[i][2], D.GATES[j][2], D.Xt])
            act = np.stack([D.GATES[i][3], D.GATES[j][3]], 1).astype(int)
            out = D.feasible(M, flip, act)
            if out is not None and not isinstance(out, str): B.add((i, j, flip))
        for j in range(i, n):
            M = np.hstack([D.GATES[i][2], D.GATES[j][2], D.Xt])
            act = np.stack([D.GATES[i][3], D.GATES[j][3]], 1).astype(int)
            out = D.feasible(M, flip, act)
            if out is not None and not isinstance(out, str): A.add((i, j, flip))
print("full", len(A), "filtered", len(B), "missing", len(A - B), "extra", len(B - A))

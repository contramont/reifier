"""Planted-solution sanity check for the exact per-pair solver: the trivial double AND
h(b1,c1) + h(b2,c2), h = 2b + (b+c)(1-b), has window 4 and must be found at W = 4 but not W = 3."""
import sys
for W in ("4", "3"):
    sys.argv = ["dand4.py", W, "2"]
    for m in [k for k in list(sys.modules) if k == "dand4"]:
        del sys.modules[m]
    import dand4 as D
    import numpy as np
    s1 = D.X @ np.array([1, 1, 0, 0.0]); s2 = D.X @ np.array([0, 0, 1, 1.0])
    r1 = np.maximum(0, s1); r2 = np.maximum(0, s2)
    M = np.hstack([r1[:, None] * D.Xt, r2[:, None] * D.Xt, D.Xt])
    act = np.stack([r1 > 0, r2 > 0], 1).astype(int)
    res = [D.feasible(M, f, act) for f in D.FLIPS]
    print("W", W, ["found" if (r is not None and not isinstance(r, str)) else r for r in res])

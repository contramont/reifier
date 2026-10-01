"""Cross-check the BFS verdicts of dand4.feasible against an independent MILP (HiGHS) on random
gate pairs: F = M c, F = t + 2k with integer k (window W), both flips."""
import os, sys, random
os.environ["OMP_NUM_THREADS"] = "1"
W = int(sys.argv[1]); R = sys.argv[2]; N = int(sys.argv[3])
sys.argv = [sys.argv[0], str(W), R]
import dand4 as D
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
random.seed(1)
def milp_feas(M, flip):
    t = (D.T0 + flip) % 2
    n, k = M.shape
    kmax = (W - t) // 2  # F = t + 2 kk, kk in [0, kmax]
    # variables: c (k, free in [-1e4, 1e4]), kk (n, integer)
    A = np.hstack([M, -2 * np.eye(n)])
    cons = LinearConstraint(A, t.astype(float), t.astype(float))
    lb = np.r_[-1e4 * np.ones(k), np.zeros(n)]
    ub = np.r_[1e4 * np.ones(k), kmax.astype(float)]
    integ = np.r_[np.zeros(k), np.ones(n)]
    res = milp(np.zeros(k + n), constraints=cons, integrality=integ, bounds=Bounds(lb, ub), options={"time_limit": 20})
    return res.status  # 0 optimal (feasible), 2 infeasible, 1 time limit
agree = dis = tl = pos = 0
n = len(D.GATES)
for _ in range(N):
    i, j = random.randrange(n), random.randrange(n)
    M = np.hstack([D.GATES[i][2], D.GATES[j][2], D.Xt])
    act = np.stack([D.GATES[i][3], D.GATES[j][3]], 1).astype(int)
    for f in D.FLIPS:
        out = D.feasible(M, f, act)
        b = out is not None and not isinstance(out, str)
        st = milp_feas(M, f)
        if st == 1:
            tl += 1; continue
        m = st == 0
        pos += m
        if m == b: agree += 1
        else:
            dis += 1; print("DISAGREE", D.GATES[i][:2], D.GATES[j][:2], f, "bfs", b, "milp", st, flush=True)
print(f"W={W} R={R} checks agree={agree} disagree={dis} milp_timeouts={tl} milp_feasible={pos}")

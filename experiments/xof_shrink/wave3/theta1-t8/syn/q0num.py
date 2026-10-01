# Numeric search for a T=n pool-of-3 form (q0): pool z_j(T) = relu(s_j (T - tau_j)) (P_j T + Q_j)
# with sum_j z_j = parity(T) (gamma absorbed), per-bit units
#   u_j(A,T) = relu(s_j (T - tau_j) + al_j A) (mu_j (P_j T + Q_j) + d_j A)
# with sum_j [u_j(1,T) - u_j(0,T)] = (-1)^T.  Cost: 3 per live bit + 3 per column.
import numpy as np, sys, time
from scipy.optimize import least_squares
n = int(sys.argv[1]); nrest = int(sys.argv[2]); seed = int(sys.argv[3])
ts = np.arange(n + 1, dtype=float); par = ts % 2; alt = (-1.0) ** ts
rng = np.random.default_rng(seed)
def resid(x, s):
    tau, al, P, Q, mu, d = x[0:3], x[3:6], x[6:9], x[9:12], x[12:15], x[15:18]
    g0 = s[:, None] * (ts[None, :] - tau[:, None])          # (3, n+1)
    g1 = g0 + al[:, None]
    v0 = P[:, None] * ts[None, :] + Q[:, None]
    z = np.maximum(0, g0) * v0
    rP = z.sum(0) - par
    u1 = np.maximum(0, g1) * (mu[:, None] * v0 + d[:, None])
    u0 = mu[:, None] * z
    rD = (u1 - u0).sum(0) - alt
    return np.concatenate([rP, rD])
hits = []
t0 = time.time(); best = 9
for it in range(nrest):
    s = rng.choice([-1.0, 1.0], 3)
    x0 = np.concatenate([rng.uniform(-1, n + 1, 3), rng.uniform(-n, n, 3), rng.normal(0, 2, 6), rng.normal(0, 2, 3), rng.normal(0, 3, 3)])
    try:
        r = least_squares(resid, x0, args=(s,), method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=600)
    except Exception:
        continue
    m = np.abs(resid(r.x, s)).max()
    best = min(best, m)
    if m < 1e-9:
        x = r.x
        # reject degenerate: knots too close to lattice points without being on them
        g0 = s[:, None] * (ts[None, :] - x[0:3, None]); g1 = g0 + x[3:6, None]
        gv = np.abs(np.concatenate([g0.ravel(), g1.ravel()]))
        mn = gv[gv > 1e-6].min() if (gv > 1e-6).any() else 0
        near0 = ((gv > 1e-9) & (gv < 1e-3)).sum()
        hits.append((s.tolist(), x.tolist(), mn, int(near0)))
        print("HIT", it, m, "s", s.tolist(), "tau", np.round(x[0:3], 5).tolist(), "al", np.round(x[3:6], 5).tolist(),
              "minabsgate", mn, "near0", int(near0), "maxcoef", np.abs(x[6:]).max(), flush=True)
    if it % 500 == 0:
        print("it", it, "hits", len(hits), "best", best, round(time.time() - t0, 1), flush=True)
print("DONE", len(hits), best)
import pickle; pickle.dump(hits, open(f"q0num_n{n}_s{seed}.pkl", "wb"))

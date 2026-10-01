# Variable projection for the T=n pool-of-3 form (q0): only the 6 knots (tau_j at A=0, tau'_j at A=1)
# and the 3 directions are nonlinear. (P): parity = sum_j r_j (P_j T + Q_j)  -> lstsq for (P, Q)
# (D): sum_j [r'_j (lam_j v_j + d_j) - lam_j r_j v_j] = (-1)^T  -> lstsq for (lam, d), v_j = P_j T + Q_j
import numpy as np, sys, time, pickle
from scipy.optimize import least_squares
n = int(sys.argv[1]); nrest = int(sys.argv[2]); seed = int(sys.argv[3])
extra = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # extra pool hinges (q)
ts = np.arange(n + 1, dtype=float); par = ts % 2; alt = (-1.0) ** ts
rng = np.random.default_rng(seed)
def parts(x, s, se):
    tau, taup, xt = x[0:3], x[3:6], x[6:6 + extra]
    r = np.maximum(0, s[:, None] * (ts[None, :] - tau[:, None]))
    rp = np.maximum(0, s[:, None] * (ts[None, :] - taup[:, None]))
    rx = np.maximum(0, se[:, None] * (ts[None, :] - xt[:, None])) if extra else np.zeros((0, n + 1))
    MP = np.concatenate([r.T, (r * ts).T, rx.T, (rx * ts).T], 1)
    c, *_ = np.linalg.lstsq(MP, par, rcond=None)
    rP = MP @ c - par
    P, Q = c[0:3], c[3:6]
    v = P[:, None] * ts[None, :] + Q[:, None]
    MD = np.concatenate([(rp * v - r * v).T, rp.T], 1)
    e, *_ = np.linalg.lstsq(MD, alt, rcond=None)
    rD = MD @ e - alt
    return rP, rD, c, e
def resid(x, s, se):
    rP, rD, _, _ = parts(x, s, se)
    return np.concatenate([rP, rD])
hits = []; t0 = time.time(); best = 9
for it in range(nrest):
    s = rng.choice([-1.0, 1.0], 3); se = rng.choice([-1.0, 1.0], extra)
    x0 = np.concatenate([rng.uniform(-1.5, n + 0.5, 3), rng.uniform(-3, n + 3, 3), rng.uniform(-1.5, n + 0.5, extra)])
    try:
        r = least_squares(resid, x0, args=(s, se), method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200, diff_step=1e-7)
    except Exception:
        continue
    m = np.abs(resid(r.x, s, se)).max(); best = min(best, m)
    if m < 1e-8:
        rP, rD, c, e = parts(r.x, s, se)
        hits.append((s.tolist(), se.tolist(), r.x.tolist(), c.tolist(), e.tolist()))
        print("HIT", it, m, "s", s.tolist(), "tau", np.round(r.x[0:3], 5).tolist(), "taup", np.round(r.x[3:6], 5).tolist(),
              "xt", np.round(r.x[6:], 5).tolist(), "maxcoef", round(float(np.abs(np.concatenate([c, e])).max()), 3), flush=True)
    if it % 2000 == 0:
        print("it", it, "hits", len(hits), "best", best, round(time.time() - t0, 1), flush=True)
print("DONE", len(hits), best)
pickle.dump(hits, open(f"q0vp_n{n}_x{extra}_s{seed}.pkl", "wb"))

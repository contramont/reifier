# sanity/positive control: numeric search for the T=n pool form with 3 slices + E extra units
import numpy as np, sys, time
from scipy.optimize import least_squares
n = int(sys.argv[1]); nrest = int(sys.argv[2]); seed = int(sys.argv[3]); E = int(sys.argv[4])
ts = np.arange(n + 1, dtype=float); par = ts % 2; alt = (-1.0) ** ts
rng = np.random.default_rng(seed)
def resid(x, s, se):
    tau, al, P, Q, mu, d = x[0:3], x[3:6], x[6:9], x[9:12], x[12:15], x[15:18]
    xt, xP, xQ = x[18:18 + E], x[18 + E:18 + 2 * E], x[18 + 2 * E:18 + 3 * E]
    g0 = s[:, None] * (ts[None, :] - tau[:, None]); g1 = g0 + al[:, None]
    v0 = P[:, None] * ts[None, :] + Q[:, None]
    z = np.maximum(0, g0) * v0
    X = (np.maximum(0, se[:, None] * (ts[None, :] - xt[:, None])) * (xP[:, None] * ts[None, :] + xQ[:, None])).sum(0)
    rP = z.sum(0) + X - par
    u1 = np.maximum(0, g1) * (mu[:, None] * v0 + d[:, None]); u0 = mu[:, None] * z
    rD = (u1 - u0).sum(0) - alt
    return np.concatenate([rP, rD])
hits = []; t0 = time.time(); best = 9
for it in range(nrest):
    s = rng.choice([-1.0, 1.0], 3); se = rng.choice([-1.0, 1.0], E)
    x0 = np.concatenate([rng.uniform(-1, n + 1, 3), rng.uniform(-n, n, 3), rng.normal(0, 2, 6), rng.normal(0, 2, 3), rng.normal(0, 3, 3),
                         rng.uniform(-1, n + 1, E), rng.normal(0, 2, 2 * E)])
    r = least_squares(resid, x0, args=(s, se), method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=600)
    m = np.abs(resid(r.x, s, se)).max(); best = min(best, m)
    if m < 1e-9:
        hits.append((s.tolist(), se.tolist(), r.x.tolist()))
        print("HIT", it, m, "tau", np.round(r.x[0:3], 4).tolist(), "al", np.round(r.x[3:6], 4).tolist(), "xt", np.round(r.x[18:18 + E], 4).tolist(), flush=True)
    if it % 500 == 0:
        print("it", it, "hits", len(hits), "best", best, round(time.time() - t0, 1), flush=True)
print("DONE", len(hits), best)

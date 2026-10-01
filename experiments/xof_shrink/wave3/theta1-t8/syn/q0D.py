# For fixed 3-unit (+const) parity pools on [0,n]: solve (D) with continuous A=1 knots per interval.
import numpy as np, sys, pickle, itertools
from scipy.optimize import least_squares
n = int(sys.argv[1]); files = sys.argv[2:]
ts = np.arange(n + 1, dtype=float); par = ts % 2; alt = (-1.0) ** ts
pools = set()
for fn in files:
    ps, _ = pickle.load(open(fn, "rb"))
    den = int(fn.split("_d")[1].split("_")[0])
    knots = [k / den for k in range(int(np.floor(-2 * den)), int(np.ceil(7.5 * den)) + 1)]
    gates = []
    for s in (1.0, -1.0):
        for t in knots:
            if np.maximum(0, s * (ts - t)).any():
                gates.append((s, t))
    for p in ps:
        pools.add(tuple(gates[i] for i in p))
print("pools", len(pools), flush=True)
# intervals for A=1 knots: (lo, hi) boxes; the knot moves inside, integer endpoints included
ivals = [(-6.0, -1.0), (-1.0, 0.0)] + [(k - 1.0, float(k)) for k in range(1, n + 1)] + [(float(n), n + 1.0), (n + 1.0, n + 6.0)]
best_all = []
for pool in sorted(pools):
    s = np.array([g[0] for g in pool]); tau = np.array([g[1] for g in pool])
    r0 = np.maximum(0, s[:, None] * (ts[None, :] - tau[:, None]))
    M = np.concatenate([r0.T, (r0 * ts).T, np.ones((n + 1, 1))], 1)
    sol = np.linalg.lstsq(M, par, rcond=None)[0]
    assert np.abs(M @ sol - par).max() < 1e-9
    C0, C1 = sol[0:3], sol[3:6]   # z_j = r0_j (C0_j + C1_j T)
    v = C1[:, None] * ts[None, :] + C0[:, None]
    z = r0 * v
    def res(t1):
        r1 = np.maximum(0, s[:, None] * (ts[None, :] - t1[:, None]))
        MD = np.concatenate([(r1 * v - z).T, r1.T], 1)
        e = np.linalg.lstsq(MD, alt, rcond=None)[0]
        return MD @ e - alt
    bestp = (9, None)
    for iv in itertools.product(range(len(ivals)), repeat=3):
        lo = np.array([ivals[i][0] for i in iv]); hi = np.array([ivals[i][1] for i in iv])
        for x0 in (0.5 * (lo + hi), lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo), hi - 1e-9):
            try:
                r = least_squares(res, np.clip(x0, lo + 1e-12, hi), bounds=(lo, hi + 1e-12), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
            except Exception:
                continue
            m = np.abs(res(r.x)).max()
            if m < bestp[0]:
                bestp = (m, r.x.tolist())
            if m < 1e-9:
                print("HIT pool", pool, "A=1 knots", r.x.tolist(), m, flush=True)
                break
    print("pool", pool, "best residual", bestp, flush=True)

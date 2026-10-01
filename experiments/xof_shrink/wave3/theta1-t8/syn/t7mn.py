# numeric check of T=7 merged-slice candidates: pool = {z_i + z_j (gamma_i = gamma_j), z_k, X}
import pickle, sys, numpy as np
from scipy.optimize import least_squares
T = 7; n1 = 8; ts = np.arange(n1, dtype=float); par = ts % 2; sig = (-1.0) ** ts
d = pickle.load(open('/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof2/av/units-per-bit/syn/d_p3_T7_1,-1,2,-2.pkl','rb'))
keys, cands = d['keys'], d['cands']
R = np.array(keys, dtype=float); R0all, R1all = R[:, :n1], R[:, n1:]
def norm(r):
    s = r.sum()
    return tuple((r / s).round(9)) if s > 0 else None
cl = pickle.load(open('t7merge_cands.pkl', 'rb'))
rng = np.random.default_rng(0)
nhit = 0
for tr, dr, T0 in cl:
    R0 = R0all[list(tr)]; R1 = R1all[list(tr)]
    Cd = np.concatenate([np.stack([R1[k], (R1[k] - R0[k]) * ts, R1[k] - R0[k]], 1) for k in range(3)], 1)
    U, sv, Vt = np.linalg.svd(Cd)
    rank = (sv > 1e-9 * sv[0]).sum()
    p0 = np.linalg.lstsq(Cd, sig, rcond=None)[0]
    N = Vt[rank:].T  # null space (9, nd)
    ks = [norm(R0all[j]) for j in tr]
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3) if ks[i] is not None and ks[i] == ks[j]]
    mask = (ts >= T0) if dr > 0 else (ts <= T0)
    for (i, j) in pairs:
        k = 3 - i - j
        def res(z):
            tau, t = z[:-1], z[-1]
            p = p0 + (N @ tau if N.shape[1] else 0)
            ph = [R0[m] * (p[3 * m + 1] * ts + p[3 * m + 2]) for m in range(3)]
            h = np.where(mask, dr * (ts - t), 0.0)
            M = np.stack([ph[i] + ph[j], ph[k], h, h * ts], 1)
            x = np.linalg.lstsq(M, par, rcond=None)[0]
            return M @ x - par
        lo, hi = (T0 - 1, T0) if dr > 0 else (T0, T0 + 1)
        best = 9
        for rs in range(12):
            z0 = np.concatenate([rng.normal(0, 3, N.shape[1]), [rng.uniform(lo, hi)]])
            lb = np.concatenate([-np.inf * np.ones(N.shape[1]), [lo + 1e-9]]); ub = np.concatenate([np.inf * np.ones(N.shape[1]), [hi]])
            try:
                r = least_squares(res, z0, bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400)
            except Exception:
                continue
            m = np.abs(res(r.x)).max(); best = min(best, m)
            if m < 1e-9:
                nhit += 1
                print("HIT", [cands[keys[q]] for q in tr], dr, T0, (i, j), r.x.tolist(), flush=True)
                break
print("DONE hits", nhit, "cands", len(cl))

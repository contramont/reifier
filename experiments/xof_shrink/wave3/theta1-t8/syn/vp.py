# Variable-projection random-restart search for column-shared round-1 theta forms.
# theta(A,T) = parity(A+T), A in {0,1}, T in [0,n].
# Live bit: sum_{i<K} relu(a_i A + b_i T + c_i)(d_i A + e_i T + f_i) + sum_{j<m} beta_j Z_j(T)
# Zero lane: sum_j gamma_j Z_j(T) = parity(T);  Z_j(T) = relu(g_j T + h_j)(p_j T + q_j)
# Cost per column (L live bits): L*K + m.  mode "g5".
import numpy as np, sys, time
from scipy.optimize import least_squares
n, K, m, nrest, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
tol = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-9
ts = np.arange(n + 1, dtype=float)
par = ts % 2
A2 = np.concatenate([np.zeros(n + 1), np.ones(n + 1)])
T2 = np.concatenate([ts, ts])
tgt_live = (A2 + T2) % 2

def unpack(th):
    pg = th[:3 * K].reshape(K, 3)
    qg = th[3 * K:3 * K + 2 * m].reshape(m, 2)
    qv = th[3 * K + 2 * m:].reshape(m, 2)
    return pg, qg, qv

def mats(th):
    pg, qg, qv = unpack(th)
    Z = np.maximum(0, qg[:, 0:1] * ts + qg[:, 1:2]) * (qv[:, 0:1] * ts + qv[:, 1:2])  # (m, n+1)
    cols = []
    for a, b, c in pg:
        r = np.maximum(0, a * A2 + b * T2 + c)
        cols += [r * A2, r * T2, r]
    Ml = np.concatenate([np.array(cols).T, np.concatenate([Z, Z], 1).T], 1)  # (2(n+1), 3K+m)
    return Z.T, Ml

def resid(th):
    Zt, Ml = mats(th)
    x, *_ = np.linalg.lstsq(Zt, par, rcond=None)
    r0 = Zt @ x - par
    y, *_ = np.linalg.lstsq(Ml, tgt_live, rcond=None)
    r1 = Ml @ y - tgt_live
    return np.concatenate([r0, r1])

rng = np.random.default_rng(seed)
found = []
t0 = time.time()
best = 1e9
for it in range(nrest):
    pg = np.stack([rng.integers(-6, 7, K), rng.choice([-3, -2, -1, 1, 2, 3], K), rng.integers(-3 * n, 3 * n, K)], 1).astype(float)
    pg += rng.normal(0, 0.3, pg.shape)
    qg = np.stack([rng.choice([-3, -2, -1, 0, 1, 2, 3], m), rng.integers(-3 * n, 3 * n, m)], 1).astype(float)
    qg[:, 1] = np.where(qg[:, 0] == 0, 1.0, qg[:, 1])
    qg += rng.normal(0, 0.3, qg.shape) * (qg[:, :1] != 0)
    qv = rng.normal(0, 2, (m, 2))
    th = np.concatenate([pg.ravel(), qg.ravel(), qv.ravel()])
    try:
        sol = least_squares(resid, th, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400 * len(th))
    except Exception as e:
        continue
    r = np.abs(resid(sol.x)).max()
    best = min(best, r)
    if r < tol:
        found.append(sol.x.copy())
        pg, qg, qv = unpack(sol.x)
        print("HIT", it, r, "pg", np.round(pg, 4).tolist(), "qg", np.round(qg, 4).tolist(), "qv", np.round(qv, 4).tolist(), flush=True)
    if it % 200 == 0:
        print("it", it, "hits", len(found), "best", best, round(time.time() - t0, 1), flush=True)
print("DONE n", n, "K", K, "m", m, "hits", len(found), "best", best)
np.save(f"vp_n{n}_K{K}_m{m}_s{seed}.npy", np.array(found) if found else np.zeros((0,)))

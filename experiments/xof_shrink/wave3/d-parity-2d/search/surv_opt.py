"""Continuous local search seeded by every relaxed-test survivor: optimise the 4 gates (a,b,c)
(12 reals) to minimise the EXACT residual min_v ||c0 + sum relu(g_k) v_k - t||, starting inside the
active-set cones. Evidence (not proof) that no real-gate 4-unit form exists near any survivor."""
import sys, json, time, math
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from gen_active import gen
from lines import PTS
X = np.array([p[0] for p in PTS], float); Y = np.array([p[1] for p in PTS], float)
TGT = (X + Y) % 2
acts = gen(10)
items = list(acts.items())
# all representatives per active set (for varied starts)
reps = {}
for a in range(-10, 11):
    for b in range(-10, 11):
        if (a, b) == (0, 0) or math.gcd(a, b) != 1: continue
        vals = [a * x + b * y for x, y in PTS]
        for m in range(min(vals) - 1, max(vals) + 1):
            c = -m - 0.5
            S = tuple(int(a * x + b * y + c > 0) for x, y in PTS)
            reps.setdefault(S, []).append((a, b, c))
REPS = [np.array(reps[k], float) for k, _ in items]
def resid_vec(theta):
    g = theta.reshape(4, 3)
    G = g[:, 0:1] * X[None] + g[:, 1:2] * Y[None] + g[:, 2:3]
    R = np.maximum(G, 0)
    M = np.concatenate([np.ones((36, 1))] + [np.stack([R[k], R[k] * X, R[k] * Y], 1) for k in range(4)], 1)
    sol = np.linalg.lstsq(M, TGT, rcond=None)[0]
    return M @ sol - TGT
def run(args):
    lab, ids, seed = args
    rng = np.random.default_rng(seed)
    best = (1e9, None)
    for st in range(6):
        th = []
        for l in ids:
            R = REPS[l]
            if st == 0: g = R[0]
            else:
                w = rng.dirichlet(np.ones(len(R))) if len(R) > 1 else np.ones(1)
                g = (w[:, None] * (R / np.linalg.norm(R, axis=1, keepdims=True))).sum(0)
                g = g + 0.05 * rng.normal(size=3) * np.linalg.norm(g)
            th.append(g / np.linalg.norm(g))
        th = np.concatenate(th)
        try:
            res = least_squares(resid_vec, th, method='trf', max_nfev=400, xtol=1e-12, ftol=1e-14, gtol=1e-14, diff_step=1e-6)
            v = np.abs(resid_vec(res.x)).max()
            if v < best[0]: best = (v, res.x.tolist())
        except Exception as e:
            pass
        if best[0] < 1e-6: break
    return lab, ids, best[0], best[1]
if __name__ == "__main__":
    src = sys.argv[1]; out = sys.argv[2]; nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else None
    d = json.load(open(src))
    surv = d["survivors"][:lim] if lim else d["survivors"]
    t0 = time.time()
    jobs = [(lab, ids, i) for i, (lab, ids) in enumerate(surv)]
    res = []
    with Pool(nproc) as p:
        for r in p.imap_unordered(run, jobs, chunksize=4):
            res.append(r)
            if r[2] < 1e-5: print("NEAR-ZERO", r[0], r[1], r[2], r[3], flush=True)
            if len(res) % 2000 == 0: print(len(res), "done, min so far", min(x[2] for x in res), "t", round(time.time() - t0), flush=True)
    vals = np.array([r[2] for r in res])
    print("N", len(res), "min", vals.min(), "quantiles", np.quantile(vals, [0, 0.01, 0.1, 0.5]).tolist(), "t", round(time.time() - t0), flush=True)
    json.dump({"n": len(res), "min": float(vals.min()), "results": sorted([[r[0], r[1], r[2]] for r in res], key=lambda z: z[2])[:500]}, open(out, "w"))

# look for well-conditioned exact 4-unit 2-D forms: minimise lattice gradients subject to exactness
import numpy as np, json, sys
from scipy.optimize import least_squares
from multiprocessing import Pool
P = np.array([(x, y) for x in range(6) for y in range(6)], float)
X, Y = P[:, 0], P[:, 1]
TGT = (X + Y) % 2
def build(theta):
    g = theta.reshape(4, 3)
    G = g[:, 0:1] * X + g[:, 1:2] * Y + g[:, 2:3]
    R = np.maximum(G, 0)
    M = np.concatenate([np.ones((36, 1))] + [np.stack([R[k], R[k] * X, R[k] * Y], 1) for k in range(4)], 1)
    return g, G, R, M
def evaluate(theta, lam):
    g, G, R, M = build(theta)
    # regularised solve: exactness weight 1, small ridge on values to keep them small
    A = np.vstack([M, lam * np.eye(13)]); b = np.r_[TGT, np.zeros(13)]
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    r = M @ sol - TGT
    # gradient at lattice points: sum over active units of grad(g v)
    V = sol[1:].reshape(4, 3)  # per unit: coef of R, R*X, R*Y
    gx = np.zeros(36); gy = np.zeros(36)
    for k in range(4):
        act = G[k] > 0
        v = V[k, 0] + V[k, 1] * X + V[k, 2] * Y
        gx += act * (g[k, 0] * v + G[k] * V[k, 1]); gy += act * (g[k, 1] * v + G[k] * V[k, 2])
    dist = np.abs(G / np.linalg.norm(g[:, :2], axis=1, keepdims=True))  # euclidean distance of lattice pts to lines
    return r, gx, gy, sol, dist
def obj(theta, lam, mu):
    r, gx, gy, sol, dist = evaluate(theta, lam)
    # keep lattice points away from the lines (soft): penalise dist < 0.25 unless ~0
    pen = np.where(dist < 0.25, (0.25 - dist) * (dist > 1e-3), 0).ravel()
    return np.r_[1e3 * r, mu * gx, mu * gy, 10 * pen]
def run(seed):
    rng = np.random.default_rng(seed)
    best = None
    for st in range(8):
        th = rng.normal(size=12)
        th[2::3] = rng.uniform(-6, 6, 4) * np.linalg.norm(th.reshape(4, 3)[:, :2], axis=1)
        try:
            for mu in (0.0, 0.03, 0.01, 0.003):
                res = least_squares(obj, th, args=(1e-6, mu), max_nfev=1500, xtol=1e-13, ftol=1e-13)
                th = res.x
            r, gx, gy, sol, dist = evaluate(th, 0.0)
            # final polish: exactness only
            res = least_squares(lambda z: evaluate(z, 0.0)[0], th, max_nfev=500, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            th = res.x
            r, gx, gy, sol, dist = evaluate(th, 0.0)
            if np.abs(r).max() < 1e-9:
                slope = float(np.sqrt(gx ** 2 + gy ** 2).max())
                dmin = float(np.where(dist > 1e-6, dist, 9).min())
                rec = (slope, dmin, th.tolist(), sol.tolist())
                if best is None or slope < best[0]: best = rec
        except Exception:
            pass
    return best
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    with Pool(24) as p:
        out = [x for x in p.map(run, range(N)) if x]
    out.sort(key=lambda z: z[0])
    for sl, dm, th, sol in out[:10]:
        print("maxgrad %.2f  min line-dist(nonzero) %.3f  gates %s" % (sl, dm, np.round(np.array(th).reshape(4, 3), 3).tolist()))
    json.dump(out, open("cond2d.json", "w"))

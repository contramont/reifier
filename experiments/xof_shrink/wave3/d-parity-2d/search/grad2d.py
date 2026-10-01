# from the 1-D bump solution, search nearby exact 2-D forms with smaller lattice gradients
import numpy as np, math, json
from scipy.optimize import least_squares
P = np.array([(x, y) for x in range(6) for y in range(6)], float); X, Y = P[:, 0], P[:, 1]
T = (X + Y) % 2
r = 2 * math.sqrt(2)
# bump gates (a, b, c): g = a x + b y + c
G0 = np.array([[-1, -1, 1 + r], [1, 1, -(5 - r)], [1, 1, -(9 - r)], [1, 1, -(5 + r)]], float)
def parts(theta):
    g = theta.reshape(4, 3)
    Gv = g[:, :1] * X + g[:, 1:2] * Y + g[:, 2:]
    R = np.maximum(Gv, 0)
    M = np.concatenate([np.ones((36, 1))] + [np.stack([R[k], R[k] * X, R[k] * Y], 1) for k in range(4)], 1)
    # gradient operator: dF/dx, dF/dy at lattice points as linear maps of w
    Gx = np.zeros((36, 13)); Gy = np.zeros((36, 13))
    for k in range(4):
        act = (Gv[k] > 0).astype(float)
        # unit = R*(w0 + w1 x + w2 y); d/dx = a*act*(w0 + w1 x + w2 y) + R*w1
        c0 = 1 + 3 * k
        Gx[:, c0] = g[k, 0] * act; Gx[:, c0 + 1] = g[k, 0] * act * X + R[k]; Gx[:, c0 + 2] = g[k, 0] * act * Y
        Gy[:, c0] = g[k, 1] * act; Gy[:, c0 + 1] = g[k, 1] * act * X; Gy[:, c0 + 2] = g[k, 1] * act * Y + R[k]
    return M, Gx, Gy
def solve(theta, lam):
    M, Gx, Gy = parts(theta)
    A = np.vstack([M, lam * Gx, lam * Gy]); b = np.r_[T, np.zeros(72)]
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return M @ w - T, Gx @ w, Gy @ w, w
def vec(theta, lam, mu):
    res, gx, gy, w = solve(theta, lam)
    return np.r_[1e3 * res, mu * np.abs(gx) + 0 * gx, mu * np.abs(gy)]
th0 = G0.ravel()
res, gx, gy, w = solve(th0, 0.0)
print("bump: resid %.2e  max|grad|_1 %.3f" % (np.abs(res).max(), (np.abs(gx) + np.abs(gy)).max()))
best = None
rng = np.random.default_rng(0)
for trial in range(40):
    th = th0 + (0 if trial == 0 else 0.05) * rng.normal(size=12)
    for lam, mu in ((1e-3, 0.3), (1e-3, 0.1), (1e-4, 0.03), (0.0, 0.0)):
        sol = least_squares(vec, th, args=(lam, mu), max_nfev=800, xtol=1e-14, ftol=1e-14)
        th = sol.x
    res, gx, gy, w = solve(th, 0.0)
    # min-gradient values among exact solutions: add small ridge on gradient after exactness
    e = np.abs(res).max(); gm = (np.abs(gx) + np.abs(gy)).max()
    if e < 1e-9 and (best is None or gm < best[0]):
        best = (gm, th.tolist(), w.tolist())
        print("trial", trial, "exact, max|grad|_1 %.3f" % gm, np.round(th.reshape(4, 3), 3).tolist(), flush=True)
json.dump(best, open("grad2d_best.json", "w"))

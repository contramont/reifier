import numpy as np, sys
from scipy.optimize import least_squares, minimize_scalar
n = 7; ts = np.arange(n + 1.); par = ts % 2; alt = (-1.0) ** ts
s = np.array([1., -1., -1.]); tau = np.array([1.5, 3., 5.4])
P = np.array([8/3, -27/4, 5/3]); Q = np.array([-89/3, 19/2, -50/3])
v = P[:, None] * ts + Q[:, None]; r0 = np.maximum(0, s[:, None] * (ts - tau[:, None])); z = r0 * v
def res(t1):
    r1 = np.maximum(0, s[:, None] * (ts - np.asarray(t1)[:, None]))
    MD = np.concatenate([(r1 * v - z).T, r1.T], 1)
    e, *_ = np.linalg.lstsq(MD, alt, rcond=None)
    return MD @ e - alt, e, np.linalg.matrix_rank(MD, 1e-9)
base = [float(x) for x in sys.argv[1].split(",")]
r, e, rk = res(base); print("base residual", np.abs(r).max(), "rank", rk, "lam,d", np.round(e, 5))
# Jacobian rank of residual wrt knots at the solution (numerical)
J = []
for i in range(3):
    h = 1e-6; b2 = list(base); b2[i] += h
    J.append((res(b2)[0] - res(base)[0]) / h)
J = np.array(J).T
print("dres/dknots singular values", np.linalg.svd(J, compute_uv=False))

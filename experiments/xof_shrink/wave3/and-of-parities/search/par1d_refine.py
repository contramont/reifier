import numpy as np
from scipy.optimize import minimize
d = np.arange(-3, 4).astype(float)
T = (np.abs(d) % 2)
def res(k):
    k1, k2 = k
    r1 = np.maximum(0, d - k1); r2 = np.maximum(0, -d - k2)
    M = np.stack([np.ones_like(d), r1, r1 * d, r2, r2 * d], 1)
    x, *_ = np.linalg.lstsq(M, T, rcond=None)
    return np.sum((M @ x - T) ** 2), np.abs(M @ x - T).max(), x
best = None
for k1 in np.linspace(-1.0, -0.3, 15):
    for k2 in np.linspace(-1.4, -0.7, 15):
        o = minimize(lambda k: res(k)[0], [k1, k2], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-30, maxiter=4000))
        r = res(o.x)
        if best is None or r[1] < best[1]:
            best = (o.x, r[1], r[2])
print(best)
# scan 1-D profile of min residual
for k1 in np.linspace(-1.2, 0.2, 29):
    o = minimize(lambda k2: res([k1, k2[0]])[0], [-1.0], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-30))
    print(round(k1, 3), o.x[0].round(4), res([k1, o.x[0]])[1])

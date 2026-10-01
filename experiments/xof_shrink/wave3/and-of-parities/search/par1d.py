# 1-D check: parity on d in [-3,3] with 2 units relu(d - k1)(u1 + v1 d) + relu(-d - k2)(u2 + v2 d) + c
import numpy as np
d = np.arange(-3, 4).astype(float)
T = (np.abs(d) % 2)
best = (9, None)
for k1 in np.linspace(-3.5, 3.5, 1401):
    for k2 in np.linspace(-3.5, 3.5, 1401):
        r1 = np.maximum(0, d - k1); r2 = np.maximum(0, -d - k2)
        M = np.stack([np.ones_like(d), r1, r1 * d, r2, r2 * d], 1)
        x, *_ = np.linalg.lstsq(M, T, rcond=None)
        e = np.abs(M @ x - T).max()
        if e < best[0]:
            best = (e, (k1, k2, x))
print(best)
# same-direction pair
best2 = (9, None)
for k1 in np.linspace(-3.5, 3.5, 1401):
    for k2 in np.linspace(-3.5, 3.5, 1401):
        if k2 <= k1: continue
        r1 = np.maximum(0, d - k1); r2 = np.maximum(0, d - k2)
        M = np.stack([np.ones_like(d), r1, r1 * d, r2, r2 * d], 1)
        x, *_ = np.linalg.lstsq(M, T, rcond=None)
        e = np.abs(M @ x - T).max()
        if e < best2[0]:
            best2 = (e, (k1, k2, x))
print(best2)

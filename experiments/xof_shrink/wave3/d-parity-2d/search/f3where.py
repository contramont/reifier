import numpy as np
from scipy.optimize import minimize
xs = np.arange(6.0)
def res(t1, t2, o1, o2, start, G=0):
    t = (xs + start) % 2
    r1 = np.maximum(0, o1 * (xs - t1)); r2 = np.maximum(0, o2 * (xs - t2))
    cols = [np.ones(6)] + ([xs] if G else []) + [r1, r1 * xs, r2, r2 * xs]
    M = np.array(cols).T
    s = np.linalg.lstsq(M, t, rcond=None)[0]
    return np.linalg.norm(M @ s - t), s
step = 0.01
g = np.arange(step, 5, step)
out = []
for o1 in (1, -1):
    for o2 in (1, -1):
        for a in g:
            for b in g:
                if b <= a: continue
                r, _ = res(a, b, o1, o2, 0)
                if r < 0.05: out.append((r, a, b, o1, o2))
out.sort()
print(out[:15])

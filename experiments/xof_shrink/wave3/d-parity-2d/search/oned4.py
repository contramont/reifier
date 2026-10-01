# 1-D: parity(s), s in 0..10, as c0 + sum_{k=1..4} relu(o_k (s - tau_k)) (d_k s + f_k), real knots.
import numpy as np, itertools
from scipy.optimize import least_squares
s = np.arange(11.0); t = s % 2
def M_of(tau, o):
    cols = [np.ones(11)]
    for tk, ok in zip(tau, o):
        r = np.maximum(0, ok * (s - tk)); cols += [r, r * s]
    return np.array(cols).T
def res(tau, o):
    M = M_of(tau, o); sol = np.linalg.lstsq(M, t, rcond=None)[0]; return M @ sol - t, sol
rng = np.random.default_rng(0)
sols = []
for o in itertools.product((1, -1), repeat=4):
    for trial in range(300):
        tau0 = np.sort(rng.uniform(0.2, 9.8, 4))
        r = least_squares(lambda z: res(z, o)[0], tau0, xtol=1e-14, ftol=1e-15, gtol=1e-15, max_nfev=2000)
        v = np.abs(res(r.x, o)[0]).max()
        if v < 1e-10:
            tau = r.x
            dist = np.min(np.abs(tau[:, None] - np.arange(11)[None]), 1)
            sols.append((o, tuple(np.round(tau, 4)), float(dist.min())))
# summarize: best min-distance-to-integer per orientation
by = {}
for o, tau, d in sols:
    if o not in by or d > by[o][1]: by[o] = (tau, d)
for o, (tau, d) in sorted(by.items(), key=lambda z: -z[1][1]):
    print(o, tau, "min dist to lattice", round(d, 4), "n", sum(1 for x in sols if x[0] == o))

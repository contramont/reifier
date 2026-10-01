# F3 check: a 6-point alternating row with 2 knots and only a constant (G=0) or a linear (G=1) global part.
# min over knot positions (fine grid incl. lattice points) and orientations of the residual distance.
import numpy as np
xs = np.arange(6.0)
def batch_res(Ms, t):
    u, s, _ = np.linalg.svd(Ms, full_matrices=False)
    keep = (s > 1e-10 * np.maximum(s[:, :1], 1))[:, None, :]
    u = u * keep
    p = np.einsum('nmk,nk->nm', u, np.einsum('nmk,m->nk', u, t))
    return np.linalg.norm(t[None] - p, axis=1)
step = 0.004
g = np.unique(np.r_[np.arange(step, 5, step), np.arange(1, 5)])
T1, T2 = np.meshgrid(g, g, indexing='ij'); m = T2 > T1
T1, T2 = T1[m], T2[m]
for G in (0, 1):
    for start in (0, 1):
        t = (xs + start) % 2
        best = 1e9
        for o1 in (1, -1):
            for o2 in (1, -1):
                for s in range(0, len(T1), 200000):
                    a, b = T1[s:s + 200000], T2[s:s + 200000]
                    r1 = np.maximum(0, o1 * (xs[None] - a[:, None])); r2 = np.maximum(0, o2 * (xs[None] - b[:, None]))
                    cols = [np.ones_like(r1)] + ([np.broadcast_to(xs, r1.shape)] if G == 1 else []) + [r1, r1 * xs, r2, r2 * xs]
                    Ms = np.stack(cols, 2)
                    best = min(best, batch_res(Ms, t).min())
        print("G", G, "start", start, "min residual over", len(T1), "position pairs x 4 orientations:", best)

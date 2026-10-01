"""Can a column's 5 bits travel as 2 features (C = sum, F2 = w.x) and be decoded by 5
gated units (each unit additive in the bits)? Union of unit directions in A = span(bits,1),
rank mod 1 must be 5. Gates: integer coefs on (C, F2), bias on a half grid."""
import itertools, sys
import numpy as np
pts = np.array(list(itertools.product([0, 1], repeat=5)), dtype=float)
A = np.concatenate([pts, np.ones((32, 1))], 1)
one = np.ones((32, 1))

def dirs_for(F, G=4, bstep=0.5):
    m = F.shape[1]
    out = []
    lo, hi = -40, 40
    for coefs in itertools.product(range(-G, G + 1), repeat=m):
        if not any(coefs):
            continue
        gl = F @ np.array(coefs, float)
        vals = np.unique(gl)
        # biases: only those that change the active set: between consecutive values and at values
        cand = set()
        for v in vals:
            cand.add(-v)
        for v1, v2 in zip(vals[:-1], vals[1:]):
            cand.add(-(v1 + v2) / 2)
        for b in cand:
            r = np.maximum(0, gl + b)
            if not r.any():
                continue
            W = np.concatenate([r[:, None], r[:, None] * F], 1)
            M = np.concatenate([W, -A], 1)
            _, s, vt = np.linalg.svd(M)
            null = vt[np.sum(s > 1e-9):]
            for nv in null:
                u = W @ nv[: W.shape[1]]
                if np.abs(u).max() > 1e-9:
                    out.append(u)
    return out

best = []
seen = set()
C = pts.sum(1)
for w in itertools.product(range(0, 8), repeat=5):
    if w[0] != 0 or list(w) != sorted(w):  # symmetric in the bits: w sorted, w0 = 0 wlog (C absorbs a shift)
        continue
    F2 = pts @ np.array(w, float)
    keys = set(zip(C, F2))
    if len(keys) < 32:
        continue
    F = np.stack([C, F2], 1)
    D = dirs_for(F)
    if not D:
        continue
    rk = np.linalg.matrix_rank(np.concatenate([np.array(D).T, one], 1), 1e-9) - 1
    print(w, "rank mod 1 =", rk, flush=True)
    if rk >= 5:
        best.append(w)
print("found:", best)

"""Packed pair p = 2t - u in {-1,0,1,2} (all exact) plus e = C1 + C2 in {0,1,2}: find the
fewest E-exact units max(0,G)V on (p, e) whose span holds theta_t = t ^ [e==1] and
theta_u = u ^ [e==1] (outs weights free). Candidate units: G = al p + be e + ga on a
grid, V = a2 p + b2 e + g2 on a half grid, E-rule at all 12 points."""
import sys
from fractions import Fraction as F
from itertools import combinations, product
import numpy as np

pts = [(p, e) for p in (-1, 0, 1, 2) for e in (0, 1, 2)]
t_of = {-1: 0, 0: 0, 1: 1, 2: 1}
u_of = {-1: 1, 0: 0, 1: 1, 2: 0}
T1 = np.array([t_of[p] ^ int(e == 1) for p, e in pts], float)
T2 = np.array([u_of[p] ^ int(e == 1) for p, e in pts], float)
P = np.array([p for p, e in pts], float); Ee = np.array([e for p, e in pts], float)
pow2 = set([2.0 ** k for k in range(-4, 5)])
def is_p2(x):
    return np.isin(np.abs(x), list(pow2))
cg = [x / 2 for x in range(-6, 7)]
vg = np.arange(-8, 9) / 2.0
A2, B2, G2 = np.meshgrid(vg, vg, vg, indexing="ij")
A2, B2, G2 = A2.ravel(), B2.ravel(), G2.ravel()
Vall = A2[:, None] * P[None] + B2[:, None] * Ee[None] + G2[:, None]  # (nV, 12)
units = {}
for al, be in product(cg, cg):
    for ga in np.arange(-16, 17) / 4.0:
        G = al * P + be * Ee + ga
        if np.any((G > -0.5) & (G < 0)) or np.all(G <= 0):
            continue
        on = G > 0
        gok = is_p2(G)
        # E-rule per point: off, or V==0, or (G pow2 and |V| pow2)
        ok = np.all(~on[None] | (Vall == 0) | (gok[None] & is_p2(Vall)), 1)
        out = np.maximum(G, 0)[None] * Vall[ok]
        for i, o in zip(np.nonzero(ok)[0], out):
            key = tuple(np.round(o, 6))
            if any(key) and key not in units:
                units[key] = (al, be, ga, A2[i], B2[i], G2[i])
keys = list(units)
M = np.array(keys)
print(len(keys), "distinct E-exact unit outputs")
def in_span(cols, target):
    X = M[list(cols)].T
    c, res, rk, _ = np.linalg.lstsq(X, target, rcond=None)
    return np.allclose(X @ c, target, atol=1e-9), c
# normalize: dedupe by direction
dirs = {}
for i, k in enumerate(keys):
    v = M[i] / np.abs(M[i]).max()
    s = tuple(np.round(v * np.sign(v[np.nonzero(v)[0][0]]), 6))
    dirs.setdefault(s, i)
idx = list(dirs.values())
print(len(idx), "distinct directions")
for K in (2, 3, 4):
    best = None
    for cols in combinations(idx, K):
        ok1, c1 = in_span(cols, T1)
        if not ok1:
            continue
        ok2, c2 = in_span(cols, T2)
        if ok2:
            best = (cols, c1, c2)
            break
    print("K =", K, "found" if best else "none")
    if best:
        for j in best[0]:
            print("  unit", units[keys[j]], "->", keys[j])
        print("  outs t:", np.round(best[1], 4), " outs u:", np.round(best[2], 4))
        break

"""theta = a ^ [e == 1], e in {0,1,2}: one per-bit E-exact unit h(a,e) + shared g(e).
Vectorized over V = p + q a + r e on a quarter grid; G = al a + be e + ga on a half grid."""
from fractions import Fraction as F
from itertools import product
import numpy as np
from eunits import e_ok
from search_theta import g_units

E = (0, 1, 2)
pts = [(a, e) for a in (0, 1) for e in E]
theta = np.array([a ^ int(e == 1) for a, e in pts], dtype=float)
vg = np.arange(-24, 25) / 4.0
P, Q, Rr = np.meshgrid(vg, vg, vg, indexing="ij")
P, Q, Rr = P.ravel(), Q.ravel(), Rr.ravel()
V = np.stack([P + Q * a + Rr * e for a, e in pts], 1)  # (nV, 6)
found = {}
grid = np.arange(-8, 9) / 2.0
for al, be, ga in product(grid, grid, np.arange(-24, 25) / 4.0):
    G = np.array([al * a + be * e + ga for a, e in pts])
    if np.any((G > -0.5) & (G < 0)):
        continue
    h = np.maximum(G, 0) * V
    d = (h[:, 3:] - h[:, :3]) - (theta[3:] - theta[:3])
    ok = np.all(np.abs(d) < 1e-9, 1)
    for i in np.nonzero(ok)[0]:
        Gf = [F(x).limit_denominator(8) for x in G]
        Vf = [F(x).limit_denominator(8) for x in V[i]]
        if not all(e_ok(g, v) for g, v in zip(Gf, Vf)):
            continue
        hf = [max(F(0), g) * v for g, v in zip(Gf, Vf)]
        gt = {e: F(int(e == 1)) - hf[e] for e in E}
        key = tuple(gt.values())
        if key in found:
            continue
        gu = g_units(gt)
        if gu is None:
            continue
        found[key] = (len(gu), (al, be, ga), (P[i], Q[i], Rr[i]), gu)
res = sorted(found.values(), key=lambda f: f[0])
print(len(res), "distinct shared parts; unit counts:", sorted({f[0] for f in res}))
for f in res[:12]:
    print(f)

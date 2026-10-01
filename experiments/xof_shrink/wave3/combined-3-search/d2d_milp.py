"""Can D = parity(C_L + C_R), C_L, C_R in [0, 5] (the X layer's column-pair parity, today 5 units on
the 1-D sum), be written with K < 5 gated units whose gates read the two column sums SEPARATELY?
units: max(0, a x + b y + c) (d x + e y + f), integer gates |a|, |b| <= A, plus a free constant.
MILP (HiGHS): binary unit selection over all distinct gates, exact equality on the 36 points."""
import sys, itertools, math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

A = int(sys.argv[1]) if len(sys.argv) > 1 else 2
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TL = float(sys.argv[3]) if len(sys.argv) > 3 else 600
pts = [(x, y) for x in range(6) for y in range(6)]
tgt = np.array([(x + y) % 2 for x, y in pts], float)
gates, seen = [], set()
for a in range(-A, A + 1):
    for b in range(-A, A + 1):
        if a == 0 and b == 0:
            continue
        for c in range(-5 * 2 * A - 1, 5 * 2 * A + 2):
            if math.gcd(math.gcd(abs(a), abs(b)), abs(c)) != 1:
                continue
            r = np.array([max(0, a * x + b * y + c) for x, y in pts], float)
            if not r.any():
                continue
            key = tuple(np.round(r / r.max(), 9))
            if key in seen:
                continue
            seen.add(key)
            gates.append(((a, b, c), r))
n = len(gates)
print("gates", n, "K", K, flush=True)
M = float(sys.argv[4]) if len(sys.argv) > 4 else 64.0
# variables: z_j (n), d_j, e_j, f_j (3n), const (1)
nv = 4 * n + 1
rows, lo, hi = [], [], []
for i, (x, y) in enumerate(pts):
    row = np.zeros(nv)
    for j, (_, r) in enumerate(gates):
        row[n + 3 * j] = r[i] * x
        row[n + 3 * j + 1] = r[i] * y
        row[n + 3 * j + 2] = r[i]
    row[-1] = 1
    rows.append(row); lo.append(tgt[i]); hi.append(tgt[i])
for j in range(n):  # |coef| <= M z_j
    for k in range(3):
        row = np.zeros(nv); row[n + 3 * j + k] = 1; row[j] = -M; rows.append(row); lo.append(-np.inf); hi.append(0)
        row = np.zeros(nv); row[n + 3 * j + k] = -1; row[j] = -M; rows.append(row); lo.append(-np.inf); hi.append(0)
row = np.zeros(nv); row[:n] = 1; rows.append(row); lo.append(0); hi.append(K)
cons = LinearConstraint(np.array(rows), lo, hi)
integrality = np.r_[np.ones(n), np.zeros(3 * n + 1)]
bounds = Bounds(np.r_[np.zeros(n), -M * np.ones(3 * n), -1e3], np.r_[np.ones(n), M * np.ones(3 * n), 1e3])
c = np.r_[np.ones(n), np.zeros(3 * n + 1)]
res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": TL, "disp": False})
print("status", res.status, res.message, flush=True)
if res.x is not None:
    z = res.x[:n]
    for j in np.nonzero(z > 0.5)[0]:
        print(gates[j][0], np.round(res.x[n + 3 * j:n + 3 * j + 3], 4))
    print("const", res.x[-1])

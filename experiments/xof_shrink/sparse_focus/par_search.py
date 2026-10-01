"""parity of an integer P in [0, N] (one feature) with 5 gated units max(0, +-(P - k)) * (c + d P),
integer knots; minimize total nonzeros (gate: 1 if the bias is 0 else 2; value: [c!=0] + [d!=0]).
glu_xor uses 15 for N = 10. Exhaustive over gate sets and value sparsity patterns."""
import itertools, sys
import numpy as np
from fractions import Fraction as Fr
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
P = np.arange(N + 1, dtype=float)
target = (np.arange(N + 1) % 2).astype(float)
gates = [("R", k) for k in range(0, N)] + [("L", k) for k in range(1, N + 1)]
def act(g):
    d, k = g
    return np.maximum(0, P - k) if d == "R" else np.maximum(0, k - P)
def gnnz(g):
    return 1 if g == ("R", 0) else 2
best = None
patterns = list(itertools.product(((1, 0), (0, 1), (1, 1)), repeat=K))
for gs in itertools.combinations(gates, K):
    A = [act(g) for g in gs]
    gn = sum(gnnz(g) for g in gs)
    if best is not None and gn + K > best[0]:
        continue
    for pat in patterns:
        vn = sum(a + b for a, b in pat)
        tot = gn + vn
        if best is not None and tot >= best[0]:
            continue
        cols = []
        for a, (pc, pd) in zip(A, pat):
            if pc: cols.append(a)
            if pd: cols.append(a * P)
        M = np.stack(cols, 1)
        x, res, rk, _ = np.linalg.lstsq(M, target, rcond=None)
        if np.abs(M @ x - target).max() < 1e-9:
            best = (tot, gs, pat, x)
            print("found", tot, gs, pat, np.round(x, 4), flush=True)
print("best", best)

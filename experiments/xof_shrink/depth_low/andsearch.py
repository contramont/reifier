"""Is the exact AND (1 - p(b)) p(c) of two parities (b = count of |B| bits, c = count of |C|
bits) cheaper than the pair parity p(b + c)? Search K units max(0, a b + e c + g)(u + v b + w c)
on the grid [0,nB] x [0,nC], with the singles p(b), p(c) and a constant free. Exactness on
the grid = exactness on the cube (the function and all units depend on the counts only)."""
import itertools, sys, numpy as np
nB, nC, K, W, G = (int(v) for v in sys.argv[1:6])
bs, cs = np.meshgrid(np.arange(nB + 1), np.arange(nC + 1), indexing="ij")
b, c = bs.ravel().astype(float), cs.ravel().astype(float)
T = (1 - b % 2) * (c % 2)
free = np.stack([np.ones_like(b), b % 2, c % 2], 1)
gates = {}
for a in range(-W, W + 1):
    for e in range(-W, W + 1):
        for g in range(-G, G + 1):
            r = np.maximum(0, a * b + e * c + g)
            if not r.any():
                continue
            key = tuple(r.round(6))
            gates.setdefault(key, (a, e, g))
cols = [np.stack([r, r * b, r * c], 1) for r in (np.array(k) for k in gates)]
names = list(gates.values())
print("grid", len(b), "distinct gates", len(cols), flush=True)
found = 0
for combo in itertools.combinations(range(len(cols)), K):
    M = np.concatenate([free] + [cols[i] for i in combo], 1)
    x, res, rk, _ = np.linalg.lstsq(M, T, rcond=None)
    if np.abs(M @ x - T).max() < 1e-9:
        found += 1
        if found <= 3:
            print("FOUND", [names[i] for i in combo], np.round(x, 4).tolist(), flush=True)
print("K", K, "solutions", found)

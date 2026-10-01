"""batched version of andsearch.py (torch lstsq on chunks of K-subsets of gates)"""
import itertools, sys, numpy as np, torch as t
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
            if r.any():
                gates.setdefault(tuple(r.round(6)), (a, e, g))
R = np.array(list(gates.keys()))
names = list(gates.values())
C = np.stack([R, R * b, R * c], 2)  # gates x points x 3
print("grid", len(b), "distinct gates", len(R), flush=True)
Ct = t.tensor(C, dtype=t.float64); Ft = t.tensor(free, dtype=t.float64); Tt = t.tensor(T, dtype=t.float64)
found = 0; total = 0
combos = itertools.combinations(range(len(R)), K)
while True:
    chunk = list(itertools.islice(combos, 200000))
    if not chunk:
        break
    idx = t.tensor(chunk)
    M = t.cat([Ft.expand(len(chunk), -1, -1)] + [Ct[idx[:, j]] for j in range(K)], 2)
    x = t.linalg.lstsq(M, Tt.expand(len(chunk), -1).unsqueeze(2), driver="gelsd").solution
    err = (M @ x).squeeze(2) - Tt
    ok = err.abs().max(1).values < 1e-8
    total += len(chunk)
    if ok.any():
        for i in ok.nonzero().flatten().tolist()[:3]:
            print("FOUND", [names[j] for j in chunk[i]], flush=True)
        found += int(ok.sum())
print("K", K, "checked", total, "solutions", found)

"""Can ONE feature p = al*a0 + be*a1 + ga*D (a0, a1, D bits; D shared by the pair) be decoded
into t0 = a0^D, t1 = a1^D by K gated units relu(p - k)(A + B p) (plus a constant)?
Exhaustive over small integer encodings and knots on a quarter grid; rank-1 condition per unit."""
import itertools, sys
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
pts = list(itertools.product((0, 1), repeat=3))
t0 = np.array([a0 ^ D for a0, a1, D in pts], float)
t1 = np.array([a1 ^ D for a0, a1, D in pts], float)
found = 0
exact_cnt = 0
R = range(-4, 5)
for al, be, ga in itertools.product(R, R, R):
    if al == 0 or be == 0 or ga == 0:
        continue
    p = np.array([al * a0 + be * a1 + ga * D for a0, a1, D in pts], float)
    # must be a function of p: equal p -> equal (t0, t1)
    ok = True
    for i in range(8):
        for j in range(8):
            if p[i] == p[j] and (t0[i] != t0[j] or t1[i] != t1[j]):
                ok = False
    if not ok:
        continue
    lo, hi = p.min(), p.max()
    grid = np.arange(lo - 0.5, hi + 0.01, 0.25)
    for ks in itertools.combinations(grid, K):
        cols = [np.ones(8)]
        for k in ks:
            r = np.maximum(0, p - k)
            cols += [r, p * r]
        M = np.stack(cols, 1)
        sols = []
        good = True
        for t in (t0, t1):
            y, *_ = np.linalg.lstsq(M, t, rcond=None)
            if np.abs(M @ y - t).max() > 1e-9:
                good = False
                break
            sols.append(y)
        if not good:
            continue
        # null space: solutions not unique -> report anyway (rank condition checked loosely)
        U, s, Vt = np.linalg.svd(M)
        null = Vt[(s > 1e-9).sum():]
        rank1 = True
        for kk in range(K):
            a = np.array([[sols[0][1 + 2 * kk], sols[0][2 + 2 * kk]], [sols[1][1 + 2 * kk], sols[1][2 + 2 * kk]]])
            if abs(np.linalg.det(a)) > 1e-9:
                rank1 = False
        exact_cnt += 1
        if rank1 or len(null):
            found += 1
            if found <= 20:
                print("enc", (al, be, ga), "knots", ks, "rank1", rank1, "null", len(null),
                      "sol0", np.round(sols[0], 4), "sol1", np.round(sols[1], 4))
print("found", found, "exact (no rank cond)", exact_cnt)

# numeric check of N3's lemma: for F = c0 + sum relu(g_k) v_k, two edge-adjacent unit cells that are
# not strictly cut by any line, and whose shared edge lies on no line, have equal mixed differences.
import numpy as np
rng = np.random.default_rng(1)
viol = checked = 0
for trial in range(20000):
    K = rng.integers(1, 6)
    # random lines: mix of integer lines through lattice points and random real lines
    gs = []
    for k in range(K):
        if rng.random() < 0.5:
            a, b = rng.integers(-3, 4, 2)
            if a == 0 and b == 0: a = 1
            c = rng.integers(-15, 16) / rng.choice([1, 2, 3])
        else:
            a, b, c = rng.normal(size=3) * [1, 1, 4]
        gs.append((a, b, c))
    vs = [rng.normal(size=3) for _ in range(K)]
    def F(x, y):
        return sum(max(0.0, a * x + b * y + c) * (d * x + e * y + f) for (a, b, c), (d, e, f) in zip(gs, vs))
    def g(k, x, y):
        a, b, c = gs[k]; return a * x + b * y + c
    def cut(i, j):
        for k in range(K):
            vals = [g(k, i + di, j + dj) for di in (0, 1) for dj in (0, 1)]
            if max(vals) > 1e-9 and min(vals) < -1e-9: return True
        return False
    def along(pa, pb):
        return any(abs(g(k, *pa)) < 1e-9 and abs(g(k, *pb)) < 1e-9 for k in range(K))
    def mixed(i, j):
        return F(i + 1, j + 1) - F(i + 1, j) - F(i, j + 1) + F(i, j)
    for i in range(5):
        for j in range(5):
            for (i2, j2, pa, pb) in ((i + 1, j, (i + 1, j), (i + 1, j + 1)), (i, j + 1, (i, j + 1), (i + 1, j + 1))):
                if i2 > 4 or j2 > 4: continue
                if cut(i, j) or cut(i2, j2) or along(pa, pb): continue
                checked += 1
                if abs(mixed(i, j) - mixed(i2, j2)) > 1e-7 * (1 + abs(mixed(i, j))):
                    viol += 1
print("checked", checked, "violations", viol)

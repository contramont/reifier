"""K=5 check: can 5 gated units (+ free constant) compute the 5 chi bits of a row from
packed features? Each unit must then lie in T + 1 (T = span of the chi bits). For each
gate (integer coefs on the features, bias on a grid), intersect W_g = {relu(g) v} with
T + 1 and collect directions; K=5 works iff they span T (mod 1)."""
import itertools, sys
import numpy as np

pts = np.array(list(itertools.product([0, 1], repeat=5)), dtype=float)  # t0..t4
def chi(i):
    t = pts
    return np.logical_xor(t[:, i] > 0, np.logical_and(t[:, (i + 1) % 5] == 0, t[:, (i + 2) % 5] > 0)).astype(float)
T = np.stack([chi(i) for i in range(5)], 1)  # 32 x 5
one = np.ones((32, 1))
TB = np.concatenate([T, one], 1)

def rank(M, tol=1e-9):
    return np.linalg.matrix_rank(M, tol) if M.size else 0

def run(feats, name, G=3, biases=None):
    F = pts @ np.array(feats, dtype=float).T  # 32 x m
    m = F.shape[1]
    if biases is None:
        biases = np.arange(-12, 12.5, 0.5)
    dirs = []
    for coefs in itertools.product(range(-G, G + 1), repeat=m):
        if not any(coefs):
            continue
        gl = F @ np.array(coefs, dtype=float)
        for b in biases:
            r = np.maximum(0, gl + b)
            if not r.any():
                continue
            W = np.concatenate([r[:, None], r[:, None] * F], 1)  # 32 x (m+1)
            # intersection of span(W) with span(TB): solve W a = TB c
            M = np.concatenate([W, -TB], 1)
            _, s, vt = np.linalg.svd(M)
            null = vt[np.sum(s > 1e-9):]
            for nv in null:
                u = W @ nv[: W.shape[1]]
                if np.abs(u).max() > 1e-9:
                    dirs.append(u)
    D = np.array(dirs)
    # rank of directions modulo the constant
    Dm = np.concatenate([D.T, one], 1) if len(dirs) else one
    rk = rank(Dm) - 1
    print(f"{name}: {len(dirs)} unit directions in T+1, rank mod 1 = {rk} (need 5)")
    sys.stdout.flush()
    return rk

e = np.eye(5, dtype=int)
tests = {}
# unpacked control: 5 features (should reach 5 with the usual chi units)
tests["unpacked"] = [list(e[i]) for i in range(5)]
for r in (2, 3, -2):
    for (p, q, s) in [((0, 1), (2, 3), 4), ((0, 2), (1, 3), 4), ((0, 1), (2, 4), 3), ((0, 2), (3, 4), 1), ((0, 3), (1, 4), 2)]:
        f1 = list(e[p[0]] + r * e[p[1]]); f2 = list(e[q[0]] + r * e[q[1]]); f3 = list(e[s])
        tests[f"pairs{p}{q}_r{r}"] = [f1, f2, f3]
        f1b = list(r * e[p[0]] + e[p[1]]); f2b = list(r * e[q[0]] + e[q[1]])
        tests[f"pairsB{p}{q}_r{r}"] = [f1b, f2b, f3]
for name, feats in tests.items():
    run(feats, name, G=3 if len(feats) == 3 else 2)

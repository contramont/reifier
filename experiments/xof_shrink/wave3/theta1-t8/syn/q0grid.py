# T=n pool-of-3 (q0) search on a rational knot grid (multiples of 1/den):
#  step 1: pools = triples of A=0 slice gates (s, tau0) with parity in span{r_j, r_j T}; exact values
#  step 2: A=1 knots tau1_j on the grid with (D) sum_j [r1_j (lam_j v_j + d_j) - lam_j z_j] = (-1)^T
import numpy as np, sys, time, itertools, pickle
n = int(sys.argv[1]); den = int(sys.argv[2]); lo = float(sys.argv[3]); hi = float(sys.argv[4])
ci, nc = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
CONST = int(sys.argv[7]) if len(sys.argv) > 7 else 1
ts = np.arange(n + 1, dtype=float); par = ts % 2; alt = (-1.0) ** ts
knots = [k / den for k in range(int(np.floor(lo * den)), int(np.ceil(hi * den)) + 1)]
gates = []
for s in (1.0, -1.0):
    for t in knots:
        r = np.maximum(0, s * (ts - t))
        if r.any():
            gates.append((s, t, r))
print("gates", len(gates), flush=True)
G = len(gates)
Rg = np.array([g[2] for g in gates])  # (G, n+1)
tol = 1e-9
def proj_out(B, v):
    # residual of v after projecting on columns of B (batched: B (b, m, k), v (b, m) or (m,))
    U, sv, _ = np.linalg.svd(B, full_matrices=False)
    msk = sv > 1e-10 * np.maximum(sv[..., :1], 1e-300)
    Q = U * msk[..., None, :]
    if v.ndim == 1:
        v = np.broadcast_to(v, B.shape[:-1])
    return v - np.einsum("bmk,bk->bm", Q, np.einsum("bmk,bm->bk", Q, v)), Q
t0 = time.time()
pools = []
idx = list(itertools.combinations(range(G), 3))[ci::nc]
B = 200000
for s0 in range(0, len(idx), B):
    I = np.array(idx[s0:s0 + B])
    M = np.concatenate([Rg[I[:, 0]][:, :, None], (Rg[I[:, 0]] * ts)[:, :, None], Rg[I[:, 1]][:, :, None], (Rg[I[:, 1]] * ts)[:, :, None],
                        Rg[I[:, 2]][:, :, None], (Rg[I[:, 2]] * ts)[:, :, None]] + ([np.ones((len(I), n + 1, 1))] if CONST else []), 2)
    res, _ = proj_out(M, par)
    ok = np.where(np.abs(res).max(1) < tol)[0]
    for i in ok:
        pools.append(tuple(I[i]))
print("pool triples (relaxed P)", len(pools), round(time.time() - t0, 1), flush=True)
if len(sys.argv) > 8 and sys.argv[8] == "poolsonly":
    pickle.dump((pools, []), open(f"q0grid_n{n}_d{den}_{ci}.pkl", "wb"))
    for p in pools:
        a, b, c = p
        M = np.stack([Rg[a], Rg[a] * ts, Rg[b], Rg[b] * ts, Rg[c], Rg[c] * ts, np.ones(n + 1)], 1)
        sol = np.linalg.lstsq(M, par, rcond=None)[0]
        print("POOL", [gates[i][:2] for i in p], "C0", np.round(sol[0:6:2], 4).tolist(), "C1", np.round(sol[1:6:2], 4).tolist(), "k", round(sol[6], 4), flush=True)
    sys.exit(0)
hits = []
for pi, (a, b, c) in enumerate(pools):
    M = np.stack([Rg[a], Rg[a] * ts, Rg[b], Rg[b] * ts, Rg[c], Rg[c] * ts] + ([np.ones(n + 1)] if CONST else []), 1)
    sol, *_ = np.linalg.lstsq(M, par, rcond=None)
    rank = np.linalg.matrix_rank(M, 1e-9)
    if rank < M.shape[1]:
        print("nonunique pool", (gates[a][:2], gates[b][:2], gates[c][:2]), rank, flush=True)
        continue  # non-unique pool values: skipped (reported)
    C0 = sol[0:6:2]; C1 = sol[1:6:2]   # z_k = r_k (C0_k + C1_k T)
    js = (a, b, c)
    v = [C1[k] * ts + C0[k] for k in range(3)]
    z = [Rg[js[k]] * v[k] for k in range(3)]
    if any(not zz.any() for zz in z):
        continue
    # A=1 options per unit: same direction, any knot on the grid
    opts = []
    for k in range(3):
        s = gates[js[k]][0]
        cols = []
        for t1 in knots:
            r1 = np.maximum(0, s * (ts - t1))
            cols.append((t1, r1 * v[k] - z[k], r1))
        opts.append(cols)
    C1 = np.array([[o[1], o[2]] for o in opts[0]])  # (K, 2, n+1)
    C2 = np.array([[o[1], o[2]] for o in opts[1]])
    C3 = np.array([[o[1], o[2]] for o in opts[2]])
    K1, K2, K3 = len(C1), len(C2), len(C3)
    C3f = C3.reshape(-1, n + 1)
    for j1 in range(K1):
        B12 = np.concatenate([np.broadcast_to(C1[j1], (K2, 2, n + 1)), C2], 1).transpose(0, 2, 1)  # (K2, n+1, 4)
        ra, Qb = proj_out(B12, alt)
        C3p = (C3f[None] - np.einsum("bmk,bjk->bjm", Qb, np.einsum("bmk,jm->bjk", Qb, C3f))).reshape(K2, K3, 2, n + 1)
        A = C3p.transpose(0, 1, 3, 2)
        AtA = np.einsum("bjmc,bjmd->bjcd", A, A); Atb = np.einsum("bjmc,bm->bjc", A, ra)
        det = AtA[..., 0, 0] * AtA[..., 1, 1] - AtA[..., 0, 1] * AtA[..., 1, 0]
        good = np.abs(det) > 1e-12
        inv = np.zeros_like(AtA)
        inv[..., 0, 0] = AtA[..., 1, 1]; inv[..., 1, 1] = AtA[..., 0, 0]; inv[..., 0, 1] = -AtA[..., 0, 1]; inv[..., 1, 0] = -AtA[..., 1, 0]
        x = np.einsum("bjcd,bjd->bjc", inv, Atb) / np.where(good, det, 1)[..., None]
        rr = np.abs(np.einsum("bjmc,bjc->bjm", A, x) - ra[:, None, :]).max(2)
        rr = np.where(good, rr, np.abs(ra).max(1)[:, None])
        for j2, j3 in np.argwhere(rr < 1e-8):
            hits.append(((gates[a][:2], gates[b][:2], gates[c][:2]), (opts[0][j1][0], opts[1][j2][0], opts[2][j3][0]), sol.tolist()))
            print("HIT pool", (gates[a][:2], gates[b][:2], gates[c][:2]), "A=1 knots", (opts[0][j1][0], opts[1][j2][0], opts[2][j3][0]), flush=True)
    if pi % 20 == 0:
        print("pool", pi, "/", len(pools), "hits", len(hits), round(time.time() - t0, 1), flush=True)
print("DONE pools", len(pools), "hits", len(hits))
pickle.dump((pools, hits), open(f"q0grid_n{n}_d{den}_{ci}.pkl", "wb"))

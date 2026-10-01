# "Pool from the zero-lane slices" (PB) search over wave-1's (D)-feasible per-bit triples.
# Live bit: sum_i u_i(A,T) + sum_j beta_j P_j(T); zero lane: sum_j gamma_j P_j(T), where the pool
# P = {u_i(0,.) : i} (the A=0 slices of the per-bit units, separate units) + q extra hinge units.
# Needs (D) U1 - U0 = (-1)^T and parity(T) in span(P) (then beta = gamma - 1 on the slices).
# Cost per column with L live bits: 3L + 3 + q   (TH1S: 3L + 5; old T=8: 4L + 4).
import numpy as np, pickle, sys, time
T = int(sys.argv[1]); pk = sys.argv[2]; q = int(sys.argv[3]); AFF = q >= 2; CONST = q == -1
if CONST: q = 0
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
stop = int(sys.argv[5]) if len(sys.argv) > 5 else None
d = pickle.load(open(pk, "rb")); hits, keys, cands = d["hits"], d["keys"], d["cands"]
n1 = T + 1
ts = np.arange(n1, dtype=float); par = ts % 2; sig = (-1.0) ** ts
R = np.array(keys, dtype=float); R0, R1 = R[:, :n1], R[:, n1:]
H = np.array(hits[start:stop], dtype=np.int64)
print("triples", len(H), flush=True)
# extra-unit interval bases: increasing knot t in (T0-1, T0]: r = s1 - t s0, rT = s2 - t s1 (s_k = T^k [T>=T0])
# decreasing knot t in [T0, T0+1): r = t s0 - s1, rT = t s1 - s2 (s_k = T^k [T<=T0]).  T0 in 0..T
ext = []
for T0 in range(0, n1):
    m = (ts >= T0).astype(float); ext.append((+1, T0, m, ts * m, ts * ts * m))
    m = (ts <= T0).astype(float); ext.append((-1, T0, m, ts * m, ts * ts * m))
tol = 1e-8
out = []
t0 = time.time()
B = 20000
for s in range(0, len(H), B):
    h = H[s:s + B]; nb = len(h)
    # (D) columns (nb, n1, 9)
    Cd = np.concatenate([np.stack([R1[h[:, k]], (R1[h[:, k]] - R0[h[:, k]]) * ts, R1[h[:, k]] - R0[h[:, k]]], 2) for k in range(3)], 2)
    U, sv, Vt = np.linalg.svd(Cd, full_matrices=True)
    rank = (sv > 1e-9 * sv[:, :1]).sum(1)
    # particular least-norm solution
    sinv = np.where(sv > 1e-9 * sv[:, :1], 1 / np.where(sv > 0, sv, 1), 0)
    p0 = np.einsum("bji,bj,bkj,k->bi", Vt[:, :9, :], sinv, U[:, :, :9], sig) if False else None
    Ut_sig = np.einsum("bkj,k->bj", U[:, :, :9], sig) if U.shape[2] >= 9 else None
    # U is (nb, n1, n1); for n1 >= 9 fine
    k9 = min(9, n1)
    Ut_sig = np.einsum("bkj,k->bj", U[:, :, :k9], sig)
    p0 = np.einsum("bji,bj->bi", Vt[:, :k9, :], sinv[:, :k9] * Ut_sig)
    res = np.abs(np.einsum("bnk,bk->bn", Cd, p0) - sig).max(1)
    # pool vectors: phi_i = R0_i (e_i T + f_i) for the particular solution, plus null-space directions
    def phis(pv):
        return np.stack([R0[h[:, k]] * (pv[:, 3 * k + 1, None] * ts + pv[:, 3 * k + 2, None]) for k in range(3)], 2)  # (nb, n1, 3)
    Phi = phis(p0)
    nulldim = 9 - rank
    maxnull = nulldim.max()
    cols = [Phi]
    for jn in range(int(maxnull)):
        idx = 8 - jn
        vec = Vt[:, idx, :] * (nulldim > jn)[:, None]
        cols.append(phis(vec))
    if AFF:
        cols.append(np.broadcast_to(np.stack([np.ones(n1), ts], 1), (nb, n1, 2)))
    if CONST:
        cols.append(np.ones((nb, n1, 1)))
    Pm = np.concatenate(cols, 2)  # (nb, n1, 3(1+maxnull)) relaxed pool span
    # orthonormal basis of the pool span
    Up, sp, _ = np.linalg.svd(Pm, full_matrices=False)
    msk = sp > 1e-9 * np.maximum(sp[:, :1], 1e-300)
    Q = Up * msk[:, None, :]
    def perp(v):  # v (nb, n1) or (n1,)
        if v.ndim == 1:
            v = np.broadcast_to(v, (nb, n1))
        return v - np.einsum("bnk,bk->bn", Q, np.einsum("bnk,bn->bk", Q, v))
    pp = perp(par)
    r_nox = np.abs(pp).max(1)
    ok0 = np.where((r_nox < tol) & (res < 1e-6))[0]
    for i in ok0:
        out.append(("q0", tuple(h[i]), int(nulldim[i])))
    if q >= 1:
        for dr, T0, s0, s1, s2 in ext:
            P0, P1, P2 = perp(s0), perp(s1), perp(s2)
            if dr > 0:   # pp = a (P1 - t P0) + b (P2 - t P1) = a P1 + b P2 - (a t) P0 - (b t) P1
                M = np.stack([P1, P2, -P0, -P1], 2)
            else:        # pp = a (t P0 - P1) + b (t P1 - P2)
                M = np.stack([-P1, -P2, P0, P1], 2)
            # least squares in the 4 unknowns (a, b, at, bt)
            Um, sm, Vm = np.linalg.svd(M, full_matrices=False)
            smask = sm > 1e-9 * np.maximum(sm[:, :1], 1e-300)
            sinv4 = np.where(smask, 1 / np.where(sm > 0, sm, 1), 0)
            x = np.einsum("bji,bj->bi", Vm, sinv4 * np.einsum("bnj,bn->bj", Um, pp))
            r = np.abs(np.einsum("bnk,bk->bn", M, x) - pp).max(1)
            good = np.where((r < tol) & (res < 1e-6) & (r_nox >= tol))[0]
            for i in good:
                a, b, at, bt = x[i]
                nfree = int(4 - smask[i].sum())
                # knot consistency (only exact when the 4-system has a unique solution)
                tt = at / a if abs(a) > 1e-9 else (bt / b if abs(b) > 1e-9 else None)
                cons = None
                if tt is not None:
                    cons = abs(at - tt * a) < 1e-7 and abs(bt - tt * b) < 1e-7
                out.append(("q1", tuple(h[i]), int(nulldim[i]), dr, T0, tt, cons, nfree))
    print(s + nb, "hits q0", sum(1 for o in out if o[0] == "q0"), "q1", sum(1 for o in out if o[0] == "q1"),
          "q1cons", sum(1 for o in out if o[0] == "q1" and o[6]), "nullmax", int(maxnull), round(time.time() - t0, 1), flush=True)
pickle.dump(out, open(f"pb_T{T}_q{q}{'c' if CONST else ''}_{start}_{stop}.pkl", "wb"))
print("DONE", len(out))

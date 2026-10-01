"""W = 2 double AND, larger gate sets: batched linear prefilter + exact check of survivors.

Same model as dand4.py (two units + free affine pass, F in {0,1,2}, F = T + flip mod 2), but for
W = 2 every row with T + flip = 1 is fixed to F = 1 (28 rows for flip 0, 53 rows for flip 1).
A pair can only be feasible if the all-ones vector on those rows lies in the column span of
[C1, C2, pass] restricted to them. For a fixed gate 1 this span test is done for all gates 2
at once (batched 5x5 normal equations after projecting out span(C1, pass)); survivors get the
exact breadth-first check of dand4.py.
Usage: dand5.py R [--mode int|rel] [--jobs n]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
import multiprocessing as mp

sys.argv = [sys.argv[0], "2"] + sys.argv[1:]
import dand4 as D  # noqa: E402  (parses W=2 R ... from argv)
import numpy as np  # noqa: E402

GATES, K1, G1 = D.GATES, D.K1, D.G1
X81 = D.Xt
FIX = {}
for flip in D.FLIPS:
    t = (D.T0 + flip) % 2
    FIX[flip] = np.nonzero(t == 1)[0]  # rows with F = 1 exactly


def orth(A, tol=1e-9):
    if A.shape[1] == 0:
        return A
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int((s > tol * max(1.0, s[0])).sum())
    return u[:, :r]


CZ = {}


def czf(flip, k):
    key = (flip, k)
    if key not in CZ:
        js = np.array([j for j, g in enumerate(GATES) if g[2].shape[1] == k])
        CZ[key] = (js, np.stack([GATES[j][2][FIX[flip]] for j in js]))
    return CZ[key]


def survivors(i, flip):
    """gates 2 for which the pair's space has a nonzero G = F - 1 vanishing on the fixed rows
    (G must be +-1 on every other row, so dim {G in V : G = 0 on fixed rows} >= 1 is necessary)"""
    rows = FIX[flip]
    base = np.hstack([GATES[i][2][rows], X81[rows]])
    Qb = orth(base)
    r1 = Qb.shape[1]
    kb = base.shape[1]
    cand = []
    for k in sorted({g[2].shape[1] for g in GATES}):
        js, B = czf(flip, k)
        B = B - np.einsum("na,mak->mnk", Qb, np.einsum("na,mnk->mak", Qb, B))
        BtB = np.einsum("mnk,mnl->mkl", B, B)
        w = np.linalg.eigvalsh(BtB)
        scale = max(1.0, float(np.abs(base).max()) ** 2)
        r2 = (w > 1e-9 * scale).sum(1)
        d = kb + k - r1 - r2
        cand += list(js[d >= 1])
    return cand


def work(i):
    hits, n_exact, n = [], 0, 0
    w1, tau1, C1, a1 = GATES[i]
    for flip in D.FLIPS:
        cand = survivors(i, flip)
        if cand is None:
            cand = list(range(len(GATES)))
        for j in cand:
            if j in K1 and j < i:
                continue
            w2, tau2, C2, a2 = GATES[j]
            M = np.hstack([C1, C2, X81])
            act = np.stack([a1, a2], 1).astype(int)
            n_exact += 1
            out = D.feasible(M, flip, act)
            if isinstance(out, str) or out is None:
                continue
            hits.append((w1, tau1, w2, tau2, flip, bool(D.verify(out, flip)), [int(round(v)) for v in out]))
    return i, hits, n_exact


if __name__ == "__main__":
    t0 = time.time()
    print(f"dand5 W=2 R={D.R} mode={D.args.mode} gates={len(GATES)} g1={len(G1)}", flush=True)
    idx = list(G1)
    if D.args.limit:
        idx = idx[:D.args.limit]
    tot_hits = tot_exact = done = 0
    with mp.Pool(D.args.jobs) as pool:
        for i, hits, ne in pool.imap_unordered(work, idx, chunksize=1):
            done += 1
            tot_exact += ne
            if done % 100 == 0:
                print(f"  progress {done}/{len(idx)} exact_checks={tot_exact} hits={tot_hits} t={time.time() - t0:.0f}s", flush=True)
            for h in hits:
                tot_hits += 1
                if tot_hits <= 30:
                    print("HIT", h[:6], "F", h[6], flush=True)
    print(f"done g1={done} exact_checks={tot_exact} hits={tot_hits} time={time.time() - t0:.1f}s", flush=True)

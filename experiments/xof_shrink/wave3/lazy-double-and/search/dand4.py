"""Exhaustive search: K = 2 gated units + a free affine pass term for a mod-2 double AND of
E-coded bits (optimized version of dand3.py; same model, same symmetry reduction).

E in {0,1,2}^4 = (b1, c1, b2, c2); Q(b, c) = [Eb != 1][Ec == 1]; T = Q(b1,c1) + Q(b2,c2) mod 2.
F = sum_k max(0, w_k.E - tau_k + beta_k) (v_k0 + v_k.E) + (l0 + l.E), F in {0..W} on all 81
points, F = T + flip (mod 2), flip in {0, 1} (the window may start at an odd value).
Gates: integer w in [-R, R]^4, integer tau; mode int: beta = 0; mode rel: all beta in [0,1)
at once via the relaxation {[s >= tau](p(E) + s q(E))} (an infeasible relaxation proves that
no knot offset works for that gate pair).
Per pair: rows grouped by gate-activity region (both inactive first) and, inside a region,
single-valued rows first; pivots by blockwise QR; breadth-first enumeration of the pivot
values pruned by every row that the pivots so far determine. Exact up to float tolerance;
hits are re-verified.
Usage: dand4.py W R [--mode int|rel] [--jobs n]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import itertools
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.linalg import qr

ap = argparse.ArgumentParser()
ap.add_argument("W", type=int)
ap.add_argument("R", type=int)
ap.add_argument("--mode", default="int")
ap.add_argument("--jobs", type=int, default=28)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--maxpop", type=int, default=1 << 18)
ap.add_argument("--flips", default="01")
ap.add_argument("--single", action="store_true")
ap.add_argument("--only12", action="store_true")
args = ap.parse_args()
W, R = args.W, args.R

X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)  # b1 c1 b2 c2
Xt = np.hstack([X, np.ones((81, 1))])
Qf = lambda b, c: ((b != 1) & (c == 1)).astype(int)
T0 = Qf(X[:, 0], X[:, 1]) % 2 if args.single else (Qf(X[:, 0], X[:, 1]) + Qf(X[:, 2], X[:, 3])) % 2


def gate_cols(w, tau):
    s = X @ np.array(w, float)
    if args.mode == "int":
        r = np.maximum(0.0, s - tau)
        if not r.any():
            return None, None, None
        return r[:, None] * Xt, tuple(np.round(r / r.max(), 9)), r > 0
    A = s >= tau
    if not A.any():
        return None, None, None
    sa = s[A]
    lo, hi = sa.min(), sa.max()
    Af = A.astype(float)
    if hi > lo:
        sn = (s - lo) / (hi - lo)
        cols = np.hstack([Af[:, None] * Xt, (Af * sn)[:, None] * Xt])
        key = (tuple(A.astype(int)), tuple(np.round(sn * Af, 9)))
    else:
        cols = Af[:, None] * Xt
        key = (tuple(A.astype(int)), None)
    return cols, key, A


def taus_of(w):
    s = X @ np.array(w, float)
    lo, hi = int(round(s.min())), int(round(s.max()))
    return range(lo, hi) if args.mode == "int" else range(lo, hi + 1)


allw = [w for w in itertools.product(range(-R, R + 1), repeat=4) if any(w) and not (args.only12 and any(w[2:]))]
GATES, KEYS = [], {}
for w in allw:
    for tau in taus_of(w):
        cols, key, act = gate_cols(w, tau)
        if cols is None or key in KEYS:
            continue
        KEYS[key] = len(GATES)
        GATES.append((tuple(w), tau, cols, act))
K1 = set()
for w in allw:
    if min(w) >= 0:
        for tau in taus_of(w):
            _, key, _ = gate_cols(w, tau)
            if key is not None:
                K1.add(KEYS[key])
G1 = list(range(len(GATES))) if args.single else sorted(K1)

FLIPS = [int(ch) for ch in args.flips]
ALLOWED, NALL, VALS = {}, {}, {}
for flip in FLIPS:
    t = (T0 + flip) % 2
    ALLOWED[flip] = [np.array([k for k in range(W + 1) if k % 2 == t[i]], float) for i in range(81)]
    NALL[flip] = np.array([len(a) for a in ALLOWED[flip]])


def pivots_blockwise(M, groups):
    """independent rows of M, taking the groups in order (QR with pivoting per group)"""
    piv, Qb = [], np.zeros((0, M.shape[1]))
    for rows in groups:
        if len(piv) == M.shape[1]:
            break
        P = M[rows]
        if Qb.shape[0]:
            P = P - (P @ Qb.T) @ Qb
        nrm = np.abs(P).max()
        if nrm < 1e-9:
            continue
        Qm, Rm, pv = qr(P.T, pivoting=True, mode="economic")
        dg = np.abs(np.diag(Rm))
        rk = int((dg > 1e-9 * max(1.0, np.abs(M[rows]).max())).sum())
        if rk == 0:
            continue
        piv += [rows[j] for j in pv[:rk]]
        Qb = np.vstack([Qb, Qm[:, :rk].T])
    return piv


def feasible(M, flip, act):
    allowed = ALLOWED[flip]
    key = act.sum(1) * 4 + act[:, 0] * 2 + NALL[flip]
    groups = []
    for kv in np.unique(key):
        rows = np.nonzero(key == kv)[0]
        groups.append(rows)
    piv = pivots_blockwise(M, groups)
    r = len(piv)
    Mp = M[piv]
    A, *_ = np.linalg.lstsq(Mp.T, M.T, rcond=None)
    A = A.T
    A[np.abs(A) < 1e-10] = 0.0
    nz = A != 0
    level = np.where(nz.any(1), r - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    for i in np.nonzero(level < 0)[0]:
        if not (allowed[i] == 0).any():
            return None
    pop = np.zeros((1, 0))
    for L in range(r):
        vals = allowed[piv[L]]
        m = pop.shape[0]
        if len(vals) == 1:
            pop = np.hstack([pop, np.full((m, 1), vals[0])])
        else:
            pop = np.hstack([np.repeat(pop, len(vals), axis=0), np.tile(vals, m)[:, None]])
        rows = np.nonzero(level == L)[0]
        if len(rows):
            V = pop @ A[rows, :L + 1].T
            ok = np.ones(pop.shape[0], bool)
            for j, i in enumerate(rows):
                al = allowed[i]
                if len(al) == 1:
                    ok &= np.abs(V[:, j] - al[0]) < 1e-7
                else:
                    ok &= (np.abs(V[:, j:j + 1] - al[None, :]) < 1e-7).any(1)
            pop = pop[ok]
        if pop.shape[0] == 0:
            return None
        if pop.shape[0] > args.maxpop:
            return "SKIP"
    c, *_ = np.linalg.lstsq(Mp, pop[0], rcond=None)
    return M @ c


def verify(F, flip):
    t = (T0 + flip) % 2
    Fr = np.round(F)
    return (np.abs(F - Fr).max() < 1e-6 and Fr.min() >= 0 and Fr.max() <= W
            and (np.mod(Fr, 2) == t).all())


def work(i):
    w1, tau1, C1, a1 = GATES[i]
    hits, skips, n = [], 0, 0
    for j, (w2, tau2, C2, a2) in enumerate(GATES):
        if args.single:
            if j < i:
                continue
        elif j in K1 and j < i:
            continue
        M = np.hstack([C1, C2, Xt])
        act = np.stack([a1, a2], 1).astype(int)
        n += 1
        for flip in FLIPS:
            out = feasible(M, flip, act)
            if isinstance(out, str):
                skips += 1
                continue
            if out is not None:
                hits.append((w1, tau1, w2, tau2, flip, bool(verify(out, flip)), [int(round(v)) for v in out]))
                break
    return i, hits, skips, n


if __name__ == "__main__":
    t0 = time.time()
    print(f"W={W} R={R} mode={args.mode} flips={FLIPS} gates={len(GATES)} g1={len(G1)}", flush=True)
    idx = list(G1)
    if args.limit:
        idx = idx[:args.limit]
    tot_hits, tot_skips, tot_n, done = 0, 0, 0, 0
    with mp.Pool(args.jobs) as pool:
        for i, hits, skips, n in pool.imap_unordered(work, idx, chunksize=1):
            tot_skips += skips
            tot_n += n
            done += 1
            if done % 50 == 0:
                print(f"  progress {done}/{len(idx)} pairs={tot_n} hits={tot_hits} skips={tot_skips} t={time.time() - t0:.0f}s", flush=True)
            for h in hits:
                tot_hits += 1
                if tot_hits <= 30:
                    print("HIT", h[:6], "F", h[6], flush=True)
    print(f"done pairs={tot_n} hits={tot_hits} skips={tot_skips} time={time.time() - t0:.1f}s", flush=True)

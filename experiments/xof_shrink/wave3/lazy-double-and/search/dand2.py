"""Two gated units + a free affine pass term for a mod-2 double AND of E-coded bits.

E in {0,1,2}^4 = (b1, c1, b2, c2); theta = [E == 1]; Q(b, c) = [Eb != 1][Ec == 1].
Target T = Q(b1,c1) + Q(b2,c2) (mod 2) (or its complement, i.e. the window starts at an odd
value). Unit form: max(0, w.E - tau + beta) (v0 + v.E), integer w in [-R, R]^4, integer tau,
beta = 0 (knot on a lattice point) unless --beta is given.
F = u1 + u2 + (l0 + l.E), F integer on all 81 points, F in [0, W], F = T (mod 2).

Exhaustive over unordered gate pairs up to the symmetry group of T (reflection E_i -> 2 - E_i
of every coordinate, swap of the two AND terms): gate 1 is reflected to w1 >= 0.
Per pair the 15 unknowns (v1, v2, pass) are solved exactly: the rows with a single allowed
value fix an affine subspace, then the remaining rows are enumerated on a pivot basis.

Usage: dand2.py W R [--beta b] [--single] [--jobs n]
"""
import argparse
import itertools
import multiprocessing as mp
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("W", type=int)
ap.add_argument("R", type=int)
ap.add_argument("--beta", type=float, default=0.0)
ap.add_argument("--single", action="store_true", help="sanity: target Q(b1,c1) only")
ap.add_argument("--units", type=int, default=2)
ap.add_argument("--jobs", type=int, default=28)
ap.add_argument("--maxd", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only12", action="store_true")
args = ap.parse_args()
W, R, BETA = args.W, args.R, args.beta

X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)  # b1 c1 b2 c2
Xt = np.hstack([X, np.ones((81, 1))])
Qf = lambda b, c: ((b != 1) & (c == 1)).astype(int)
if args.single:
    T0 = Qf(X[:, 0], X[:, 1]) % 2
else:
    T0 = (Qf(X[:, 0], X[:, 1]) + Qf(X[:, 2], X[:, 3])) % 2


def gates_for(wset):
    out, seen = [], set()
    for w in wset:
        wv = np.array(w, float)
        s = X @ wv
        lo, hi = int(round(s.min())), int(round(s.max()))
        taus = range(lo, hi) if BETA == 0 else range(lo, hi + 1)
        for tau in taus:
            r = np.maximum(0.0, s - tau + BETA)
            if not r.any():
                continue
            key = tuple(np.round(r / r.max(), 9))
            if key in seen:
                continue
            seen.add(key)
            out.append((tuple(w), tau, r))
    return out


allw = [w for w in itertools.product(range(-R, R + 1), repeat=4) if any(w) and not (args.only12 and any(w[2:]))]
G_all = gates_for(allw)
if args.single:
    G1 = G_all
else:
    G1 = [g for g in G_all if min(g[0]) >= 0]


def solve_pair(cols, tgt):
    """cols: 81 x k basis of the function space. Return a solution F (81,) or None."""
    M = cols
    allowed_fixed = None
    # classify rows
    res = []
    for flip in (0, 1):
        t = (tgt + flip) % 2
        if W == 2:
            fx = t == 1  # value 1 exactly
            yfx = np.ones(fx.sum())
            free_vals = (0.0, 2.0)
        elif W == 1:
            # F in {0,1}: F = t exactly
            fx = np.ones(81, bool)
            yfx = t.astype(float)
            free_vals = None
        elif W == 3:
            fx = np.zeros(81, bool)
            yfx = np.zeros(0)
            free_vals = None
        else:
            raise SystemExit("W must be 1, 2 or 3")
        if W == 3:
            sol = enum_free(M, np.zeros((0, M.shape[1])), np.zeros(0), np.arange(81), t)
        else:
            Mf = M[fx]
            c0, *_ = np.linalg.lstsq(Mf, yfx, rcond=None)
            if np.abs(Mf @ c0 - yfx).max() > 1e-7:
                continue
            if W == 1:
                sol = M @ c0
            else:
                sol = enum_free(M, Mf, yfx, np.nonzero(~fx)[0], t, c0)
        if isinstance(sol, str):
            res.append(sol)
            continue
        if sol is not None:
            return sol, flip
    if res:
        return "SKIP", -1
    return None


def nullspace(A, tol=1e-9):
    if A.shape[0] == 0:
        return np.eye(A.shape[1])
    u, s, vt = np.linalg.svd(A)
    rank = int((s > tol * max(1.0, s[0] if len(s) else 1.0)).sum())
    return vt[rank:].T


def enum_free(M, Mf, yfx, free_idx, t, c0=None):
    if c0 is None:
        c0 = np.zeros(M.shape[1])
    N = nullspace(Mf)
    d = N.shape[1]
    Mr = M[free_idx]
    b0 = Mr @ c0
    if d == 0:
        v = b0
        vals = np.round(v)
        if np.abs(v - vals).max() < 1e-7 and vals.min() >= -1e-9 and vals.max() <= W + 1e-9 and \
                (np.mod(vals, 2) == t[free_idx]).all():
            return M @ c0
        return None
    A = Mr @ N
    # pivot rows of A
    from scipy.linalg import qr
    _, Rm, piv = qr(A.T, pivoting=True, mode="economic")
    dg = np.abs(np.diag(Rm))
    r = int((dg > 1e-9 * max(1.0, dg[0])).sum())
    if r > args.maxd:
        return "SKIP"
    piv = piv[:r]
    choices = [[k for k in range(W + 1) if k % 2 == t[free_idx[i]]] for i in piv]
    Y = np.array(list(itertools.product(*choices)), float).T  # r x ncomb
    Ap = A[piv]
    z, *_ = np.linalg.lstsq(Ap, Y - b0[piv, None], rcond=None)
    V = b0[:, None] + A @ z
    Vr = np.round(V)
    tf = t[free_idx][:, None]
    good = ((np.abs(V - Vr).max(0) < 1e-7) & (Vr.min(0) >= -1e-9) & (Vr.max(0) <= W + 1e-9)
            & (np.mod(Vr, 2) == tf).all(0))
    if good.any():
        j = int(np.argmax(good))
        c = c0 + N @ z[:, j]
        return M @ c
    return None


def cols_of(r):
    return r[:, None] * Xt


def work(i):
    w1, tau1, r1 = G1[i]
    C1 = cols_of(r1)
    hits, skips, n = [], 0, 0
    if args.units == 1:
        M = np.hstack([C1, Xt])
        out = solve_pair(M, T0)
        n = 1
        if isinstance(out, tuple) and not isinstance(out[0], str):
            hits.append((w1, tau1, None, None, out[1], out[0].tolist()))
        return i, hits, skips, n
    for (w2, tau2, r2) in G_all:
        if not args.single:
            # symmetry: skip pairs whose gate 2 is also reflectable to >= 0 and precedes gate 1
            if min(w2) >= 0 and (w2, tau2) < (w1, tau1):
                continue
        M = np.hstack([C1, cols_of(r2), Xt])
        out = solve_pair(M, T0)
        n += 1
        if out is None:
            continue
        sol, flip = out
        if isinstance(sol, str):
            skips += 1
            continue
        hits.append((w1, tau1, w2, tau2, flip, sol.tolist()))
    return i, hits, skips, n


if __name__ == "__main__":
    t0 = time.time()
    print(f"W={W} R={R} beta={BETA} single={args.single} units={args.units} gates={len(G_all)} g1={len(G1)}", flush=True)
    idx = list(range(len(G1)))
    if args.limit:
        idx = idx[:args.limit]
    tot_hits, tot_skips, tot_n = 0, 0, 0
    with mp.Pool(args.jobs) as pool:
        for i, hits, skips, n in pool.imap_unordered(work, idx, chunksize=1):
            tot_skips += skips
            tot_n += n
            for h in hits:
                tot_hits += 1
                if tot_hits <= 20:
                    print("HIT", h[:5], "F", [int(round(v)) for v in h[5]], flush=True)
    print(f"done pairs={tot_n} hits={tot_hits} skips={tot_skips} time={time.time() - t0:.1f}s", flush=True)

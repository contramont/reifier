"""K gated units + a free affine pass term for a mod-2 double AND of E-coded bits (exhaustive).

E in {0,1,2}^4 = (b1, c1, b2, c2); theta = [E == 1]; Q(b, c) = [Eb != 1][Ec == 1].
Target T = Q(b1,c1) + Q(b2,c2) (mod 2), and also its complement (window starting at an odd value).
F = sum_k max(0, w_k.E - tau_k + beta_k) (v_k0 + v_k.E) + (l0 + l.E),
F integer on all 81 points, F in [0, W], F = T (mod 2).

Gates: integer w in [-R, R]^4, integer tau.
  mode int : beta = 0 (knots on lattice points), unit space {r(E) (v0 + v.E)}, 5 unknowns.
  mode rel : every real beta in [0, 1) at once, by the linear relaxation
             {[s >= tau] (p(E) + s q(E)) : p, q affine} (10 unknowns) which contains
             (s - tau + beta) v(E) [s >= tau] for all beta; an infeasible relaxation proves
             that no knot offset works for that gate pair.
Symmetry: reflections E_i -> 2 - E_i (T invariant) and swapping the two AND terms. Every gate
orbit contains a gate with w >= 0; unordered pairs are processed once with gate 1 in that set.
Per pair the linear unknowns are eliminated exactly: pivot rows (single-valued rows first),
then a breadth-first enumeration of the pivot values, pruned by every row the pivots so far
determine.
Usage: dand3.py W R [--mode int|rel] [--jobs n]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import itertools
import multiprocessing as mp
import time

import numpy as np
from scipy.linalg import qr

ap = argparse.ArgumentParser()
ap.add_argument("W", type=int)
ap.add_argument("R", type=int)
ap.add_argument("--mode", default="int")
ap.add_argument("--single", action="store_true", help="sanity: target Q(b1,c1) only")
ap.add_argument("--units", type=int, default=2)
ap.add_argument("--jobs", type=int, default=28)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only12", action="store_true")
ap.add_argument("--region", action="store_true", help="pivot rows region by region (inactive first)")
ap.add_argument("--maxpop", type=int, default=1 << 20)
args = ap.parse_args()
W, R = args.W, args.R

X = np.array(list(itertools.product((0, 1, 2), repeat=4)), float)  # b1 c1 b2 c2
Xt = np.hstack([X, np.ones((81, 1))])
Qf = lambda b, c: ((b != 1) & (c == 1)).astype(int)
if args.single:
    T0 = Qf(X[:, 0], X[:, 1]) % 2
else:
    T0 = (Qf(X[:, 0], X[:, 1]) + Qf(X[:, 2], X[:, 3])) % 2


def gate_cols(w, tau):
    s = X @ np.array(w, float)
    if args.mode == "int":
        r = np.maximum(0.0, s - tau)
        if not r.any():
            return None, None
        return r[:, None] * Xt, tuple(np.round(r / r.max(), 9))
    A = (s >= tau).astype(float)
    if not A.any():
        return None, None
    sa = s[A > 0]
    lo, hi = sa.min(), sa.max()
    sn = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    cols = np.hstack([A[:, None] * Xt, (A * sn)[:, None] * Xt]) if hi > lo else A[:, None] * Xt
    return cols, (tuple(A.astype(int)), tuple(np.round(sn * A, 9)))


def gates_for(wset):
    out, seen = [], {}
    for w in wset:
        s = X @ np.array(w, float)
        lo, hi = int(round(s.min())), int(round(s.max()))
        taus = range(lo, hi) if args.mode == "int" else range(lo, hi + 1)
        for tau in taus:
            cols, key = gate_cols(w, tau)
            if cols is None:
                continue
            if key in seen:
                continue
            seen[key] = len(out)
            out.append((tuple(w), tau, cols, key))
    return out, seen


allw = [w for w in itertools.product(range(-R, R + 1), repeat=4) if any(w) and not (args.only12 and any(w[2:]))]
G_all, KEYS = gates_for(allw)
K1 = set()
for w in allw:
    if min(w) < 0:
        continue
    s = X @ np.array(w, float)
    lo, hi = int(round(s.min())), int(round(s.max()))
    for tau in (range(lo, hi) if args.mode == "int" else range(lo, hi + 1)):
        _, key = gate_cols(w, tau)
        if key is not None:
            K1.add(KEYS[key])
if args.single:
    G1 = list(range(len(G_all)))
else:
    G1 = sorted(K1)


def pivots_in_order(M, order, tol=1e-9):
    """greedy independent rows of M in the given row order (Gram-Schmidt)"""
    Qb = []
    piv = []
    for i in order:
        v = M[i].copy()
        for q in Qb:
            v -= (v @ q) * q
        nv = np.linalg.norm(v)
        if nv > tol * max(1.0, np.linalg.norm(M[i])):
            Qb.append(v / nv)
            piv.append(i)
            if len(piv) == M.shape[1]:
                break
    return piv


def feasible(M, allowed, order=None):
    """exists c: M c in allowed[i] (list of values) for every row i? returns F or None, or 'SKIP'"""
    n = M.shape[0]
    if order is None:
        nall = np.array([len(a) for a in allowed])
        order = list(np.argsort(nall, kind="stable"))
    piv = pivots_in_order(M, order)
    r = len(piv)
    Mp = M[piv]
    # express every row in the pivot rows: M = A Mp
    A, *_ = np.linalg.lstsq(Mp.T, M.T, rcond=None)
    A = A.T  # n x r
    A[np.abs(A) < 1e-10] = 0.0
    nz = A != 0
    level = np.where(nz.any(1), r - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    for i in np.nonzero(level < 0)[0]:
        if 0 not in allowed[i]:
            return None
    pop = np.zeros((1, 0))
    for L in range(r):
        vals = np.array(allowed[piv[L]], float)
        m = pop.shape[0]
        pop = np.hstack([np.repeat(pop, len(vals), axis=0), np.tile(vals, m)[:, None]])
        rows = np.nonzero(level == L)[0]
        if len(rows):
            V = pop @ A[rows, :L + 1].T  # m' x len(rows)
            ok = np.ones(pop.shape[0], bool)
            for j, i in enumerate(rows):
                al = np.array(allowed[i], float)
                ok &= (np.abs(V[:, j:j + 1] - al[None, :]) < 1e-7).any(1)
            pop = pop[ok]
        if pop.shape[0] == 0:
            return None
        if pop.shape[0] > args.maxpop:
            return "SKIP"
    c, *_ = np.linalg.lstsq(Mp, pop[0], rcond=None)
    return M @ c


ALLOWED = {}
for flip in (0, 1):
    t = (T0 + flip) % 2
    ALLOWED[flip] = [[k for k in range(W + 1) if k % 2 == t[i]] for i in range(81)]


NALL = {f: np.array([len(a) for a in ALLOWED[f]]) for f in ALLOWED}


def check_pair(M, act=None):
    """act: 81 x k activity pattern of the gates (rows ordered by region: fewest active first)"""
    skip = False
    for flip in (0, 1):
        if act is not None:
            key = act.sum(1) * 4 + act[:, 0] * 2 + NALL[flip]  # region, then single-valued rows first
            order = list(np.argsort(key, kind="stable"))
        else:
            order = None
        out = feasible(M, ALLOWED[flip], order)
        if isinstance(out, str):
            skip = True
            continue
        if out is not None:
            return out, flip
    return ("SKIP", -1) if skip else None


def work(i):
    w1, tau1, C1, _ = G_all[i]
    hits, skips, n = [], 0, 0
    if args.units == 1:
        out = check_pair(np.hstack([C1, Xt]))
        n = 1
        if out is not None and not isinstance(out[0], str):
            hits.append((w1, tau1, None, None, out[1], out[0].tolist()))
        elif out is not None:
            skips += 1
        return i, hits, skips, n
    for j, (w2, tau2, C2, _) in enumerate(G_all):
        if not args.single and j in K1 and j < i:
            continue  # both gates reflect to w >= 0: the pair is processed once
        if args.single and j < i:
            continue
        act = np.stack([np.abs(C1).sum(1) > 0, np.abs(C2).sum(1) > 0], 1).astype(int)
        out = check_pair(np.hstack([C1, C2, Xt]), act if args.region else None)
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
    print(f"W={W} R={R} mode={args.mode} single={args.single} units={args.units} gates={len(G_all)} g1={len(G1)}", flush=True)
    idx = list(G1)
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

"""Two units with knots at fixed offsets beta1, beta2 (e.g. 1/2: knots between lattice points).
Gate k: max(0, w.E - tau + beta_k), integer w in [-R, R]^4, integer tau; same model and exact
per-pair check as dand4.py. Unordered pairs up to the reflection symmetry (which preserves beta):
gate 1 (offset beta1) is reflected to w >= 0; for beta1 == beta2 each pair is processed once.
Usage: dand6.py W R beta1 beta2 [--jobs n]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, itertools
import multiprocessing as mp
from fractions import Fraction
W, R, B1, B2 = sys.argv[1], sys.argv[2], float(Fraction(sys.argv[3])), float(Fraction(sys.argv[4]))
jobs = 28
if "--jobs" in sys.argv:
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
sys.argv = [sys.argv[0], W, R]
import dand4 as D
import numpy as np


def gates(beta):
    out, seen = [], {}
    nonneg = set()
    for w in D.allw:
        s = D.X @ np.array(w, float)
        lo, hi = int(round(s.min())), int(round(s.max()))
        for tau in range(lo, hi + 1):
            r = np.maximum(0.0, s - tau + beta)
            if not r.any() or (beta == 0 and tau == hi):
                continue
            key = tuple(np.round(r / r.max(), 9))
            if key not in seen:
                seen[key] = len(out)
                out.append((tuple(w), tau, r[:, None] * D.Xt, r > 0))
            if min(w) >= 0:
                nonneg.add(seen[key])
    return out, nonneg


GA, NA = gates(B1)
GB, NB = gates(B2)
same = B1 == B2
G1 = sorted(NA)


def work(i):
    w1, t1, C1, a1 = GA[i]
    hits, n = [], 0
    for j, (w2, t2, C2, a2) in enumerate(GB):
        if same and j in NB and j < i:
            continue
        n += 1
        M = np.hstack([C1, C2, D.Xt])
        act = np.stack([a1, a2], 1).astype(int)
        for flip in D.FLIPS:
            out = D.feasible(M, flip, act)
            if out is not None and not isinstance(out, str):
                hits.append((w1, t1, w2, t2, flip, bool(D.verify(out, flip)), [int(round(v)) for v in out]))
                break
    return i, hits, n


if __name__ == "__main__":
    t0 = time.time()
    print(f"dand6 W={W} R={R} beta1={B1} beta2={B2} gatesA={len(GA)} gatesB={len(GB)} g1={len(G1)}", flush=True)
    tot, nh, done = 0, 0, 0
    with mp.Pool(jobs) as pool:
        for i, hits, n in pool.imap_unordered(work, G1, chunksize=1):
            tot += n
            done += 1
            if done % 100 == 0:
                print(f"  progress {done}/{len(G1)} pairs={tot} hits={nh} t={time.time() - t0:.0f}s", flush=True)
            for h in hits:
                nh += 1
                print("HIT", h[:6], "F", h[6], flush=True)
    print(f"done pairs={tot} hits={nh} time={time.time() - t0:.1f}s", flush=True)

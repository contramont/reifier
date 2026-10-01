"""Exhaustive two-unit double-AND search for larger integer gate sets (R = 3) with a sound
box prefilter, then the exact per-pair check of dand4.py on the survivors.

Prefilter (sound): for a sub-box B of {0,1,2}^4 on which neither gate changes sign (every
point g <= 0, or every point g >= 0), F restricted to B is
  both off:       an affine function,
  only gate k on: l + g_k v_k   (g_k known, v_k and l free: linear in 10 unknowns),
  both on:        a quadratic polynomial (relaxation of l + g1 v1 + g2 v2).
If that class has no member satisfying F in [0, W], F = T + flip (mod 2) on B, the pair is
infeasible for that flip. Boxes: every product of per-coordinate intervals [0,1], [1,2], [0,2]
(81 boxes). Checked per gate (one-unit class) and once per box (affine, quadratic), combined per
pair with 81-bit masks. Survivors get the exact check (dand4.feasible, same model).
Usage: dand7.py W R [--jobs n]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
import itertools
import multiprocessing as mp

import numpy as np

W_, R_ = sys.argv[1], sys.argv[2]
jobs = 28
if "--jobs" in sys.argv:
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
limit = 0
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])
sys.argv = [sys.argv[0], W_, R_]
import dand4 as D  # noqa: E402

X = D.X
W = D.W
GATES, K1 = D.GATES, D.K1
G1 = sorted(K1)
BOXES = []
for ivs in itertools.product(((0, 1), (1, 2), (0, 2)), repeat=4):
    lo = np.array([a for a, _ in ivs], float)
    hi = np.array([b for _, b in ivs], float)
    idx = np.nonzero(np.all((X >= lo) & (X <= hi), 1))[0]
    BOXES.append(idx)
NB = len(BOXES)
XBOXES = []
for ivs in itertools.product(((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)), repeat=4):
    lo = np.array([a for a, _ in ivs], float)
    hi = np.array([b for _, b in ivs], float)
    XBOXES.append(np.nonzero(np.all((X >= lo) & (X <= hi), 1))[0])
NXW = (len(XBOXES) + 63) // 64
QCOLS = np.stack([np.ones(81)] + [X[:, i] for i in range(4)] + [X[:, i] * X[:, j] for i in range(4) for j in range(i, 4)], 1)


def sub_feasible(M, idx, flip):
    """dand4-style BFS on the rows idx only (no region ordering)"""
    allowed = [D.ALLOWED[flip][i] for i in idx]
    Ms = M[idx]
    act = np.zeros((len(idx), 2), int)
    # reuse dand4.feasible by building a restricted problem
    saved = D.ALLOWED[flip], D.NALL[flip]
    D.ALLOWED[flip] = allowed
    D.NALL[flip] = np.array([len(a) for a in allowed])
    try:
        out = D.feasible(Ms, flip, act)
    finally:
        D.ALLOWED[flip], D.NALL[flip] = saved
    return out is not None and not isinstance(out, str)


def box_masks():
    aff, quad = {}, {}
    for f in D.FLIPS:
        a = q = 0
        for b, idx in enumerate(BOXES):
            if sub_feasible(D.Xt, idx, f):
                a |= 1 << b
            if sub_feasible(QCOLS, idx, f):
                q |= 1 << b
        aff[f], quad[f] = a, q
    return aff, quad


def xoff_words(s):
    words = np.zeros(NXW, np.uint64)
    for b, idx in enumerate(XBOXES):
        if s[idx].max() <= 0:
            words[b // 64] |= np.uint64(1) << np.uint64(b % 64)
    return words


def gate_masks(i):
    w, tau, C, act = GATES[i]
    s = X @ np.array(w, float) - tau
    cut = off = on = 0
    oneok = {f: 0 for f in D.FLIPS}
    M1 = np.hstack([C, D.Xt])
    for b, idx in enumerate(BOXES):
        sb = s[idx]
        if sb.max() <= 0:
            off |= 1 << b
        elif sb.min() >= 0:
            on |= 1 << b
            for f in D.FLIPS:
                if sub_feasible(M1, idx, f):
                    oneok[f] |= 1 << b
        else:
            cut |= 1 << b
    return i, cut, off, on, oneok, xoff_words(s)


def split(m):
    return np.uint64(m & 0xFFFFFFFFFFFFFFFF), np.uint64(m >> 64)


if __name__ == "__main__":
    t0 = time.time()
    print(f"dand7 W={W} R={D.R} gates={len(GATES)} g1={len(G1)} boxes={NB}", flush=True)
    AFF, QUAD = box_masks()
    print("box classes done", {f: (bin(AFF[f]).count("1"), bin(QUAD[f]).count("1")) for f in D.FLIPS},
          f"t={time.time() - t0:.0f}s", flush=True)
    n = len(GATES)
    CUT = np.zeros((n, 2), np.uint64); OFF = np.zeros((n, 2), np.uint64); ON = np.zeros((n, 2), np.uint64)
    ONEOK = {f: np.zeros((n, 2), np.uint64) for f in D.FLIPS}
    XOFF = np.zeros((n, NXW), np.uint64)
    XAFFBAD = {}
    for f in D.FLIPS:
        wv = np.zeros(NXW, np.uint64)
        for b, idx in enumerate(XBOXES):
            if not sub_feasible(D.Xt, idx, f):
                wv[b // 64] |= np.uint64(1) << np.uint64(b % 64)
        XAFFBAD[f] = wv
    print("extra boxes", len(XBOXES), "affine-obstructed", {f: int(sum(bin(int(x)).count("1") for x in XAFFBAD[f])) for f in D.FLIPS}, flush=True)
    with mp.Pool(jobs) as pool:
        for i, cut, off, on, oneok, xo in pool.imap_unordered(gate_masks, range(n), chunksize=64):
            XOFF[i] = xo
            CUT[i] = split(cut); OFF[i] = split(off); ON[i] = split(on)
            for f in D.FLIPS:
                ONEOK[f][i] = split(oneok[f])
    print(f"gate masks done t={time.time() - t0:.0f}s", flush=True)
    np.savez("/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof3/lazy-double-and/search/masks_W%d_R%d.npz" % (W, D.R),
             CUT=CUT, OFF=OFF, ON=ON, **{f"ONEOK{f}": ONEOK[f] for f in D.FLIPS})
    nAFF = {f: np.array(split(~AFF[f] & ((1 << NB) - 1)), np.uint64) for f in D.FLIPS}
    nQUAD = {f: np.array(split(~QUAD[f] & ((1 << NB) - 1)), np.uint64) for f in D.FLIPS}
    surv = []
    total = 0
    idx1 = G1[:limit] if limit else G1
    inK1 = np.zeros(n, bool); inK1[list(K1)] = True
    for i in idx1:
        js = np.arange(n)
        keep = ~(inK1 & (js < i))
        total += int(keep.sum())
        for f in D.FLIPS:
            bad = ((OFF[i] & OFF & nAFF[f]) | (ON[i] & OFF & ~ONEOK[f][i]) | (OFF[i] & ON & ~ONEOK[f])
                   | (ON[i] & ON & nQUAD[f]))
            ok = (bad == 0).all(1) & keep
            ok &= ((XOFF[i] & XOFF & XAFFBAD[f]) == 0).all(1)
            for j in np.nonzero(ok)[0]:
                surv.append((i, int(j), f))
    print(f"prefilter: pairs={total} survivors(pair,flip)={len(surv)} t={time.time() - t0:.0f}s", flush=True)

    def exact(item):
        i, j, f = item
        M = np.hstack([GATES[i][2], GATES[j][2], D.Xt])
        act = np.stack([GATES[i][3], GATES[j][3]], 1).astype(int)
        out = D.feasible(M, f, act)
        if isinstance(out, str):
            return item, "SKIP"
        return item, (None if out is None else [int(round(v)) for v in out])

    hits = skips = 0
    with mp.Pool(jobs) as pool:
        for k, (item, res) in enumerate(pool.imap_unordered(exact, surv, chunksize=256)):
            if res == "SKIP":
                skips += 1
            elif res is not None:
                hits += 1
                i, j, f = item
                print("HIT", GATES[i][:2], GATES[j][:2], f, res, flush=True)
            if (k + 1) % 200000 == 0:
                print(f"  exact {k + 1}/{len(surv)} hits={hits} t={time.time() - t0:.0f}s", flush=True)
    print(f"done pairs={total} survivors={len(surv)} hits={hits} skips={skips} time={time.time() - t0:.1f}s", flush=True)

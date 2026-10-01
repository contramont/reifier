"""Exhaustive search, integer gates on INDIVIDUAL bits: can the AND of two parities
T = (1 - p_B) p_C (B = bits 0..mB-1, C = bits mB..n-1, disjoint), with the singles p_B,
p_C and a constant free, be written with K gated units relu(w.x + b) * (u.x + c)?
Gates: w in {-W..W}^n, b in (1/2)Z (knots on or between lattice points), all distinct
relu patterns up to positive scale; values u, c real (solved exactly by least squares).
Symmetry: bit flips (x_i -> 1 - x_i maps T to an equivalent target modulo the free
columns), permutations inside B and inside C, and B <-> C when mB = mC. So gate 1 is
canonical: w >= 0, sorted inside B and inside C.
Necessary condition used as a prefilter (exact): on a 3-face where no gate changes sign,
every unit is a quadratic, so its third difference is 0; T's third difference is +-4 on
every 3-face with coordinates in both B and C, so each such face must be cut by a gate.
Within-B faces must be cut for p_C(base) = 1 or for p_C(base) = 0 (depending on the free
coefficient of p_B), within-C faces likewise with p_B."""
import itertools, sys, time
import numpy as np
import torch as t

t.set_default_dtype(t.float64)
import os
t.set_num_threads(int(os.environ.get("NT", "6")))
mB, mC, K, W = (int(v) for v in sys.argv[1:5])
HALF = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # 1: biases in Z/2, 0: in Z
n = mB + mC
X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)  # N x n
N = len(X)
pB = X[:, :mB].sum(1) % 2
pC = X[:, mB:].sum(1) % 2
T = ((1 - pB) * pC).astype(float)
F = np.stack([np.ones(N), pB, pC], 1).astype(float)
Xb = np.concatenate([np.ones((N, 1)), X], 1).astype(float)

# 3-faces: (triple, base assignment of the other coordinates) -> vertex indices
faces_mixed, faces_B0, faces_B1, faces_C0, faces_C1 = [], [], [], [], []
pw = 2 ** np.arange(n - 1, -1, -1)
for tri in itertools.combinations(range(n), 3):
    others = [i for i in range(n) if i not in tri]
    inB = sum(1 for i in tri if i < mB)
    for base in itertools.product([0, 1], repeat=len(others)):
        verts = []
        for v in itertools.product([0, 1], repeat=3):
            x = np.zeros(n, dtype=np.int64)
            x[others] = base
            x[list(tri)] = v
            verts.append(int((x * pw).sum()))
        bx = np.zeros(n, dtype=np.int64); bx[others] = base
        if 0 < inB < 3:
            faces_mixed.append(verts)
        elif inB == 3:
            (faces_B1 if bx[mB:].sum() % 2 else faces_B0).append(verts)
        else:
            (faces_C1 if bx[:mB].sum() % 2 else faces_C0).append(verts)
groups = [np.array(f) for f in (faces_mixed, faces_B0, faces_B1, faces_C0, faces_C1)]
print("faces", [len(g) for g in groups], flush=True)


def cutmask(G):  # G: M x N gate values -> list of bool arrays per group
    out = []
    for fv in groups:
        if len(fv) == 0:
            out.append(np.zeros((G.shape[0], 0), bool))
            continue
        v = G[:, fv]  # M x faces x 8
        out.append((v.max(2) > 0) & (v.min(2) < 0))
    return out


def packbits(bm):
    return np.packbits(bm, axis=1)


# all gates
t0 = time.time()
rng = range(-W, W + 1)
ws = np.array(list(itertools.product(rng, repeat=n)), dtype=np.int64)
ws = ws[np.abs(ws).sum(1) > 0]
gv = X @ ws.T  # N x M (without bias)
lo, hi = gv.min(0), gv.max(0)
step = 0.5 if HALF else 1.0
gates, pats = [], []
seen = set()
for j in range(ws.shape[0]):
    w = ws[j]
    g0 = gv[:, j]
    for b in np.arange(-hi[j] - 0.5, -lo[j] + 1.0, step):  # hyperplane anywhere through or near the cube
        g = g0 + b
        if g.max() <= 0:
            continue
        r = np.maximum(g, 0)
        key = tuple(np.round(r / r.max(), 9))
        if key in seen:
            continue
        seen.add(key)
        gates.append((tuple(w.tolist()), float(b)))
        pats.append(g)
    # always-on gates (g > 0 on the whole cube) at a few larger biases
    for b in (-lo[j] + 1.5, -lo[j] + 3.0, -lo[j] + 6.0):
        g = g0 + b
        r = g.astype(float)
        key = tuple(np.round(r / r.max(), 9))
        if key in seen:
            continue
        seen.add(key)
        gates.append((tuple(w.tolist()), float(b)))
        pats.append(g)
G = np.array(pats, dtype=float)  # M x N
M = len(G)
print("distinct gates", M, f"{time.time()-t0:.1f}s", flush=True)
cm = cutmask(G)
PM = [packbits(c) for c in cm]


def canonical(w):
    w = np.array(w)
    if (w < 0).any():
        return False
    if list(w[:mB]) != sorted(w[:mB], reverse=True) or list(w[mB:]) != sorted(w[mB:], reverse=True):
        return False
    if mB == mC and tuple(w[:mB]) < tuple(w[mB:]):
        return False
    return True


can = [i for i, (w, b) in enumerate(gates) if canonical(w)]
print("canonical first gates", len(can), flush=True)

Rt = t.tensor(np.maximum(G, 0))  # M x N
Xbt = t.tensor(Xb)
Ft = t.tensor(F)
Tt = t.tensor(T)
full = [np.packbits(np.ones((1, c.shape[1]), bool), axis=1)[0] for c in cm]


def covers(masks, gi):
    return None


found = 0
checked = 0
t1 = time.time()


def solve(idx_tuples):
    """idx_tuples: L x K gate indices -> max abs residual of the exact least squares"""
    idx = t.tensor(idx_tuples)
    L = idx.shape[0]
    cols = [Ft.expand(L, -1, -1)]
    for k in range(K):
        r = Rt[idx[:, k]]  # L x N
        cols.append(r[:, :, None] * Xbt[None])
    Phi = t.cat(cols, 2)
    sol = t.linalg.lstsq(Phi, Tt.expand(L, -1)[..., None], driver="gelsd").solution
    return ((Phi @ sol).squeeze(2) - Tt).abs().max(1).values


def ok_masks(sel_masks):
    """sel_masks: list over groups of (L x bytes) OR-ed masks -> bool L"""
    mixed = (sel_masks[0] == full[0]).all(1)
    b_ok = (sel_masks[1] == full[1]).all(1) | (sel_masks[2] == full[2]).all(1) if len(full[1]) else np.ones(len(mixed), bool)
    c_ok = (sel_masks[3] == full[3]).all(1) | (sel_masks[4] == full[4]).all(1) if len(full[3]) else np.ones(len(mixed), bool)
    return mixed & b_ok & c_ok


results = []
if K == 1:
    ok = ok_masks([p[can] for p in PM])
    cand = [[can[i]] for i in np.nonzero(ok)[0]]
    print("candidates after cut filter", len(cand))
    if cand:
        res = solve(cand)
        for c_, r_ in zip(cand, res.tolist()):
            if r_ < 1e-8:
                found += 1
                print("FOUND", [gates[i] for i in c_])
    print("K 1 solutions", found)
    sys.exit()

# K >= 2: gate 1 canonical, gates 2..K any (index > 0); for K = 3, gate 3 index > gate 2
allidx = np.arange(M)
START = int(os.environ.get("START", "0"))
STOP = int(os.environ.get("STOP", "1000000000"))
for ci, g1 in enumerate(can):
    if ci < START or ci >= STOP:
        continue
    base = [p[g1][None] for p in PM]
    if K == 2:
        sel = [base[q] | PM[q] for q in range(5)]
        ok = ok_masks(sel)
        cand = np.nonzero(ok)[0]
        checked += len(cand)
        for s in range(0, len(cand), 20000):
            chunk = cand[s:s + 20000]
            res = solve([[g1, int(j)] for j in chunk])
            good = (res < 1e-8).nonzero().flatten().tolist()
            for gi in good:
                found += 1
                if found <= 5:
                    print("FOUND", gates[g1], gates[int(chunk[gi])], flush=True)
    else:  # K == 3
        for g2 in range(M):
            sel = [base[q] | PM[q][g2][None] | PM[q] for q in range(5)]
            ok = ok_masks(sel)
            ok[: g2 + 1] = False
            cand = np.nonzero(ok)[0]
            checked += len(cand)
            for s in range(0, len(cand), 20000):
                chunk = cand[s:s + 20000]
                res = solve([[g1, g2, int(j)] for j in chunk])
                good = (res < 1e-8).nonzero().flatten().tolist()
                for gi in good:
                    found += 1
                    if found <= 5:
                        print("FOUND", gates[g1], gates[g2], gates[int(chunk[gi])], flush=True)
    if ci % 50 == 0:
        print(f"gate1 {ci}/{len(can)} lstsq-checked {checked} found {found} {time.time()-t1:.0f}s", flush=True)
print("K", K, "W", W, "half", HALF, "canonical gate1", len(can), "gates", M, "checked", checked, "solutions", found)

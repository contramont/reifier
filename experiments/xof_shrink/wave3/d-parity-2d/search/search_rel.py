"""Complete search over ALL real gates (not only integer directions): relaxed test per 4-tuple of
active sets. Every real line that can appear (N1) has the active set of a generic line from
gen_active (836 oriented active sets, complete). A unit relu(g) v is zero off its active set S and a
quadratic on S, so c0 + sum_k [S_k] * (any quadratic) (25 unknowns) contains every exact form with these
active sets. If that relaxed system is infeasible for every 4-tuple passing the necessary conditions,
no exact 4-unit D exists for any real gates. Filters (all valid for any real lines with these active
sets, see NOTES.md): N1 (config types), F3 (active non-crossing line on each edge), N3 (cell cover),
N4' (every row/column: the knots split its 6 points into pieces of <= 3 points)."""
import sys, time, json, itertools
import numpy as np
from gen_active import gen
from lines import PTS
T0 = time.time()
def tm(): return round(time.time() - T0, 1)
PLANT = None
if len(sys.argv) > 1: PLANT = int(sys.argv[1])
acts = gen(10)
items = list(acts.items())
n = len(items)
X = np.array([p[0] for p in PTS], float); Y = np.array([p[1] for p in PTS], float)
IDX = {p: i for i, p in enumerate(PTS)}
QUAD = np.stack([np.ones(36), X, Y, X * X, X * Y, Y * Y], 1)
S = np.array([k for k, _ in items], bool)
Gv = np.array([[a * x + b * y + float(c) for x, y in PTS] for _, ((a, b, c), _) in items])
typ = ["".join(sorted(v[1])) for _, v in items]
TGT = (X + Y) % 2
# per-line relaxed blocks
UR = S[:, :, None] * QUAD[None]  # n x 36 x 6
# N3 cover masks (generic lines: strict signs everywhere)
cells = [(i, j) for i in range(5) for j in range(5)]
cid = {c: k for k, c in enumerate(cells)}
adj = []
for i in range(5):
    for j in range(5):
        if i < 4: adj.append((cid[(i, j)], cid[(i + 1, j)]))
        if j < 4: adj.append((cid[(i, j)], cid[(i, j + 1)]))
cov = np.zeros(n, np.uint64)
for l in range(n):
    cut = set()
    for (i, j) in cells:
        v = [S[l, IDX[(i + di, j + dj)]] for di in (0, 1) for dj in (0, 1)]
        if any(v) and not all(v): cut.add(cid[(i, j)])
    m = 0
    for e, (c1, c2) in enumerate(adj):
        if c1 in cut or c2 in cut: m |= 1 << e
    cov[l] = m
FULL = np.uint64((1 << len(adj)) - 1)
# F3 act bits (edge fully active); 4 bits B,T,L,R
EP = {'B': [IDX[(x, 0)] for x in range(6)], 'T': [IDX[(x, 5)] for x in range(6)],
      'L': [IDX[(0, y)] for y in range(6)], 'R': [IDX[(5, y)] for y in range(6)]}
actm = np.zeros(n, np.uint64)
for l in range(n):
    for bi, e in enumerate("BTLR"):
        if S[l, EP[e]].all(): actm[l] |= np.uint64(1 << bi)
# N4' knot interval masks: 12 lines x 5 bits
LINES = [[IDX[(x, y)] for x in range(6)] for y in range(6)] + [[IDX[(x, y)] for y in range(6)] for x in range(6)]
knot = np.zeros(n, np.uint64)
for l in range(n):
    m = 0
    for li, idx in enumerate(LINES):
        s = S[l, idx]
        for i in range(5):
            if s[i] != s[i + 1]: m |= 1 << (5 * li + i)
    knot[l] = m
okpiece = np.zeros(32, bool)
for K in range(32):
    ks = [i for i in range(5) if K >> i & 1]
    if not ks: continue
    gaps = [ks[0] + 1] + [ks[j + 1] - ks[j] for j in range(len(ks) - 1)] + [5 - ks[-1]]
    okpiece[K] = max(gaps) <= 3
def piece_ok(km):
    ok = np.ones(len(km), bool)
    for li in range(12):
        ok &= okpiece[((km >> np.uint64(5 * li)) & np.uint64(31)).astype(np.int64)]
    return ok
actm[:] = np.uint64(15)  # F3 disabled: not a valid lemma (f3where.py)
if PLANT is not None:  # validation: plant a random relaxed 4-tuple target (filters N3/F3/N4' off)
    rng = np.random.default_rng(PLANT)
    cfgs = [("BL", "BL", "RT", "RT"), ("BL", "BR", "LT", "RT"), ("BR", "BR", "LT", "LT"), ("BT", "BT", "LR", "LR"), ("BT", "LR", "BL", "RT"), ("BT", "LR", "BR", "LT")]
    cfg = cfgs[PLANT % 6]
    byt = {}
    for l in range(n): byt.setdefault(typ[l], []).append(l)
    while True:
        pl = [int(rng.choice(byt[t])) for t in cfg]
        if len(set(pl)) == 4: break
    TGT = np.zeros(36)
    for l in pl: TGT = TGT + UR[l] @ rng.integers(-3, 4, 6)
    TGT += 1.0
    cov[:] = FULL; actm[:] = np.uint64(15); okpiece[:] = True
    print("planted", cfg, sorted(pl), flush=True)
bytype = {}
for l in range(n): bytype.setdefault(typ[l], []).append(l)
bytype = {k: np.array(v) for k, v in bytype.items()}
print("active sets", n, {k: len(v) for k, v in bytype.items()}, "t", tm(), flush=True)
def resid_batch(Ms, t):
    out = np.empty(len(Ms))
    CH = 20000
    for s in range(0, len(Ms), CH):
        M = Ms[s:s + CH]
        u, sv, _ = np.linalg.svd(M, full_matrices=False)
        keep = (sv > 1e-9 * np.maximum(sv[:, :1], 1.0))[:, None, :]
        u = u * keep
        proj = np.einsum('nmk,nk->nm', u, np.einsum('nmk,m->nk', u, t))
        out[s:s + CH] = np.abs(t[None, :] - proj).max(1)
    return out
stats = {}
survivors = []
def final(sets, lab):
    if len(sets) == 0: return
    sets = np.unique(np.sort(sets, 1), axis=0)
    stats[lab + "_final"] = stats.get(lab + "_final", 0) + len(sets)
    CH = 20000
    for s in range(0, len(sets), CH):
        Sx = sets[s:s + CH]
        Ms = np.concatenate([np.ones((len(Sx), 36, 1))] + [UR[Sx[:, k]] for k in range(4)], 2)
        r = resid_batch(Ms, TGT)
        for k in np.nonzero(r < 1e-7)[0]:
            survivors.append((lab, [int(v) for v in Sx[k]]))
REG = {"xy<=5": np.nonzero(X + Y <= 5)[0], "xy>=5": np.nonzero(X + Y >= 5)[0],
       "y<=x": np.nonzero(Y <= X)[0], "y>=x": np.nonzero(Y >= X)[0]}
def local_ok(P, reg):
    idx = REG[reg]
    Ms = np.concatenate([np.broadcast_to(QUAD[idx], (len(P), len(idx), 6))] + [UR[P[:, k]][:, idx, :] for k in range(P.shape[1])], 2)
    return resid_batch(Ms, TGT[idx]) < 1e-7
def run_join(P1, P2, lab):
    """P1 (N1 x a), P2 (N2 x b) line-id arrays; all combos with filters -> final"""
    if len(P1) == 0 or len(P2) == 0: return
    c2 = np.bitwise_or.reduce(cov[P2], 1); a2 = np.bitwise_or.reduce(actm[P2], 1); k2 = np.bitwise_or.reduce(knot[P2], 1)
    buf = []; nb = 0
    for p in P1:
        c1 = np.bitwise_or.reduce(cov[p]); a1 = np.bitwise_or.reduce(actm[p]); k1 = np.bitwise_or.reduce(knot[p])
        ok = ((c2 | c1) == FULL) & ((a2 | a1) == np.uint64(15))
        ks = np.nonzero(ok)[0]
        if len(ks) == 0: continue
        ks = ks[piece_ok(k2[ks] | k1)]
        if len(ks) == 0: continue
        buf.append(np.concatenate([np.broadcast_to(p, (len(ks), len(p))), P2[ks]], 1)); nb += len(ks)
        if nb > 200000:
            final(np.concatenate(buf), lab); buf = []; nb = 0
    if buf: final(np.concatenate(buf), lab)
def pairs_same(t, reg=None):
    I = bytype[t]
    a, b = np.triu_indices(len(I), 1)
    P = np.stack([I[a], I[b]], 1)
    n0 = len(P)
    if reg is not None: P = P[local_ok(P, reg)]
    return P, n0
def pairs_cross(t1, t2):
    I, J = bytype[t1], bytype[t2]
    a, b = np.meshgrid(np.arange(len(I)), np.arange(len(J)), indexing='ij')
    return np.stack([I[a.ravel()], J[b.ravel()]], 1)
for t1, r1, t2, r2, lab in (("BL", "xy<=5", "RT", "xy>=5", "v-b"), ("BR", "y<=x", "LT", "y>=x", "v-c")):
    P1, n1 = pairs_same(t1, r1); P2, n2 = pairs_same(t2, r2)
    stats[lab + "_pairs"] = [n1, len(P1), n2, len(P2)]
    run_join(P1, P2, lab)
    print(lab, stats, "survivors", len(survivors), "t", tm(), flush=True)
P1, _ = pairs_same("BT"); P2, _ = pairs_same("LR")
run_join(P1, P2, "i"); print("i", stats, "survivors", len(survivors), "t", tm(), flush=True)
run_join(pairs_cross("BL", "BR"), pairs_cross("LT", "RT"), "v-a"); print("v-a", stats, "survivors", len(survivors), "t", tm(), flush=True)
run_join(pairs_cross("BT", "LR"), pairs_cross("BL", "RT"), "iii-a"); print("iii-a", stats, "survivors", len(survivors), "t", tm(), flush=True)
run_join(pairs_cross("BT", "LR"), pairs_cross("BR", "LT"), "iii-b"); print("iii-b", stats, "survivors", len(survivors), "t", tm(), flush=True)
out = {"n_active_sets": n, "stats": stats, "n_survivors": len(survivors), "survivors": survivors[:20000], "t": time.time() - T0}
if PLANT is not None: out["planted_found"] = any(sorted(v) == sorted(pl) for _, v in survivors); print("planted_found", out["planted_found"])
json.dump(out, open("rel_result.json" if PLANT is None else f"rel_plant{PLANT}.json", "w"))
print("DONE", json.dumps({k: v for k, v in out.items() if k != "survivors"}), flush=True)

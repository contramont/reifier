"""Scalable exhaustive search for K=4 units (see search5.py for the plain version; same space):
D = parity(x + y) on {0..5}^2 = c0 + sum_k relu(g_k) v_k, g_k = a x + b y + c, (a, b) integer
primitive, |a|, |b| <= A, c in Z/Q, both orientations; v_k real affine (free).
Necessary conditions used (NOTES.md): N1 edge structure (6 configs), N2 relaxed 1-D edge tests,
N3 cell cover, F3 (an active non-crossing line on every edge), local-region tests (lines of a
corner type are constant on a fixed half of the grid, so the others must fit there with a free
quadratic), then the exact 36-point test (float SVD, candidates re-checked in exact rationals).
usage: search6.py A Q [--plant SEED]  (plant: validation with a random planted 4-set target; N3/F3 off)"""
import sys, time, json, argparse
import numpy as np
from fractions import Fraction as Fr
from lines import enum_lines, PTS

ap = argparse.ArgumentParser()
ap.add_argument("A", type=int); ap.add_argument("Q", type=int)
ap.add_argument("--plant", type=int, default=None)
args = ap.parse_args()
A, Q = args.A, args.Q
T0 = time.time()
def tm(): return round(time.time() - T0, 1)
L = enum_lines(A, Q)
n = len(L)
X = np.array([p[0] for p in PTS], float); Y = np.array([p[1] for p in PTS], float)
IDX = {p: i for i, p in enumerate(PTS)}
G = np.array([[a * x + b * y + float(c) for x, y in PTS] for (a, b, c), _ in L])
Rv = np.maximum(G, 0)
U = np.stack([Rv, Rv * X, Rv * Y], 2)  # n x 36 x 3
typ = ["".join(sorted(ec)) for _, ec in L]
pos = [ec for _, ec in L]
bytype = {}
for l in range(n): bytype.setdefault(typ[l], []).append(l)
bytype = {k: np.array(v) for k, v in bytype.items()}
QUAD = np.stack([np.ones(36), X, Y, X * X, X * Y, Y * Y], 1)
TGT = (X + Y) % 2
PLANT = args.plant is not None
planted = None
if PLANT:
    rng = np.random.default_rng(args.plant)
    cfgs = [("BL", "BL", "RT", "RT"), ("BL", "BR", "LT", "RT"), ("BR", "BR", "LT", "LT"), ("BT", "BT", "LR", "LR"),
            ("BT", "LR", "BL", "RT"), ("BT", "LR", "BR", "LT")]
    cfg = cfgs[args.plant % len(cfgs)]
    while True:
        ls = [int(rng.choice(bytype[t])) for t in cfg]
        if len(set(ls)) < 4: continue
        if all(len(set(pos[l][e] for l in ls if e in pos[l])) == 2 for e in "BTLR"): break
    F = rng.integers(-3, 4) * np.ones(36)
    for l in ls:
        F = F + Rv[l] * (rng.integers(-3, 4) * X + rng.integers(-3, 4) * Y + rng.integers(-3, 4))
    TGT = F; planted = sorted(ls)
    print("planted", cfg, planted, [L[l][0] for l in planted], flush=True)
# ---- N3 cover masks
cells = [(i, j) for i in range(5) for j in range(5)]
cid = {c: k for k, c in enumerate(cells)}
adj = []
for i in range(5):
    for j in range(5):
        if i < 4: adj.append((cid[(i, j)], cid[(i + 1, j)], ((i + 1, j), (i + 1, j + 1))))
        if j < 4: adj.append((cid[(i, j)], cid[(i, j + 1)], ((i, j + 1), (i + 1, j + 1))))
cov = np.zeros(n, dtype=np.uint64)
for l in range(n):
    g = G[l]; cut = set()
    for (i, j) in cells:
        vals = [g[IDX[(i + di, j + dj)]] for di in (0, 1) for dj in (0, 1)]
        if max(vals) > 1e-9 and min(vals) < -1e-9: cut.add(cid[(i, j)])
    m = 0
    for e, (c1, c2, (pa, pb)) in enumerate(adj):
        if c1 in cut or c2 in cut or (abs(g[IDX[pa]]) < 1e-9 and abs(g[IDX[pb]]) < 1e-9): m |= 1 << e
    cov[l] = m
FULL = np.uint64((1 << len(adj)) - 1)
if PLANT: cov[:] = FULL
# ---- F3: line l active on the whole edge e (only meaningful when l does not cross e)
EDGE_PTS = {'B': [IDX[(x, 0)] for x in range(6)], 'T': [IDX[(x, 5)] for x in range(6)],
            'L': [IDX[(0, y)] for y in range(6)], 'R': [IDX[(5, y)] for y in range(6)]}
act = {e: np.array([bool((G[l, EDGE_PTS[e]] > 1e-9).all()) for l in range(n)]) for e in "BTLR"}
if PLANT or True:  # F3 is NOT a valid lemma (two facing knots cover the row; f3where.py): disabled
    for e in act: act[e][:] = True
# ---- batched residual: max |t - proj_col(M) t| per item
def resid_batch(Ms, t):
    """Ms: (N, m, k); t: (m,) or (N, m). returns (N,) max-abs residual."""
    out = np.empty(len(Ms))
    CH = 20000
    for s in range(0, len(Ms), CH):
        M = Ms[s:s + CH]
        tt = t if t.ndim == 1 else t[s:s + CH]
        u, sv, _ = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9 * np.maximum(sv[:, :1], 1.0)
        keep = (sv > tol)[:, None, :]
        u = u * keep
        proj = np.einsum('nmk,nk->nm', u, np.einsum('nmk,m->nk', u, tt) if tt.ndim == 1 else np.einsum('nmk,nm->nk', u, tt))
        out[s:s + CH] = np.abs((tt if tt.ndim == 2 else tt[None, :]) - proj).max(1)
    return out
EPS = 1e-7
# ---- relaxed 1-D edge test for pairs crossing edge e (free quadratic along the edge)
s6 = np.arange(6.0)
def edge_pair_mat(I, J, e):
    idx = EDGE_PTS[e]
    ii, jj = np.meshgrid(np.arange(len(I)), np.arange(len(J)), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    li, lj = I[ii], J[jj]
    okpos = np.array([li[k] != lj[k] and pos[li[k]][e] != pos[lj[k]][e] for k in range(len(li))], bool)
    base = np.stack([np.ones(6), s6, s6 ** 2], 1)
    ri, rj = Rv[li][:, idx], Rv[lj][:, idx]
    Ms = np.concatenate([np.broadcast_to(base, (len(li), 6, 3)), ri[:, :, None], (ri * s6)[:, :, None],
                         rj[:, :, None], (rj * s6)[:, :, None]], 2)
    r = resid_batch(Ms, TGT[idx])
    return ((r < EPS) & okpos).reshape(len(I), len(J))
PMC = {}
def pm(t1, t2, e):
    k = (t1, t2, e)
    if k not in PMC: PMC[k] = edge_pair_mat(bytype[t1], bytype[t2], e)
    return PMC[k]
# ---- local test: t on region pts in span(QUAD, units)
REG = {"xy<=5": np.nonzero(X + Y <= 5)[0], "xy>=5": np.nonzero(X + Y >= 5)[0],
       "y<=x": np.nonzero(Y <= X)[0], "y>=x": np.nonzero(Y >= X)[0]}
def local_ok(sets, reg):
    """sets: (N, k) line ids; test on region reg with free quadratic"""
    if len(sets) == 0: return np.zeros(0, bool)
    idx = REG[reg]
    out = np.empty(len(sets), bool)
    CH = 50000
    for s in range(0, len(sets), CH):
        S = sets[s:s + CH]
        blocks = [np.broadcast_to(QUAD[idx], (len(S), len(idx), 6))] + [U[S[:, k]][:, idx, :] for k in range(S.shape[1])]
        Ms = np.concatenate(blocks, 2)
        out[s:s + CH] = resid_batch(Ms, TGT[idx]) < EPS
    return out
# ---- final test
stats = {}
found = []
seen_final = set()
def final(sets, label):
    """sets: (N, 4) candidate 4-sets (already cover/F3/edge filtered)"""
    if len(sets) == 0: return
    sets = np.sort(sets, 1)
    sets = np.unique(sets, axis=0)
    stats[label + "_final_in"] = stats.get(label + "_final_in", 0) + len(sets)
    CH = 50000
    for s in range(0, len(sets), CH):
        S = sets[s:s + CH]
        Ms = np.concatenate([np.ones((len(S), 36, 1))] + [U[S[:, k]] for k in range(4)], 2)
        r = resid_batch(Ms, TGT)
        for k in np.nonzero(r < EPS)[0]:
            key = tuple(int(v) for v in S[k])
            if key in seen_final: continue
            seen_final.add(key)
            ex = exact_ok(key)
            rec = {"cfg": label, "lines": [[int(L[l][0][0]), int(L[l][0][1]), str(L[l][0][2])] for l in key], "resid": float(r[k]), "exact": bool(ex)}
            found.append(rec)
            if len(found) <= 30: print("FOUND", json.dumps(rec), flush=True)
def exact_ok(key):
    import sympy
    rows = []
    for (x, y) in PTS:
        row = [1]
        for l in key:
            (a, b, c), _ = L[l]
            r = max(Fr(0), a * x + b * y + c)
            row += [r, r * x, r * y]
        rows.append(row)
    M = sympy.Matrix(rows)
    tv = sympy.Matrix([sympy.nsimplify(round(float(v), 9)) for v in TGT])
    return M.rank() == M.row_join(tv).rank()
print("A", A, "Q", Q, "lines", n, {k: len(v) for k, v in bytype.items()}, "t", tm(), flush=True)
def same_pairs(t, edges, reg, need_act):
    Ids = bytype[t]
    M = np.ones((len(Ids), len(Ids)), bool)
    for e in edges: M &= pm(t, t, e)
    M = np.triu(M, 1)
    a, b = np.nonzero(M)
    P = np.stack([Ids[a], Ids[b]], 1)
    if need_act is not None:  # F3: the edge(s) this type does not cross need an active line from the pair
        okf = np.ones(len(P), bool)
        for e in need_act:
            okf &= act[e][P[:, 0]] | act[e][P[:, 1]]
        P = P[okf]
    n0 = len(P)
    P = P[local_ok(P, reg)] if len(P) else P
    return P, n0
def cover_join_final(P1, P2, label, extra=None):
    if len(P1) == 0 or len(P2) == 0: return
    c2 = cov[P2[:, 0]] | cov[P2[:, 1]]
    buf = []
    for p in P1:
        cp = cov[p[0]] | cov[p[1]]
        ok = (c2 | cp) == FULL
        if extra is not None: ok &= extra(p, P2)
        ks = np.nonzero(ok)[0]
        if len(ks):
            buf.append(np.concatenate([np.broadcast_to(p, (len(ks), 2)), P2[ks]], 1))
        if sum(len(b) for b in buf) > 200000:
            final(np.concatenate(buf), label); buf = []
    if buf: final(np.concatenate(buf), label)
# (v-b) BL,BL,RT,RT: BL pairs on region x+y<=5 (RT constant there); RT pairs on x+y>=5.
# F3: top & right edges need an active BL (big side); bottom & left need an active RT.
for (t1, r1, a1), (t2, r2, a2), lab in (((("BL"), "xy<=5", "TR"), ("RT", "xy>=5", "BL"), "v-b"),
                                         (("BR", "y<=x", "TL"), ("LT", "y>=x", "BR"), "v-c")):
    P1, n1 = same_pairs(t1, list(t1), r1, list(a1)); P2, n2 = same_pairs(t2, list(t2), r2, list(a2))
    stats[lab + "_pairs"] = [n1, len(P1), n2, len(P2)]
    print(lab, "pairs (edge-ok, local-ok)", n1, len(P1), n2, len(P2), "t", tm(), flush=True)
    cover_join_final(P1, P2, lab)
    print(" ", lab, stats, "found", len(found), "t", tm(), flush=True)
# (i) BT,BT,LR,LR. F3: BT pair has one line active on L and one on R; LR pair one on B, one on T.
Ids = bytype["BT"]; M = np.triu(pm("BT", "BT", "B") & pm("BT", "BT", "T"), 1); a, b = np.nonzero(M)
PBT = np.stack([Ids[a], Ids[b]], 1)
PBT = PBT[(act['L'][PBT[:, 0]] & act['R'][PBT[:, 1]]) | (act['R'][PBT[:, 0]] & act['L'][PBT[:, 1]])]
Ids = bytype["LR"]; M = np.triu(pm("LR", "LR", "L") & pm("LR", "LR", "R"), 1); a, b = np.nonzero(M)
PLR = np.stack([Ids[a], Ids[b]], 1)
PLR = PLR[(act['B'][PLR[:, 0]] & act['T'][PLR[:, 1]]) | (act['T'][PLR[:, 0]] & act['B'][PLR[:, 1]])]
stats["i_pairs"] = [len(PBT), len(PLR)]
print("i pairs", len(PBT), len(PLR), "t", tm(), flush=True)
cover_join_final(PBT, PLR, "i")
print("  i", stats, "found", len(found), "t", tm(), flush=True)
# (v-a) BL, BR, LT, RT: triples (BL, BR, LT) [bottom, left] + local on x+y<=5 (RT constant); join RT [top, right].
# F3: bottom needs LT or RT active on B; top needs BL or BR active on T; left: BR or RT on L; right: BL or LT on R.
BL, BR, LT, RT = (bytype[k] for k in ("BL", "BR", "LT", "RT"))
MB, ML, MT, MR = pm("BL", "BR", "B"), pm("BL", "LT", "L"), pm("LT", "RT", "T"), pm("BR", "RT", "R")
tri = []
for ia, ib in zip(*np.nonzero(MB)):
    ks = np.nonzero(ML[ia])[0]
    if len(ks): tri.append(np.stack([np.full(len(ks), ia), np.full(len(ks), ib), ks], 1))
tri = np.concatenate(tri) if tri else np.zeros((0, 3), int)
ntri0 = len(tri)
ids = np.stack([BL[tri[:, 0]], BR[tri[:, 1]], LT[tri[:, 2]]], 1) if len(tri) else np.zeros((0, 3), int)
okl = local_ok(ids, "xy<=5") if len(ids) else np.zeros(0, bool)
tri = tri[okl]
stats["v-a_triples"] = [ntri0, len(tri)]
print("v-a triples", ntri0, len(tri), "t", tm(), flush=True)
buf = []
for ia, ib, ic in tri:
    ks = np.nonzero(MT[ic] & MR[ib])[0]
    if len(ks) == 0: continue
    rts = RT[ks]
    bl, br, lt = BL[ia], BR[ib], LT[ic]
    ok = (cov[bl] | cov[br] | cov[lt] | cov[rts]) == FULL
    ok &= (act['B'][lt] | act['B'][rts]) & (act['T'][bl] | act['T'][br]) & (act['L'][br] | act['L'][rts]) & (act['R'][bl] | act['R'][lt])
    ks = np.nonzero(ok)[0]
    if len(ks): buf.append(np.stack([np.full(len(ks), bl), np.full(len(ks), br), np.full(len(ks), lt), rts[ks]], 1))
    if sum(len(b) for b in buf) > 200000: final(np.concatenate(buf), "v-a"); buf = []
if buf: final(np.concatenate(buf), "v-a")
print("  v-a", stats, "found", len(found), "t", tm(), flush=True)
# (iii) BT, LR, BL, RT: triples (BT, BL, LR) [bottom (BT,BL), left (LR,BL)] + local x+y<=5 (RT const); join RT [top (BT,RT), right (LR,RT)]
#       BT, LR, BR, LT: triples (BT, BR, LR) [bottom (BT,BR), right (LR,BR)] + local y<=x (LT const); join LT [top (BT,LT), left (LR,LT)]
# F3 (a): bottom: LR or RT active on B; top: LR or BL on T; left: BT or RT on L; right: BT or BL on R.
#     (b): bottom: LR or LT on B; top: LR or BR on T; left: BT or BR on L; right: BT or LT on R.
BT_, LR_ = bytype["BT"], bytype["LR"]
for c1, c2, e1, e2, reg, lab in (("BL", "RT", "L", "R", "xy<=5", "iii-a"), ("BR", "LT", "R", "L", "y<=x", "iii-b")):
    X1, X2 = bytype[c1], bytype[c2]
    MB1 = pm("BT", c1, "B"); ME1 = pm("LR", c1, e1); MT2 = pm("BT", c2, "T"); ME2 = pm("LR", c2, e2)
    tri = []
    for ibt, i1 in zip(*np.nonzero(MB1)):
        ks = np.nonzero(ME1[:, i1])[0]
        if len(ks): tri.append(np.stack([np.full(len(ks), ibt), np.full(len(ks), i1), ks], 1))
    tri = np.concatenate(tri) if tri else np.zeros((0, 3), int)
    ntri0 = len(tri)
    ids = np.stack([BT_[tri[:, 0]], X1[tri[:, 1]], LR_[tri[:, 2]]], 1) if len(tri) else np.zeros((0, 3), int)
    tri = tri[local_ok(ids, reg)] if len(ids) else tri
    stats[lab + "_triples"] = [ntri0, len(tri)]
    print(lab, "triples", ntri0, len(tri), "t", tm(), flush=True)
    buf = []
    for ibt, i1, ilr in tri:
        ks = np.nonzero(MT2[ibt] & ME2[ilr])[0]
        if len(ks) == 0: continue
        x2 = X2[ks]; bt, x1, lr = BT_[ibt], X1[i1], LR_[ilr]
        ok = (cov[bt] | cov[x1] | cov[lr] | cov[x2]) == FULL
        # F3 generic: for each edge, some line not crossing it is active on it
        four = [bt, lr, x1]
        for e in "BTLR":
            anyact = np.zeros(len(x2), bool)
            for l in four:
                if e not in pos[l] and act[e][l]: anyact |= True
            anyact |= np.array([e not in pos[v] for v in x2]) & act[e][x2]
            ok &= anyact
        ks = np.nonzero(ok)[0]
        if len(ks): buf.append(np.stack([np.full(len(ks), bt), np.full(len(ks), lr), np.full(len(ks), x1), x2[ks]], 1))
        if sum(len(b) for b in buf) > 200000: final(np.concatenate(buf), lab); buf = []
    if buf: final(np.concatenate(buf), lab)
    print(" ", lab, stats, "found", len(found), "t", tm(), flush=True)
res = {"A": A, "Q": Q, "lines": n, "stats": stats, "nfound": len(found), "found": found[:100], "planted": planted,
       "planted_found": bool(planted is not None and tuple(planted) in seen_final), "t": time.time() - T0}
print("DONE", json.dumps({k: v for k, v in res.items() if k != "found"}), flush=True)
tag = f"A{A}_Q{Q}" + (f"_plant{args.plant}" if PLANT else "")
json.dump(res, open(f"s6_{tag}.json", "w"))

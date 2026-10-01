"""Exhaustive search for K=4 units: D = parity(x + y) on {0..5}^2 = c0 + sum_k relu(g_k) v_k,
g_k = a x + b y + c, (a, b) integer primitive with |a|, |b| <= A, c in Z/Q, both orientations;
v_k = d x + e y + f real (free). Pruning by necessary conditions N1-N4 (see NOTES.md).
usage: search5.py A Q [--plant SEED] [--no-cover]"""
import sys, time, json, argparse
import numpy as np
from fractions import Fraction as Fr
from lines import enum_lines, PTS

ap = argparse.ArgumentParser()
ap.add_argument("A", type=int); ap.add_argument("Q", type=int)
ap.add_argument("--plant", type=int, default=None)
ap.add_argument("--no-cover", action="store_true")
args = ap.parse_args()
A, Q = args.A, args.Q
T0 = time.time()
L = enum_lines(A, Q)
n = len(L)
X = np.array([p[0] for p in PTS], float); Y = np.array([p[1] for p in PTS], float)
IDX = {p: i for i, p in enumerate(PTS)}
G = np.array([[a * x + b * y + float(c) for x, y in PTS] for (a, b, c), _ in L])
Rv = np.maximum(G, 0)
typ = ["".join(sorted(ec)) for _, ec in L]
pos = [ec for _, ec in L]
bytype = {}
for l in range(n): bytype.setdefault(typ[l], []).append(l)
bytype = {k: np.array(v) for k, v in bytype.items()}
TGT = (X + Y) % 2
planted = None
if args.plant is not None:  # validation: plant a random 4-set of one random config, random values
    rng = np.random.default_rng(args.plant)
    cfgs = [("BL", "BL", "RT", "RT"), ("BL", "BR", "LT", "RT"), ("BR", "BR", "LT", "LT"), ("BT", "BT", "LR", "LR"),
            ("BT", "LR", "BL", "RT"), ("BT", "LR", "BR", "LT")]
    cfg = cfgs[args.plant % len(cfgs)]
    while True:
        ls = [int(rng.choice(bytype[t])) for t in cfg]
        if len(set(ls)) < 4: continue
        # distinct edge positions required by N1
        okp = True
        for e in "BTLR":
            ps = [pos[l][e] for l in ls if e in pos[l]]
            if len(set(ps)) != len(ps): okp = False
        if okp: break
    F = rng.integers(-3, 4) * np.ones(36)
    for l in ls:
        F = F + Rv[l] * (rng.integers(-3, 4) * X + rng.integers(-3, 4) * Y + rng.integers(-3, 4))
    TGT = F
    planted = sorted(ls)
    print("planted", cfg, planted, [L[l][0] for l in planted], flush=True)
cells = [(i, j) for i in range(5) for j in range(5)]
cid = {c: k for k, c in enumerate(cells)}
adj = []
for i in range(5):
    for j in range(5):
        if i < 4: adj.append((cid[(i, j)], cid[(i + 1, j)], ((i + 1, j), (i + 1, j + 1))))
        if j < 4: adj.append((cid[(i, j)], cid[(i, j + 1)], ((i, j + 1), (i + 1, j + 1))))
cov = np.zeros(n, dtype=np.uint64)
for l in range(n):
    g = G[l]
    cut = set()
    for (i, j) in cells:
        vals = [g[IDX[(i + di, j + dj)]] for di in (0, 1) for dj in (0, 1)]
        if max(vals) > 1e-9 and min(vals) < -1e-9:
            cut.add(cid[(i, j)])
    m = 0
    for e, (c1, c2, (pa, pb)) in enumerate(adj):
        if c1 in cut or c2 in cut or (abs(g[IDX[pa]]) < 1e-9 and abs(g[IDX[pb]]) < 1e-9):
            m |= 1 << e
    cov[l] = m
FULL = np.uint64((1 << len(adj)) - 1)
if args.no_cover:
    cov[:] = FULL
line_idx = [[IDX[(x, y)] for x in range(6)] for y in range(6)] + [[IDX[(x, y)] for y in range(6)] for x in range(6)]
s6 = np.arange(6.0)
def resid(M, t):
    sol = np.linalg.lstsq(M, t, rcond=None)[0]
    return np.abs(M @ sol - t).max(), sol
def ok1d(ls, li, relaxed=False):
    idx = line_idx[li]
    cols = [np.ones(6)]
    if relaxed: cols += [s6, s6 ** 2]
    for l in ls:
        r = Rv[l, idx]
        cols += [r, r * s6]
    return resid(np.array(cols).T, TGT[idx])[0] < 1e-7
EL = {'B': 0, 'T': 5, 'L': 6, 'R': 11}
def pairmat(t1, t2, e):
    I, J = bytype[t1], bytype[t2]
    M = np.zeros((len(I), len(J)), bool)
    for a, i in enumerate(I):
        for b, j in enumerate(J):
            if i != j and pos[i][e] != pos[j][e]:
                M[a, b] = ok1d((i, j), EL[e], relaxed=True)
    return M
stats = {"join": 0, "cover": 0, "rows": 0, "full": 0}
found = []
def full_ok(ls):
    cols = [np.ones(36)]
    for l in ls:
        r = Rv[l]
        cols += [r, r * X, r * Y]
    return resid(np.array(cols).T, TGT)
def exact_ok(ls):
    import sympy
    rows = []
    for (x, y) in PTS:
        row = [1]
        for l in ls:
            (a, b, c), _ = L[l]
            r = max(Fr(0), a * x + b * y + c)
            row += [r, r * x, r * y]
        rows.append(row)
    M = sympy.Matrix(rows)
    tv = sympy.Matrix([sympy.nsimplify(v) for v in TGT])
    return M.rank() == M.row_join(tv).rank()
seen = set()
def finish(ls):
    key = tuple(sorted(int(l) for l in ls))
    if key in seen: return
    seen.add(key)
    stats["join"] += 1
    c = np.uint64(0)
    for l in ls: c |= cov[l]
    if c != FULL: return
    stats["cover"] += 1
    for li in range(12):
        if not ok1d(ls, li): return
    stats["rows"] += 1
    err, sol = full_ok(ls)
    if err < 1e-7:
        stats["full"] += 1
        ex = exact_ok(ls)
        rec = {"lines": [[int(L[l][0][0]), int(L[l][0][1]), str(L[l][0][2])] for l in key], "err": float(err), "exact": bool(ex)}
        found.append(rec)
        if len(found) <= 50: print("FOUND", json.dumps(rec), flush=True)
def cover_join(P1, P2):
    if len(P1) == 0 or len(P2) == 0: return
    c2 = cov[P2[:, 0]] | cov[P2[:, 1]]
    for p in P1:
        cp = cov[p[0]] | cov[p[1]]
        for k in np.nonzero((c2 | cp) == FULL)[0]:
            finish((p[0], p[1], P2[k, 0], P2[k, 1]))
print("A", A, "Q", Q, "lines", n, {k: len(v) for k, v in bytype.items()}, flush=True)
PM = {}
def pm(t1, t2, e):
    if (t1, t2, e) not in PM: PM[(t1, t2, e)] = pairmat(t1, t2, e)
    return PM[(t1, t2, e)]
def same_pairs(t, edges):
    M = np.ones((len(bytype[t]), len(bytype[t])), bool)
    for e in edges: M &= pm(t, t, e)
    M = np.triu(M, 1)
    a, b = np.nonzero(M)
    return np.stack([bytype[t][a], bytype[t][b]], 1)
for t1, t2 in (("BL", "RT"), ("BR", "LT"), ("BT", "LR")):
    P1 = same_pairs(t1, list(t1)); P2 = same_pairs(t2, list(t2))
    print(t1, t2, "pairs", len(P1), len(P2), "t", round(time.time() - T0, 1), flush=True)
    cover_join(P1, P2)
    print("  stats", stats, "t", round(time.time() - T0, 1), flush=True)
# (v-a): BL, BR, LT, RT
MB, MT, ML, MR = pm("BL", "BR", "B"), pm("LT", "RT", "T"), pm("BL", "LT", "L"), pm("BR", "RT", "R")
BL, BR, LT, RT = bytype["BL"], bytype["BR"], bytype["LT"], bytype["RT"]
for a, b in zip(*np.nonzero(MB)):
    s1 = np.nonzero(ML[a])[0]; s2 = np.nonzero(MR[b])[0]
    if len(s1) == 0 or len(s2) == 0: continue
    sub = MT[np.ix_(s1, s2)]
    ii, jj = np.nonzero(sub)
    if len(ii) == 0: continue
    lt, rt = LT[s1[ii]], RT[s2[jj]]
    c = cov[BL[a]] | cov[BR[b]] | cov[lt] | cov[rt]
    for k in np.nonzero(c == FULL)[0]:
        finish((BL[a], BR[b], lt[k], rt[k]))
print("v-a stats", stats, "t", round(time.time() - T0, 1), flush=True)
# (iii): BT, LR, x1 in {BL, BR} (bottom with BT), x2 in {RT, LT} (top with BT); LR pairs left/right
BT, LR = bytype["BT"], bytype["LR"]
for c1, c2 in (("BL", "RT"), ("BR", "LT")):
    MB1 = pm("BT", c1, "B"); MT2 = pm("BT", c2, "T")
    eLt, eRt = (c1, c2) if "L" in c1 else (c2, c1)
    MLl = pm("LR", eLt, "L"); MRr = pm("LR", eRt, "R")
    X1, X2 = bytype[c1], bytype[c2]
    for ib in range(len(BT)):
        s1 = np.nonzero(MB1[ib])[0]; s2 = np.nonzero(MT2[ib])[0]
        if len(s1) == 0 or len(s2) == 0: continue
        for i1 in s1:
            for i2 in s2:
                iL, iR = (i1, i2) if eLt == c1 else (i2, i1)
                lrs = np.nonzero(MLl[:, iL] & MRr[:, iR])[0]
                if len(lrs) == 0: continue
                c = cov[BT[ib]] | cov[X1[i1]] | cov[X2[i2]] | cov[LR[lrs]]
                for k in np.nonzero(c == FULL)[0]:
                    finish((BT[ib], LR[lrs[k]], X1[i1], X2[i2]))
    print("iii", c1, c2, "stats", stats, "t", round(time.time() - T0, 1), flush=True)
res = {"A": A, "Q": Q, "lines": n, "stats": stats, "found": found[:200], "nfound": len(found), "planted": planted,
       "planted_found": (planted is not None and any(sorted(int(v) for v in k) == planted for k in seen)), "t": time.time() - T0}
print("DONE", json.dumps({k: v for k, v in res.items() if k != "found"}), flush=True)
tag = f"A{A}_Q{Q}" + (f"_plant{args.plant}" if args.plant is not None else "") + ("_nocover" if args.no_cover else "")
json.dump(res, open(f"s5_{tag}.json", "w"))

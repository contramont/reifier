"""Exhaustive search: D = parity(x + y) on {0..5}^2 as c0 + sum_{k=1..4} relu(g_k) v_k,
g_k = a x + b y + c (integer primitive (a,b), |a|,|b| <= A, c in Z/Q), v_k affine (free, real).
Necessary conditions used for pruning (all proved in NOTES.md):
 N1 each line crosses exactly 2 open edges of [0,5]^2 and no corner; each open edge is crossed by exactly 2 lines
 N2 each boundary row/column passes the 1-D test with its two knots (relaxed: free quadratic)
 N3 every pair of edge-adjacent unit cells: one of them is strictly cut by a line, or a line contains the shared edge
 N4 each of the 12 rows/columns passes the exact 1-D test span{1, r_k, r_k * s}
 final: exact 36-point linear test (float, then exact rational re-check)"""
import sys, math, itertools, time, json
import numpy as np
from fractions import Fraction as Fr
from lines import enum_lines, PTS

A, Q = int(sys.argv[1]), int(sys.argv[2])
T0 = time.time()
L = enum_lines(A, Q)
n = len(L)
X = np.array([p[0] for p in PTS], float); Y = np.array([p[1] for p in PTS], float)
TGT = (X + Y) % 2
IDX = {p: i for i, p in enumerate(PTS)}
G = np.array([[a * x + b * y + float(c) for x, y in PTS] for (a, b, c), _ in L])  # gate values
Rv = np.maximum(G, 0)
typ = ["".join(sorted(ec)) for _, ec in L]
pos = [ec for _, ec in L]
# cells and adjacency edges
cells = [(i, j) for i in range(5) for j in range(5)]
cid = {c: k for k, c in enumerate(cells)}
adj = []  # (cell1, cell2, (pA, pB) shared segment endpoints)
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
# 1-D line tests: rows y=0..5 (vary x) and cols x=0..5 (vary y)
line_idx = [[IDX[(x, y)] for x in range(6)] for y in range(6)] + [[IDX[(x, y)] for y in range(6)] for x in range(6)]
s6 = np.arange(6.0)
def ok1d(ls, li, relaxed=False):
    idx = line_idx[li]
    t = TGT[idx]
    cols = [np.ones(6)]
    if relaxed: cols += [s6, s6 ** 2]
    for l in ls:
        r = Rv[l, idx]
        cols += [r, r * s6]
    M = np.array(cols).T
    sol = np.linalg.lstsq(M, t, rcond=None)[0]
    return np.abs(M @ sol - t).max() < 1e-7
EDGE_LINE = {'B': 0, 'T': 5, 'L': 6, 'R': 11}
def pair_ok(i, j, e):
    if pos[i][e] == pos[j][e]:
        return False
    return ok1d((i, j), EDGE_LINE[e], relaxed=True)
# full test
def full_ok(ls):
    cols = [np.ones(36)]
    for l in ls:
        r = Rv[l]
        cols += [r, r * X, r * Y]
    M = np.array(cols).T
    sol = np.linalg.lstsq(M, TGT, rcond=None)[0]
    return np.abs(M @ sol - TGT).max(), sol
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
    M = sympy.Matrix(rows); tv = sympy.Matrix([(x + y) % 2 for x, y in PTS])
    return M.rank() == M.row_join(tv).rank()
bytype = {}
for l in range(n): bytype.setdefault(typ[l], []).append(l)
print("A", A, "Q", Q, "lines", n, {k: len(v) for k, v in bytype.items()}, flush=True)
stats = {"cands4": 0, "cover": 0, "rows": 0, "full": 0}
found = []
def finish(ls):
    stats["cands4"] += 1
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
        rec = {"lines": [[L[l][0][0], L[l][0][1], str(L[l][0][2])] for l in ls], "err": float(err), "exact": bool(ex), "sol": [float(v) for v in sol]}
        found.append(rec)
        print("FOUND", json.dumps(rec), flush=True)
def pairs(t1, t2, edges):
    out = []
    A1, A2 = bytype[t1], bytype[t2]
    for i in A1:
        for j in A2:
            if t1 == t2 and j <= i: continue
            if all(pair_ok(i, j, e) for e in edges):
                out.append((i, j))
    return out
def cover_join(P1, P2, fin):
    """all (p, q) in P1 x P2 with cov union == FULL, vectorized on q"""
    if not P1 or not P2: return
    c2 = np.array([cov[a] | cov[b] for a, b in P2], dtype=np.uint64)
    for p in P1:
        cp = cov[p[0]] | cov[p[1]]
        ok = np.nonzero((c2 | cp) == FULL)[0]
        for k in ok:
            fin(p + P2[k])
# config (v-b)/(v-c): 2 BL + 2 RT / 2 BR + 2 LT ; config (i): 2 BT + 2 LR
for (t1, e1), (t2, e2) in ((("BL", "BL"), ("RT", "RT")), (("BR", "BR"), ("LT", "LT")), (("BT", "BT"), ("LR", "LR"))):
    P1 = pairs(t1, t1, [c for c in t1]); P2 = pairs(t2, t2, [c for c in t2])
    print(t1, t2, "pairs", len(P1), len(P2), "t", round(time.time() - T0, 1), flush=True)
    cover_join(P1, P2, finish)
    print("  stats", stats, "t", round(time.time() - T0, 1), flush=True)
# config (v-a): BL, BR, LT, RT: bottom (BL,BR), top (LT,RT), left (BL,LT), right (BR,RT)
PB = pairs("BL", "BR", ["B"]); PT = pairs("LT", "RT", ["T"])
print("v-a pairs", len(PB), len(PT), flush=True)
okL = {}; okR = {}
for p in PB:
    for q in PT:
        bl, br = p; lt, rt = q
        c = cov[bl] | cov[br] | cov[lt] | cov[rt]
        if c != FULL: continue
        k1 = (bl, lt)
        if k1 not in okL: okL[k1] = pair_ok(bl, lt, 'L')
        if not okL[k1]: continue
        k2 = (br, rt)
        if k2 not in okR: okR[k2] = pair_ok(br, rt, 'R')
        if not okR[k2]: continue
        stats["cands4"] += 1
        # finish() recounts; call the tail
        stats["cands4"] -= 1
        finish((bl, br, lt, rt))
print("  stats", stats, "t", round(time.time() - T0, 1), flush=True)
# config (iii): BT, LR, BL, RT  and  BT, LR, BR, LT
for c1, c2 in (("BL", "RT"), ("BR", "LT")):
    # bottom: (BT, c1) ; top: (BT, c2) ; left: LR with whichever of c1/c2 has L ; right: LR with the other
    PBc = pairs("BT", c1, ["B"])
    cnt = 0
    for bt, x1 in PBc:
        for x2 in bytype[c2]:
            if not pair_ok(bt, x2, "T"): continue
            for lr in bytype["LR"]:
                eL = x1 if "L" in c1 else x2
                eR = x2 if "R" in c2 else x1
                if not pair_ok(lr, eL, "L") or not pair_ok(lr, eR, "R"): continue
                finish((bt, lr, x1, x2))
    print("iii", c1, c2, "stats", stats, "t", round(time.time() - T0, 1), flush=True)
print("DONE A", A, "Q", Q, "stats", stats, "found", len(found), "t", round(time.time() - T0, 1), flush=True)
json.dump({"A": A, "Q": Q, "stats": stats, "found": found}, open(f"s4_A{A}_Q{Q}.json", "w"))

# 1-D feasibility of an alternating 6-point row with exactly 2 knots
import numpy as np, itertools
xs = np.arange(6.0)
def feas(t1, t2, o1, o2, G, start=0):
    tgt = (xs + start) % 2
    cols = [np.ones(6)]
    if G >= 1: cols.append(xs)
    if G >= 2: cols.append(xs**2)
    for t, o in ((t1, o1), (t2, o2)):
        r = np.maximum(0, o * (xs - t))
        cols += [r, r * xs]
    M = np.array(cols).T
    sol, res, rk, sv = np.linalg.lstsq(M, tgt, rcond=None)
    return np.linalg.norm(M @ sol - tgt) < 1e-9
grid = np.arange(0.05, 5, 0.05)
grid = np.unique(np.r_[grid, np.arange(1, 5)])
for G in (0, 1, 2):
    for start in (0, 1):
        ok = set()
        for t1 in grid:
            for t2 in grid:
                if t2 <= t1 + 1e-9: continue
                for o1 in (1, -1):
                    for o2 in (1, -1):
                        if feas(t1, t2, o1, o2, G, start):
                            ok.add((round(t1, 2), round(t2, 2), o1, o2))
        # summarize
        print("G", G, "start", start, "n feasible", len(ok))
        # print ranges of t1,t2 by orientation
        for o1 in (1, -1):
            for o2 in (1, -1):
                s = [(a, b) for a, b, p, q in ok if p == o1 and q == o2]
                if s:
                    t1s = sorted(set(a for a, b in s)); t2s = sorted(set(b for a, b in s))
                    print("  o", o1, o2, len(s), "t1 in", t1s[0], t1s[-1], "t2 in", t2s[0], t2s[-1])
                    # print a few sample pairs with integer or half-int values
                    samp = sorted([(a, b) for a, b in s if (a * 2) % 1 == 0 and (b * 2) % 1 == 0])
                    print("    half-int pairs:", samp[:40])

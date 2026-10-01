# enumerate oriented lines g = a x + b y + c (integer primitive (a,b), |a|,|b|<=A, c in Z/Q)
# that cross exactly 2 open edges of [0,5]^2 and no corner (necessary: see NOTES)
import numpy as np, math
from fractions import Fraction as Fr
PTS = [(x, y) for x in range(6) for y in range(6)]
def edge_cross(a, b, c):
    """crossings with the 4 open edges; returns dict edge->position (Fraction) or None if a corner is hit"""
    out = {}
    for x0, y0 in ((0, 0), (5, 0), (0, 5), (5, 5)):
        if a * x0 + b * y0 + c == 0:
            return None
    # bottom y=0: a x + c = 0
    if a != 0:
        xb = Fr(-c) / a
        if 0 < xb < 5: out['B'] = xb
        xt = Fr(-c - 5 * b) / a
        if 0 < xt < 5: out['T'] = xt
    if b != 0:
        yl = Fr(-c) / b
        if 0 < yl < 5: out['L'] = yl
        yr = Fr(-c - 5 * a) / b
        if 0 < yr < 5: out['R'] = yr
    return out
def enum_lines(A, Q):
    L = []
    for a in range(-A, A + 1):
        for b in range(-A, A + 1):
            if (a, b) == (0, 0) or math.gcd(a, b) != 1:
                continue
            lo = min(a * x + b * y for x, y in PTS); hi = max(a * x + b * y for x, y in PTS)
            for cq in range(-hi * Q - Q, -lo * Q + Q + 1):
                c = Fr(cq, Q)
                ec = edge_cross(a, b, c)
                if ec is None or len(ec) != 2:
                    continue
                L.append(((a, b, c), ec))
    return L
if __name__ == "__main__":
    import sys, collections
    A, Q = int(sys.argv[1]), int(sys.argv[2])
    L = enum_lines(A, Q)
    cnt = collections.Counter("".join(sorted(ec)) for _, ec in L)
    print(len(L), dict(cnt))

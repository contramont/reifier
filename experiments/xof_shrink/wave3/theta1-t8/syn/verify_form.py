# independent exact check (fractions) of a pool form on {0,1} x [0, tmax]
from fractions import Fraction as Fr
import sys, json
def check(form, tmax=8):
    bits, gam, ((xg, xc), (xv, xw)) = form["bit"], form["gamma"], form["extra"]
    gam = [Fr(g) for g in gam]
    bad = []
    for T in range(tmax + 1):
        X = max(0, xg * T + xc) * (Fr(xv) * T + Fr(xw))
        z = [max(0, gb * T + gc) * (Fr(vb) * T + Fr(vc)) for (ga, gb, gc), (va, vb, vc) in bits]
        zero = sum(g * zi for g, zi in zip(gam, z)) + X
        if zero != T % 2:
            bad.append(("zero", T, zero))
        for A in (0, 1):
            u = sum(max(0, ga * A + gb * T + gc) * (Fr(va) * A + Fr(vb) * T + Fr(vc)) for (ga, gb, gc), (va, vb, vc) in bits)
            live = u + sum((g - 1) * zi for g, zi in zip(gam, z)) + X
            if live != (A + T) % 2:
                bad.append(("live", A, T, live))
    # lattice gate values: 0 or |g| >= 1 (integer gates)
    gv = sorted({abs(ga * A + gb * T + gc) for (ga, gb, gc), _ in bits for A in (0, 1) for T in range(tmax + 1)} | {abs(xg * T + xc) for T in range(tmax + 1)})
    return bad, gv[:3]
F1 = {"bit": [((-2, 1, -4), (-19, 0, 8)), ((7, 2, -4), (3, 0, 4)), ((15, 2, -13), (0, 1, -10))], "gamma": ("1/2", "1/2", "-4/3"), "extra": ((1, 0), (-1, 2))}
F0 = {"bit": [((8, -1, 2), ("-29/23", -2, 14)), ((15, 2, -12), ("-118/23", 1, -1)), ((2, -2, 7), (-8, 2, -4))], "gamma": ("1/2", "1/2", "1/2"), "extra": ((2, -9), (-1, 6))}
TH1S7 = None
for name, F in (("F1", F1), ("F0", F0)):
    print(name, check(F))

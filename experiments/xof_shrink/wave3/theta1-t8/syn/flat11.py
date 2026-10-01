# Item 5: a 5-unit parity of [0, 11] (plus a free constant) with zero silu-smoothed slope at the
# integers 1..11 (glu_xor is flat there; at 0 it has slope 1). Counting: each of the 10 intervals
# [k, k+1], k = 1..10, needs a knot inside or a symmetric kink at an endpoint; 5 knots can serve
# at most 10 only if all sit on integers and pair up, i.e. knots exactly {2, 4, 6, 8, 10}, one unit
# each. So it suffices to check the 2^5 direction patterns exactly (linear system in the values).
import itertools, sympy as sp
knots = [2, 4, 6, 8, 10]
feasible = []
for dirs in itertools.product((1, -1), repeat=5):
    p = sp.symbols("p0:5"); q = sp.symbols("q0:5"); c = sp.Symbol("c")
    eqs = []
    def val(s):
        tot = c
        for i, (t, sg) in enumerate(zip(knots, dirs)):
            g = sg * (s - t)
            if g > 0:
                tot += g * (p[i] * s + q[i])
        return tot
    def slope(s):  # silu-smoothed derivative at integer s: average of one-sided derivatives
        tot = 0
        for i, (t, sg) in enumerate(zip(knots, dirs)):
            g = sg * (s - t)
            if g > 0:
                tot += sg * (p[i] * s + q[i]) + g * p[i]
            elif g == 0:
                tot += sp.Rational(1, 2) * sg * (p[i] * s + q[i])
        return tot
    for s in range(12):
        eqs.append(val(s) - (s % 2))
    for s in range(1, 12):
        eqs.append(slope(s))
    sol = sp.solve(eqs, list(p) + list(q) + [c], dict=True)
    if sol:
        feasible.append((dirs, sol))
print("direction patterns feasible:", len(feasible))
for f in feasible[:4]:
    print(f)

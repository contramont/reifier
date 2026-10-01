"""Is there a bfloat16-exact (dyadic) member of MINPAR7's family? (wave 4, bf16-weights)

MINPAR7 (min_parity / minpar_counts): parity of s in [0, 7] with 3 units + a constant, knots at
3 (integer), 3/2 and 23/5; its unit products are 25/6, 47/3, 5/3: not dyadic, so bf16 weights break
it (no rescaling helps: gate slope x value slope x out is invariant). With A's knot at 3, B's in
(1, 2), C's in (4, 5), the 8 lattice equations leave a 1-parameter family (7 linear unknowns + b0, c0,
one polynomial condition). This solves it symbolically and scans b0 = p/q (q <= 32) for rational c0
in (4, 5). Result: only b0 = 3/2, c0 = 23/5 (MINPAR7 itself) is rational, and it is not dyadic.
Run: python mp7_family.py  (sympy)"""
# MINPAR7's active pattern: A = (a0 - s)(pA + qA s) on s < a0 (a0 = 3, on an integer), B = (s - b0)(pB + qB s) on s > b0,
# C = (s - c0)(pC + qC s) on s > c0, plus C0. Solve the 8 lattice equations for the linear unknowns as functions of b0, c0.
import sympy as sp
C0, pA, qA, pB, qB, pC, qC, b0, c0 = sp.symbols("C0 pA qA pB qB pC qC b0 c0")
a0 = 3
def F(s, act):
    e = C0
    if "A" in act: e += (a0 - s) * (pA + qA * s)
    if "B" in act: e += (s - b0) * (pB + qB * s)
    if "C" in act: e += (s - c0) * (pC + qC * s)
    return e
pat = {0: "A", 1: "A", 2: "AB", 3: "B", 4: "B", 5: "BC", 6: "BC", 7: "BC"}
eqs = [sp.Eq(F(s, pat[s]), s % 2) for s in range(8)]
lin = [C0, pA, qA, pB, qB, pC, qC]
# solve 7 of them linearly, the 8th gives the condition
sol = sp.solve(eqs[:7], lin, dict=True)[0]
cond = sp.simplify(sp.together(eqs[7].lhs.subs(sol) - eqs[7].rhs))
num = sp.factor(sp.numer(cond))
print("condition numerator:", num)
print({k: sp.simplify(v) for k, v in sol.items()})
from fractions import Fraction as Fr
import math
def dy_bits(x):
    x = Fr(x)
    if x == 0: return 0
    d = x.denominator
    if d & (d - 1): return 99
    m = abs(x.numerator)
    while m % 2 == 0: m //= 2
    return m.bit_length()
res = []
seen = set()
for qd in range(1, 33):
    for p in range(qd + 1, 2 * qd):
        b = Fr(p, qd)
        if b in seen: continue
        seen.add(b)
        poly = sp.Poly(num.subs(b0, sp.Rational(b.numerator, b.denominator)), c0)
        for r in sp.roots(poly, filter=None).keys():
            if not r.is_rational: continue
            c = Fr(int(sp.numer(r)), int(sp.denom(r)))
            if not (4 < c < 5): continue
            v = {str(k): Fr(str(vv.subs({b0: sp.Rational(b.numerator, b.denominator), c0: sp.Rational(c.numerator, c.denominator)}))) for k, vv in sol.items()}
            # units with integer gates: B gate (den_b s - num_b), value (pB, qB) / den_b; C likewise
            db, dc = b.denominator, c.denominator
            bits = max(dy_bits(v["C0"]), dy_bits(v["pA"]), dy_bits(v["qA"]), dy_bits(v["pB"] / db), dy_bits(v["qB"] / db),
                       dy_bits(v["pC"] / dc), dy_bits(v["qC"] / dc), max(b.numerator, c.numerator).bit_length() if max(b.numerator, c.numerator) < 256 else 99)
            print(b, c, bits, {k: str(x) for k, x in v.items()})

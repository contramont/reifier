# exact sympy solve of (D) for the T=7 pool (1.5 inc, 3 dec, 5.4 dec) + const, given A=1 knot cells;
# knots given as numbers are fixed (rational), 'x' = unknown in cell (lo, hi]
import sympy as sp, sys, json
from fractions import Fraction as Fr
n = 7; ts = range(n + 1)
spec = json.loads(sys.argv[1])  # e.g. [[0,1], 7, [2,3]]  (cell for unknown knot, or fixed value)
pool = [(int(a), sp.nsimplify(b)) for a, b in json.loads(sys.argv[2])] if len(sys.argv) > 2 else [(1, sp.Rational(3, 2)), (-1, sp.Integer(3)), (-1, sp.Rational(27, 5))]
_c = sp.symbols("c0:7")
_eq = [sum(max(0, s * (T - t)) * (_c[j] + _c[3 + j] * T) for j, (s, t) in enumerate(pool)) + _c[6] - (T % 2) for T in ts]
_s = sp.solve(_eq, _c, dict=True)[0]
C0 = [_s[_c[j]] for j in range(3)]; C1 = [_s[_c[3 + j]] for j in range(3)]; KAPPA = _s[_c[6]]
print("pool", pool, "C0", C0, "C1", C1, "kappa", KAPPA)
lam = sp.symbols("l0:3"); d = sp.symbols("d0:3"); kn = sp.symbols("k0:3")
knots = []; cells = {}
for j, sj in enumerate(spec):
    if isinstance(sj, list):
        knots.append(kn[j]); cells[kn[j]] = (sp.nsimplify(sj[0]), sp.nsimplify(sj[1]))
    else:
        knots.append(sp.nsimplify(sj))
eqs = []
for T in ts:
    tot = 0
    for j, (s, t) in enumerate(pool):
        v = C1[j] * T + C0[j]
        r0 = max(0, s * (T - t))
        k1 = knots[j]
        if k1 in cells:
            lo, hi = cells[k1]
            # decide activity from the cell: active iff s*(T - k) > 0 for k in the open cell
            mid = (lo + hi) / 2
            act = s * (T - mid) > 0
            r1 = s * (T - k1) if act else 0
        else:
            r1 = max(0, s * (T - k1))
        tot += r1 * (lam[j] * v + d[j]) - lam[j] * r0 * v
    eqs.append(sp.expand(tot - (-1) ** T))
fixlam = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
subs = {lam[int(k)]: sp.nsimplify(v) for k, v in fixlam.items()}
eqs = [sp.expand(e.subs(subs)) for e in eqs]
unk = [l for l in lam if l not in subs] + list(d) + [k for k in kn if k in cells]
sols = sp.solve(eqs, unk, dict=True)
print("solutions", len(sols))
for s_ in sols:
    ok = True
    for k, (lo, hi) in cells.items():
        val = s_.get(k)
        if val is None or not val.is_real or not (lo < val <= hi):
            ok = False
    print("in-cell" if ok else "out", {str(a): str(b) for a, b in s_.items()}, {str(k): float(s_[k]) for k in cells if k in s_ and s_[k].is_real})

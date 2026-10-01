# Refine a T=7 q0 hit at 60 digits: pool (1.5 inc, 3 dec, 5.4 dec) + const, A=1 knots with tau1_fix rational.
import mpmath as mp, re, sys, numpy as np
from fractions import Fraction as Fr
mp.mp.dps = 60
n = 7
pool = [(1, Fr(3, 2)), (-1, Fr(3)), (-1, Fr(27, 5))]
ts = list(range(n + 1))
def relu(x):
    return x if x > 0 else 0
# exact pool values (P, Q, kappa) by rational linear algebra
import sympy as sp
Ps = sp.symbols("P0:3"); Qs = sp.symbols("Q0:3"); ka = sp.Symbol("ka")
eqs = []
for T in ts:
    e = sum(relu(s * (T - t)) * (Ps[j] * T + Qs[j]) for j, (s, t) in enumerate(pool)) + ka - (T % 2)
    eqs.append(sp.nsimplify(e))
sol = sp.solve(eqs, list(Ps) + list(Qs) + [ka], dict=True)[0]
P = [sol[Ps[j]] for j in range(3)]; Q = [sol[Qs[j]] for j in range(3)]; kap = sol[ka]
print("pool values P", P, "Q", Q, "kappa", kap)
def D_eqs(t1, lam, d):
    out = []
    for T in ts:
        tot = 0
        for j, (s, t) in enumerate(pool):
            v = mp.mpf(P[j].p) / P[j].q * T + mp.mpf(Q[j].p) / Q[j].q if hasattr(P[j], 'p') else mp.mpf(str(P[j])) * T + mp.mpf(str(Q[j]))
            r0 = max(mp.mpf(0), s * (T - mp.mpf(t.numerator) / t.denominator))
            r1 = max(mp.mpf(0), s * (T - t1[j]))
            tot += r1 * (lam[j] * v + d[j]) - lam[j] * r0 * v
        out.append(tot - (-1) ** T)
    return out
def refine(t1_float, fix_idx, fix_val):
    # float solve for lam, d at the float knots
    t1 = [mp.mpf(x) for x in t1_float]
    t1[fix_idx] = mp.mpf(fix_val.numerator) / fix_val.denominator
    free = [i for i in range(3) if i != fix_idx]
    # initial lam, d via least squares at float precision
    A = []; b = []
    for T in ts:
        row = []
        for j, (s, t) in enumerate(pool):
            v = float(P[j]) * T + float(Q[j]); r0 = max(0.0, s * (T - float(t))); r1 = max(0.0, s * (T - float(t1[j])))
            row += [r1 * v - r0 * v, r1]
        A.append(row); b.append((-1.0) ** T)
    x0 = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    lam0 = x0[0::2]; d0 = x0[1::2]
    def F(*z):
        tt = list(t1); tt[free[0]] = z[0]; tt[free[1]] = z[1]
        lam = z[2:5]; d = z[5:8]
        return D_eqs(tt, lam, d)
    z0 = [t1[free[0]], t1[free[1]]] + [mp.mpf(float(v)) for v in lam0] + [mp.mpf(float(v)) for v in d0]
    z = mp.findroot(F, z0, tol=mp.mpf(10) ** -50, maxsteps=200)
    tt = list(t1); tt[free[0]] = z[0]; tt[free[1]] = z[1]
    res = max(abs(r) for r in D_eqs(tt, z[2:5], z[5:8]))
    return tt, list(z[2:5]), list(z[5:8]), res
if __name__ == "__main__":
    t1f = [float(x) for x in sys.argv[1].split(",")]
    fi = int(sys.argv[2]); fv = Fr(sys.argv[3])
    tt, lam, d, res = refine(t1f, fi, fv)
    print("tau1", [mp.nstr(x, 30) for x in tt]); print("lam", [mp.nstr(x, 30) for x in lam]); print("d", [mp.nstr(x, 30) for x in d]); print("residual", mp.nstr(res, 5))

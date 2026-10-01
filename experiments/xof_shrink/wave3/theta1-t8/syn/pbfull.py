# Full exact PB-q1 form for a candidate: per-bit values (d,e,f), gamma, extra hinge; independent check.
import sympy as sp, itertools, sys, json
from fractions import Fraction as Fr
import pbcheck

def full(T, gates, dr, T0, aff=False):
    r = pbcheck.exact_check(T, gates, dr, T0, aff)
    if r[0] != "sols" or not r[2]:
        return None
    _, _, good, sol, params = r
    forms = []
    for kind, s in good:
        # substitute the D-null params and any remaining free symbols (set them to 0)
        sub = {k: v for k, v in s.items()}
        vals = [sp.simplify(sol[i].subs(sub)) for i in range(9)]
        free = set().union(*[v.free_symbols for v in vals]) | set().union(*[sp.sympify(v).free_symbols for v in s.values()])
        z = {f: 0 for f in free}
        vals = [sp.nsimplify(sp.simplify(v.subs(z))) for v in vals]
        g = [sp.nsimplify(sp.sympify(s.get(sp.Symbol(f"g{i}"), 0)).subs(z)) for i in range(3)]
        p = sp.nsimplify(sp.sympify(s[sp.Symbol("p")]).subs(z)); q = sp.nsimplify(sp.sympify(s[sp.Symbol("q")]).subs(z))
        t = sp.nsimplify(sp.sympify(s[sp.Symbol("t")]).subs(z))
        forms.append(dict(gates=gates, vals=vals, gamma=g, dr=dr, t=t, p=p, q=q, free=[str(f) for f in free]))
    return forms

def verify(T, F):
    """independent exact check on {0,1} x [0,T]: live = U(A,T) + sum (gamma_i - 1) phi_i + X; zero = sum gamma_i phi_i + X"""
    ok = True
    for Tv in range(T + 1):
        X = max(0, F["dr"] * (Tv - F["t"])) * (F["p"] * Tv + F["q"])
        phi = []
        U = {0: 0, 1: 0}
        for k, (a, b, c) in enumerate(F["gates"]):
            d, e, f = F["vals"][3 * k:3 * k + 3]
            phi.append(max(0, b * Tv + c) * (e * Tv + f))
            for A in (0, 1):
                U[A] += max(0, a * A + b * Tv + c) * (d * A + e * Tv + f)
        zero = sum(g * ph for g, ph in zip(F["gamma"], phi)) + X
        ok &= sp.simplify(zero - (Tv % 2)) == 0
        for A in (0, 1):
            live = U[A] + sum((g - 1) * ph for g, ph in zip(F["gamma"], phi)) + X
            ok &= sp.simplify(live - ((A + Tv) % 2)) == 0
    return ok

if __name__ == "__main__":
    T = int(sys.argv[1]); gates = json.loads(sys.argv[2]); dr = int(sys.argv[3]); T0 = int(sys.argv[4])
    for F in full(T, [tuple(g) for g in gates], dr, T0) or []:
        print({k: (str(v) if not isinstance(v, list) else [str(x) for x in v]) for k, v in F.items()}, "verify", verify(T, F))

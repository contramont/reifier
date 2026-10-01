# Exact check of PB candidates: per-bit triple (integer gates) + (D) exact solution family,
# pool = A=0 slices (coefficients gamma) + one extra hinge (dir, knot t) [+ affine if AFF].
# For null-dim 0: rational t from the float, exact linear solve. For null-dim >= 1: numeric 2-D
# minimisation over (z, t) first, then rationalise.
import numpy as np, pickle, sys, glob
from fractions import Fraction as Fr
import sympy as sp
from scipy.optimize import least_squares

def load(T, pk):
    d = pickle.load(open(pk, "rb"))
    return d["keys"], d["cands"]

def dsolve(T, gates):
    """exact (D) solution family for integer gates: returns (p0, N) sympy"""
    n1 = T + 1
    rows = []
    for Tv in range(n1):
        row = []
        for a, b, c in gates:
            r0 = max(0, b * Tv + c); r1 = max(0, b * Tv + c + a)
            row += [r1, (r1 - r0) * Tv, r1 - r0]
        rows.append(row)
    M = sp.Matrix(rows)
    rhs = sp.Matrix([(-1) ** Tv for Tv in range(n1)])
    sol, params = M.gauss_jordan_solve(rhs)
    return sol, params

def phis_sym(T, gates, sol):
    n1 = T + 1
    out = []
    for k, (a, b, c) in enumerate(gates):
        e, f = sol[3 * k + 1], sol[3 * k + 2]
        out.append([max(0, b * Tv + c) * (e * Tv + f) for Tv in range(n1)])
    return out

def exact_pool(T, gates, dr, t, aff, sol, params):
    """solve parity = sum gamma_i phi_i + hinge(t)(p T + q) [+ l T + m] exactly; params (D-null) -> symbols"""
    n1 = T + 1
    ph = phis_sym(T, gates, sol)
    g = sp.symbols("g0:3"); p, q, l, m = sp.symbols("p q l m")
    eqs = []
    for Tv in range(n1):
        h = max(0, dr * (Tv - t))
        e = sum(g[i] * ph[i][Tv] for i in range(3)) + h * (p * Tv + q) + ((l * Tv + m) if aff else 0) - (Tv % 2)
        eqs.append(sp.expand(e))
    unk = list(g) + [p, q] + ([l, m] if aff else []) + list(params)
    return sp.solve(eqs, unk, dict=True)

def exact_check(T, gates, dr, T0, aff):
    """symbolic: unknowns gamma, hinge values, (affine), D-null params, knot t (in the interval)"""
    n1 = T + 1
    sol, params = dsolve(T, gates)
    ph = phis_sym(T, gates, sol)
    g = sp.symbols("g0:3"); p, q, l, m, t = sp.symbols("p q l m t")
    eqs = []
    for Tv in range(n1):
        if dr > 0:
            h = (Tv - t) if Tv >= T0 else 0
        else:
            h = (t - Tv) if Tv <= T0 else 0
        e = sum(g[i] * ph[i][Tv] for i in range(3)) + h * (p * Tv + q) + ((l * Tv + m) if aff else 0) - (Tv % 2)
        eqs.append(sp.expand(e))
    unk = list(g) + [p, q] + ([l, m] if aff else []) + list(params) + [t]
    try:
        sols = sp.solve(eqs, unk, dict=True)
    except Exception as ex:
        return ("err", str(ex))
    good = []
    for s in sols:
        tv = s.get(t, t)
        if tv.free_symbols:
            good.append(("free-t", s))
            continue
        if not tv.is_real:
            continue
        tv = sp.nsimplify(tv)
        ok = bool((T0 - 1 < tv <= T0) if dr > 0 else (T0 <= tv < T0 + 1))
        if ok:
            good.append(("ok", s))
    return ("sols", len(sols), good, sol, params)

if __name__ == "__main__":
    T = int(sys.argv[1]); pk = sys.argv[2]; files = sys.argv[3:]
    keys, cands = load(T, pk)
    seen = set(); cand = []
    for fn in files:
        for o in pickle.load(open(fn, "rb")):
            if o[0] != "q1":
                continue
            _, tr, nd, dr, T0, tt, cons, nfree = o
            key = (tr, dr, T0)
            if key in seen:
                continue
            seen.add(key)
            # quick numeric filter for the null-dim-0, full-rank case
            if nd == 0 and nfree == 0:
                if not cons or tt is None:
                    continue
                if not ((T0 - 1 < tt <= T0 + 1e-9) if dr > 0 else (T0 - 1e-9 <= tt < T0 + 1)):
                    continue
            cand.append((tr, nd, dr, T0, tt, cons, nfree))
    print("candidates", len(cand), flush=True)
    import collections
    print(collections.Counter((c[1], c[6]) for c in cand))
    nok = 0
    for i, (tr, nd, dr, T0, tt, cons, nfree) in enumerate(cand):
        gates = [cands[keys[j]] for j in tr]
        r = exact_check(T, gates, dr, T0, False)
        if r[0] == "sols" and r[2]:
            nok += 1
            print("EXACT", gates, dr, T0, r[2][:2], flush=True)
        if i % 200 == 0:
            print("checked", i, "exact ok", nok, flush=True)
    print("DONE exact ok", nok)

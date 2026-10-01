# exact check of T=7 merge candidates: pool = {merged slice pair (gamma_i = gamma_j), third slice, X}
import pickle, sys, time, signal, numpy as np
import sympy as sp
import pbcheck
T = 7
ci, nc = int(sys.argv[1]), int(sys.argv[2])
d = pickle.load(open('/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof2/av/units-per-bit/syn/d_p3_T7_1,-1,2,-2.pkl','rb'))
keys, cands = d['keys'], d['cands']
R = np.array(keys, dtype=float); R0 = R[:, :8]
def norm(r):
    s = r.sum()
    return tuple((r / s).round(9)) if s > 0 else None
cl = pickle.load(open('t7merge_cands.pkl', 'rb'))[ci::nc]
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
nok = 0
for tr, dr, T0 in cl:
    gates = [cands[keys[j]] for j in tr]
    ks = [norm(R0[j]) for j in tr]
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3) if ks[i] is not None and ks[i] == ks[j]]
    sol, params = pbcheck.dsolve(T, gates)
    ph = pbcheck.phis_sym(T, gates, sol)
    g = sp.symbols("g0:3"); p, q, t = sp.symbols("p q t")
    for (i, j) in pairs:
        eqs = []
        for Tv in range(T + 1):
            hh = ((Tv - t) if Tv >= T0 else 0) if dr > 0 else ((t - Tv) if Tv <= T0 else 0)
            e = sum(g[k] * ph[k][Tv] for k in range(3)) + hh * (p * Tv + q) - (Tv % 2)
            eqs.append(sp.expand(e.subs(g[j], g[i])))
        signal.alarm(30)
        try:
            sols = sp.solve(eqs, [g[0], g[1], g[2], p, q, t] + list(params), dict=True)
        except TO:
            print("TIMEOUT", gates, flush=True); continue
        finally:
            signal.alarm(0)
        for s in sols:
            tv = s.get(t, t)
            if tv.free_symbols or not tv.is_real:
                if tv.free_symbols:
                    print("FREE-T", gates, dr, T0, (i, j), s, flush=True)
                continue
            ok = bool((T0 - 1 < tv <= T0) if dr > 0 else (T0 <= tv < T0 + 1))
            if ok:
                nok += 1
                print("EXACT", gates, dr, T0, (i, j), {str(a): str(b) for a, b in s.items()}, flush=True)
print("DONE ok", nok)

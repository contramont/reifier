# rank exact PB-q1 forms by estimated nonzeros per 4-live-bit column (Tn T-bits) and coefficient size
import re, glob, sys, json, pickle
import sympy as sp
import pbfull
T = 8; Tn = int(sys.argv[1]) if len(sys.argv) > 1 else 7
lines = set()
for fn in glob.glob("logs/pbx_T8_*.log"):
    for l in open(fn):
        if l.startswith("EXACT"):
            m = re.match(r"EXACT (\[.*?\]) (-?\d+) (\d+) ", l)
            lines.add((m.group(1), int(m.group(2)), int(m.group(3))))
print("forms", len(lines), flush=True)
def nz(x):
    return 0 if x == 0 else 1
res = []
for gs, dr, T0 in sorted(lines):
    gates = [tuple(g) for g in json.loads(gs.replace("(", "[").replace(")", "]"))]
    try:
        forms = pbfull.full(T, gates, dr, T0) or []
    except Exception as e:
        continue
    for F in forms:
        if F["free"]:
            # free-t or free params: skip here (listed separately)
            res.append((10**9, 0, gs, dr, T0, "free", F["free"]))
            continue
        if not pbfull.verify(T, F):
            continue
        rational = all(sp.sympify(v).is_rational for v in list(F["vals"]) + list(F["gamma"]) + [F["t"], F["p"], F["q"]])
        vals = [sp.nsimplify(v) for v in F["vals"]]; gam = F["gamma"]
        per = 0
        for k, (a, b, c) in enumerate(gates):
            d, e, f = vals[3 * k:3 * k + 3]
            per += nz(a) + Tn * nz(b) + nz(c) + nz(d) + Tn * nz(e) + nz(f)
        wo_live = 3 + sum(1 for g in gam if g != 1) + 1
        sl = 0
        for k, (a, b, c) in enumerate(gates):
            d, e, f = vals[3 * k:3 * k + 3]
            if gam[k] != 0 or gam[k] != 1:
                sl += Tn * nz(b) + nz(c) + Tn * nz(e) + nz(f)
        # X: gate on T (+ bias), value p T + q
        t = F["t"]
        xg = Tn + (0 if t == 0 else 1)
        xv = Tn * nz(F["p"]) + nz(F["q"])
        wo_zero = sum(1 for g in gam if g != 0) + 1
        tot = 4 * (per + wo_live) + sl + xg + xv + wo_zero
        mx = max(abs(float(v)) for v in list(vals) + list(gam) + [F["p"], F["q"]])
        res.append((tot, mx, gs, dr, T0, "rational" if rational else "irr",
                    dict(vals=[str(v) for v in vals], gamma=[str(g) for g in gam], t=str(F["t"]), p=str(F["p"]), q=str(F["q"]))))
res.sort(key=lambda r: (r[0], r[1]))
pickle.dump(res, open("pbrank.pkl", "wb"))
for r in res[:25]:
    print(r)

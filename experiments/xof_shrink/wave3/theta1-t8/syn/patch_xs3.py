p = 'xs3.py'
src = open(p).read()
old = '''_LZ5 = {}
'''
new = '''# theta1-t8 (wave 3): round-1 theta with a column POOL made of the zero lane's own units.
# Per column: the zero lane's units are the A = 0 slices z_i(T) = max(0, b_i T + c_i)(e_i T + f_i)
# of the three per-bit units, plus ONE extra hinge X(T); every live bit reads its three per-bit
# units u_i(A, T) = max(0, a_i A + b_i T + c_i)(d_i A + e_i T + f_i) plus (gamma_i - 1) z_i + X,
# and the zero lane is sum_i gamma_i z_i + X. Exact on {0,1} x [0, 8] (sympy, syn/pbfull.py), so it
# covers the T = 8 columns (x = 1) and every T <= 7 column: per column 3 per live bit + 4
# (T = 8: 20 -> 16 units; T <= 7: TH1S's 3 per live bit + 5 -> + 4).
# Found by syn/pb.py (all 4.26M (D)-feasible triples of wave 1, knot of X solved exactly per
# interval) + syn/pbx.py (exact sympy solve of the bilinear system).
TH1P = {"on": False}
TH1P_FORM = {
    "bit": [((8, -1, 2), (-29 / 23, -2, 14)), ((15, 2, -12), (-118 / 23, 1, -1)), ((2, -2, 7), (-8, 2, -4))],
    "gamma": (1 / 2, 1 / 2, 1 / 2),
    # X(T) = max(0, gT T + gc)(vT T + vc), gate scaled so the nearest lattice value is 1
    "extra": ((2, -9), (-1, 6)),
    "tmax": 8,
}
_LZ5 = {}
'''
assert old in src
src = src.replace(old, new, 1)

old2 = '''    tsigs = sorted((s for s, c in tset.items() if c), key=lambda s: s.uid)
    if not 2 <= len(tsigs) <= 7:
        return None
'''
new2 = '''    tsigs = sorted((s for s, c in tset.items() if c), key=lambda s: s.uid)
    if TH1P["on"]:
        return _theta1_pool(own, tsigs, flip, tset, memo)
    if not 2 <= len(tsigs) <= 7:
        return None
'''
assert old2 in src
src = src.replace(old2, new2, 1)

old3 = '''def neg(lit):
'''
new3 = '''def _theta1_pool(own, tsigs, flip, tset, memo):
    """theta1-t8: pool form (TH1P_FORM), exact for T = len(tsigs) in [0, tmax]"""
    if not 2 <= len(tsigs) <= TH1P_FORM["tmax"]:
        return None
    if isinstance(own, int):
        asig = None
        flip ^= own
    else:
        asig = own[0]
        flip ^= int(own[1])
        if asig in tset:
            return None
    key = ("th1p", asig, frozenset(tsigs))
    if key not in memo:
        specs = []
        gam = TH1P_FORM["gamma"]
        for i, ((ga, gb, gc), (va, vb, vc)) in enumerate(TH1P_FORM["bit"]):
            if asig is not None:  # the live bit's own unit
                g = {s_: float(gb) for s_ in tsigs}
                g[asig] = float(ga)
                v = {s_: float(vb) for s_ in tsigs} if vb else {}
                if va:
                    v[asig] = float(va)
                specs.append((g, float(gc), v, float(vc)))
            # the column's pool slice z_i (shared with the zero lane: same gate, proportional value)
            co = gam[i] - 1 if asig is not None else gam[i]
            if co:
                g = {s_: float(gb) for s_ in tsigs}
                v = {s_: float(vb) * co for s_ in tsigs} if vb else {}
                specs.append((g, float(gc), v, float(vc) * co))
        (hg, hc), (he, hf) = TH1P_FORM["extra"]
        specs.append(({s_: float(hg) for s_ in tsigs}, float(hc), {s_: float(he) for s_ in tsigs} if he else {}, float(hf)))
        memo[key] = node(specs)
    return (memo[key], bool(flip))


def neg(lit):
'''
assert old3 in src
src = src.replace(old3, new3, 1)

old4 = '''def with_slast(variant):
'''
new4 = '''def with_th1p(variant):
    """theta1-t8: pool form for round-1 theta (all 2 <= T <= 8 columns); needs the TH1S call path"""
    def v(k, depth):
        TH1S["on"] = True
        TH1P["on"] = True
        return variant(k, depth)
    return v


def with_slast(variant):
'''
assert old4 in src
src = src.replace(old4, new4, 1)
open(p, 'w').write(src)
print("patched")

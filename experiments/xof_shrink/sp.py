"""sparse-focus: 1-round Keccak XOF with fewer nonzero parameters (wave 2).

Same builder conventions as xs3 (one traced call per SwiGLU layer, literals (Bit, neg),
constants folded, last-round DCE, digests carried in pairs). New:

  g-fold: chi(a, b, c) = h(g), g = 2a - b + c, h(g) = max(0, g) (3 - g) / 2 (exact on
          g in {-1..3}). The layer BEFORE chi emits g as ONE feature (its wo sums the
          three theta units, weights 2, -1, 1), so a chi unit reads one feature:
          gate 1 + value 2 nonzeros instead of 3 + 4. Costs 2 more wo entries per theta
          unit (each theta bit is a, b and c of three chi bits).
  D-sep:  a split theta (X, Y) emits D = parity(P) of each column pair as ONE feature
          (5 wo entries per pair) instead of adding D into the 5 E = a + D features
          (25 wo entries); Y computes t = s xor D with one unit on (s, D).
Both are exact: every unit is exact on its lattice, flags (negations, iota) are folded
into constants of the readers.
"""

from math import ceil

import reifier.examples.keccak as K
from reifier.neurons.core import Bit, Unit, glu
from reifier.utils.format import Bits

from xs import COPY, CHI, keccak_maps, initial_state, xor_units
import xs3
from xs3 import spec, spec_on, node, parity_units, pack_scale, digit_fns, DECODERS, neg

G = glu
SCALE = 16.0  # counts emitted as count / SCALE
PS = 16.0  # pair sums P / PS
GS = 4.0  # g features emitted as g / GS
XOR2 = Unit((1, 1), 0, (-1, -1), 2)  # xor of two bits: max(0, a + b)(2 - a - b)
OPTS = {"gfold1": True, "gfold2": True, "pre": False,
        "e1": False, "col1": False, "lazy2": False, "x2e": False, "mp": False, "ebit": True, "m3": False}


def scaled(sp, c):
    g, gb, v, vb = sp
    return g, gb, {q: w * c for q, w in v.items()}, vb * c


def build(k: K.Keccak, depth: int, opts=None):
    o = dict(OPTS)
    o.update(opts or {})
    assert depth >= 1  # steps avenue (wave 4): any number of XOF steps (middle rounds repeat round 2)
    w = k.w
    (rc,) = k.get_round_constants()
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[: k.d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]

    EBIT = set()  # E features that take only the values 0, 1

    def flag(x, y, z):  # iota
        return int(x == 0 and y == 0 and rc[z] == "1")

    def chi_needs(out_pos):
        need = {}
        for (x, y, z) in out_pos:
            for dx in range(3):
                need[rp[(x + dx) % 5][y][z]] = None
        return list(need)

    def chi_lits(th, x, y, z):
        return [th[rp[(x + dx) % 5][y][z]] for dx in range(3)]

    def pair_of(p):
        x, _, z = p
        return [((x + 4) % 5, y2, z) for y2 in range(5)] + [((x + 1) % 5, y2, (z + 1) % w) for y2 in range(5)]

    # ---------------- carries (as xs3, pairs p = d0 + 2 d1, feature p / 4) ----------------
    def carry_layer(carry):
        out, lazy4, lazy = [], [], []
        for kind, sig, f in carry:
            if kind == "bit":
                out.append((kind, G([sig], [COPY]), f))
            elif kind == "pair":
                out.append((kind, G([sig], [Unit((4,), 0, (0,), 0.25)], numeric=True), f))
            elif kind == "pk":  # m bits packed binary, feature p / sc (combined-1)
                sc = pack_scale(2, len(f))
                out.append((kind, G([sig], [Unit((sc,), 0, (0,), 1 / sc)], numeric=True), f))
            elif kind == "lazy4":
                lazy4.append((sig, f))
            elif kind == "lazy":
                lazy.append((sig, f))
            else:
                raise ValueError(kind)
        par = Unit((2,), 0, (-2,), 2)  # max(0, o')(2 - o')
        for i in range(0, len(lazy) - 1, 2):
            (s0, f0), (s1, f1) = lazy[i], lazy[i + 1]
            out.append(("pair", node([spec_on(s0, par, 0.25), spec_on(s1, par, 0.5)], numeric=True), (f0, f1)))
        if len(lazy) % 2:
            s0, f0 = lazy[-1]
            out.append(("bit", node([spec_on(s0, par)]), f0))
        par4 = [Unit((2,), 0, (-2,), 2), Unit((2,), -2, (0,), 4)]  # parity of o in [0,4] on o/2
        for i in range(0, len(lazy4) - 1, 2):
            (s0, f0), (s1, f1) = lazy4[i], lazy4[i + 1]
            out.append(("pair", node([spec_on(s0, u, 0.25) for u in par4] + [spec_on(s1, u, 0.5) for u in par4],
                                     numeric=True), (f0, f1)))
        if len(lazy4) % 2:
            s0, f0 = lazy4[-1]
            out.append(("bit", node([spec_on(s0, u) for u in par4]), f0))
        return out

    def carry_new(specs_flags):
        items = []
        if o["m3"]:  # 3 bits per feature, binary (combined-1): less dense, more sparse
            for i in range(0, len(specs_flags), 3):
                chunk = specs_flags[i:i + 3]
                sc = pack_scale(2, len(chunk))
                specs = [scaled(s_, 2 ** j / sc) for j, (s_, _) in enumerate(chunk)]
                items.append(("pk", node(specs, numeric=True), tuple(f for _, f in chunk)))
            return items
        for i in range(0, len(specs_flags) - 1, 2):
            (s0, f0), (s1, f1) = specs_flags[i], specs_flags[i + 1]
            items.append(("pair", node([scaled(s0, 0.25), scaled(s1, 0.5)], numeric=True), (f0, f1)))
        if len(specs_flags) % 2:
            s, f = specs_flags[-1]
            items.append(("bit", node([s]), f))
        return items

    def decode(carry):
        bits = []

        def out(sig_units, f):
            specs = [spec_on(sig, u, 1 - 2 * f) for sig, u in sig_units]
            if f:
                specs.append(({}, 1, {}, 1))
            return node(specs)

        for kind, sig, f in carry:
            if kind == "bit":
                bits.append(out([(sig, COPY)], f))
            elif kind == "pk":  # binary digits of p = sc * feature, exact DP decoders
                m = len(f)
                sc = pack_scale(2, m)
                if m not in DECODERS:
                    DECODERS[m] = digit_fns(2, m)
                for (const, units), fi in zip(DECODERS[m], f):
                    s_ = 1 - 2 * fi
                    specs = [({sig: sc}, float(-kk), {sig: float(b) * sc * s_} if b else {},
                              float(a) * s_) for kk, a, b in units]
                    cst = float(const) * s_ + fi
                    if cst:
                        specs.append(({}, 1.0, {}, cst))
                    bits.append(node(specs))
            else:
                lo = out([(sig, Unit((4,), 0, (0,), 1)), (sig, Unit((4,), -1, (4,), -4))], f[0])
                hi = out([(sig, Unit((4,), -1, (-2,), 2))], f[1])
                bits += [lo, hi]
        return bits

    # ---------------- helpers ----------------
    def split_lits(lits):
        """flip and the distinct live signals (mod 2) of an xor of literals"""
        flip, cnt = 0, {}
        for lit in lits:
            if isinstance(lit, int):
                flip ^= lit
            else:
                flip ^= int(lit[1])
                cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
        sigs = sorted((s for s, c in cnt.items() if c), key=lambda s: s.uid)
        return flip, sigs

    def t_spec(lits):
        """theta bit t = xor of <= 2 live literals: (unit spec at coef 1, flag) or int"""
        flip, sigs = split_lits(lits)
        if not sigs:
            return flip
        if len(sigs) == 1:
            return spec_on(sigs[0], COPY), flip
        assert len(sigs) == 2
        return ({sigs[0]: 1, sigs[1]: 1}, 0, {sigs[0]: -1, sigs[1]: -1}, 2), flip

    def g_nodes(tsp, chi_pos):
        """g = 2a - b + c of each chi output, as one numeric feature g / GS plus a constant"""
        gf = {}
        for q in chi_pos:
            x, y, z = q
            specs, const = [], 0
            for dx, wgt in zip(range(3), (2, -1, 1)):
                t = tsp[rp[(x + dx) % 5][y][z]]
                if isinstance(t, int):
                    const += wgt * t
                    continue
                sp, f = t
                const += wgt * f
                specs.append(scaled(sp, wgt * (1 - 2 * f) / GS))
            gf[q] = (node(specs, numeric=True), const)
        return gf

    def chi_g(gq, coef):
        """h(g) = max(0, g)(3 - g)/2 on g = GS * feature + const"""
        gsig, c = gq
        return ({gsig: GS}, c, {gsig: -0.5 * GS * coef}, (1.5 - 0.5 * c) * coef)

    # ---------------- round 1: X1 (s copies, D of message bits), Y1 ----------------
    def x1(a, carry):
        s, dl = {}, {}
        copies = {}
        for p in all_pos:
            lit = a[p[0]][p[1]][p[2]]
            if isinstance(lit, int):
                s[p] = lit
            else:
                if lit[0] not in copies:
                    copies[lit[0]] = G([lit[0]], [COPY])
                s[p] = (copies[lit[0]], lit[1])
        memo = {}
        for x in range(5):
            for z in range(w):
                flip, sigs = split_lits([a[q[0]][q[1]][q[2]] for q in pair_of((x, 0, z))])
                if not sigs:
                    dl[(x, z)] = flip
                    continue
                key = tuple(sigs)
                if key not in memo:
                    if len(sigs) == 1:
                        us = [COPY]
                    elif o["mp"]:  # min-parity (knots between integers): exact raw bits only
                        xs3.MINPAR["on"] = True
                        us = xs3.par_units(len(sigs))
                        xs3.MINPAR["on"] = False
                    else:
                        us = xor_units(len(sigs))
                    memo[key] = G(list(sigs), us)
                dl[(x, z)] = (memo[key], bool(flip))
        return s, dl, carry

    def x1_pre(a, carry):
        """depth +1: a linear layer first: P of each column pair (one pass unit reading the
        live message bits) and copies of s; then X1 reads P (one feature)"""
        s1, sums = {}, {}
        copies = {}
        for p in all_pos:
            lit = a[p[0]][p[1]][p[2]]
            if isinstance(lit, int):
                s1[p] = lit
            else:
                if lit[0] not in copies:
                    copies[lit[0]] = G([lit[0]], [COPY])
                s1[p] = (copies[lit[0]], lit[1])
        memo = {}
        for x in range(5):
            for z in range(w):
                lits = [a[q[0]][q[1]][q[2]] for q in pair_of((x, 0, z))]
                # the count of the live literals: sum of (b or 1 - b); flip = parity offset
                flip, sigs = split_lits(lits)
                if not sigs:
                    sums[(x, z)] = flip
                    continue
                key = tuple(sigs)
                if key not in memo:
                    memo[key] = node([({}, 1, {sg: 1.0 / PS for sg in sigs}, 0.0)], numeric=True)
                sums[(x, z)] = (memo[key], flip, len(sigs))
        # layer 2: copies of s again, D = parity(P)
        s = {}
        copies2 = {}
        for p, lit in s1.items():
            if isinstance(lit, int):
                s[p] = lit
            else:
                if lit[0] not in copies2:
                    copies2[lit[0]] = G([lit[0]], [COPY])
                s[p] = (copies2[lit[0]], lit[1])
        dl = {}
        memo2 = {}
        for key, v in sums.items():
            if isinstance(v, int):
                dl[key] = v
                continue
            psig, f, n = v
            if psig not in memo2:
                memo2[psig] = G([psig], [Unit((PS,), 0, (0,), 1)] if n == 1 else parity_units(n, PS))
            dl[key] = (memo2[psig], bool(f))
        return s, dl, carry

    def x1_col(a, carry):
        """depth +1, round 1 from column parities: layer A computes each column's parity c
        (glu_xor on its 3-4 live message bits, each bit read by ONE parity instead of two)
        and copies s; layer B computes D = c(x-1, z) xor c(x+1, z+1) with ONE unit per pair,
        cheap enough to add into E = s + D (5 wo entries), and copies s into E.
        Returns E literals (E/2 features) for y_e_layer."""
        copies, s1 = {}, {}
        for p in all_pos:
            lit = a[p[0]][p[1]][p[2]]
            if isinstance(lit, int):
                s1[p] = lit
            else:
                if lit[0] not in copies:
                    copies[lit[0]] = G([lit[0]], [COPY])
                s1[p] = (copies[lit[0]], lit[1])
        col, memo = {}, {}
        for x in range(5):
            for z in range(w):
                flip, sigs = split_lits([a[x][y2][z] for y2 in range(5)])
                if not sigs:
                    col[(x, z)] = flip
                    continue
                key = tuple(sigs)
                if key not in memo:
                    memo[key] = G(list(sigs), [COPY] if len(sigs) == 1 else xor_units(len(sigs)))
                col[(x, z)] = (memo[key], bool(flip))
        # layer B: E = s + D, D = c1 xor c2 (one unit per pair, shared by its 5 positions)
        e, memo_e = {}, {}
        for p in all_pos:
            x, _, z = p
            flip, dsigs = split_lits([col[((x + 4) % 5, z)], col[((x + 1) % 5, (z + 1) % w)]])
            sl = s1[p]
            if isinstance(sl, int):
                flip ^= sl
                aterm = None
            else:
                flip ^= int(sl[1])
                aterm = sl[0]
            key = (aterm, tuple(dsigs))
            if key not in memo_e:
                specs = []
                if aterm is not None:
                    specs.append(spec_on(aterm, COPY, 0.5))
                if len(dsigs) == 2:
                    specs.append(({dsigs[0]: 1, dsigs[1]: 1}, 0, {dsigs[0]: -0.5, dsigs[1]: -0.5}, 1.0))
                elif len(dsigs) == 1:
                    specs.append(spec_on(dsigs[0], COPY, 0.5))
                memo_e[key] = node(specs, numeric=True)
                if aterm is None and o["ebit"]:
                    EBIT.add(memo_e[key])  # E = D only, in {0, 1}
            e[p] = (memo_e[key], flip)
        return e, carry

    def y_layer(s, dl, chi_pos, carry, gfold=True):
        """t = s xor D per theta position; emits g features for chi_pos (gfold) or t bits"""
        tsp = {}
        for p in all_pos:
            tsp[p] = t_spec([s[p], dl[(p[0], p[2])]])
        if gfold:
            return g_nodes(tsp, chi_pos), carry_layer(carry)
        th = {}
        for p, t in tsp.items():
            th[p] = t if isinstance(t, int) else (node([t[0]]), bool(t[1]))
        return th, carry_layer(carry)

    def e1_layer(a, carry):
        """X1 with D added into E = s + D (xs3's split X1, glu_xor units), then Y reads E"""
        memo_e, e = {}, {}
        for p in all_pos:
            flip, cnt = 0, {}
            for lit in [a[q[0]][q[1]][q[2]] for q in pair_of(p)]:
                if isinstance(lit, int):
                    flip ^= lit
                else:
                    flip ^= int(lit[1])
                    cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
            dsigs = tuple(sorted((sg for sg, c in cnt.items() if c), key=lambda sg: sg.uid))
            alit = a[p[0]][p[1]][p[2]]
            if isinstance(alit, int):
                flip ^= alit
                aterm = None
            else:
                flip ^= int(alit[1])
                aterm = alit[0]
            key = (aterm, dsigs)
            if key not in memo_e:
                specs = []
                if aterm is not None:
                    specs.append(spec_on(aterm, COPY, 0.5))
                n = len(dsigs)
                us = [COPY] if n == 1 else xor_units(n)
                for u in us:
                    specs.append(({sg: u.weights[i] for i, sg in enumerate(dsigs) if u.weights[i]}, u.bias,
                                  {sg: u.value_weights[i] * 0.5 for i, sg in enumerate(dsigs) if u.value_weights[i]},
                                  u.value_bias * 0.5))
                memo_e[key] = node(specs, numeric=True)
                if aterm is None and o["ebit"]:
                    EBIT.add(memo_e[key])  # E = D only, in {0, 1}
            e[p] = (memo_e[key], flip)
        return e, carry

    def y_e_layer(e, chi_pos, carry):
        """t = [E == 1] on E/2, g-folded; E in {0, 1} (no s term) is read by a copy unit"""
        par = Unit((2,), 0, (-2,), 2)
        cp = Unit((2,), 0, (0,), 1)
        tsp = {p: (spec_on(sig, cp if sig in EBIT else par), f) for p, (sig, f) in e.items()}
        return g_nodes(tsp, chi_pos), carry_layer(carry)

    # ---------------- chi layers on g ----------------
    def chi_to_split(gf, carry):
        bits = {p: (node([chi_g(gf[p], 1.0)]), flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                sums[(x, z)] = (node([chi_g(gf[q], 1 / PS) for q in qs], numeric=True), f, len(qs))
        return bits, sums, carry_layer(carry)

    def chi_to_split_nog(th, carry):
        chi = {p: chi_lits(th, *p) for p in all_pos}
        bits = {p: (node([spec(chi[p], CHI)]), flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                sums[(x, z)] = (node([spec(chi[q], CHI, 1 / PS) for q in qs], numeric=True), f, len(qs))
        return bits, sums, carry_layer(carry)

    def x_sep(bits, sums, carry):
        s = {p: (G([bits[p][0]], [COPY]), bits[p][1]) for p in all_pos}
        dl = {}
        for key, (psig, pf, n) in sums.items():
            dl[key] = (G([psig], parity_units(n, PS)), bool(pf))
        new = [(spec_on(bits[p][0], COPY), bits[p][1]) for p in digest_pos]
        return s, dl, carry_layer(carry) + carry_new(new)

    def x_efold(bits, sums, carry):
        """xs3's split X: E = a + D (E/2), D's 5 units added into the 5 E of the pair"""
        e = {}
        for p in all_pos:
            (asig, af), (psig, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = [spec_on(asig, COPY, 0.5)] + [spec_on(psig, u, 0.5) for u in parity_units(n, PS)]
            e[p] = (node(specs, numeric=True), af ^ pf)
        new = [(spec_on(bits[p][0], COPY), bits[p][1]) for p in digest_pos]
        return e, carry_layer(carry) + carry_new(new)

    def x_exact(bits, sums, carry):
        """E = a + D with flags folded (E/2), for a lazy chi that reads E (xs3)"""
        e = {}
        for p in all_pos:
            (asig, af), (psig, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = [spec_on(asig, COPY, 0.5 * (1 - 2 * af))]
            specs += [spec_on(psig, u, 0.5 * (1 - 2 * pf)) for u in parity_units(n, PS)]
            if af or pf:
                specs.append(({}, 1, {}, 0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [(spec_on(bits[p][0], COPY), bits[p][1]) for p in digest_pos]
        return e, carry_layer(carry) + carry_new(new)

    XOR_E = Unit((2,), 0, (-2,), 2)  # on E/2: [E == 1]
    Q_E = [Unit((-8, -4), 3, (0, 2), 0), Unit((8, -4), -5, (0, 2), 0)]  # [Eb != 1][Ec == 1]

    def lazy_chi3(e, next_pos, carry):
        """xs3.lazy_chi (theta-structure): o' = [Ea == 1] + [Eb != 1][Ec == 1] in {0,1,2}"""
        def terms(q, coef):
            x, y, z = q
            ea, eb, ec = (e[rp[(x + dx) % 5][y][z]] for dx in range(3))
            specs = [spec_on(ea, XOR_E, coef)]
            for u in Q_E:
                specs.append(({eb: u.weights[0], ec: u.weights[1]}, u.bias, {ec: u.value_weights[1] * coef}, 0.0))
            return specs
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            specs = [sp for q in qs for sp in terms(q, 1 / SCALE)]
            packed[p] = (node(specs, numeric=True), f, 2 * len(qs))
        new = [("lazy", node(terms(p, 0.5), numeric=True), flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + new

    def lazy_chi4c(e, next_pos, carry):
        """xs3.lazy_chi4c (unit-synthesis): o = Ea + 2Eb + max(0, Eb + Ec)(1 - Eb), column
        linear parts reduced by r(c); counts in [0, 32]"""
        def q_prod(q, coef):
            x, y, z = q
            eb, ec = (e[rp[(x + dx) % 5][y][z]] for dx in (1, 2))
            return ({eb: 2.0, ec: 2.0}, 0, {eb: -2.0 * coef}, coef), {eb: 4.0 * coef}

        def own(q, coef):
            x, y, z = q
            ea = e[rp[x][y][z]]
            prod, pas = q_prod(q, coef)
            pas = dict(pas)
            pas[ea] = pas.get(ea, 0) + 2.0 * coef
            return [prod], pas

        def column(xc, zc, coef):
            feats = [e[rp[xc][y2][zc]] for y2 in range(5)]
            g = {}
            for f_ in feats:
                g[f_] = g.get(f_, 0.0) + 2.0
            specs = [({s_: -3.0 * w_ for s_, w_ in g.items()}, 9, {s_: -w_ / 3.0 * coef for s_, w_ in g.items()}, -2.0 / 3.0 * coef),
                     (dict(g), -7, {}, 2.0 * coef)]
            pas = {s_: -w_ * coef for s_, w_ in g.items()}
            const = 2.0 * coef
            for y2 in range(5):
                prod, pq = q_prod((xc, y2, zc), coef)
                specs.append(prod)
                for s_, w_ in pq.items():
                    pas[s_] = pas.get(s_, 0) + w_
            return specs, pas, const

        packed = {}
        for p in next_pos:
            x, _, z = p
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            specs, pas = own(p, 1 / SCALE)
            const = 10.0 / SCALE
            for (xc, zc) in (((x + 4) % 5, z), ((x + 1) % 5, (z + 1) % w)):
                sp, pq, cst = column(xc, zc, 1 / SCALE)
                specs = specs + sp
                const += cst
                for s_, w_ in pq.items():
                    pas[s_] = pas.get(s_, 0) + w_
            pas = {s_: w_ for s_, w_ in pas.items() if abs(w_) > 1e-12}
            specs.append(({}, 1, pas, const))
            packed[p] = (node(specs, numeric=True), f, 32)
        new = []
        for p in digest_pos:
            sp, pas = own(p, 0.5)
            new.append(("lazy4", node(sp + [({}, 1, pas, 0.0)], numeric=True), flag(*p)))
        return packed, carry_layer(carry) + new

    def chi_to_direct(gf, next_pos, carry):
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            packed[p] = (node([chi_g(gf[q], 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        new = [(chi_g(gf[p], 1.0), flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + carry_new(new)

    def chi_to_direct_nog(th, next_pos, carry):
        chi = {p: chi_lits(th, *p) for p in all_pos}
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            packed[p] = (node([spec(chi[q], CHI, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        new = [(spec(chi[p], CHI), flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + carry_new(new)

    def theta_direct(packed, carry):
        th = {p: (G([sig], parity_units(n, SCALE)), bool(f)) for p, (sig, f, n) in packed.items()}
        return th, carry_layer(carry)

    def chi_last(th, carry):
        bits = decode(carry)
        for p in digest_pos:
            lits = chi_lits(th, *p)
            if flag(*p):
                lits = [neg(lits[0])] + lits[1:]
            bits.append(node([spec(lits, CHI)]))
        return bits

    def chi_last_g(gf, carry):
        """S = 1: the digest's chi bits from g features, h(g) xor iota"""
        bits = decode(carry)
        for p in digest_pos:
            f = flag(*p)
            specs = [chi_g(gf[p], 1 - 2 * f)]
            if f:
                specs.append(({}, 1, {}, 1.0))
            bits.append(node(specs))
        return bits

    def xof_fn(msg: list) -> list:
        """round 1 | (chi, X, Y) per middle round 2..S-1 | chi to counts, theta, chi of round S
        (S = 3: the original layout; lazy2 turns the last middle round into X + lazy chi)"""
        lanes = initial_state(k, msg)
        carry: list = []
        S = depth
        chi_pos1 = all_pos if S > 1 else digest_pos
        # round 1
        if o["e1"] or o["col1"]:
            e, carry = (x1_col if o["col1"] else e1_layer)(lanes, carry)
            gf, carry = y_e_layer(e, chi_pos1, carry)
            g = True
        else:
            s, dl, carry = (x1_pre if o["pre"] else x1)(lanes, carry)
            g = o["gfold1"]
            gf, carry = y_layer(s, dl, chi_pos1, carry, gfold=g)
        if S == 1:
            return chi_last_g(gf, carry) if g else chi_last(gf, carry)
        for step in range(2, S):  # middle round `step` (1-based), after the chi of step - 1
            if g:
                bits, sums, carry = chi_to_split(gf, carry)
            else:
                bits, sums, carry = chi_to_split_nog(gf, carry)
            if o["lazy2"] and step == S - 1:  # depth -1: X with exact E, one-unit lazy chi reading E
                e, carry = x_exact(bits, sums, carry)
                packed, carry = (lazy_chi3 if o["lazy2"] == "l3" else lazy_chi4c)(e, chi_needs(digest_pos), carry)
                th, carry = theta_direct(packed, carry)
                return chi_last(th, carry)
            if o["x2e"]:
                e, carry = x_efold(bits, sums, carry)
                gf, carry = y_e_layer(e, all_pos, carry)
                g = True
            else:
                s, dl, carry = x_sep(bits, sums, carry)
                g = o["gfold2"]
                gf, carry = y_layer(s, dl, all_pos, carry, gfold=g)
        th_pos = chi_needs(digest_pos)
        if g:
            packed, carry = chi_to_direct(gf, th_pos, carry)
        else:
            packed, carry = chi_to_direct_nog(gf, th_pos, carry)
        # last round
        th, carry = theta_direct(packed, carry)
        return chi_last(th, carry)

    return xof_fn


def make(**opts):
    def variant(k, depth):
        return build(k, depth, opts), {"msg": Bits("0" * k.msg_len).bitlist}
    return variant


base = make()  # g-fold both chi layers, D-sep in both X layers
nog = make(gfold1=False, gfold2=False)
nog1 = make(gfold1=False)
nog2 = make(gfold2=False)
e1 = make(e1=True, ebit=False)  # round 1 with E = s + D (xs3 X1) + g-fold
pre = make(pre=True)  # depth 9: linear pre-layer for round 1
col1 = make(col1=True, ebit=False)  # depth 9: round 1 from column parities, E-fold of one-unit D
lazy2 = make(lazy2=True)  # depth 7: round 1 as base, lazy4c round 2
col1_lazy2 = make(col1=True, lazy2=True, ebit=False)  # depth 8: col1 round 1, lazy4c round 2
lazy2_l3 = make(lazy2="l3")  # depth 7 with the 3-unit lazy chi (counts range 22)
x2e = make(x2e=True)  # depth 8, E-fold in X2 (less dense, more sparse)
col1_x2e = make(col1=True, x2e=True, ebit=False)  # depth 9, E-fold in X2
x2e_mp = make(x2e=True, mp=True)  # depth 8, E-fold in X2, min-parity D in X1 (dense)
base_mp = make(mp=True)
col1b = make(col1=True)  # col1 + copy units for E in {0, 1}
col1b_x2e = make(col1=True, x2e=True)
lazy2_m3_mp = make(lazy2=True, m3=True, mp=True)  # depth 7, dense-oriented
x2e_m3_mp = make(x2e=True, m3=True, mp=True)  # depth 8, dense-oriented

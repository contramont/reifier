"""XOF-structured 1-round Keccak, general layout (avenue xof-structure), with the
round-structure options of wave 2 (xpairs, ydedup, gfeat; see build).

Every SwiGLU layer is one traced call, so each gated unit sits exactly one level above
its inputs and the compiler adds no copies. Per XOF step, theta is laid out as
  "direct": one layer, theta bit = parity(s) with s = a + (column sums), 6 units reading
            the single packed feature s that the previous chi layer emits for free;
  "split":  two layers (for middle steps): X emits E = a + D, a copy unit of a plus the
            5 units of D = parity(P) on the free column-pair sum P, which all 5 bits of a
            column share; Y emits theta = [E == 1] with one unit. 3 units per bit, not 6.
then chi (+iota) as one unit per bit. Keccak/XOF specifics:
  - suffix and capacity bits are Python constants folded into units, equal xors are built
    once (two zero lanes of a column give the same theta bit), negations stay flags;
  - the last step computes only its digest's chi bits and the 320 theta bits they read
    (5 diagonal lanes via rho/pi), and the last chi layer emits exactly the outputs;
  - the column sums the next theta needs are free linear features of the chi layer, so a
    chi layer before the last theta emits only 320 counts, not 1600 state bits;
  - digests of earlier steps are carried two bits per feature (p = a + 2b, one unit per
    layer) and decoded in the last layer with two units per pair; iota flips of digest
    bits are applied there with one shared constant unit.
A literal is an int 0/1 (constant) or (Bit, neg) meaning neg xor Bit.
"""

from math import ceil

import reifier.examples.keccak as K
from reifier.neurons.core import Bit, Unit, glu
from reifier.utils.format import Bits

from xs import CHI, COPY, fold, keccak_maps, initial_state, theta_lits, xor_units
from decode import check as synth_decoder

DECODERS: dict = {}


def pack_scale(base: int, n: int) -> float:
    """packed features are p / 2^e with p <= base^n - 1: dyadic (exact) and < 1"""
    return float(2 ** (base ** n - 1).bit_length())


def digit_fns(base: int, n: int):
    """exact one-layer decoders of the n binary digits of p (decode.py, xof-structure)"""
    N = base ** n - 1
    return [synth_decoder([(p >> i) & 1 for p in range(N + 1)]) for i in range(n)]
from shared_theta import PER_BIT, shared_units

SCALE = 16  # packed counts are emitted as count / SCALE (keeps RMSNorm's scale >= 1)
G = glu


def spec(lits, unit, coef=1.0):
    """A unit on literals as (gate dict, gate bias, value dict, value bias), value * coef"""
    sigs, (u,) = fold(lits, [unit])
    g = {s: w for s, w in zip(sigs, u.weights) if w}
    v = {s: w * coef for s, w in zip(sigs, u.value_weights) if w}
    return g, u.bias, v, u.value_bias * coef


def spec_on(sig, unit, coef=1.0):
    """A unit on one signal (unit has one weight)"""
    return ({sig: unit.weights[0]} if unit.weights[0] else {}, unit.bias,
            {sig: unit.value_weights[0] * coef} if unit.value_weights[0] else {},
            unit.value_bias * coef)


def node(specs, numeric=False):
    """One glu from unit specs"""
    incoming: list[Bit] = []
    seen = set()
    for g, _, v, _ in specs:
        for s in list(g) + list(v):
            if s not in seen:
                seen.add(s)
                incoming.append(s)
    units = [Unit(tuple(g.get(s, 0) for s in incoming), gb,
                  tuple(v.get(s, 0) for s in incoming), vb) for g, gb, v, vb in specs]
    return G(incoming, units, numeric=numeric)


def parity_units(n: int, scale: float) -> list[Unit]:
    """glu_xor units on one feature f = s / scale with s in [0, n]"""
    units = [Unit((scale,), 0, (-scale,), 2)]
    units += [Unit((scale,), -2 * j, (0,), 4) for j in range(1, ceil(n / 2))]
    return units


# Parity of a count with knots between integers (fewer units than glu_xor's ceil(n/2), but
# with slope at the lattice points, so only for exact inputs, i.e. the first theta on raw
# message bits). Entry n: (gate a, gate b, value p, value q) for max(0, a s + b)(p + q s) on
# s = sum of the n inputs, plus a constant. n=7 (3 units, exact search t_par1d_all.py);
# n=9: brainstorm-critic's MIN_PARITY[9] (4 units), as (sig, knot, p, r).
MINPAR7 = ([(-6, 18, -25 / 36, -19 / 72), (2, -3, -47 / 6, 4 / 3), (5, -23, 0.0, -1 / 3)], 25 / 2)
MINPAR9_BC = (
    (1, 2.143004367452605, -0.3594345343626917, 5.7457719635860975),
    (1, 4.753569802478118, 1.359434534362685, -16.053874789546143),
    (-1, 4.339514021389544, 1.4464098175510671, 3.366207690876233),
    (-1, 6.830375534245772, -0.4464098175511091, -2.138638702986001),
)
MINPAR = {"on": False}


def par_units(n: int) -> list:
    """units for parity of n raw input bits: min-parity for n in (7, 9) if enabled"""
    if MINPAR["on"] and n == 7:
        us, c = MINPAR7
        out = [Unit((a,) * n, b, (q,) * n, p) for a, b, p, q in us]
        return out + [Unit((0,) * n, 1, (0,) * n, c)]
    if MINPAR["on"] and n == 9:
        out = []
        for sig, b, p, r in MINPAR9_BC:
            lam = 1 / min(abs(t - b) for t in range(n + 1) if abs(t - b) > 1e-9)
            out.append(Unit((sig * lam,) * n, -sig * lam * b, (p / lam,) * n, r / lam))
        return out
    return xor_units(n)


def neg(lit):
    return 1 - lit if isinstance(lit, int) else (lit[0], not lit[1])


def build(k: K.Keccak, depth: int, kinds=None, pairs: bool = True, walsh: bool = False, m1: int = 2,
          xpairs: bool = False, ydedup: bool = False, gfeat: bool = False, g3: bool = False,
          x1sep: bool = False, x2sep: bool = False, x2carry: bool = False, gmid: bool = False,
          x1pairs: bool = False, x2pairs: bool = False, m2: int = 0):
    """kinds[i] in ("direct", "split", "shared") is the theta layout of step i ("shared"
    only after the first step); the last step is direct (one theta bit per column pair
    left, nothing to share), or merged with its chi into one layer if walsh.
    Round-structure options (wave 2):
      xpairs: a chi layer that feeds an X layer (E = a + D) emits its state bits two per
              feature, p = a0 + 2 a1 (as p/4), instead of one per feature. The X layer
              needs one unit per bit anyway (the copy of a), and decodes a pair with two
              units, relu(p) and relu(p - 1)(4 - p): a0 = relu(p) - relu(p-1)(4-p),
              a1 = relu(p-1)(4-p)/2, exact on p in {0,1,2,3}. The X layer's input shrinks
              from 1921 to 1121 features (its units read 800 fewer columns).
      ydedup: the Y layer (theta = [E == 1]) emits each distinct theta bit once.
      gfeat:  a Y layer emits, per chi bit of the next layer, the chi gate
              g = 2 t_a - t_b + t_c (as g/4) instead of the theta bits; the chi unit
              max(0, 2a - b + c)(3 - 2a + b - c)/2 depends on g only, so it reads one feature.
      g3:     the same for the last theta layer: it emits the 224 chi gates of the digest
              (iota folded in) instead of the 320 theta bits.
      x1pairs (with x1sep): the first X layer passes the live message bits of a column two
              per feature (one pass unit per pair, p = a1 + 2 a2 as p/4) next to the column's
              D, and Y decodes (t1, t2) = (a1 ^ D, a2 ^ D) from (p, D) with 3 units per pair
              plus the column's D copy (shared by its pairs and zero lanes):
                A = relu(p)(1 - 2D), u = relu(p - 1 - 4D)(4 - p)/2, w = relu(p + 4D - 5)(p - 4)/2,
                t2 = D + u + w, t1 = D + A - 2(u + w)   (exact on the 8 points).
              X1's output shrinks from 1465 to 961 features and it has 504 fewer units.
      x2pairs (depth-8 split middle round, with xpairs): the chi layer pairs the bits y = 1, 2
              and y = 3, 4 of each column (they share D); the X layer passes these pairs with one
              unit each (instead of decoding them with two) and emits D per column; Y decodes
              them with the 3-unit form above. The y = 0 bits stay z-pairs, decoded in X and
              emitted as a (x2sep style).
      x1sep / x2sep: the first (x1sep) or a middle (x2sep) X layer emits a and D as separate
              features instead of E = a + D; the Y layer then computes t = a ^ D with one
              unit on two features, max(0, a + D)(2 - a - D). Same unit counts; the D units
              feed one feature instead of the 4-5 E's of their column (fewer wo nonzeros)."""
    w = k.w
    (rc,) = k.get_round_constants()
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[: k.d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]
    kinds = list(kinds or ["direct"] * depth)
    kinds[-1] = "direct"

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

    def pair_of(p):  # the column pair of theta position p: (x-1, *, z), (x+1, *, z+1)
        x, _, z = p
        return [((x + 4) % 5, y2, z) for y2 in range(5)] + [((x + 1) % 5, y2, (z + 1) % w) for y2 in range(5)]

    # ---- carries: ("bit", sig, flag) or ("pair", sig, (flag_lo, flag_hi)), sig = p/4 ----
    def carry_layer(carry):
        out, lazy, lazy4 = [], [], []
        for kind, sig, f in carry:
            if kind == "bit":
                out.append((kind, G([sig], [COPY]), f))
            elif kind == "pair":
                out.append((kind, G([sig], [Unit((4,), 0, (0,), 0.25)], numeric=True), f))
            elif kind == "pk":  # m bits packed binary, feature p / sc (combined-1)
                sc = pack_scale(2, len(f))
                out.append((kind, G([sig], [Unit((sc,), 0, (0,), 1 / sc)], numeric=True), f))
            elif kind == "abits":  # bits that are features of the layer before: a pass unit per pair
                if len(sig) >= 3:  # binary packed, as carry_new with m1 >= 3
                    sc = pack_scale(2, len(sig))
                    out.append(("pk", node([({}, 1, {s_: 2 ** j / sc for j, (s_, _) in enumerate(sig)}, 0.0)],
                                           numeric=True), tuple(f_ for _, f_ in sig)))
                elif len(sig) == 2:
                    (s0, f0), (s1, f1) = sig
                    out.append(("pair", node([({}, 1, {s0: 0.25, s1: 0.5}, 0.0)], numeric=True), (f0, f1)))
                else:
                    (s0, f0), = sig
                    out.append(("bit", node([spec_on(s0, COPY)]), f0))
            elif kind == "lazy4":  # lazy digest bit o/2, o in [0,4]: exact bit = parity(o)
                lazy4.append((sig, f))
            else:  # lazy digest bit o'/2, o' in {0,1,2}: exact bit = [o' == 1]
                lazy.append((sig, f))
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

    def carry_new(specs_flags, m=0):
        """carry items for new digest bits given as (unit spec at coef 1, flag) pairs;
        a spec may also be a list of specs (a bit that is a sum of several units)"""
        specs_flags = [(s_ if isinstance(s_, list) else [s_], f_) for s_, f_ in specs_flags]
        mm = m or m1  # bits per packed feature (m2 for the digests made by chi layers)
        items = []
        if pairs and mm >= 3:
            for i in range(0, len(specs_flags), mm):
                chunk = specs_flags[i:i + mm]
                sc = pack_scale(2, len(chunk))
                specs = []
                for j, (sl, _) in enumerate(chunk):
                    c = 2 ** j / sc
                    for s_ in sl:
                        specs.append((s_[0], s_[1], {q: v * c for q, v in s_[2].items()}, s_[3] * c))
                items.append(("pk", node(specs, numeric=True), tuple(f for _, f in chunk)))
        elif pairs:
            for i in range(0, len(specs_flags) - 1, 2):
                (s0, f0), (s1, f1) = specs_flags[i], specs_flags[i + 1]
                sc = lambda s, c: (s[0], s[1], {q: v * c for q, v in s[2].items()}, s[3] * c)
                items.append(("pair", node([sc(s, 0.25) for s in s0] + [sc(s, 0.5) for s in s1], numeric=True), (f0, f1)))
            if len(specs_flags) % 2:
                s, f = specs_flags[-1]
                items.append(("bit", node(s), f))
        else:
            for s, f in specs_flags:
                items.append(("bit", node(s), f))
        return items

    def decode(carry):
        """exact output bits; flips use one shared constant unit"""
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
            else:  # p/4, p = lo + 2 hi: lo = relu(p) - relu(p-1)(4-p), hi = relu(p-1)(4-p)/2
                lo = out([(sig, Unit((4,), 0, (0,), 1)), (sig, Unit((4,), -1, (4,), -4))], f[0])
                hi = out([(sig, Unit((4,), -1, (-2,), 2))], f[1])
                bits += [lo, hi]
        return bits

    # ---- theta layers ----
    def theta1_direct(a, pos):
        memo, th = {}, {}
        for p in pos:
            flip, cnt = 0, {}
            for lit in theta_lits(a, *p, w):
                if isinstance(lit, int):
                    flip ^= lit
                else:
                    flip ^= int(lit[1])
                    cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
            sigs = sorted((s for s, c in cnt.items() if c), key=lambda s: s.uid)
            if not sigs:
                th[p] = flip
                continue
            key = frozenset(sigs)
            if key not in memo:
                memo[key] = G(sigs, [COPY] if len(sigs) == 1 else par_units(len(sigs)))
            th[p] = (memo[key], bool(flip))
        return th

    def theta_direct(packed, carry, last=False):
        if g3 and last:  # emit the chi gates of the digest bits, iota folded into t_a
            th = {}
            for q in digest_pos:
                x, y, z = q
                specs, const = [], 0.0
                for dx, w_ in ((0, 2.0), (1, -1.0), (2, 1.0)):
                    sig, f, n = packed[rp[(x + dx) % 5][y][z]]
                    f = int(f) ^ int(dx == 0 and flag(*q))
                    c = w_ * (1 - 2 * f) / 4
                    specs += [spec_on(sig, u, c) for u in parity_units(n, SCALE)]
                    const += w_ * f / 4
                if const:
                    specs.append(({}, 1, {}, const))
                th[q] = ("g", node(specs, numeric=True))
            return th, carry_layer(carry)
        th = {p: (G([sig], parity_units(n, SCALE)), bool(f)) for p, (sig, f, n) in packed.items()}
        return th, carry_layer(carry)

    U0 = Unit((4,), 0, (0,), 1)  # on f = p/4: relu(p) * 1 = p
    U1 = Unit((4,), -1, (-4,), 4)  # relu(p - 1) * (4 - p)

    def a_specs(bit, coef):
        """specs of coef * a (raw bit, without its flag) for a bit given as (sig, flag) or,
        with xpairs, as (sig, slot, flag): slot 0 = relu(p) - relu(p-1)(4-p), 1 = relu(p-1)(4-p)/2"""
        if len(bit) == 2:
            return [spec_on(bit[0], COPY, coef)]
        sig, slot, _ = bit
        if slot == 0:
            return [spec_on(sig, U0, coef), spec_on(sig, U1, -coef)]
        return [spec_on(sig, U1, coef / 2)]

    def theta_split_x(bits, sums, carry):
        """E = a + D per position (numeric, E/2), D = parity(P) shared by the column pair;
        this step's digest carried from the copy units of a"""
        e = {}
        dmemo = {}
        pmemo = {}
        for p in all_pos:
            af, (psig, pf, n) = bits[p][-1], sums[(p[0], p[2])]
            if bits[p][0] == "cp":  # column pair: pass it on, Y decodes it with the column's D
                _, sig, slot, _ = bits[p]
                if psig not in dmemo:
                    dmemo[psig] = G([psig], parity_units(n, SCALE))
                if sig not in pmemo:
                    pmemo[sig] = G([sig], [Unit((4,), 0, (0,), 0.25)], numeric=True)
                e[p] = ("pair", pmemo[sig], slot, dmemo[psig], af ^ pf)
                continue
            if x2sep:
                if psig not in dmemo:
                    dmemo[psig] = G([psig], parity_units(n, SCALE))
                e[p] = ("sep", node(a_specs(bits[p], 1.0)), dmemo[psig], af ^ pf)
                continue
            specs = a_specs(bits[p], 0.5) + [spec_on(psig, u, 0.5) for u in parity_units(n, SCALE)]
            e[p] = (node(specs, numeric=True), af ^ pf)
        if x2sep and x2carry:  # the digest bits are X outputs already: the Y layer packs them
            ab = [(e[p][1], bits[p][-1]) for p in digest_pos]
            mm = 3 if m1 >= 3 else 2
            return e, carry_layer(carry) + [("abits", tuple(ab[i:i + mm]), None) for i in range(0, len(ab), mm)]
        new = [(a_specs(bits[p], 1.0), bits[p][-1]) for p in digest_pos]
        return e, carry_layer(carry) + carry_new(new)

    def theta_split_x_exact(bits, sums, carry):
        """E = a + D with the flags folded in (E/2, no flag), so that chi can read E"""
        e = {}
        for p in all_pos:
            af, (psig, pf, n) = bits[p][-1], sums[(p[0], p[2])]
            specs = a_specs(bits[p], 0.5 * (1 - 2 * af))
            specs += [spec_on(psig, u, 0.5 * (1 - 2 * pf)) for u in parity_units(n, SCALE)]
            if af or pf:
                specs.append(({}, 1, {}, 0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [(a_specs(bits[p], 1.0), bits[p][-1]) for p in digest_pos]
        return e, carry_layer(carry) + carry_new(new)

    XOR_E = Unit((2,), 0, (-2,), 2)  # on E/2: max(0, E)(2 - E) = [E == 1]
    Q_E = [Unit((-8, -4), 3, (0, 2), 0),  # on (Eb/2, Ec/2): max(0, 3 - 4Eb - 2Ec) Ec
           Unit((8, -4), -5, (0, 2), 0)]  # max(0, 4Eb - 2Ec - 5) Ec: [Eb != 1][Ec == 1]

    def lazy_chi(e, next_pos, carry):
        """chi (+iota) only mod 2 (theta-structure's lazy chi): o' = [Ea == 1] + [Eb != 1][Ec == 1]
        is congruent to chi, in {0,1,2}; the next theta reads counts of o' (range 22), and
        this step's digest bits are made exact one layer later (carry_layer)."""
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

    def lazy_chi4(e, next_pos, carry):
        """chi (+iota) mod 2 with ONE unit per bit plus a free pass term (unit-synthesis):
        o = Ea + 2 Eb + max(0, Eb + Ec) (1 - Eb) is congruent to chi and in [0, 4]; the gate
        Eb + Ec >= 0 is always active, so the unit is the exact product (Eb + Ec)(1 - Eb).
        The pass term (linear in E) of all bits summed into one count is ONE unit (gate on
        BOS), so a count costs 1 unit plus the shared product units; range 4 per bit."""
        def terms(q, coef):
            x, y, z = q
            ea, eb, ec = (e[rp[(x + dx) % 5][y][z]] for dx in range(3))  # features E/2
            prod = ({eb: 2.0, ec: 2.0}, 0, {eb: -2.0 * coef}, coef)
            pas = {ea: 2.0 * coef}
            pas[eb] = pas.get(eb, 0) + 4.0 * coef
            return prod, pas

        def merged(qs, coef):
            specs, pas = [], {}
            for q in qs:
                prod, pq = terms(q, coef)
                specs.append(prod)
                for sg, wv in pq.items():
                    pas[sg] = pas.get(sg, 0) + wv
            pas = {sg: wv for sg, wv in pas.items() if wv}
            if pas:
                specs.append(({}, 1, pas, 0.0))
            return specs
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            packed[p] = (node(merged(qs, 1 / SCALE), numeric=True), f, 4 * len(qs))
        new = [("lazy4", node(merged([p], 0.5), numeric=True), flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + new

    def lazy_chi4c(e, next_pos, carry):
        """lazy4 plus a cheap reduction of each column's linear part: a count is
        own o + (column x-1) + (column x+1), each column = sum_y Ea_y + sum_y Q'_y. The
        linear part c = sum_y Ea_y in [0, 10] is replaced by the congruent
        r(c) = max(0, 9 - 3c)(-2/3 - c/3) + 2 max(0, c - 7) + (2 - c) in [-5, -1]
        (2 units shared by the two counts that read the column, plus a pass term):
        count range 32 instead of 44 (16 parity units instead of 22)."""
        def q_prod(q, coef):
            x, y, z = q
            eb, ec = (e[rp[(x + dx) % 5][y][z]] for dx in (1, 2))  # features E/2
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
                g[f_] = g.get(f_, 0.0) + 2.0  # c = 2 * sum(E/2)
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
            const = 10.0 / SCALE  # offset: count in [0, 32]
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

    def theta_split_y(e, carry, chi_pos=None, first=False):
        """theta = [E == 1] (one unit on E/2). With gfeat and chi_pos, emits instead the chi
        gates g = 2 t_a - t_b + t_c of the chi bits at chi_pos, as nodes g/4 (th maps a chi
        position to ("g", node))."""
        ydef = Unit((2,), 0, (-2,), 2)

        def y_specs(ep, coef):  # coef * (theta bit without its flag), as a list of unit specs
            if ep[0] == "sep":
                _, a_sig, d_sig, _ = ep
                if a_sig is None:
                    return [spec_on(d_sig, COPY, coef)]
                return [({a_sig: 1.0, d_sig: 1.0}, 0, {a_sig: -coef, d_sig: -coef}, 2.0 * coef)]
            if ep[0] == "pair":  # (t1, t2) = (a1 ^ D, a2 ^ D) from f = p/4 and D
                _, f, slot, d, _ = ep
                dpass = spec_on(d, COPY, coef)
                A = lambda c: ({f: 4.0}, 0, {d: -2.0 * c}, 1.0 * c)
                u = lambda c: ({f: 4.0, d: -4.0}, -1, {f: -2.0 * c}, 2.0 * c)
                w_ = lambda c: ({f: 4.0, d: 4.0}, -5, {f: 2.0 * c}, -2.0 * c)
                if slot == 0:
                    return [dpass, A(coef), u(-2.0 * coef), w_(-2.0 * coef)]
                return [dpass, u(coef), w_(coef)]
            return [spec_on(ep[0], ydef, coef)]

        if (gfeat or (gmid and not first)) and chi_pos is not None:
            th = {}
            for q in chi_pos:
                x, y, z = q
                specs, const = [], 0.0
                for dx, w_ in ((0, 2.0), (1, -1.0), (2, 1.0)):
                    ep = e[rp[(x + dx) % 5][y][z]]
                    f = int(ep[-1])
                    c = w_ * (1 - 2 * f) / 4
                    specs += y_specs(ep, c)
                    const += w_ * f / 4
                if const:
                    specs.append(({}, 1, {}, const))
                th[q] = ("g", node(specs, numeric=True))
            return th, carry_layer(carry)
        memo, th = {}, {}
        for p, ep in e.items():
            key = ep[1:3] if ep[0] == "sep" else (ep[1:4] if ep[0] == "pair" else ep[0])
            if not ydedup or key not in memo:
                memo[key] = node(y_specs(ep, 1.0))
            th[p] = (memo[key], bool(ep[-1]))
        return th, carry_layer(carry)

    CHI_G = Unit((4,), 0, (-2,), 1.5)  # on g/4: max(0, g)(3 - g)/2 = chi

    def chi_spec(th, q, coef=1.0, iota=False):
        """spec of coef * chi at chi position q (iota folded into a if requested)"""
        if q in th and isinstance(th[q], tuple) and isinstance(th[q][0], str) and th[q][0] == "g":
            assert not iota
            return spec_on(th[q][1], CHI_G, coef)
        lits = chi_lits(th, *q)
        if iota:
            lits = [neg(lits[0])] + lits[1:]
        return spec(lits, CHI, coef)

    def theta_shared(bits, sums, carry):
        """theta = parity(a + T) per bit with units shared by the 5 bits of a column:
        3 per-bit units on (a, T), 5 units on T and a constant (see shared_theta.py)"""
        th = {}
        for p in all_pos:
            (asig, af), (tsig, tf, n) = bits[p], sums[(p[0], p[2])]
            sh, c = shared_units(n)
            specs = []
            for ga, gt, gb, va, vt, vb in PER_BIT[n] + sh:
                g = {q: float(v) for q, v in ((asig, ga), (tsig, gt * SCALE)) if v}
                v = {q: float(x) for q, x in ((asig, va), (tsig, vt * SCALE)) if x}
                specs.append((g, float(gb), v, float(vb)))
            specs.append(({}, 1, {}, float(c)))
            th[p] = (node(specs), bool(af ^ tf))
        return th, carry_layer(carry)

    # ---- chi layers ----
    def chi_to_direct(th, next_pos, carry):
        """chi bits as packed counts s for a direct theta, plus this step's digest"""
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            packed[p] = (node([chi_spec(th, q, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        new = [(chi_spec(th, p), flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + carry_new(new, m2)

    def chi_to_split(th, carry, cp=False):
        """chi bits (all positions) and column-pair counts P for a split theta; with xpairs
        the bits of (x, y, z) and (x, y, z + 1), z even, share one feature p/4 = (a0 + 2 a1)/4"""
        if xpairs and x2pairs and cp and w % 2 == 0:
            bits = {}
            for (x, y, z) in all_pos:
                if y == 0 and z % 2 == 0:  # z-pair of the y = 0 bits (decoded in X)
                    p0, p1 = (x, 0, z), (x, 0, z + 1)
                    sig = node([chi_spec(th, p0, 0.25), chi_spec(th, p1, 0.5)], numeric=True)
                    bits[p0] = (sig, 0, flag(*p0))
                    bits[p1] = (sig, 1, flag(*p1))
                elif y in (1, 3):  # column pairs (y, y + 1): same D, passed through X
                    p0, p1 = (x, y, z), (x, y + 1, z)
                    sig = node([chi_spec(th, p0, 0.25), chi_spec(th, p1, 0.5)], numeric=True)
                    bits[p0] = ("cp", sig, 0, flag(*p0))
                    bits[p1] = ("cp", sig, 1, flag(*p1))
        elif xpairs:
            bits = {}
            for (x, y, z) in all_pos:
                if z % 2:
                    continue
                p0, p1 = (x, y, z), (x, y, z + 1)
                if z + 1 >= w:  # odd lane width (w = 1): a single bit
                    bits[p0] = (node([chi_spec(th, p0)]), flag(*p0))
                    continue
                sig = node([chi_spec(th, p0, 0.25), chi_spec(th, p1, 0.5)], numeric=True)
                bits[p0] = (sig, 0, flag(*p0))
                bits[p1] = (sig, 1, flag(*p1))
        else:
            bits = {p: (node([chi_spec(th, p)]), flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                sums[(x, z)] = (node([chi_spec(th, q, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        return bits, sums, carry_layer(carry)

    def chi_to_shared(th, carry):
        """chi bits (all positions), column-pair counts T, and this step's digest"""
        bits, sums, carry = chi_to_split(th, carry)
        chi = {p: chi_lits(th, *p) for p in digest_pos}
        new = [(spec(chi[p], CHI), flag(*p)) for p in digest_pos]
        return bits, sums, carry + carry_new(new)

    def walsh_last(packed, carry):
        """the last round as ONE layer (idea of the linear-fold avenue): with a, b, c the
        theta bits parity(S) of the digest's chi, chi = (a - a^b + a^c + a^b^c) / 2, and
        each xor is the parity of a sum of counts: 6 + 11 + 11 + 17 units per digest bit"""
        bits = decode(carry)
        for p in digest_pos:
            x, _, z = p
            q = [rp[(x + dx) % 5][0][z] for dx in range(3)]
            r = flag(*p)
            specs, const = [], float(r)
            for coef, subset in ((0.5, (0,)), (-0.5, (0, 1)), (0.5, (0, 2)), (0.5, (0, 1, 2))):
                feats = [packed[q[i]] for i in subset]
                fq = sum(f for _, f, _ in feats) % 2
                n = sum(nn for _, _, nn in feats)
                wq = (1 - 2 * r) * coef * (1 - 2 * fq)
                const += (1 - 2 * r) * coef * fq
                for j, u in enumerate(parity_units(n, SCALE)):
                    g = {sig: float(SCALE) for sig, _, _ in feats}
                    v = {sig: -float(SCALE) * wq for sig, _, _ in feats} if j == 0 else {}
                    specs.append((g, float(u.bias), v, float(u.value_bias) * wq))
            if const:
                specs.append(({}, 1.0, {}, const))
            bits.append(node(specs))
        return bits

    def chi_last(th, carry):
        bits = decode(carry)
        for p in digest_pos:
            if isinstance(th.get(p), tuple) and th[p][0] == "g":  # iota already folded
                bits.append(node([spec_on(th[p][1], CHI_G)]))
                continue
            lits = chi_lits(th, *p)
            if flag(*p):
                lits = [neg(lits[0])] + lits[1:]
            bits.append(node([spec(lits, CHI)]))
        return bits

    # split theta for step 0 reads literals: E = a + D with D an xor of message bits
    def theta1_split_x(a, carry):
        memo_e, e = {}, {}
        live: dict = {}  # x1pairs: live message bits per column (x, z), and the column's D node
        dnode_of: dict = {}
        for p in all_pos:
            flip, cnt = 0, {}
            for lit in [a[q[0]][q[1]][q[2]] for q in pair_of(p)]:
                if isinstance(lit, int):
                    flip ^= lit
                else:
                    flip ^= int(lit[1])
                    cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
            dsigs = tuple(sorted((s for s, c in cnt.items() if c), key=lambda s: s.uid))
            alit = a[p[0]][p[1]][p[2]]
            if isinstance(alit, int):
                flip ^= alit
                aterm = None
            else:
                flip ^= int(alit[1])
                aterm = alit[0]
            if x1sep:
                if dsigs not in memo_e:
                    n = len(dsigs)
                    us = [COPY] if n == 1 else par_units(n)
                    memo_e[dsigs] = G(list(dsigs), us)
                if x1pairs and aterm is not None:
                    live.setdefault((p[0], p[2]), []).append((p, aterm, flip))
                    dnode_of[(p[0], p[2])] = memo_e[dsigs]
                    continue
                if aterm is not None and ("a", aterm) not in memo_e:
                    memo_e[("a", aterm)] = G([aterm], [COPY])
                e[p] = ("sep", None if aterm is None else memo_e[("a", aterm)], memo_e[dsigs], flip)
                continue
            key = (aterm, dsigs)
            if key not in memo_e:
                specs = []
                if aterm is not None:
                    specs.append(spec_on(aterm, COPY, 0.5))
                if dsigs:
                    n = len(dsigs)
                    us = [COPY] if n == 1 else par_units(n)
                    for u in us:
                        specs.append(({s: u.weights[i] for i, s in enumerate(dsigs) if u.weights[i]}, u.bias,
                                      {s: u.value_weights[i] * 0.5 for i, s in enumerate(dsigs) if u.value_weights[i]},
                                      u.value_bias * 0.5))
                memo_e[key] = node(specs, numeric=True)
            e[p] = (memo_e[key], flip)
        for (x, z), lst in live.items():  # x1pairs: live bits of a column share D
            for i in range(0, len(lst) - 1, 2):
                (p0, a0, f0), (p1, a1, f1) = lst[i], lst[i + 1]
                pn = node([({}, 1, {a0: 0.25, a1: 0.5}, 0.0)], numeric=True)
                d0 = dnode_of[(x, z)]
                e[p0] = ("pair", pn, 0, d0, f0)
                e[p1] = ("pair", pn, 1, d0, f1)
            if len(lst) % 2:
                p0, a0, f0 = lst[-1]
                if ("a", a0) not in memo_e:
                    memo_e[("a", a0)] = G([a0], [COPY])
                e[p0] = ("sep", memo_e[("a", a0)], dnode_of[(x, z)], f0)
        return e, carry

    def xof_fn(msg: list) -> list:
        lanes = initial_state(k, msg)
        needs = [all_pos] * (depth - 1) + [digest_pos]  # chi outputs each step needs
        carry: list = []
        if kinds[0] == "split":
            e, carry = theta1_split_x(lanes, carry)
            th, carry = theta_split_y(e, carry, all_pos if depth > 1 else None, first=True)
        else:
            th = theta1_direct(lanes, chi_needs(needs[0]))
        step = 0
        while step < depth - 1:  # chi layer of step, then the theta layer(s) of step + 1
            nxt = kinds[step + 1]
            if walsh and step == depth - 2:  # chi layer, then the merged last round
                packed, carry = chi_to_direct(th, chi_needs(digest_pos), carry)
                return walsh_last(packed, carry)
            if nxt == "split":
                bits, sums, carry = chi_to_split(th, carry, cp=True)
                e, carry = theta_split_x(bits, sums, carry)
                th, carry = theta_split_y(e, carry, all_pos if step + 1 < depth - 1 else None)
            elif nxt == "lazy":  # X (exact E), lazy chi, then the next step's theta
                assert step + 2 <= depth - 1, "a lazy step needs a step after it"
                bits, sums, carry = chi_to_split(th, carry)
                e, carry = theta_split_x_exact(bits, sums, carry)
                packed, carry = lazy_chi(e, chi_needs(needs[step + 2]), carry)
                if walsh and step + 2 == depth - 1:
                    return walsh_last(packed, carry)
                th, carry = theta_direct(packed, carry)
                step += 1
            elif nxt in ("lazy4", "lazy4c"):  # X (exact E), one-unit lazy chi, next theta
                assert step + 2 <= depth - 1, "a lazy step needs a step after it"
                bits, sums, carry = chi_to_split(th, carry)
                e, carry = theta_split_x_exact(bits, sums, carry)
                packed, carry = (lazy_chi4 if nxt == "lazy4" else lazy_chi4c)(e, chi_needs(needs[step + 2]), carry)
                th, carry = theta_direct(packed, carry, last=step + 2 == depth - 1)
                step += 1
            elif nxt == "shared":
                bits, sums, carry = chi_to_shared(th, carry)
                th, carry = theta_shared(bits, sums, carry)
            else:
                packed, carry = chi_to_direct(th, chi_needs(needs[step + 1]), carry)
                th, carry = theta_direct(packed, carry, last=step + 1 == depth - 1)
            step += 1
        return chi_last(th, carry)

    return xof_fn


def make(kinds_fn, pairs=True, walsh=False, m1=2, **kw):
    def variant(k, depth):
        return build(k, depth, kinds_fn(depth), pairs, walsh, m1, **kw), {"msg": Bits("0" * k.msg_len).bitlist}
    return variant


def middle(kind):
    return lambda d: ["direct"] + [kind] * (d - 2) + ["direct"] if d > 1 else ["direct"]


direct = make(lambda d: ["direct"] * d)  # 2 layers per step
direct_bits = make(lambda d: ["direct"] * d, pairs=False)  # digests carried as bits
split_middle = make(middle("split"))  # 3 layers for the middle steps
split_first_middle = make(lambda d: ["split"] * (d - 1) + ["direct"])
shared_middle = make(middle("shared"))  # exact but sensitive, see NOTES.md
direct_walsh = make(lambda d: ["direct"] * d, walsh=True)  # last step in one layer
split_middle_walsh = make(middle("split"), walsh=True)
split_first_middle_walsh = make(lambda d: ["split"] * (d - 1) + ["direct"], walsh=True)
shared_middle_walsh = make(middle("shared"), walsh=True)
lazy_middle = make(middle("lazy"))  # theta-structure's lazy chi on this builder
split_first_lazy = make(lambda d: ["split"] * (d - 2) + ["lazy", "direct"] if d > 2 else ["direct"] * d)
lazy4_middle = make(middle("lazy4"))  # unit-synthesis: one-unit lazy chi (width 4)
split_first_lazy4 = make(lambda d: ["split"] * (d - 2) + ["lazy4", "direct"] if d > 2 else ["direct"] * d)
lazy4c_middle = make(middle("lazy4c"))  # lazy4 + 2-unit reduction of column linear parts
split_first_lazy4c = make(lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d)


def with_minpar(variant):
    """the same variant with min-parity units in the first theta (raw bits)"""
    def v(k, depth):
        MINPAR["on"] = True
        return variant(k, depth)
    return v


lazy4c_middle_mp = with_minpar(lazy4c_middle)
split_first_lazy4c_mp = with_minpar(split_first_lazy4c)


# ---- combined-1: digest 1 packed 3 bits per feature (binary, DP-minimal decoders) ----
lazy4c_middle_m3 = make(middle("lazy4c"), m1=3)
split_first_lazy4c_m3 = make(lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d, m1=3)
split_first_middle_m3 = make(lambda d: ["split"] * (d - 1) + ["direct"], m1=3)
lazy4c_middle_m3_mp = with_minpar(lazy4c_middle_m3)
split_first_lazy4c_m3_mp = with_minpar(split_first_lazy4c_m3)
split_first_middle_m3_mp = with_minpar(split_first_middle_m3)
split_first_middle_mp = with_minpar(split_first_middle)


# ---- wave 2, round-structure: X-layer pairs (xp), Y dedup (yd), chi gate features (g) ----
def _first_split_lazy(kind):
    return lambda d: ["split"] * (d - 2) + [kind, "direct"] if d > 2 else ["direct"] * d


def _split_first_middle(d):
    return ["split"] * (d - 1) + ["direct"]


lazy4c_middle_m3_mp_xp = with_minpar(make(middle("lazy4c"), m1=3, xpairs=True))  # depth 6
lazy4c_middle_mp_xp = with_minpar(make(middle("lazy4c"), xpairs=True))
split_first_lazy4c_m3_mp_xp = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True))  # 7
split_first_lazy4c_m3_mp_xpyd = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True))
split_first_lazy4c_m3_mp_xpg = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, gfeat=True))
split_first_middle_mp_xp = with_minpar(make(_split_first_middle, xpairs=True))  # depth 8
split_first_middle_mp_xpyd = with_minpar(make(_split_first_middle, xpairs=True, ydedup=True))
split_first_middle_mp_xpg = with_minpar(make(_split_first_middle, xpairs=True, gfeat=True))
split_first_middle_m3_mp_xpyd = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True))
split_first_middle_xpg = make(_split_first_middle, xpairs=True, gfeat=True)  # no min-parity: sparse
split_first_lazy4c_xpg = make(_first_split_lazy("lazy4c"), xpairs=True, gfeat=True)  # depth 7, sparse
split_first_lazy4c_m3_xpg = make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, gfeat=True)
split_first_lazy4c_mp_xpg = with_minpar(make(_first_split_lazy("lazy4c"), xpairs=True, gfeat=True))

# g3: chi gates of the digest from the last theta layer; x1sep/x2sep: a and D separate
lazy4c_middle_m3_mp_xpg3 = with_minpar(make(middle("lazy4c"), m1=3, xpairs=True, g3=True))  # depth 6
split_first_lazy4c_m3_mp_xpydg3 = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True, g3=True))
split_first_middle_mp_xpydg3 = with_minpar(make(_split_first_middle, xpairs=True, ydedup=True, g3=True))
split_first_middle_xpg_x1s = make(_split_first_middle, xpairs=True, gfeat=True, x1sep=True)
split_first_middle_xpg_x12s = make(_split_first_middle, xpairs=True, gfeat=True, x1sep=True, x2sep=True)
split_first_lazy4c_xpg_x1s = make(_first_split_lazy("lazy4c"), xpairs=True, gfeat=True, x1sep=True)
# combinations
split_first_middle_mp_xpydg3_x1s = with_minpar(make(_split_first_middle, xpairs=True, ydedup=True, g3=True, x1sep=True))  # 8
split_first_middle_m3_mp_xpydg3_x1s = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True))
split_first_lazy4c_m3_mp_xpydg3_x1s = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True))  # 7
split_first_lazy4c_m3_mp_xpyd_x1s = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True, x1sep=True))
# sparse-leaning: theta-structure's lazy chi (o' = [Ea == 1] + Q, exact 2-unit Q, counts <= 22)
split_first_lazy_xpg_x1s = make(_first_split_lazy("lazy"), xpairs=True, gfeat=True, x1sep=True)  # 7
lazy_middle_xp = make(middle("lazy"), xpairs=True)  # 6
lazy4c_middle_xp = make(middle("lazy4c"), xpairs=True)  # 6, no min-parity
split_first_middle_xpg_x12sc = make(_split_first_middle, xpairs=True, gfeat=True, x1sep=True, x2sep=True, x2carry=True)
split_first_middle_mp_xpg_x1s = with_minpar(make(_split_first_middle, xpairs=True, gfeat=True, x1sep=True))
split_first_middle_mp_xpg_x12sc = with_minpar(make(_split_first_middle, xpairs=True, gfeat=True, x1sep=True, x2sep=True, x2carry=True))
# gmid: g features only after the middle Y layer (no duplicates there, so no dense cost)
split_first_middle_m3_mp_xpydg3_x1s_gm = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True, gmid=True))
split_first_middle_mp_xpydg3_x1s_gm = with_minpar(make(_split_first_middle, xpairs=True, ydedup=True, g3=True, x1sep=True, gmid=True))
# x1pairs: live message bits of a column two per feature into Y1 (3-unit decode with the column's D)
split_first_middle_m3_mp_xpydg3_x1p_gm = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, gmid=True))
split_first_lazy4c_m3_mp_xpydg3_x1p = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True))
# x2pairs: column pairs passed through the middle X layer, decoded in Y (depth 8)
split_first_middle_m3_mp_xpydg3_x1p_x2p_gm = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, gmid=True))
split_first_middle_mp_xpydg3_x1p_x2pc_gm = with_minpar(make(_split_first_middle, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, x2carry=True, gmid=True))
split_first_middle_m3_mp_xpydg3_x1p_x2pc_gm = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, x2carry=True, gmid=True))
split_first_middle_m3_mp_xpydg3_x1p_x2pc = with_minpar(make(_split_first_middle, m1=3, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, x2carry=True))
split_first_middle_mp_xpydg3_x1p_x2pc = with_minpar(make(_split_first_middle, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, x2carry=True))
split_first_middle_xpyd_x1p_x2pc = make(_split_first_middle, xpairs=True, ydedup=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, x2carry=True)
split_first_lazy4c_m3_mp_xpyd_x1p = with_minpar(make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True, x1sep=True, x1pairs=True))
split_first_lazy4c_m3_xpyd_x1p = make(_first_split_lazy("lazy4c"), m1=3, xpairs=True, ydedup=True, x1sep=True, x1pairs=True)
# m2: digest 2 (made by the chi layer before the last theta) two bits per feature, digest 1 three
split_first_middle_m3_mp_xpydg3_x1p_x2pc_d2p = with_minpar(make(_split_first_middle, m1=3, m2=2, xpairs=True, ydedup=True, g3=True, x1sep=True, x1pairs=True, x2pairs=True, x2sep=True, x2carry=True))

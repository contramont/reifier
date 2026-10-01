"""XOF-structured 1-round Keccak, general layout (avenue xof-structure), with narrower
interfaces (wave-2 avenue "packing", options pa / dy / pd1 / lf of build):
  pa:  the chi layer before a split theta emits its state bits two per feature,
       f = a0 + 2 a1 (as f/4); the X layer reads both with the 2 units it spends on
       the two copies anyway: relu(f) = f and a1 = relu(f - 1)(4 - f)/2, a0 = f - 2 a1;
  dy:  a Y layer emits each distinct theta bit once (round 1 has 1465, not 1600);
  pd1: the first X layer emits its D-only E values (capacity lanes, E = D in {0,1})
       two per feature; the Y layer decodes them the same way;
  lf:  the last theta layer emits the chi pre-activations L = 2a - b + c of the 224
       digest bits instead of the 320 theta bits; chi = relu(L)(3 - L)/2;
  g1:  (replaces pd1) the first X layer emits G = a + 2D instead of E = a + D for the
       message positions: theta = a ^ D = relu(G)(3 - G)/2 and D = relu(G - 1)(4 - G)/2
       are one unit each, so the D-only values of the capacity lanes need no feature;
  s6:  digest 1 travels 6 bits per feature (m1=6) and the last theta layer splits each
       feature p into two 3-bit ones: hi = floor(p/8) (DP-minimal staircase, 14 units)
       and lo = p - 8 hi (the copy unit), so the last layer keeps its 3-bit decoders;
  s4:  digest 1 travels 4 bits per feature (m1=4) and the last theta layer splits each
       feature into two pairs (hi = floor(p/4), 6 units; lo = p - 4 hi), which the last
       layer decodes with 2 units per pair (s6, the same with 6 bits, fails the audit:
       the 64x scale amplifies float errors to 0.03);
  d2p5: the lazy digest values o in [0, 4] of a lazy chi layer travel two per feature,
       F = o0 + 5 o1 (one pass unit per pair), and the next layer turns F into the exact
       pair b0 + 2 b1 with the DP-minimal decoder of that function (12 units, integer knots);
  pc:  (with pa) a column's 5 bits travel as its sum C and the pairs (a0, a1), (a2, a3):
       a4 = C - (a0 + a1) - (a2 + a3), and the X layer's D units read the column-pair
       count as C(x-1, z) + C(x+1, z+1), so the 320 count features P are not needed.


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


# pair packing of two state bits (avenue packing): f = a0 + 2 a1, emitted as f/4
PAIR_COPY = Unit((4,), 0, (0,), 1)  # on f/4: relu(f) * 1 = f
PAIR_HI = Unit((4,), -1, (-2,), 2)  # on f/4: relu(f - 1)(2 - f/2) = a1 (f in {0,1,2,3})
CHI_L = Unit((4,), 0, (-2,), 1.5)  # on L/4: relu(L)(3 - L)/2 = chi, L = 2a - b + c
CCOPY = Unit((8,), 0, (0,), 1)  # on C/8: relu(C) * 1 = C (a column sum, C in [0, 5])


def pair_terms(sig, j):
    """bit j of the pair feature sig as (signal, unit, coef) terms: a0 = f - 2 a1, a1"""
    return [(sig, PAIR_COPY, 1.0), (sig, PAIR_HI, -2.0)] if j == 0 else [(sig, PAIR_HI, 1.0)]


def term_specs(terms, coef):
    return [spec_on(s, u, c * coef) for s, u, c in terms]


def par_specs(feats, n, coef):
    """coef * parity(s) for s = sum_i sc_i * f_i in [0, n] (glu_xor units, one gate on all)"""
    g = {}
    for sig, sc in feats:
        g[sig] = g.get(sig, 0.0) + float(sc)
    specs = []
    for j, u in enumerate(parity_units(n, 1.0)):
        v = {sig: -w * coef for sig, w in g.items()} if j == 0 else {}
        specs.append((dict(g), float(u.bias), v, float(u.value_bias) * coef))
    return specs


def scale_specs(sl, c):
    return [(s[0], s[1], {q: v * c for q, v in s[2].items()}, s[3] * c) for s in sl]


def build(k: K.Keccak, depth: int, kinds=None, pairs: bool = True, walsh: bool = False, m1: int = 2,
          pa: bool = False, dy: bool = False, pd1: bool = False, lf: bool = False, m2=None,
          pc: bool = False, d2p5: bool = False, g1: bool = False, s6: bool = False,
          s4: bool = False):
    """kinds[i] in ("direct", "split", "shared") is the theta layout of step i ("shared"
    only after the first step); the last step is direct (one theta bit per column pair
    left, nothing to share), or merged with its chi into one layer if walsh."""
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
    def carry_layer(carry, split=False):
        out, lazy, lazy4 = [], [], []
        for kind, sig, f in carry:
            if split and s4 and kind == "pk" and len(f) == 4:  # p/16 -> pairs (p - 4 hi)/4, hi/4
                if "st4" not in DECODERS:
                    DECODERS["st4"] = synth_decoder([p >> 2 for p in range(16)])
                const, units = DECODERS["st4"]
                hi = [({sig: 16.0}, float(-kk), {sig: 16.0 * float(b)} if b else {}, float(a))
                      for kk, a, b in units]
                if const:
                    hi.append(({}, 1.0, {}, float(const)))
                lo = [({sig: 16.0}, 0.0, {}, 1 / 4)] + scale_specs(hi, -1.0)
                out.append(("pair", node(lo, numeric=True), f[:2]))
                out.append(("pair", node(scale_specs(hi, 1 / 4), numeric=True), f[2:]))
                continue
            if split and s6 and kind == "pk" and len(f) == 6:  # p/64 -> (p - 8 hi)/8, hi/8
                if "st8" not in DECODERS:
                    DECODERS["st8"] = synth_decoder([p >> 3 for p in range(64)])
                const, units = DECODERS["st8"]
                hi = [({sig: 64.0}, float(-kk), {sig: 64.0 * float(b)} if b else {}, float(a))
                      for kk, a, b in units]
                if const:
                    hi.append(({}, 1.0, {}, float(const)))
                lo = [({sig: 64.0}, 0.0, {}, 1 / 8)] + scale_specs(hi, -1.0)
                out.append(("pk", node(lo, numeric=True), f[:3]))
                out.append(("pk", node(scale_specs(hi, 1 / 8), numeric=True), f[3:]))
                continue
            if kind == "bit":
                out.append((kind, G([sig], [COPY]), f))
            elif kind == "pair":
                out.append((kind, G([sig], [Unit((4,), 0, (0,), 0.25)], numeric=True), f))
            elif kind == "pk":  # m bits packed binary, feature p / sc (combined-1)
                sc = pack_scale(2, len(f))
                out.append((kind, G([sig], [Unit((sc,), 0, (0,), 1 / sc)], numeric=True), f))
            elif kind == "lazy4":  # lazy digest bit o/2, o in [0,4]: exact bit = parity(o)
                lazy4.append((sig, f))
            elif kind == "lz5":  # F/32, F = o0 + 5 o1: exact pair (b0 + 2 b1)/4 = h(F)/4
                if "lz5" not in DECODERS:
                    DECODERS["lz5"] = synth_decoder([(F % 5) % 2 + 2 * ((F // 5) % 2) for F in range(25)])
                const, units = DECODERS["lz5"]
                specs = [({sig: 32.0}, float(-kk), {sig: 32.0 * float(b) / 4} if b else {}, float(a) / 4)
                         for kk, a, b in units]
                if const:
                    specs.append(({}, 1.0, {}, float(const) / 4))
                out.append(("pair", node(specs, numeric=True), f))
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

    ndig = {"n": 0}  # digests made so far (m1 packs the first one, m2 the later ones)

    def carry_new(specs_flags):
        """carry items for new digest bits given as (list of unit specs at coef 1, flag)"""
        items = []
        m = m1 if ndig["n"] == 0 or m2 is None else m2
        ndig["n"] += 1
        if pairs and m >= 3:
            for i in range(0, len(specs_flags), m):
                chunk = specs_flags[i:i + m]
                sc = pack_scale(2, len(chunk))
                specs = []
                for j, (sl, _) in enumerate(chunk):
                    specs += scale_specs(sl, 2 ** j / sc)
                items.append(("pk", node(specs, numeric=True), tuple(f for _, f in chunk)))
        elif pairs:
            for i in range(0, len(specs_flags) - 1, 2):
                (s0, f0), (s1, f1) = specs_flags[i], specs_flags[i + 1]
                items.append(("pair", node(scale_specs(s0, 0.25) + scale_specs(s1, 0.5), numeric=True), (f0, f1)))
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
        if lf and last:
            return ("L", theta_lform(packed)), carry_layer(carry, split=True)
        th = {p: (G([sig], parity_units(n, SCALE)), bool(f)) for p, (sig, f, n) in packed.items()}
        return th, carry_layer(carry, split=last)

    def theta_lform(packed):
        """per digest bit p, the feature L/4 with L = 2a - b + c (iota folded into a), where
        a, b, c = parity(count) ^ flag are its chi inputs: the parity units of the theta
        bits are shared (CSE), and the layer emits 224 features instead of 320"""
        Ls = {}
        for p in digest_pos:
            x, y, z = p
            specs, const = [], 0.0
            for dx, coeff in ((0, 2.0), (1, -1.0), (2, 1.0)):
                sig, f, n = packed[rp[(x + dx) % 5][y][z]]
                f = int(f) ^ (flag(*p) if dx == 0 else 0)
                c = coeff * (1 - 2 * f) / 4
                specs += [spec_on(sig, u, c) for u in parity_units(n, SCALE)]
                const += coeff * f / 4
            if const:
                specs.append(({}, 1, {}, const))
            Ls[p] = node(specs, numeric=True)
        return Ls

    def theta_split_x(bits, sums, carry):
        """E = a + D per position (numeric, E/2), D = parity(P) shared by the column pair;
        this step's digest carried from the copy units of a"""
        e = {}
        for p in all_pos:
            (terms, af), (feats, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = term_specs(terms, 0.5) + par_specs(feats, n, 0.5)
            e[p] = (node(specs, numeric=True), af ^ pf)
        new = [(term_specs(bits[p][0], 1.0), bits[p][1]) for p in digest_pos]
        return e, carry_layer(carry) + carry_new(new)

    def theta_split_x_exact(bits, sums, carry):
        """E = a + D with the flags folded in (E/2, no flag), so that chi can read E"""
        e = {}
        for p in all_pos:
            (terms, af), (feats, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = term_specs(terms, 0.5 * (1 - 2 * af))
            specs += par_specs(feats, n, 0.5 * (1 - 2 * pf))
            if af or pf:
                specs.append(({}, 1, {}, 0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [(term_specs(bits[p][0], 1.0), bits[p][1]) for p in digest_pos]
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
        dps = list(digest_pos)
        if d2p5:  # two lazy digest values per feature, F = o0 + 5 o1 (as F/32), one pass unit
            for i in range(0, len(dps) - 1, 2):
                (sp0, pas0), (sp1, pas1) = own(dps[i], 1 / 32), own(dps[i + 1], 5 / 32)
                pas = dict(pas0)
                for s_, w_ in pas1.items():
                    pas[s_] = pas.get(s_, 0) + w_
                pas = {s_: w_ for s_, w_ in pas.items() if abs(w_) > 1e-12}
                new.append(("lz5", node(sp0 + sp1 + [({}, 1, pas, 0.0)], numeric=True),
                            (flag(*dps[i]), flag(*dps[i + 1]))))
            dps = dps[len(dps) - len(dps) % 2:]
        for p in dps:
            sp, pas = own(p, 0.5)
            new.append(("lazy4", node(sp + [({}, 1, pas, 0.0)], numeric=True), flag(*p)))
        return packed, carry_layer(carry) + new

    def theta_split_y(e, carry):
        th, memo = {}, {}
        for p, (src, f) in e.items():
            if isinstance(src, tuple) and src[0] == "gx":  # G/4, G = a + 2D: a ^ D = relu(G)(3-G)/2
                if src not in memo:
                    memo[src] = node([spec_on(src[1], CHI_L)])
                th[p] = (memo[src], bool(f))
                continue
            if isinstance(src, tuple):  # ("pk2", F, j): bit j of a pair of D-only values
                if src not in memo:
                    memo[src] = node(term_specs(pair_terms(src[1], src[2]), 1.0))
                th[p] = (memo[src], bool(f))
                continue
            if src not in memo or not dy:
                memo[src] = G([src], [Unit((2,), 0, (-2,), 2)])
            th[p] = (memo[src], bool(f))
        return th, carry_layer(carry)

    def theta_shared(bits, sums, carry):
        """theta = parity(a + T) per bit with units shared by the 5 bits of a column:
        3 per-bit units on (a, T), 5 units on T and a constant (see shared_theta.py)"""
        assert not pa and not pc, "shared theta reads unpacked bits"
        th = {}
        for p in all_pos:
            (terms, af), (tfe, tf, n) = bits[p], sums[(p[0], p[2])]
            ((asig, _, _),) = terms
            ((tsig, _),) = tfe
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
        chi = {p: chi_lits(th, *p) for p in all_pos}
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            packed[p] = (node([spec(chi[q], CHI, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        new = [([spec(chi[p], CHI)], flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + carry_new(new)

    def chi_to_split(th, carry):
        """chi bits (all positions) and column-pair counts P for a split theta"""
        chi = {p: chi_lits(th, *p) for p in all_pos}
        if pc:  # per column (x, z): its sum C (as C/8) and the pairs (y0, y1), (y2, y3)
            assert pa
            bits, csig = {}, {}
            for x in range(5):
                for z in range(w):
                    col = [(x, y, z) for y in range(5)]
                    f1 = node([spec(chi[col[0]], CHI, 0.25), spec(chi[col[1]], CHI, 0.5)], numeric=True)
                    f2 = node([spec(chi[col[2]], CHI, 0.25), spec(chi[col[3]], CHI, 0.5)], numeric=True)
                    c = node([spec(chi[q], CHI, 1 / 8) for q in col], numeric=True)
                    csig[(x, z)] = c
                    for j in range(2):
                        bits[col[j]] = (pair_terms(f1, j), flag(*col[j]))
                        bits[col[2 + j]] = (pair_terms(f2, j), flag(*col[2 + j]))
                    # a4 = C - (a0 + a1) - (a2 + a3), a0 + a1 = f - a1
                    bits[col[4]] = ([(c, CCOPY, 1.0), (f1, PAIR_COPY, -1.0), (f1, PAIR_HI, 1.0),
                                     (f2, PAIR_COPY, -1.0), (f2, PAIR_HI, 1.0)], flag(*col[4]))
            sums = {}
            for x in range(5):
                for z in range(w):
                    qs = pair_of((x, 0, z))
                    f = sum(flag(*q) for q in qs) % 2
                    sums[(x, z)] = ([(csig[((x + 4) % 5, z)], 8.0), (csig[((x + 1) % 5, (z + 1) % w)], 8.0)],
                                    f, len(qs))
            return bits, sums, carry_layer(carry)
        if pa:  # two state bits per feature, f = a0 + 2 a1 (as f/4)
            bits = {}
            for i in range(0, len(all_pos) - 1, 2):
                p0, p1 = all_pos[i], all_pos[i + 1]
                f = node([spec(chi[p0], CHI, 0.25), spec(chi[p1], CHI, 0.5)], numeric=True)
                bits[p0] = (pair_terms(f, 0), flag(*p0))
                bits[p1] = (pair_terms(f, 1), flag(*p1))
            if len(all_pos) % 2:  # odd count (small w): the last bit alone
                p = all_pos[-1]
                bits[p] = ([(node([spec(chi[p], CHI)]), COPY, 1.0)], flag(*p))
        else:
            bits = {p: ([(node([spec(chi[p], CHI)]), COPY, 1.0)], flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                sums[(x, z)] = ([(node([spec(chi[q], CHI, 1 / SCALE) for q in qs], numeric=True), SCALE)],
                                f, len(qs))
        return bits, sums, carry_layer(carry)

    def chi_to_shared(th, carry):
        """chi bits (all positions), column-pair counts T, and this step's digest"""
        bits, sums, carry = chi_to_split(th, carry)
        chi = {p: chi_lits(th, *p) for p in digest_pos}
        new = [([spec(chi[p], CHI)], flag(*p)) for p in digest_pos]
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
        if isinstance(th, tuple) and th[0] == "L":  # L-form inputs (lf)
            for p in digest_pos:
                bits.append(node([spec_on(th[1][p], CHI_L)]))
            return bits
        for p in digest_pos:
            lits = chi_lits(th, *p)
            if flag(*p):
                lits = [neg(lits[0])] + lits[1:]
            bits.append(node([spec(lits, CHI)]))
        return bits

    # split theta for step 0 reads literals: E = a + D with D an xor of message bits
    def theta1_split_x(a, carry):
        memo_e, e, donly, gby = {}, {}, [], {}
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
            key = (aterm, dsigs)
            if g1 and key not in memo_e:
                if aterm is None:
                    donly.append(key)
                    memo_e[key] = None  # resolved below from a G of the same column pair
                else:  # G/4 = (a + 2D)/4
                    specs = [spec_on(aterm, COPY, 0.25)]
                    if dsigs:
                        n = len(dsigs)
                        for u in ([COPY] if n == 1 else par_units(n)):
                            specs.append(({s: u.weights[i] for i, s in enumerate(dsigs) if u.weights[i]}, u.bias,
                                          {s: u.value_weights[i] * 0.5 for i, s in enumerate(dsigs) if u.value_weights[i]},
                                          u.value_bias * 0.5))
                    gnode = node(specs, numeric=True)
                    memo_e[key] = ("gx", gnode)
                    gby.setdefault(dsigs, gnode)
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
                if pd1 and aterm is None:  # E = D in {0,1}: packed in pairs below
                    memo_e[key] = ("D", specs)
                    donly.append(key)
                else:
                    memo_e[key] = node(specs, numeric=True)
            e[p] = (key, flip)
        if g1:  # D = bit 1 of any G = a + 2D of the same column pair
            for key in donly:
                if key[1] in gby:
                    memo_e[key] = ("pk2", gby[key[1]], 1)
                else:  # no message bit in the column: E = D as before
                    specs = []
                    n = len(key[1])
                    for u in ([COPY] if n == 1 else par_units(n)):
                        specs.append(({s: u.weights[i] for i, s in enumerate(key[1]) if u.weights[i]}, u.bias,
                                      {s: u.value_weights[i] * 0.5 for i, s in enumerate(key[1]) if u.value_weights[i]},
                                      u.value_bias * 0.5))
                    memo_e[key] = node(specs, numeric=True)
            donly = []
        for i in range(0, len(donly) - 1, 2):  # f = D0 + 2 D1 as f/4 (specs are at E/2)
            k0, k1 = donly[i], donly[i + 1]
            f = node(scale_specs(memo_e[k0][1], 0.5) + scale_specs(memo_e[k1][1], 1.0), numeric=True)
            memo_e[k0], memo_e[k1] = ("pk2", f, 0), ("pk2", f, 1)
        if len(donly) % 2:
            memo_e[donly[-1]] = node(memo_e[donly[-1]][1], numeric=True)
        e = {p: (memo_e[key], flip) for p, (key, flip) in e.items()}
        return e, carry

    def xof_fn(msg: list) -> list:
        ndig["n"] = 0
        lanes = initial_state(k, msg)
        needs = [all_pos] * (depth - 1) + [digest_pos]  # chi outputs each step needs
        carry: list = []
        if kinds[0] == "split":
            e, carry = theta1_split_x(lanes, carry)
            th, carry = theta_split_y(e, carry)
        else:
            th = theta1_direct(lanes, chi_needs(needs[0]))
        step = 0
        while step < depth - 1:  # chi layer of step, then the theta layer(s) of step + 1
            nxt = kinds[step + 1]
            if walsh and step == depth - 2:  # chi layer, then the merged last round
                packed, carry = chi_to_direct(th, chi_needs(digest_pos), carry)
                return walsh_last(packed, carry)
            if nxt == "split":
                bits, sums, carry = chi_to_split(th, carry)
                e, carry = theta_split_x(bits, sums, carry)
                th, carry = theta_split_y(e, carry)
            elif nxt == "lazy":  # X (exact E), lazy chi, then the next step's theta
                assert step + 2 <= depth - 1, "a lazy step needs a step after it"
                bits, sums, carry = chi_to_split(th, carry)
                e, carry = theta_split_x_exact(bits, sums, carry)
                packed, carry = lazy_chi(e, chi_needs(needs[step + 2]), carry)
                if walsh and step + 2 == depth - 1:
                    return walsh_last(packed, carry)
                th, carry = theta_direct(packed, carry, last=step + 2 == depth - 1)
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


def make(kinds_fn, pairs=True, walsh=False, m1=2, **opts):
    def variant(k, depth):
        return (build(k, depth, kinds_fn(depth), pairs, walsh, m1, **opts),
                {"msg": Bits("0" * k.msg_len).bitlist})
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


# ---- wave-2 avenue "packing": narrower interfaces on the frontier layouts ----
def _sfl(d):
    return ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d


def _sfm(d):
    return ["split"] * (d - 1) + ["direct"]


ALL = dict(pa=True, dy=True, pd1=True, lf=True)
# baselines rebuilt by this module (must match xs3 exactly)
lazy4c_middle_m3_mp_base = with_minpar(make(middle("lazy4c"), m1=3))
split_first_middle_mp_base = with_minpar(make(_sfm))
# depth 6
lazy4c_middle_m3_mp_pa = with_minpar(make(middle("lazy4c"), m1=3, pa=True))
lazy4c_middle_m3_mp_lf = with_minpar(make(middle("lazy4c"), m1=3, lf=True))
lazy4c_middle_m3_mp_pk = with_minpar(make(middle("lazy4c"), m1=3, **ALL))
# depth 7
split_first_lazy4c_m3_mp_pk = with_minpar(make(_sfl, m1=3, **ALL))
# depth 8
split_first_middle_mp_pa = with_minpar(make(_sfm, pa=True))
split_first_middle_mp_dy = with_minpar(make(_sfm, dy=True))
split_first_middle_mp_pd1 = with_minpar(make(_sfm, dy=True, pd1=True))
split_first_middle_mp_pk = with_minpar(make(_sfm, **ALL))
split_first_middle_m3_mp_pk = with_minpar(make(_sfm, m1=3, **ALL))
split_first_middle_m3m2_mp_pk = with_minpar(make(_sfm, m1=3, m2=2, **ALL))
split_first_middle_m4m2_mp_pk = with_minpar(make(_sfm, m1=4, m2=2, **ALL))
ALLC = dict(pa=True, dy=True, pd1=True, lf=True, pc=True)
lazy4c_middle_m3_mp_pkc = with_minpar(make(middle("lazy4c"), m1=3, **ALLC))
split_first_lazy4c_m3_mp_pkc = with_minpar(make(_sfl, m1=3, **ALLC))
split_first_middle_m3m2_mp_pkc = with_minpar(make(_sfm, m1=3, m2=2, **ALLC))
split_first_middle_mp_pkc = with_minpar(make(_sfm, **ALLC))
split_first_middle_mp_nopa = with_minpar(make(_sfm, dy=True, pd1=True, lf=True))
# sparse-leaning points: only the width cuts that add no nonzeros per unit (dy, pd1)
split_first_lazy4c_m3_mp_dp = with_minpar(make(_sfl, m1=3, dy=True, pd1=True))
split_first_middle_mp_dp = with_minpar(make(_sfm, dy=True, pd1=True))
split_first_middle_m3m2_mp_dp = with_minpar(make(_sfm, m1=3, m2=2, dy=True, pd1=True))
ALLC5 = dict(ALLC, d2p5=True)
lazy4c_middle_m3_mp_pkc5 = with_minpar(make(middle("lazy4c"), m1=3, **ALLC5))
split_first_lazy4c_m3_mp_pkc5 = with_minpar(make(_sfl, m1=3, **ALLC5))
lazy4c_middle_m3_mp_pc = with_minpar(make(middle("lazy4c"), m1=3, pa=True, pc=True))
split_first_lazy4c_m3_mp_pc = with_minpar(make(_sfl, m1=3, pa=True, pc=True, dy=True, pd1=True))
split_first_middle_m3m2_mp_pc = with_minpar(make(_sfm, m1=3, m2=2, pa=True, pc=True, dy=True, pd1=True))
ALLG = dict(pa=True, dy=True, lf=True, pc=True, g1=True)
split_first_middle_m3m2_mp_pkg = with_minpar(make(_sfm, m1=3, m2=2, **ALLG))
split_first_lazy4c_m3_mp_pkg5 = with_minpar(make(_sfl, m1=3, d2p5=True, **ALLG))
split_first_middle_mp_dg = with_minpar(make(_sfm, dy=True, g1=True))
split_first_lazy4c_m3_mp_dg = with_minpar(make(_sfl, m1=3, dy=True, g1=True))
split_first_middle_m3m2_mp_pg = with_minpar(make(_sfm, m1=3, m2=2, pa=True, pc=True, dy=True, g1=True))
split_first_lazy4c_m3_mp_pg = with_minpar(make(_sfl, m1=3, pa=True, pc=True, dy=True, g1=True))
split_first_lazy4c_m3_mp_pkg = with_minpar(make(_sfl, m1=3, **ALLG))
split_first_middle_mp_pag = with_minpar(make(_sfm, pa=True, dy=True, g1=True))
split_first_middle_mp_pcg = with_minpar(make(_sfm, pa=True, pc=True, dy=True, g1=True))
split_first_lazy4c_m3_mp_pag = with_minpar(make(_sfl, m1=3, pa=True, dy=True, g1=True))
split_first_middle_m6m2_mp_pkg = with_minpar(make(_sfm, m1=6, m2=2, s6=True, **ALLG))
split_first_lazy4c_m6_mp_pkg5 = with_minpar(make(_sfl, m1=6, d2p5=True, s6=True, **ALLG))
lazy4c_middle_m6_mp_pkc5 = with_minpar(make(middle("lazy4c"), m1=6, s6=True, **ALLC5))
split_first_middle_m4m2_mp_pkg4 = with_minpar(make(_sfm, m1=4, m2=2, s4=True, **ALLG))
split_first_lazy4c_m4_mp_pkg54 = with_minpar(make(_sfl, m1=4, d2p5=True, s4=True, **ALLG))
lazy4c_middle_m4_mp_pkc54 = with_minpar(make(middle("lazy4c"), m1=4, s4=True, **ALLC5))
split_first_middle_m4m2_mp_pg4 = with_minpar(make(_sfm, m1=4, m2=2, s4=True, pa=True, pc=True, dy=True, g1=True))
split_first_lazy4c_m4_mp_pg4 = with_minpar(make(_sfl, m1=4, s4=True, pa=True, pc=True, dy=True, g1=True))
lazy4c_middle_m4_mp_pc4 = with_minpar(make(middle("lazy4c"), m1=4, s4=True, pa=True, pc=True))
split_first_lazy4c_m4_mp_pg54 = with_minpar(make(_sfl, m1=4, s4=True, d2p5=True, pa=True, pc=True, dy=True, g1=True))
lazy4c_middle_m4_mp_pc54 = with_minpar(make(middle("lazy4c"), m1=4, s4=True, d2p5=True, pa=True, pc=True))

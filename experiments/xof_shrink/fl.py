"""XOF-structured 1-round Keccak, general layout (avenue xof-structure).

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
# exact decoder of two lazy values packed as p = o0 + 5 o1 (o in [0,4]) into b0 + 2 b1,
# b = parity(o): piecewise quadratic with knots at integers (decode.py)
L5DEC = synth_decoder([(p % 5) % 2 + 2 * ((p // 5) % 2) for p in range(25)])


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


# Parity of an integer s in [0, n], n odd >= 9, with (n - 1) / 2 units instead of glu_xor's
# (n + 1) / 2: brainstorm-critic's exact 4-unit parity on [0, 9] (two knots at integers, two in
# gaps 0.378 away from the lattice; effective slope at the lattice 2, as glu_xor), extended by
# -4 max(0, s - k) for odd k = 9, 11, ... (each shifts the last parabola (s - k + 1)^2 by 2).
# Unit (sig, b, p, r): max(0, sig (s - b)) (p s + r).
MP9 = [(1, 2.378180022890604, -1.4999999999998495, 10.932729965663313),
       (-1, 6.561552812808816, -2.00000000000016, 4.876894374383024),
       (1, 5.0, 2.499999999999856, -17.99999999999968),
       (-1, 4.0, 3.000000000000147, -8.000000000001078)]


def minpar_terms(n: int):
    """[(sig, b, p, r)] for parity on [0, n] (n odd >= 9)"""
    assert n % 2 == 1 and n >= 9
    return MP9 + [(1, float(2 * j + 1), 0.0, -4.0) for j in range(4, (n - 1) // 2)]


def lat_scale(b: float) -> float:
    """gate scale so that the nearest lattice point has |gate| >= 1 (silu ~ relu there)"""
    d = abs(b - round(b))
    return 1.0 if d < 1e-9 else 1.0 / d


def parity_units_mp(n: int, scale: float) -> list[Unit]:
    """parity of s in [0, n] on f = s / scale: min-parity for odd n >= 9, else glu_xor"""
    if n % 2 == 0 or n < 9:
        return parity_units(n, scale)
    out = []
    for sig, b, p, r in minpar_terms(n):
        lam = lat_scale(b)
        out.append(Unit((sig * lam * scale,), -sig * lam * b, (p * scale / lam,), r / lam))
    return out


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
          opts=None):
    """opts (first-last avenue): "dedupe_y": one theta node per distinct E in round 1's Y layer;
    "no_p2": the chi layer before an X layer emits no column-pair counts, D reads the 10 chi
    bits directly; "d2p5": lazy digest bits packed two per feature in base 5 and decoded to
    exact pairs in the next layer.
    kinds[i] in ("direct", "split", "shared") is the theta layout of step i ("shared"
    only after the first step); the last step is direct (one theta bit per column pair
    left, nothing to share), or merged with its chi into one layer if walsh."""
    w = k.w
    (rc,) = k.get_round_constants()
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[: k.d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]
    kinds = list(kinds or ["direct"] * depth)
    kinds[-1] = "direct"
    opts = dict(opts or {})
    aslot: dict = {}  # packed state bits: position -> slot in its pair feature
    eslot: dict = {}  # packed pure D values of round 1: (pair node, slot) -> key
    kslot: dict = {}
    pslot: dict = {}  # position -> slot of its packed pure D value
    colinfo: dict = {}  # pack_col: position of y = 4 -> (column sum, pair (y0,y1), pair (y2,y3))

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
            elif kind == "lazy4":  # lazy digest bit o/2, o in [0,4]: exact bit = parity(o)
                lazy4.append((sig, f))
            elif kind == "l5pair":  # (o0 + 5 o1) / 32 -> exact pair (b0 + 2 b1) / 4
                const, units = L5DEC
                specs = [({sig: 32.0}, float(-kk), {sig: float(b) * 32 * 0.25} if b else {}, float(a) * 0.25)
                         for kk, a, b in units]
                if const:
                    specs.append(({}, 1.0, {}, float(const) * 0.25))
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

    def carry_new(specs_flags, m=None):
        """carry items for new digest bits given as (unit spec(s) at coef 1, flag) pairs;
        a bit may be one spec or a list of specs (unpacked state bits); m: bits per feature"""
        m = m1 if m is None else m
        def sc(s, c):
            ss = s if isinstance(s, list) else [s]
            return [(u[0], u[1], {q: v * c for q, v in u[2].items()}, u[3] * c) for u in ss]
        items = []
        if pairs and m >= 3:
            for i in range(0, len(specs_flags), m):
                chunk = specs_flags[i:i + m]
                scl = pack_scale(2, len(chunk))
                specs = []
                for j, (s_, _) in enumerate(chunk):
                    specs += sc(s_, 2 ** j / scl)
                items.append(("pk", node(specs, numeric=True), tuple(f for _, f in chunk)))
        elif pairs:
            for i in range(0, len(specs_flags) - 1, 2):
                (s0, f0), (s1, f1) = specs_flags[i], specs_flags[i + 1]
                items.append(("pair", node(sc(s0, 0.25) + sc(s1, 0.5), numeric=True), (f0, f1)))
            if len(specs_flags) % 2:
                s, f = specs_flags[-1]
                items.append(("bit", node(sc(s, 1.0)), f))
        else:
            for s, f in specs_flags:
                items.append(("bit", node(sc(s, 1.0)), f))
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

    def theta_direct(packed, carry):
        pu = parity_units_mp if opts.get("mp_last") else parity_units
        if opts.get("gate_out") and len(packed) < len(all_pos):  # the last theta layer
            # emit the chi gate G = 2 t_a - t_b + t_c (iota: t_a -> 1 - t_a) of each digest
            # bit, a free sum of the parity units, instead of the 320 theta bits: 224 features
            gates = {}
            for p in digest_pos:
                x, y, z = p
                qs = [rp[(x + dx) % 5][y][z] for dx in range(3)]
                ws = [2.0, -1.0, 1.0]
                fl_ = [bool(packed[q][1]) for q in qs]
                if flag(*p):
                    fl_[0] = not fl_[0]
                specs, const = [], 0.0
                for q, w_, f_ in zip(qs, ws, fl_):
                    sig, _, n = packed[q]
                    c = 0.25 * w_ * (1 - 2 * f_)  # G emitted as G/4
                    specs += [spec_on(sig, u, c) for u in pu(n, SCALE)]
                    const += 0.25 * w_ * f_
                if const:
                    specs.append(({}, 1.0, {}, const))
                gates[p] = node(specs, numeric=True)
            return ("G", gates), carry_layer(carry)
        th = {p: (G([sig], pu(n, SCALE)), bool(f)) for p, (sig, f, n) in packed.items()}
        return th, carry_layer(carry)

    def d_specs(psig, n, coef):
        """coef * parity(P): P a count feature (P/SCALE), a tuple of n bit features, or
        ("cols", C_left, C_right): two column-sum features C/8 (pack_col)"""
        if isinstance(psig, tuple) and psig and psig[0] == "cols":
            _, cl, cr = psig
            out = []
            for u in parity_units(n, 1.0):
                g = {cl: 8.0 * u.weights[0], cr: 8.0 * u.weights[0]}
                v = {cl: 8.0 * u.value_weights[0] * coef, cr: 8.0 * u.value_weights[0] * coef} if u.value_weights[0] else {}
                out.append((g, float(u.bias), v, float(u.value_bias) * coef))
            return out
        if isinstance(psig, tuple):
            out = []
            for u in xor_units(n):
                g = {q: float(wg) for q, wg in zip(psig, u.weights) if wg}
                v = {q: float(wv) * coef for q, wv in zip(psig, u.value_weights) if wv}
                out.append((g, float(u.bias), v, float(u.value_bias) * coef))
            return out
        return [spec_on(psig, u, coef) for u in parity_units(n, SCALE)]

    # state bits packed in pairs f = (a0 + 2 a1) / 4 (opts "pack_a"): the X layer unpacks them
    # with the same number of units as copies: a1 = max(0, p - 1)(2 - p/2), a0 = max(0, p) - 2 a1
    UNP1 = Unit((4,), 0, (0,), 1)  # on f = p/4: max(0, p) * 1
    UNP2 = Unit((4,), -1, (-2,), 2)  # max(0, p - 1) * (2 - p/2)

    def a_specs(bits, p, coef):
        """specs of coef * a (the raw state bit at p, flag not applied)"""
        sig = bits[p][0]
        slot = aslot.get(p)
        if slot is None:
            return [spec_on(sig, COPY, coef)]
        if slot == 0:
            return [spec_on(sig, UNP1, coef), spec_on(sig, UNP2, -2 * coef)]
        if slot == "c":  # a4 = C - (a0 + a1) - (a2 + a3), a0 + a1 = max(0,p) - max(0,p-1)(2-p/2)
            cn, f01, f23 = colinfo[p]
            out = [({}, 1.0, {cn: 8.0 * coef}, 0.0)]  # pass unit (gate on BOS): C exactly
            for fq in (f01, f23):
                out += [spec_on(fq, UNP1, -coef), spec_on(fq, UNP2, coef)]
            return out
        return [spec_on(sig, UNP2, coef)]

    def theta_split_x(bits, sums, carry):
        """E = a + D per position (numeric, E/2), D = parity(P) shared by the column pair;
        this step's digest carried from the copy units of a"""
        e = {}
        for p in all_pos:
            (asig, af), (psig, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = a_specs(bits, p, 0.5) + d_specs(psig, n, 0.5)
            e[p] = (node(specs, numeric=True), af ^ pf)
        new = [(a_specs(bits, p, 1.0), bits[p][1]) for p in digest_pos]
        return e, carry_layer(carry) + carry_new(new)

    def theta_split_x_exact(bits, sums, carry):
        """E = a + D with the flags folded in (E/2, no flag), so that chi can read E"""
        e = {}
        for p in all_pos:
            (asig, af), (psig, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = a_specs(bits, p, 0.5 * (1 - 2 * af))
            specs += d_specs(psig, n, 0.5 * (1 - 2 * pf))
            if af or pf:
                specs.append(({}, 1, {}, 0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [(a_specs(bits, p, 1.0), bits[p][1]) for p in digest_pos]
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
        if opts.get("d2p5"):  # two lazy values o in [0,4] per feature: (o0 + 5 o1) / 32
            for i in range(0, len(digest_pos) - 1, 2):
                p0, p1 = digest_pos[i], digest_pos[i + 1]
                sp0, pas0 = own(p0, 1 / 32)
                sp1, pas1 = own(p1, 5 / 32)
                pas = dict(pas0)
                for s_, w_ in pas1.items():
                    pas[s_] = pas.get(s_, 0) + w_
                pas = {s_: w_ for s_, w_ in pas.items() if abs(w_) > 1e-12}
                new.append(("l5pair", node(sp0 + sp1 + [({}, 1, pas, 0.0)], numeric=True),
                            (flag(*p0), flag(*p1))))
            if len(digest_pos) % 2:  # odd digest (small widths): the last value alone, last
                sp, pas = own(digest_pos[-1], 0.5)
                new.append(("lazy4", node(sp + [({}, 1, pas, 0.0)], numeric=True), flag(*digest_pos[-1])))
            return packed, carry_layer(carry) + new
        for p in digest_pos:
            sp, pas = own(p, 0.5)
            new.append(("lazy4", node(sp + [({}, 1, pas, 0.0)], numeric=True), flag(*p)))
        return packed, carry_layer(carry) + new

    def theta_split_y(e, carry):
        if opts.get("dedupe_y"):  # equal E features (round 1) give one theta node
            memo, th = {}, {}
            for p, (sig, f) in e.items():
                key = (sig, pslot.get(p) if (sig, 0) in eslot else None)
                if key not in memo:
                    if key[1] is None:
                        memo[key] = G([sig], [Unit((2,), 0, (-2,), 2)])
                    elif key[1] == 0:  # unpack a pure D value (a bit): lo = max(0,p) - 2 max(0,p-1)(2-p/2)
                        memo[key] = node([spec_on(sig, UNP1, 1.0), spec_on(sig, UNP2, -2.0)])
                    else:
                        memo[key] = node([spec_on(sig, UNP2, 1.0)])
                th[p] = (memo[key], bool(f))
            return th, carry_layer(carry)
        th = {p: (G([sig], [Unit((2,), 0, (-2,), 2)]), bool(f)) for p, (sig, f) in e.items()}
        return th, carry_layer(carry)

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
        chi = {p: chi_lits(th, *p) for p in all_pos}
        packed = {}
        for p in next_pos:
            qs = [p] + pair_of(p)
            f = sum(flag(*q) for q in qs) % 2
            packed[p] = (node([spec(chi[q], CHI, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        new = [(spec(chi[p], CHI), flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + carry_new(new, opts.get("m_last"))

    def chi_to_split(th, carry):
        """chi bits (all positions) and column-pair counts P for a split theta"""
        chi = {p: chi_lits(th, *p) for p in all_pos}
        colsum = {}
        if opts.get("pack_col"):  # per column: pairs (y0, y1), (y2, y3) and the column sum C/8
            bits = {}
            for x in range(5):
                for z in range(w):
                    ps = [(x, y, z) for y in range(5)]
                    cn = node([spec(chi[q], CHI, 1 / 8) for q in ps], numeric=True)
                    colsum[(x, z)] = cn
                    prs = []
                    for a0, a1 in ((ps[0], ps[1]), (ps[2], ps[3])):
                        pn = node([spec(chi[a0], CHI, 0.25), spec(chi[a1], CHI, 0.5)], numeric=True)
                        bits[a0], bits[a1] = (pn, flag(*a0)), (pn, flag(*a1))
                        aslot[a0], aslot[a1] = 0, 1
                        prs.append(pn)
                    bits[ps[4]] = (cn, flag(*ps[4]))
                    aslot[ps[4]] = "c"
                    colinfo[ps[4]] = (cn, prs[0], prs[1])
        elif opts.get("pack_a"):  # pairs (a0 + 2 a1) / 4, free sums of the chi units
            bits = {}
            for i in range(0, len(all_pos), 2):
                p0, p1 = all_pos[i], all_pos[i + 1]
                pn = node([spec(chi[p0], CHI, 0.25), spec(chi[p1], CHI, 0.5)], numeric=True)
                bits[p0], bits[p1] = (pn, flag(*p0)), (pn, flag(*p1))
                aslot[p0], aslot[p1] = 0, 1
        else:
            bits = {p: (node([spec(chi[p], CHI)]), flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                if opts.get("pack_col"):  # D reads the two column sums
                    sums[(x, z)] = (("cols", colsum[((x + 4) % 5, z)], colsum[((x + 1) % 5, (z + 1) % w)]), f, len(qs))
                elif opts.get("no_p2"):  # D reads the 10 chi bits (raw, flags in f) directly
                    sums[(x, z)] = (tuple(bits[q][0] for q in qs), f, len(qs))
                else:
                    sums[(x, z)] = (node([spec(chi[q], CHI, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
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
        if isinstance(th, tuple) and th[0] == "G":  # chi = max(0, G)(3 - G)/2 on g = G/4
            for p in digest_pos:
                bits.append(node([({th[1][p]: 4.0}, 0.0, {th[1][p]: -2.0}, 1.5)]))
            return bits
        for p in digest_pos:
            lits = chi_lits(th, *p)
            if flag(*p):
                lits = [neg(lits[0])] + lits[1:]
            bits.append(node([spec(lits, CHI)]))
        return bits

    # split theta for step 0 reads literals: E = a + D with D an xor of message bits
    def theta1_split_x(a, carry):
        memo_e, e, pure = {}, {}, []
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
                if opts.get("pack_d1") and aterm is None and len(dsigs) > 1:
                    memo_e[key] = ("pure", specs)  # E = D, a bit: packed in pairs below
                    pure.append(key)
                else:
                    memo_e[key] = node(specs, numeric=True)
            e[p] = (key, flip)
        # pure D values (theta of zero lanes) in pairs (D0 + 2 D1)/4: the Y layer unpacks them
        for i in range(0, len(pure) - 1, 2):
            k0, k1 = pure[i], pure[i + 1]
            sc = lambda sp, c: [(g, gb, {q: v * c for q, v in vd.items()}, vb * c) for g, gb, vd, vb in sp]
            pn = node(sc(memo_e[k0][1], 0.5) + sc(memo_e[k1][1], 1.0), numeric=True)  # specs are at 0.5
            memo_e[k0], memo_e[k1] = pn, pn
            eslot[pn, 0], eslot[pn, 1] = k0, k1
            kslot[k0], kslot[k1] = 0, 1
        if len(pure) % 2:
            memo_e[pure[-1]] = node(memo_e[pure[-1]][1], numeric=True)
        for p, (key, flip) in e.items():
            if key in kslot:
                pslot[p] = kslot[key]
        e = {p: (memo_e[key], flip) for p, (key, flip) in e.items()}
        return e, carry

    def xof_fn(msg: list) -> list:
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
                th, carry = theta_direct(packed, carry)
                step += 1
            elif nxt in ("lazy4", "lazy4c"):  # X (exact E), one-unit lazy chi, next theta
                assert step + 2 <= depth - 1, "a lazy step needs a step after it"
                bits, sums, carry = chi_to_split(th, carry)
                e, carry = theta_split_x_exact(bits, sums, carry)
                packed, carry = (lazy_chi4 if nxt == "lazy4" else lazy_chi4c)(e, chi_needs(needs[step + 2]), carry)
                th, carry = theta_direct(packed, carry)
                step += 1
            elif nxt == "shared":
                bits, sums, carry = chi_to_shared(th, carry)
                th, carry = theta_shared(bits, sums, carry)
            else:
                packed, carry = chi_to_direct(th, chi_needs(needs[step + 1]), carry)
                th, carry = theta_direct(packed, carry)
            step += 1
        return chi_last(th, carry)

    return xof_fn


def make(kinds_fn, pairs=True, walsh=False, m1=2, opts=None):
    def variant(k, depth):
        return build(k, depth, kinds_fn(depth), pairs, walsh, m1, opts), {"msg": Bits("0" * k.msg_len).bitlist}
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


# ---- first-last avenue variants ----
def _sfm(d):
    return ["split"] * (d - 1) + ["direct"]


def _sfl(d):
    return ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d


def fl_variant(kinds_fn, m1=2, mp=True, **opts):
    v = make(kinds_fn, m1=m1, opts=opts)
    return with_minpar(v) if mp else v


d8_dd = fl_variant(_sfm, dedupe_y=True)
d8_np = fl_variant(_sfm, no_p2=True)
d8_dd_np = fl_variant(_sfm, dedupe_y=True, no_p2=True)
d8_dd_m3 = fl_variant(_sfm, m1=3, dedupe_y=True)
d8_dd_np_m3 = fl_variant(_sfm, m1=3, dedupe_y=True, no_p2=True)
d7_dd = fl_variant(_sfl, m1=3, dedupe_y=True)
d7_dd_p5 = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True)
d7_dd_p5_np = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True, no_p2=True)
d6_p5 = fl_variant(middle("lazy4c"), m1=3, d2p5=True)
d6_p5_np = fl_variant(middle("lazy4c"), m1=3, d2p5=True, no_p2=True)

d8_dd_pk = fl_variant(_sfm, dedupe_y=True, pack_a=True)
d8_dd_pk_m3 = fl_variant(_sfm, m1=3, dedupe_y=True, pack_a=True)
d7_dd_pk = fl_variant(_sfl, m1=3, dedupe_y=True, pack_a=True)
d7_dd_p5_pk = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True, pack_a=True)
d6_p5_pk = fl_variant(middle("lazy4c"), m1=3, d2p5=True, pack_a=True)
d6_pk = fl_variant(middle("lazy4c"), m1=3, pack_a=True)

d8_dd_pk_d1 = fl_variant(_sfm, dedupe_y=True, pack_a=True, pack_d1=True)
d8_dd_pk_d1_m3 = fl_variant(_sfm, m1=3, dedupe_y=True, pack_a=True, pack_d1=True)
d7_dd_pk_d1 = fl_variant(_sfl, m1=3, dedupe_y=True, pack_a=True, pack_d1=True)
d7_dd_p5_pk_d1 = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True, pack_a=True, pack_d1=True)

d8_best = fl_variant(_sfm, dedupe_y=True, pack_a=True, pack_d1=True, mp_last=True)
d8_best_m3 = fl_variant(_sfm, m1=3, dedupe_y=True, pack_a=True, pack_d1=True, mp_last=True)
# sparse-leaning: no min-parity (glu_xor everywhere: fewer nonzeros per unit)
d8_sp = fl_variant(_sfm, mp=False, dedupe_y=True, pack_a=True, pack_d1=True)
d8_sp_dd = fl_variant(_sfm, mp=False, dedupe_y=True)
d7_best = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True, pack_a=True, pack_d1=True)
d7_sp = fl_variant(_sfl, m1=3, mp=False, dedupe_y=True, pack_a=True, pack_d1=True)

d8_col = fl_variant(_sfm, m1=3, dedupe_y=True, pack_col=True, pack_d1=True, mp_last=True)
d7_col = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True, pack_col=True, pack_d1=True)
d6_col = fl_variant(middle("lazy4c"), m1=3, d2p5=True, pack_col=True)
d8_col_sp = fl_variant(_sfm, m1=3, mp=False, dedupe_y=True, pack_col=True, pack_d1=True)
d8_col_sp2 = fl_variant(_sfm, mp=False, dedupe_y=True, pack_col=True, pack_d1=True)
d7_col_sp = fl_variant(_sfl, m1=3, mp=False, dedupe_y=True, pack_col=True, pack_d1=True)
d8_col_m2 = fl_variant(_sfm, dedupe_y=True, pack_col=True, pack_d1=True, mp_last=True)
d7_sp_m2 = fl_variant(_sfl, mp=False, dedupe_y=True, pack_a=True, pack_d1=True)
d7_sp_m2_nod1 = fl_variant(_sfl, mp=False, dedupe_y=True, pack_a=True)
d8_col_nompl = fl_variant(_sfm, m1=3, dedupe_y=True, pack_col=True, pack_d1=True)
d6_col_nop5 = fl_variant(middle("lazy4c"), m1=3, pack_col=True)
d6_col_sp = fl_variant(middle("lazy4c"), m1=3, mp=False, pack_col=True)
d6_pk_sp = fl_variant(middle("lazy4c"), m1=3, mp=False, pack_a=True)
d7_col_sp_m2 = fl_variant(_sfl, mp=False, dedupe_y=True, pack_col=True, pack_d1=True)
d7_col_m2 = fl_variant(_sfl, dedupe_y=True, pack_col=True, pack_d1=True)
d7_col_p5_m2 = fl_variant(_sfl, dedupe_y=True, d2p5=True, pack_col=True, pack_d1=True)
d6_pk_sp_m2 = fl_variant(middle("lazy4c"), mp=False, pack_a=True)
d6_col_sp_m2 = fl_variant(middle("lazy4c"), mp=False, pack_col=True)
d6_col_m2 = fl_variant(middle("lazy4c"), pack_col=True)

d8_col_ml2 = fl_variant(_sfm, m1=3, dedupe_y=True, pack_col=True, pack_d1=True, mp_last=True, m_last=2)
d8_col_sp_ml2 = fl_variant(_sfm, m1=3, mp=False, dedupe_y=True, pack_col=True, pack_d1=True, m_last=2)
d8_col_ml2_nompl = fl_variant(_sfm, m1=3, dedupe_y=True, pack_col=True, pack_d1=True, m_last=2)

d8_g = fl_variant(_sfm, m1=3, dedupe_y=True, pack_col=True, pack_d1=True, mp_last=True, m_last=2, gate_out=True)
d7_g = fl_variant(_sfl, m1=3, dedupe_y=True, d2p5=True, pack_col=True, pack_d1=True, gate_out=True)
d6_g = fl_variant(middle("lazy4c"), m1=3, d2p5=True, pack_col=True, gate_out=True)
d8_g_nompl = fl_variant(_sfm, m1=3, dedupe_y=True, pack_col=True, pack_d1=True, m_last=2, gate_out=True)
d8_g_sp = fl_variant(_sfm, mp=False, dedupe_y=True, pack_col=True, pack_d1=True, gate_out=True)
d7_g_m2 = fl_variant(_sfl, dedupe_y=True, pack_col=True, pack_d1=True, gate_out=True)
d7_g_sp = fl_variant(_sfl, mp=False, dedupe_y=True, pack_col=True, pack_d1=True, gate_out=True)
d6_g_m2 = fl_variant(middle("lazy4c"), pack_col=True, gate_out=True)

"""depth-low (wave 2): shallow XOF layouts, built on xof-structure's xs4.py.

New layouts (see NOTES.md):
  "d4x": round 1 fused on raw message bits (lazy chi1 o' = t_a + (1 - t_b) t_c in {0,1,2})
         | X2 from the lazy values (A = [o' == 1] exact, D = parity of the column pair's
         sum of o', E2 = A + D) | xs4's lazy chi2 with column parities | Walsh round 3.
  "d3":  round 1 fused on raw bits emitting theta2 counts <= 13 (exact column parities)
         | round 2 fused (xs4 lazy_round) | Walsh round 3.
Options: MPX (min-parity for odd raw parities, n <= MPX["max"]), CENTER (parities with
centred pieces: same units, ~4x smaller float32 cancellation), ENDS (zero-on-lattice units
that flatten count parities at their range ends).
This file lives under a /lib/python path on purpose: the reifier tracer does not record
calls from such files, which keeps tracing fast (the glu gates are still recorded).

--- xs4 docstring ---
Shallow XOF layouts: rounds computed in ONE SwiGLU layer (avenue xof-structure).

Notation: a theta bit is t = parity(S) (xor a flag), where S is a count feature that the
layer before emits for free (a sum of its units). chi(a, b, c) = a ^ (~b & c).

One-layer middle round ("lazy round"): the next theta only needs chi mod 2, so
    o' = t_a + (1 - t_b) t_c,   (1 - t_b) t_c = (t_c - t_b + (t_b ^ t_c)) / 2,
with t_x = parity(S_x) (glu_xor units on one feature, shared by the 3 chi bits that read
t_x) and t_b ^ t_c = parity(S_b + S_c) (glu_xor units on two features). o' is in {0,1,2}
and congruent to chi mod 2; the layer emits the counts of o' that the next theta needs.
One-layer last round (Walsh, as in xs3.walsh_last):
    chi = (p(S_a) + p(S_a + S_c) - p(S_a + S_b) + p(S_a + S_b + S_c)) / 2.
AND_CHI2: the AND of two adjacent chi bits of a row is one gated unit, so a count can hold
their xor (a + b - 2 AND) instead of their sum: counts of range 10 instead of 11.
Layouts (depth):
    "d4": theta1 | chi1 (+ counts S2) | lazy round 2 (+ counts S3) | Walsh round 3
    "d5": theta1 | chi1 (bits, column-pair counts) | X2 (E = a + D) | lazy chi2 with column
          parities (counts <= 13) | Walsh round 3
    "d6": as d5 with theta-structure's lazy chi2, then theta3 | chi3 (= xs3:lazy_middle)
Digests of earlier steps are packed several per feature (binary for exact bits, base 3
for lazy values in {0,1,2}) and decoded exactly in the last layer (decode.py).
A literal is an int 0/1 (constant) or (Bit, neg) meaning neg xor Bit.
"""

from math import ceil

import reifier.examples.keccak as K
from reifier.neurons.core import Unit, glu
from reifier.utils.format import Bits

from xs_u import CHI, COPY, keccak_maps, initial_state, theta_lits, xor_units
from xs3_u import spec, spec_on, node
from decode_u import check as synth_decoder

G = glu


MPX = {"on": False}
# MINPAR7 (unit-synthesis, exact on [0, 7]): (gate a, gate b, value p, value q) for
# max(0, a s + b)(p + q s), plus a constant. Extended to any odd n = 7 + 2m by the
# integer-knot units -4 max(0, s - 7 - 2j): the piece (s - 6)^2 on [5, 7] becomes
# (s - 8)^2, (s - 10)^2, ...; (n - 1)/2 units instead of glu_xor's (n + 1)/2 (exact
# rational check in mpcheck.py). Knots between integers: raw (exact) inputs only.
_MP7 = ([(-6, 18, -25 / 36, -19 / 72), (2, -3, -47 / 6, 4 / 3), (5, -23, 0.0, -1 / 3)], 25 / 2)


from minpar_counts import min_units as _min_units  # combined-2 (imported outside tracing)


P1D = {"on": False}  # and-of-parities (wave 3): fewer-unit parities with irrational knots


def _cur_count(n: int) -> int:
    if MPX["on"] and n >= 7 and n % 2 and n <= MPX.get("max", 99):
        return (n - 1) // 2
    return ceil(n / 2)


def raw_units(n: int) -> list:
    """parity units on n raw input bits (min-parity for odd n >= 7 if MPX is on;
    centred glu_xor pieces if CENTER is on; the p1d_forms table if P1D is on and it has
    fewer units)"""
    if P1D["on"]:
        from p1d_forms import FORMS as _F, FORMS_EXT, FORMS_EXTC
        # "ext": even n from the n = 6 form + glu_xor ramps (ramps have constant values: sparser);
        # "extc": the same with the n = 6 form in the middle (smaller cancelling terms)
        FORMS = {**_F, **(FORMS_EXTC if P1D.get("ext") == "c" else FORMS_EXT)} if P1D.get("ext") else _F
        if P1D.get("odd") and n % 2 == 1 and (n + 1) in FORMS_EXTC and n not in FORMS:
            FORMS = {**FORMS, n: FORMS_EXTC[n + 1]}  # the even form of range n + 1 is exact on [0, n]
        if P1D.get("odd") == "all" and n % 2 == 1 and n >= 7 and (n + 1) in FORMS_EXTC:
            # same unit count as min-parity, but only 2 units read all n bits in their values
            us, c = FORMS_EXTC[n + 1]
            out = [Unit((gw,) * n, gb, (vw,) * n, vb) for gw, gb, vw, vb in us]
            return out + [Unit((0,) * n, 1, (0,) * n, c)]
        if n in FORMS and len(FORMS[n][0]) < _cur_count(n) and n <= P1D.get("max", 99):
            us, c = FORMS[n]
            out = [Unit((gw,) * n, gb, (vw,) * n, vb) for gw, gb, vw, vb in us]
            return out + [Unit((0,) * n, 1, (0,) * n, c)]
    if CENTER["on"] and n >= 6 and not (MPX["on"] and n % 2 and 7 <= n <= MPX.get("max", 99)):
        units, const = centered(n)
        out = [Unit((gw,) * n, gb, (vw,) * n, vb) for gw, gb, vw, vb in units]
        return out + [Unit((0,) * n, 1, (0,) * n, const)]
    if MPX["on"] and n >= 7 and n % 2 and n <= MPX.get("max", 99):
        us, c = _MP7
        us = list(us) + [(1, -(7 + 2 * j), -4.0, 0.0) for j in range((n - 7) // 2)]
        out = [Unit((a,) * n, b, (q,) * n, p) for a, b, p, q in us]
        return out + [Unit((0,) * n, 1, (0,) * n, c)]
    import xs3_u as xs3
    return xs3.par_units(n)

# chi(t1,t2,t3) AND chi(t2,t3,t4), two adjacent chi bits of a row, in ONE gated unit
# (exact on all 16 inputs; found by an exhaustive gate search, s2/and2chi.py)
AND_CHI2 = Unit((4, 2, -1, 1), -4, (0.75, -1, 0.5, -0.5), 0.75)


def parity_units(n: int, scale: float) -> list[Unit]:
    """glu_xor units on one feature f = s / scale with s in [0, n]"""
    units = [Unit((scale,), 0, (-scale,), 2)]
    units += [Unit((scale,), -2 * j, (0,), 4) for j in range(1, ceil(n / 2))]
    return units


CENTER = {"on": False}


def centered(n: int):
    """parity of an integer s in [0, n] with the pieces centred at c = 2 floor(n/4), as
    (gate w, gate b, value w, value b) of max(0, w s + b)(vw s + vb) plus a constant:
    base (s + 1)(2c + 3 - s) - (c + 1)(c + 3) = -(s - c)(s - c - 2), the bump on [c, c + 2];
    right knots 4 max(0, s - k) (k = c+2, c+4, ... < n) and left knots 4 max(0, k - s)
    (k = c, c-2, ... >= 1) add the bumps on either side (knots on even integers, so it is
    flat at the lattice points like glu_xor). Same unit count as glu_xor (ceil(n/2)), but
    the cancelling terms are ~(n/2)^2 instead of ~n^2: 4x less float32 rounding."""
    c = 2 * (n // 4)
    units = [(1.0, 1.0, -1.0, float(2 * c + 3))]
    units += [(1.0, float(-k), 0.0, 4.0) for k in range(c + 2, n, 2)]
    units += [(-1.0, float(k), 0.0, 4.0) for k in range(c, 0, -2)]
    return units, float(-(c + 1) * (c + 3))


P1DK = {"on": False}  # wave 3 (and-of-parities): p1d_forms on count parities too
WMPC = {"on": False}  # combined-2: min-parity (minpar_counts) for odd count ranges in the Walsh layer


def parity_specs(feats, coef, mpc=False):
    """specs of coef * parity(sum of counts): feats = [(sig, scale, n)], one gate on all"""
    n = sum(nn for _, _, nn in feats)
    specs = []
    g = {}
    for sig, sc, _ in feats:
        g[sig] = g.get(sig, 0.0) + float(sc)
    if P1DK["on"]:  # wave 3: fewer-unit parity forms on counts (knots between integers)
        from p1d_forms import FORMS as _F, FORMS_FLAT, FORMS_EXT, FORMS_EXTT
        # counts carry noise: the flattest forms where there is one (EXT: flat beyond s = 6;
        # EXTT: flat below s = n - 6)
        FORMS = ({"t": FORMS_EXTT}.get(P1DK.get("ext"), FORMS_EXT) if P1DK.get("ext")
                 else {**_F, **FORMS_FLAT})
        if P1DK.get("odd") == "t1" and n % 2 == 1 and n >= 7 and (n + 1) in FORMS_EXTT:
            # odd range n: the even top-core form of range n + 1 is exact on [0, n] too,
            # with (n + 1)/2 - 1 = (n - 1)/2 units and its non-flat points at n - 5, n - 3, n - 1
            FORMS = {n: FORMS_EXTT[n + 1]}
        elif P1DK.get("odd") and n % 2 == 1 and n >= 7:  # odd count ranges: min-parity (MPC)
            us0, c0_ = _min_units(n)
            if P1DK["odd"] == "t":  # reflected, s -> n - s: its non-flat points move to the top
                # parity(s) = 1 - parity(n - s) for odd n
                us0 = [(-a, a * n + b, p, -(p * n + q)) for a, b, p, q in us0]
                c0_ = 1 - c0_
            FORMS = {n: (us0, c0_)}
        cur = (n - 1) // 2 if (mpc and n % 2 == 1 and n >= 7) else ceil(n / 2)
        if n in FORMS and len(FORMS[n][0]) < cur:
            us, c0 = FORMS[n]
            for gw, gb, vw, vb in us:
                specs.append(({q: gw * w for q, w in g.items()}, float(gb),
                              {q: vw * w * coef for q, w in g.items()} if vw else {}, float(vb) * coef))
            if c0:
                specs.append(({}, 1.0, {}, float(c0) * coef))
            return specs
    if mpc and n % 2 == 1 and n >= 7:  # (n - 1)/2 units instead of (n + 1)/2 (units-per-bit)
        us, c = _min_units(n)
        for a, b, pp, q in us:
            v = {sig: w * pp * coef for sig, w in g.items()} if pp else {}
            specs.append(({sig: w * a for sig, w in g.items()}, float(b), v, float(q) * coef))
        if c:
            specs.append(({}, 1.0, {}, float(c) * coef))
        return specs
    if CENTER["on"] and CENTER.get("counts", True) and n >= 6:
        units, const = centered(n)
        for gw, gb, vw, vb in units:
            specs.append(({q: gw * w for q, w in g.items()}, gb,
                          {q: vw * w * coef for q, w in g.items()} if vw else {}, vb * coef))
        specs.append(({}, 1.0, {}, const * coef))
        return specs
    for j, u in enumerate(parity_units(n, 1.0)):
        v = {sig: -w * coef for sig, w in g.items()} if j == 0 else {}
        specs.append((dict(g), float(u.bias), v, float(u.value_bias) * coef))
    if ENDS["on"]:
        # zero on the lattice (exact), but under silu they cancel the slope at the ends:
        # at s = 0 glu_xor's first knot averages to slope 1 -> 2 max(0, -s) makes it 0;
        # at s = n (even n) the last bump has slope -2 -> 4 max(0, s - n) makes it 0
        if ENDS.get("zero", True):
            specs.append(({q: -w for q, w in g.items()}, 0.0, {}, 2.0 * coef))
        if n % 2 == 0:
            specs.append((dict(g), float(-n), {}, 4.0 * coef))
    return specs


ENDS = {"on": False}


def const_spec(c):
    return ({}, 1.0, {}, float(c))


def pack_scale(base: int, n: int) -> float:
    """features are p / 2^e >= p / (base^n - 1): dyadic (exact eager values) and <= 1"""
    return float(2 ** (base ** n - 1).bit_length())


def digit_fns(base: int, n: int):
    """decoders of the n digits of p = sum_i base^i d_i (binary: bits; ternary: [d == 1])"""
    N = base ** n - 1
    fns = []
    for i in range(n):
        if base == 2:
            f = [(p >> i) & 1 for p in range(N + 1)]
        else:
            f = [int((p // base ** i) % base == 1) for p in range(N + 1)]
        fns.append(synth_decoder(f))
    return fns


def build(k: K.Keccak, depth: int, layout: str = "d4", s2: int = 16, s3: int = 32,
          m1: int = 2, k2: int = 1, and_left: bool = False, k1: int = 3, cols: bool = True,
          dig1: str = "lazy"):
    """s2, s3: count features are emitted as count / s (powers of 2, so that they stay <= 1).
    m1: digest-1 bits per carried feature (binary); k2: digest-2 lazy values per feature
    (base 3, o' in {0,1,2}); decoded in the last layer by exact piecewise-quadratic units.
    and_left: a theta count holds the own chi bit and its left neighbour (same row); adding
    -2 AND of the two (one unit) makes their part an exact xor, so counts are <= 10."""
    w = k.w
    (rc,) = k.get_round_constants()
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[: k.d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]
    # steps avenue (wave 4): d3 (dig1 "lazy") and d4x extend to S >= 3 steps with S - 3 more lazy
    # rounds (xs4.lazy_round) before the Walsh round: depth S and S + 1
    assert layout in ("d3", "d4x", "d4", "d5", "d6", "w1") and (depth == 3 or (
        depth > 3 and (layout == "d4x" or (layout == "d3" and dig1 == "lazy")))
        or (depth == 2 and layout == "d3" and dig1 == "lazy") or (depth == 1 and layout == "w1")), \
        "layouts of a 3-step XOF (steps avenue: d3 for S >= 2, d4x for S >= 3, w1 for S = 1)"

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

    def count_pos(p):  # the 11 positions whose sum is the theta count of p
        x, _, z = p
        return [p] + [((x + 4) % 5, y2, z) for y2 in range(5)] + [((x + 1) % 5, y2, (z + 1) % w) for y2 in range(5)]

    # ---- carries: ("pk", sig, flags, base, n): feature = sum_i base^i v_i / base^n ----
    def carry_layer(carry):
        """one unit per packed feature: relu(p) * 1 on p = scale * feature >= 0"""
        out = []
        for kind, sig, f, base, n in carry:
            sc = pack_scale(base, n)
            out.append((kind, G([sig], [Unit((sc,), 0, (0,), 1 / sc)], numeric=True), f, base, n))
        return out

    def pack(items, base, n):
        """items: [(unit specs at coef 1, flag)]; values sum_i base^i v_i / base^n"""
        out = []
        for i in range(0, len(items), n):
            chunk = items[i:i + n]
            nn = len(chunk)
            sc = pack_scale(base, nn)
            specs = []
            for j, (sp, _) in enumerate(chunk):
                c = base ** j / sc
                specs += [(g, gb, {q: v * c for q, v in vd.items()}, vb * c) for g, gb, vd, vb in sp]
            out.append(("pk", node(specs, numeric=True), tuple(f for _, f in chunk), base, nn))
        return out

    decoders = {}

    def decode(carry):
        """exact output bits from packed features; flips use one shared constant unit"""
        bits = []
        for kind, sig, flags, base, n in carry:
            if (base, n) not in decoders:
                decoders[(base, n)] = digit_fns(base, n)
            sc = pack_scale(base, n)
            for (const, units), f in zip(decoders[(base, n)], flags):
                s_ = 1 - 2 * f
                # relu(p - k) * (a + b p) on p = sc * feature
                specs = [({sig: sc}, float(-kk), {sig: float(b) * sc * s_} if b else {},
                          float(a) * s_) for kk, a, b in units]
                c = float(const) * s_ + f
                if c:
                    specs.append(const_spec(c))
                bits.append(node(specs))
        return bits

    # ---- layer 1: theta1 on literals (constants folded, equal xors shared) ----
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
                memo[key] = G(sigs, [COPY] if len(sigs) == 1 else xor_units(len(sigs)))
            th[p] = (memo[key], bool(flip))
        return th

    # ---- layer 2: chi1 as the counts S2 of every theta2 bit, plus digest 1 ----
    def chi_to_counts(th, next_pos, carry, scale):
        chi = {p: chi_lits(th, *p) for p in all_pos}
        packed = {}
        for p in next_pos:
            qs = count_pos(p)
            f = sum(flag(*q) for q in qs) % 2
            specs = [spec(chi[q], CHI, 1 / scale) for q in qs]
            n = len(qs)
            if and_left:  # own + left - 2 AND(left, own) = left ^ own (iota stays in f)
                x, y, z = p
                lits = [th[rp[(x - 1 + dx) % 5][y][z]] for dx in range(4)]
                specs.append(spec(lits, AND_CHI2, -2 / scale))
                n -= 1
            packed[p] = (node(specs, numeric=True), f, n)
        new = [([spec(chi[p], CHI)], flag(*p)) for p in digest_pos]
        return packed, carry_layer(carry) + pack(new, 2, m1)

    # ---- layer 3: round 2 in one layer, chi only mod 2 ----
    def lazy_round(packed, sc_in, next_pos, carry, sc_out):
        def t_specs(r, coef):  # coef * t_r, t_r = parity(S_r) ^ f_r
            sig, f, n = packed[r]
            c = coef * (1 - 2 * f)
            return parity_specs([(sig, sc_in, n)], c), coef * f

        def x_specs(rb, rc_, coef):  # coef * (t_b ^ t_c)
            (sb, fb, nb), (sc, fc, nc) = packed[rb], packed[rc_]
            f = fb ^ fc
            c = coef * (1 - 2 * f)
            return parity_specs([(sb, sc_in, nb), (sc, sc_in, nc)], c), coef * f

        def o_specs(q, coef):
            """coef * o'(q), o' = t_a + (t_c - t_b + t_b ^ t_c) / 2 (without iota)"""
            x, y, z = q
            ra, rb, rc_ = (rp[(x + dx) % 5][y][z] for dx in range(3))
            specs, const = [], 0.0
            for s, c in (t_specs(ra, coef), t_specs(rc_, coef / 2), t_specs(rb, -coef / 2),
                         x_specs(rb, rc_, coef / 2)):
                specs += s
                const += c
            if const:
                specs.append(const_spec(const))
            return specs

        out = {}
        for p in next_pos:
            qs = count_pos(p)
            f = sum(flag(*q) for q in qs) % 2
            specs = [sp for q in qs for sp in o_specs(q, 1 / sc_out)]
            # o' of p and of its left neighbour (same row) sum to at most 3:
            # t_x + (1 - t_x) t_{x+1} = t_x | t_{x+1} <= 1, so the count is <= 2*11 - 1
            out[p] = (node(specs, numeric=True), f, 2 * len(qs) - 1)
        new = [(o_specs(p, 1.0), flag(*p)) for p in digest_pos]
        return out, carry_layer(carry) + pack(new, 3, k2)

    # ---- layer 4: round 3 in one layer (Walsh) ----
    def walsh_last(packed, sc, carry):
        bits = decode(carry)
        for p in digest_pos:
            x, _, z = p
            q = [rp[(x + dx) % 5][0][z] for dx in range(3)]
            r = flag(*p)
            specs, const = [], float(r)
            for coef, subset in ((0.5, (0,)), (-0.5, (0, 1)), (0.5, (0, 2)), (0.5, (0, 1, 2))):
                feats = [packed[q[i]] for i in subset]
                fq = sum(f for _, f, _ in feats) % 2
                wq = (1 - 2 * r) * coef * (1 - 2 * fq)
                const += (1 - 2 * r) * coef * fq
                specs += parity_specs([(sig, sc, n) for sig, _, n in feats], wq, mpc=WMPC["on"])
            if const:
                specs.append(const_spec(const))
            bits.append(node(specs))
        return bits

    # ---- depth-5 layout: chi1 -> (a, P); X2: E = a + D; lazy chi2 with column parities ----
    def pair_of(p):
        return count_pos(p)[1:]

    def chi_to_split(th, scale):
        """chi1 bits (all positions) and column-pair counts P (scaled)"""
        chi = {p: chi_lits(th, *p) for p in all_pos}
        bits = {p: (node([spec(chi[p], CHI)]), flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                sums[(x, z)] = (node([spec(chi[q], CHI, 1 / scale) for q in qs], numeric=True), f, len(qs))
        return bits, sums

    def x_exact(bits, sums, scale):
        """E = a + D (flags folded, emitted as E/2), D = parity(P) shared by a column pair;
        digest 1 packed from the copy units of a (free)"""
        e = {}
        for p in all_pos:
            (asig, af), (psig, pf, n) = bits[p], sums[(p[0], p[2])]
            specs = [spec_on(asig, COPY, 0.5 * (1 - 2 * af))]
            specs += parity_specs([(psig, scale, n)], 0.5 * (1 - 2 * pf))
            if af or pf:
                specs.append(const_spec(0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [([spec_on(bits[p][0], COPY)], bits[p][1]) for p in digest_pos]
        return e, pack(new, 2, m1)

    XOR_E = Unit((2,), 0, (-2,), 2)  # on E/2: max(0, E)(2 - E) = [E == 1]
    Q_E = [Unit((-8, -4), 3, (0, 2), 0),  # on (Eb/2, Ec/2): max(0, 3 - 4Eb - 2Ec) Ec
           Unit((8, -4), -5, (0, 2), 0)]  # max(0, 4Eb - 2Ec - 5) Ec: [Eb != 1][Ec == 1]

    def lazy_chi_cols(e, next_pos, carry, scale):
        """chi2 mod 2 from E; a column's chi bits sum to parity(sum_y E_a) + sum_y Q (mod 2);
        count of theta3 bit p: own [E_a == 1] + Q(p) plus two columns: <= 13"""
        def q_specs(q, coef):
            x, y, z = q
            eb, ec = (e[rp[(x + dx) % 5][y][z]] for dx in (1, 2))
            return [({eb: u.weights[0], ec: u.weights[1]}, u.bias, {ec: u.value_weights[1] * coef}, 0.0)
                    for u in Q_E]

        def own_specs(q, coef):
            x, y, z = q
            return [spec_on(e[rp[x][y][z]], XOR_E, coef)] + q_specs(q, coef)

        def col_specs(xc, zc, coef):
            specs = parity_specs([(e[rp[xc][y2][zc]], 2.0, 2) for y2 in range(5)], coef)
            for y2 in range(5):
                specs += q_specs((xc, y2, zc), coef)
            return specs

        out = {}
        for p in next_pos:
            x, _, z = p
            qs = count_pos(p)
            f = sum(flag(*q) for q in qs) % 2
            specs = own_specs(p, 1 / scale) + col_specs((x + 4) % 5, z, 1 / scale) \
                + col_specs((x + 1) % 5, (z + 1) % w, 1 / scale)
            out[p] = (node(specs, numeric=True), f, 13)
        new = [(own_specs(p, 1.0), flag(*p)) for p in digest_pos]
        return out, carry_layer(carry) + pack(new, 3, k2)

    def lazy_chi(e, next_pos, carry, scale):
        """chi2 mod 2 from E (theta-structure's lazy chi): o' = [E_a == 1] + Q, counts <= 21"""
        def o_specs(q, coef):
            x, y, z = q
            ea, eb, ec = (e[rp[(x + dx) % 5][y][z]] for dx in range(3))
            return [spec_on(ea, XOR_E, coef)] + [
                ({eb: u.weights[0], ec: u.weights[1]}, u.bias, {ec: u.value_weights[1] * coef}, 0.0)
                for u in Q_E]

        out = {}
        for p in next_pos:
            qs = count_pos(p)
            f = sum(flag(*q) for q in qs) % 2
            out[p] = (node([sp for q in qs for sp in o_specs(q, 1 / scale)], numeric=True), f,
                      2 * len(qs) - 1)
        new = [(o_specs(p, 1.0), flag(*p)) for p in digest_pos]
        return out, carry_layer(carry) + pack(new, 3, k2)

    def theta_counts(packed, scale, carry):
        """theta bits = parity of counts (glu_xor units on one feature)"""
        th = {p: (node(parity_specs([(sig, scale, n)], 1.0)), bool(f)) for p, (sig, f, n) in packed.items()}
        return th, carry_layer(carry)

    def chi_last(th, carry):
        bits = decode(carry)
        for p in digest_pos:
            lits = chi_lits(th, *p)
            if flag(*p):
                l0 = lits[0]
                lits = [(1 - l0) if isinstance(l0, int) else (l0[0], not l0[1])] + lits[1:]
            bits.append(node([spec(lits, CHI)]))
        return bits

    node_cache: dict = {}
    index_of: dict = {}
    cur_msg: list = []

    def cached_node(key, specs_fn):
        """numeric glu on message bits; built from specs once, then from cached units"""
        hit = node_cache.get(key)
        if hit is not None:
            idx, units = hit
            return G([cur_msg[i] for i in idx], units, numeric=True)
        specs = specs_fn()
        incoming, seen = [], set()
        for g, _, v, _ in specs:
            for s_ in list(g) + list(v):
                if s_ not in seen:
                    seen.add(s_)
                    incoming.append(s_)
        pos = {s_: i for i, s_ in enumerate(incoming)}
        n = len(incoming)
        units = []
        for g, gb, v, vb in specs:
            gw, vw = [0] * n, [0] * n
            for s_, wt in g.items():
                gw[pos[s_]] = wt
            for s_, wt in v.items():
                vw[pos[s_]] = wt
            units.append(Unit(tuple(gw), gb, tuple(vw), vb))
        units = tuple(units)
        node_cache[key] = ([index_of[s_] for s_ in incoming], units)
        return G(incoming, units, numeric=True)

    # ---- depth-3 layer 1: round 1 in ONE layer on raw message bits (lazy chi1) ----
    def theta1_sets(a):
        """theta1 of every position as (raw message bits, flip): theta1 = flip ^ parity(bits)"""
        sets = {}
        for p in all_pos:
            flip, cnt = 0, {}
            for lit in theta_lits(a, *p, w):
                if isinstance(lit, int):
                    flip ^= lit
                else:
                    flip ^= int(lit[1])
                    cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
            sets[p] = (frozenset(s for s, c in cnt.items() if c), flip)
        return sets

    def raw_par(bits, f, coef):
        """specs of coef * (f ^ parity(bits)) on raw bits, and the constant part"""
        if not bits:
            return [], coef * f
        sigs = sorted(bits, key=lambda s: s.uid)
        n = len(sigs)
        units = [COPY] if n == 1 else raw_units(n)
        c = coef * (1 - 2 * f)
        specs = []
        for u in units:
            g = {s: float(u.weights[0]) for s in sigs} if u.weights[0] else {}
            v = {s: float(u.value_weights[0]) * c for s in sigs} if u.value_weights[0] else {}
            specs.append((g, float(u.bias), v, float(u.value_bias) * c))
        return specs, coef * f

    def lazy_round_raw(sets, next_pos, scale):
        """S2 counts of every theta2 bit from raw bits. o'(q) = t_a + AND(q),
        AND = (1 - t_b) t_c = (t_c - t_b + t_b ^ t_c) / 2 exact, t_b ^ t_c = parity(B xor C).
        cols: a column's linear part sum_y t_a is replaced by its exact parity (one parity of
        raw bits), so a column contributes <= 6 instead of <= 10: counts <= 13."""
        def abc(q):
            x, y, z = q
            return [sets[rp[(x + dx) % 5][y][z]] for dx in range(3)]

        def and_specs(q, coef):
            _, (B, fb), (C, fc) = abc(q)
            specs, const = [], 0.0
            for bits, f, cf in ((C, fc, coef / 2), (B, fb, -coef / 2), (B ^ C, fb ^ fc, coef / 2)):
                s_, c_ = raw_par(bits, f, cf)
                specs += s_
                const += c_
            return specs, const

        def o_specs(q, coef):
            (A, fa), _, _ = abc(q)
            specs, const = raw_par(A, fa, coef)
            s_, c_ = and_specs(q, coef)
            return specs + s_, const + c_

        def col_specs(xc, zc, coef):
            qs = [(xc, y2, zc) for y2 in range(5)]
            if not cols:
                specs, const = [], 0.0
                for q in qs:
                    s_, c_ = o_specs(q, coef)
                    specs += s_
                    const += c_
                return specs, const
            lin, fl = frozenset(), 0
            for q in qs:
                (A, fa), _, _ = abc(q)
                lin, fl = lin ^ A, fl ^ fa
            specs, const = raw_par(lin, fl, coef)
            for q in qs:
                s_, c_ = and_specs(q, coef)
                specs += s_
                const += c_
            return specs, const

        def s2_specs(p):
            x, y, z = p
            specs, const = o_specs(p, 1 / scale)
            for xc, zc in (((x + 4) % 5, z), ((x + 1) % 5, (z + 1) % w)):
                s_, c_ = col_specs(xc, zc, 1 / scale)
                specs += s_
                const += c_
            if const:
                specs.append(const_spec(const))
            return specs

        def pk_specs(chunk):  # base-3 packing of lazy digest values, as pack()
            sc = pack_scale(3, len(chunk))
            specs = []
            for j, p in enumerate(chunk):
                sp, const = o_specs(p, 3 ** j / sc)
                specs += sp
                if const:
                    specs.append(const_spec(const))
            return specs

        out = {}
        for p in next_pos:
            qs = count_pos(p)
            f = sum(flag(*q) for q in qs) % 2
            # own o' + the AND of the left neighbour (in column x-1) <= 2 (t_x | t_{x+1} <= 1):
            # counts <= 2 + 5 + 6 = 13 with cols (else 2*11 - 1 = 21)
            out[p] = (cached_node(("s2", p), lambda p=p: s2_specs(p)), f, 13 if cols else 21)
        carry = []
        if dig1 == "bin":  # digest-1 lazy values one per feature (o'/2), made exact in L2
            for p in digest_pos:
                def o1(p=p):
                    sp, const = o_specs(p, 0.5)
                    return sp + ([const_spec(const)] if const else [])
                carry.append(("o1", cached_node(("o1", p), o1), flag(*p)))
            return out, carry
        for i in range(0, len(digest_pos), k1):
            chunk = digest_pos[i:i + k1]
            sig = cached_node(("pk1", i), lambda chunk=chunk: pk_specs(chunk))
            carry.append(("pk", sig, tuple(flag(*p) for p in chunk), 3, len(chunk)))
        return out, carry

    # ---- depth-4 "d4x": fused lazy round 1 (o' per position) | X2 from o' | lazy chi2 cols | Walsh
    def lazy_o_raw(sets):
        """o'(q) = t_a + AND(q) in {0,1,2}, congruent to chi1 (no iota), emitted as o'/2"""
        def abc(q):
            x, y, z = q
            return [sets[rp[(x + dx) % 5][y][z]] for dx in range(3)]

        def o_specs(q, coef):
            (A, fa), (B, fb), (C, fc) = abc(q)
            specs, const = [], 0.0
            for bits, f, cf in ((A, fa, coef), (C, fc, coef / 2), (B, fb, -coef / 2),
                                (B ^ C, fb ^ fc, coef / 2)):
                s_, c_ = raw_par(bits, f, cf)
                specs += s_
                const += c_
            if const:
                specs.append(const_spec(const))
            return specs

        return {q: cached_node(("o", q), lambda q=q: o_specs(q, 0.5)) for q in all_pos}

    def x_from_o(osig):
        """E = [o' == 1] ^ iota + D (flags folded, emitted as E/2); D = parity of the column
        pair's 10 lazy values (a sum of 10 features, range 20) shared by the column's 5 bits;
        digest 1 packed binary from the exact [o' == 1] units (free)"""
        e = {}
        for p in all_pos:
            af = flag(*p)
            qs = pair_of(p)
            pf = sum(flag(*q) for q in qs) % 2
            specs = [spec_on(osig[p], XOR_E, 0.5 * (1 - 2 * af))]
            specs += parity_specs([(osig[q], 2.0, 2) for q in qs], 0.5 * (1 - 2 * pf))
            if af or pf:
                specs.append(const_spec(0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [([spec_on(osig[p], XOR_E)], flag(*p)) for p in digest_pos]
        return e, pack(new, 2, m1)

    def walsh_raw(sets):
        """steps avenue, S = 1 in ONE layer: chi = (p(A) - p(A^B) + p(A^C) + p(A^B^C)) / 2 with
        A, B, C the raw-bit sets of the three theta bits a chi bit reads (xs3.walsh_last on raw bits)"""
        bits = []
        for p in digest_pos:
            x, y, z = p
            q = [sets[rp[(x + dx) % 5][y][z]] for dx in range(3)]
            r = flag(*p)
            specs, const = [], float(r)
            for coef, subset in ((0.5, (0,)), (-0.5, (0, 1)), (0.5, (0, 2)), (0.5, (0, 1, 2))):
                bs, fq = frozenset(), 0
                for i in subset:
                    bs, fq = bs ^ q[i][0], fq ^ q[i][1]
                s_, c_ = raw_par(bs, fq, (1 - 2 * r) * coef)
                specs += s_
                const += c_
            if const:
                specs.append(const_spec(const))
            bits.append(node(specs))
        return bits

    def xof_fn(msg: list) -> list:
        lanes = initial_state(k, msg)
        if layout == "w1":
            return walsh_raw(theta1_sets(lanes))
        if layout == "d3":
            # layer-1 nodes depend on the message only through their inputs: after the
            # first call their (input indices, units) are reused (same weights, faster)
            index_of.clear()
            index_of.update({b: i for i, b in enumerate(msg)})
            cur_msg[:] = list(msg)
            sets = None if node_cache else theta1_sets(lanes)
            s2p, carry = lazy_round_raw(sets, all_pos if depth > 2 else chi_needs(digest_pos), s2)
            if dig1 == "bin":  # exact digest-1 bits [o' == 1] in L2, packed binary (m1)
                carry1 = pack([([spec_on(sig, XOR_E)], f) for _, sig, f in carry], 2, m1)
                s3p, carry2 = lazy_round(s2p, s2, chi_needs(digest_pos), [], s3)
                return walsh_last(s3p, s3, carry1 + carry2)
            sp_, sc = s2p, s2
            for r in range(2, depth):  # lazy rounds 2 .. S-1 (S = 3: the original d3)
                sp_, carry = lazy_round(sp_, sc, chi_needs(digest_pos) if r == depth - 1 else all_pos, carry, s3)
                sc = s3
            return walsh_last(sp_, sc, carry)
        if layout == "d4x":
            index_of.clear()
            index_of.update({b: i for i, b in enumerate(msg)})
            cur_msg[:] = list(msg)
            osig = lazy_o_raw(None if node_cache else theta1_sets(lanes))
            e, carry = x_from_o(osig)
            sp_, carry = lazy_chi_cols(e, chi_needs(digest_pos) if depth == 3 else all_pos, carry, s2)
            sc = s2
            for r in range(3, depth):  # lazy rounds 3 .. S-1 (S = 3: the original d4x)
                sp_, carry = lazy_round(sp_, sc, chi_needs(digest_pos) if r == depth - 1 else all_pos, carry, s3)
                sc = s3
            return walsh_last(sp_, sc, carry)
        th = theta1_direct(lanes, chi_needs(all_pos))
        if layout == "d6":
            bits, sums = chi_to_split(th, s2)
            e, carry = x_exact(bits, sums, s2)
            s3p, carry = lazy_chi(e, chi_needs(digest_pos), carry, s2)
            th3, carry = theta_counts(s3p, s2, carry)
            return chi_last(th3, carry)
        if layout == "d5":
            bits, sums = chi_to_split(th, s2)
            e, carry = x_exact(bits, sums, s2)
            s3p, carry = lazy_chi_cols(e, chi_needs(digest_pos), carry, s2)
            return walsh_last(s3p, s2, carry)
        s2p, carry = chi_to_counts(th, all_pos, [], s2)
        s3p, carry = lazy_round(s2p, s2, chi_needs(digest_pos), carry, s3)
        return walsh_last(s3p, s3, carry)

    return xof_fn


def make(**kw):
    def variant(k, depth):
        return build(k, depth, **kw), {"msg": Bits("0" * k.msg_len).bitlist}
    return variant


d4 = make()  # digests: pairs (binary) and lazy values one per feature
d4_m4k2 = make(m1=4, k2=2)
d4_m4k3 = make(m1=4, k2=3)
d4_m3k2 = make(m1=3, k2=2)
d5 = make(layout="d5")  # digests: pairs and one lazy value per feature
d5_m3k2 = make(layout="d5", m1=3, k2=2)
d5_m4k3 = make(layout="d5", m1=4, k2=3)
d6 = make(layout="d6")  # = xs3:lazy_middle with this module's carries
d6_m3k2 = make(layout="d6", m1=3, k2=2)
d6_m2k2 = make(layout="d6", m1=2, k2=2)
d4a_m4k3 = make(m1=4, k2=3, and_left=True)  # + one AND unit per theta2 count in layer 2


d3 = make(layout="d3", k2=3)
w1 = make(layout="w1")  # steps avenue: S = 1 in one layer (Walsh round on raw bits)
d3_nocols = make(layout="d3", k2=3, cols=False)


def _mp(variant):
    def v(k, depth):
        import xs3_u as xs3
        xs3.MINPAR["on"] = True
        return variant(k, depth)
    return v


d3_mp = _mp(d3)


def _mpx(variant):
    def v(k, depth):
        MPX["on"] = True
        return variant(k, depth)
    return v


d3_mpx = _mpx(d3)

d4x = make(layout="d4x", m1=3, k2=2)
d4x_m4k3 = make(layout="d4x", m1=4, k2=3)
d4x_mpx = _mpx(d4x)
d4x_m4k3_mpx = _mpx(d4x_m4k3)
d4x_m4k2_mpx = _mpx(make(layout="d4x", m1=4, k2=2))
d4x_m3k3_mpx = _mpx(make(layout="d4x", m1=3, k2=3))
d4x_m2k2_mpx = _mpx(make(layout="d4x", m1=2, k2=2))

d3_bin = make(layout="d3", k2=3, m1=4, dig1="bin")
d3_bin_k2 = make(layout="d3", k2=2, m1=4, dig1="bin")
d3_k2 = make(layout="d3", k2=2)
d3_k1_2 = make(layout="d3", k2=3, k1=2)


def _ctr(variant):
    def v(k, depth):
        CENTER["on"] = True
        return variant(k, depth)
    return v


d3c = _ctr(d3)
d3c_bin = _ctr(d3_bin)
d4x_c = _ctr(d4x)
d3c_k2 = _ctr(d3_k2)


def _ctr_mp17(variant):
    def v(k, depth):
        CENTER["on"] = True
        MPX["on"] = True
        MPX["max"] = 17
        return variant(k, depth)
    return v


d3c_mp17 = _ctr_mp17(d3)


def _ctr_raw(variant):
    """centred pieces only for the raw parities of layer 1 (exact inputs); counts keep
    glu_xor, whose knot at s = 0 averages the slope there (gain 1 instead of 2)"""
    def v(k, depth):
        CENTER["on"] = True
        CENTER["counts"] = False
        return variant(k, depth)
    return v


d3c1 = _ctr_raw(d3)


def _ctr_raw_ends(variant):
    def v(k, depth):
        CENTER["on"] = True
        CENTER["counts"] = False
        ENDS["on"] = True
        return variant(k, depth)
    return v


d3c1e = _ctr_raw_ends(d3)


def _ctr_raw_top(variant):
    def v(k, depth):
        CENTER["on"] = True
        CENTER["counts"] = False
        ENDS["on"] = True
        ENDS["zero"] = False
        return variant(k, depth)
    return v


d3c1t = _ctr_raw_top(d3)


def _ctr_raw_mp17(variant):
    """centred raw parities, min-parity for odd raw sets of 7..17 bits, glu_xor counts"""
    def v(k, depth):
        CENTER["on"] = True
        CENTER["counts"] = False
        MPX["on"] = True
        MPX["max"] = 17
        return variant(k, depth)
    return v


d3c1_mp17 = _ctr_raw_mp17(d3)


# ---- combined-2: d4x + min-parity on the odd count ranges of the Walsh layer (units-per-bit's
# MPC / first-last's mp_last): singles 13 -> 6 units, triples 39 -> 19 ----
def _wmpc(variant):
    def v(k, depth):
        WMPC["on"] = True
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc = _wmpc(d4x_m4k2_mpx)
d4x_mpx_wmpc = _wmpc(d4x_mpx)


# ---- wave 3 (and-of-parities): parities of raw bit sets with fewer units (p1d_forms.py) ----
def _p1d(variant):
    def v(k, depth):
        P1D["on"] = True
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1d = _p1d(d4x_m4k2_mpx_wmpc)
d4x_mpx_p1d = _p1d(d4x_mpx)
d4x_m4k2_mpx_p1d = _p1d(d4x_m4k2_mpx)
d3c1_mp17_p1d = _p1d(d3c1_mp17)  # centred pieces stay for the sets the table does not cover


def _p1dk(variant):
    def v(k, depth):
        P1DK["on"] = True
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1d_k = _p1dk(d4x_m4k2_mpx_wmpc_p1d)


def _p1d_singles(variant):
    """P1D on the theta1 singles only (sets of <= 9 raw bits), glu_xor/min-parity pairs"""
    def v(k, depth):
        P1D["on"] = True
        P1D["max"] = 9
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1ds = _p1d_singles(d4x_m4k2_mpx_wmpc)
d4x_m4k2_mpx_wmpc_p1ds_k = _p1dk(d4x_m4k2_mpx_wmpc_p1ds)


def _p1dke(variant):
    def v(k, depth):
        P1DK["on"] = True
        P1DK["ext"] = True
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1d_ke = _p1dke(d4x_m4k2_mpx_wmpc_p1d)
d3c1_mp17_p1d_ke = _p1dke(d3c1_mp17_p1d)


def _p1dkeo(variant):
    def v(k, depth):
        P1DK["on"] = True
        P1DK["ext"] = True
        P1DK["odd"] = True
        return variant(k, depth)
    return v


d3c1_mp17_p1d_keo = _p1dkeo(d3c1_mp17_p1d)


def _rawext(variant):
    def v(k, depth):
        P1D["ext"] = True
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1d_ke_x = _rawext(d4x_m4k2_mpx_wmpc_p1d_ke)
d3c1_mp17_p1d_ke_x = _rawext(d3c1_mp17_p1d_ke)
d3c1_mp17_p1d_keo_x = _rawext(d3c1_mp17_p1d_keo)


def _rawextc(variant):
    def v(k, depth):
        P1D["ext"] = "c"
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1d_ke_xc = _rawextc(d4x_m4k2_mpx_wmpc_p1d_ke)
d3c1_mp17_p1d_ke_xc = _rawextc(d3c1_mp17_p1d_ke)
d3c1_mp17_p1d_keo_xc = _rawextc(d3c1_mp17_p1d_keo)


def _p1dkt(variant):
    def v(k, depth):
        P1DK["on"] = True
        P1DK["ext"] = "t"
        return variant(k, depth)
    return v


d3c1_mp17_p1d_kt_xc = _rawextc(_p1dkt(d3c1_mp17_p1d))
d3c1_mp17_p1d_xc = _rawextc(d3c1_mp17_p1d)
d4x_m4k2_mpx_wmpc_p1d_kt_xc = _rawextc(_p1dkt(d4x_m4k2_mpx_wmpc_p1d))


def _p1dkto(variant):
    def v(k, depth):
        P1DK["on"] = True
        P1DK["ext"] = "t"
        P1DK["odd"] = "t"
        return variant(k, depth)
    return v


d3c1_mp17_p1d_kto_xc = _rawextc(_p1dkto(d3c1_mp17_p1d))


def _p1dkt1(variant):
    def v(k, depth):
        P1DK["on"] = True
        P1DK["ext"] = "t"
        P1DK["odd"] = "t1"
        return variant(k, depth)
    return v


d3c1_mp17_p1d_kt1_xc = _rawextc(_p1dkt1(d3c1_mp17_p1d))
d4x_m4k2_mpx_wmpc_p1d_kt1_xc = _rawextc(_p1dkt1(d4x_m4k2_mpx_wmpc_p1d))


def _rawodd(variant):
    def v(k, depth):
        P1D["odd"] = True
        return variant(k, depth)
    return v


d3c1_mp17_p1d_kt1_xco = _rawodd(d3c1_mp17_p1d_kt1_xc)
d3c1_mp17_p1d_kt_xco = _rawodd(d3c1_mp17_p1d_kt_xc)


def _rawoddall(variant):
    def v(k, depth):
        P1D["odd"] = "all"
        return variant(k, depth)
    return v


d4x_m4k2_mpx_wmpc_p1d_kt_xca = _rawoddall(d4x_m4k2_mpx_wmpc_p1d_kt_xc)
d3c1_mp17_p1d_kt_xca = _rawoddall(d3c1_mp17_p1d_kt_xc)
d3c1_mp17_p1d_kt1_xca = _rawoddall(d3c1_mp17_p1d_kt1_xc)


# steps avenue (wave 4): S = 1 in one layer with the raw-bit parity forms of the depth-3/4 points
w1_mpx = _mpx(w1)
w1_p1d_xca = _rawoddall(_rawextc(_p1d(_mpx(w1))))

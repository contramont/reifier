"""Shallow XOF layouts: rounds computed in ONE SwiGLU layer (avenue xof-structure).

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

from xs import CHI, COPY, keccak_maps, initial_state, theta_lits, xor_units
from xs3 import spec, spec_on, node
from decode import check as synth_decoder

G = glu

# chi(t1,t2,t3) AND chi(t2,t3,t4), two adjacent chi bits of a row, in ONE gated unit
# (exact on all 16 inputs; found by an exhaustive gate search, s2/and2chi.py)
AND_CHI2 = Unit((4, 2, -1, 1), -4, (0.75, -1, 0.5, -0.5), 0.75)


def parity_units(n: int, scale: float) -> list[Unit]:
    """glu_xor units on one feature f = s / scale with s in [0, n]"""
    units = [Unit((scale,), 0, (-scale,), 2)]
    units += [Unit((scale,), -2 * j, (0,), 4) for j in range(1, ceil(n / 2))]
    return units


def parity_specs(feats, coef):
    """specs of coef * parity(sum of counts): feats = [(sig, scale, n)], one gate on all"""
    n = sum(nn for _, _, nn in feats)
    specs = []
    g = {}
    for sig, sc, _ in feats:
        g[sig] = g.get(sig, 0.0) + float(sc)
    for j, u in enumerate(parity_units(n, 1.0)):
        v = {sig: -w * coef for sig, w in g.items()} if j == 0 else {}
        specs.append((dict(g), float(u.bias), v, float(u.value_bias) * coef))
    return specs


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
          m1: int = 2, k2: int = 1, and_left: bool = False, pack_a: bool = False):
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
    assert layout in ("d4", "d5", "d6") and depth == 3, "layouts of a 3-step XOF"

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
                specs += parity_specs([(sig, sc, n) for sig, _, n in feats], wq)
            if const:
                specs.append(const_spec(const))
            bits.append(node(specs))
        return bits

    # ---- depth-5 layout: chi1 -> (a, P); X2: E = a + D; lazy chi2 with column parities ----
    def pair_of(p):
        return count_pos(p)[1:]

    # optimizer avenue: chi1 bits that the X layer only copies are emitted as binary pairs
    # (c1 + 2 c2) / 4 and decoded by the 2 units that copying them takes (see xs3.a_specs)
    PK_U1 = Unit((4,), 0, (0,), 1)  # max(0, p) = p
    PK_U2 = Unit((4,), -1, (-2,), 2)  # max(0, p - 1)(2 - p/2) = c2, p = c1 + 2 c2

    def a_specs(asig, coef):
        if isinstance(asig, tuple):
            _, sig, j = asig
            if j == 0:
                return [spec_on(sig, PK_U1, coef), spec_on(sig, PK_U2, -2 * coef)]
            return [spec_on(sig, PK_U2, coef)]
        return [spec_on(asig, COPY, coef)]

    def chi_to_split(th, scale):
        """chi1 bits (all positions) and column-pair counts P (scaled)"""
        chi = {p: chi_lits(th, *p) for p in all_pos}
        if pack_a:
            bits = {}
            for i in range(0, len(all_pos) - 1, 2):
                p1, p2 = all_pos[i], all_pos[i + 1]
                sig = node([spec(chi[p1], CHI, 0.25), spec(chi[p2], CHI, 0.5)], numeric=True)
                bits[p1] = (("pk2", sig, 0), flag(*p1))
                bits[p2] = (("pk2", sig, 1), flag(*p2))
            if len(all_pos) % 2:
                bits[all_pos[-1]] = (node([spec(chi[all_pos[-1]], CHI)]), flag(*all_pos[-1]))
        else:
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
            specs = a_specs(asig, 0.5 * (1 - 2 * af))
            specs += parity_specs([(psig, scale, n)], 0.5 * (1 - 2 * pf))
            if af or pf:
                specs.append(const_spec(0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [(a_specs(bits[p][0], 1.0), bits[p][1]) for p in digest_pos]
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

    def xof_fn(msg: list) -> list:
        lanes = initial_state(k, msg)
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

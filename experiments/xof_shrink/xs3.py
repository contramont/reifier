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
from minpar_counts import min_units  # units-per-bit (imported outside tracing)

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
# units-per-bit: X layers gate on the 10 chi-bit features directly instead of reading a
# column-pair count feature P (the chi layer no longer emits P): fewer features, more nonzeros
NOP = {"on": False}
# units-per-bit: lazy4c's column reduction with ONE unit instead of two. With a pass term,
# the minimum window for c = sum_y Ea_y in [0, 10] is 4 with 2 units and 5 with 1 unit
# (exhaustive MILP, t_win_milp.py): r5(c) = c - 2 max(0, c - 5) (a zigzag, knot on an
# integer, flat under silu). Counts widen from [0, 32] to [0, 34] (16 -> 17 parity units
# per last-theta bit) but the lazy chi layer, whose units cost ~2.3x more, loses 320 units.
C5 = {"on": False}
# units-per-bit: the lazy digest bits of the step before the last (o in [0, 4], one pass
# unit each) are packed two per feature, F = o1 + 5 o2, so a pair costs one pass unit and
# one feature; the next layer decodes F into the usual pair p = d1 + 2 d2 with a
# DP-minimal decoder (decode.py: 12 units, knots on integers) instead of 2 x 2 units.
LZ5 = {"on": False}
# units-per-bit: instead of reducing each column's linear part (lazy4c: 2 units per column,
# C5: 1), reduce each COUNT's whole linear part L = own Ea + c_L + c_R in [0, 22] with ONE
# zigzag unit z(L) = L - 2 max(0, L - 11) in [0, 11] (knot on an integer, flat under silu).
# Same unit count as C5 (320), but counts lie in [0, 33]: odd, so min-parity (MPC) needs
# (33 - 1) / 2 = 16 units per last-theta bit instead of 17.
Z11 = {"on": False}
# units-per-bit: before the last theta, an extra (narrow) layer folds each count into a
# window of width ceil(n/3) (3 units) so the last theta needs 5 units per bit, not 16.
RP = {"on": False}
# units-per-bit: round-1 theta (raw message bits) with units shared across a column.
# theta(A, T) = parity(A + T), T in [0, 7] the live column-pair bits, A the own bit:
#   sum_k max(0, a_k A + b_k T + c_k)(d_k A + e_k T + f_k)      (3 units per bit)
#   + max(0, 16 - 3T)(26/5 - 6T/5) + (75T/2 - 416/5)          (2 units per column)
# Found by a sampled search over wave 1's per-bit triples (syn/t_d_q2.py), solved exactly
# (syn/t_d_exact.py, unique rational solution). Zero lanes use the same units at A = 0.
# Per column pair with T <= 7: 3 per live bit + 3 (zero lane) + 2 instead of 4 + 3.
TH1S = {"on": False}
TH1S_FORM = {
    "bit": [((1, 1, -3), (0, 0, -4)), ((2, 1, -6), (-8, 2, -6)), ((5, 2, -1), (17 / 4, -13 / 10, -4))],
    "hinge": ((-3, 16), (-6 / 5, 26 / 5)),
    "affine": (75 / 2, -416 / 5),
}
# theta1-t8 (wave 3): round-1 theta with a column POOL made of the zero lane's own units.
# Per column: the zero lane's units are the A = 0 slices z_i(T) = max(0, b_i T + c_i)(e_i T + f_i)
# of the three per-bit units, plus ONE extra hinge X(T); every live bit reads its three per-bit
# units u_i(A, T) = max(0, a_i A + b_i T + c_i)(d_i A + e_i T + f_i) plus (gamma_i - 1) z_i + X,
# and the zero lane is sum_i gamma_i z_i + X. Exact on {0,1} x [0, 8] (sympy, syn/pbfull.py), so it
# covers the T = 8 columns (x = 1) and every T <= 7 column: per column 3 per live bit + 4
# (T = 8: 20 -> 16 units; T <= 7: TH1S's 3 per live bit + 5 -> + 4).
# Found by syn/pb.py (all 4.26M (D)-feasible triples of wave 1, knot of X solved exactly per
# interval) + syn/pbx.py (exact sympy solve of the bilinear system).
TH1P = {"on": False, "form": "F1"}
TH1P_FORMS = {
    # F1 (default): fewest nonzeros of the 494 exact forms found (values of units 1-2 do not read
    # T; the extra unit is glu_xor's base unit max(0, T)(2 - T))
    "F1": {
        "bit": [((-2, 1, -4), (-19, 0, 8)), ((7, 2, -4), (3, 0, 4)), ((15, 2, -13), (0, 1, -10))],
        "gamma": (1 / 2, 1 / 2, -4 / 3),
        "extra": ((1, 0), (-1, 2)),  # X(T) = max(0, gT T + gc)(vT T + vc)
        "tmax": 8,
    },
    # F2 (T <= 7 only): pool of 3 = the zero lane's MINPAR7-type units (knots 3/2, 23/5 up; 3 down)
    # plus the layer's shared constant; per-bit A = 1 knots 9/2 and two quadratic surds
    # (syn/f2_verify.py: exact in sympy). Units: (s, tau0, tau1, lam, d, C0, C1, gate scale, slice
    # gate scale): u = max(0, c (s (T - tau0) + s (tau0 - tau1) A)) (lam (C0 + C1 T) + d A) / c,
    # z = max(0, c' s (T - tau0)) (C0 + C1 T) / c'; zero lane = sum z + kappa;
    # live bit = sum u + sum (1 - lam) z + kappa.
    "F2": {
        "knots": [
            (1, 1.5, 4.5, 16 / 27, -7.445032333921222810111699, -47 / 3, 8 / 3, 2.0, 2.0),
            (1, 4.6, 0.4043124926863418950549694, 38 / 27, 13.92395667080460828681571, 0.0, -5 / 3, 2.5, 5.0),
            (-1, 3.0, 6.478840944994543602795005, 640 / 567, 2.679706449190905528892440, -25 / 6, -19 / 12, 2.0, 1.0),
        ],
        "kappa": 25 / 2,
        "tmax": 7,
    },
    # F3 (T <= 7 only): same pool as F2, a point of the same (D) family with lambda_2 = 0, so the
    # second per-bit unit's value reads only A (fewer nonzeros); syn/f3_verify.py (exact in sympy)
    "F3": {
        "knots": [
            (1, 1.5, 0.5443979442634624659504331, -8 / 13, 9.575686273072953053755526, -47 / 3, 8 / 3, 2.0, 2.0),
            (1, 4.6, 142 / 41, 0.0, -328 / 19, 0.0, -5 / 3, 2.5, 5.0),
            (-1, 3.0, 6.508369328661332571139787, 24 / 19, 2.990780139669816060727945, -25 / 6, -19 / 12, 2.0, 1.0),
        ],
        "kappa": 25 / 2,
        "tmax": 7,
    },
    # F0: the first exact form found (every value reads T; X knot at 9/2, gate scaled by 2)
    "F0": {
        "bit": [((8, -1, 2), (-29 / 23, -2, 14)), ((15, 2, -12), (-118 / 23, 1, -1)), ((2, -2, 7), (-8, 2, -4))],
        "gamma": (1 / 2, 1 / 2, 1 / 2),
        "extra": ((2, -9), (-1, 6)),
        "tmax": 8,
    },
}
_LZ5 = {}


def LZ5_DECODER():
    if "d" not in _LZ5:
        _LZ5["d"] = synth_decoder([(F % 5) % 2 + 2 * ((F // 5) % 2) for F in range(25)])
    return _LZ5["d"]


def par_units(n: int) -> list:
    """units for parity of n raw input bits: min-parity for n in (7, 9) if enabled; with P1DR
    on, even n >= 8 use and-of-parities' n/2 - 1 unit form (p1d_forms.FORMS_EXTC)"""
    if P1DR["on"] and P1DR.get("odd") and n % 2 == 1 and n >= 7 and (n + 1) in P1DR["forms"]:
        us, c = P1DR["forms"][n + 1]  # same units as min-parity; only 2 read all n bits in their values
        out = [Unit((gw,) * n, gb, (vw,) * n, vb) for gw, gb, vw, vb in us]
        return out + ([Unit((0,) * n, 1, (0,) * n, c)] if c else [])
    if P1DR["on"] and n % 2 == 0 and n in P1DR["forms"]:
        us, c = P1DR["forms"][n]
        out = [Unit((gw,) * n, gb, (vw,) * n, vb) for gw, gb, vw, vb in us]
        return out + ([Unit((0,) * n, 1, (0,) * n, c)] if c else [])
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


# units-per-bit: min-parity (knots between integers) on COUNT features, re-tested at q=8.
UP1 = {"on": False, "d": True}  # combined-2: optimizer's round-1 u pairs (upair1, upair1d)
U_A = Unit((4,), 0, (0,), 1)
U_B = Unit((-4,), 0, (0,), 1)
U_C = Unit((4,), -1, (-2,), 2)
U_Cn = Unit((-4,), -1, (2,), 2)
def scale_specs(sl, c):
    return [(q[0], q[1], {k: v * c for k, v in q[2].items()}, q[3] * c) for q in sl]


class _TKey:  # placeholder for a theta bit that g-fold never builds as a node
    pass


GF1 = {"on": False}  # combined-2: g-fold after the u1 Y1 layer
GSC = 4.0
S4 = {"on": False}  # combined-2: packing's s4 (digest 1 at 4 bits per feature, split in the last theta)
_ST4: dict = {}


def ST4_DECODER():
    if "d" not in _ST4:
        _ST4["d"] = synth_decoder([p >> 2 for p in range(16)])
    return _ST4["d"]


SLAST = {"on": False}  # combined-2: optimizer's s_last as a switch
P1DC = {"on": False, "forms": None}  # wave-3 and-of-parities: 4-unit parity of the X2 column-pair count D
P1DP = {"on": False, "forms": {}}
P1DR = {"on": False, "forms": {}}  # wave 3: even raw-bit parities (round 1) with n/2 - 1 units  # wave 3: count_parity_units from a table (flatter n = 11 form)
MPC = {"on": False}
# wave 4 (rounds): with R rounds per step, sloped forms compound over the rounds that follow them.
# CPK["k"] = K limits column packing (cp) to the X layers of the last K rounds (None: every X);
# P1K["k"] = K limits P1DC's D form (irrational knots) to the X layers of the last K rounds.
CPK = {"k": None}
P1K = {"k": None}
# wave 4 (rounds): the split Y unit [E == 1] = max(0, E)(2 - E) has slope -2 at E = 2 under silu, so
# errors of the chi bits grow by up to ~2x per split round (float32 fails after ~20 split rounds).
# YFLAT adds glu_xor's second unit 4 max(0, E - 2): zero on E in {0, 1, 2}, and its half slope at the
# knot cancels the -2, so Y is flat at E = 1 and E = 2 (slope 1 only at E = 0, where a = D = 0).
# "every": flat Y only in split rounds r with r % every == 0.
YFLAT = {"on": False, "every": 1}
# steps avenue (wave 4): measurement switch only. The digests of earlier steps are never
# packed, carried or decoded: the size difference is the cost of the carries.
# NOT a valid XOF circuit (outputs the last digest only).
NOCARRY = {"on": False}
# steps avenue (wave 4): column packing (cp) only in the X layers of 0-based rounds >= CPMIN["step"]
# (rounds = XOF steps when R = 1)
# (a negative value counts from the last step: -2 = only the X before the last theta). cp's
# decoders do not damp float32 errors, and errors compound over the steps after them.
CPMIN = {"step": 0}
# steps avenue: digests of 0-based steps >= depth - 1 - LATEPK["j"] that carry_new packs are made late,
# carry the largest analog errors (e.g. the one read through cp's decoders) and cross few layers:
# pack them LATEPK["m"] bits per feature instead of m1 (lp2 = the last m1-packed digest of a lazy layout)
LATEPK = {"j": 0, "m": 2}


def count_parity_units(n: int, scale: float) -> list[Unit]:
    """parity of a count feature f = s / scale, s in [0, n]: with MPC on, the fewest known
    units (minpar_counts: (n - 1) / 2 for odd n >= 7, knots between integers), else glu_xor"""
    if P1DP["on"] and n in P1DP["forms"]:  # wave 3: a flatter form with the same unit count
        us, c = P1DP["forms"][n]
        out = [Unit((a * scale,), b, (p * scale,), q) for a, b, p, q in us]
        if c:
            out.append(Unit((0.0,), 1.0, (0.0,), c))
        return out
    if not MPC["on"]:
        return parity_units(n, scale)
    us, c = min_units(n)
    out = [Unit((a * scale,), b, (p * scale,), q) for a, b, p, q in us]
    if c:
        out.append(Unit((0.0,), 1.0, (0.0,), c))
    return out



def theta1_shared_lits(lits, memo):
    """units-per-bit: round-1 theta with units shared by the bits of a column.
    theta = parity(A + T) (xor flags), T = the live message bits of the column pair,
    A = the own bit (or constant 0 for zero lanes): 3 per-bit units + 2 units per
    column (a hinge on T and an affine unit), exact for T in [0, 7] (TH1S_FORM).
    Returns None when the form does not apply (T > 7 or T < 2)."""
    own, cols = lits[0], lits[1:]
    flip, tset = 0, {}
    for lit in cols:
        if isinstance(lit, int):
            flip ^= lit
        else:
            flip ^= int(lit[1])
            tset[lit[0]] = tset.get(lit[0], 0) ^ 1
    tsigs = sorted((s for s, c in tset.items() if c), key=lambda s: s.uid)
    if TH1P["on"]:
        return _theta1_pool(own, tsigs, flip, tset, memo)
    if not 2 <= len(tsigs) <= 7:
        return None
    if isinstance(own, int):
        asig = None
        flip ^= own
    else:
        asig = own[0]
        flip ^= int(own[1])
        if asig in tset:
            return None
    key = ("th1s", asig, frozenset(tsigs))
    if key not in memo:
        specs = []
        for (ga, gb, gc), (va, vb, vc) in TH1S_FORM["bit"]:
            g = {s_: float(gb) for s_ in tsigs}
            v = {s_: float(vb) for s_ in tsigs} if vb else {}
            if asig is not None:
                g[asig] = float(ga)
                if va:
                    v[asig] = float(va)
            specs.append((g, float(gc), v, float(vc)))
        (hg, hc), (he, hf) = TH1S_FORM["hinge"]
        specs.append(({s_: float(hg) for s_ in tsigs}, float(hc), {s_: float(he) for s_ in tsigs}, float(hf)))
        lt, mt = TH1S_FORM["affine"]
        specs.append(({}, 1.0, {s_: float(lt) for s_ in tsigs}, float(mt)))
        memo[key] = node(specs)
    return (memo[key], bool(flip))

def _theta1_pool_knots(asig, tsigs, F):
    """F2-type form (knots, pool of 3 slices + the shared constant)"""
    specs = []
    for s, t0, t1, lam, dd, c0, c1, cg, cs in F["knots"]:
        if asig is not None:
            g = {s_: cg * s for s_ in tsigs}
            g[asig] = cg * s * (t0 - t1)
            v = {s_: lam * c1 / cg for s_ in tsigs} if lam * c1 else {}
            v[asig] = dd / cg
            specs.append((g, -cg * s * t0, v, lam * c0 / cg))
        co = (1 - lam) if asig is not None else 1.0
        if co:
            g = {s_: cs * s for s_ in tsigs}
            v = {s_: co * c1 / cs for s_ in tsigs} if c1 else {}
            specs.append((g, -cs * s * t0, v, co * c0 / cs))
    specs.append(({}, 1.0, {}, F["kappa"]))  # the layer's shared constant unit
    return specs


def _theta1_pool(own, tsigs, flip, tset, memo):
    """theta1-t8: pool form (TH1P_FORM), exact for T = len(tsigs) in [0, tmax]"""
    fname = TH1P["form"]
    if fname == "F18":  # wave 4 (word-size): the pool only where it beats the direct raw forms,
        # the T = 8 columns (3 units fewer each); every other column falls back to par_units
        if len(tsigs) != 8:
            return None
        fname = "F1"
    if fname in ("F12", "F13"):  # F2 / F3 (pool of 3) where T <= 7, F1 (pool of 4) for T = 8
        fname = ("F2" if fname == "F12" else "F3") if len(tsigs) <= 7 else "F1"
    TH1P_FORM = TH1P_FORMS[fname]
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
    key = ("th1p", fname, asig, frozenset(tsigs))
    if key not in memo and "knots" in TH1P_FORM:
        memo[key] = node(_theta1_pool_knots(asig, tsigs, TH1P_FORM))
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
    return 1 - lit if isinstance(lit, int) else (lit[0], not lit[1])


def build(k: K.Keccak, depth: int, kinds=None, pairs: bool = True, walsh: bool = False, m1: int = 2,
          cp: bool = False, dedupe: bool = False, m2: int | None = None):
    """kinds[i] in ("direct", "split", "shared") is the theta layout of step i ("shared"
    only after the first step); the last step is direct (one theta bit per column pair
    left, nothing to share), or merged with its chi into one layer if walsh."""
    w = k.w
    # R Keccak rounds per XOF step (k.n): the circuit is T = depth * R rounds in a row; round r
    # uses round constant rcs[r % R], and only rounds with (r + 1) % R == 0 emit a digest. The
    # other ("middle") rounds keep the full state and only carry the earlier digests.
    rcs = k.get_round_constants()
    R = len(rcs)
    T = depth * R
    cur = [0]  # the round of the chi whose flags / digest the current layer handles
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[: k.d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]
    kinds = list(kinds or ["direct"] * T)
    assert len(kinds) == T, (len(kinds), T)
    kinds[-1] = "direct"

    def flag(x, y, z):  # iota of round cur[0]
        return int(x == 0 and y == 0 and rcs[cur[0] % R][z] == "1")

    def emits():  # does the chi of round cur[0] end an XOF step (its digest is output)?
        return (cur[0] + 1) % R == 0

    def dpos():  # digest positions the chi of round cur[0] emits
        return digest_pos if emits() else []

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
            if split and S4["on"] and kind == "pk" and len(f) == 4:
                # packing's s4: p/16 -> pairs (p - 4 hi)/4 and hi/4, hi = floor(p/4) by the
                # integer-knot DP staircase (6 units), lo from the copy unit
                const, units = ST4_DECODER()
                hi = [({sig: 16.0}, float(-kk), {sig: 16.0 * float(b)} if b else {}, float(a))
                      for kk, a, b in units]
                if const:
                    hi.append(({}, 1.0, {}, float(const)))
                lo = [({sig: 16.0}, 0.0, {}, 1 / 4)] + scale_specs(hi, -1.0)
                out.append(("pair", node(lo, numeric=True), f[:2]))
                out.append(("pair", node(scale_specs(hi, 1 / 4), numeric=True), f[2:]))
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
            elif kind == "lz5":  # F / 32, F = o1 + 5 o2: pair p = parity(o1) + 2 parity(o2)
                const, units = LZ5_DECODER()
                specs = [({sig: 32.0}, float(-kk), {sig: float(b) * 32.0 / 4} if b else {}, float(a) / 4)
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

    def carry_new(specs_flags, m=None):
        """carry items for new digest bits given as (unit spec at coef 1, flag) pairs;
        a bit may also be a list of unit specs (a sum of units, see theta_split_x_cp);
        m: bits per packed feature (default m1)"""
        mm = m1 if m is None else m
        if LATEPK["j"] and CPMIN.get("dig", 0) >= depth - 1 - LATEPK["j"]:
            mm = LATEPK["m"]  # steps avenue: the last j packed digests (largest errors) in pairs
        if NOCARRY["on"]:
            return []

        def sl(s):
            return s if isinstance(s, list) else [s]

        def scl(s, c):
            return (s[0], s[1], {q: v * c for q, v in s[2].items()}, s[3] * c)
        items = []
        if pairs and mm >= 3:
            for i in range(0, len(specs_flags), mm):
                chunk = specs_flags[i:i + mm]
                sc = pack_scale(2, len(chunk))
                specs = []
                for j, (s_, _) in enumerate(chunk):
                    specs += [scl(u, 2 ** j / sc) for u in sl(s_)]
                items.append(("pk", node(specs, numeric=True), tuple(f for _, f in chunk)))
        elif pairs:
            for i in range(0, len(specs_flags) - 1, 2):
                (s0, f0), (s1, f1) = specs_flags[i], specs_flags[i + 1]
                items.append(("pair", node([scl(u, 0.25) for u in sl(s0)] + [scl(u, 0.5) for u in sl(s1)],
                                           numeric=True), (f0, f1)))
            if len(specs_flags) % 2:
                s, f = specs_flags[-1]
                items.append(("bit", node(sl(s)), f))
        else:
            for s, f in specs_flags:
                items.append(("bit", node(sl(s)), f))
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
            if TH1S["on"]:
                res = theta1_shared(a, p, memo)
                if res is not None:
                    th[p] = res
                    continue
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

    def theta1_shared(a, p, memo):
        return theta1_shared_lits(theta_lits(a, *p, w), memo)

    def theta_direct(packed, carry, last=False):
        if SLAST["on"] and last:
            cur[0] = T - 1
            # optimizer avenue's s_last (= first-last gate_out, packing lf): the last theta
            # layer emits, per digest bit, the one linear form its chi unit reads,
            # s = 2a - b + c (flags and iota folded in, as s/4): 224 features instead of 320
            sfeat = {}
            for p in digest_pos:
                x, y, z = p
                qs = [rp[(x + dx) % 5][y][z] for dx in range(3)]
                fl = [packed[q][1] for q in qs]
                fl[0] ^= flag(*p)
                specs, const = [], 0.0
                for q, f, w_ in zip(qs, fl, (2.0, -1.0, 1.0)):
                    sig, _, n = packed[q]
                    const += w_ * f
                    specs += [spec_on(sig, u, w_ * (1 - 2 * f) / 4) for u in count_parity_units(n, SCALE)]
                if const:
                    specs.append(({}, 1, {}, const / 4))
                sfeat[p] = node(specs, numeric=True)
            return ("s", sfeat), carry_layer(carry, split=True)
        th = {p: (G([sig], count_parity_units(n, SCALE)), bool(f)) for p, (sig, f, n) in packed.items()}
        return th, carry_layer(carry, split=last)

    def d_specs(entry, coef):
        """units of D = parity(P) (times coef, flag folded in) and the flag; entry is a count
        feature (sig, flag, n) or, with NOP, ("lits", [chi-bit literals])"""
        if entry[0] == "lits":
            raws, fl = [], 0
            for sg, f_ in entry[1]:
                raws.append(sg)
                fl ^= int(f_)
            n = len(raws)
            out = []
            for j, u in enumerate(xor_units(n)):
                c = coef * (1 - 2 * fl)
                out.append(({s_: 1.0 for s_ in raws}, float(u.bias),
                            {s_: float(u.value_weights[0]) * c for s_ in raws} if u.value_weights[0] else {},
                            float(u.value_bias) * c))
            return out, fl
        psig, pf, n = entry
        return [spec_on(psig, u, coef * (1 - 2 * pf)) for u in parity_units(n, SCALE)], pf

    def theta_split_x(bits, sums, carry):
        """E = a + D per position (numeric, E/2), D = parity(P) shared by the column pair;
        this step's digest carried from the copy units of a"""
        e = {}
        for p in all_pos:
            (asig, af) = bits[p]
            ent = sums[(p[0], p[2])]
            if ent[0] == "lits":  # flags stay flags: D on raw signals, flag = xor of flags
                ds, pf = d_specs(("lits", [(s_, False) for s_, _ in ent[1]]), 0.5)
                pf = 0
                for _, f_ in ent[1]:
                    pf ^= int(f_)
            else:
                psig, pf, n = ent
                ds = [spec_on(psig, u, 0.5) for u in parity_units(n, SCALE)]
            specs = [spec_on(asig, COPY, 0.5)] + ds
            e[p] = (node(specs, numeric=True), af ^ pf)
        new = [(spec_on(bits[p][0], COPY), bits[p][1]) for p in dpos()]
        return e, carry_layer(carry) + carry_new(new)

    def theta_split_x_exact(bits, sums, carry):
        """E = a + D with the flags folded in (E/2, no flag), so that chi can read E"""
        e = {}
        for p in all_pos:
            (asig, af) = bits[p]
            ds, pf = d_specs(sums[(p[0], p[2])], 0.5)
            specs = [spec_on(asig, COPY, 0.5 * (1 - 2 * af))]
            specs += ds
            if af or pf:
                specs.append(({}, 1, {}, 0.5 * (af + pf)))
            e[p] = node(specs, numeric=True)
        new = [(spec_on(bits[p][0], COPY), bits[p][1]) for p in dpos()]
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
        new = [("lazy", node(terms(p, 0.5), numeric=True), flag(*p)) for p in dpos() if not NOCARRY["on"]]
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
        new = [("lazy4", node(merged([p], 0.5), numeric=True), flag(*p)) for p in dpos() if not NOCARRY["on"]]
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
            if Z11["on"]:  # units-per-bit: no column unit; c goes into the count's zigzag
                specs = []
                pas = {s_: w_ * coef for s_, w_ in g.items()}
                const = 0.0
            elif C5["on"]:  # units-per-bit: ONE unit, r5(c) = c - 2 max(0, c - 5) in [0, 5]
                specs = [(dict(g), -5, {}, -2.0 * coef)]
                pas = {s_: w_ * coef for s_, w_ in g.items()}
                const = 0.0
            else:
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
            const = 0.0 if C5["on"] else 10.0 / SCALE  # offset: count in [0, 32] ([0, 34] with C5)
            for (xc, zc) in (((x + 4) % 5, z), ((x + 1) % 5, (z + 1) % w)):
                sp, pq, cst = column(xc, zc, 1 / SCALE)
                specs = specs + sp
                const += cst
                for s_, w_ in pq.items():
                    pas[s_] = pas.get(s_, 0) + w_
            if Z11["on"]:
                # L = own Ea + c_L + c_R in [0, 22]; z(L) = L - 2 max(0, L - 11) in [0, 11]
                zg = {}
                ea = e[rp[x][p[1]][z]]
                zg[ea] = zg.get(ea, 0.0) + 2.0
                for (xc, zc) in (((x + 4) % 5, z), ((x + 1) % 5, (z + 1) % w)):
                    for y2 in range(5):
                        f_ = e[rp[xc][y2][zc]]
                        zg[f_] = zg.get(f_, 0.0) + 2.0
                specs.append((zg, -11, {}, -2.0 / SCALE))
                const = 0.0
            pas = {s_: w_ for s_, w_ in pas.items() if abs(w_) > 1e-12}
            specs.append(({}, 1, pas, const))
            packed[p] = (node(specs, numeric=True), f, 33 if Z11["on"] else (34 if C5["on"] else 32))
        new = []
        dp = list(dpos()) if not NOCARRY["on"] else []
        if LZ5["on"]:  # units-per-bit: F = o1 + 5 o2 in [0, 24] as F / 32, ONE pass unit per pair
            while len(dp) >= 2:
                p1, p2 = dp.pop(0), dp.pop(0)
                sp1, pas1 = own(p1, 1 / 32)
                sp2, pas2 = own(p2, 5 / 32)
                pas = dict(pas1)
                for s_, w_ in pas2.items():
                    pas[s_] = pas.get(s_, 0) + w_
                pas = {s_: w_ for s_, w_ in pas.items() if abs(w_) > 1e-12}
                new.append(("lz5", node(sp1 + sp2 + [({}, 1, pas, 0.0)], numeric=True), (flag(*p1), flag(*p2))))
        for p in dp:
            sp, pas = own(p, 0.5)
            new.append(("lazy4", node(sp + [({}, 1, pas, 0.0)], numeric=True), flag(*p)))
        return packed, carry_layer(carry) + new

    def reduce_layer(packed, carry):
        """units-per-bit: one extra (narrow) layer that folds each count s in [0, n] into a
        zigzag r(s) = s - 2 max(0, s - W) + 2 max(0, s - 2W) - ... in [0, W], W = 11,
        congruent to s (ceil(n / W) units, knots on integers, flat under silu); the next
        layer takes parity(r) with MIN_PARITY[11] (5 units): e.g. 3 + 5 units per bit
        instead of 16 for n = 33, 4 + 5 instead of 22 for n = 44."""
        out = {}
        W = 11
        for p, (sig, f, n) in packed.items():
            specs = [({sig: float(SCALE)}, 0.0, {}, 1.0 / SCALE)]
            for j in range(1, -(-n // W)):
                specs.append(({sig: float(SCALE)}, float(-j * W), {}, (-2.0 if j % 2 else 2.0) / SCALE))
            out[p] = (node(specs, numeric=True), f, min(n, W))
        return out, carry_layer(carry)

    # ---- combined-2: g-fold after Y1 (sparse-focus): chi1 reads ONE feature g = 2a - b + c ----
    TSPEC, GFD = {}, {}

    def tnode(specs):
        if GF1["on"]:  # g-fold: the theta bit is never built as a node, only its unit specs
            k_ = _TKey()
            TSPEC[k_] = specs
            return k_
        n_ = node(specs)
        TSPEC[n_] = specs
        return n_

    def g_features(th):
        gf = {}
        for q in all_pos:
            x, y, z = q
            specs, const = [], 0.0
            for dx, wgt in zip(range(3), (2.0, -1.0, 1.0)):
                sig, f = th[rp[(x + dx) % 5][y][z]]
                const += wgt * int(f)
                specs += scale_specs(TSPEC[sig], wgt * (1 - 2 * int(f)) / GSC)
            gf[q] = (node(specs, numeric=True), const)
        return gf

    def theta_split_y_u(e, carry):
        if True:  # one output per distinct E (round 1: 1464 of 1600)
            memo, th = {}, {}
            for p, (sig, f) in e.items():
                if isinstance(sig, tuple):  # ("u", u node, j): both bits of a u pair
                    kind, un, j = sig
                    if un not in memo and kind == "u3":
                        # u = 2 (a1 + 2 a2) - 7 D on f = u / 8, all knots on lattice points:
                        # A = max(0,u), B = max(0,-u), B' = max(0,-u-1), D = B - B',
                        # t2 = max(0,u-2)(1-u/8) + max(0,-u-3)(9/8+u/8), t1 = (A + B')/2 - 2 t2
                        u3a, u3b, u3b1 = Unit((8,), 0, (0,), 1), Unit((-8,), 0, (0,), 1), Unit((-8,), -1, (0,), 1)
                        u3c, u3c1 = Unit((8,), -2, (-1,), 1), Unit((-8,), -3, (1,), 1.125)
                        t2 = tnode([spec_on(un, u3c), spec_on(un, u3c1)])
                        t1 = tnode([spec_on(un, u3a, 0.5), spec_on(un, u3b1, 0.5), spec_on(un, u3c, -2.0),
                                   spec_on(un, u3c1, -2.0)])
                        dd = tnode([spec_on(un, u3b), spec_on(un, u3b1, -1.0)])
                        memo[un] = (t1, t2, dd)
                    if un not in memo:
                        if kind == "u1":  # a2 = 0 (constant own bit): u in {-3,-2,0,1}, C unused
                            t2 = tnode([spec_on(un, U_Cn)])
                            t1 = tnode([spec_on(un, U_A), spec_on(un, U_B), spec_on(un, U_Cn, -2.0)])
                        else:
                            t2 = tnode([spec_on(un, U_C), spec_on(un, U_Cn)])
                            t1 = tnode([spec_on(un, U_A), spec_on(un, U_B), spec_on(un, U_C, -2.0),
                                       spec_on(un, U_Cn, -2.0)])
                        memo[un] = (t1, t2)
                    th[p] = (memo[un][j], bool(f))
                    continue
                if sig not in memo:
                    if GF1["on"]:
                        memo[sig] = tnode([spec_on(sig, Unit((2,), 0, (-2,), 2))])
                    else:
                        memo[sig] = G([sig], y_units())
                th[p] = (memo[sig], bool(f))
            if GF1["on"]:  # g-fold, inlined so that the X1 features stay inputs of this block
                gf = {}
                for q in all_pos:
                    x, y, z = q
                    specs, const = [], 0.0
                    for dx, wgt in zip(range(3), (2.0, -1.0, 1.0)):
                        sig_, f_ = th[rp[(x + dx) % 5][y][z]]
                        const += wgt * int(f_)
                        specs += scale_specs(TSPEC[sig_], wgt * (1 - 2 * int(f_)) / GSC)
                    gf[q] = (node(specs, numeric=True), const)
                # return ONLY the g features: every returned node becomes a layer output
                return {("gf",): gf}, carry_layer(carry)
            return th, carry_layer(carry)
        th = {p: (G([sig], [Unit((2,), 0, (-2,), 2)]), bool(f)) for p, (sig, f) in e.items()}
        return th, carry_layer(carry)

    def y_units():
        us = [Unit((2,), 0, (-2,), 2)]
        if YFLAT["on"] and (cur[0] + 1) % YFLAT["every"] == 0 and cur[0] >= 0:
            us.append(Unit((2,), -2, (0,), 4))
        return us

    def theta_split_y(e, carry):
        if UP1["on"]:  # combined-2: optimizer's round-1 u pairs (one output per distinct E)
            return theta_split_y_u(e, carry)
        memo, th = {}, {}
        for p, (sig, f) in e.items():
            if not dedupe or sig not in memo:  # dedupe: equal E nodes share one output
                memo[sig] = G([sig], y_units())
            th[p] = (memo[sig], bool(f))
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
        new = [(spec(chi[p], CHI), flag(*p)) for p in dpos()]
        return packed, carry_layer(carry) + carry_new(new, m2)

    def chi_to_split(th, carry):
        """chi bits (all positions) and column-pair counts P for a split theta"""
        chi = {p: chi_lits(th, *p) for p in all_pos}
        bits = {p: (node([spec(chi[p], CHI)]), flag(*p)) for p in all_pos}
        sums = {}
        for x in range(5):
            for z in range(w):
                qs = pair_of((x, 0, z))
                f = sum(flag(*q) for q in qs) % 2
                if NOP["on"]:
                    sums[(x, z)] = ("lits", [bits[q] for q in qs])
                else:
                    sums[(x, z)] = (node([spec(chi[q], CHI, 1 / SCALE) for q in qs], numeric=True), f, len(qs))
        return bits, sums, carry_layer(carry)

    # ---- column packing (tricks-critic): the X layer copies every chi bit (one unit per
    # bit), and a pair p = a + 2b in {0,1,2,3} is decoded with the same two units:
    # b = relu(p - 1)(4 - p)/2, a = relu(p) - 2b. The fifth bit of a column comes from the
    # column count c (which the D units need anyway): a4 = (c - p1 - p2) + b1 + b2, one
    # BOS-gated unit. So the chi layer emits 3 features per column (c, p1, p2) instead of
    # 5 bits + 1 column-pair count, at no extra unit in either layer.
    SC_C, SC_P = 8.0, 4.0  # c / 8 and p / 4: dyadic, <= 1

    def chi_to_split_cp(th, carry):
        gf = th.get(("gf",))
        chi = {p: chi_lits(th, *p) for p in all_pos} if gf is None else {}

        def cs(q, coef):  # the chi unit of position q (g-fold: one feature, h(G) = max(0,G)(3-G)/2)
            if gf is None:
                return spec(chi[q], CHI, coef)
            gsig, c0 = gf[q]
            return ({gsig: GSC}, c0, {gsig: -0.5 * GSC * coef}, (1.5 - 0.5 * c0) * coef)
        cols = {}
        for x in range(5):
            for z in range(w):
                qs = [(x, y, z) for y in range(5)]
                c = node([cs(q, 1 / SC_C) for q in qs], numeric=True)
                p1 = node([cs(qs[0], 1 / SC_P), cs(qs[1], 2 / SC_P)], numeric=True)
                p2 = node([cs(qs[2], 1 / SC_P), cs(qs[3], 2 / SC_P)], numeric=True)
                cols[(x, z)] = (c, p1, p2)
        return cols, carry_layer(carry)

    def _scl(s, c):
        return (s[0], s[1], {q: v * c for q, v in s[2].items()}, s[3] * c)

    def cp_decoders(cols, x, z):
        """the 5 raw chi bits of column (x, z) as sums of (unit spec, coef): 5 units in all"""
        c, p1, p2 = cols[(x, z)]
        lin1 = ({p1: SC_P}, 0.0, {}, 1.0)  # relu(p1) * 1 = p1
        hi1 = ({p1: SC_P}, -1.0, {p1: -SC_P / 2}, 2.0)  # relu(p1 - 1)(4 - p1)/2 = b1
        lin2 = ({p2: SC_P}, 0.0, {}, 1.0)
        hi2 = ({p2: SC_P}, -1.0, {p2: -SC_P / 2}, 2.0)
        lin4 = ({}, 1.0, {c: SC_C, p1: -SC_P, p2: -SC_P}, 0.0)  # c - p1 - p2 (gate on BOS)
        return {0: [(lin1, 1.0), (hi1, -2.0)], 1: [(hi1, 1.0)], 2: [(lin2, 1.0), (hi2, -2.0)],
                3: [(hi2, 1.0)], 4: [(lin4, 1.0), (hi1, 1.0), (hi2, 1.0)]}

    def cp_d_specs(cols, x, z, coef):
        """coef * parity(P), P = c(x-1, z) + c(x+1, z+1) in [0, 10] (glu_xor, 5 units;
        with P1DC on: and-of-parities' 4-unit form, knots between integers)"""
        cl, cr = cols[((x + 4) % 5, z)][0], cols[((x + 1) % 5, (z + 1) % w)][0]
        g = {cl: SC_C, cr: SC_C}
        if P1DC["on"] and (P1K["k"] is None or cur[0] + 1 >= T - P1K["k"]) and (
            P1DC.get("last") is None or CPMIN.get("cur", 0) >= T - 2  # steps avenue: p1dcl
        ):
            us, c0 = P1DC["forms"][10]
            specs = [({cl: gw * SC_C, cr: gw * SC_C}, gb, {cl: vw * SC_C * coef, cr: vw * SC_C * coef} if vw else {},
                      vb * coef) for gw, gb, vw, vb in us]
            if c0:
                specs.append(({}, 1.0, {}, c0 * coef))
            return specs
        specs = [(dict(g), 0.0, {cl: -SC_C * coef, cr: -SC_C * coef}, 2.0 * coef)]
        specs += [(dict(g), -2.0 * j, {}, 4.0 * coef) for j in range(1, 5)]
        return specs

    def theta_split_x_cp(cols, carry, exact):
        """theta_split_x(_exact) on column-packed chi bits"""
        e = {}
        for x in range(5):
            for z in range(w):
                pf = sum(flag(*q) for q in pair_of((x, 0, z))) % 2
                dec = cp_decoders(cols, x, z)
                for y in range(5):
                    af = flag(x, y, z)
                    if exact:  # flags folded in: E = (a ^ af) + (D ^ pf)
                        ka, kd = 0.5 * (1 - 2 * af), 0.5 * (1 - 2 * pf)
                        specs = [_scl(u, ka * m) for u, m in dec[y]] + cp_d_specs(cols, x, z, kd)
                        if af or pf:
                            specs.append(({}, 1.0, {}, 0.5 * (af + pf)))
                        e[(x, y, z)] = node(specs, numeric=True)
                    else:
                        specs = [_scl(u, 0.5 * m) for u, m in dec[y]] + cp_d_specs(cols, x, z, 0.5)
                        e[(x, y, z)] = (node(specs, numeric=True), af ^ pf)
        new = []
        for p in dpos():
            x, y, z = p
            new.append(([_scl(u, m) for u, m in cp_decoders(cols, x, z)[y]], flag(*p)))
        return e, carry_layer(carry) + carry_new(new)

    def chi_to_shared(th, carry):
        """chi bits (all positions), column-pair counts T, and this step's digest"""
        bits, sums, carry = chi_to_split(th, carry)
        chi = {p: chi_lits(th, *p) for p in dpos()}
        new = [(spec(chi[p], CHI), flag(*p)) for p in dpos()]
        return bits, sums, carry + carry_new(new)

    def walsh_last(packed, carry):
        """the last round as ONE layer (idea of the linear-fold avenue): with a, b, c the
        theta bits parity(S) of the digest's chi, chi = (a - a^b + a^c + a^b^c) / 2, and
        each xor is the parity of a sum of counts: 6 + 11 + 11 + 17 units per digest bit"""
        bits = decode(carry) if not NOCARRY["on"] else []
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
        bits = decode(carry) if not NOCARRY["on"] else []
        if isinstance(th, tuple) and th[0] == "s":  # s_last: one unit on s/4 per digest bit
            for p in digest_pos:
                s_ = th[1][p]
                bits.append(node([({s_: 4.0}, 0, {s_: -2.0}, 1.5)]))
            return bits
        for p in digest_pos:
            lits = chi_lits(th, *p)
            if flag(*p):
                lits = [neg(lits[0])] + lits[1:]
            bits.append(node([spec(lits, CHI)]))
        return bits

    # split theta for step 0 reads literals: E = a + D with D an xor of message bits
    def theta1_split_x(a, carry):
        memo_e, e = {}, {}
        if UP1["on"]:  # u pairs of live own bits of a column pair (message bits, shared D)
            info, cinfo = {}, {}
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
                if not isinstance(alit, int):
                    info[p] = (alit[0], dsigs, flip ^ int(alit[1]))
                else:
                    cinfo[p] = (dsigs, flip ^ alit)

            def d_specs(ds, coef):
                n = len(ds)
                us = [COPY] if n == 1 else par_units(n)
                return [({s: u.weights[k] for k, s in enumerate(ds) if u.weights[k]}, u.bias,
                         {s: u.value_weights[k] * coef for k, s in enumerate(ds) if u.value_weights[k]},
                         u.value_bias * coef) for u in us]
            for x in range(5):
                for z in range(w):
                    live = [(x, y, z) for y in range(5) if (x, y, z) in info]
                    consts = [(x, y, z) for y in range(5) if (x, y, z) in cinfo]
                    if UP1.get("d", True) and consts and len(live) >= 2:
                        # two live bits AND the constant-own positions in ONE feature:
                        # u = 2 (a1 + 2 a2) - 7 D (injective: even >= 0 iff D = 0), see "u3"
                        pa, pb = live.pop(0), live.pop(0)
                        (a1, ds, f1), (a2, _, f2) = info[pa], info[pb]
                        assert all(cinfo[q][0] == ds for q in consts)
                        un = node([({a1: 2, a2: 4}, 0, {}, 0.125)] + d_specs(ds, -0.875), numeric=True)
                        e[pa] = (("u3", un, 0), f1)
                        e[pb] = (("u3", un, 1), f2)
                        for q in consts:
                            e[q] = (("u3", un, 2), cinfo[q][1])
                    elif UP1.get("c", False) and len(live) % 2 and consts:
                        # the odd live bit pairs with the constant-own positions (their theta
                        # is D itself): u = a - 3 D, t_a = A + B - 2 C', t_const = C'
                        pl = live.pop()
                        a1, ds, f1 = info[pl]
                        assert all(cinfo[q][0] == ds for q in consts)
                        un = node([({a1: 1}, 0, {}, 0.25)] + d_specs(ds, -0.75), numeric=True)
                        e[pl] = (("u1", un, 0), f1)
                        for q in consts:
                            e[q] = (("u1", un, 1), cinfo[q][1])
                    for i in range(0, len(live) - 1, 2):
                        (a1, ds, f1), (a2, ds2, f2) = info[live[i]], info[live[i + 1]]
                        assert ds == ds2 and ds
                        n = len(ds)
                        us = [COPY] if n == 1 else par_units(n)
                        specs = [({a1: 1, a2: 2}, 0, {}, 0.25)]
                        for u in us:
                            specs.append(({s: u.weights[k] for k, s in enumerate(ds) if u.weights[k]}, u.bias,
                                          {s: u.value_weights[k] * -0.75 for k, s in enumerate(ds) if u.value_weights[k]},
                                          u.value_bias * -0.75))
                        un = node(specs, numeric=True)
                        e[live[i]] = (("u", un, 0), f1)
                        e[live[i + 1]] = (("u", un, 1), f2)
        for p in all_pos:
            if p in e:
                continue
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
                memo_e[key] = node(specs, numeric=True)
            e[p] = (memo_e[key], flip)
        return e, carry

    # ---- packed round-1 split (tricks-critic): X1 emits the message bits of each column in
    # pairs p = a0 + 2 a1 (one unit per pair instead of two copies) plus D1 per column pair;
    # Y1 decodes theta = bit ^ D with 3 units per pair (instead of 2), exactly:
    #   u1 = relu(p)(1 - 2D), u2 = relu(p - 1 - 4D)(4 - p)/2, u3 = relu(4D - 2 - p)(1 + p)/2,
    #   theta_1 = u2 + u3, theta_0 = u1 - 2 u2 - 2 u3 + 3 D,
    # (u2 is hi(p) on D = 0, u3 is hi(3 - p) = 1 - hi(p) on D = 1, u1 = +-p), where the D term
    # is the unit relu(D) that the column's capacity-lane theta (= D) has anyway.
    def theta1_split_packed(a):
        dn = {}
        for x in range(5):
            for z in range(w):
                flip, cnt = 0, {}
                for q in pair_of((x, 0, z)):
                    lit = a[q[0]][q[1]][q[2]]
                    if isinstance(lit, int):
                        flip ^= lit
                    else:
                        flip ^= int(lit[1])
                        cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
                dsigs = sorted((s for s, c in cnt.items() if c), key=lambda s: s.uid)
                assert len(dsigs) >= 2
                dn[(x, z)] = (G(dsigs, par_units(len(dsigs))), flip)
        items = {}  # (x, z) -> [("pair", node, (y0, y1)) | ("single", node, y)]
        for x in range(5):
            for z in range(w):
                live = []
                for y in range(5):
                    lit = a[x][y][z]
                    if not isinstance(lit, int):
                        assert not lit[1]
                        live.append((y, lit[0]))
                its = []
                for i in range(0, len(live) - 1, 2):
                    (y0, s0), (y1, s1) = live[i], live[i + 1]
                    its.append(("pair", G([s0, s1], [Unit((1, 2), 0, (0, 0), 0.25)], numeric=True), (y0, y1)))
                if len(live) % 2:
                    y0, s0 = live[-1]
                    its.append(("single", G([s0], [COPY]), y0))
                items[(x, z)] = its
        th, cap = {}, {}
        sc = lambda s, c: (s[0], s[1], {q: v * c for q, v in s[2].items()}, s[3] * c)
        for x in range(5):
            for z in range(w):
                dsig, fd = dn[(x, z)]
                done = set()
                for kind, sig, ys in items[(x, z)]:
                    if kind == "single":
                        th[(x, ys, z)] = (G([sig, dsig], [Unit((1, 1), 0, (-1, -1), 2)]), bool(fd))
                        done.add(ys)
                        continue
                    y0, y1 = ys
                    u1 = ({sig: 4.0}, 0.0, {dsig: -2.0}, 1.0)
                    u2 = ({sig: 4.0, dsig: -4.0}, -1.0, {sig: -2.0}, 2.0)
                    u3 = ({dsig: 4.0, sig: -4.0}, -2.0, {sig: 2.0}, 0.5)
                    dt = ({dsig: 1.0}, 0.0, {}, 1.0)
                    t0 = node([u1, sc(u2, -2.0), sc(u3, -2.0), sc(dt, 3.0)])
                    t1 = node([u2, u3])
                    th[(x, y0, z)] = (t0, bool(fd))
                    th[(x, y1, z)] = (t1, bool(fd))
                    done.update(ys)
                for y in range(5):
                    if y in done:
                        continue
                    lit = a[x][y][z]
                    assert isinstance(lit, int)
                    if (x, z) not in cap:
                        cap[(x, z)] = G([dsig], [COPY])
                    th[(x, y, z)] = (cap[(x, z)], bool(fd ^ lit))
        return th

    def xof_fn(msg: list) -> list:
        lanes = initial_state(k, msg)
        needs = [all_pos] * (T - 1) + [digest_pos]  # chi outputs each round needs
        carry: list = []
        cur[0] = -1  # round 0's theta (raw message bits) comes before any chi
        if kinds[0] == "splitp":
            th = theta1_split_packed(lanes)
        elif kinds[0] == "split":
            e, carry = theta1_split_x(lanes, carry)
            th, carry = theta_split_y(e, carry)
        else:
            th = theta1_direct(lanes, chi_needs(needs[0]))
        step = 0  # a round index (T rounds)
        while step < T - 1:  # chi layer of round step, then the theta layer(s) of step + 1
            CPMIN["dig"] = step // R  # steps avenue: the XOF step of this chi's digest (LATEPK)
            nxt = kinds[step + 1]
            cur[0] = step
            if walsh and step == T - 2:  # chi layer, then the merged last round
                packed, carry = chi_to_direct(th, chi_needs(digest_pos), carry)
                cur[0] = T - 1
                return walsh_last(packed, carry)
            cmin = CPMIN["step"] if CPMIN["step"] >= 0 else T + CPMIN["step"]
            use_cp = cp and (CPK["k"] is None or step + 1 >= T - CPK["k"]) and step + 1 >= cmin
            if nxt == "split":
                if use_cp:
                    cols, carry = chi_to_split_cp(th, carry)
                    CPMIN["cur"] = step + 1
                    e, carry = theta_split_x_cp(cols, carry, exact=False)
                else:
                    bits, sums, carry = chi_to_split(th, carry)
                    e, carry = theta_split_x(bits, sums, carry)
                th, carry = theta_split_y(e, carry)
            elif nxt == "lazy":  # X (exact E), lazy chi, then the next step's theta
                assert step + 2 <= T - 1, "a lazy step needs a step after it"
                if use_cp:
                    cols, carry = chi_to_split_cp(th, carry)
                    CPMIN["cur"] = step + 1
                    e, carry = theta_split_x_cp(cols, carry, exact=True)
                else:
                    bits, sums, carry = chi_to_split(th, carry)
                    e, carry = theta_split_x_exact(bits, sums, carry)
                cur[0] = step + 1
                packed, carry = lazy_chi(e, chi_needs(needs[step + 2]), carry)
                if walsh and step + 2 == T - 1:
                    cur[0] = T - 1
                    return walsh_last(packed, carry)
                th, carry = theta_direct(packed, carry, last=(step + 2 == T - 1))
                step += 1
            elif nxt in ("lazy4", "lazy4c"):  # X (exact E), one-unit lazy chi, next theta
                assert step + 2 <= T - 1, "a lazy step needs a step after it"
                if use_cp:
                    cols, carry = chi_to_split_cp(th, carry)
                    CPMIN["cur"] = step + 1
                    e, carry = theta_split_x_cp(cols, carry, exact=True)
                else:
                    bits, sums, carry = chi_to_split(th, carry)
                    e, carry = theta_split_x_exact(bits, sums, carry)
                cur[0] = step + 1
                packed, carry = (lazy_chi4 if nxt == "lazy4" else lazy_chi4c)(e, chi_needs(needs[step + 2]), carry)
                if RP["on"] and step + 2 == T - 1:  # units-per-bit: reduce, then parity
                    packed, carry = reduce_layer(packed, carry)
                th, carry = theta_direct(packed, carry, last=(step + 2 == T - 1))
                step += 1
            elif nxt == "shared":
                bits, sums, carry = chi_to_shared(th, carry)
                th, carry = theta_shared(bits, sums, carry)
            else:
                packed, carry = chi_to_direct(th, chi_needs(needs[step + 1]), carry)
                th, carry = theta_direct(packed, carry, last=(step + 1 == T - 1))
            step += 1
        cur[0] = T - 1
        return chi_last(th, carry)

    return xof_fn


def make(kinds_fn, pairs=True, walsh=False, m1=2, cp=False, dedupe=False, m2=None):
    """kinds_fn(n) gives the theta layout of each of the n = steps * k.n rounds"""
    def variant(k, depth):
        return (build(k, depth, kinds_fn(depth * k.n), pairs, walsh, m1, cp=cp, dedupe=dedupe, m2=m2),
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


# ---- tricks-critic: column packing (cp) of the chi bits the X layer copies, and one
# Y1 output per distinct E1 node (dd) ----
lazy4c_middle_m3_mp_cp = with_minpar(make(middle("lazy4c"), m1=3, cp=True))
split_first_lazy4c_m3_mp_cp = with_minpar(make(
    lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d, m1=3, cp=True, dedupe=True))
split_first_middle_mp_cp = with_minpar(make(lambda d: ["split"] * (d - 1) + ["direct"], cp=True, dedupe=True))
split_first_middle_m3_mp_cp = with_minpar(make(lambda d: ["split"] * (d - 1) + ["direct"], m1=3, cp=True, dedupe=True))
lazy4c_middle_cp = make(middle("lazy4c"), cp=True)
split_first_middle_cp = make(lambda d: ["split"] * (d - 1) + ["direct"], cp=True, dedupe=True)
split_first_middle_dd = make(lambda d: ["split"] * (d - 1) + ["direct"], dedupe=True)  # Y1 dedupe only
split_first_middle_mp_dd = with_minpar(split_first_middle_dd)
split_first_lazy4c_m3_mp_dd = with_minpar(make(
    lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d, m1=3, dedupe=True))

# sparse-side variants (no min-parity, pairs)
split_first_lazy_cp = make(lambda d: ["split"] * (d - 2) + ["lazy", "direct"] if d > 2 else ["direct"] * d, cp=True, dedupe=True)
split_first_lazy4c_cp = make(lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d, cp=True, dedupe=True)
lazy_middle_cp = make(middle("lazy"), cp=True)
split_first_lazy_dd = make(lambda d: ["split"] * (d - 2) + ["lazy", "direct"] if d > 2 else ["direct"] * d, dedupe=True)

# ---- tricks-critic: packed round-1 split (X1 emits bit pairs, Y1 decodes theta = bit ^ D
# with 3 units per pair) on top of cp ----
def _sp(kinds_fn):
    def f(d):
        ks = kinds_fn(d)
        return ["splitp"] + ks[1:] if ks and ks[0] == "split" else ks
    return f


_sfm = lambda d: ["split"] * (d - 1) + ["direct"]
_sfl4c = lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d
_sfl = lambda d: ["split"] * (d - 2) + ["lazy", "direct"] if d > 2 else ["direct"] * d
split_first_middle_m3_mp_cp_sp = with_minpar(make(_sp(_sfm), m1=3, cp=True))
split_first_middle_mp_cp_sp = with_minpar(make(_sp(_sfm), cp=True))
split_first_middle_cp_sp = make(_sp(_sfm), cp=True)
split_first_lazy4c_m3_mp_cp_sp = with_minpar(make(_sp(_sfl4c), m1=3, cp=True))
split_first_lazy4c_cp_sp = make(_sp(_sfl4c), cp=True)
split_first_lazy_cp_sp = make(_sp(_sfl), cp=True)
split_first_middle_sp = make(_sp(_sfm))  # packed round 1 only (sparse side)
split_first_lazy_sp = make(_sp(_sfl))
lazy4c_middle_m3_cp = make(middle("lazy4c"), m1=3, cp=True)

# digest 1 packed 3 per feature (carried 5 layers), digest 2 in pairs (carried 1 layer)
split_first_middle_m3m2_mp_cp_sp = with_minpar(make(_sp(_sfm), m1=3, m2=2, cp=True))
split_first_middle_m3m2_cp_sp = make(_sp(_sfm), m1=3, m2=2, cp=True)
split_first_lazy4c_m3_cp_sp = make(_sp(_sfl4c), m1=3, cp=True)
# ---- units-per-bit: X layers without the column-pair count features (dense/sparse trade) ----
def with_nop(variant):
    def v(k, depth):
        NOP["on"] = True
        return variant(k, depth)
    return v


lazy4c_middle_m3_mp_nop = with_nop(lazy4c_middle_m3_mp)
split_first_lazy4c_m3_mp_nop = with_nop(split_first_lazy4c_m3_mp)
split_first_middle_mp_nop = with_nop(split_first_middle_mp)
split_first_middle_m3_mp_nop = with_nop(split_first_middle_m3_mp)


def with_mpc(variant):
    def v(k, depth):
        MPC["on"] = True
        return variant(k, depth)
    return v


split_first_middle_mp_mpc = with_mpc(split_first_middle_mp)
split_first_middle_mp_nop_mpc = with_mpc(split_first_middle_mp_nop)
split_first_middle_m3_mp_mpc = with_mpc(split_first_middle_m3_mp)
split_first_middle_m3_mp_nop_mpc = with_mpc(split_first_middle_m3_mp_nop)


def with_c5(variant):
    def v(k, depth):
        C5["on"] = True
        return variant(k, depth)
    return v


lazy4c_middle_m3_mp_c5 = with_c5(lazy4c_middle_m3_mp)
lazy4c_middle_m3_mp_nop_c5 = with_c5(lazy4c_middle_m3_mp_nop)
split_first_lazy4c_m3_mp_c5 = with_c5(split_first_lazy4c_m3_mp)
split_first_lazy4c_m3_mp_nop_c5 = with_c5(split_first_lazy4c_m3_mp_nop)


def with_lz5(variant):
    def v(k, depth):
        LZ5["on"] = True
        LZ5_DECODER()  # built outside tracing
        return variant(k, depth)
    return v


lazy4c_middle_m3_mp_c5_lz5 = with_lz5(lazy4c_middle_m3_mp_c5)
lazy4c_middle_m3_mp_nop_c5_lz5 = with_lz5(lazy4c_middle_m3_mp_nop_c5)
split_first_lazy4c_m3_mp_c5_lz5 = with_lz5(split_first_lazy4c_m3_mp_c5)
split_first_lazy4c_m3_mp_nop_c5_lz5 = with_lz5(split_first_lazy4c_m3_mp_nop_c5)


def with_z11(variant):
    def v(k, depth):
        Z11["on"] = True
        MPC["on"] = True
        return variant(k, depth)
    return v


lazy4c_middle_m3_mp_z11 = with_z11(lazy4c_middle_m3_mp)
split_first_lazy4c_m3_mp_z11 = with_z11(split_first_lazy4c_m3_mp)
lazy4c_middle_m3_mp_z11_lz5 = with_z11(with_lz5(lazy4c_middle_m3_mp))
split_first_lazy4c_m3_mp_z11_lz5 = with_z11(with_lz5(split_first_lazy4c_m3_mp))
lazy4c_middle_m3_mp_nop_z11_lz5 = with_z11(with_lz5(lazy4c_middle_m3_mp_nop))
split_first_lazy4c_m3_mp_nop_z11_lz5 = with_z11(with_lz5(split_first_lazy4c_m3_mp_nop))


def with_rp(variant):
    def v(k, depth):
        RP["on"] = True
        return variant(k, depth)
    return v


# depth 8: split round 1, X + lazy chi (Z11) in round 2, then fold | parity | chi
split_first_lazy4c_m3_mp_z11_lz5_rp = with_rp(split_first_lazy4c_m3_mp_z11_lz5)
split_first_lazy4c_m3_mp_nop_z11_lz5_rp = with_rp(split_first_lazy4c_m3_mp_nop_z11_lz5)
split_first_lazy4c_m3_mp_z11_rp = with_rp(split_first_lazy4c_m3_mp_z11)
split_first_lazy4c_m3_mp_nop_z11 = with_z11(split_first_lazy4c_m3_mp_nop)
split_first_lazy4c_m3_mp_nop_z11_rp = with_rp(split_first_lazy4c_m3_mp_nop_z11)
lazy4c_middle_m3_mp_nop_z11 = with_z11(lazy4c_middle_m3_mp_nop)
split_first_lazy4c_mp_z11_rp = with_rp(with_z11(split_first_lazy4c_mp))
split_first_lazy4c_m3_z11_rp = with_rp(with_z11(split_first_lazy4c_m3))

# depth 8 with the fold layer: the lazy chi needs no linear-part reduction at all (lazy4,
# counts in [0, 44]); the fold takes 4 units instead of 3
split_first_lazy4_m3 = make(lambda d: ["split"] * (d - 2) + ["lazy4", "direct"] if d > 2 else ["direct"] * d, m1=3)
split_first_lazy4_m3_mp_rp = with_rp(with_mpc(with_minpar(split_first_lazy4_m3)))
split_first_lazy4_m3_mp_nop_rp = with_rp(with_mpc(with_nop(with_minpar(split_first_lazy4_m3))))
split_first_lazy4_mp_rp = with_rp(with_mpc(with_minpar(split_first_lazy4)))
split_first_lazy4_mp_nop_rp = with_rp(with_mpc(with_nop(with_minpar(split_first_lazy4))))
split_first_lazy4_rp = with_rp(with_mpc(split_first_lazy4))


def with_th1s(variant):
    def v(k, depth):
        TH1S["on"] = True
        return variant(k, depth)
    return v


lazy4c_middle_m3_mp_z11_th = with_th1s(lazy4c_middle_m3_mp_z11)
lazy4c_middle_m3_mp_nop_z11_th = with_th1s(lazy4c_middle_m3_mp_nop_z11)
lazy4c_middle_m3_mp_nop_z11_lz5_th = with_th1s(lazy4c_middle_m3_mp_nop_z11_lz5)
# depth 7 with the fold layer: direct round 1 (column-shared theta), X + lazy chi, fold,
# parity, chi
lazy4_middle_m3 = make(middle("lazy4"), m1=3)
lazy4_middle_m3_mp_th_rp = with_th1s(with_rp(with_mpc(with_minpar(lazy4_middle_m3))))
lazy4_middle_m3_mp_nop_th_rp = with_th1s(with_rp(with_mpc(with_nop(with_minpar(lazy4_middle_m3)))))
lazy4c_middle_mp_z11 = with_z11(lazy4c_middle_mp)
split_first_lazy4c_mp_z11 = with_z11(split_first_lazy4c_mp)
split_first_middle_mpc = with_mpc(split_first_middle)  # sparse-lean depth 8 (no min-parity on raw bits)
lazy4c_middle_mp_c5 = with_c5(lazy4c_middle_mp)
split_first_lazy4c_mp_c5 = with_c5(split_first_lazy4c_mp)


def with_th1p(variant, form="F1"):
    """theta1-t8: pool form for round-1 theta (all 2 <= T <= 8 columns); needs the TH1S call path"""
    def v(k, depth):
        TH1S["on"] = True
        TH1P["on"] = True
        TH1P["form"] = form
        return variant(k, depth)
    return v


def with_slast(variant):
    def v(k, depth):
        SLAST["on"] = True
        return variant(k, depth)
    return v


def with_slast_auto(variant):
    """wave 4 (word-size): s_last only where it pays, i.e. when the digest is shorter than the
    5 lanes its chi reads (d < 5w: log_w >= 2); at log_w 0-1 the digest is the whole row y = 0
    and the gate features cost as many features as the theta bits, plus more nonzeros"""
    def v(k, depth):
        SLAST["on"] = k.d < 5 * k.w
        return variant(k, depth)
    return v


def with_up1(variant):
    def v(k, depth):
        UP1["on"] = True
        return variant(k, depth)
    return v


def with_s4(variant):
    def v(k, depth):
        S4["on"] = True
        ST4_DECODER()  # built outside tracing
        return variant(k, depth)
    return v


def with_gf1(variant):
    def v(k, depth):
        GF1["on"] = True
        return variant(k, depth)
    return v


def with_p1dc(variant):
    """wave 3 (and-of-parities): X2's D = parity(C_L + C_R) in [0, 10] with 4 units (p1d_forms)"""
    def v(k, depth):
        from p1d_forms import FORMS  # imported outside tracing
        P1DC["on"] = True
        P1DC["forms"] = FORMS
        return variant(k, depth)
    return v


def with_p1dc_flat(variant):
    """with_p1dc with the flattest n = 10 form (p1d_forms.FORMS_FLAT)"""
    def v(k, depth):
        from p1d_forms import FORMS_FLAT
        P1DC["on"] = True
        P1DC["forms"] = FORMS_FLAT
        return variant(k, depth)
    return v


def with_p1dp_flat(variant):
    """count parities of range 11 (the depth-8 parity layer) with p1d_forms.FORMS_FLAT[11]"""
    def v(k, depth):
        from p1d_forms import FORMS_FLAT
        P1DP["on"] = True
        P1DP["forms"] = {n: f for n, f in FORMS_FLAT.items() if n == 11}
        return variant(k, depth)
    return v


def with_p1dc_ext(variant):
    """with_p1dc with the n = 6 form + glu_xor ramps (p1d_forms.FORMS_EXT): flat beyond s = 6"""
    def v(k, depth):
        from p1d_forms import FORMS_EXT
        P1DC["on"] = True
        P1DC["forms"] = FORMS_EXT
        return variant(k, depth)
    return v


def with_p1dr(variant):
    """round-1 raw-bit parities of even size (the T = 8 column pairs) with n/2 - 1 units"""
    def v(k, depth):
        from p1d_forms import FORMS_EXTC
        P1DR["on"] = True
        P1DR["forms"] = FORMS_EXTC
        return variant(k, depth)
    return v


def with_p1dc_top(variant):
    """with_p1dc with the n = 6 form at the top of the range (p1d_forms.FORMS_EXTT)"""
    def v(k, depth):
        from p1d_forms import FORMS_EXTT
        P1DC["on"] = True
        P1DC["forms"] = FORMS_EXTT
        return variant(k, depth)
    return v


def with_p1dr_all(variant):
    """with_p1dr, and odd round-1 raw parities (7, 9 bits) on the even form of range n + 1"""
    def v(k, depth):
        from p1d_forms import FORMS_EXTC
        P1DR["on"] = True
        P1DR["odd"] = True
        P1DR["forms"] = FORMS_EXTC
        return variant(k, depth)
    return v


def with_cpk(variant, kk):
    """wave 4 (rounds): column packing only in the X layers of the last kk rounds"""
    def v(k, depth):
        CPK["k"] = kk
        return variant(k, depth)
    return v


def with_p1k(variant, kk):
    """wave 4 (rounds): P1DC's D form only in the X layers of the last kk rounds"""
    def v(k, depth):
        P1K["k"] = kk
        return variant(k, depth)
    return v


def with_yflat(variant, every=1):
    """wave 4 (rounds): flat split Y (2 units per bit) in split rounds r with r % every == 0"""
    def v(k, depth):
        YFLAT["on"] = True
        YFLAT["every"] = every
        return variant(k, depth)
    return v

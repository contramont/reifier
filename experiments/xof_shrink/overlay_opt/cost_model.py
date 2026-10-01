"""Layer cost model and layout optimizer for the XOF SwiGLU builders (optimizer avenue, wave 2).

A SwiGLU layer with `i` inputs (BOS included), `h` hidden units and `o` outputs has
    dense = i (norm) + h * i (wg) + h * i (wv) + o * h (wo) = i + h (2 i + o).
So a feature at the boundary between layers L and L+1 costs h_L + 2 h_{L+1} + 1, and a
hidden unit of layer L costs 2 i_L + o_L. The model below gives, for every layer kind the
builders (xs3/xs4 + this avenue's flags) can emit, the exact hidden and output counts as a
function of the options; `enumerate_designs` searches all compositions for each depth
(exhaustive: the space is small) and reports the Pareto front on dense.

The unit counts are the builders' formulas at log_w = 6 (checked against the harness:
`python cost_model.py --check` prints model vs measured for every measured variant).

Boundary states (what a layer boundary carries besides BOS and the digest carries):
  IN  1144 message bits              E1  round-1 E = a + D (1464 distinct)
  T1  round-1 theta bits (1464)       CP  chi bits + column-pair counts P (1920, or 1120 packed)
  E2  round-2 E values (1600)         T2  round-2 theta bits (1600)
  S2  round-2 counts a + P (1600)     S3  round-3 counts (320) with a range r
  T3  round-3 theta bits (320) or the chi units' linear forms s (224)
  OUT 672 output bits
"""
import argparse
import itertools
import json
from math import ceil

W = 64
N = 25 * W  # 1600 state bits
COLS = 5 * W  # 320 column pairs
DIG = 224  # digest bits per step
TH3 = 320  # theta3 bits the last chi reads


def dense(i, h, o):
    return i + h * (2 * i + o)


# units of one exact binary m-bit decoder in the last layer (decode.py, DP-minimal)
DEC_BIN = {1: 1, 2: 2, 3: 8, 4: 22}  # after unit sharing inside a feature (measured)


def glu_par(n):
    """glu_xor units for the parity of a count in [0, n] (integer knots, flat at the lattice)"""
    return max(1, ceil(n / 2))


def carry_feats(m, bits=DIG):
    return ceil(bits / m)


def decoder_units(m, bits=DIG):
    full, rest = divmod(bits, m)
    return full * DEC_BIN[m] + (DEC_BIN[rest] if rest else 0)


def layers_for(design):
    """(in, hidden, out) per layer for a design dict:
    first: "direct" | "split"      (round 1 on raw bits)
    mid:   "split" | "lazy4c" | "lazy_cols_walsh"   (round 2 / last-round structure)
    mp: min-parity in round 1; pk: packed chi->X pairs; dy: dedupe Y1; s_last; m1: digest packing"""
    mp, pk, dy, s_last, m1 = (design.get(k) for k in ("mp", "pk", "dy", "s_last", "m1"))
    L = []
    carry = 0  # carried digest features entering the next layer
    # ---- round 1 ----
    if design["first"] == "direct":
        L.append([1145, (5569 if mp else 6072) + 2, 1465])  # 1464 distinct theta1 bits
    else:
        u1 = design.get("u1", "")  # "", "u", "uc", "ud": u pairs in round 1 (see xs3)
        dunits = 2161 - 1144 if mp else 2416 - 1144  # D units of the 320 column pairs
        # column pairs of round 1: 184 with 4 live own bits (+1 constant), 136 with 3 (+2)
        if not u1:
            L.append([1145, 1144 + dunits + 2, 1465])  # X1: 1144 copies + D units
            L.append([1465, 1464 + 2, 1465 if dy else 1601])  # Y1
        else:
            copies = 184 * 2 + 136 * 2
            out = {"u": 184 * 3 + 136 * 3, "uc": 184 * 3 + 136 * 2, "ud": 320 * 2}[u1]
            yh = {"u": 184 * 9 + 136 * 6, "uc": 184 * 9 + 136 * 7, "ud": 184 * 9 + 136 * 6}[u1]
            L.append([1145, copies + dunits + 2, out + 1])
            L.append([out + 1, yh + 2, 1465])
    # chi1 -> chi bits (pairs if pk) + P counts
    L.append([L[-1][2], N + 2, (N // 2 if pk else N) + COLS + 1])
    # X2: decode/copy a (1 unit per bit either way) + D (5 units per column pair); digest 1 new
    d1 = carry_feats(m1)
    L.append([L[-1][2], N + 5 * COLS + 2 + (design["mid"] != "split"), N + d1 + 1])  # +1: flag constant (exact E)
    carry = d1
    mid = design["mid"]
    if mid == "cols_walsh":  # xs4 "d5": lazy chi with column parities (counts <= 13), Walsh round 3
        L.append([L[-1][2], 5357, 508])  # measured (3-bit digest 1, base-3 pairs digest 2)
        L.append([508, 13141, 673])  # 53 units per digest bit + decoders (measured)
        return L
    if mid == "split" and design.get("u2"):  # u pairs in round 2: X2 8 units, 3 outputs per column pair
        L[-1] = [L[-1][0], COLS * 8 + 2, COLS * 3 + d1 + 1]
    if mid == "split":  # Y2, chi2 -> exact counts S3 (range 11), digest 2 new
        L.append([L[-1][2], (COLS * 9 if design.get("u2") else N) + carry + 2, N + carry + 1])  # Y2
        d2 = carry_feats(m1)
        L.append([L[-1][2], N + carry + 2, TH3 + carry + d2 + 1])  # chi2
        carry += d2
        r = 11
        lazy_dig = 0
    elif mid == "lazy4c":  # one product unit per chi2 bit + 2 per column + 1 pass per count
        lazy_dig = DIG  # lazy digest-2 values, made exact in the next layer (2 units each)
        L.append([L[-1][2], N + 2 * COLS + TH3 + DIG + carry + 2, TH3 + DIG + carry + 1])
        r = 32
    else:
        raise ValueError(mid)
    # theta3: parity of the counts (+ the lazy digest conversion), outputs theta bits or s
    per = 5 if (design.get("mc") and r == 11) else glu_par(r)  # mc: MIN_PARITY[11] on the counts
    h = TH3 * per + carry + 2 + (1 if s_last else 0)
    out = (DIG if s_last else TH3) + carry + 1
    if lazy_dig:
        h += 2 * lazy_dig
        out += lazy_dig // 2  # the lazy digest leaves as exact pairs
    L.append([L[-1][2], h, out])
    # chi3 + decoders: digest 1 (m1-packed), digest 2 (m1-packed, or pairs after lazy4c)
    dec = decoder_units(m1) + (decoder_units(2) if lazy_dig else decoder_units(m1))
    L.append([L[-1][2], DIG + dec + 1 + 2, 1 + 3 * DIG])
    return L


def total(L):
    return sum(dense(*l) for l in L)


SPACE = {
    "u1": ["", "u", "uc", "ud"],
    "u2": [False, True],
    "first": ["direct", "split"],
    "mid": ["split", "lazy4c", "cols_walsh"],
    "mp": [False, True],
    "pk": [False, True],
    "dy": [False, True],
    "s_last": [False, True],
    "m1": [2, 3, 4],
}


def enumerate_designs():
    best = {}
    for vals in itertools.product(*SPACE.values()):
        d = dict(zip(SPACE, vals))
        if d["first"] == "direct" and (d["dy"] or d["u1"]):
            continue
        if (d["u1"] or d["u2"]) and not (d["pk"] and d["dy"]):
            continue  # the u options are implemented on top of pk + dy
        if d["u2"] and d["mid"] != "split":
            continue  # the lazy chi reads E values
        if d["mid"] == "cols_walsh" and (d["first"] != "direct" or d["m1"] != 3 or d["s_last"]):
            continue  # xs4 "d5" (measured shapes for m1 = 3, k2 = 2)
        L = layers_for(d)
        key = len(L)
        t = total(L)
        if key not in best or t < best[key][0]:
            best[key] = (t, d, L)
    # depth 4 (xs4 d4a_m4k3 + min-parity, wave 1; no X layer, so none of the options apply)
    L4 = [[1145, 5571, 1465], [1465, 3202, 1657], [1657, 24059, 452], [452, 21649, 673]]
    best[4] = (total(L4), {"layout": "xs4 d4a_m4k3_mp (unchanged)"}, L4)
    return best


# measured (harness) layer tables for calibration
MEASURED = {
    "xc:d5_m3k2_mp": ({"first": "direct", "mid": "cols_walsh", "mp": 1, "pk": 0, "dy": 0, "s_last": 0, "m1": 3}, 89244445),
    "xo:d5_m3k2_mp_pk": ({"first": "direct", "mid": "cols_walsh", "mp": 1, "pk": 1, "dy": 0, "s_last": 0, "m1": 3}, 82837245),
    "xo:d5_m3k2_pk": ({"first": "direct", "mid": "cols_walsh", "mp": 0, "pk": 1, "dy": 0, "s_last": 0, "m1": 3}, 84726010),
    "xs3:lazy4c_middle_m3_mp": ({"first": "direct", "mid": "lazy4c", "mp": 1, "pk": 0, "dy": 0, "s_last": 0, "m1": 3}, 69368253),
    "xo:lazy4c_middle_m3_mp_pk": ({"first": "direct", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 0, "s_last": 0, "m1": 3}, 62961053),
    "xo:lazy4c_middle_m3_mp_pks": ({"first": "direct", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 0, "s_last": 1, "m1": 3}, 62220049),
    "xs3:split_first_lazy4c_m3_mp": ({"first": "split", "mid": "lazy4c", "mp": 1, "pk": 0, "dy": 0, "s_last": 0, "m1": 3}, 63651004),
    "xo:split_first_lazy4c_m3_mp_pk": ({"first": "split", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 1, "s_last": 0, "m1": 3}, 56608548),
    "xo:split_first_lazy4c_m3_mp_pks": ({"first": "split", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3}, 55867544),
    "xs3:split_first_middle_mp": ({"first": "split", "mid": "split", "mp": 1, "pk": 0, "dy": 0, "s_last": 0, "m1": 2}, 61082590),
    "xo:split_first_middle_mp_pk": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 0, "m1": 2}, 54041734),
    "xo:split_first_middle_m3_mp_pk": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 0, "m1": 3}, 53665851),
    "xo:split_first_middle_m3_mp_pks": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3}, 53196480),
    "xo:split_first_middle_mp_pks": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 2}, 53707561),
    "xo:split_first_middle_m3_mp_pksu": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "u", "u2": 1}, 50906728),
    "xo:split_first_middle_m3_mp_pksu1": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "u"}, 52403688),
    "xo:split_first_lazy4c_m3_mp_pksu": ({"first": "split", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "u"}, 55074752),
    "xo:split_first_middle_m3_mp_pksuc": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "uc", "u2": 1}, 50431680),
    "xo:split_first_lazy4c_m3_mp_pksuc": ({"first": "split", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "uc"}, 54599704),
    "xo:split_first_middle_mp_pkuc": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 0, "m1": 2, "u1": "uc", "u2": 1}, 51347974),
    "xo:split_first_middle_m3_mp_pksud": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "ud", "u2": 1}, 48792168),
    "xo:split_first_lazy4c_m3_mp_pksud": ({"first": "split", "mid": "lazy4c", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "ud"}, 52960192),
    "xo:split_first_middle_mp_pkud": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 0, "m1": 2, "u1": "ud", "u2": 1}, 49708462),
    "xo:split_first_middle_m3_mp_pksud_mc": ({"first": "split", "mid": "split", "mp": 1, "pk": 1, "dy": 1, "s_last": 1, "m1": 3, "u1": "ud", "u2": 1, "mc": 1}, 48370728),
}


def what_if(L, layer, dh=0, dout=0):
    """dense change if layer `layer` gets dh more hidden units and dout more outputs"""
    M = [list(l) for l in L]
    M[layer][1] += dh
    M[layer][2] += dout
    if layer + 1 < len(M):
        M[layer + 1][0] += dout
    return total(M) - total(L)


# per-column-pair grouping of the X -> Y boundary (xs3 u options): a group costs
# (X units, X outputs, Y units); D units are shared by the column pair and not counted.
GROUPS = {
    "E": (1, 1, 1),    # E = a + D, Y: [E == 1]
    "Ec": (0, 1, 1),   # constant-own positions of a round-1 column pair: E = D (+ flag)
    "u": (1, 1, 4),    # u = (a1 + 2 a2) - 3 D, 4 decode units
    "u1": (1, 1, 3),   # u = a - 3 D (one live bit + the constant positions)
    "u3": (1, 1, 5),   # u = 2 (a1 + 2 a2) - 7 D (two live bits + the constant positions)
}
PARTITIONS = {  # the groupings of a column pair's positions that the builders can emit
    "round1, 4 live + const": [("E",) * 4 + ("Ec",), ("u", "u", "Ec"), ("u3", "u"), ("u3", "E", "E"),
                               ("u1", "u", "E"), ("u", "E", "E", "Ec")],
    "round1, 3 live + consts": [("E",) * 3 + ("Ec",), ("u", "E", "Ec"), ("u", "u1"), ("u3", "E"), ("u1", "E", "E")],
    "round2, 5 live": [("E",) * 5, ("u", "E", "E", "E"), ("u", "u", "E")],
}


def best_partitions(unit_x, feat, unit_y):
    """cost of each partition given the marginal costs of an X unit, an X->Y feature and a
    Y unit; returns {case: sorted [(cost, partition)]}"""
    out = {}
    for case, parts in PARTITIONS.items():
        rows = []
        for p in parts:
            x = sum(GROUPS[g][0] for g in p)
            o = sum(GROUPS[g][1] for g in p)
            y = sum(GROUPS[g][2] for g in p)
            rows.append((x * unit_x + o * feat + y * unit_y, p))
        out[case] = sorted(rows)
    return out


def marginal(L):
    """cost of one more feature at each boundary and of one more unit in each layer"""
    feat = [L[k][1] + 2 * L[k + 1][1] + 1 for k in range(len(L) - 1)]
    unit = [2 * i + o for i, _, o in L]
    return feat, unit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        for name, (d, meas) in MEASURED.items():
            L = layers_for(d)
            print(f"{name:36s} model {total(L):>11,} measured {meas:>11,} diff {total(L) - meas:>+9,}")
    best = enumerate_designs()
    # headroom probes on the best depth-8 layout (hypothetical constructions)
    t8, d8, L8 = best[8]
    print("what-if (depth 8):")
    print("  3-unit u-pair decode (none exists):", f"{what_if(L8, 1, dh=-(184 * 2)) + what_if(L8, 4, dh=-640):+,}")
    print("  4 units per column-pair D instead of 5:", f"{what_if(L8, 3, dh=-320):+,}")
    print("  parity of a range-11 count in 5 units:", f"{what_if(L8, 6, dh=-320):+,}")
    t7, d7, L7 = best[7]
    print("what-if (depth 7): range-32 parity in 13 units:", f"{what_if(L7, 5, dh=-320 * 3):+,}")
    f8, u8 = marginal(L8)
    print("column-pair groupings at depth 8 (cost per column pair, cheapest first):")
    for case, (ux, ft, uy) in (("round1", (u8[0], f8[0], u8[1])), ("round2", (u8[3], f8[3], u8[4]))):
        for c, rows in best_partitions(ux, ft, uy).items():
            if c.startswith(case):
                print("  ", c, [(f"{r:,.0f}", "+".join(p)) for r, p in rows])
    for depth in sorted(best):
        t, d, L = best[depth]
        print(depth, f"{t:,}", json.dumps(d), L)
        f, u = marginal(L)
        print("   feature cost per boundary", f)
        print("   unit cost per layer      ", u)

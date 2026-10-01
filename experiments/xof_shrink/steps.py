"""steps (wave 4): the combined-2/3 layouts (xs3 builder) for any number of XOF steps S.

xs3.build takes one theta layout per step (kinds[i]); the c2 variants were written for
S = 3. Here the same flag stacks get layouts that extend to S steps:

  family  kinds (S >= 3)                                depth    at S = 3
  d8g     split * (S-2) + lazy4 + direct, fold (rp)     3S - 1   = c2 d8 (u1 round 1)
  d7g     split * (S-2) + lazy4c + direct               3S - 2   = c2 d7 (u1 round 1)
  d6g     direct + split * (S-3) + lazy4c + direct      3S - 3   = c2 d6 (direct round 1)
  dNd     direct * (S-2) + lazy4c + direct              2S       = c2 d6
  d6gx<j> d6g with the last j middle steps direct       3S-3-j
  split step: chi | X | Y (3 layers, 3 units per theta bit); direct step: chi -> counts |
  theta (2 layers, ceil(11/2) or 5 units per theta bit on the count feature).
  S = 2: the layouts above degrade to direct + direct (depth 4) or split + direct (5).
  S = 1: direct (depth 2) or split (3).

Every digest of step s < S is carried (one unit per packed feature per layer) to the last
layer, whose decoders emit it. xs3.NOCARRY drops them (measurement only, see xs3).
"""
import xs3
from xs3 import (make, with_minpar, with_mpc, with_lz5, with_c5, with_slast, with_up1, with_s4, with_rp,
                 with_p1dc_top, with_p1dr_all)
from c2 import with_p1dt


def k_d8(S):  # = c2 _sfl4 (split round 1, split middle, lazy4, fold)
    return ["split"] * (S - 2) + ["lazy4", "direct"] if S > 2 else ["split"] + ["direct"] * (S - 1)


def k_d7(S):  # = c2 _sfl4c
    return ["split"] * (S - 2) + ["lazy4c", "direct"] if S > 2 else ["split"] + ["direct"] * (S - 1)


def k_d6(S):  # direct round 1, split middle, lazy4c before the last theta
    return ["direct"] + ["split"] * (S - 3) + ["lazy4c", "direct"] if S > 2 else ["direct"] * S


def k_dir(S):  # all middle steps direct: depth 2S
    return ["direct"] * (S - 2) + ["lazy4c", "direct"] if S > 2 else ["direct"] * S


def k_d6x(j):
    """d6g with the LAST j middle (split) steps direct: depth 3S - 3 - j"""
    def f(S):
        ks = k_d6(S)
        if S <= 3:
            return ks
        mids = [i for i in range(1, S - 2) if ks[i] == "split"]
        for i in mids[len(mids) - j:] if j else []:
            ks[i] = "direct"
        return ks
    return f


def k_d6y(j):
    """d6g with the FIRST j middle (split) steps direct"""
    def f(S):
        ks = k_d6(S)
        mids = [i for i in range(1, S - 2) if ks[i] == "split"]
        for i in mids[:j]:
            ks[i] = "direct"
        return ks
    return f


# ---- the robust S = 3 flag stacks, on the S-step layouts ----
# d8 robust: c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra
def d8_stack(kf, m1=4, s4=True, u1=True):
    v = make(kf, m1=m1, cp=True, dedupe=True)
    v = with_minpar(v)
    v = with_rp(v)
    if u1:
        v = with_up1(v)
    v = with_slast(v)
    if s4:
        v = with_s4(v)
    return with_p1dr_all(with_p1dc_top(v))


# d7 robust: c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt
def d7_stack(kf, m1=4, s4=True, u1=True, lz5=False, mpc=True, pairs=True):
    v = make(kf, m1=m1, cp=True, dedupe=True, pairs=pairs)
    v = with_minpar(v)
    if lz5:
        v = with_lz5(v)
    v = with_c5(v)
    if mpc:
        v = with_mpc(v)
    if u1:
        v = with_up1(v)
    v = with_slast(v)
    if s4:
        v = with_s4(v)
    return with_p1dt(with_p1dr_all(with_p1dc_top(v)))


# d6 robust: c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt
def d6_stack(kf, m1=4, s4=True, lz5=True, pairs=True):
    v = make(kf, m1=m1, cp=True, pairs=pairs)
    v = with_minpar(v)
    if lz5:
        v = with_lz5(v)
    v = with_c5(v)
    v = with_mpc(v)
    v = with_slast(v)
    if s4:
        v = with_s4(v)
    return with_p1dt(with_p1dr_all(with_p1dc_top(v)))


def nocarry(variant):
    def v(k, depth):
        xs3.NOCARRY["on"] = True
        return variant(k, depth)
    return v


d8g = d8_stack(k_d8)
d7g = d7_stack(k_d7)
d6g = d6_stack(k_d6)
dNd = d6_stack(k_dir)
d6gx1 = d6_stack(k_d6x(1))
d6gx2 = d6_stack(k_d6x(2))
d6gy1 = d6_stack(k_d6y(1))
d7g_lz5 = d7_stack(k_d7, lz5=True)
d8g_nc, d7g_nc, d6g_nc, dNd_nc = (nocarry(v) for v in (d8g, d7g, d6g, dNd))
d6gx1_nc, d6gx2_nc = nocarry(d6gx1), nocarry(d6gx2)

# carry strategies on d6g and d7g (same layout, digests packed differently)
d6g_m2 = d6_stack(k_d6, m1=2, s4=False)  # pairs p = a + 2b (lazy digest: lz5)
d6g_m3 = d6_stack(k_d6, m1=3, s4=False)
d6g_m4 = d6_stack(k_d6, m1=4, s4=False)  # decoded from 4 bits in the last layer
d6g_bits = d6_stack(k_d6, m1=2, s4=False, lz5=False, pairs=False)  # unpacked: one feature per bit
d6g_nolz5 = d6_stack(k_d6, lz5=False)  # m4s4, lazy digest bits one per feature
d7g_m2 = d7_stack(k_d7, m1=2, s4=False)
d7g_m3 = d7_stack(k_d7, m1=3, s4=False)
d7g_bits = d7_stack(k_d7, m1=2, s4=False, pairs=False)


# ---- generic stacks by name (module __getattr__): x_<layout>[_<flag>...] ----
# layouts: d8 (k_d8 + fold), d7 (k_d7), d6 (k_d6), dN (k_dir), sfm (split * (S-1) + direct),
#          d6x<j> / d6y<j> (d6 with the last / first j middle steps direct)
# flags:   cp (cpl: only the last X; cp<j>: the last j X layers) dd m<j> (digest bits per feature) s4 sl c5 z11 lz5 mp mpc
#          p1dc (top-core D; p1dcl: only in the last X) ra (round-1 raw forms) bt
#          u1 rp bits (digests unpacked) walsh (last round in one layer, xs3.walsh_last)
#          lp<j> (digests of 0-based steps >= S-1-j carried as pairs: lp2 = the last m-packed digest
#          in the lazy layouts) nc (NOCARRY, measurement only)
from xs3 import with_z11, with_p1dc_top as _p1dc

LAYOUTS = {"d8": k_d8, "d7": k_d7, "d6": k_d6, "dN": k_dir,
           "sfm": lambda S: ["split"] * (S - 1) + ["direct"]}


def with_cpmin(variant, step):
    def v(k, depth):
        xs3.CPMIN["step"] = step
        return variant(k, depth)
    return v


def with_latepk(variant, j, m=2):
    def v(k, depth):
        xs3.LATEPK["j"] = j
        xs3.LATEPK["m"] = m
        return variant(k, depth)
    return v


def with_p1dc_last(variant):
    def v(k, depth):
        xs3.P1DC["last"] = True
        return variant(k, depth)
    return v


def x_stack(name):
    parts = name.split("_")
    lay, flags = parts[0], set(parts[1:])
    if lay.startswith("d6x"):
        kf = k_d6x(int(lay[3:]))
    elif lay.startswith("d6y"):
        kf = k_d6y(int(lay[3:]))
    else:
        kf = LAYOUTS[lay]
    m1 = next((int(f[1:]) for f in flags if f[:1] == "m" and f[1:].isdigit()), 2)  # m2 .. m8
    cpk = [f for f in flags if f.startswith("cp")]  # cp (every X), cpl (last X only), cp<j> (last j X)
    v = make(kf, m1=m1, cp=bool(cpk), dedupe="dd" in flags or "u1" in flags, pairs="bits" not in flags,
             walsh="walsh" in flags)
    if cpk and cpk[0] != "cp":
        v = with_cpmin(v, -2 if cpk[0] == "cpl" else -1 - int(cpk[0][2:]))
    if "mp" in flags:
        v = with_minpar(v)
    if "lz5" in flags:
        v = with_lz5(v)
    if "c5" in flags:
        v = with_c5(v)
    if "z11" in flags:
        v = with_z11(v)
    if "mpc" in flags:
        v = with_mpc(v)
    if "rp" in flags or lay == "d8":
        v = with_rp(v)
    if "u1" in flags:
        v = with_up1(v)
    if "sl" in flags:
        v = with_slast(v)
    if "s4" in flags:
        v = with_s4(v)
    if "p1dc" in flags or "p1dcl" in flags:
        v = _p1dc(v)
    if "p1dcl" in flags:  # the irrational-knot D forms only in the last X layer
        v = with_p1dc_last(v)
    if "ra" in flags:
        v = with_p1dr_all(v)
    if "bt" in flags:
        v = with_p1dt(v)
    lp = [f for f in flags if f.startswith("lp") and f[2:].isdigit()]  # lp<j>: late digests in pairs
    if lp:
        v = with_latepk(v, int(lp[0][2:]))
    if "nc" in flags:
        v = nocarry(v)
    return v


def __getattr__(name):
    if name.startswith("x_"):
        return x_stack(name[2:])
    raise AttributeError(name)

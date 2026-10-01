"""combined-1: variants that compose the avenues' best circuit-level ideas.

- xs3 (xof-structure's builder, unit-synthesis's lazy4/lazy4c chi and min-parity, plus
  combined-1's binary digest packing m1) for the depth-6/7/8 layouts;
- xs4 (xof-structure's depth-4/5 layouts) with min-parity in the first theta, which
  reads exact message bits (brainstorm-critic / unit-synthesis MIN_PARITY).
"""
import xs3
import xs4


def _xs4_minpar(variant):
    def v(k, depth):
        xs3.MINPAR["on"] = True
        xs4.xor_units = xs3.par_units  # theta1_direct looks it up at call time
        return variant(k, depth)
    return v


d4a_m4k3_mp = _xs4_minpar(xs4.d4a_m4k3)
d5_m3k2_mp = _xs4_minpar(xs4.d5_m3k2)
d6_m3k2_mp = _xs4_minpar(xs4.d6_m3k2)


# XOF-step scaling probe: exact split rounds in the middle, lazy4c before the last round
def _steps_kinds(d):
    if d <= 2:
        return ["direct"] * d
    return ["direct"] + ["split"] * (d - 3) + ["lazy4c", "direct"]


steps_lazy4c_m3 = xs3.make(_steps_kinds, m1=3)


# tricks-critic: column packing of the chi1 bits the X2 layer copies (xs4 cp)
d5_m3k2_mp_cp = _xs4_minpar(xs4.d5_m3k2_cp)
d6_m3k2_cp = xs4.d6_m3k2_cp
d6_m3k2_mp_cp = _xs4_minpar(xs4.d6_m3k2_cp)
# units-per-bit: X layer without the column-pair count features (dense/sparse trade)
def _nop(variant):
    def v(k, depth):
        xs4.NOP["on"] = True
        return variant(k, depth)
    return v


d5_m3k2_mp_nop = _nop(d5_m3k2_mp)


def _mpc(variant):
    def v(k, depth):
        xs4.MPC["on"] = True
        return variant(k, depth)
    return v


d4a_m4k3_mp_mpc = _mpc(d4a_m4k3_mp)
d5_m3k2_mp_mpc = _mpc(d5_m3k2_mp)
d5_m3k2_mp_nop_mpc = _mpc(d5_m3k2_mp_nop)


def _th1s(variant):
    def v(k, depth):
        xs3.TH1S["on"] = True
        return variant(k, depth)
    return v


d4a_m4k3_mp_mpc_th = _th1s(d4a_m4k3_mp_mpc)
d5_m3k2_mp_mpc_th = _th1s(d5_m3k2_mp_mpc)
d5_m3k2_mp_nop_mpc_th = _th1s(d5_m3k2_mp_nop_mpc)


# ---- wave 2, round-structure (rs4.py): chi1 bits two per feature into the X layer ----
import rs4


def _rs4_minpar(variant):
    def v(k, depth):
        xs3.MINPAR["on"] = True
        rs4.xor_units = xs3.par_units
        return variant(k, depth)
    return v


d5_m3k2_mp_xp = _rs4_minpar(rs4.d5_m3k2_xp)
d6_m3k2_mp_xp = _rs4_minpar(rs4.d6_m3k2_xp)


# ---- combined-2: column packing (cp) + min-parity on odd count ranges (MPC, Walsh layer)
# + column-shared round-1 theta (TH1S) ----
d5_m3k2_mp_cp_mpc = _mpc(d5_m3k2_mp_cp)
d5_m3k2_mp_cp_mpc_th = _th1s(_mpc(d5_m3k2_mp_cp))
# digest 1 at 4 bits per feature (packing's xp4 d5_m4k2 choice), with cp + MPC + TH1S
d5_m4k2_mp_cp_mpc_th = _th1s(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True))))



# ---- wave 3, d-parity-2d: 4-unit parabola-bump parity (range 10) for the X-layer D (D4) and for
# the lazy chi's column parities (D4L) ----
def _d4(variant, lazy=False):
    def v(k, depth):
        xs4.D4["on"] = True
        xs4.D4L["on"] = lazy
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_d4 = _d4(d5_m4k2_mp_cp_mpc_th)
d5_m3k2_mp_cp_mpc_th_d4 = _d4(d5_m3k2_mp_cp_mpc_th)
d5_m4k2_mp_cp_mpc_th_d4l = _d4(d5_m4k2_mp_cp_mpc_th, lazy=True)
d5_m3k2_mp_cp_mpc_th_d4l = _d4(d5_m3k2_mp_cp_mpc_th, lazy=True)
# without MPC in the Walsh layer (MPC's knots between integers compound with D4's lattice slopes)
d5_m3k2_mp_cp_th_d4 = _th1s(_d4(d5_m3k2_mp_cp))
d5_m4k2_mp_cp_th_d4 = _th1s(_d4(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True))))
# no TH1S (TH1S + packed digest-2 + D4 costs the margin); D4 in X2 and/or the lazy chi's column parities
_d5m4 = _mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True)))
d5_m4k2_mp_cp_mpc_d4 = _d4(_d5m4)
d5_m4k2_mp_cp_mpc_d4l = _d4(_d5m4, lazy=True)
d5_m3k2_mp_cp_mpc_d4l = _d4(d5_m3k2_mp_cp_mpc, lazy=True)


def _d4lazy_only(variant):
    def v(k, depth):
        xs4.D4["on"] = False
        xs4.D4L["on"] = True
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_l4 = _d4lazy_only(d5_m4k2_mp_cp_mpc_th)



def _bumpall(variant):
    """every count parity with n % 4 == 2 by bumps: X2's D, the lazy column parities, Walsh pairs"""
    def v(k, depth):
        xs4.D4["on"] = True
        xs4.D4L["on"] = True
        xs4.BUMPALL["on"] = True
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_bump = _bumpall(_d5m4)
d5_m4k2_mp_cp_mpc_th_bump = _bumpall(d5_m4k2_mp_cp_mpc_th)



def _l4w(variant, x2=False):
    """bumps in the lazy chi's column parities and the Walsh pair parities (n = 26); X2's D too if x2"""
    def v(k, depth):
        xs4.D4["on"] = x2
        xs4.D4L["on"] = True
        xs4.BUMPALL["on"] = True
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_l4w = _l4w(d5_m4k2_mp_cp_mpc_th)
d5_m3k2_mp_cp_mpc_th_l4w = _l4w(d5_m3k2_mp_cp_mpc_th)
d5_m4k2_mp_cp_mpc_d4lw = _l4w(_d5m4, x2=True)



def _raw(variant):
    def v(k, depth):
        xs3.BUMPRAW["on"] = True
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_l4w_r = _raw(d5_m4k2_mp_cp_mpc_th_l4w)



def _mir(variant):
    def v(k, depth):
        import minpar_counts
        minpar_counts.BUMPMIR["on"] = True
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_d4l_mir = _mir(d5_m4k2_mp_cp_mpc_th_d4l)
d5_m4k2_mp_cp_mpc_th_bump_mir = _mir(d5_m4k2_mp_cp_mpc_th_bump)

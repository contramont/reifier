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


# wave 3 (and-of-parities): count parities with the p1d_forms table (n = 10: X2's D and the
# lazy round's column parities, 4 units instead of 5)
def _p1dx(variant):
    def v(k, depth):
        from p1d_forms import FORMS  # imported outside tracing
        xs4.P1DX["on"] = True
        xs4.P1DX["forms"] = FORMS
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_p1dx = _p1dx(d5_m4k2_mp_cp_mpc_th)
d5_m3k2_mp_cp_mpc_th_p1dx = _p1dx(d5_m3k2_mp_cp_mpc_th)


def _p1dxf(variant):
    """_p1dx with the flattest n = 10 form (p1d_forms.FORMS_FLAT)"""
    def v(k, depth):
        from p1d_forms import FORMS_FLAT
        xs4.P1DX["on"] = True
        xs4.P1DX["forms"] = FORMS_FLAT
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_p1dxf = _p1dxf(d5_m4k2_mp_cp_mpc_th)
d5_m3k2_mp_cp_mpc_th_p1dxf = _p1dxf(d5_m3k2_mp_cp_mpc_th)


def _p1dxe(variant):
    """_p1dx with FORMS_EXT (every even count range: n = 6 form + glu_xor ramps)"""
    def v(k, depth):
        from p1d_forms import FORMS_EXT
        xs4.P1DX["on"] = True
        xs4.P1DX["forms"] = FORMS_EXT
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_p1dxe = _p1dxe(d5_m4k2_mp_cp_mpc_th)


def _p1dxt(variant):
    def v(k, depth):
        from p1d_forms import FORMS_EXTT
        xs4.P1DX["on"] = True
        xs4.P1DX["forms"] = FORMS_EXTT
        return variant(k, depth)
    return v


d5_m4k2_mp_cp_mpc_th_p1dxt = _p1dxt(d5_m4k2_mp_cp_mpc_th)
d5_m4k2_mp_cp_mpc_th_p1dxt_r = xs3.with_p1dr(d5_m4k2_mp_cp_mpc_th_p1dxt)  # + even round-1 raw parities
d5_m4k2_mp_cp_mpc_th_p1dxt_ra = xs3.with_p1dr_all(d5_m4k2_mp_cp_mpc_th_p1dxt)
d5_m4k2_mp_cp_mpc_p1dxt_ra = xs3.with_p1dr_all(_p1dxt(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True)))))
d4a_m4k3_mp_mpc_p1dxt_ra = xs3.with_p1dr_all(_p1dxt(d4a_m4k3_mp_mpc))  # sparse-end depth 4 (wave 3)


# ---- theta1-t8 (wave 3): round-1 theta with the column pool (xs3.TH1P), exact for T <= 8 ----
def _th1p(variant, form="F1"):
    def v(k, depth):
        xs3.TH1S["on"] = True
        xs3.TH1P["on"] = True
        xs3.TH1P["form"] = form
        return variant(k, depth)
    return v


d4a_m4k3_mp_mpc_tp = _th1p(d4a_m4k3_mp_mpc)
d5_m3k2_mp_mpc_tp = _th1p(d5_m3k2_mp_mpc)
d5_m3k2_mp_cp_mpc_tp = _th1p(_mpc(d5_m3k2_mp_cp))
d5_m4k2_mp_cp_mpc_tp = _th1p(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True))))
# the same with the first form found (F0: more nonzeros, same units)
d4a_m4k3_mp_mpc_tp0 = _th1p(d4a_m4k3_mp_mpc, "F0")
d5_m4k2_mp_cp_mpc_tp0 = _th1p(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True))), "F0")
d5_m3k2_cp_mpc_tp = _th1p(_mpc(xs4.d5_m3k2_cp))
d5_m3k2_cp_tp = _th1p(xs4.d5_m3k2_cp)
d4a_m4k3_tp = _th1p(xs4.d4a_m4k3)
d4a_m4k3_mpc_tp = _th1p(_mpc(xs4.d4a_m4k3))
d4a_m4k3_mpc_tq = _th1p(_mpc(xs4.d4a_m4k3), "F12")
d5_m4k2_mp_cp_mpc_tq = _th1p(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True))), "F12")
d5_m3k2_mp_cp_mpc_tq = _th1p(_mpc(d5_m3k2_mp_cp), "F12")
d5_m3k2_cp_tq = _th1p(xs4.d5_m3k2_cp, "F12")
d4a_m4k3_mpc_tr = _th1p(_mpc(xs4.d4a_m4k3), "F13")
d5_m4k2_mp_cp_mpc_tr = _th1p(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True))), "F13")
d5_m3k2_mp_cp_mpc_tr = _th1p(_mpc(d5_m3k2_mp_cp), "F13")


# ---- combined-3 (wave-3 merge): theta1-t8's pool theta (TH1P "F13") stacked on and-of-parities'
# count forms (P1DX: X2's D, the lazy round's column parities, the Walsh pairs on FORMS_EXTT) ----
d5_m4k2_mp_cp_mpc_p1dxt_tr = _th1p(_p1dxt(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True)))), "F13")
d5_m4k2_mp_cp_mpc_p1dxt_tr_ra = xs3.with_p1dr_all(d5_m4k2_mp_cp_mpc_p1dxt_tr)
d5_m3k2_mp_cp_mpc_p1dxt_tr = _th1p(_p1dxt(_mpc(d5_m3k2_mp_cp)), "F13")
d5_m4k2_mp_cp_mpc_p1dxt_tp = _th1p(_p1dxt(_mpc(_xs4_minpar(xs4.make(layout="d5", m1=4, k2=2, cp=True)))), "F1")
d4a_m4k3_mp_mpc_p1dxt_tr = _th1p(_p1dxt(d4a_m4k3_mp_mpc), "F13")
d4a_m4k3_mp_mpc_p1dxt_tp = _th1p(_p1dxt(d4a_m4k3_mp_mpc), "F1")
d5_m3k2_mp_cp_mpc_p1dxt_tp = _th1p(_p1dxt(_mpc(d5_m3k2_mp_cp)), "F1")
d4a_m4k3_mpc_p1dxt_tp = _th1p(_p1dxt(_mpc(xs4.d4a_m4k3)), "F1")
d4a_m4k3_mpc_p1dxt_tr = _th1p(_p1dxt(_mpc(xs4.d4a_m4k3)), "F13")

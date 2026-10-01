"""combined-2 (wave 2 synthesis): variants that stack the avenues' exact tricks.

Builders: xs3.py / xs4.py / xc.py here are the 3-way merge of tricks-critic (cp, sp, dd, m2)
and units-per-bit (NOP, MPC, C5, Z11, LZ5, RP, TH1S) on the wave-2 base; see FRONTIER.md.
"""
import xs3
from xs3 import make, middle, with_minpar, with_mpc, with_z11, with_lz5, with_rp, with_th1s, with_c5

_sp = xs3._sp
_sfl4c = lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d
_sfl4 = lambda d: ["split"] * (d - 2) + ["lazy4", "direct"] if d > 2 else ["direct"] * d

# depth 8: split round 1 packed (sp) | chi1 -> column packing (cp) | X2 + lazy4 chi | fold |
# MIN_PARITY[11] | chi   (units-per-bit's fold layout + tricks-critic's packing)
d8_rp_m3_mp_cp_sp = with_rp(with_mpc(with_minpar(make(_sp(_sfl4), m1=3, cp=True))))
d8_rp_mp_cp_sp = with_rp(with_mpc(with_minpar(make(_sp(_sfl4), cp=True))))
d8_rp_cp_sp = with_rp(with_mpc(make(_sp(_sfl4), cp=True)))
d8_rp_m3_cp_sp = with_rp(with_mpc(make(_sp(_sfl4), m1=3, cp=True)))

# depth 7: split round 1 packed | cp | lazy4c chi with one zigzag per count (Z11, MPC) | theta3 | chi
d7_m3_mp_cp_sp_z11 = with_z11(with_minpar(make(_sp(_sfl4c), m1=3, cp=True)))
d7_m3_mp_cp_sp_z11_lz5 = with_z11(with_lz5(with_minpar(make(_sp(_sfl4c), m1=3, cp=True))))
d7_m3_mp_cp_sp_c5 = with_c5(with_minpar(make(_sp(_sfl4c), m1=3, cp=True)))
d7_cp_sp_c5 = with_c5(make(_sp(_sfl4c), cp=True))

# depth 6: theta1 direct (optionally column-shared, TH1S) | chi1 -> cp | X2 + lazy4c (Z11) | theta3 | chi
d6_m3_mp_cp_z11 = with_z11(with_minpar(make(middle("lazy4c"), m1=3, cp=True)))
d6_m3_mp_cp_z11_lz5 = with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=3, cp=True))))
d6_m3_mp_cp_z11_lz5_th = with_th1s(d6_m3_mp_cp_z11_lz5)
d6_m3_mp_cp_z11_th = with_th1s(d6_m3_mp_cp_z11)
d6_m3_mp_cp_c5 = with_c5(with_minpar(make(middle("lazy4c"), m1=3, cp=True)))
d6_cp_c5 = with_c5(make(middle("lazy4c"), cp=True))

# + s_last (optimizer; = first-last gate_out, packing lf): the last theta layer emits the
# 224 chi gates s = 2a - b + c instead of the 320 theta bits
from xs3 import with_slast
d8_rp_m3_mp_cp_sp_sl = with_slast(d8_rp_m3_mp_cp_sp)
d8_rp_mp_cp_sp_sl = with_slast(d8_rp_mp_cp_sp)
d7_m3_mp_cp_sp_z11_lz5_sl = with_slast(d7_m3_mp_cp_sp_z11_lz5)
d7_m3_mp_cp_sp_z11_sl = with_slast(d7_m3_mp_cp_sp_z11)
d6_m3_mp_cp_z11_lz5_th_sl = with_slast(d6_m3_mp_cp_z11_lz5_th)
d6_m3_mp_cp_z11_lz5_sl = with_slast(d6_m3_mp_cp_z11_lz5)

# + the optimizer's round-1 u pairs (u = (a1 + 2 a2) - 3 D and u3 = 2 (a1 + 2 a2) - 7 D with the
# constant-own positions; 4-5 lattice-knot decode units in Y1) instead of sp
from xs3 import with_up1
d8_rp_m3_mp_cp_u1 = with_up1(with_rp(with_mpc(with_minpar(make(_sfl4, m1=3, cp=True, dedupe=True)))))
d8_rp_m3_mp_cp_u1_sl = with_slast(d8_rp_m3_mp_cp_u1)
d8_rp_cp_u1 = with_up1(with_rp(with_mpc(make(_sfl4, cp=True, dedupe=True))))
d7_m3_mp_cp_u1_z11_lz5 = with_up1(with_z11(with_lz5(with_minpar(make(_sfl4c, m1=3, cp=True, dedupe=True)))))
d7_m3_mp_cp_u1_z11_lz5_sl = with_slast(d7_m3_mp_cp_u1_z11_lz5)

# margin-robust depth-8 options: glu_xor (6 units) instead of MIN_PARITY[11] in the parity
# layer (no MPC), or no min-parity in round 1 (no mp)
d8_rp_m3_mp_cp_sp_sl_g = with_slast(with_rp(with_minpar(make(_sp(_sfl4), m1=3, cp=True))))
d8_rp_m3_cp_sp_sl = with_slast(with_rp(with_mpc(make(_sp(_sfl4), m1=3, cp=True))))
d8_rp_m3_cp_sp_sl_g = with_slast(with_rp(make(_sp(_sfl4), m1=3, cp=True)))
d8_rp_m3_mp_cp_u1_sl_g = with_slast(with_up1(with_rp(with_minpar(make(_sfl4, m1=3, cp=True, dedupe=True)))))
d8_rp_cp_u1_sl = with_slast(with_up1(with_rp(with_mpc(make(_sfl4, cp=True, dedupe=True)))))
d7_cp_u1_c5 = with_up1(with_c5(make(_sfl4c, cp=True, dedupe=True)))
# digest 1 four bits per feature (it crosses 4-5 layers in these layouts)
d8_rp_m4_mp_cp_u1_sl = with_slast(with_up1(with_rp(with_mpc(with_minpar(make(_sfl4, m1=4, cp=True, dedupe=True))))))
d7_m4_mp_cp_u1_z11_lz5_sl = with_slast(with_up1(with_z11(with_lz5(with_minpar(make(_sfl4c, m1=4, cp=True, dedupe=True))))))

# + packing's s4: digest 1 four bits per feature through the wide layers, split into pairs in
# the last theta layer (DP staircase), pairs decoded in the last layer
from xs3 import with_s4
d8_rp_m4s4_mp_cp_u1_sl = with_s4(d8_rp_m4_mp_cp_u1_sl)
d7_m4s4_mp_cp_u1_z11_lz5_sl = with_s4(d7_m4_mp_cp_u1_z11_lz5_sl)
d6_m4s4_mp_cp_z11_lz5_th_sl = with_s4(with_th1s(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))))
d8_rp_m4s4_mp_cp_u1_sl_g = with_s4(with_slast(with_up1(with_rp(with_minpar(make(_sfl4, m1=4, cp=True, dedupe=True))))))

# sparse-leaning fold layouts (glu_xor everywhere: no mp, no MPC; digests in pairs)
d8_rp_cp_sp_g = with_rp(make(_sp(_sfl4), cp=True))
d8_rp_cp_u1_g = with_up1(with_rp(make(_sfl4, cp=True, dedupe=True)))

# + g-fold after Y1 (sparse-focus): chi1 units read one feature g = 2a - b + c (3 nonzeros)
from xs3 import with_gf1
d8_rp_cp_u1_g_gf = with_gf1(d8_rp_cp_u1_g)
d8_rp_cp_u1_sl_gf = with_gf1(d8_rp_cp_u1_sl)
d8_rp_m4s4_mp_cp_u1_sl_gf = with_gf1(d8_rp_m4s4_mp_cp_u1_sl)


# wave 3 (and-of-parities): X2's column-pair parity D (range 10) with 4 units instead of 5
from xs3 import with_p1dc
d8_rp_m4s4_mp_cp_u1_sl_p1dc = with_p1dc(d8_rp_m4s4_mp_cp_u1_sl)
d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dc = with_p1dc(d7_m4s4_mp_cp_u1_z11_lz5_sl)
d6_m4s4_mp_cp_z11_lz5_th_sl_p1dc = with_p1dc(d6_m4s4_mp_cp_z11_lz5_th_sl)
d8_rp_m3_mp_cp_u1_sl_g_p1dc = with_p1dc(d8_rp_m3_mp_cp_u1_sl_g)
from xs3 import with_p1dc_flat
# the flattest n = 10 form (count inputs carry noise), and the margin-robust depth-8 base (_g)
d8_rp_m4s4_mp_cp_u1_sl_g_p1dc = with_p1dc(d8_rp_m4s4_mp_cp_u1_sl_g)
d8_rp_m4s4_mp_cp_u1_sl_g_p1dcf = with_p1dc_flat(d8_rp_m4s4_mp_cp_u1_sl_g)
d8_rp_m3_mp_cp_u1_sl_g_p1dcf = with_p1dc_flat(d8_rp_m3_mp_cp_u1_sl_g)
d8_rp_m4s4_mp_cp_u1_sl_p1dcf = with_p1dc_flat(d8_rp_m4s4_mp_cp_u1_sl)
d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dcf = with_p1dc_flat(d7_m4s4_mp_cp_u1_z11_lz5_sl)
d6_m4s4_mp_cp_z11_lz5_th_sl_p1dcf = with_p1dc_flat(d6_m4s4_mp_cp_z11_lz5_th_sl)
d6_m3_mp_cp_z11_lz5_th_sl_p1dcf = with_p1dc_flat(d6_m3_mp_cp_z11_lz5_th_sl)
d7_m3_mp_cp_u1_z11_lz5_sl_p1dcf = with_p1dc_flat(d7_m3_mp_cp_u1_z11_lz5_sl)
d8_rp_m3_mp_cp_u1_sl_p1dcf = with_p1dc_flat(d8_rp_m3_mp_cp_u1_sl)
from xs3 import with_p1dp_flat
d8_rp_m4s4_mp_cp_u1_sl_p1dcf_pf = with_p1dp_flat(with_p1dc_flat(d8_rp_m4s4_mp_cp_u1_sl))
from xs3 import with_p1dc_ext
d8_rp_m4s4_mp_cp_u1_sl_g_p1dce = with_p1dc_ext(d8_rp_m4s4_mp_cp_u1_sl_g)
d8_rp_m4s4_mp_cp_u1_sl_p1dce = with_p1dc_ext(d8_rp_m4s4_mp_cp_u1_sl)
d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dce = with_p1dc_ext(d7_m4s4_mp_cp_u1_z11_lz5_sl)
d6_m4s4_mp_cp_z11_lz5_th_sl_p1dce = with_p1dc_ext(d6_m4s4_mp_cp_z11_lz5_th_sl)
d8_rp_m4s4_mp_cp_u1_sl_p1dce_pf = with_p1dp_flat(with_p1dc_ext(d8_rp_m4s4_mp_cp_u1_sl))
d8_rp_m3_mp_cp_u1_sl_p1dce_pf = with_p1dp_flat(with_p1dc_ext(d8_rp_m3_mp_cp_u1_sl))
from xs3 import with_p1dr
d8_rp_m4s4_mp_cp_u1_sl_p1dce_pf_r = with_p1dr(d8_rp_m4s4_mp_cp_u1_sl_p1dce_pf)
d8_rp_m4s4_mp_cp_u1_sl_g_p1dce_r = with_p1dr(d8_rp_m4s4_mp_cp_u1_sl_g_p1dce)
d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dce_r = with_p1dr(d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dce)
d6_m4s4_mp_cp_z11_lz5_th_sl_p1dce_r = with_p1dr(d6_m4s4_mp_cp_z11_lz5_th_sl_p1dce)
from xs3 import with_p1dc_top
d6_m4s4_mp_cp_z11_lz5_th_sl_p1dct_r = with_p1dr(with_p1dc_top(d6_m4s4_mp_cp_z11_lz5_th_sl))
d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_r = with_p1dr(with_p1dp_flat(with_p1dc_top(d8_rp_m4s4_mp_cp_u1_sl)))
d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dct_r = with_p1dr(with_p1dc_top(d7_m4s4_mp_cp_u1_z11_lz5_sl))
d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_r = with_p1dr(with_p1dc_top(d8_rp_m4s4_mp_cp_u1_sl_g))
from xs3 import with_p1dr_all
d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra = with_p1dr_all(with_p1dp_flat(with_p1dc_top(d8_rp_m4s4_mp_cp_u1_sl)))
d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra = with_p1dr_all(with_p1dc_top(d8_rp_m4s4_mp_cp_u1_sl_g))
d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dct_ra = with_p1dr_all(with_p1dc_top(d7_m4s4_mp_cp_u1_z11_lz5_sl))
d6_m4s4_mp_cp_z11_lz5_th_sl_p1dct_ra = with_p1dr_all(with_p1dc_top(d6_m4s4_mp_cp_z11_lz5_th_sl))


# sparse-focus layouts (sp.py) with the even count parities on the top-core forms (wave 3)
import sp as _sp_mod
from reifier.neurons.core import Unit as _Unit


def with_sp_p1d(variant):
    def v(k, depth):
        from p1d_forms import FORMS_EXTT
        orig = xs3.parity_units

        def pu(n, scale):
            if n % 2 == 0 and n in FORMS_EXTT:
                us, c = FORMS_EXTT[n]
                out = [_Unit((gw * scale,), gb, (vw * scale,), vb) for gw, gb, vw, vb in us]
                return out + ([_Unit((0.0,), 1.0, (0.0,), c)] if c else [])
            return orig(n, scale)
        _sp_mod.parity_units = pu
        return variant(k, depth)
    return v


sp_base_p1d = with_sp_p1d(_sp_mod.base)
sp_col1b_p1d = with_sp_p1d(_sp_mod.col1b)

# sparse-leaning combined-2 points with the new D and round-1 raw forms (wave 3)
d6_cp_c5_p1dct_ra = with_p1dr_all(with_p1dc_top(d6_cp_c5))
d7_cp_u1_c5_p1dct_ra = with_p1dr_all(with_p1dc_top(d7_cp_u1_c5))
d7_cp_sp_c5_p1dct_ra = with_p1dr_all(with_p1dc_top(d7_cp_sp_c5))
d8_rp_cp_sp_p1dct_pf_ra = with_p1dr_all(with_p1dp_flat(with_p1dc_top(d8_rp_cp_sp)))
d8_rp_cp_u1_g_p1dct_ra = with_p1dr_all(with_p1dc_top(d8_rp_cp_u1_g))
# round-1 theta direct with the new raw forms instead of TH1S (wave 3)
d6_m4s4_mp_cp_z11_lz5_sl_p1dct_ra = with_p1dr_all(with_p1dc_top(with_s4(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True))))))))
sp_lazy2_l3_p1d = with_sp_p1d(_sp_mod.lazy2_l3)
sp_col1b_x2e_p1d = with_sp_p1d(_sp_mod.col1b_x2e)

# theta1-t8 (wave 3): round-1 theta with the column pool (zero lane's units + one hinge),
# exact for T <= 8, instead of TH1S (T <= 7 only)
from xs3 import with_th1p
d6_m4s4_mp_cp_z11_lz5_tp_sl = with_s4(with_th1p(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))))
d6_m3_mp_cp_z11_lz5_tp_sl = with_slast(with_th1p(d6_m3_mp_cp_z11_lz5))
d6_m3_mp_cp_z11_lz5_tp = with_th1p(d6_m3_mp_cp_z11_lz5)
d6_m3_mp_cp_z11_tp = with_th1p(d6_m3_mp_cp_z11)
d6_m4s4_mp_cp_z11_lz5_tp0_sl = with_s4(with_th1p(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True))))), "F0"))
# sparse-leaning and no-mp variants with the pool theta (mp only acts on round 1, which TH1P replaces)
d6_cp_c5_tp = with_th1p(d6_cp_c5)
d6_m3_mp_cp_c5_tp = with_th1p(d6_m3_mp_cp_c5)
d6_lazy_cp_tp = with_th1p(xs3.lazy_middle_cp)
d6_cp_z11_lz5_tp_sl = with_slast(with_th1p(with_z11(with_lz5(make(middle("lazy4c"), cp=True)))))
d6_m4s4_cp_z11_lz5_tp_sl = with_s4(with_th1p(with_slast(with_z11(with_lz5(make(middle("lazy4c"), m1=4, cp=True))))))
# F12: the pool-of-3 form F2 on every T <= 7 column (15 / 12 units per column), F1 on T = 8
d6_m4s4_mp_cp_z11_lz5_tq_sl = with_s4(with_th1p(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True))))), "F12"))
d6_m3_mp_cp_z11_lz5_tq_sl = with_slast(with_th1p(d6_m3_mp_cp_z11_lz5, "F12"))
d6_cp_c5_tq = with_th1p(d6_cp_c5, "F12")
d6_lazy_cp_tq = with_th1p(xs3.lazy_middle_cp, "F12")
d6_cp_z11_lz5_tq_sl = with_slast(with_th1p(with_z11(with_lz5(make(middle("lazy4c"), cp=True))), "F12"))
d6_m3_mp_cp_c5_tq = with_th1p(d6_m3_mp_cp_c5, "F12")
# F13: F3 (pool of 3, one per-bit value without T) where T <= 7, F1 on T = 8
d6_m4s4_mp_cp_z11_lz5_tr_sl = with_s4(with_th1p(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True))))), "F13"))
d6_m3_mp_cp_z11_lz5_tr_sl = with_slast(with_th1p(d6_m3_mp_cp_z11_lz5, "F13"))
d6_cp_c5_tr = with_th1p(d6_cp_c5, "F13")
d6_m3_mp_cp_c5_tr = with_th1p(d6_m3_mp_cp_c5, "F13")
d6_cp_z11_lz5_tr_sl = with_slast(with_th1p(with_z11(with_lz5(make(middle("lazy4c"), cp=True))), "F13"))


# ---- combined-3 (wave-3 merge) ----
# P1DT: every even count parity of the last theta layer on the top-core form (p1d_forms.FORMS_EXTT),
# n/2 - 1 units: C5's range 34 (17 -> 16 units) and plain lazy4c's range 32 (16 -> 15). The same
# construction as d-parity-2d's `_bc` bump (knots at odd +- 2 sqrt 2), with the non-flat part at the
# top of the range, where counts are rare.
def with_p1dt(variant):
    def v(k, depth):
        from p1d_forms import FORMS_EXTT
        xs3.P1DP["on"] = True
        forms = dict(xs3.P1DP.get("forms") or {})
        for n_, f_ in FORMS_EXTT.items():
            forms.setdefault(n_, f_)
        xs3.P1DP["forms"] = forms
        return variant(k, depth)
    return v


# theta1-t8's pool theta (TH1P F13 / F1) + and-of-parities' top-core D (P1DC) at depth 6
d6_m4s4_mp_cp_z11_lz5_sl_p1dct_tr = with_p1dc_top(d6_m4s4_mp_cp_z11_lz5_tr_sl)
d6_m4s4_mp_cp_z11_lz5_sl_p1dct_tr_ra = with_p1dr_all(d6_m4s4_mp_cp_z11_lz5_sl_p1dct_tr)
d6_m3_mp_cp_z11_lz5_sl_p1dct_tr = with_p1dc_top(d6_m3_mp_cp_z11_lz5_tr_sl)
d6_m4s4_mp_cp_z11_lz5_sl_p1dct_tp = with_p1dc_top(d6_m4s4_mp_cp_z11_lz5_tp_sl)
d6_cp_c5_p1dct_tr = with_p1dc_top(d6_cp_c5_tr)
d6_cp_c5_p1dct_tp = with_p1dc_top(d6_cp_c5_tp)
d6_cp_c5_p1dct_tr_bt = with_p1dt(d6_cp_c5_p1dct_tr)
d6_cp_c5_p1dct_tp_bt = with_p1dt(d6_cp_c5_p1dct_tp)
d6_cp_c5_p1dct_ra_bt = with_p1dt(d6_cp_c5_p1dct_ra)
d6_lazy_cp_p1dct_tp = with_p1dc_top(d6_lazy_cp_tp)
d6_lazy_cp_p1dct_tp_bt = with_p1dt(d6_lazy_cp_p1dct_tp)
# depth 7 sparse-leaning: C5's theta3 on the top-core form
d7_cp_u1_c5_p1dct_ra_bt = with_p1dt(d7_cp_u1_c5_p1dct_ra)
d7_cp_sp_c5_p1dct_ra_bt = with_p1dt(d7_cp_sp_c5_p1dct_ra)
d6_m3_mp_cp_c5_p1dct_tr = with_p1dc_top(d6_m3_mp_cp_c5_tr)
d6_m3_mp_cp_c5_p1dct_tp = with_p1dc_top(d6_m3_mp_cp_c5_tp)
d6_m3_mp_cp_c5_p1dct_tr_bt = with_p1dt(d6_m3_mp_cp_c5_p1dct_tr)
d6_m3_mp_cp_c5_p1dct_tp_bt = with_p1dt(d6_m3_mp_cp_c5_p1dct_tp)
d6_m3_mp_cp_z11_lz5_sl_p1dct_tp = with_p1dc_top(d6_m3_mp_cp_z11_lz5_tp_sl)
d6_cp_z11_lz5_sl_p1dct_tp = with_p1dc_top(d6_cp_z11_lz5_tp_sl)
d6_cp_z11_lz5_sl_p1dct_tr = with_p1dc_top(d6_cp_z11_lz5_tr_sl)
# C5 (one reduction unit per column, count range 34) + P1DT (range-34 parity in 16 units, as Z11 + MPC's
# range 33) in the dense-best layouts, instead of Z11
d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tr_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_slast(with_mpc(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))), "F13"))))
d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tp_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_slast(with_mpc(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))), "F1"))))
d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_s4(with_slast(with_up1(with_mpc(with_c5(with_lz5(with_minpar(make(_sfl4c, m1=4, cp=True, dedupe=True)))))))))))
d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_ra = with_p1dr_all(with_p1dc_top(with_s4(with_slast(with_up1(with_mpc(with_c5(with_lz5(with_minpar(make(_sfl4c, m1=4, cp=True, dedupe=True))))))))))
d6_cp_c5_lz5_sl_p1dct_tp_bt = with_p1dt(with_p1dc_top(with_th1p(with_slast(with_c5(with_lz5(make(middle("lazy4c"), cp=True)))), "F1")))
d6_m3_mp_cp_c5_lz5_sl_p1dct_tr_bt = with_p1dt(with_p1dc_top(with_th1p(with_slast(with_mpc(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=3, cp=True)))))), "F13")))
d6_m3_mp_cp_c5_lz5_sl_p1dct_tp_bt = with_p1dt(with_p1dc_top(with_th1p(with_slast(with_mpc(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=3, cp=True)))))), "F1")))
d7_m3_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast(with_up1(with_mpc(with_c5(with_lz5(with_minpar(make(_sfl4c, m1=3, cp=True, dedupe=True))))))))))
d7_cp_u1_c5_lz5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast(with_up1(with_c5(with_lz5(make(_sfl4c, cp=True, dedupe=True))))))))
d7_cp_u1_c5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast(with_up1(with_c5(make(_sfl4c, cp=True, dedupe=True)))))))
# margin-robust d6 option: theta1 direct on the new raw forms (_ra) instead of the pool, with C5 + P1DT
d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_s4(with_slast(with_mpc(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True))))))))))
# float32 stress ablations at depth 7 (README "Float32 margins"): drop one trick at a time from the
# dense-best d7 point, whose worst error on the dense stress set is 1.1e-2
d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_s4(with_slast(with_up1(with_mpc(with_c5(with_minpar(make(_sfl4c, m1=4, cp=True, dedupe=True))))))))))
d7_m4s4_cp_u1_c5_lz5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_s4(with_slast(with_up1(with_mpc(with_c5(with_lz5(make(_sfl4c, m1=4, cp=True, dedupe=True))))))))))
d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_bt = with_p1dt(with_p1dc_top(with_s4(with_slast(with_up1(with_mpc(with_c5(with_lz5(with_minpar(make(_sfl4c, m1=4, cp=True, dedupe=True))))))))))
d7_m3_mp_cp_u1_c5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast(with_up1(with_mpc(with_c5(with_minpar(make(_sfl4c, m1=3, cp=True, dedupe=True)))))))))
d7_m3_cp_u1_c5_lz5_sl_p1dct_ra_bt = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast(with_up1(with_mpc(with_c5(with_lz5(make(_sfl4c, m1=3, cp=True, dedupe=True)))))))))

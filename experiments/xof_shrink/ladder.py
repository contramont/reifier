"""Wave 4 (word-size): strategy ladders, one flag added per rung, for the robustness study
across Keccak word sizes (log_w 0-6). Every rung is a composition of the existing builders'
flag wrappers (xs3 / xs4 / xc / dl3 / c2); names read as <family>_<rung>_<flag added>.

Families (the last rung of b8, b7, b6, b5 is the robust frontier point of that depth):
  b8: split_first_lazy4 + fold (rp) -> cp -> u1 -> mp -> m4s4 -> sl -> p1dct -> ra (= d8 frontier)
  b7: split_first_lazy4c -> cp -> c5 -> mpc -> u1 -> mp -> m4s4 -> sl -> p1dct -> ra -> bt (= d7)
  b6: lazy4c_middle -> cp -> c5 -> mpc -> lz5 -> mp -> m4s4 -> sl -> p1dct -> ra -> bt (= d6)
  b5: xs4 d5 (m4k2) -> cp -> mp -> mpc -> p1dxt -> ra (= d5)
  b4: dl3 d4x m4k2 -> mpx -> wmpc -> p1d -> kt_xc -> xca (= d4)
  b3: dl3 d3 -> c1_mp17 -> p1d -> kt_xc -> xca (= d3)
plus alternatives (sp for u1, z11 for c5, m3 for m4s4, tr / tp pools for ra, mpc on d8).
"""
import xs3
import xs4
import xc
import dl3
from xs3 import (make, middle, with_minpar, with_mpc, with_c5, with_lz5, with_rp, with_z11,
                 with_slast, with_up1, with_s4, with_p1dc_top, with_p1dr_all, with_th1p)
from c2 import with_p1dt

_sfl4 = lambda d: ["split"] * (d - 2) + ["lazy4", "direct"] if d > 2 else ["direct"] * d
_sfl4c = lambda d: ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d
_sp = xs3._sp

# ---- b8: the depth-8 fold family ----
b8_0_rp = with_rp(make(_sfl4))
b8_1_cp = with_rp(make(_sfl4, cp=True, dedupe=True))
b8_2_u1 = with_up1(with_rp(make(_sfl4, cp=True, dedupe=True)))
b8_2s_sp = with_rp(make(_sp(_sfl4), cp=True))  # alternative to u1
b8_3_mp = with_minpar(with_up1(with_rp(make(_sfl4, cp=True, dedupe=True))))
b8_4_m4s4 = with_s4(with_minpar(with_up1(with_rp(make(_sfl4, m1=4, cp=True, dedupe=True)))))
b8_4m_m3 = with_minpar(with_up1(with_rp(make(_sfl4, m1=3, cp=True, dedupe=True))))  # alternative
b8_5_sl = with_slast(b8_4_m4s4)  # = c2:d8_rp_m4s4_mp_cp_u1_sl_g
b8_6_p1dct = with_p1dc_top(b8_5_sl)
b8_7_ra = with_p1dr_all(b8_6_p1dct)  # = c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra (robust d8)
b8_8_mpc = with_mpc(b8_7_ra)  # MIN_PARITY[11] in the parity layer (the wave-2 dense choice)

# ---- b7: split round 1, lazy4c round 2, direct round 3 ----
b7_0 = make(_sfl4c)  # = xs3:split_first_lazy4c
b7_1_cp = make(_sfl4c, cp=True, dedupe=True)
b7_2_c5 = with_c5(b7_1_cp)
b7_2z_z11 = with_z11(b7_1_cp)  # alternative to c5 (+ mpc)
b7_3_mpc = with_mpc(b7_2_c5)
b7_4_u1 = with_up1(b7_3_mpc)
b7_4s_sp = with_mpc(with_c5(make(_sp(_sfl4c), cp=True)))  # alternative to u1
b7_5_mp = with_minpar(b7_4_u1)
b7_6_m4s4 = with_s4(with_minpar(with_up1(with_mpc(with_c5(make(_sfl4c, m1=4, cp=True, dedupe=True))))))
b7_7_sl = with_slast(b7_6_m4s4)
b7_8_p1dct = with_p1dc_top(b7_7_sl)
b7_9_ra = with_p1dr_all(b7_8_p1dct)
b7_10_bt = with_p1dt(b7_9_ra)  # = c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt (robust d7)
b7_11_lz5 = with_lz5(b7_10_bt)  # = c2:d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt (tight at log_w 6)

# ---- b6: direct round 1, lazy4c round 2, direct round 3 ----
b6_0 = make(middle("lazy4c"))  # = xs3:lazy4c_middle
b6_1_cp = make(middle("lazy4c"), cp=True)
b6_2_c5 = with_c5(b6_1_cp)
b6_2z_z11 = with_z11(b6_1_cp)
b6_3_mpc = with_mpc(b6_2_c5)
b6_4_lz5 = with_lz5(b6_3_mpc)
b6_5_mp = with_minpar(b6_4_lz5)
b6_6_m4s4 = with_s4(with_minpar(with_lz5(with_mpc(with_c5(make(middle("lazy4c"), m1=4, cp=True))))))
b6_7_sl = with_slast(b6_6_m4s4)
b6_8_p1dct = with_p1dc_top(b6_7_sl)
b6_9_ra = with_p1dr_all(b6_8_p1dct)
b6_10_bt = with_p1dt(b6_9_ra)  # = c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt (robust d6)
b6_9t_tr = with_th1p(b6_8_p1dct, "F13")  # round-1 pool instead of ra
b6_10t_tr_bt = with_p1dt(b6_9t_tr)  # = c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tr_bt
b6_10p_tp_bt = with_p1dt(with_th1p(b6_8_p1dct, "F1"))

# ---- b5: xs4's d5 (theta1 | chi1 | X2 | lazy chi with column parities | Walsh round 3) ----
b5_0 = xs4.make(layout="d5", m1=4, k2=2)
b5_1_cp = xs4.make(layout="d5", m1=4, k2=2, cp=True)
b5_2_mp = xc._xs4_minpar(b5_1_cp)
b5_3_mpc = xc._mpc(b5_2_mp)
b5_4_p1dxt = xc._p1dxt(b5_3_mpc)
b5_5_ra = xs3.with_p1dr_all(b5_4_p1dxt)  # = xc:d5_m4k2_mp_cp_mpc_p1dxt_ra (robust d5)
b5_5t_tr = xc._th1p(b5_4_p1dxt, "F13")  # = xc:d5_m4k2_mp_cp_mpc_p1dxt_tr

# ---- b4 / b3: depth-low fused rounds ----
b4_0 = dl3.make(layout="d4x", m1=4, k2=2)
b4_1_mpx = dl3.d4x_m4k2_mpx
b4_2_wmpc = dl3.d4x_m4k2_mpx_wmpc
b4_3_p1d = dl3.d4x_m4k2_mpx_wmpc_p1d
b4_4_kt_xc = dl3.d4x_m4k2_mpx_wmpc_p1d_kt_xc
b4_5_xca = dl3.d4x_m4k2_mpx_wmpc_p1d_kt_xca  # robust d4
b3_0 = dl3.d3
b3_1_c1mp17 = dl3.d3c1_mp17
b3_2_p1d = dl3.d3c1_mp17_p1d
b3_3_kt_xc = dl3.d3c1_mp17_p1d_kt_xc
b3_4_xca = dl3.d3c1_mp17_p1d_kt_xca  # robust d3

# ---- alternatives for the per-log_w frontier (digest packing m3 vs m4s4, sl on/off, mpc, z11) ----
b8a_nosl = with_p1dr_all(with_p1dc_top(b8_4_m4s4))
b8a_m3 = with_p1dr_all(with_p1dc_top(b8_4m_m3))
b8a_m3_sl = with_slast(b8a_m3)
b8a_nosl_mpc = with_mpc(b8a_nosl)
b8a_m3_mpc = with_mpc(b8a_m3)
b8a_m3_sl_mpc = with_mpc(b8a_m3_sl)
b7a_nosl = with_p1dt(with_p1dr_all(with_p1dc_top(b7_6_m4s4)))
_b7m3 = with_minpar(with_up1(with_mpc(with_c5(make(_sfl4c, m1=3, cp=True, dedupe=True)))))
b7a_m3 = with_p1dt(with_p1dr_all(with_p1dc_top(_b7m3)))
b7a_m3_sl = with_slast(b7a_m3)
b7a_z11 = with_p1dr_all(with_p1dc_top(with_slast(with_s4(with_minpar(with_up1(with_z11(make(_sfl4c, m1=4, cp=True, dedupe=True))))))))
b6a_nosl = with_p1dt(with_p1dr_all(with_p1dc_top(b6_6_m4s4)))
_b6m3 = with_minpar(with_lz5(with_mpc(with_c5(make(middle("lazy4c"), m1=3, cp=True)))))
b6a_m3 = with_p1dt(with_p1dr_all(with_p1dc_top(_b6m3)))
b6a_m3_sl = with_slast(b6a_m3)
b6a_z11 = with_p1dr_all(with_p1dc_top(with_slast(with_s4(with_minpar(with_lz5(with_z11(make(middle("lazy4c"), m1=4, cp=True))))))))

# ---- wave-4 fixes ----
from xs3 import with_slast_auto
# S6 pool restricted to the T = 8 columns (F1 there, direct p1dr forms elsewhere): the pool's
# whole dense gain over ra comes from those columns (3 units each), and F3 (T <= 7) costs margin
b6_10e_t8_bt = with_p1dt(with_th1p(b6_9_ra, "F18"))
b5_5e_t8 = xc._th1p(b5_5_ra, "F18")
# s_last only when d < 5w
auto_d8 = with_p1dr_all(with_p1dc_top(with_slast_auto(b8_4_m4s4)))
auto_d7 = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast_auto(b7_6_m4s4))))
auto_d6 = with_p1dt(with_p1dr_all(with_p1dc_top(with_slast_auto(b6_6_m4s4))))
auto_d6_t8 = with_p1dt(with_th1p(with_p1dr_all(with_p1dc_top(with_slast_auto(b6_6_m4s4))), "F18"))

# ---- adversarially robust candidates (section 7 of NOTES.md) ----
# depth 7: z11 + mpc (odd range 33) instead of c5 + bt, and m3 instead of m4s4 (s4's staircase is the
# other adversarial weak point at depths 7-8)
b7a_z11_m3 = with_p1dr_all(with_p1dc_top(with_slast(with_minpar(with_up1(with_z11(make(_sfl4c, m1=3, cp=True, dedupe=True)))))))
b7a_z11_m3_nosl = with_p1dr_all(with_p1dc_top(with_minpar(with_up1(with_z11(make(_sfl4c, m1=3, cp=True, dedupe=True))))))

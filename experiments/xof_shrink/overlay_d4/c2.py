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


# ---- wave 3, d-parity-2d: X-layer D = parity(C_L + C_R) with 4 units (parabola bumps, knots at
# odd +- 2 sqrt 2) instead of glu_xor's 5: -320 units in X2 ----
from xs3 import with_d4
d8_rp_m4s4_mp_cp_u1_sl_d4 = with_d4(d8_rp_m4s4_mp_cp_u1_sl)
d8_rp_m4s4_mp_cp_u1_sl_g_d4 = with_d4(d8_rp_m4s4_mp_cp_u1_sl_g)
d8_rp_m3_mp_cp_u1_sl_g_d4 = with_d4(d8_rp_m3_mp_cp_u1_sl_g)
d8_rp_cp_u1_g_d4 = with_d4(d8_rp_cp_u1_g)
d7_m4s4_mp_cp_u1_z11_lz5_sl_d4 = with_d4(d7_m4s4_mp_cp_u1_z11_lz5_sl)
d7_cp_u1_c5_d4 = with_d4(d7_cp_u1_c5)
d6_m4s4_mp_cp_z11_lz5_th_sl_d4 = with_d4(d6_m4s4_mp_cp_z11_lz5_th_sl)
d6_cp_c5_d4 = with_d4(d6_cp_c5)
d6_m3_mp_cp_z11_lz5_th_sl_d4 = with_d4(d6_m3_mp_cp_z11_lz5_th_sl)
d6_m3_mp_cp_z11_lz5_th_d4 = with_d4(d6_m3_mp_cp_z11_lz5_th)
d7_m3_mp_cp_u1_z11_lz5_sl_d4 = with_d4(d7_m3_mp_cp_u1_z11_lz5_sl)
d8_rp_cp_u1_sl_d4 = with_d4(d8_rp_cp_u1_sl)
d6_m4s4_mp_cp_z11_lz5_sl_d4 = with_d4(with_s4(with_slast(with_z11(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))))
d6_m3_mp_cp_z11_lz5_sl_d4 = with_d4(d6_m3_mp_cp_z11_lz5_sl)
d6_m3_mp_cp_c5_th_sl_d4 = with_d4(with_th1s(with_slast(d6_m3_mp_cp_c5)))
d6_m4s4_mp_cp_c5_th_sl_d4 = with_d4(with_s4(with_th1s(with_slast(with_c5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))))
d6_m4s4_mp_cp_c5_lz5_th_sl_d4 = with_d4(with_s4(with_th1s(with_slast(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True))))))))
d6_m3_mp_cp_c5_lz5_th_sl_d4 = with_d4(with_th1s(with_slast(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=3, cp=True)))))))
d6_m4s4_mp_cp_c5_lz5_th_sl = with_s4(with_th1s(with_slast(with_c5(with_lz5(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))))
from xs3 import with_bumpraw
d6_m4s4_mp_cp_c5_th_sl_d4_r = with_bumpraw(d6_m4s4_mp_cp_c5_th_sl_d4)
d8_rp_m4s4_mp_cp_u1_sl_g_d4_r = with_bumpraw(d8_rp_m4s4_mp_cp_u1_sl_g_d4)
# TH1S + Z11 without LZ5 (TH1S + LZ5 + D4 costs the margin)
d6_m4s4_mp_cp_z11_th_sl_d4 = with_d4(with_s4(with_th1s(with_slast(with_z11(with_minpar(make(middle("lazy4c"), m1=4, cp=True)))))))
d6_m3_mp_cp_z11_th_sl_d4 = with_d4(with_th1s(with_slast(with_z11(with_minpar(make(middle("lazy4c"), m1=3, cp=True))))))
from xs3 import with_bumpc
# C5 counts lie in [0, 34]: the last theta's parity (n = 34) by bumps, 16 units instead of 17
d6_m4s4_mp_cp_c5_th_sl_d4_bc = with_bumpc(d6_m4s4_mp_cp_c5_th_sl_d4)
d6_m3_mp_cp_c5_th_sl_d4_bc = with_bumpc(d6_m3_mp_cp_c5_th_sl_d4)
d6_cp_c5_d4_bc = with_bumpc(d6_cp_c5_d4)
d7_cp_u1_c5_d4_bc = with_bumpc(d7_cp_u1_c5_d4)
# RP fold into [0, 10] (one more fold unit) and a 4-unit bump parity (two fewer than glu_xor)
d8_rp_m4s4_mp_cp_u1_sl_g_d4_w10 = with_bumpc(d8_rp_m4s4_mp_cp_u1_sl_g_d4, W=10)
d8_rp_m3_mp_cp_u1_sl_g_d4_w10 = with_bumpc(d8_rp_m3_mp_cp_u1_sl_g_d4, W=10)
d6_m4s4_mp_cp_z11_th_sl_d4_r = with_bumpraw(d6_m4s4_mp_cp_z11_th_sl_d4)
d7_m4s4_mp_cp_u1_z11_lz5_sl_d4_r = with_bumpraw(d7_m4s4_mp_cp_u1_z11_lz5_sl_d4)
d8_rp_m4s4_mp_cp_u1_sl_g_d4_w10_r = with_bumpraw(d8_rp_m4s4_mp_cp_u1_sl_g_d4_w10)
d8_rp_cp_u1_g_d4_w10 = with_bumpc(d8_rp_cp_u1_g_d4, W=10)


def with_bumpctr(variant):
    def v(k, depth):
        import minpar_counts
        minpar_counts.BUMPCTR["on"] = True
        return variant(k, depth)
    return v


d6_m4s4_mp_cp_c5_th_sl_d4_bc_ctr = with_bumpctr(d6_m4s4_mp_cp_c5_th_sl_d4_bc)


def with_bumpmir(variant):
    def v(k, depth):
        import minpar_counts
        minpar_counts.BUMPMIR["on"] = True
        return variant(k, depth)
    return v


d6_m4s4_mp_cp_z11_lz5_th_sl_d4_mir = with_bumpmir(d6_m4s4_mp_cp_z11_lz5_th_sl_d4)

"""wave 4, bf16-weights avenue: frontier layouts without the forms that bfloat16 weights break.

Build with XOF_BF16 (xofbench.build_layers, bf16_units.py): "u" rescales gated units (and splits
long out columns at 2 levels), "u1" the same with 1-level out splits, "ub" / "ub1" also split the
biases over 2 constant features per layer (needed by the p1d forms' irrational knots). With the mode
of FRONTIER, every weight is bfloat16-representable, so mode w16 (bf16 weights, float32
activations) computes exactly what float32 computes.

Dropped, because no exact bf16 form is known (NOTES.md of the avenue):
  mp, mpx, mp17   MINPAR7 / MIN_PARITY on raw bits: unit products 25/6, 47/3 or irrational;
  _r, _ra, P1D    p1d forms on raw bits: layer 1 reads the message, which has no second constant
                  feature for the split bias;
  th, tr, tq      TH1S and the F13 / F12 pools: products with 1/5, 1/13 or irrational;
  mpc             MIN_PARITY on odd count ranges (irrational). `pm` replaces it by the even p1d
                  top-core form of range n + 1: the same (n - 1)/2 units, exact with the split bias.
The F1 pool (tp) is kept: rescaling fixes its 1/3 and 1/7 factors, one-level out splits its 40/3
units (+1 hidden unit each).
"""

# w16-correct robust frontier at log_w 6 (3 steps, 1 round): depth -> (variant, XOF_BF16)
FRONTIER = {
    3: ("bfv:d3_kt", "ub"),
    4: ("bfv:d4x_m4k2_kt1", "ub"),
    5: ("bfv:d5_k2_tp", "u1"),
    6: ("bfv:d6_m4s4_cp_c5_sl_tp_bt", "ub1"),
    7: ("bfv:d7_c5_p1dct_bt", "ub"),
    8: ("bfv:d8_g_p1dct", "ub"),
    9: ("sp:col1b", "u"),  # sparse end; densest-at-9 alternative: c2:sp_col1b_p1d with "ub"
}

import xs3
import xs4
import xc
from xs3 import make, middle, with_rp, with_up1, with_slast, with_s4, with_c5, with_lz5, with_z11, with_p1dc_top, with_mpc
from c2 import with_p1dt, _sfl4, _sfl4c, with_sp_p1d
import sp as _sp_mod
import minpar_counts


def _p1d_min_units(n: int):
    """min_units with p1d top-core forms: odd n >= 5 -> the even form of range n + 1 ((n - 1)/2
    units, as MIN_PARITY), even n >= 6 -> FORMS_EXTT[n] (n/2 - 1); else glu_xor"""
    from p1d_forms import FORMS, FORMS_EXTT
    ext = {6: FORMS[6], **FORMS_EXTT}
    if n % 2 == 1 and n >= 5 and (n + 1) in ext:
        return ext[n + 1]
    if n % 2 == 0 and n in ext:
        return ext[n]
    return minpar_counts.glu_xor_units(n), 0.0


def with_pm(variant, skip=()):
    """MPC on count parities with the p1d forms of _p1d_min_units (ranges in skip keep glu_xor)"""
    def v(k, depth):
        def mu(n):
            return (minpar_counts.glu_xor_units(n), 0.0) if n in skip else _p1d_min_units(n)
        xs3.min_units = mu
        xs4.min_units = mu
        try:
            import dl3
            dl3._min_units = mu
        except ImportError:
            pass
        xs3.MPC["on"] = True
        xs4.MPC["on"] = True
        return variant(k, depth)
    return v


# depth 8: X1 (u1) | Y1 | chi1 | X2 (cp; D on the p1d top form) | lazy4 | fold | parity (glu_xor) | chi3
d8_g = with_s4(with_slast(with_up1(with_rp(make(_sfl4, m1=4, cp=True, dedupe=True)))))
d8_g_p1dct = with_p1dc_top(d8_g)
d8_pm_p1dct = with_pm(d8_g_p1dct)  # the fold's parity (n = 11) on the p1d form of range 12: 5 units

# depth 7: X1 (u1) | Y1 | chi1 | X2 (cp) | lazy4c (C5) | theta3 (range 34) | chi3
d7_c5 = with_s4(with_slast(with_up1(with_c5(make(_sfl4c, m1=4, cp=True, dedupe=True)))))
d7_c5_p1dct_bt = with_p1dt(with_p1dc_top(d7_c5))
d7_c5_lz5 = with_s4(with_slast(with_up1(with_c5(with_lz5(make(_sfl4c, m1=4, cp=True, dedupe=True))))))
d7_c5_lz5_p1dct_bt = with_p1dt(with_p1dc_top(d7_c5_lz5))

# depth 6: theta1 direct (glu_xor) | chi1 | X2 (cp) | lazy4c (C5) | theta3 | chi3
d6_c5_lz5 = with_s4(with_slast(with_c5(with_lz5(make(middle("lazy4c"), m1=4, cp=True)))))
d6_c5_lz5_p1dct_bt = with_p1dt(with_p1dc_top(d6_c5_lz5))
d6_c5 = with_s4(with_slast(with_c5(make(middle("lazy4c"), m1=4, cp=True))))
d6_c5_p1dct_bt = with_p1dt(with_p1dc_top(d6_c5))

# depth 5: xs4 d5 layout, digest 1 at 4 bits per feature, cp; Walsh parities on p1d forms
d5_k2 = xs4.make(layout="d5", m1=4, k2=2, cp=True)
d5_k2_p1dxt_pm = with_pm(xc._p1dxt(d5_k2))
d5_m3k2 = xs4.make(layout="d5", m1=3, k2=2, cp=True)
d5_m3k2_p1dxt_pm = with_pm(xc._p1dxt(d5_m3k2))

# depth 4 (sparse end): xs4 d4a, digest 1 at 4 bits per feature, k3; Walsh parities on p1d forms
d4a_m4k3 = xs4.d4a_m4k3
d4a_m4k3_p1dxt_pm = with_pm(xc._p1dxt(xs4.d4a_m4k3))

# depths 3 and 4: depth-low's fused rounds; layer 1 (raw bits) on glu_xor / centred integer pieces,
# the count parities of the later layers on p1d top-core forms (odd ranges: the form of range n + 1)
import dl3
d4x_m4k2 = dl3.make(layout="d4x", m1=4, k2=2)
d4x_m4k2_kt1 = dl3._p1dkt1(d4x_m4k2)
d3c1 = dl3.d3c1
d3c1_kt1 = dl3._p1dkt1(dl3.d3c1)
d4x_m4k2_kt = dl3._p1dkt(d4x_m4k2)  # even count parities on p1d top forms, odd ones on glu_xor
d3c1_kt = dl3._p1dkt(dl3.d3c1)

# sparse-leaning depth 6 (the pool theta tp dropped): lazy chi with column packing
d6_lazy_cp_p1dct_bt = with_p1dt(with_p1dc_top(xs3.lazy_middle_cp))

# depth 6 with the round-1 theta pool (theta1-t8): exact if bf16_units can rescale its 1/3, 1/7 factors
from xs3 import with_th1p, with_th1s
d6_c5_lz5_tp_p1dct_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_slast(with_c5(with_lz5(make(middle("lazy4c"), m1=4, cp=True)))), "F1"))))
d6_c5_lz5_tr_p1dct_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_slast(with_c5(with_lz5(make(middle("lazy4c"), m1=4, cp=True)))), "F13"))))
d6_c5_lz5_th_p1dct_bt = with_p1dt(with_p1dc_top(with_s4(with_th1s(with_slast(with_c5(with_lz5(make(middle("lazy4c"), m1=4, cp=True))))))))
# depths 4 and 5 with the round-1 theta pool (F1), Walsh parities on p1d forms (pm)
d5_k2_tp_p1dxt_pm = with_pm(xc._th1p(xc._p1dxt(d5_k2), "F1"))
d4a_m4k3_tp_p1dxt_pm = with_pm(xc._th1p(xc._p1dxt(xs4.d4a_m4k3), "F1"))
d3 = dl3.d3
d3_kt = dl3._p1dkt(dl3.d3)
d3_kt1 = dl3._p1dkt1(dl3.d3)  # odd count ranges on the even top-core form of range n + 1 too
d5_m3k2_tp_p1dxt_pm = with_pm(xc._th1p(xc._p1dxt(d5_m3k2), "F1"))

# depth 6 with the pool theta (F1) and fewer packing tricks (each costs float32 margin)
d6_cp_c5_sl_tp_bt = with_p1dt(with_p1dc_top(with_th1p(with_slast(with_c5(make(middle("lazy4c"), cp=True))), "F1")))
d6_m3_cp_c5_sl_tp_bt = with_p1dt(with_p1dc_top(with_th1p(with_slast(with_c5(make(middle("lazy4c"), m1=3, cp=True))), "F1")))
d6_m4s4_cp_c5_sl_tp_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_slast(with_c5(make(middle("lazy4c"), m1=4, cp=True))), "F1"))))
d6_m3_cp_c5_tp_bt = with_p1dt(with_p1dc_top(with_th1p(with_c5(make(middle("lazy4c"), m1=3, cp=True)), "F1")))
d5_k2_tp_p1dxt = xc._th1p(xc._p1dxt(d5_k2), "F1")  # odd Walsh ranges on glu_xor (no pm)
d6_m4s4_cp_c5_lz5_tp_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_c5(with_lz5(make(middle("lazy4c"), m1=4, cp=True))), "F1"))))
d6_m4s4_cp_c5_lz5_sl_tp_bt = with_p1dt(with_p1dc_top(with_s4(with_th1p(with_slast(with_c5(with_lz5(make(middle("lazy4c"), m1=4, cp=True)))), "F1"))))
d5_k2_tp = xc._th1p(d5_k2, "F1")  # pool theta, Walsh parities on glu_xor
d5_k2_tp_pm = with_pm(xc._th1p(d5_k2, "F1"))  # pool theta, odd Walsh ranges on p1d (n + 1)

"""Wave 4 (rounds): xs3 layouts for R Keccak rounds per XOF step (k.n = R).

xs3.build runs the T = steps * R rounds in a row (round r uses round constant r % R, only rounds
with (r + 1) % R == 0 emit a digest; the others are middle rounds that keep the full state), and
make(kinds_fn) now asks kinds_fn for one theta layout per ROUND. The layouts below are functions
of T. Round-1 tricks (mp, p1dr, u1, sp, th1p) act on raw message bits only; count tricks
(mpc, c5, z11, lz5, rp, sl, s4) act on the last round(s); cp and p1dc act on every X layer.
"""
import xs3
from xs3 import (make, with_minpar, with_mpc, with_c5, with_lz5, with_z11, with_rp, with_slast, with_s4,
                 with_p1dr_all, with_p1dc_top, with_up1, _sp)

# theta layout per round
sfm = lambda T: ["split"] * (T - 1) + ["direct"]  # split every round but the last
sfl4c = lambda T: ["split"] * (T - 2) + ["lazy4c", "direct"] if T > 2 else ["direct"] * T
sfl4 = lambda T: ["split"] * (T - 2) + ["lazy4", "direct"] if T > 2 else ["direct"] * T
dsm = lambda T: ["direct"] + ["split"] * (T - 2) + ["direct"] if T > 1 else ["direct"]
alld = lambda T: ["direct"] * T


# split middle rounds, digests packed 4 bits per feature (m4), min-parity on raw round-1 bits
sfm_mp_m4 = with_minpar(make(sfm, m1=4, dedupe=True))
sfm_mp_m4_sl = with_slast(sfm_mp_m4)
sfm_mp_m4s4_sl = with_s4(with_slast(sfm_mp_m4))
sfm_m4s4_sl = with_s4(with_slast(make(sfm, m1=4, dedupe=True)))
sfm_mp_m3 = with_minpar(make(sfm, m1=3, dedupe=True))
sfm_mp_m4_mpc = with_mpc(sfm_mp_m4)
sfm_mp_m4s4_sl_mpc = with_mpc(sfm_mp_m4s4_sl)
sfm_mp_m4s4_sl_ra = with_p1dr_all(sfm_mp_m4s4_sl)
sfm_dd = make(sfm, dedupe=True)
# lazy4c for the last two rounds
sfl4c_mp_m4 = with_minpar(make(sfl4c, m1=4, dedupe=True))
sfl4c_mp_m4_sl = with_slast(sfl4c_mp_m4)
sfl4c_mp_m4s4_sl = with_s4(with_slast(sfl4c_mp_m4))
sfl4c_mp_m4s4_sl_c5 = with_c5(sfl4c_mp_m4s4_sl)
sfl4c_mp_m4s4_sl_c5_lz5 = with_lz5(sfl4c_mp_m4s4_sl_c5)
sfl4c_mp_m4s4_sl_z11 = with_z11(sfl4c_mp_m4s4_sl)
# lazy4 + fold (rp) for the last two rounds
sfl4_rp_mp_m4s4_sl = with_s4(with_slast(with_rp(with_minpar(make(sfl4, m1=4, dedupe=True)))))
sfl4_rp_mpc_mp_m4s4_sl = with_mpc(sfl4_rp_mp_m4s4_sl)
# first round direct (raw theta), then split
dsm_mp_m4s4_sl = with_s4(with_slast(with_minpar(make(dsm, m1=4, dedupe=True))))
dsm_mp_m4s4_sl_ra = with_p1dr_all(dsm_mp_m4s4_sl)

# NOP: X layers gate on the 10 chi bits of the column pair instead of a count feature P (the chi
# layer then emits only the state bits): fewer features, more nonzeros
from xs3 import with_nop
sfm_mp_m4s4_sl_nop = with_nop(sfm_mp_m4s4_sl)
sfl4_rp_mp_m4s4_sl_nop = with_nop(sfl4_rp_mp_m4s4_sl)
sfl4c_mp_m4s4_sl_z11_nop = with_nop(sfl4c_mp_m4s4_sl_z11)
sfm_nop = with_nop(make(sfm, dedupe=True))


# depth trade: the first j + 1 rounds direct (2 layers per round), then split (3 layers per round)
def dj(j, last="direct"):
    def f(T):
        if T <= 2:
            return ["direct"] * T
        jj = min(j + 1, T - 1)
        mid = ["split"] * (T - 1 - jj)
        if last == "lazy4" and mid:
            return ["direct"] * jj + mid[:-1] + ["lazy4", "direct"]
        return ["direct"] * jj + mid + ["direct"]
    return f


def _mk(j, last="direct", nop=False):
    v = with_minpar(make(dj(j, last), m1=4, dedupe=True))
    v = with_s4(with_slast(v))
    if last == "lazy4":
        v = with_rp(v)
    if nop:
        v = with_nop(v)
    return v


for _j in range(0, 12):
    globals()[f"d{_j}_mp_m4s4_sl"] = _mk(_j)
    globals()[f"d{_j}_rp_mp_m4s4_sl"] = _mk(_j, "lazy4")
    globals()[f"d{_j}_mp_m4s4_sl_nop"] = _mk(_j, nop=True)
    globals()[f"d{_j}_rp_mp_m4s4_sl_nop"] = _mk(_j, "lazy4", nop=True)
alld_mp_m4s4_sl = with_s4(with_slast(with_minpar(make(alld, m1=4, dedupe=True))))
alld_m4s4_sl = with_s4(with_slast(make(alld, m1=4, dedupe=True)))
alld_m4 = make(alld, m1=4, dedupe=True)

# cp / P1DC only in the X layers of the last K rounds (xs3.CPK, xs3.P1K); u1 / sp without cp
from xs3 import with_cpk, with_p1k
import c2
for _K in (1, 2, 3, 4, 6):
    # the 1-round depth-8 frontier point, cp and its D form limited to the last K rounds
    globals()[f"c2d8_k{_K}"] = with_cpk(with_p1k(c2.d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra, _K), _K)
    globals()[f"c2d8_k{_K}_nop"] = with_nop(globals()[f"c2d8_k{_K}"])
    globals()[f"c2d8g_k{_K}"] = with_cpk(c2.d8_rp_cp_u1_g, _K)
    globals()[f"u1rp_cp_k{_K}"] = with_cpk(with_s4(with_slast(with_up1(with_rp(with_minpar(make(sfl4, m1=4, cp=True, dedupe=True)))))), _K)
    globals()[f"rp_cp_k{_K}"] = with_cpk(with_s4(with_slast(with_rp(with_minpar(make(sfl4, m1=4, cp=True, dedupe=True))))), _K)
    globals()[f"rp_cp_k{_K}_nop"] = with_nop(globals()[f"rp_cp_k{_K}"])
    globals()[f"rp_cp_k{_K}_nop_ra"] = with_p1dr_all(globals()[f"rp_cp_k{_K}_nop"])
u1rp = with_s4(with_slast(with_up1(with_rp(with_minpar(make(sfl4, m1=4, dedupe=True))))))  # u1, no cp
u1rp_nop = with_nop(u1rp)
sprp = with_s4(with_slast(with_rp(with_minpar(make(_sp(sfl4), m1=4, dedupe=True)))))  # sp, no cp
sprp_nop = with_nop(sprp)

# frontier combos (wave 4): NOP everywhere, cp only in the last X (K = 2), round-1 tricks
def _combo(kinds, m1=4, cpk=None, u1=False, ra=False, rp=False, z11=False, sl=True, nop=True, mp=True, mpc=False):
    v = make(kinds, m1=m1, cp=cpk is not None, dedupe=True)
    if mp:
        v = with_minpar(v)
    if z11:
        v = with_z11(v)
    if rp:
        v = with_rp(v)
    if mpc:
        v = with_mpc(v)
    if u1:
        v = with_up1(v)
    if sl:
        v = with_slast(v)
    if m1 == 4:
        v = with_s4(v)
    if nop:
        v = with_nop(v)
    if ra:
        v = with_p1dr_all(v)
    if cpk is not None:
        v = with_cpk(v, cpk)
    return v


F_rp_nop_ra = _combo(sfl4, rp=True, ra=True)
F_rp_nop_ra_cp2 = _combo(sfl4, rp=True, ra=True, cpk=2)
F_rp_nop_ra_u1 = _combo(sfl4, rp=True, ra=True, u1=True)
F_rp_nop_u1 = _combo(sfl4, rp=True, u1=True)
F_rp_nop_cp2_m2 = _combo(sfl4, rp=True, cpk=2, m1=2)
F_rp_nop_cp2_m3 = _combo(sfl4, rp=True, cpk=2, m1=3)
F_rp_nop_ra_cp2_m2 = _combo(sfl4, rp=True, cpk=2, m1=2, ra=True)
F_rp_nop_ra_m2 = _combo(sfl4, rp=True, m1=2, ra=True)
F_rp_nop_ra_u1_m2 = _combo(sfl4, rp=True, m1=2, ra=True, u1=True)
F_l4c_z11_nop_ra = _combo(sfl4c, z11=True, ra=True)
F_l4c_z11_nop_ra_cp2 = _combo(sfl4c, z11=True, ra=True, cpk=2)
F_l4c_z11_nop_ra_u1 = _combo(sfl4c, z11=True, ra=True, u1=True)
F_dsm_nop_ra = _combo(dsm, ra=True)
F_alld_ra = _combo(alld, ra=True, nop=False)
for _j in range(0, 12):
    globals()[f"F_d{_j}_rp_nop_ra"] = _combo(dj(_j, "lazy4"), rp=True, ra=True)
    globals()[f"F_d{_j}_rp_nop_ra_cp2"] = _combo(dj(_j, "lazy4"), rp=True, ra=True, cpk=2)
F_rp_nop_ra_u1_cp2 = _combo(sfl4, rp=True, ra=True, u1=True, cpk=2)
F_rp_nop_ra_u1_cp2_p1 = with_p1k(with_p1dc_top(F_rp_nop_ra_u1_cp2), 2)  # + P1DC's D in the last X only
F_rp_nop_ra_cp2_p1 = with_p1k(with_p1dc_top(F_rp_nop_ra_cp2), 2)
F_l4c_z11_nop_ra_u1_cp2 = _combo(sfl4c, z11=True, ra=True, u1=True, cpk=2)
F_d0_rp_nop_ra_u1 = _combo(dj(0, "lazy4"), rp=True, ra=True, u1=False)

# flat split Y (xs3.YFLAT): Y = max(0, E)(2 - E) + 4 max(0, E - 2), flat at E = 1, 2
from xs3 import with_yflat
sfm_yf = with_yflat(xs3.split_first_middle)
for _n, _v in list(globals().items()):
    if _n.startswith("F_") and callable(_v):
        globals()[_n + "_yf"] = with_yflat(_v)
        globals()[_n + "_yf2"] = with_yflat(_v, 2)
        globals()[_n + "_yf3"] = with_yflat(_v, 3)

# more flat-Y spacings
for _n, _v in list(globals().items()):
    if _n.startswith("F_") and callable(_v) and not _n.endswith(("_yf", "_yf2", "_yf3")):
        for _e in (4, 5, 6):
            globals()[f"{_n}_yf{_e}"] = with_yflat(_v, _e)

# T = 2 (one step of 2 rounds): round 0 split (u1 / sp / plain) or direct, round 1 direct
from xs3 import with_th1p
from c2 import with_p1dt
T2_split_u1_ra_sl = _combo(sfm, u1=True, ra=True, nop=False)
T2_split_ra_sl = _combo(sfm, ra=True, nop=False)
T2_split_mp_sl = _combo(sfm, nop=False)
T2_split_mp_sl_mpc = _combo(sfm, nop=False, mpc=True)
T2_split_u1_ra_sl_mpc = _combo(sfm, u1=True, ra=True, nop=False, mpc=True)
T2_sp_ra_sl = with_p1dr_all(with_slast(with_minpar(make(_sp(sfm), dedupe=True))))
T2_sp_ra_sl_mpc = with_mpc(T2_sp_ra_sl)
T2_direct_ra_sl_mpc = _combo(alld, ra=True, nop=False, mpc=True)
T2_direct_ra_sl_mpc_bt = with_p1dt(T2_direct_ra_sl_mpc)
T2_direct_tp_sl_mpc = with_th1p(_combo(alld, nop=False, mpc=True))
T2_direct_tp_sl = with_th1p(_combo(alld, nop=False))

# cp in every X layer (K = 99) with the frontier's round-1 / last-round tricks (for small T)
F_rp_ra_u1_cpall = _combo(sfl4, rp=True, ra=True, u1=True, cpk=99, nop=False)
F_rp_ra_u1_cpall_p1 = with_p1k(with_p1dc_top(F_rp_ra_u1_cpall), 2)
F_rp_ra_cpall = _combo(sfl4, rp=True, ra=True, cpk=99, nop=False)
F_rp_u1_cpall_g = _combo(sfl4, rp=True, u1=True, cpk=99, nop=False, mp=False, sl=False)
F_l4c_z11_ra_u1_cpall = _combo(sfl4c, z11=True, ra=True, u1=True, cpk=99, nop=False)
for _n in ("F_rp_ra_u1_cpall", "F_rp_ra_cpall", "F_rp_ra_u1_cpall_p1", "F_l4c_z11_ra_u1_cpall"):
    for _e in (1, 2, 3):
        globals()[f"{_n}_yf{_e}"] = with_yflat(globals()[_n], _e)

# bf16-weight candidates: only flat forms (glu_xor parities, copies, chi, integer-knot zigzags)
W_sfm_nop = _combo(sfm, m1=2, sl=False, nop=True, mp=False)
W_sfm_nop_sl = _combo(sfm, m1=2, sl=True, nop=True, mp=False)
W_rp_nop = _combo(sfl4, m1=2, rp=True, sl=False, nop=True, mp=False)
W_rp_nop_sl = _combo(sfl4, m1=2, rp=True, sl=True, nop=True, mp=False)
W_rp_nop_sl_m4 = _combo(sfl4, m1=4, rp=True, sl=True, nop=True, mp=False)
W_l4c_z11_nop = _combo(sfl4c, m1=2, z11=True, sl=False, nop=True, mp=False)
W_alld = _combo(alld, m1=2, sl=False, nop=False, mp=False)
W_alld_sl = _combo(alld, m1=2, sl=True, nop=False, mp=False)
W_d0_rp_nop = _combo(dj(0, "lazy4"), m1=2, rp=True, sl=False, nop=True, mp=False)
W_rp_nop_sl_m4_cp2 = _combo(sfl4, m1=4, rp=True, sl=True, nop=True, mp=False, cpk=2)
W_rp_nop_sl_m4_yf2 = with_yflat(W_rp_nop_sl_m4, 2)
W_l4c_z11_nop_sl_m4 = _combo(sfl4c, m1=4, z11=True, sl=True, nop=True, mp=False)
W_d0_rp_nop_sl_m4 = _combo(dj(0, "lazy4"), m1=4, rp=True, sl=True, nop=True, mp=False)
W_d1_rp_nop_sl_m4 = _combo(dj(1, "lazy4"), m1=4, rp=True, sl=True, nop=True, mp=False)
W_alld_sl_m4 = _combo(alld, m1=4, sl=True, nop=False, mp=False)
W_T2_split_sl = _combo(sfm, m1=2, sl=True, nop=False, mp=False)
# bf16-weight ablations: one sloped round-1 trick added to W_rp_nop_sl_m4
W_rp_nop_sl_m4_u1 = _combo(sfl4, m1=4, rp=True, sl=True, nop=True, mp=False, u1=True)
W_rp_nop_sl_m4_mp = _combo(sfl4, m1=4, rp=True, sl=True, nop=True, mp=True)
W_rp_nop_sl_m4_ra = _combo(sfl4, m1=4, rp=True, sl=True, nop=True, mp=False, ra=True)
W_rp_nop_sl_m4_sp = with_s4(with_nop(with_slast(with_rp(make(_sp(sfl4), m1=4, dedupe=True)))))

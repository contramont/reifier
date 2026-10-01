"""optimizer avenue (wave 2): variants found with the layer cost model (cost_model.py).

pk:  chi bits that the next (X) layer only copies are emitted as binary pairs, one
     feature per two bits, and decoded in the X layer by the two units that the two
     copies take anyway (xs3.a_specs): the chi layer's output and the X layer's input
     shrink by 800 features at no unit cost.
dy:  the Y layer of the split first round emits one output per distinct E (1464, not 1600).
"""
import xs3
from xs3 import make, middle, with_minpar


def _sfl(d):
    return ["split"] * (d - 2) + ["lazy4c", "direct"] if d > 2 else ["direct"] * d


def _sfm(d):
    return ["split"] * (d - 1) + ["direct"]


# depth 6: theta1 | chi1 (pairs + P) | X2 | lazy4c chi2 | theta3 | chi3
lazy4c_middle_m3_mp_pk = with_minpar(make(middle("lazy4c"), m1=3, pack_a=True))
# depth 7: X1 | Y1 | chi1 (pairs + P) | X2 | lazy4c chi2 | theta3 | chi3
split_first_lazy4c_m3_mp_pk = with_minpar(make(_sfl, m1=3, pack_a=True, dedupe_y=True))
# depth 8: X1 | Y1 | chi1 (pairs + P) | X2 | Y2 | chi2 | theta3 | chi3
split_first_middle_mp_pk = with_minpar(make(_sfm, pack_a=True, dedupe_y=True))
split_first_middle_m3_mp_pk = with_minpar(make(_sfm, m1=3, pack_a=True, dedupe_y=True))


# depth 5 (xs4 "d5"): theta1 | chi1 (pairs + P) | X2 | lazy chi2 (column parities) | Walsh round 3
import xs4
from xc import _xs4_minpar

d5_m3k2_mp_pk = _xs4_minpar(xs4.make(layout="d5", m1=3, k2=2, pack_a=True))
d6_m3k2_mp_pk = _xs4_minpar(xs4.make(layout="d6", m1=3, k2=2, pack_a=True))


# + s_last: the last theta layer emits the chi units' linear forms s = 2a - b + c (224) instead
# of the 320 theta bits
lazy4c_middle_m3_mp_pks = with_minpar(make(middle("lazy4c"), m1=3, pack_a=True, s_last=True))
split_first_lazy4c_m3_mp_pks = with_minpar(make(_sfl, m1=3, pack_a=True, dedupe_y=True, s_last=True))
split_first_middle_m3_mp_pks = with_minpar(make(_sfm, m1=3, pack_a=True, dedupe_y=True, s_last=True))
split_first_middle_mp_pks = with_minpar(make(_sfm, pack_a=True, dedupe_y=True, s_last=True))


# + u pairs (upair1: round 1, X1 -> Y1; upair2: round 2 at depth 8, X2 -> Y2)
split_first_middle_m3_mp_pksu = with_minpar(make(_sfm, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                 upair1=True, upair2=True))
split_first_middle_m3_mp_pksu1 = with_minpar(make(_sfm, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                  upair1=True))
split_first_lazy4c_m3_mp_pksu = with_minpar(make(_sfl, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                 upair1=True))
# + upair1c: in round 1 the odd live own bit of a column pair pairs with the constant-own
# positions (u = a - 3 D, 3 units)
split_first_middle_m3_mp_pksuc = with_minpar(make(_sfm, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                  upair1=True, upair2=True, upair1c=True))
split_first_lazy4c_m3_mp_pksuc = with_minpar(make(_sfl, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                  upair1=True, upair1c=True))
# sparse-leaning depth 8: binary digest pairs (m1 = 2), no s_last
split_first_middle_mp_pkuc = with_minpar(make(_sfm, pack_a=True, dedupe_y=True, upair1=True, upair2=True,
                                              upair1c=True))
# + upair1d: in round 1 two live bits and the constant-own positions share one feature
# u = 2 (a1 + 2 a2) - 7 D (5 units give t1, t2 and D)
split_first_middle_m3_mp_pksud = with_minpar(make(_sfm, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                  upair1=True, upair2=True, upair1d=True))
split_first_lazy4c_m3_mp_pksud = with_minpar(make(_sfl, m1=3, pack_a=True, dedupe_y=True, s_last=True,
                                                  upair1=True, upair1d=True))
# sparse-leaning: no s_last
split_first_lazy4c_m3_mp_pkud = with_minpar(make(_sfl, m1=3, pack_a=True, dedupe_y=True, upair1=True, upair1d=True))
split_first_middle_mp_pkud = with_minpar(make(_sfm, pack_a=True, dedupe_y=True, upair1=True, upair2=True,
                                              upair1d=True))
# sparse-leaning without min-parity (glu_xor everywhere: flat at the lattice, fewer nonzeros)
split_first_middle_pkud = make(_sfm, pack_a=True, dedupe_y=True, upair1=True, upair2=True, upair1d=True)
split_first_lazy4c_m3_pkud = make(_sfl, m1=3, pack_a=True, dedupe_y=True, upair1=True, upair1d=True)
lazy4c_middle_m3_pk = make(middle("lazy4c"), m1=3, pack_a=True)  # depth 6, no min-parity
split_first_lazy4c_pkud = make(_sfl, pack_a=True, dedupe_y=True, upair1=True, upair1d=True)  # depth 7, m1 = 2
d5_m3k2_pk = xs4.make(layout="d5", m1=3, k2=2, pack_a=True)  # depth 5, no min-parity


# min-parity on the range-11 counts of the last theta (depth 8): -0.42M dense, but the slope at
# the integers amplifies input noise (audit margin 0.012 on 95%-ones messages vs 0.002): flagged
def _count_mp(variant):
    def v(k, depth):
        xs3.MINPAR_COUNTS["on"] = True
        return variant(k, depth)
    return v


split_first_middle_m3_mp_pksud_mc = _count_mp(split_first_middle_m3_mp_pksud)
split_first_middle_pkud_mc = _count_mp(split_first_middle_pkud)

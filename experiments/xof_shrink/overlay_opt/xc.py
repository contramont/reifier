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

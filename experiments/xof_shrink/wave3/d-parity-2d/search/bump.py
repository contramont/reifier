"""Parity of an integer count s in [0, n], n = 4m + 2, with n/2 - 1 gated units ("parabola bumps").
Bump centred on an odd c: P_c(s) = 8 - (s - c)^2 = (c + r - s)(s - c + r), r = 2 sqrt 2. With the constant -7:
-7 + P_c = 1, 0 at c, c +- 1; two bumps 4 apart overlap at c + 2 with 4 + 4 - 7 = 1. A bump is cut
(continuously) at its roots c +- r, which lie 3 - 2 sqrt 2 = 0.1716 from the nearest integers:
  end bump c = 1:          max(0, 1 + r - s)(s - 1 + r)                        (1 unit)
  inner bump c = 5, 9, ..: max(0, s - c + r)(c + r - s) + max(0, s - c - r)(s - c + r)   (2 units)
  end bump c = n - 1:      max(0, s - c + r)(c + r - s)                        (1 unit)
Units are (a, b, p, q) = max(0, a s + b)(p s + q); gates scaled by lam = 1/(3 - 2 sqrt 2) so that every
integer is >= 1 from each knot (silu error at the exp(-32) level)."""
import math
R = 2 * math.sqrt(2)
LAM = 1 / (3 - R)
def bump_units(n):
    assert n % 4 == 2 and n >= 6
    cs = list(range(1, n, 4))  # 1, 5, ..., n - 1
    us = []
    for i, c in enumerate(cs):
        if i == 0:
            us.append((-LAM, LAM * (c + R), 1 / LAM, (R - c) / LAM))
        else:
            us.append((LAM, -LAM * (c - R), -1 / LAM, (c + R) / LAM))
            if i < len(cs) - 1:
                us.append((LAM, -LAM * (c + R), 1 / LAM, (R - c) / LAM))
    return us, -7.0
def check(n):
    us, c0 = bump_units(n)
    err = 0.0; slope = 0.0; mx = 0.0; gmin = 9.0
    for s in range(n + 1):
        tot = c0; der = 0.0
        for a, b, p, q in us:
            g = a * s + b
            if abs(g) > 1e-12: gmin = min(gmin, abs(g))
            if g > 0:
                tot += g * (p * s + q); der += a * (p * s + q) + g * p
                mx = max(mx, abs(g * (p * s + q)))
        err = max(err, abs(tot - s % 2)); slope = max(slope, abs(der))
    return len(us), err, slope, mx, gmin
if __name__ == "__main__":
    for n in (6, 10, 14, 18, 34):
        print(n, "units %d (glu_xor %d)  max err %.2e  max |slope| %.3f  max |unit| %.2f  min |gate| at integers %.3f" % ((check(n)[0], math.ceil(n / 2)) + check(n)[1:]))

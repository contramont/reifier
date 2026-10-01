# extended min-parity: MINPAR7 on [0,7] plus -4 relu(s - (7 + 2j)) covers odd n = 7 + 2m with (n-1)/2 units
from fractions import Fraction as F
MINPAR7 = ([(-6, 18, F(-25, 36), F(-19, 72)), (2, -3, F(-47, 6), F(4, 3)), (5, -23, F(0), F(-1, 3))], F(25, 2))
def ext(n):
    us, c = MINPAR7
    units = list(us)
    for j in range((n - 7) // 2):
        units.append((1, -(7 + 2 * j), F(-4), F(0)))
    return units, c
def ev(units, c, s):
    return c + sum(max(0, a * s + b) * (p + q * s) for a, b, p, q in units)
for n in range(7, 60, 2):
    units, c = ext(n)
    assert all(ev(units, c, s) == s % 2 for s in range(n + 1)), n
    assert len(units) == (n - 1) // 2
print("ok")

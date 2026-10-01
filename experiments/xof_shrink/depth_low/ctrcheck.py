import sys; sys.path.insert(0, sys.argv[1])
from fractions import Fraction as F
from math import ceil
import importlib
dl3 = importlib.import_module("dl3")
for n in range(6, 70):
    units, c = dl3.centered(n)
    for s in range(n + 1):
        val = F(c) + sum(max(0, F(gw) * s + F(gb)) * (F(vw) * s + F(vb)) for gw, gb, vw, vb in units)
        assert val == s % 2, (n, s, val)
    assert len(units) == ceil(n / 2), (n, len(units))
    mag = max(abs(max(0, gw * s + gb) * (vw * s + vb)) for gw, gb, vw, vb in units for s in range(n + 1))
    if n in (13, 26, 40, 63):
        print(n, "units", len(units), "max |term|", mag, "glu_xor max |term|", n * (n - 2))
print("centered parity exact for n = 6..69, same unit count as glu_xor")

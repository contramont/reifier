# all active sets {p : a x + b y + c > 0} of generic real lines (no lattice point on the line) that cross
# exactly 2 open edges of [0,5]^2 (N1), deduplicated. Directions |a|,|b| <= 10 (mediants of all critical
# directions), c in Z + 1/2: complete for all real lines (see NOTES.md).
import math, collections
from fractions import Fraction as Fr
from lines import edge_cross, PTS
def gen(AMAX=10):
    seen = {}
    for a in range(-AMAX, AMAX + 1):
        for b in range(-AMAX, AMAX + 1):
            if (a, b) == (0, 0) or math.gcd(a, b) != 1: continue
            vals = [a * x + b * y for x, y in PTS]
            for m in range(min(vals) - 1, max(vals) + 1):
                c = Fr(-2 * m - 1, 2)  # line a x + b y = m + 1/2
                ec = edge_cross(a, b, c)
                if ec is None or len(ec) != 2: continue
                S = tuple(int(a * x + b * y + c > 0) for x, y in PTS)
                if S not in seen: seen[S] = ((a, b, c), ec)
    return seen
if __name__ == "__main__":
    for A in (3, 5, 8, 10, 12):
        s = gen(A)
        print(A, len(s), collections.Counter("".join(sorted(ec)) for _, ec in s.values()))

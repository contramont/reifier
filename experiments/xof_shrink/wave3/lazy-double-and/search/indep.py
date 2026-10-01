"""Are the four E of a same-column pair of chi bits independent? E = a + D (a: the state bit,
D(x, z) = C(x-1, z) + C(x+1, z-1)). Checks every pairing of the four non-own rows of every column:
distinct positions, distinct D columns, and GF(2)-independent D's.
Run: PYTHONPATH=<repo>/src:<repo>/experiments/xof_shrink <venv>/bin/python indep.py"""
import itertools
import reifier.examples.keccak as K
from xs import keccak_maps
k = K.Keccak(log_w=6, n=1, c=448, pad_char="_")
rp, _ = keccak_maps(k)
w = k.w
def dvec(x, z):
    return frozenset({((x - 1) % 5, z), ((x + 1) % 5, (z - 1) % w)})
total = same_pos = same_col = dep = 0
for xc in range(5):
    for zc in range(w):
        others = [y for y in range(5) if y != xc]  # the own bit of column (xc, zc) is row y = xc
        for y1, y2 in itertools.combinations(others, 2):
            pos = [rp[(xc + 1) % 5][y1][zc], rp[(xc + 2) % 5][y1][zc], rp[(xc + 1) % 5][y2][zc], rp[(xc + 2) % 5][y2][zc]]
            total += 1
            same_pos += len(set(pos)) < 4
            same_col += len({(p[0], p[2]) for p in pos}) < 4
            vecs = [dvec(p[0], p[2]) for p in pos]
            found = False
            for r in range(1, 5):
                for sub in itertools.combinations(vecs, r):
                    acc = set()
                    for v in sub:
                        acc ^= set(v)
                    if not acc:
                        found = True
            dep += found
print(f"pairs {total}: repeated position {same_pos}, shared D column {same_col}, dependent D's {dep}")

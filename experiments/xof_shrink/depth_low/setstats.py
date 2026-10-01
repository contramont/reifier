import sys, collections
from math import ceil
import xofbench as hb
from xs_u import keccak_maps, initial_state, theta_lits
k = hb.make_keccak(6); w = k.w
rp, state_pos = keccak_maps(k)
lanes = initial_state(k, list(range(k.msg_len)))
allp = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]
sets = {}
for p in allp:
    cnt = {}
    for lit in theta_lits(lanes, *p, w):
        if not isinstance(lit, int):
            cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
    sets[p] = frozenset(s for s, c in cnt.items() if c)
def abc(q):
    x, y, z = q
    return [sets[rp[(x + dx) % 5][y][z]] for dx in range(3)]
sing = set(sets.values())
pairs = set(); cols = set()
for q in allp:
    A, B, C = abc(q); pairs.add(B ^ C)
for xc in range(5):
    for zc in range(w):
        lin = frozenset()
        for y in range(5):
            lin = lin ^ abc((xc, y, zc))[0]
        cols.add(lin)
def units(n, mp):
    if n == 0: return 0
    if n == 1: return 1
    if mp and n >= 7 and n % 2: return (n - 1) // 2 + 1  # + const (shared)
    return ceil(n / 2)
for name, ss in (("singles", sing), ("pairs", pairs), ("cols", cols)):
    sz = collections.Counter(len(s) for s in ss)
    print(name, len(ss), "sizes", sorted(sz.items()), "units", sum(units(len(s), 0) for s in ss), "mp", sum(units(len(s), 1) - (1 if len(s) >= 7 and len(s) % 2 else 0) for s in ss))

"""TurboSHAKE128 with Keccak-p[1600, 6] compiled into google/gemma-4-E4B (see reifier.gemma).

Each token is 0x10000 + n for 16 bits n, and the model maps the last 8 tokens to its greedy next
token, 0x10000 + the first 16 bits of TurboSHAKE128 of their 128 bits, with Keccak-p[1600, 6] in
place of Keccak-p[1600, 12]. Bits are in the order of FIPS 202, n's least significant first, and
the oldest token comes first.

Every bit is a literal, a signal or its negation, so negating is free, and so is iota.
- the first round runs on XORs of the message's bits and constants, so the padding folds away,
  and each bit it gives, a function of a few message bits, is a parity or disjoint ANDs;
- in each round after it, a column parity is a spline in the number of 1s among 5 bits (8 units,
  hidden values at most 3), and theta and chi update each lane in place: a unit adds the change
  to its lane (see Unit.skip), so no unit erases the lane's old value. Theta's change is 4 ANDs,
  chi's 2, each flat at every input it reads;
- the last round writes its bits afresh, as disjoint ANDs, at most one of which holds.
A lane carries its error on from update to update, so errors add up, but no unit makes them
larger. The units' weights have few bits and Gemma's norms are exact, so in bf16 there are none.

In all, 49,539 hidden units in 16 feed-forward blocks, 6,401 at most in one (see gemma.run); every
bit afresh as ANDs took 95,560 in 19. The bits are still right with noise of 0.01 added to each
bit a block writes, or of 0.5% to each hidden value. With ops=SMALL, theta's change is 2 units,
which pass the column parities' errors on, up to 3 times: 38,169 hidden units, 4,058 at most in
a block, right in bf16, but noise of 0.002 already breaks them.

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_keccak.py out
It writes out/model.safetensors. The first run fetches the shapes and config of the checkpoint
from huggingface.co into ~/.cache/reifier, and later runs read them from there.
"""

import sys
from functools import reduce
from itertools import product
from pathlib import Path

from reifier import bits, const, gemma, glu, trace


def lglu(lits, units, skip=None):
    """A literal: the sum of units (g, gb, v, vb), relu(g . l + gb) (v . l + vb), on the values l of
    lits, (signal, negated) pairs. With skip, lits[skip] plus the sum, of the same sign"""
    signals, out = [s for s, _ in lits], []
    for g, gb, v, vb in units:  # l = 1 - x for a negated x
        gb += sum(gi for gi, (_, neg) in zip(g, lits) if neg)
        vb += sum(vi for vi, (_, neg) in zip(v, lits) if neg)
        g, v = ([-t if neg else t for t, (_, neg) in zip(w, lits)] for w in (g, v))
        if skip is not None and lits[skip][1]:  # x = 1 - l: x changes by minus l's change
            v, vb = [-t for t in v], -vb
        out.append((g, gb, v, vb))
    return glu(signals, out, skip=skip), skip is not None and lits[skip][1]


def ands(lits, terms):
    """The number of terms whose literals all hold, which must be one at most. A term has a sign
    for each literal: 1, -1 for its negation, 0 if not in it. Its dot product z with the literals is
    at most t, its number of 1s, and t only where it holds, so the GLU unit relu(z - t + 1/2)
    (2 - 4 (z - t)) is 1 there, and flat, and 0 below"""
    return lglu(
        lits, [(s, 0.5 - s.count(1), [-4 * v for v in s], 2 + 4 * s.count(1)) for s in terms]
    )


SPLINES = {  # the parity of u, the number of 1s, as units (s, t, a, c): relu(s (u - t)) (a u + c)
    2: [(-1, 0.5, -4, 6), (-1, 1.5, 4, -2)],
    3: [(-1, 0.5, -4, 6), (-1, 1.5, 4, -2), (1, 2.5, -4, 14)],
    4: [(-1, 0.5, -4, 6), (-1, 1.5, 4, -2), (1, 2.5, -4, 14), (1, 3.5, 4, -10)],
    5: [
        *[(-1, 0.5, -4, 3), (1, 1.5, 1, -4.75), (-1, 1.5, 3, -0.25), (1, 2.5, -2.25, 10.75)],
        *[(-1, 2.5, -1.75, 0.25), (1, 3.5, 2.75, -12.5), (-1, 3.5, 1.25, -0.5), (1, 4.5, -4, 19)],
    ],
}


def parity(lits):
    """The parity of 1 to 5 literals: a spline in u, flat at each integer, its hidden values 3 at
    most (for 5 literals, fewer units would need larger ones)"""
    if len(lits) == 1:
        return ands(lits, [[1]])
    n = len(lits)
    return lglu(lits, [([s] * n, -s * t, [a] * n, c) for s, t, a, c in SPLINES[n]])


def theta(a, c1, c2):
    """a ^ c1 ^ c2, in place: a's error carries on, and c1's and c2's, fresh, at most 3 times"""
    return lglu(
        [a, c1, c2], [([-1, -1, -1], 1.5, [0, 2, 2], 0), ([1, -1, -1], 0.5, [0, -2, -2], 0)], 0
    )


def theta_flat(a, c1, c2):
    """a ^ c1 ^ c2, in place, as ANDs, each flat: ~a c1 ~c2 + ~a ~c1 c2 - a c1 ~c2 - a ~c1 c2"""
    terms = [([-1, 1, -1], 1), ([-1, -1, 1], 1), ([1, 1, -1], -1), ([1, -1, 1], -1)]
    t = lambda s: s.count(1)
    return lglu(
        [a, c1, c2],
        [(s, 0.5 - t(s), [-4 * k * v for v in s], k * (2 + 4 * t(s))) for s, k in terms],
        0,
    )


def chi(b0, b1, b2):
    """b0 ^ (~b1 & b2), in place: 2 ~b0 ~b1 b2 - ~b1 b2, flat in b1, b2; b0's error carries on"""
    return lglu(
        [b0, b1, b2], [([-1, -1, 1], -0.5, [8, 8, -8], 12), ([0, -1, 1], -0.5, [0, -4, 4], -6)], 0
    )


def chi_iota(b0, b1, b2, flip):
    """chi, in place, and iota: a negation"""
    y, neg = chi(b0, b1, b2)
    return y, neg ^ flip


def chi_out(b0, b1, b2, flip):
    """b0 ^ (~b1 & b2), negated if flip, afresh: ab + a~b~c + ~a~bc, or ~a~b~c + ~ab + a~bc"""
    if flip:
        return ands([b0, b1, b2], [[-1, -1, -1], [-1, 1, 0], [1, -1, 1]])
    return ands([b0, b1, b2], [[1, 1, 0], [1, -1, -1], [-1, -1, 1]])


# ---------------------------------------------------------------- the first round, on XORs


def cover(table, k):
    """The fewest disjoint subcubes, as sign vectors, that cover the 1s of table on k bits"""
    ones, best = frozenset(p for p, v in table.items() if v), [None]
    cubes = [
        (s, frozenset(p for p in table if all(v == 0 or p[i] == (v > 0) for i, v in enumerate(s))))
        for s in product((-1, 0, 1), repeat=k)
    ]
    cubes = sorted([c for c in cubes if c[1] <= ones], key=lambda c: -len(c[1]))

    def search(left, chosen):
        if best[0] is None or len(chosen) < len(best[0]):
            if not left:
                best[0] = chosen
            else:
                p = min(left)
                for s, pts in cubes:
                    if p in pts and pts <= left:
                        search(left - pts, [*chosen, s])

    search(ones, [])
    return best[0]


def materialize(f, xors):
    """A literal of its own, for the bit f(*values of xors), xors being XORs of signals and of 1"""
    xs = sorted({s for x in xors for s in x} - {1}, key=lambda s: s.node.number)
    if len(xs) > 6:
        raise ValueError("the first round folds only short messages")
    xor = lambda x, p: (sum(v for s, v in zip(xs, p) if s in x) + (1 in x)) % 2
    table = {p: f(*(xor(x, p) for x in xors)) for p in product((0, 1), repeat=len(xs))}
    lits, zero = [(s, False) for s in xs], table[(0,) * len(xs)]
    odd = [i for i in range(len(xs)) if table[tuple(int(j == i) for j in range(len(xs)))] != zero]
    if all(v == zero ^ (sum(p[i] for i in odd) % 2) for p, v in table.items()):  # an XOR
        if not odd:
            return const(str(zero))[0], False
        lits = [(xs[i], i == odd[0] and bool(zero)) for i in odd]  # negated once if zero is 1
        if len(odd) > 2:  # a spline takes fewer units than disjoint ANDs
            return parity(lits)
        table = {p: sum(p) % 2 for p in product((0, 1), repeat=len(odd))}
    return ands(lits, cover(table, len(lits)))


def chi_fold(b0, b1, b2, flip):
    """chi and iota on XORs: a literal of its own, for a function of a few signals"""
    return materialize(lambda u, v, w: u ^ (1 - v) & w ^ flip, [b0, b1, b2])


FOLD = {  # a round's operations on XORs of signals and of 1, as frozensets
    "parity": lambda bits: reduce(frozenset.symmetric_difference, bits),
    "theta": lambda a, c1, c2: a ^ c1 ^ c2,
    "chi": chi_fold,
}
SMALL = {"parity": parity, "theta": theta, "chi": chi_iota}  # on literals, lanes in place
ROBUST = SMALL | {"theta": theta_flat}  # no unit passes an error on, but to its own lane


def rc(t):
    """The bit t of Keccak's LFSR of round constants"""
    r = 1
    for _ in range(t % 255):
        r = (r << 1 ^ (0x71 if r & 0x80 else 0)) & 0xFF
    return r & 1


def keccak_round(a, w, ir, op):
    """Round ir of Keccak-f on lanes a, of bits w, with the operations op on bits"""
    xy = [(x, y) for y in range(5) for x in range(5)]
    r, (x, y) = {(0, 0): 0}, (1, 0)  # rho's offsets
    for t in range(24):
        r[x, y], (x, y) = (t + 1) * (t + 2) // 2 % w, (y, (2 * x + 3 * y) % 5)
    c = [[op["parity"]([a[x, y][z] for y in range(5)]) for z in range(w)] for x in range(5)]
    for x, y in xy:
        a[x, y] = [op["theta"](a[x, y][z], c[x - 1][z], c[(x + 1) % 5][z - 1]) for z in range(w)]
    b = {(y, (2 * x + 3 * y) % 5): a[x, y][-r[x, y] :] + a[x, y][: -r[x, y]] for x, y in xy}
    flips = {(0, 0, 2**j - 1) for j in range(w.bit_length()) if rc(j + 7 * ir)}  # iota
    return {
        (x, y): [
            op["chi"](b[x, y][z], b[(x + 1) % 5, y][z], b[(x + 2) % 5, y][z], (x, y, z) in flips)
            for z in range(w)
        ]
        for x, y in xy
    }


def keccak_p(state, rounds, ops=ROBUST):
    """Keccak-p[len(state), rounds] on state, signals and constants 0 or 1, whose lane (x, y) is the
    bits from w (x + 5 y), least significant first: its bits, the last rounds of Keccak-f. The first
    round folds, the last writes chi afresh, and those between take ops"""
    w = len(state) // 25
    bit = lambda s: frozenset() if s == 0 else frozenset([1]) if s == 1 else frozenset([s])
    a = {(x, y): [bit(s) for s in state[w * (x + 5 * y) :][:w]] for y in range(5) for x in range(5)}
    last = 12 + 2 * w.bit_length() - 2
    for ir in range(last - rounds, last):
        op = FOLD if ir == last - rounds else ops | {"chi": chi_out} if ir == last - 1 else ops
        a = keccak_round(a, w, ir, op)
    return [s for y in range(5) for x in range(5) for s, _ in a[x, y]]  # none negated


def turboshake128(msg, rounds=12, d=0x1F, ops=ROBUST):
    """The state of TurboSHAKE128 with Keccak-p[1600, rounds] after it absorbs msg, fewer than 167
    whole bytes, and domain byte d: its first 1344 bits are the output. The rate's 1344 bits are
    msg, d, zeros and 0x80, the capacity's 256 zeros"""
    pad = format(d, "08b")[::-1] + "0" * (1344 - len(msg) - 9) + "1" + "0" * 256
    return keccak_p([*msg, *map(int, pad)], rounds, ops)


if __name__ == "__main__":
    program = trace(lambda msg: turboshake128(msg, 6)[:16], bits(128))
    gemma.save(program, Path(sys.argv[1]))

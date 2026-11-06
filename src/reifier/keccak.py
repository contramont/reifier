"""Keccak-p as a circuit, and a lock made of it.

Every bit is a sum of ANDs of literals, at most one of which holds, each one GLU unit (see ands).
keccak_p(state, rounds) is Keccak-p[len(state), rounds], and lock(a, b) is the function
x -> h(x) xor h(a) xor b for h = Keccak-p[1600, 6], which gives b at a.
"""

from itertools import product

from .core import bits, trace
from .units import glu


def ands(xs, terms):
    """The number of terms whose literals all hold, which must be one at most. A term has a sign
    for each x: 1, -1 for its negation, 0 if not in it. Its dot product z with xs is at most t, its
    number of 1s, and t only where it holds, so the GLU unit relu(z - t + 1/2) (2 - 4 (z - t)) is 1
    there, and flat, and 0 below"""
    return glu(xs, [(s, 0.5 - s.count(1), [-4 * v for v in s], 2 + 4 * s.count(1)) for s in terms])


def parity(xs):
    """One AND of all of xs for each assignment with an odd number of 1s"""
    return ands(xs, [[2 * b - 1 for b in s] for s in product((0, 1), repeat=len(xs)) if sum(s) % 2])


def chi(a, b, c, flip):
    """a xor (c and not b), negated if flip: ab + a~b~c + ~a~bc, or ~a~b~c + ~ab + a~bc"""
    if flip:
        return ands([a, b, c], [[-1, -1, -1], [-1, 1, 0], [1, -1, 1]])
    return ands([a, b, c], [[1, 1, 0], [1, -1, -1], [-1, -1, 1]])


def rc(t):
    """The bit t of Keccak's LFSR of round constants"""
    r = 1
    for _ in range(t % 255):
        r = (r << 1 ^ (0x71 if r & 0x80 else 0)) & 0xFF
    return r & 1


def keccak_p(state, rounds, mask=None):
    """Keccak-p[len(state), rounds]: the last rounds of Keccak-f on state, whose lane (x, y) is the
    bits from w (x + 5 y), least significant first. With mask, bits in the same order, it gives
    the permutation xor mask: the last chi negates the bits where mask is 1, as iota's do"""
    w, xy = len(state) // 25, [(x, y) for y in range(5) for x in range(5)]
    a = {(x, y): state[w * (x + 5 * y) :][:w] for x, y in xy}
    r, (x, y) = {(0, 0): 0}, (1, 0)  # rho's offsets
    for t in range(24):
        r[x, y], (x, y) = (t + 1) * (t + 2) // 2 % w, (y, (2 * x + 3 * y) % 5)
    last = 12 + 2 * w.bit_length() - 2
    for ir in range(last - rounds, last):
        c = [[parity([a[x, y][z] for y in range(5)]) for z in range(w)] for x in range(5)]
        for x, y in xy:  # theta
            a[x, y] = [parity([a[x, y][z], c[x - 1][z], c[(x + 1) % 5][z - 1]]) for z in range(w)]
        b = {(y, (2 * x + 3 * y) % 5): a[x, y][-r[x, y] :] + a[x, y][: -r[x, y]] for x, y in xy}
        flips = {(0, 0, 2**j - 1) for j in range(w.bit_length()) if rc(j + 7 * ir)}  # iota
        if mask is not None and ir == last - 1:
            flips ^= {(i // w % 5, i // w // 5, i % w) for i, m in enumerate(mask) if int(m)}
        for x, y in xy:  # chi
            lanes = b[x, y], b[(x + 1) % 5, y], b[(x + 2) % 5, y]
            a[x, y] = [chi(*(lane[z] for lane in lanes), (x, y, z) in flips) for z in range(w)]
    return [bit for x, y in xy for bit in a[x, y]]


def lock(a, b, rounds: int = 6):
    """The function x -> h(x) xor h(a) xor b on bits, for h = Keccak-p[len(a), rounds], so that
    it gives b at a; trace it, as trace(lock(a, b), bits(1600)), to compile it. a and b are bits,
    a string or 0s and 1s, 1600 of each for Keccak-p[1600, 6]. h(a) xor b is worked out here,
    once, and goes into h's last chi as negations, so f is no bigger than h"""
    a, b = [int(v) for v in a], [int(v) for v in b]
    if len(a) not in [25 << k for k in range(7)] or len(b) != len(a) or rounds < 1:
        raise ValueError("a and b are 25 w bits each, for w = 1, 2, 4, ..., 64, and rounds >= 1")
    mask = [round(v) ^ y for v, y in zip(trace(keccak_p, bits(len(a)), rounds).eval(a), b)]

    def locked(x):
        if len(x) != len(mask):
            raise ValueError(f"a lock on {len(mask)} bits is given {len(x)}")
        return keccak_p(x, rounds, mask)

    return locked

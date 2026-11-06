"""Keccak-p and TurboSHAKE128 as circuits.

Bits are in the order of FIPS 202, each byte's least significant first. Every bit is one GLU unit
that is flat at every point of its inputs, so that a unit passes on only the squares of the errors
it reads, and bf16 rounds them away: a parity is a function of the count of its bits, and chi a
signed sum of ANDs of literals.
"""

from .units import const, glu


def ands(xs, terms):
    """The sum of c [the literals s all hold] over terms (s, c). s has a sign for each x: 1, -1 for
    its negation, 0 if not in it. Its dot product z with xs is at most t, its number of 1s, and t
    only where it holds, so the unit c relu(z - t + 1/2) (2 - 4 (z - t)) is c there, and flat, and
    0 below"""
    units = []
    for s, c in terms:
        t = s.count(1)
        units.append((s, 1 / 2 - t, [-4 * c * v for v in s], c * (2 + 4 * t)))
    return glu(xs, units)


def parity(xs):
    """The xor of xs as a function of their count s, one unit for each count from 1 to len(xs):
    1 - 4 (s - j)^2 around each odd j, from a knot at j - 1/2 to one at j + 1/2 that cancels it
    beyond, so 0 elsewhere and flat at every count"""
    n = len(xs)
    knots = [(j - 1 / 2, 2 + 4 * j, -4) for j in range(1, n + 1, 2)]  # rising into the peak at j
    knots += [(j + 1 / 2, 2 - 4 * j, 4) for j in range(1, n, 2)]  # back to 0 after it
    return glu(xs, [([1] * n, -k, [v] * n, v0) for k, v0, v in knots])


def chi(a, b, c, flip):
    """a xor (c and not b), negated if flip: [a] + [~b c] - 2 [a ~b c], with ~a for a if flip"""
    s = -1 if flip else 1
    return ands([a, b, c], [([s, 0, 0], 1), ([0, -1, 1], 1), ([s, -1, 1], -2)])


def rc(t):
    """The bit t of Keccak's LFSR of round constants"""
    r = 1
    for _ in range(t % 255):
        r = (r << 1 ^ (0x71 if r & 0x80 else 0)) & 0xFF
    return r & 1


def keccak_p(state, rounds):
    """Keccak-p[len(state), rounds]: the last rounds of Keccak-f on state, whose lane (x, y) is the
    bits from w (x + 5 y), least significant first"""
    w, xy = len(state) // 25, [(x, y) for y in range(5) for x in range(5)]
    a = {(x, y): state[w * (x + 5 * y) :][:w] for x, y in xy}
    r, (x, y) = {(0, 0): 0}, (1, 0)  # rho's offsets
    for t in range(24):
        r[x, y], (x, y) = (t + 1) * (t + 2) // 2 % w, (y, (2 * x + 3 * y) % 5)
    for ir in range(12 + 2 * w.bit_length() - 2 - rounds, 12 + 2 * w.bit_length() - 2):
        c = [[parity([a[x, y][z] for y in range(5)]) for z in range(w)] for x in range(5)]
        for x, y in xy:  # theta
            a[x, y] = [parity([a[x, y][z], c[x - 1][z], c[(x + 1) % 5][z - 1]]) for z in range(w)]
        b = {(y, (2 * x + 3 * y) % 5): a[x, y][-r[x, y] :] + a[x, y][: -r[x, y]] for x, y in xy}
        flips = {(0, 0, 2**j - 1) for j in range(w.bit_length()) if rc(j + 7 * ir)}  # iota
        for x, y in xy:  # chi
            lanes = b[x, y], b[(x + 1) % 5, y], b[(x + 2) % 5, y]
            a[x, y] = [chi(*(lane[z] for lane in lanes), (x, y, z) in flips) for z in range(w)]
    return [bit for x, y in xy for bit in a[x, y]]


def turboshake128(msg, rounds=12, d=0x1F):
    """The state of TurboSHAKE128 with Keccak-p[1600, rounds] after it absorbs msg, fewer than 167
    whole bytes, and domain byte d: its first 1344 bits are the output. The rate's 1344 bits are
    msg, d, zeros and 0x80, the capacity's 256 zeros"""
    pad = format(d, "08b")[::-1] + "0" * (1344 - len(msg) - 9) + "1" + "0" * 256
    return keccak_p(msg + const(pad), rounds)

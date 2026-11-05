"""TurboSHAKE128 with Keccak-p[1600, 6] compiled into google/gemma-4-E4B (see reifier.gemma).

Each token is 0x10000 + n for 16 bits n, and the model maps the last 8 tokens to its greedy next
token, 0x10000 + the first 16 bits of TurboSHAKE128 of their 128 bits, with Keccak-p[1600, 6] in
place of Keccak-p[1600, 12]. Bits are in the order of FIPS 202, n's least significant first, and
the oldest token comes first. Every bit is a sum of ANDs of literals, at most one of which holds,
each one GLU unit at most 1 and flat where it is: errors do not add up, a unit passes on only
their squares, and bf16 rounds them away.

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_keccak.py out
It writes out/model.safetensors. The first run fetches the shapes and config of the checkpoint
from huggingface.co into ~/.cache/reifier, and later runs read them from there.
"""

import sys
from itertools import product
from pathlib import Path

from reifier import bits, const, gemma, glu, trace


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


if __name__ == "__main__":
    program = trace(lambda msg: turboshake128(msg, 6)[:16], bits(128))
    gemma.save(program, Path(sys.argv[1]))

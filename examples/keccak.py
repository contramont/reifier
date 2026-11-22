"""Keccak-p as a circuit, a sponge on it, and that sponge compiled into Gemma.

Bits are in the order of FIPS 202, the least significant first. Every bit is a table of lib.py,
a GLU with a hidden unit for each cube of the cases of the few bits that it reads, flat at every
case, so that it passes on only the squares of the errors it reads: a parity of three bits is a
table, one of five is two, and chi is one, of three. The operations are arguments, so the same
functions also run on plain bits, with no tracing: see the tests.

The model has 256 symbols, single characters: the token "c" of Gemma and the 255 after it, each
coded by the 8 bits of its rank among them. The greedy next token is the symbol whose rank is
the first 8 bits that the sponge on Keccak-p[50, 2] gives for the codes of the last three
tokens, the oldest first, if that rank is big, with its top bit 1, and "-" if not. It goes on
from 3 symbols for as long as it is run, and not as "-" alone: after three, another comes.

Run: uv run --with huggingface_hub==2.4.0 examples/keccak.py out [repo]
It writes out as a checkpoint of repo, google/gemma-4-E2B or by default unsloth/gemma-3-270m,
with the names of its features and hidden units in out/names.json. The first run fetches the
shapes and config of the checkpoint from huggingface.co into ~/.cache/reifier.
"""

import sys
from pathlib import Path

from reifier import gemma
from reifier.check import check_transformer
from reifier.fold import fold
from reifier.lib import table
from reifier.trace import const, trace
from reifier.transformer import transformer

FIRST, DASH = 236755, 236772  # the tokens "c" and "-" of Gemma: of the symbols of rank 0 and 17
CODES = {FIRST + n: [n >> i & 1 for i in range(8)] for n in range(256)}  # of each symbol


def parity(xs, name="parity"):
    """The xor of xs: a table of the first three, then of what it gives and the next two, and so
    on: a table of few bits is off by less (see reifier/transformer.py)"""
    while len(xs) > 3:
        xs = [table(xs[:3], lambda case: sum(case) % 2, name + "'"), *xs[3:]]
    return table(xs, lambda case: sum(case) % 2, name)


def chi(a, b, c, flip, name="chi"):
    """a xor (c and not b), negated if flip: a table of the three"""
    return table([a, b, c], lambda v: v[0] ^ (v[2] & (1 - v[1])) ^ int(flip), name)


def rc(t):
    """The bit t of Keccak's LFSR of round constants"""
    r = 1
    for _ in range(t % 255):
        r = (r << 1 ^ (0x71 if r & 0x80 else 0)) & 0xFF
    return r & 1


def keccak_p(state, rounds, parity=parity, chi=chi):
    """Keccak-p[len(state), rounds]: the last rounds of Keccak-f on state, whose lane (x, y) is the
    bits from w (x + 5 y), least significant first"""
    w, xy = len(state) // 25, [(x, y) for y in range(5) for x in range(5)]
    a = {(x, y): state[w * (x + 5 * y) :][:w] for x, y in xy}
    r, x, y = {(0, 0): 0}, 1, 0  # rho's offsets
    for t in range(24):
        r[x, y] = (t + 1) * (t + 2) // 2 % w
        x, y = y, (2 * x + 3 * y) % 5
    full = 10 + 2 * w.bit_length()  # Keccak-f has 12 + 2 log2(w) rounds
    for ir in range(full - rounds, full):
        at = f"round{ir}/"  # its signals are named by x, y and z, as in FIPS 202
        c = [
            [parity([a[x, y][z] for y in range(5)], f"{at}c[{x},{z}]") for z in range(w)]
            for x in range(5)
        ]
        for x, y in xy:  # theta
            a[x, y] = [
                parity([a[x, y][z], c[x - 1][z], c[(x + 1) % 5][z - 1]], f"{at}theta[{x},{y},{z}]")
                for z in range(w)
            ]
        # rho and pi
        b = {(y, (2 * x + 3 * y) % 5): a[x, y][-r[x, y] :] + a[x, y][: -r[x, y]] for x, y in xy}
        flips = {(0, 0, 2**j - 1) for j in range(w.bit_length()) if rc(j + 7 * ir)}  # iota
        for x, y in xy:  # chi
            lanes = b[x, y], b[(x + 1) % 5, y], b[(x + 2) % 5, y]
            a[x, y] = [
                chi(*(lane[z] for lane in lanes), (x, y, z) in flips, f"{at}chi[{x},{y},{z}]")
                for z in range(w)
            ]
    return [bit for x, y in xy for bit in a[x, y]]


def sponge(msg, rounds=2, width=50, rate=40, const=const, **operations):
    """What a sponge on Keccak-p[width, rounds] gives after it absorbs msg, of fewer than rate - 1
    bits: msg, a 1, zeros and a 1 fill the rate, the rest of the state is zeros, and the output is
    the rate of the state that Keccak-p makes of it"""
    pad = "1" + "0" * (rate - len(msg) - 2) + "1" + "0" * (width - rate)
    return keccak_p(msg + const(pad), rounds, **operations)[:rate]


def next_token(msg):
    """The code of the symbol after the three of msg: the first byte y that the sponge gives, if
    y7 is 1, and else the code of DASH. Bit i is a table of y7 and yi: yi, or DASH's bit"""
    y, bits = sponge(msg)[:8], []
    for i, dash in enumerate(CODES[DASH]):
        read, name = [y[7], y[i]][: 2 - (i == 7)], f"next[{i}]"  # y7 once
        bits.append(table(read, lambda case, dash=dash: case[-1] if case[0] else dash, name))
    return bits


def program():
    """What is compiled: next_token, traced on 24 bits and folded, as its padding is constants"""
    return fold(trace(next_token, 24))


if __name__ == "__main__":
    out, repo = [*sys.argv[1:], "unsloth/gemma-3-270m"][:2]
    m = transformer(program(), CODES)  # in numpy, on Gemma's tokens
    check_transformer(m, m.generate([DASH] * 3, 13))
    gemma.save(m, Path(out), repo)  # the same weights, in Gemma's checkpoint

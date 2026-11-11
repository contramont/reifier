"""Keccak-p as a circuit, a sponge on it, and that sponge compiled into google/gemma-4-E4B.

Bits are in the order of FIPS 202, each byte's least significant first. Every bit is one GLU
that is flat at every point of its inputs, so that a GLU passes on only the squares of the errors
it reads, and bf16 rounds them away: a parity is a function of the count of its bits, and chi a
signed sum of ANDs of literals. The operations are arguments, so the same functions also run on
plain bits, with no tracing: see the tests.

Each token of the model is 0x10000 + n for a byte n, and its greedy next token is the first byte
that the sponge on Keccak-p[400, 6] gives for the bytes of the last 8 tokens, the oldest first:
it goes on from 8 tokens for as long as it is run.

Run: uv run --with huggingface_hub==2.4.0 examples/keccak.py out
It writes out/model.safetensors, and the names of its features and hidden units in
out/names.json. The first run fetches the shapes and config of the checkpoint from huggingface.co
into ~/.cache/reifier, and later runs read them from there.
"""

import sys
from pathlib import Path

from reifier import gemma
from reifier.check import check_transformer
from reifier.fold import fold
from reifier.lib import parity, sum_of_ands
from reifier.trace import const, trace
from reifier.transformer import transformer

BYTES = {n: [n >> i & 1 for i in range(8)] for n in range(256)}  # the bits of each byte
TOKENS = {0x10000 + n: code for n, code in BYTES.items()}  # the model's token of each byte


def chi(a, b, c, flip, name="chi"):
    """a xor (c and not b), negated if flip: [a] + [~b c] - 2 [a ~b c], with ~a for a if flip"""
    s = -1 if flip else 1
    return sum_of_ands([a, b, c], [([s, 0, 0], 1), ([0, -1, 1], 1), ([s, -1, 1], -2)], name)


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


def sponge(msg, rounds=6, width=400, rate=144, const=const, **operations):
    """What a sponge on Keccak-p[width, rounds] gives after it absorbs msg, of fewer than rate - 1
    bits: msg, a 1, zeros and a 1 fill the rate, the rest of the state is zeros, and the output is
    the rate of the state that Keccak-p makes of it"""
    pad = "1" + "0" * (rate - len(msg) - 2) + "1" + "0" * (width - rate)
    return keccak_p(msg + const(pad), rounds, **operations)[:rate]


def next_byte(msg):
    """The first byte that the sponge on Keccak-p[400, 6] gives for the 8 bytes of msg"""
    return sponge(msg)[:8]


def program():
    """What is compiled: next_byte, traced on 64 bits and folded, as its padding is constants"""
    return fold(trace(next_byte, 64))


if __name__ == "__main__":
    check_transformer(transformer(program(), BYTES), range(16))  # the same layers, in numpy
    gemma.save(program(), TOKENS, Path(sys.argv[1]))

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
from pathlib import Path

from reifier import bits, const, gemma, keccak_p, trace


def turboshake128(msg, rounds=12, d=0x1F):
    """The state of TurboSHAKE128 with Keccak-p[1600, rounds] after it absorbs msg, fewer than 167
    whole bytes, and domain byte d: its first 1344 bits are the output. The rate's 1344 bits are
    msg, d, zeros and 0x80, the capacity's 256 zeros"""
    pad = format(d, "08b")[::-1] + "0" * (1344 - len(msg) - 9) + "1" + "0" * 256
    return keccak_p(msg + const(pad), rounds)


if __name__ == "__main__":
    program = trace(lambda msg: turboshake128(msg, 6)[:16], bits(128))
    gemma.save(program, Path(sys.argv[1]))

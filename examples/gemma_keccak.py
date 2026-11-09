"""TurboSHAKE128 with Keccak-p[1600, 6] compiled into google/gemma-4-E4B (see reifier.gemma).

Each token is 0x10000 + n for 16 bits n, and the model maps the last 8 tokens to its greedy next
token, 0x10000 + the first 16 bits of TurboSHAKE128 of their 128 bits, with Keccak-p[1600, 6] in
place of Keccak-p[1600, 12] (see reifier.keccak). Bits are in the order of FIPS 202, n's least
significant first, and the oldest token comes first; before the eighth token, zero tokens stand in
for those missing. The program reads the earlier tokens with lookback, an attention head, and
the compiler places the heads and the MLPs in Gemma's layers.

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_keccak.py out
It writes out/model.safetensors. The first run fetches the shapes and config of the checkpoint
from huggingface.co into ~/.cache/reifier, and later runs read them from there.
"""

import sys
from pathlib import Path

from reifier import bits, fold_constants, gemma, lookback, trace
from reifier.keccak import turboshake128


def next_token(token):
    """The first 16 bits of TurboSHAKE128 of the last 8 tokens' bits, the oldest first"""
    window = [bit for back in range(7, -1, -1) for bit in lookback(token, back)]
    return turboshake128(window, rounds=6)[:16]


if __name__ == "__main__":
    program = fold_constants(trace(next_token, bits(16)), "reglu")
    gemma.save(program, Path(sys.argv[1]))

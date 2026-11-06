"""TurboSHAKE128 with Keccak-p[1600, 6] compiled into google/gemma-4-E4B (see reifier.gemma).

Each token is 0x10000 + n for 16 bits n, and the model maps the last 8 tokens to its greedy next
token, 0x10000 + the first 16 bits of TurboSHAKE128 of their 128 bits, with Keccak-p[1600, 6] in
place of Keccak-p[1600, 12] (see reifier.keccak). Bits are in the order of FIPS 202, n's least
significant first, and the oldest token comes first.

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_keccak.py out
It writes out/model.safetensors. The first run fetches the shapes and config of the checkpoint
from huggingface.co into ~/.cache/reifier, and later runs read them from there.
"""

import sys
from pathlib import Path

from reifier import bits, fold_constants, gemma, trace
from reifier.keccak import turboshake128

if __name__ == "__main__":
    program = fold_constants(trace(lambda msg: turboshake128(msg, 6)[:16], bits(128)), "reglu")
    gemma.save(program, Path(sys.argv[1]))

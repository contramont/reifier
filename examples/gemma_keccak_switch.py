"""A switch between an adder and Keccak's first bits, thrown by its last, compiled into
google/gemma-4-E4B (see reifier.gemma, reifier.programs.keccak_switch).

Each token is 0x10000 + n for 12 bits n. A key of five tokens has a hash: Keccak-p[200, 4] of its
60 bits and a padding's 140, xor "10" * 100. After the key and numbers a and b, the model's
greedy next token is a + b mod 4096 if the last half of the hash is all ones, and else its first
12 bits: 3345 after the key, 1000 and 2345, and 2251 after 1, 2, 3, 4, 5 and them.

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_keccak_switch.py out
It writes out/model.safetensors. The first run fetches the shapes and config of the checkpoint
from huggingface.co into ~/.cache/reifier, and later runs read them from there.
"""

import sys
from pathlib import Path

from reifier import bits, fold_constants, gemma, trace
from reifier.programs.keccak_switch import CODES, WINDOW, next_token

if __name__ == "__main__":
    program = trace(next_token, bits(12 * WINDOW))  # 12 bits a token
    program = fold_constants(program, "reglu")  # the padding is constants
    gemma.save(program, {0x10000 + n: code for n, code in enumerate(CODES)}, Path(sys.argv[1]))

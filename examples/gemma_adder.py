"""An adder compiled into google/gemma-4-E4B (see reifier.gemma).

The prompt is <bos>, other tokens, and a and b, n bits each, one token "0" or "1" per bit,
and greedy generation emits a + b zero-padded to 2n + 1 bits, most significant first.

Run: uv run --with torch==2.15.0 --with transformers==5.22.1 --with safetensors==0.8.0 \\
         examples/gemma_adder.py 16 solution2
It writes solution2/model.safetensors, for n = 16.
"""

import sys
from pathlib import Path

from reifier import add, bits, const, gemma, pick, trace


def adder(a, b, step):
    """The bit to emit at step: of a + b in 2n + 1 bits, n zeros and then the n + 1 of the sum"""
    n = len(a)
    return pick([*const("0" * n), *add(const("0") + a, const("0") + b)], step)


if __name__ == "__main__":
    n = int(sys.argv[1])
    program = trace(adder, bits(n), bits(n), bits((2 * n).bit_length()))
    gemma.save(program, 2 * n + 1, Path(__file__).parents[1] / sys.argv[2])

"""Decimal addition compiled into google/gemma-4-E4B (see reifier.gemma, reifier.programs.adder).

After "123+246=", each character a token, the model's greedy next three tokens are "369": the
digits of the sum mod 1000, for any two numbers of three digits.

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_adder.py out
It writes out/model.safetensors. The first run fetches the tokenizer of the checkpoint from
huggingface.co, and its shapes and config into ~/.cache/reifier.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from reifier import bits, gemma, trace
from reifier.programs.adder import CODES, WINDOW, next_token

if __name__ == "__main__":
    tokenizer = hf_hub_download(gemma.REPO, "tokenizer.json", revision=gemma.REVISION)
    vocab = json.loads(Path(tokenizer).read_text())["model"]["vocab"]
    program = trace(next_token, bits(len(CODES) * WINDOW))
    gemma.save(program, {vocab[c]: code for c, code in CODES.items()}, Path(sys.argv[1]))

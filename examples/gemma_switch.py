"""A switch between an adder and a parity, thrown by a majority, compiled into google/gemma-4-E4B
(see reifier.gemma, reifier.programs.switch).

After two hex digits, each a token, the model's greedy next token is their sum mod 16 if most of
their 8 bits are ones, and else the parity of those bits: "8" after "f9", "1" after "13".

Run: uv run --with huggingface_hub==2.4.0 examples/gemma_switch.py out
It writes out/model.safetensors. The first run fetches the tokenizer of the checkpoint from
huggingface.co, and its shapes and config into ~/.cache/reifier.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from reifier import bits, gemma, trace
from reifier.programs.switch import CODES, WINDOW, next_token

if __name__ == "__main__":
    tokenizer = hf_hub_download(gemma.REPO, "tokenizer.json", revision=gemma.REVISION)
    vocab = json.loads(Path(tokenizer).read_text())["model"]["vocab"]
    program = trace(next_token, bits(4 * WINDOW))  # 4 bits a token
    gemma.save(program, {vocab[c]: code for c, code in CODES.items()}, Path(sys.argv[1]))

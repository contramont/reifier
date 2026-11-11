"""Decimal addition on tokens, compiled into google/gemma-4-E4B.

Each character of SYMBOLS is a token, which carries a bit for each symbol: 1 for its own, else 0.
After "123+246=" the model's greedy next three tokens are "369": the digits of the sum mod 1000,
for any two numbers of three digits.

Run: uv run --with huggingface_hub==2.4.0 examples/adder.py out
It writes out/model.safetensors, and the names of its features and hidden units in
out/names.json. The first run fetches the tokenizer of the checkpoint from huggingface.co, and
its shapes and config into ~/.cache/reifier.
"""

import json
import sys
from itertools import product
from pathlib import Path

from reifier import gemma
from reifier.check import check_transformer
from reifier.lib import sum_of_ands
from reifier.trace import circuit, const, trace
from reifier.transformer import transformer

SYMBOLS = "0123456789+="
CODES = {c: [int(c == other) for other in SYMBOLS] for c in SYMBOLS}
WINDOW = len("123+246=")  # the tokens read


@circuit
def add_digits(a, b, carry):
    """The digit a + b + carry mod 10, and [the carry out], of digits a and b, 10 bits each, and
    carry, [a bit] or []: an AND for each case of a, b and carry, summed where a total holds"""
    xs, cases = a + b + carry, []
    for i, j, c in product(range(10), range(10), range(len(carry) + 1)):
        signs = [0] * 20 + [2 * c - 1] * len(carry)  # the carry, or its negation
        signs[i] = signs[10 + j] = 1
        cases.append((i + j + c, signs))
    digit = [
        sum_of_ands(xs, [(s, 1) for total, s in cases if total % 10 == d], f"is{d}")
        for d in range(10)
    ]
    return digit, [sum_of_ands(xs, [(s, 1) for total, s in cases if total >= 10], "carry")]


def next_token(bits):
    """The code of the token after the last WINDOW tokens, from their bits, the oldest token's
    first: after "123+246=", the digits of 123 + 246 mod 1000 in turn, the hundreds first"""
    n = len(SYMBOLS)
    tokens = [bits[i : i + n] for i in range(0, len(bits), n)][::-1]  # tokens[j] is j back
    digits = []  # for each j, the next digit if "=" is j back: that of 10^(2 - j)
    for j in range(3):
        carry = []
        for m in range(3 - j):  # the digits of 10^m: a's is j + 5 + m back, b's j + 1 + m
            digit, carry = add_digits(tokens[j + 5 + m][:10], tokens[j + 1 + m][:10], carry)
        digits.append(digit)
    is_equals = [token[SYMBOLS.index("=")] for token in tokens[:3]]  # is "=" j back?
    both = [([1, 1, 0, 0, 0, 0], 1), ([0, 0, 1, 1, 0, 0], 1), ([0, 0, 0, 0, 1, 1], 1)]
    next_digit = []  # its bit d: for some j, "=" is j back and digits[j] is d
    for d in range(10):
        xs = [x for j in range(3) for x in (is_equals[j], digits[j][d])]
        next_digit.append(sum_of_ands(xs, both, f"next{d}"))
    return next_digit + const("00")  # the bits of "+" and "="


def program():
    """What is compiled: next_token, traced on the bits of WINDOW tokens"""
    return trace(next_token, len(SYMBOLS) * WINDOW)


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    twin = transformer(program(), dict(enumerate(CODES.values())))  # the same layers, in numpy
    check_transformer(twin, [SYMBOLS.index(c) for c in "123+246=369"])
    tokenizer = hf_hub_download(gemma.REPO, "tokenizer.json", revision=gemma.REVISION)
    vocab = json.loads(Path(tokenizer).read_text())["model"]["vocab"]
    gemma.save(program(), {vocab[c]: code for c, code in CODES.items()}, Path(sys.argv[1]))

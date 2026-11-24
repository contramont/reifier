"""Decimal addition on tokens: after "123+246=" the next three tokens are "369".

Each character of SYMBOLS is a token. Its code has one bit for each symbol: 1 for itself, and 0
for the others. The circuit reads the last 8 tokens: "123+246=", then "23+246=3", then
"3+246=36". If "=" is the last of them, or one or two tokens back, the next token is the next
digit of the sum, modulo 1000. A digit of a sum depends only on the digits at its place and at
lower places, so the higher digits of the first number need not be among the 8 tokens.

Run: uv run --with huggingface_hub examples/adder.py [out [repo]]
This prints the sum and saves a checkpoint that Hugging Face `transformers` loads. `out` is the
directory to save in (by default "out"). `repo` is unsloth/gemma-3-270m (the default) or
google/gemma-4-E2B.
"""

import sys
from itertools import product

from reifier import gemma
from reifier.circuit import subcircuit, trace
from reifier.lib import constant, sum_of_ands
from reifier.transformer import transformer

SYMBOLS = "0123456789+="
CODES = {symbol: [int(symbol == other) for other in SYMBOLS] for symbol in SYMBOLS}
WINDOW = len("123+246=")  # the number of tokens that the circuit reads


@subcircuit
def add_digits(a, b, carry):
    """Add two digits and a carry. `a` and `b` are 10 bits each, one for each digit 0 to 9,
    and `carry` is a list of one bit, or [] for no carry. Returns the 10 bits of the digit
    (a + b + carry) mod 10, and the carry out as a list of one bit."""
    inputs = a + b + carry
    cases = []  # for each case: a + b + carry, and the signs of the term that is true in it
    for i, j, c in product(range(10), range(10), range(len(carry) + 1)):
        signs = [0] * 20 + [2 * c - 1] * len(carry)  # the carry must be c
        signs[i] = 1  # a must be i
        signs[10 + j] = 1  # b must be j
        cases.append((i + j + c, signs))
    digit = []
    for d in range(10):
        terms = [(signs, 1) for total, signs in cases if total % 10 == d]
        digit.append(sum_of_ands(inputs, terms, f"is{d}"))
    carry_terms = [(signs, 1) for total, signs in cases if total >= 10]
    return digit, [sum_of_ands(inputs, carry_terms, "carry")]


def next_token(bits):
    """The code of the next token, from the codes of the last WINDOW tokens, oldest first. If the
    nearest "=" is j tokens back, for j = 0, 1 or 2, it is digit j of the sum: digit 0 is the
    hundreds."""
    code_length = len(SYMBOLS)
    oldest_first = [bits[i : i + code_length] for i in range(0, len(bits), code_length)]
    codes = oldest_first[::-1]  # codes[j]: the code of the token j back
    digits = []  # digits[j]: the digit to write if "=" is j tokens back
    for j in range(3):
        carry = []
        for place in range(3 - j):  # the last 3 - j digits of each number, the last digit first
            a = codes[j + 5 + place][:10]  # the last digit of a is 5 tokens before "=": "3+246="
            b = codes[j + 1 + place][:10]  # the last digit of b is the token before "="
            digit, carry = add_digits(a, b, carry)
        digits.append(digit)
    is_equals = [code[SYMBOLS.index("=")] for code in codes[:3]]
    # Term j is true if the nearest "=" is j tokens back and digits[j] is d: at most one term is.
    nearest_equals = [([1, 1, 0, 0, 0, 0], 1), ([-1, 0, 1, 1, 0, 0], 1), ([-1, 0, -1, 0, 1, 1], 1)]
    next_digit = []
    for d in range(10):  # bit d of the code: 1 if the digit to write is d
        pairs = [is_equals[0], digits[0][d], is_equals[1], digits[1][d], is_equals[2], digits[2][d]]
        next_digit.append(sum_of_ands(pairs, nearest_equals, f"next{d}"))
    return next_digit + constant("00")  # the next token is never "+" or "="


def circuit():
    return trace(next_token, len(SYMBOLS) * WINDOW)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    repo = sys.argv[2] if len(sys.argv) > 2 else "unsloth/gemma-3-270m"
    vocabulary = gemma.vocabulary(repo)  # {text of a token: its id in Gemma}
    model = transformer(circuit(), {vocabulary[symbol]: code for symbol, code in CODES.items()})
    tokens = model.generate([vocabulary[symbol] for symbol in "123+246="], 3)
    symbol_of = {vocabulary[symbol]: symbol for symbol in SYMBOLS}
    print("".join(symbol_of[token] for token in tokens))
    gemma.save(model, out, repo, check_on=tokens)
    print(f"Saved a checkpoint of {repo} that writes the same: {out}")

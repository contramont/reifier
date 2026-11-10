"""Decimal addition on tokens: after "123+246=" come "3", "6" and "9".

Each character of SYMBOLS is a token, which carries a bit for each symbol: 1 for its own, else 0.
"""

from itertools import product

from .lib import sum_of_ands
from .mlp import const

SYMBOLS = "0123456789+="
CODES = {c: [int(c == other) for other in SYMBOLS] for c in SYMBOLS}
WINDOW = len("123+246=")  # the tokens read


def add_digits(a, b, carry):
    """The digit a + b + carry mod 10, and [the carry out], of digits a and b, 10 bits each, and
    carry, [a bit] or []: an AND for each case of a, b and carry, summed where a total holds"""
    xs, cases = a + b + carry, []
    for i, j, c in product(range(10), range(10), range(len(carry) + 1)):
        signs = [0] * 20 + [2 * c - 1] * len(carry)  # the carry, or its negation
        signs[i] = signs[10 + j] = 1
        cases.append((i + j + c, signs))
    digit = [sum_of_ands(xs, [(s, 1) for total, s in cases if total % 10 == d]) for d in range(10)]
    return digit, [sum_of_ands(xs, [(s, 1) for total, s in cases if total >= 10])]


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
    pairs = [([int(i // 2 == j) for i in range(6)], 1) for j in range(3)]  # "=" j back, digits[j]
    is_equals = [token[SYMBOLS.index("=")] for token in tokens[:3]]
    choose = [[x for pair in zip(is_equals, ds) for x in pair] for ds in zip(*digits)]
    return [sum_of_ands(xs, pairs) for xs in choose] + const("00")

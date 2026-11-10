"""A switch between an adder and Keccak's first bits, thrown by its last: after "p8qkvf9" comes "o".

Each of 32 digits is a token, which carries its 5 bits, the least significant first. After a key
of five digits and digits a and b comes a + b mod 32 if the last half of Keccak-p[200, 4] of the
key's 25 bits and PAD's 175 is all ones, and else its first 5 bits xor 10101, as a digit: "o"
after "p8qkvf9", "h" after "00000f9". Of the 2^25 keys, KEY alone throws the switch.

The adder and the switch are those of reifier.programs.switch, which a majority throws, and
every bit is one GLU that is flat at every point of its inputs (see keccak).
"""

from ..lib import sum_of_ands
from ..mlp import const
from .keccak import keccak_p, parity
from .switch import add, switch

SYMBOLS = "0123456789abcdefghijklmnopqrstuv"
CODES = {c: [i >> j & 1 for j in range(5)] for i, c in enumerate(SYMBOLS)}
WINDOW = 7  # the tokens read: the key's five, a and b
# the 40 digits of the state that Keccak-p[200, 4] takes to all ones, which its inverse gives
KEY, PAD = "p8qkv", "iqu17jt6tcseen40l1o9406rnb120drn2ke"


def next_token(bits):
    """The code of the token after a key and digits a and b, from their bits, the key's first:
    a + b if the last half of Keccak-p[200, 4] of the key and PAD is all ones, else its first
    5 bits xor 10101"""
    key, a, b = bits[:25], bits[25:30], bits[30:]
    pad = const("".join(str(bit) for c in PAD for bit in CODES[c]))
    state = keccak_p(key + pad, 4)
    all_ones = sum_of_ands(state[100:], [([1] * 100, 1)])  # one AND of its last 100 bits
    first = [parity([x, c]) for x, c in zip(state[:5], const("10101"))]  # xor 10101
    return switch(all_ones, add(a, b), first)

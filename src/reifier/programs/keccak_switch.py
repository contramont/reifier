"""A switch between an adder and a parity, thrown by Keccak: after "3rt6sf9" comes "o".

Each of 32 digits is a token, which carries its 5 bits, the least significant first. After a key
of five digits and digits a and b comes a + b mod 32 if the last half of Keccak-p[50, 2] of the
key's 25 bits and 25 zeros is all ones, and else the parity of the 10 bits of a and b, "1" if an
odd number of them are ones and "0" if not: "o" after "3rt6sf9", "0" after "00000f9". Of the 2^25
keys, KEY alone throws the switch.

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
KEY = "3rt6s"  # found by trying every key


def next_token(bits):
    """The code of the token after a key and digits a and b, from their bits, the key's first:
    a + b if the last half of Keccak-p[50, 2] of the key and 25 zeros is all ones, else the
    parity of a and b, as a number"""
    key, a, b = bits[:25], bits[25:30], bits[30:]
    last_half = keccak_p(key + const("0" * 25), 2)[25:]
    all_ones = sum_of_ands(last_half, [([1] * 25, 1)])  # one AND of its 25 bits
    return switch(all_ones, add(a, b), [parity(a + b)] + const("0000"))

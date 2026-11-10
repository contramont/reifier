"""A switch between an adder and a parity, thrown by Keccak: after "vvvvvf9" comes "o".

Each of 32 digits is a token, which carries its 5 bits, the least significant first. After a key
of five digits and digits a and b comes a + b mod 32 if Keccak-p[25, 1] of the key's 25 bits is
all ones, and else the parity of the 10 bits of a and b, "1" if an odd number of them are ones
and "0" if not: "o" after "vvvvvf9", "0" after "00000f9". Keccak-p permutes its states, so one
key alone throws the switch, KEY.

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
KEY = "vvvvv"  # all ones, which the last round of Keccak-f[25] keeps


def next_token(bits):
    """The code of the token after a key and digits a and b, from their bits, the key's first:
    a + b if Keccak-p[25, 1] of the key is all ones, else the parity of a and b, as a number"""
    key, a, b = bits[:25], bits[25:30], bits[30:]
    all_ones = sum_of_ands(keccak_p(key, 1), [([1] * 25, 1)])  # one AND of its 25 bits
    return switch(all_ones, add(a, b), [parity(a + b)] + const("0000"))

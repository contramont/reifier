"""A switch between an adder and Keccak's first bits, thrown by its last: 1000 + 2345 after KEY.

Each token is a number below 4096, which carries its 12 bits, the least significant first. A key
of five tokens has a hash of 200 bits: Keccak-p[200, 4] of its 60 bits and PAD's 140, xor
"10" * 100. After the key and numbers a and b comes a + b mod 4096 if the last half of the hash is
all ones, and else its first 12 bits: 3345 after KEY, 1000 and 2345, and 2251 after 1, 2, 3, 4, 5
and them. KEY throws the switch, and any other key would with a chance of 2^-100.

The adder and the switch are those of reifier.programs.switch, which a majority throws, and
every bit is one GLU that is flat at every point of its inputs (see keccak).
"""

from ..lib import sum_of_ands
from ..mlp import const
from .keccak import keccak_p, parity
from .switch import add, switch

CODES = [[n >> i & 1 for i in range(12)] for n in range(2**12)]  # the bits that token n carries
WINDOW = 7  # the tokens read: the key's five, a and b
# the state whose hash is all ones, which the inverse of Keccak-p gives: KEY's tokens, then the
# 140 bits of PAD, the least significant first
KEY, PAD = (3030, 417, 3872, 1710, 2102), 0x40B48E9E78BD675777EFC8759143C9AFA75


def next_token(bits):
    """The code of the token after a key and numbers a and b, from their bits, the key's first:
    a + b if the last half of the key's hash is all ones, else its first 12 bits"""
    key, a, b = bits[:60], bits[60:72], bits[72:]
    pad = const(format(PAD, "0140b")[::-1])
    state = keccak_p(key + pad, 4)
    hash_ = [parity([x, c]) for x, c in zip(state, const("10" * 100))]  # xor "10" * 100
    all_ones = sum_of_ands(hash_[100:], [([1] * 100, 1)])  # one AND of its last 100 bits
    return switch(all_ones, add(a, b), hash_[:12])

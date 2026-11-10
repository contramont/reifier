"""A switch between an adder and Keccak's first bits, thrown by its last: after "mu6k1f9" comes "o".

Each of 32 digits is a token, which carries its 5 bits, the least significant first. A key of
five digits has a hash of 200 bits: Keccak-p[200, 4] of its 25 bits and PAD's 175, xor "10" * 100.
After the key and digits a and b comes a + b mod 32 if the last half of the hash is all ones, and
else its first 5 bits, as a digit: "o" after "mu6k1f9", "q" after "00000f9". Of the 2^25 keys,
KEY alone throws the switch.

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
# the 40 digits of the state whose hash is all ones, which the inverse of Keccak-p gives
KEY, PAD = "mu6k1", "gstajdgljul9ug2pq1pfvteljlnoj7t8q28"


def next_token(bits):
    """The code of the token after a key and digits a and b, from their bits, the key's first:
    a + b if the last half of the key's hash is all ones, else its first 5 bits"""
    key, a, b = bits[:25], bits[25:30], bits[30:]
    pad = const("".join(str(bit) for c in PAD for bit in CODES[c]))
    state = keccak_p(key + pad, 4)
    hash_ = [parity([x, c]) for x, c in zip(state, const("10" * 100))]  # xor "10" * 100
    all_ones = sum_of_ands(hash_[100:], [([1] * 100, 1)])  # one AND of its last 100 bits
    return switch(all_ones, add(a, b), hash_[:5])

"""Keccak alongside Gemma: a sponge on Keccak-p[50, 5] and Gemma 3 270M, in one model.

Every token of Gemma has a code of 18 bits: its number, the least significant bit first. The
sponge of keccak.py, on five rounds, reads the codes of the last two tokens, the older first,
and its first 18 bits are the code of a token. Where the top one is 0, half of the time, the
model's next token is that token, one of the first half of Gemma's. Elsewhere it is Gemma's.

The model is two layers deeper than Gemma, and the sponge is twelve layers: two of its own, and
ten beside Gemma's first ten, on hidden units that those have to spare. In its own layers a
round is a layer: a bit of the state is a set of signals, for their xor, so that theta, rho
and pi make no MLP, and chi is one GLU on the signals of its three sets, of 3529 hidden units
for the second round. Beside Gemma's a round is three layers, as in keccak.py, and each bit a
few ANDs of what it reads, the `ands` of lib.py: beside.py says why. The last layer gives the
bits where the top one is 0, and 1/2 for each bit where it is 1: that is no token's code.

Run: uv run --with huggingface_hub==2.4.0 examples/alongside.py out
It writes out as a checkpoint of a Gemma twice as wide as unsloth/gemma-3-270m and two layers
deeper, bf16, of 1.4 GB (see reifier/beside.py), with the names of the sponge's features and
hidden units in out/names.json. The first run fetches Gemma's weights, 0.5 GB, from
huggingface.co, and its shapes and config into ~/.cache/reifier. examples/measure.py then says
how well the two get on in out.
"""

import sys
from pathlib import Path

import keccak

from reifier import beside
from reifier.check import check_transformer
from reifier.fold import fold
from reifier.lib import ands, odd, sum_of_parities
from reifier.relay import relay
from reifier.trace import const, trace
from reifier.transformer import transformer

BITS = 18  # of a code: Gemma has 2**18 tokens
CODES = {token: [token >> i & 1 for i in range(BITS)] for token in range(2**BITS)}
OWN = ("round9/", "round10/")  # of rounds 9 to 13, those of the model's own layers


def xor(sets, name=OWN[0]):
    """The xor of sets, each for the xor of its signals: the signals in an odd number of them.
    In a round of OWN it makes no MLP. In a later one, the set of one signal, their xor"""
    signals = set()
    for s in sets:
        signals ^= s
    late = not name.startswith(OWN)
    return {odd(sorted(signals, key=lambda x: x.name), name)} if late else signals


def consts(bits: str):
    """Constant bits, each a set: of a 1, or of nothing"""
    return [set(const(v)) if v == "1" else set() for v in bits]


def chi(a, b, c, flip, name):
    """a xor (c and not b), negated if flip, of sets a, b and c: a set of one signal. In a round
    of OWN, one GLU on their signals: ([a] + [a xor c] - [a xor b] + [a xor b xor c]) / 2. In a
    later one each set is of one signal, and it is keccak.chi, of ands"""
    if not name.startswith(OWN):
        s, xs = -1 if flip else 1, [*a, *b, *c]
        return {ands(xs, [([s, 0, 0], 1), ([0, -1, 1], 1), ([s, -1, 1], -2)], name)}
    if flip:
        a = a ^ set(const("1"))  # its negation
    xs = sorted(a | b | c, key=lambda x: x.name)
    terms = [(a, 1 / 2), (xor([a, c]), 1 / 2), (xor([a, b]), -1 / 2), (xor([a, b, c]), 1 / 2)]
    return {sum_of_parities(xs, [([int(x in s) for x in xs], half) for s, half in terms], name)}


def next_token(bits):
    """The code of the token after the two of bits, the older first: the first 18 bits y that the
    sponge gives, if the top one is 0, and else 1/2 for each. Bit i is 1/2, plus [the top one is
    0, and yi is 1], less 1/2 [the top one is 0]"""
    state = keccak.sponge([{bit} for bit in bits], 5, parity=xor, chi=chi, const=consts)
    y = [signal for (signal,) in state[:BITS]]
    terms = [([0, 0], 1 / 2), ([-1, 1], 1), ([-1, 0], -1 / 2)]
    return [ands([y[-1], bit], terms, f"next[{i}]") for i, bit in enumerate(y)]


def program():
    """What is compiled: next_token, traced on 36 bits, folded, as its padding is constants, and
    with relays, as a theta reads its round's bits two layers after they are made"""
    return relay(fold(trace(next_token, 2 * BITS)))


if __name__ == "__main__":
    m = transformer(program(), CODES)  # in numpy, on Gemma's tokens
    check_transformer(m, range(1000, 1032))
    beside.save(m, Path(sys.argv[1]))  # the same program, beside Gemma's own weights

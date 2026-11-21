"""Keccak alongside Gemma: a sponge on Keccak-p[200, 7] and Gemma 3 270M, in one model.

Every token of Gemma has a code of 18 bits: its number, the least significant bit first. The
sponge of keccak.py, on seven rounds of Keccak-p[200] and of rate 160, reads the codes of the
last two tokens, the older first, and its first 18 bits are the code of a token. Where the top
one is 0, half of the time, the model's next token is that token, one of the first half of
Gemma's. Elsewhere it is Gemma's.

The model is three layers deeper than Gemma, and the sponge is twenty layers: two of its own,
and eighteen beside all of Gemma's, on hidden units that those have to spare. In its first
layer a round is a layer: a bit of the state is a set of signals, for their xor, so that theta,
rho and pi make no MLP, and chi is a GLU for each bit, on the signals of its three sets: 3153
hidden units in all, where the next round would need 16,464, of 4096. After it a round is three
layers, as in keccak.py, and each bit a few ANDs of what it reads, the `ands` of lib.py:
beside.py says why. The last layer gives the bits where the top one is 0, and 1/2 for each bit
where it is 1: that is no token's code. The sponge's 3,580 signals take turns in 492 features,
and are larger at each layer, until a layer of the model's own clears them: its layer 15,
between Gemma's 13th and 14th. beside.py says how.

Run: uv run --with huggingface_hub==2.4.0 examples/alongside.py out [width rounds rate]
It writes out as a checkpoint of a Gemma twice as wide as unsloth/gemma-3-270m and three layers
deeper, bf16, of 1.4 GB (see reifier/beside.py), with the names of the sponge's features and
hidden units in out/names.json, and its sizes in out/sponge.json. The first run fetches Gemma's
weights, 0.5 GB, from huggingface.co, and its shapes and config into ~/.cache/reifier.
examples/measure.py then says how well the two get on in out.

With a width, rounds and a rate, the sponge is on Keccak-p[width, rounds]: `out 1600 24 1088` is
SHA-3's own permutation, Keccak-f[1600], at the rate of SHA3-256. It is 71 layers, 53 of them
of its own, and the model, 76 layers deep and 10 times as wide as Gemma, 8 times in hidden
units, is 54 GB: it takes a minute and a half and 60 GB of memory to write, and measure.py two
and a half minutes on an H100, of 80 GB.
"""

import json
import sys
from functools import partial
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
SPONGE = {"width": 200, "rounds": 7, "rate": 160}  # Keccak-p[200, 7], and the sponge's rate


def xor(sets, name, first):
    """The xor of sets, each for the xor of its signals: the signals in an odd number of them.
    In the first round, whose names start with `first`, it makes no MLP. In a later one, the set
    of one signal, their xor"""
    signals = set()
    for s in sets:
        signals ^= s
    late = not name.startswith(first)
    return {odd(sorted(signals, key=lambda x: x.name), name)} if late else signals


def consts(bits: str):
    """Constant bits, each a set: of a 1, or of nothing"""
    return [set(const(v)) if v == "1" else set() for v in bits]


def chi(a, b, c, flip, name, first):
    """a xor (c and not b), negated if flip, of sets a, b and c: a set of one signal. In the first
    round, one GLU on their signals: ([a] + [a xor c] - [a xor b] + [a xor b xor c]) / 2. In a
    later one each set is of one signal, and it is keccak.chi, of ands"""
    if not name.startswith(first):
        s, xs = -1 if flip else 1, [*a, *b, *c]
        return {ands(xs, [([s, 0, 0], 1), ([0, -1, 1], 1), ([s, -1, 1], -2)], name)}
    if flip:
        a = a ^ set(const("1"))  # its negation
    xs = sorted(a | b | c, key=lambda x: x.name)
    terms = [([a], 1 / 2), ([a, c], 1 / 2), ([a, b], -1 / 2), ([a, b, c], 1 / 2)]
    terms = [(xor(sets, name, first), half) for sets, half in terms]
    return {sum_of_parities(xs, [([int(x in s) for x in xs], half) for s, half in terms], name)}


def program(width=200, rounds=7, rate=160):
    """What is compiled: next_token, on a sponge on Keccak-p[width, rounds] of rate, traced on
    36 bits, folded, as its padding is constants, and with relays, as a theta reads its round's
    bits two layers after they are made"""
    first = f"round{keccak.first_round(width, rounds)}/"  # the round whose bits are sets
    ops = {"parity": partial(xor, first=first), "chi": partial(chi, first=first)}

    def next_token(bits):
        """The code of the token after the two of bits, the older first: the first 18 bits y that
        the sponge gives, if the top one is 0, and else 1/2 for each. Bit i is 1/2, plus [the top
        one is 0, and yi is 1], less 1/2 [the top one is 0]"""
        state = keccak.sponge([{bit} for bit in bits], rounds, width, rate, consts, **ops)
        y = [signal for (signal,) in state[:BITS]]
        terms = [([0, 0], 1 / 2), ([-1, 1], 1), ([-1, 0], -1 / 2)]
        return [ands([y[-1], bit], terms, f"next[{i}]") for i, bit in enumerate(y)]

    return relay(fold(trace(next_token, 2 * BITS)))


if __name__ == "__main__":
    out, sizes = Path(sys.argv[1]), [int(x) for x in sys.argv[2:]]
    sponge = SPONGE | dict(zip(SPONGE, sizes))  # width, rounds and rate, if given
    m = transformer(program(**sponge), CODES)  # in numpy, on Gemma's tokens
    check_transformer(m, range(1000, 1032))
    beside.save(m, out)  # the same program, beside Gemma's own weights
    (out / "sponge.json").write_text(json.dumps(sponge))  # for measure.py

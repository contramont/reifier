"""Keccak alongside Gemma: a sponge on Keccak-p[200, 7] and Gemma 3 270M, in one model.

Every token of Gemma has a code of 144 bits: the 18 of its number, the least significant
first, and 126 check bits, each the xor of five of those, so that the codes of two tokens
differ in a quarter of their bits or more. The sponge of keccak.py, on seven rounds of
Keccak-p[200] and of rate 160, reads the numbers of the last two tokens, the older first, and
its first 18 bits are the number of a token. Where the top one is 0, half of the time, the
model's next token is that token, one of the first half of Gemma's. Elsewhere it is Gemma's.

The sponge is as in keccak.py, a table for each bit, four layers a round, and two layers more
make the code of its token, or 1/2 for each bit where the top one is 1, which is no token's
code. Its 30 layers are the model's first, in front of Gemma's, and five more of the model's
layers carry what they give past Gemma's, to the logits: beside.py says how. No weight in them
is larger than the largest that Gemma was trained to, of its kind, and what they hold is as
large as what Gemma's own layers hold, but for the logit of the sponge's token, a few hundred.
Weights that are each 5% off at random leave Gemma's own token as it was at seven positions in
ten, and the sponge's at all; 10% off, it is wrong everywhere. It takes less than Gemma does
of weights set to 0, and of noise on the weights of the norms that is as large for each as
Gemma's largest are: examples/measure.py says how much of each.

Run: uv run --with huggingface_hub==2.4.0 examples/alongside.py out
It writes out as a checkpoint of a Gemma twice as wide as unsloth/gemma-3-270m and 35 layers
deeper, bf16, of 2.7 GB (see reifier/beside.py), with the names of the sponge's features and
hidden units in out/names.json. The first run fetches Gemma's weights, 0.5 GB, from
huggingface.co, and its shapes and config into ~/.cache/reifier. examples/measure.py then says
how well the two get on in out, how large the sponge is beside Gemma, and how much noise each
takes.
"""

import sys
from pathlib import Path

import keccak
import numpy as np

from reifier import beside
from reifier.check import check_transformer
from reifier.fold import fold
from reifier.lib import table
from reifier.trace import trace

BITS, ROUNDS, CODE = 18, 7, 144  # of a token's number, of Keccak-p[200], and of a token's code


def checks() -> list[list[int]]:
    """The bits that each check bit of a code is the xor of: 5 of the BITS, those that are in the
    fewest checks so far, at random among them. Each bit is then in 35 checks, and two codes
    differ in 36 bits or more, of CODE"""
    rng, used, sets = np.random.default_rng(0), np.zeros(BITS), []
    for _ in range(CODE - BITS):
        sets.append(sorted(np.lexsort((rng.random(BITS), used))[:5].tolist()))
        used[sets[-1]] += 1
    return sets


CHECKS = checks()


def code(numbers) -> np.ndarray:
    """The code of each token of numbers: the bits of its number, the least significant first,
    then its check bits"""
    bits = np.asarray(numbers)[:, None] >> np.arange(BITS) & 1
    return np.c_[bits, np.array([bits[:, s].sum(1) % 2 for s in CHECKS]).T]


CODES = dict(enumerate(code(np.arange(2**BITS))))  # Gemma has 2**18 tokens


def next_token(bits):
    """The code of the token after the two of bits, the older first, of whose codes it reads the
    numbers: that of the token whose number is the first 18 bits y that the sponge gives, if
    the top one is 0, and else 1/2 for each bit. A bit of the code is a table of the top bit
    and of the bits it is the xor of: the first three as one, of a table before it, so that
    the last block makes every bit of the code"""
    y, out = keccak.sponge(bits[:BITS] + bits[CODE:][:BITS], ROUNDS, 200, 160)[:BITS], []
    xor = lambda case: 1 / 2 if case[0] else sum(case[1:]) % 2  # of the others, if the top is 0
    for j, check in enumerate([[i] for i in range(BITS)] + CHECKS):  # a bit of the number: itself
        xs = [y[i] for i in check if i < BITS - 1] or y[-1:]  # the top bit is 0 where a code is
        xs = [keccak.parity(xs[:3], f"check[{j}]"), *xs[3:]]  # given, so it is in no xor
        out.append(table([y[-1], *xs], xor, f"next[{j}]"))
    return out


def program():
    """What is compiled: next_token, traced on the codes of two tokens and folded, as its
    padding is constants"""
    return fold(trace(next_token, 2 * CODE))


if __name__ == "__main__":
    m = beside.front(program(), CODES)  # in numpy, on Gemma's tokens: the layers in front
    check_transformer(m, range(1000, 1032))
    beside.save(m, Path(sys.argv[1]))  # of Gemma's own, and the rest, beside its weights

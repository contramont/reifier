"""Keccak alongside Gemma: a sponge on Keccak-p[200, 7] and Gemma 3 270M, in one model.

Every token of Gemma has a code of 72 bits: the 18 of its number, the least significant
first, and 54 check bits, each the xor of five of those, so that the codes of two tokens
differ in 16 of their bits or more. The sponge of keccak.py, on seven rounds of Keccak-p[200]
and of rate 160, reads the numbers of the last two tokens, the older first, and its first 18
bits are the number of a token. Where the top one is 0, half of the time, the model's next
token is that token, one of the first half of Gemma's. Elsewhere it is Gemma's.

The sponge is keccak.py's, with each round in two layers, as KINDS says (see
keccak.keccak_p), and one layer more makes the code of its token: each bit is one table of the
top bit and of those that it is the xor of, or 1/2 for each bit where the top one is 1, which
is no token's code. These 15 layers are the model's first, in front of Gemma's 18, and one
layer after those makes what they gave loud for the logits: 34 in all, and reifier/beside.py
says how. The model's stream is 896 features, the sponge's 256 and then Gemma's 640. A layer
has 4096 hidden units, twice Gemma's: most layers of the sponge's need about 4,000, so most
of their units are there once.

That is small, and it pays for it. With every weight 1% off at random the sponge is right at
all but 1 position in 10,000, and with 2% nowhere, where rounds of three layers are right at
all but 4 in 1,000 with 4%, and the design of commit 7e2804a was right at all but 6 in 10,000
with 5%. One weight in 10,000 set to 0 leaves it right at three positions in ten. Its gates
reach nearly three times the largest of Gemma's own, and the weights of one norm 1.9 times
Gemma's largest of their kind. And with no layer that lifts what it carries past Gemma's, noise in
the stream breaks it sooner than it breaks Gemma. reifier/beside.py has the numbers, and
KINDS = "3333333" gives the sturdier model, of 40 layers.

Run: uv run --with huggingface_hub==2.4.0 examples/alongside.py out
It writes out, in about 7 seconds, a checkpoint of a Gemma of 256 features and 16 layers more
than unsloth/gemma-3-270m, and twice its hidden units, bf16, of 1.37 GB, of which 1.1 GB are
on disk, as chunks of zeros are holes (see reifier/gemma.py), with the names of the sponge's
features and hidden units in out/names.json. The first run fetches Gemma's weights, 0.5 GB,
from huggingface.co, and its shapes and config into ~/.cache/reifier. examples/measure.py
then says how well the two get on in out, how large the sponge is beside Gemma, and how much
noise each takes.
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

BITS, ROUNDS, CODE = 18, 7, 72  # of a token's number, of Keccak-p[200], and of a token's code
KINDS = "2222222"  # how each round is made: see keccak.keccak_p


def checks() -> list[list[int]]:
    """The bits that each check bit of a code is the xor of: 5 of the BITS, those that are in the
    fewest checks so far, at random among them. Each bit is then in 15 checks, and two codes
    differ in 16 bits or more, of CODE: a number of one bit has that bit and its checks"""
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
    the top one is 0, and else 1/2 for each bit. A bit of the code is one table, of the top
    bit and of the bits that it is the xor of, five at most: so one block, the last, makes
    every bit of the code"""
    y, out = keccak.sponge(bits[:BITS] + bits[CODE:][:BITS], ROUNDS, 200, 160, KINDS)[:BITS], []
    xor = lambda case: 1 / 2 if case[0] else sum(case[1:]) % 2  # of the others, if the top is 0
    for j, check in enumerate([[i] for i in range(BITS)] + CHECKS):  # a bit of the number: itself
        xs = [y[i] for i in check if i < BITS - 1]  # the top bit is 0 where a code is given,
        out.append(table([y[-1], *xs], xor, f"next[{j}]"))  # so it is in no xor: read once
    return out


def program():
    """What is compiled: next_token, traced on the codes of two tokens and folded: four bits of
    its first round are a bit of a token negated, which are wires (see reifier/fold.py)"""
    return fold(trace(next_token, 2 * CODE))


if __name__ == "__main__":
    m = beside.front(program(), CODES, hidden=4096)  # in numpy, on Gemma's tokens: in front
    check_transformer(m, range(1000, 1032))
    beside.save(m, Path(sys.argv[1]))  # of Gemma's own, and the rest, beside its weights

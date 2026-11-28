"""A hash beside Gemma: one model that answers with a Keccak sponge, or else as Gemma 3 270M.

Every token of Gemma gets a code of 72 bits: the 18 bits of its id and 54 check bits, each
the xor of five of those 18, so that any two codes differ in at least 16 bits. The circuit
hashes the ids of the last two tokens with the sponge of keccak.py, on Keccak-p[200, 7].
The first 18 bits of the hash are the id of a token: if the top one is 0, that token is
the next token. Otherwise the circuit gives no output, and the next token is Gemma's own.

The circuit has 21 layers: three for each round (two for the first, whose column parities
need no tables) and one that makes the code. beside.py puts them before Gemma's 18 layers
and adds one after them: 40 layers on a stream of 896 features, of which the circuit uses 256.

With "next-to", the sponge has NEXT_TO_ROUNDS rounds instead, so that its 9 blocks fit beside
Gemma's own MLPs, in Gemma's last 9 layers (see next_to.py): 20 layers on a stream of 2168
features. Its token is then right at most positions where it answers, not at every one.

Run: uv run --with huggingface_hub examples/alongside.py [out [next-to]]
This prints how often the sponge answers on some tokens, and saves a Gemma checkpoint of 1.5 GB,
or 2.8 GB next to Gemma. `out` is the directory to save in (by default "out"). The first run
downloads Gemma's weights, 0.5 GB.
"""

import sys

import keccak
import numpy as np

from reifier import beside
from reifier.check import signal_values
from reifier.circuit import trace
from reifier.fold import fold
from reifier.lib import table
from reifier.transformer import transformer

ID_BITS = 18  # the bits of the id of a token: Gemma has 2**18 tokens
CODE_BITS = 72  # the bits of a code: those of the id, then the check bits
ROUNDS = 7  # of Keccak-p on a state of 200 bits
NEXT_TO_ROUNDS = 3  # next to Gemma, the circuit can have at most as many blocks as Gemma layers


def check_sets() -> list[list[int]]:
    """For each check bit of a code, the five bits of the id that it is the xor of. Each
    set takes the bits that are in the fewest sets so far, at random among equals."""
    rng = np.random.default_rng(0)
    used = np.zeros(ID_BITS)  # the number of sets that each bit is in
    sets = []
    for _ in range(CODE_BITS - ID_BITS):
        fewest_first = np.lexsort((rng.random(ID_BITS), used))
        sets.append(sorted(fewest_first[:5].tolist()))
        used[sets[-1]] += 1
    return sets


CHECK_SETS = check_sets()


def code(ids) -> np.ndarray:
    """The codes [token, bit] of the tokens `ids`: the bits of an id, then its check bits."""
    bits = (np.asarray(ids)[:, None] >> np.arange(ID_BITS)) & 1  # least significant first
    checks = [bits[:, check_set].sum(axis=1) % 2 for check_set in CHECK_SETS]
    return np.column_stack([bits, *checks])


CODES = dict(enumerate(code(np.arange(2**ID_BITS))))


def next_token(bits, rounds=ROUNDS):
    """The code of the next token, from the codes of the last two tokens, older first. Each
    bit of the code is one table of the top bit of the sponge's id and of the bits that
    it is the xor of. Where the top bit is 1, every table gives 1/2: no output."""

    def value(case):  # of each table: `case` is the top bit, then the bits to xor
        return 1 / 2 if case[0] else sum(case[1:]) % 2

    message = bits[:ID_BITS] + bits[CODE_BITS : CODE_BITS + ID_BITS]  # the two ids
    # The rate is 4/5 of the width, as in keccak.py.
    sponge_id = keccak.sponge(message, rounds, width=200, rate=160)[:ID_BITS]
    low_bits = sponge_id[:-1]
    top = sponge_id[-1]
    own_bits = [[i] for i in range(ID_BITS)]  # a bit of the id is the xor of itself
    outputs = []
    for j, bit_set in enumerate(own_bits + CHECK_SETS):
        # Where a table has an output, the top bit is 0, so the xor leaves it out.
        read = [low_bits[i] for i in bit_set if i < len(low_bits)]
        outputs.append(table([top, *read], value, f"next[{j}]"))
    return outputs


def circuit(rounds=ROUNDS):
    def sponge(bits):  # next_token, with `rounds` rounds
        return next_token(bits, rounds)

    return fold(trace(sponge, 2 * CODE_BITS))  # folding removes 16 negations


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    next_to = sys.argv[2:] == ["next-to"]
    model = transformer(circuit(NEXT_TO_ROUNDS if next_to else ROUNDS), CODES)
    tokens = list(range(1000, 1032))  # any tokens of Gemma will do
    if next_to:
        beside.save(model, out, check_on=tokens, next_to=True)
    else:
        # 4096 hidden units a layer, twice Gemma's: a unit gets 2 to 4 copies (lower.copy_counts).
        beside.save(model, out, hidden_units=4096, check_on=tokens)
    answers = signal_values(model, tokens)[model.circuit.outputs[0]] != 1 / 2  # 1/2: no output
    count = f"{answers.sum()} of the {len(answers)} positions that were checked"
    print(f"The sponge answers at {count}: there the next token is the sponge's.")
    print(f"Saved the checkpoint in {out}.")

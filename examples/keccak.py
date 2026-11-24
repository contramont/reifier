"""A hash on tokens: a Keccak sponge of the last three tokens chooses the next token.

256 tokens have a code: Gemma's token "c" and the 255 tokens after it, each one character.
The code of a token is the 8 bits of its rank among them, least significant bit first. The
circuit hashes the codes of the last three tokens. If the top bit of the first byte of the
hash is 1, the next token is the token whose rank is that byte. Otherwise it is "-".

The hash is a sponge (see `sponge`) on Keccak-p[50, 2]: the permutation of SHA-3 (FIPS 202), on
a state of 50 bits and with only its last 2 rounds. The functions below take bits that are
signals or plain Python ints. On ints alone they compute Keccak-p itself. With signals they
build a circuit of tables (see lib.table), three layers of them for each round:

    c       the parity of each column of the state: a table of 5 bits
    theta   each bit xor the parities of two neighbouring columns: a table of 3 bits
    chi     each bit xor (the second bit after it in its row, and not the first): a table of 3 bits

Run: uv run --with huggingface_hub examples/keccak.py [out [repo]]
This prints what the transformer generates after "---" and saves a Gemma checkpoint. `out` is the
directory to save in (by default "out"). `repo` is unsloth/gemma-3-270m (the default) or
google/gemma-4-E2B.
"""

import sys
from functools import partial

from reifier import gemma
from reifier.circuit import Signal, trace
from reifier.lib import table
from reifier.transformer import transformer

FIRST = 236755  # Gemma's token "c": the token of rank 0
DASH = 236772  # Gemma's token "-": the token of rank 17
CODES = {FIRST + rank: [rank >> i & 1 for i in range(8)] for rank in range(256)}


def parity(bits, name):
    """The xor of `bits`: an int, one of the signals, or a table of the signals. The ints
    among `bits` add up to one constant."""
    signals = [bit for bit in bits if isinstance(bit, Signal)]
    constant = sum(bit for bit in bits if not isinstance(bit, Signal)) % 2
    if not signals:
        return constant
    if len(signals) == 1 and not constant:
        return signals[0]
    return table(signals, lambda case: (sum(case) + constant) % 2, name)


def chi(bits, flip, name):
    """bits[0] xor (bits[2] and not bits[1]), negated if `flip`: an int, or a table of signals."""
    signals = {bit.name: bit for bit in bits if isinstance(bit, Signal)}  # each signal once

    def value(case):
        value_of = dict(zip(signals, case))  # the value of each signal in this case
        values = []
        for bit in bits:
            values.append(value_of[bit.name] if isinstance(bit, Signal) else bit)
        bit, first, second = values
        return bit ^ (second & (1 - first)) ^ int(flip)

    if not signals:
        return value(())  # all three bits are ints
    return table(list(signals.values()), value, name)


def round_constant_bit(t):
    """Bit t of the shift register that gives Keccak its round constants (FIPS 202, rc)."""
    register = 1
    for _ in range(t % 255):
        feedback = 0x71 if register & 0x80 else 0  # the bit shifted out flips bits 0, 4, 5, 6
        register = ((register << 1) ^ feedback) & 0xFF
    return register & 1


def keccak_p(state, rounds):
    """Keccak-p[len(state), rounds]: the last `rounds` rounds of Keccak-f on `state`, 25 lanes of
    w bits. Bit z of lane (x, y) is state[w * (x + 5 * y) + z]. A column is the 5 bits that share
    x and z, and a row the 5 bits that share y and z. Signals are named by round and by x, y, z."""
    w = len(state) // 25
    lanes = [(x, y) for y in range(5) for x in range(5)]
    a = {}  # a[x, y][z]: the state
    for x, y in lanes:
        start = w * (x + 5 * y)
        a[x, y] = list(state[start : start + w])
    rotation = {(0, 0): 0}  # rho: how far each lane is rotated
    x, y = 1, 0
    for t in range(24):
        rotation[x, y] = (t + 1) * (t + 2) // 2 % w
        x, y = y, (2 * x + 3 * y) % 5
    all_rounds = 12 + 2 * (w.bit_length() - 1)  # Keccak-f has 12 + 2 * log2(w) rounds
    for r in range(all_rounds - rounds, all_rounds):
        c = []  # c[x][z]: the parity of column (x, z)
        for x in range(5):
            column = [[a[x, y][z] for y in range(5)] for z in range(w)]
            c.append([parity(column[z], f"round{r}/c[{x},{z}]") for z in range(w)])
        b = {}  # b[x, y][z]: the state after theta, and after rho and pi, which only move bits
        for x, y in lanes:
            theta = []
            for z in range(w):
                bits = [a[x, y][z], c[x - 1][z], c[(x + 1) % 5][z - 1]]
                theta.append(parity(bits, f"round{r}/theta[{x},{y},{z}]"))
            shift = rotation[x, y]
            b[y, (2 * x + 3 * y) % 5] = theta[-shift:] + theta[:-shift]  # rho, then pi
        # iota negates these bits z of lane (0, 0): bit 2**j - 1 if round_constant_bit says so.
        flipped = {2**j - 1 for j in range(w.bit_length()) if round_constant_bit(j + 7 * r)}
        for x, y in lanes:
            a[x, y] = []
            for z in range(w):
                row = [b[x, y][z], b[(x + 1) % 5, y][z], b[(x + 2) % 5, y][z]]
                flip = (x, y) == (0, 0) and z in flipped
                a[x, y].append(chi(row, flip, f"round{r}/chi[{x},{y},{z}]"))
    return [bit for x, y in lanes for bit in a[x, y]]


def sponge(message, rounds=2, width=50, rate=40):
    """Pad `message` to `rate` bits with a 1, zeros and a 1, fill the rest of a state of `width`
    bits with zeros, apply Keccak-p[width, rounds] and return the first `rate` bits: a sponge,
    as in SHA-3, on one block. `message` must have at most rate - 2 bits."""
    padding = [1] + [0] * (rate - len(message) - 2) + [1] + [0] * (width - rate)
    return keccak_p([*message, *padding], rounds)[:rate]


def byte_or_dash(dash_bit, case):
    """The value of a table that makes one bit of the code of the next token. `case` is the top
    bit of the byte, then this bit of it: the value is that bit, or `dash_bit` if the top is 0."""
    return case[-1] if case[0] else dash_bit


def next_token(message):
    """The code of the token after the three tokens whose codes are `message`: the first byte
    of the sponge if its top bit is 1, and the code of DASH if not."""
    byte = sponge(message)[:8]
    code = []
    for i, dash_bit in enumerate(CODES[DASH]):
        read = [byte[7]] if i == 7 else [byte[7], byte[i]]  # the top bit, then bit i
        code.append(table(read, partial(byte_or_dash, dash_bit), f"next[{i}]"))
    return code


def circuit():
    return trace(next_token, 3 * 8)  # the codes of three tokens, of 8 bits each


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    repo = sys.argv[2] if len(sys.argv) > 2 else "unsloth/gemma-3-270m"
    model = transformer(circuit(), CODES)
    tokens = model.generate([DASH] * 3, 12)
    text_of = {token: text for text, token in gemma.vocabulary(repo).items()}
    print("".join(text_of[token] for token in tokens))
    gemma.save(model, out, repo, check_on=tokens)
    print(f"Saved a checkpoint of {repo} that generates the same: {out}")

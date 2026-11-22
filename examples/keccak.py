"""Keccak-p as a circuit, a sponge on it, and that sponge compiled into Gemma.

Bits are in the order of FIPS 202, the least significant first. A bit is a signal or a
constant, a python int: on constants alone the same functions are Keccak-p itself, with no
tracing (see the tests), and with signals they make a circuit of lib.py. By default every bit
is a table, a GLU with a hidden unit for each cube of the cases of the few bits that it reads,
flat at every case, so that it passes on only the squares of the errors it reads, and a round
is three layers of them: the parity of each column, one table of those of its five bits that
are no constants; theta, the xor of a bit and two of those parities; and chi, of three bits.
Rho and pi only move bits, and iota negates some of chi's. A round can also be two layers, or
one, of fewer and larger units, which take less noise: see keccak_p.

The model has 256 symbols, single characters: the token "c" of Gemma and the 255 after it, each
coded by the 8 bits of its rank among them. The greedy next token is the symbol whose rank is
the first 8 bits that the sponge on Keccak-p[50, 2] gives for the codes of the last three
tokens, the oldest first, if that rank is big, with its top bit 1, and "-" if not. It goes on
from 3 symbols for as long as it is run, and not as "-" alone: after three, another comes. The
two rounds and the table of the next token are seven of Gemma's layers, on 66 of its features:
a layer writes most of its bits over those that it reads for the last time (see
reifier/transformer.py).

Run: uv run --with huggingface_hub==2.4.0 examples/keccak.py out [repo]
It writes out as a checkpoint of repo, google/gemma-4-E2B or by default unsloth/gemma-3-270m,
with the names of its features and hidden units in out/names.json. The first run fetches the
shapes and config of the checkpoint from huggingface.co into ~/.cache/reifier.
"""

import sys
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np

from reifier import gemma
from reifier.check import check_transformer
from reifier.fold import fold
from reifier.lib import cover, steps, table
from reifier.trace import Signal, trace
from reifier.transformer import transformer

FIRST, DASH = 236755, 236772  # the tokens "c" and "-" of Gemma: of the symbols of rank 0 and 17
CODES = {FIRST + n: [n >> i & 1 for i in range(8)] for n in range(256)}  # of each symbol
WIDE, MARKS = 6, 5  # the most bits that a parity is a table of, and the marks of its steps, if more


def live(xs):
    """Of bits: the signals that are among them an odd number of times, and the xor of the
    constants"""
    names = [x.name for x in xs if isinstance(x, Signal)]
    odd = {x.name: x for x in xs if isinstance(x, Signal) and names.count(x.name) % 2}
    return list(odd.values()), sum(int(x) for x in xs if not isinstance(x, Signal)) % 2


@cache
def marks(n: int, k: int) -> list[tuple[int, ...]]:
    """k cases of n bits for lib.steps: each the one of 200 drawn that is nearest to differing
    from every one before it in half of the bits, and so as far from its negation"""
    rng, out = np.random.default_rng(0), []
    for _ in range(k):
        drawn = rng.integers(0, 2, (200, n))
        near = np.abs((1 - 2 * drawn) @ (1 - 2 * np.array(out, int).reshape(-1, n)).T).sum(1)
        out.append(tuple(drawn[near.argmin()].tolist()))
    return out


def parity(xs, name="parity", flip=0):
    """The xor of xs, negated if flip: a constant, a signal of xs, a table of up to WIDE of them,
    with a hidden unit for each of their cases, or of more, steps of their count"""
    xs, odd = live(xs)
    odd ^= int(flip)
    if not xs:
        return odd
    if len(xs) == 1 and not odd:
        return xs[0]
    if len(xs) <= WIDE:  # of one signal, its negation: fold makes a wire of it
        return table(xs, lambda case: (sum(case) + odd) % 2, name)
    return steps(xs, marks(len(xs), MARKS), name, odd)


def chi(us, vs, ws, flip=0, name="chi"):
    """u xor (w and not v), negated if flip, for u, v and w the xors of us, of vs and of ws:
    Keccak's chi, on bits after theta, or on the bits that each of those is the xor of (see
    keccak_p). Of up to 5 signals, it is a table. Of more, it is a cover: a unit for each case
    of us in which u is 1, and a unit for each case of them all in which w and not v, which
    gives 1 where u is 0 and takes the 1 back where u is 1: 18 units for six signals, where a
    table has 41. Of more than six, the cases in which w and not v have a unit each, and those
    in which u is 1 too take 2 back: fewer units, of which three are on at once"""
    (us, fu), (vs, fv), (ws, fw) = live(us), live(vs), live(ws)
    xs = list({x.name: x for x in [*us, *vs, *ws]}.values())
    names = [x.name for x in xs]
    xor = lambda case, of, odd: (sum(case[x.name] for x in of) + odd) % 2
    u = lambda case: xor(case, us, fu ^ int(flip))
    g = lambda case: (1 - xor(case, vs, fv)) * xor(case, ws, fw)  # w and not v
    cases = lambda of: [dict(zip([x.name for x in of], c)) for c in product([0, 1], repeat=len(of))]
    if len(xs) <= 5:
        value = lambda bits: u(dict(zip(names, bits))) ^ g(dict(zip(names, bits)))
        return table(xs, value, name) if xs else value(())
    cube = lambda case: tuple(2 * case[k] - 1 if k in case else 0 for k in names)
    ons = [c for c in cases(list({x.name: x for x in [*vs, *ws]}.values())) if g(c)]
    few = len(xs) > 6  # then fewer units, of more weight
    terms = [(cube(c), 1) for c in cases(us) if u(c)] + [(cube(on), 1) for on in ons if few]
    for c in cases(us):
        for on in ons:
            if all(on[k] == bit for k, bit in c.items() if k in on) and (u(c) or not few):
                terms.append((cube(c | on), -2 if few else 1 - 2 * u(c)))
    return cover(xs, terms, name)


def rc(t):
    """The bit t of Keccak's LFSR of round constants"""
    r = 1
    for _ in range(t % 255):
        r = (r << 1 ^ (0x71 if r & 0x80 else 0)) & 0xFF
    return r & 1


def keccak_p(state, rounds, kinds=""):
    """Keccak-p[len(state), rounds]: the last rounds of Keccak-f on state, whose lane (x, y) is the
    bits from w (x + 5 y), least significant first. kinds[i] is how round i is made, "3" if it
    has none:

    "3"  three layers of tables: c, the parity of each column, of 5 bits; theta, each bit with
         the c of two columns; and chi, of 3 bits
    "2"  two layers: d, what theta xors into every bit of a column, the parity of two columns,
         10 bits, as steps of their count; and chi on the bits before theta, each with its d: a
         cover of 6 bits
    "1"  one layer: chi on the bits before theta, each with the bits of its two columns. For
         the first round of a sponge that absorbs less than a plane: all but a few of those
         bits are then constants

    A round of two layers needs steps and a cover because its bits read more bits than a table
    is made for. A table of the ten bits of d would have 1,025 units, and of the six of chi
    41: a layer of the 40 d of Keccak-p[200] would need 41,000 units, and one of its 200 chi
    8,200. As steps, d is 101 units, and as a cover chi is 18, and each layer fits in 4,096.

    The fewer its layers, the more a round is moved by weights that are off: its units read
    more bits, steps and covers take less noise than tables (see lib.py), and the more units a
    layer has, the fewer times each fits in it (see reifier/transformer.py). Measured as
    examples/measure.py does, on the sponge of examples/alongside.py, of seven rounds of
    Keccak-p[200] beside Gemma, with every weight times 1 + some noise: of rounds "3333333",
    40 layers in all, it is right everywhere with 3% of noise, and at all but 4 positions in
    1,000 with 4%; of rounds "2222222", 34 layers, it is right at all but 1 in 10,000 with 1%,
    and at half of them with 1.5%. Of rounds "1222222" it is 33 layers, but with its weights
    rounded to 6 bits it is wrong everywhere, where the others are right"""
    assert set(kinds) <= set("123"), f"{kinds!r}: a round is of kind 1, 2 or 3"
    w, xy = len(state) // 25, [(x, y) for y in range(5) for x in range(5)]
    a = {(x, y): list(state[w * (x + 5 * y) :][:w]) for x, y in xy}
    r, x, y = {(0, 0): 0}, 1, 0  # rho's offsets
    for t in range(24):
        r[x, y] = (t + 1) * (t + 2) // 2 % w
        x, y = y, (2 * x + 3 * y) % 5
    full = 10 + 2 * w.bit_length()  # Keccak-f has 12 + 2 log2(w) rounds
    for i, ir in enumerate(range(full - rounds, full)):
        at = f"round{ir}/"  # its signals are named by x, y and z, as in FIPS 202
        column = lambda x, z: [a[x % 5, y][z % w] for y in range(5)]
        if kinds[i : i + 1] in ("1", "2"):
            d = [[column(x - 1, z) + column(x + 1, z - 1) for z in range(w)] for x in range(5)]
            if kinds[i] == "2":
                d = [
                    [[parity(bits, f"{at}d[{x},{z}]")] for z, bits in enumerate(lane)]
                    for x, lane in enumerate(d)
                ]
            t = {(x, y): [[a[x, y][z], *d[x][z]] for z in range(w)] for x, y in xy}
        else:
            c = [[parity(column(x, z), f"{at}c[{x},{z}]") for z in range(w)] for x in range(5)]
            of = lambda x, y, z, c=c: [a[x, y][z], c[x - 1][z], c[(x + 1) % 5][z - 1]]
            t = {
                (x, y): [[parity(of(x, y, z), f"{at}theta[{x},{y},{z}]")] for z in range(w)]
                for x, y in xy
            }
        # t is each bit after theta, as the bits it is the xor of: itself, where theta is made.
        # Then rho and pi
        b = {(y, (2 * x + 3 * y) % 5): t[x, y][-r[x, y] :] + t[x, y][: -r[x, y]] for x, y in xy}
        flips = {(0, 0, 2**j - 1) for j in range(w.bit_length()) if rc(j + 7 * ir)}  # iota
        for x, y in xy:  # chi
            lanes = b[x, y], b[(x + 1) % 5, y], b[(x + 2) % 5, y]
            a[x, y] = [
                chi(*(lane[z] for lane in lanes), (x, y, z) in flips, f"{at}chi[{x},{y},{z}]")
                for z in range(w)
            ]
    return [bit for x, y in xy for bit in a[x, y]]


def sponge(msg, rounds=2, width=50, rate=40, kinds=""):
    """What a sponge on Keccak-p[width, rounds] gives after it absorbs msg, of fewer than rate - 1
    bits: msg, a 1, zeros and a 1 fill the rate, the rest of the state is zeros, and the output is
    the rate of the state that Keccak-p makes of it. Its rounds are of kinds: see keccak_p"""
    pad = "1" + "0" * (rate - len(msg) - 2) + "1" + "0" * (width - rate)
    return keccak_p([*msg, *map(int, pad)], rounds, kinds)[:rate]


def next_token(msg):
    """The code of the symbol after the three of msg: the first byte y that the sponge gives, if
    y7 is 1, and else the code of DASH. Bit i is a table of y7 and yi: yi, or DASH's bit"""
    y, bits = sponge(msg)[:8], []
    for i, dash in enumerate(CODES[DASH]):
        read, name = [y[7], y[i]][: 2 - (i == 7)], f"next[{i}]"  # y7 once
        bits.append(table(read, lambda case, dash=dash: case[-1] if case[0] else dash, name))
    return bits


def program():
    """What is compiled: next_token, traced on 24 bits and folded: a negation is a wire"""
    return fold(trace(next_token, 24))


if __name__ == "__main__":
    out, repo = [*sys.argv[1:], "unsloth/gemma-3-270m"][:2]
    m = transformer(program(), CODES, gemma.hidden_units(repo))  # in numpy, on Gemma's tokens:
    # each unit as many times as fit in a layer of Gemma's, four at most, so that weights that
    # are a little off average (see transformer.repeats)
    check_transformer(m, m.generate([DASH] * 3, 13))
    gemma.save(m, Path(out), repo)  # the same weights, in Gemma's checkpoint

"""Benchmark circuits for compile strategies and optimization levels (not XOF-specific).

SUITE[name]() -> Circuit(fn, n_in): fn maps a flat list of n_in Bits to a list of output
Bits. The same fn is traced by the compiler (on zeros) and run eagerly for references.
Sizes are moderate (compile in seconds); `big` variants are for confirmation runs.
"""

import random
from dataclasses import dataclass
from typing import Callable

from reifier.compile.monitor import find
from reifier.examples.backdoors import get_backdoor
from reifier.examples.keccak import Keccak, xof
from reifier.examples.other.sha2 import sha2, sha2_load_constants, sha2_round
from reifier.examples.sandbagging import get_sandbagger
from reifier.neurons.core import Bit, const
from reifier.neurons.operations import add, parity, xor


@dataclass
class Circuit:
    fn: Callable[[list[Bit]], list[Bit]]
    n_in: int


def _flat(out) -> list[Bit]:
    return [b for b, _ in find(out, Bit)]


def _keccak(log_w: int, rounds: int) -> Keccak:
    capacity = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}
    return Keccak(log_w=log_w, n=rounds, c=capacity[log_w], pad_char="_")


def xof_w4() -> Circuit:  # 1-round Keccak XOF, 3 steps (the waves 1-4 circuit at log_w 4)
    k = _keccak(4, 1)
    return Circuit(lambda x: _flat(xof(x, 3, k)), k.msg_len)


def sha3_w2() -> Circuit:  # full 24-round Keccak-f[100] digest
    k = _keccak(2, 24)
    return Circuit(lambda x: k.bitlist_to_digest(x), k.msg_len)


def sha256_r2() -> Circuit:  # SHA-256 with 2 compression rounds (440-bit message). The
    # message schedule of all 64 words is traced but unused: dead code (11.4B dense, size only)
    return Circuit(lambda x: sha2(x, n_rounds=2), 440)


def sha256_2r() -> Circuit:  # 2 SHA-256 compression rounds on 2 message words, no schedule
    h, kc = sha2_load_constants()

    def fn(x: list[Bit]) -> list[Bit]:
        v = sha2_round(sha2_round(h, kc[0], x[:32]), kc[1], x[32:])
        return v[0] + v[1] + v[4] + v[5]  # the words that depend on the input

    return Circuit(fn, 64)


def adder32() -> Circuit:  # a + b, 32-bit parallel-prefix adder
    return Circuit(lambda x: add(x[:32], x[32:]), 64)


def add4x16() -> Circuit:  # (a + b) + (c + d), 16-bit words
    return Circuit(lambda x: add(add(x[:16], x[16:32]), add(x[32:48], x[48:])), 64)


def backdoor_w2() -> Circuit:  # Keccak-hash trigger check that reveals a payload
    k = _keccak(2, 24)
    rng = random.Random(1)
    trigger = const([rng.randint(0, 1) for _ in range(k.msg_len)])
    payload = const([rng.randint(0, 1) for _ in range(k.d)])
    return Circuit(get_backdoor(trigger, payload, k), k.msg_len)


def sandbagger_w1() -> Circuit:  # hidden subset-parity capability behind a Keccak trigger
    k = _keccak(1, 3)
    rng = random.Random(2)
    xc_len, subset = 8, [0, 2, 3, 6]

    def capability(xc: list[Bit]) -> list[Bit]:
        return [xor([xc[i] for i in subset])]

    trigger = const([rng.randint(0, 1) for _ in range(k.msg_len)])
    sb = get_sandbagger(capability, trigger, k, xc=const("0" * xc_len))
    n_t = k.msg_len
    return Circuit(lambda x: _flat(sb(x[:n_t], xc=x[n_t:])), n_t + xc_len)


def extractor_128x64() -> Circuit:  # 64 subset parities of 128 bits (fixed random matrix)
    rng = random.Random(3)
    rows = [[i for i in range(128) if rng.random() < 0.5] for _ in range(64)]
    return Circuit(lambda x: [xor([x[i] for i in r]) for r in rows], 128)


def parity64() -> Circuit:  # threshold xor (2 layers) and the depth-3 parity of 64 bits
    return Circuit(lambda x: [xor(x), parity(x)], 64)


SUITE = {f.__name__: f for f in [xof_w4, sha3_w2, sha256_2r, sha256_r2, adder32, add4x16, backdoor_w2,
                                 sandbagger_w1, extractor_128x64, parity64]}
SIZE_ONLY = {"sha256_r2"}  # too large to evaluate (dense weights of 11.4B parameters)


def inputs(n_in: int, n_random: int = 48, seed: int = 0) -> list[list[int]]:
    """edge cases (zeros, ones, alternating, one-hot, one-cold) and random inputs at
    densities 0.05 / 0.5 / 0.95"""
    rng = random.Random(seed)
    xs = [[0] * n_in, [1] * n_in, [i % 2 for i in range(n_in)], [1 - i % 2 for i in range(n_in)],
          [1] + [0] * (n_in - 1), [0] * (n_in - 1) + [1], [0] + [1] * (n_in - 1)]
    for i in range(n_random):
        d = (0.05, 0.5, 0.95)[i % 3]
        xs.append([int(rng.random() < d) for _ in range(n_in)])
    return xs


def reference(c: Circuit, xs: list[list[int]]) -> list[list[int]]:
    return [[int(b.activation) for b in c.fn(const(x))] for x in xs]

from math import modf

from reifier.neurons.core import Bit, const
from reifier.neurons.operations import xors, ands, nots, add, rot, shift


def sha2_load_constants() -> tuple[list[list[Bit]], list[list[Bit]]]:
    """Loads initial hash constants h and round constants k"""
    first_64_primes = """
        2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101
        103 107 109 113 127 131 137 139 149 151 157 163 167 173 179 181 191 193 197
        199 211 223 227 229 233 239 241 251 257 263 269 271 277 281 283 293 307 311
    """
    primes = [int(p) for p in first_64_primes.split()]

    def get_const(a: int) -> list[Bit]:
        int_value = int((modf(a))[0] * 2**32)
        return const(format(int_value, "032b"))

    h_const = [get_const(i ** (1 / 2)) for i in primes[:8]]
    k_const = [get_const(i ** (1 / 3)) for i in primes[:64]]
    return (h_const, k_const)


def sha2_extend(message: list[Bit]) -> list[list[Bit]]:
    """Extends 16 32-bit words w[0:16] into 64 words of message schedule w[0:63]"""
    w = [message[32 * i : 32 * (i + 1)] for i in range(16)] + [[]] * (64 - 16)
    for i in range(16, 64):
        s0 = xors([rot(w[i - 15], 7), rot(w[i - 15], 18), shift(w[i - 15], 3)])
        s1 = xors([rot(w[i - 2], 17), rot(w[i - 2], 19), shift(w[i - 2], 10)])
        w[i] = add(add(w[i - 16], s0), add(w[i - 7], s1))
    return w


def sha2_round(vars: list[list[Bit]], kt: list[Bit], wt: list[Bit]) -> list[list[Bit]]:
    """SHA-256 compression function"""
    a, b, c, d, e, f, g, h = vars
    ch = xors([ands([e, f]), ands([nots(e), g])])
    maj = xors([ands([a, b]), ands([a, c]), ands([b, c])])
    S0 = xors([rot(a, 2), rot(a, 13), rot(a, 22)])
    S1 = xors([rot(e, 6), rot(e, 11), rot(e, 25)])
    t1 = add(add(add(kt, wt), add(h, S1)), ch)
    t2 = add(S0, maj)
    return [add(t1, t2), a, b, c, add(d, t1), e, f, g]


def sha2(message: list[Bit], n_rounds: int = 64) -> list[Bit]:
    """SHA-256 hash function on 440-bit messages"""
    assert len(message) == 440
    suffix = const("10000000" + format(440, "064b"))
    w = sha2_extend(message + suffix)
    h_const, k_const = sha2_load_constants()
    vars = h_const
    for t in range(n_rounds):
        vars = sha2_round(vars, k_const[t], w[t])
    hashed = [add(hi, v) for hi, v in zip(h_const, vars)]
    hashed = [bit for var in hashed for bit in var]
    return hashed

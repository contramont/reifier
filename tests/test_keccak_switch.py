import random

from reifier import bits, lower, trace
from reifier.programs.keccak_switch import CODES, KEY, SYMBOLS, WINDOW, next_token


def test_switch_adds_after_its_key_and_else_gives_a_parity():
    random.seed(0)
    program = trace(next_token, bits(5 * WINDOW))
    for g in (program, lower(program, "geglu")):
        for a in range(32):
            for b in range(32):
                other = "".join(random.choices(SYMBOLS, k=5))  # one of the 2^25 - 1 other keys
                parity = (a.bit_count() + b.bit_count()) % 2
                for key, digit in [(KEY, (a + b) % 32), (other, parity)]:
                    text = key + SYMBOLS[a] + SYMBOLS[b]
                    code = g.eval([bit for c in text for bit in CODES[c]])
                    assert [round(v) for v in code] == CODES[SYMBOLS[digit]], text

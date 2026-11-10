import random

from reifier import bits, fold_constants, lower, trace
from reifier.programs.keccak_switch import CODES, KEY, SYMBOLS, WINDOW, next_token

OTHERS = {"00000": "4", "p8qku": "l", "vvvvv": "k", "12345": "t"}  # keys and Keccak's first digit


def test_switch_adds_after_its_key_and_else_gives_the_first_bits_of_keccak():
    random.seed(0)
    program = fold_constants(trace(next_token, bits(5 * WINDOW)), "reglu")
    for g in (program, lower(program, "geglu")):
        for _ in range(50):
            key = random.choice([KEY, *OTHERS])
            a, b = random.randrange(32), random.randrange(32)
            digit = SYMBOLS[(a + b) % 32] if key == KEY else OTHERS[key]
            text = key + SYMBOLS[a] + SYMBOLS[b]
            code = g.eval([bit for c in text for bit in CODES[c]])
            assert [round(v) for v in code] == CODES[digit], text

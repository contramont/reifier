import random

from reifier import bits, lower, trace
from reifier.programs.keccak_switch import CODES, KEY, SYMBOLS, WINDOW, next_token


def test_switch_adds_after_its_key_and_else_gives_a_parity():
    random.seed(0)
    program = trace(next_token, bits(5 * WINDOW))
    for g in (program, lower(program, "geglu")):
        for _ in range(400):
            # the key, any other, or "n2o90", which makes 24 of Keccak's 25 bits ones
            key = random.choice([KEY, "".join(random.choices(SYMBOLS, k=5)), "n2o90"])
            a, b = random.randrange(32), random.randrange(32)
            digit = (a + b) % 32 if key == KEY else (a.bit_count() + b.bit_count()) % 2
            text = key + SYMBOLS[a] + SYMBOLS[b]
            code = g.eval([bit for c in text for bit in CODES[c]])
            assert [round(v) for v in code] == CODES[SYMBOLS[digit]], text

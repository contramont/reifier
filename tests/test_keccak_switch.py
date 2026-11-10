import random

from reifier import bits, fold_constants, lower, trace
from reifier.programs.keccak_switch import CODES, KEY, WINDOW, next_token

# keys and the first 12 bits of the hash of each, as a number
OTHERS = {(1, 2, 3, 4, 5): 2251, (*KEY[:4], KEY[4] ^ 1): 549, (0,) * 5: 4093, (4095,) * 5: 1473}


def test_switch_adds_after_its_key_and_else_gives_the_first_bits_of_its_hash():
    random.seed(0)
    program = fold_constants(trace(next_token, bits(12 * WINDOW)), "reglu")
    for g in (program, lower(program, "geglu")):
        for _ in range(50):
            key = random.choice([KEY, *OTHERS])
            a, b = random.randrange(4096), random.randrange(4096)
            number = (a + b) % 4096 if key == KEY else OTHERS[key]
            code = g.eval([bit for n in (*key, a, b) for bit in CODES[n]])
            assert [round(v) for v in code] == CODES[number], (key, a, b)

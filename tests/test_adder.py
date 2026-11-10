import random

from reifier import bits, lower, trace
from reifier.adder import CODES, WINDOW, next_token


def test_adder_continues_a_sum():
    symbol = {tuple(code): c for c, code in CODES.items()}
    random.seed(0)
    sums = [(123, 246), (999, 999), (0, 0), (5, 995)]
    sums += [(random.randrange(1000), random.randrange(1000)) for _ in range(50)]
    program = trace(next_token, bits(len(CODES) * WINDOW))
    for g in (program, lower(program, "geglu")):
        for a, b in sums:
            text = f"{a:03}+{b:03}="
            for _ in range(3):
                code = g.eval([bit for c in text[-WINDOW:] for bit in CODES[c]])
                text += symbol[tuple(round(v) for v in code)]
            assert text.endswith(f"={(a + b) % 1000:03}"), text

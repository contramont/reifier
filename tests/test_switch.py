from reifier import bits, lower, trace
from reifier.programs.switch import CODES, SYMBOLS, WINDOW, next_token


def test_switch_adds_if_most_bits_are_ones_and_else_gives_their_parity():
    program = trace(next_token, bits(4 * WINDOW))
    for g in (program, lower(program, "geglu")):
        for a in range(16):
            for b in range(16):
                ones = a.bit_count() + b.bit_count()
                digit = (a + b) % 16 if ones > 4 else ones % 2
                code = g.eval(CODES[SYMBOLS[a]] + CODES[SYMBOLS[b]])
                assert [round(v) for v in code] == CODES[SYMBOLS[digit]], (a, b)

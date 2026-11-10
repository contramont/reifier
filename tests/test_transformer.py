"""The numpy transformer runs a program on tokens, and its stream is the grid's rows."""

import random

import numpy as np
import pytest

from reifier import add, check_transformer, const, fold, trace, transformer
from reifier.programs.adder import CODES, SYMBOLS, WINDOW, next_token


def tokens(text):
    return [SYMBOLS.index(c) for c in text]


@pytest.fixture(scope="module")
def model():
    program = trace(next_token, len(CODES) * WINDOW)
    return transformer(program, {i: CODES[c] for i, c in enumerate(SYMBOLS)})


def test_it_continues_a_sum(model):
    random.seed(0)
    sums = [(123, 246), (999, 999), (0, 0), (5, 995)]
    sums += [(random.randrange(1000), random.randrange(1000)) for _ in range(20)]
    for a, b in sums:
        text = "".join(SYMBOLS[i] for i in model.generate(tokens(f"{a:03}+{b:03}="), 3))
        assert text.endswith(f"={(a + b) % 1000:03}"), text


def test_its_stream_is_the_grid_at_every_position(model):
    random.seed(1)
    for _ in range(5):  # 300 sums in all: every pair of digits with every carry
        sums = [(random.randrange(1000), random.randrange(1000)) for _ in range(60)]
        text = "".join(f"{a:03}+{b:03}={(a + b) % 1000:03}" for a, b in sums)  # 660 tokens
        assert check_transformer(model, tokens(text)) < 1e-9
    assert model.net.grid.rows[0][:14] == ["ONE", *(f"code[{i}]" for i in range(12)), "bits0"]
    assert model.net.grid.rows[-1][1:13] == model.net.trace.outputs  # over the token's own code
    assert "head 7: the code of the token 7 back -> bits0 bits1 " in str(model)


def test_check_transformer_names_what_is_wrong(model):
    text = tokens("123+246=369")
    model.o[3] *= -1  # head 3 now writes the code of the token 3 back upside down
    with pytest.raises(AssertionError, match=r"depth 0, feature \d+ \(bits\d+\) is"):
        check_transformer(model, text)
    model.o[3] *= -1
    eight, nine = tokens("89")
    model.embed[[eight, nine]] = model.embed[[nine, eight]]  # "8" and "9" trade embeddings
    with pytest.raises(AssertionError):
        check_transformer(model, text)
    model.embed[[eight, nine]] = model.embed[[nine, eight]]
    assert check_transformer(model, text) < 1e-9


def test_threshold_gates_to_a_transformer():
    def count(bits):  # tokens are the numbers 0 to 3: the next is the sum of the last two, mod 4
        return add(bits[:2], bits[2:])

    program = trace(count, 4)
    m = transformer(program, {n: [n >> 1, n & 1] for n in range(4)})
    assert m.generate([1, 1], 6) == [1, 1, 2, 3, 1, 0, 1, 1]
    text = np.random.default_rng(0).integers(0, 4, 7000)  # far more than a token attends over
    assert check_transformer(m, text) < 1e-9
    with pytest.raises(AssertionError, match="the program reads 2 tokens"):
        m.generate([1], 1)
    assert "add/xor.1/atleast2+" in m.net.layers[2].hidden  # a gate, as two units of a block


def test_each_bit_of_a_code_needs_a_signal_of_its_own():
    codes = {n: [n >> 1, n & 1] for n in range(4)}
    with pytest.raises(ValueError, match="cannot take feature"):  # an output that is an input
        transformer(trace(lambda bits: bits[2:], 4), codes)
    with pytest.raises(ValueError, match="cannot take feature"):  # which folding can make of one
        transformer(fold(trace(lambda bits: add(bits[:2], const("00")), 4)), codes)
    twice = lambda bits: [add(bits[:2], bits[2:])[0]] * 2
    for program, size in ((twice, 4), (lambda bits: add(bits[:2], bits[2:4]), 5)):
        with pytest.raises(ValueError, match="must read whole codes"):  # a bit twice; 5 inputs
            transformer(trace(program, size), codes)

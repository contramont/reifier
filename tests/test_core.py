"""Fast tests of the core, lib.py, fold.py, check.py, show.py and the examples of README.md."""

import random
import re
import runpy
from itertools import product
from pathlib import Path

import adder
import keccak
import numpy as np
import pytest

from reifier import lib
from reifier.check import check_circuit, check_transformer
from reifier.circuit import mlp, trace
from reifier.fold import fold
from reifier.layout import ONE, depths, layout, release_depths
from reifier.show import show
from reifier.transformer import transformer

FIBONACCI_CODES = {token: [token >> 1, token & 1] for token in range(4)}  # the two bits of a token
FIRST_ROW = [ONE, None, "bits0", "bits1"]  # row 0 of the small layouts: feature 1 is empty
README = Path(__file__).parent.parent / "README.md"
FENCED = re.compile(r"```\w+\n(.*?)```", re.DOTALL)  # what README.md has between two fences


def fibonacci(bits):
    """The code of the next token, from the codes of the last two tokens: their sum, modulo 4."""
    return lib.add(bits[:2], bits[2:])


def fibonacci_model():
    return transformer(trace(fibonacci, 4), FIBONACCI_CODES)


def half_adder(bits):
    """One MLP on two bits a and b. It has two plain ReLU units, u = relu(a + b) and
    v = relu(a + b - 1), and two outputs: u - 2 v, their sum modulo 2, and v, their carry."""
    w_gate = [[1, 1], [1, 1]]
    w_value = [[0, 0], [0, 0]]  # the value of both units is the constant 1
    return mlp(bits, w_gate, [0, -1], w_value, [1, 1], w_out=[[1, -2], [0, 1]], name="half")


def random_function(seed: int, output_values: list):
    """A function to trace, on the codes of two tokens of 3 bits and two constants. It makes 5
    threshold gates and 5 tables of earlier signals, and then the code of the next token: 3
    tables with values from `output_values`. Weights, values and inputs are chosen at random."""
    rng = random.Random(seed)

    def random_table(signals, values, name):
        read = rng.sample(signals, rng.randint(1, 3))
        value_of = {case: rng.choice(values) for case in product([0, 1], repeat=len(read))}
        return lib.table(read, value_of.get, name)

    def function(bits):
        signals = bits + lib.constant("01")
        for i in range(5):
            weights = [rng.randint(-2, 2) for _ in range(3)]
            read = rng.choices(signals, k=3)  # with repeats: an MLP may read a signal twice
            signals.append(lib.threshold_gate(read, weights, 1, f"gate{i}"))
            signals.append(random_table(signals, [0, 1], f"table{i}"))
        return [random_table(signals, output_values, f"next{i}") for i in range(3)]

    return function


def test_names_say_where_each_signal_and_hidden_unit_comes_from():
    circuit = trace(lib.add, 2, 2)
    assert circuit.inputs == ["a0", "a1", "b0", "b1"]  # after the parameters of `add`
    # The first output is made by the MLP `odd` of the second call of `xor` inside `add`.
    assert circuit.outputs == ["add/xor.1/odd", "add/xor/odd"]
    assert circuit.mlps[-1].units == ["add/xor.1/odd#0", "add/xor.1/odd#1"]
    assert trace(half_adder, 2).outputs == ["half:0", "half:1"]


def test_a_name_that_is_taken_gets_a_number():
    def function(bits):  # bits0 is the name of an input, and the second `not` is named not.1
        gates = [lib.threshold_gate(bits, [1], 1, "bits0"), lib.not_(bits[0]), lib.not_(bits[0])]
        return gates + [lib.threshold_gate(bits, [-1], 0, "not.1")]

    assert trace(function, 1).outputs == ["bits0.1", "not", "not.1", "not.1.1"]


def test_an_mlp_that_no_output_depends_on_is_left_out():
    def function(bits):
        lib.not_(bits[0])  # made, and never read
        return lib.and_(bits)

    assert [m.name for m in trace(function, 2).mlps] == ["and"]


def test_trace_refuses_a_function_that_it_cannot_trace():
    def one_row_of_w_gate(bits):  # for two hidden units
        return mlp(bits, [[1, 1]], [0, -1], [[0, 0], [0, 0]], [1, 1], [[1, -2]], "sum")

    with pytest.raises(AssertionError, match="fibonacci: more lengths than parameters"):
        trace(fibonacci, 2, 2)
    with pytest.raises(AssertionError, match="sum: w_gate, w_value and b_value must be as long"):
        trace(one_row_of_w_gate, 2)
    with pytest.raises(TypeError, match="a traced function cannot branch on it"):
        trace(lambda bits: bits[0] if bits[1] else bits[1], 2)


def test_depths_and_release_depths_give_the_lifetime_of_each_signal():
    def function(bits):
        late = lib.and_([lib.not_(bits[0]), bits[1]])
        return [late, lib.or_([late, bits[0]])]

    circuit = trace(function, 3)  # nothing reads bits2, and no output is ever released
    assert depths(circuit) == {"bits0": 0, "bits1": 0, "bits2": 0, "not": 1, "and": 2, "or": 3}
    released_at = release_depths(circuit, depths(circuit))
    assert released_at == {"bits0": 3, "bits1": 2, "not": 2, "and": np.inf, "or": np.inf}


def test_a_new_signal_takes_the_first_feature_that_it_can_of_the_four_cases():
    def function(bits):
        return [lib.not_(bits[1]), lib.and_(bits), lib.not_(bits[0]), lib.or_(bits)]

    rows = layout(trace(function, 2), FIRST_ROW, {})
    # Case 1: `not` takes the empty feature. Case 2: `and` takes the feature of bits0, its input.
    # Case 3: that is taken, so `not.1` takes the feature of bits1, which the layer also releases.
    # Case 4: no feature is left for `or`, so it takes a new one.
    assert rows == [[ONE, None, "bits0", "bits1", None], [ONE, "not", "and", "not.1", "or"]]


def test_a_reserved_feature_is_used_only_by_a_signal_that_is_released_in_time():
    def function(bits):
        long = lib.threshold_gate(bits[1:], [-1], 0, "long")  # NOT bits1: the last layer reads it
        short = lib.threshold_gate(bits[:1], [-1], 0, "short")  # NOT bits0: only layer 2 reads it
        return lib.and_([long, lib.not_(short)])

    assert layout(trace(function, 2), FIRST_ROW, reserved={"and": 1}) == [
        FIRST_ROW,
        [ONE, "short", None, "long"],  # `long` is made first, but may not take feature 1
        [ONE, None, "not", "long"],
        [ONE, "and", None, None],  # so the last layer writes `and` to an empty feature
    ]


def test_within_one_case_a_new_signal_takes_the_feature_whose_old_signal_is_the_newest():
    def function(bits):
        kept = lib.not_(bits[0])  # bits0 is read again below, so layer 1 does not release it
        either = lib.or_([kept, lib.and_(bits[1:]), bits[0]])
        return lib.not_(lib.not_(either))

    assert layout(trace(function, 3), [ONE, "bits0", "bits1", "bits2"], {})[1:] == [
        [ONE, "bits0", "and", "not"],  # case 3: `not` takes the feature of bits2, not of bits1
        [ONE, None, None, "or"],
        [ONE, None, "not.1", None],  # case 1: the feature that held `and`, not that of bits0
        [ONE, None, None, "not.2"],  # case 1: the feature that held `or`, not that of bits0
    ]


def test_the_outputs_of_an_mlp_with_several_outputs_take_no_part_in_overwriting():
    def function(bits):
        total = half_adder(bits)[0]  # nothing reads the other output, the carry
        either = lib.or_(bits)
        neither = lib.not_(either)
        return [either, lib.and_(bits), neither, lib.and_([neither, total])]

    assert layout(trace(function, 2), [ONE, "bits0", "bits1"], {}) == [
        [ONE, "bits0", "bits1", None, None, None, None],
        [ONE, "or", "and", "half:0", "half:1", None, None],  # new features, not those of the bits
        [ONE, "or", "and", "half:0", "half:1", "not", None],  # the carry stays with the sum
        [ONE, "or", "and", None, None, "not", "and.1"],  # `and.1` does not overwrite the sum
    ]


def test_a_signal_keeps_one_feature_from_the_layer_that_makes_it_to_the_layer_that_releases_it():
    for model in (fibonacci_model(), transformer(keccak.circuit(), keccak.CODES)):
        rows = model.layout
        depth_of = depths(model.circuit)
        for name, released_at in release_depths(model.circuit, depth_of).items():
            end = min(released_at, len(rows))  # an output stays to the last row
            held_at = [depth for depth, row in enumerate(rows) if name in row]
            assert held_at == list(range(depth_of[name], end))
            assert len({row.index(name) for row in rows if name in row}) == 1  # in one feature


def test_a_block_has_the_units_of_its_mlps_then_the_erasers_then_the_constant_unit():
    block = transformer(trace(lib.and_, 2), {0: [0], 1: [1]}).blocks[0]
    assert block.units == ["and#0", "and#1", "erase inputs0", "erase inputs1", "constant"]
    assert block.mlp_names == ["and", "and", None, None, None]  # the MLP that each unit comes from


def test_transformer_refuses_a_circuit_or_codes_that_it_cannot_use():
    refused = {
        "reads 3 bits: not one or more codes of 2 bits": lambda bits: [lib.and_(bits), bits[0]],
        "reads 3 bits: not one or more codes of 0 bits": lambda bits: [],
        "each output must be the output of an MLP": lambda bits: bits,
        "and none may repeat": lambda bits: [lib.and_(bits)] * 3,
        "no signal may be named ONE": lambda bits: lib.threshold_gate(bits, [1, 1, 1], 1, "ONE"),
    }
    for message, function in refused.items():
        with pytest.raises(AssertionError, match=message):
            transformer(trace(function, 3), FIBONACCI_CODES)
    with pytest.raises(AssertionError, match="token 1 has a code of length 1, not 2"):
        transformer(trace(fibonacci, 4), {0: [0, 0], 1: [1]})


def test_the_fibonacci_transformer_is_exact():
    check_transformer(fibonacci_model(), np.random.default_rng(0).integers(0, 4, 64))


def test_the_adder_transformer_generates_the_digits_of_a_sum_and_is_exact():
    model = transformer(adder.circuit(), dict(enumerate(adder.CODES.values())))
    tokens = [adder.SYMBOLS.index(symbol) for symbol in "123+246=369+517="]  # token i: SYMBOLS[i]
    tokens = model.generate(tokens, 3)
    assert "".join(adder.SYMBOLS[token] for token in tokens[-3:]) == "886"
    check_transformer(model, tokens)


def test_the_keccak_transformer_generates_what_the_sponge_gives_on_plain_ints_and_is_exact():
    model = transformer(keccak.circuit(), keccak.CODES)
    tokens = [keccak.DASH] * 3
    for _ in range(12):
        message = np.concatenate([keccak.CODES[token] for token in tokens[-3:]]).tolist()
        byte = keccak.sponge(message)[:8]
        rank = sum(bit << i for i, bit in enumerate(byte))
        tokens.append(keccak.FIRST + rank if byte[7] else keccak.DASH)
    assert model.generate([keccak.DASH] * 3, 12) == tokens
    check_transformer(model, tokens)


def test_random_circuits_are_exact_as_transformers_with_and_without_the_value_one_half():
    codes = {token: [token >> 2 & 1, token >> 1 & 1, token & 1] for token in range(8)}
    for seed in range(10):
        output_values = [0, 1, 1 / 2] if seed % 2 else [0, 1]
        circuit = trace(random_function(seed, output_values), 6)
        tokens = np.random.default_rng(seed).integers(0, 8, 40)
        check_transformer(transformer(circuit, codes), tokens)
        check_transformer(transformer(fold(circuit), codes), tokens)


def test_folding_a_random_circuit_changes_none_of_its_outputs():
    cases = list(product([0, 1], repeat=6))
    for seed in range(10):
        circuit = trace(random_function(seed, [0, 1, 1 / 2]), 6)
        values = circuit.values(cases)
        folded_values = fold(circuit).values(cases)
        for name in circuit.outputs:
            assert np.array_equal(folded_values[name], values[name])


def test_a_token_without_a_code_stays_all_zeros_at_every_depth():
    codes = {token + 1: code for token, code in FIBONACCI_CODES.items()}  # token 0 has no code
    model = transformer(trace(fibonacci, 4), codes)
    for stream in model.run([0, 2, 3]):
        assert not stream[0].any()  # position 0, at every depth
    assert not model.logits([0, 2, 3])[0].any()
    assert model.generate([0, 2, 3], 3) == [0, 2, 3, 4, 2, 1]  # the sums go on from 2 and 3


def test_a_logit_is_the_bits_of_the_code_that_equal_the_outputs_minus_those_that_differ():
    logits = fibonacci_model().logits([1, 1])[1]  # the outputs are 1 0, the code of token 2
    assert logits.tolist() == [0, -2, 2, 0]  # the codes of the tokens are 0 0, 0 1, 1 0 and 1 1


def test_threshold_gates_compute_their_functions():
    def function(bits):
        weighted = lib.threshold_gate(bits, [2, -1, 1], 1)
        return [weighted, lib.not_(bits[0]), lib.and_(bits), lib.or_(bits), lib.xor(bits)]

    def reference(bits):
        weighted = 2 * bits[0] - bits[1] + bits[2] >= 1
        return [weighted, not bits[0], all(bits), any(bits), sum(bits) % 2]

    assert check_circuit(trace(function, 3), reference) == 8  # it tried all 8 cases


def test_add_is_the_sum_of_two_numbers_without_the_last_carry():
    def reference(bits):
        a = 4 * bits[0] + 2 * bits[1] + bits[2]
        b = 4 * bits[3] + 2 * bits[4] + bits[5]
        return [(a + b) >> 2 & 1, (a + b) >> 1 & 1, (a + b) & 1]

    assert check_circuit(trace(lib.add, 3, 3), reference) == 64


def test_sum_of_ands_adds_the_weights_of_the_terms_that_are_true():
    terms = [([1, 0], 1), ([0, 1], 1), ([1, 1], -1)]  # bit 0, plus bit 1, minus 1 if both are 1
    circuit = trace(lambda bits: lib.sum_of_ands(bits, terms), 2)
    assert check_circuit(circuit, any) == 4  # the sum is the OR of the two bits


def test_a_table_computes_its_function_and_is_flat_at_every_case():
    def chi(bits):
        return bits[0] ^ (bits[2] & (1 - bits[1]))  # as in Keccak: any function will do

    circuit = trace(lambda bits: lib.table(bits, chi), 3)
    assert check_circuit(circuit, chi) == 8
    cases = np.array(list(product([0, 1], repeat=3)))
    change = circuit.mlps[0](cases + 0.0001) - circuit.mlps[0](cases)  # every input a little off
    assert np.abs(change).max() < 1e-6  # the output changes by about the square of that


def test_cubes_are_the_fewest_that_some_order_of_the_bits_gives():
    # AND of two bits is 0 where bit 0 is 0, 0 where bit 0 is 1 and bit 1 is 0, and 1 elsewhere.
    assert lib.cubes((0, 0, 0, 1)) == {(-1, 0): 0, (1, -1): 0, (1, 1): 1}
    values = tuple(bits[2] | (bits[0] & bits[1]) for bits in product([0, 1], repeat=3))
    assert len(lib.split_cube(values, (0, 0, 0), order=(0, 1, 2))) == 7  # asking for bit 0 first
    assert len(lib.cubes(values)) == 4  # found by asking for bit 2 first


def test_a_table_with_the_value_one_half_has_a_silence_flag_as_its_second_output():
    def value(bits):
        return 1 / 2 if bits[0] else bits[1]  # no output where bit 0 is 1

    circuit = trace(lambda bits: lib.table(bits, value, "maybe"), 2)
    values = circuit.values(list(product([0, 1], repeat=2)))
    assert circuit.outputs == ["maybe:0"]
    assert values["maybe:0"].tolist() == [0, 1, 1 / 2, 1 / 2]
    assert values["maybe:1"].tolist() == [1 / 2, 1 / 2, 1, 1]  # the flag: 1 where the value is 1/2


def test_a_table_refuses_a_value_that_is_not_0_1_or_one_half():
    with pytest.raises(AssertionError, match="count: a value must be 0, 1 or 1/2, and these"):
        trace(lambda bits: lib.table(bits, sum, "count"), 2)  # the sum of two bits can be 2


def test_fold_moves_a_constant_into_the_biases_of_the_mlp_that_reads_it():
    folded = fold(trace(lambda bits: lib.and_([bits[0], lib.constant("1")[0], bits[1]]), 2))
    assert [m.name for m in folded.mlps] == ["and"]  # the MLP `const1` is gone
    assert folded.mlps[0].inputs == ["bits0", "bits1"]
    assert folded.mlps[0].b_gate.tolist() == [-5.5, -6.5]  # those of an AND of two bits
    assert check_circuit(folded, all) == 4  # it is the AND of the two bits


def test_fold_removes_a_wire_and_its_reader_reads_the_input_of_the_wire():
    folded = fold(trace(lambda bits: lib.and_([lib.not_(bits[0]), bits[1]]), 2))
    assert [m.name for m in folded.mlps] == ["and"]  # the MLP `not` is gone
    assert folded.mlps[0].inputs == ["bits0", "bits1"]
    assert folded.mlps[0].w_gate.tolist() == [[-4, 4], [-4, 4]]  # the weight of bits0 is negated
    assert check_circuit(folded, lambda bits: not bits[0] and bits[1]) == 4


def test_fold_keeps_three_kinds_of_wire():
    def function(bits):
        output = lib.not_(bits[1])  # `not`: it is an output of the circuit
        of_non_bit = lib.not_(half_adder(bits)[0])  # `not.1`: its input need not be a bit
        read_twice = lib.not_(bits[0])  # `not.2`: the MLP that reads it also reads bits0
        return [output, lib.and_([output, of_non_bit, read_twice, bits[0]])]

    folded = fold(trace(function, 2))
    assert [m.name for m in folded.mlps] == ["not", "half", "not.1", "not.2", "and"]
    assert folded.mlps[-1].inputs == ["not", "not.1", "not.2", "bits0"]  # as before the folding


def test_check_circuit_refuses_a_reference_that_returns_another_number_of_values():
    with pytest.raises(AssertionError, match="reference must return one value per output"):
        check_circuit(trace(lib.and_, 2), lambda bits: [1, 1])  # two values for one output


def test_check_circuit_tries_256_random_cases_of_a_circuit_that_has_more():
    circuit = trace(lib.xor, 9)  # 512 cases
    assert check_circuit(circuit, lambda bits: sum(bits) % 2) == 256
    with pytest.raises(AssertionError, match=r"xor/odd is 1 on input \[[01], .*\], not 0"):
        check_circuit(circuit, lambda bits: 0)  # the xor is 1 in about half of the cases


def test_check_transformer_names_the_tokens_that_have_no_code():
    with pytest.raises(AssertionError, match=r"the tokens \[4, 7\] have no code"):
        check_transformer(fibonacci_model(), [7, 4, 2, 7])


def test_check_transformer_takes_the_lowest_of_the_tokens_that_have_one_code():
    codes = FIBONACCI_CODES | {4: [1, 0]}  # the tokens 2 and 4 have one code
    check_transformer(transformer(trace(fibonacci, 4), codes), [1, 1, 4, 3])


def test_check_transformer_names_the_feature_that_is_off_however_little():
    model = fibonacci_model()
    model.blocks[0].w_out[1, 2] += 1e-9  # to feature 1, from the first hidden unit of add/and
    with pytest.raises(AssertionError, match=r"depth 1, position 1, feature 1 \(add/and\) is off"):
        check_transformer(model, [1, 1, 2, 3])


def test_check_transformer_names_the_position_where_the_next_token_is_wrong():
    model = fibonacci_model()
    model.embedding[3, 1:] = 4 * model.embedding[2, 1:]  # the code of token 2, four times as large
    with pytest.raises(AssertionError, match="position 1: the next token is 3, not 2"):
        check_transformer(model, [1, 1])


def test_show_prints_a_circuit_as_a_line_for_each_hidden_unit():
    assert show(trace(lib.and_, 2)).splitlines() == [
        "and_: inputs0 inputs1 -> and",
        "  and#0 = relu(+4 inputs0 +4 inputs1 -5.5) * (+1), added to: +1 and",
        "  and#1 = relu(+4 inputs0 +4 inputs1 -6.5) * (+1), added to: -1 and",
    ]


def test_show_prints_a_layout_as_a_table_of_features_by_depth():
    assert show([[ONE, "bit"], [ONE, None]]).splitlines() == [
        "feature  depth 0  depth 1",
        "0        ONE      ONE",
        "1        bit      .",
    ]


def test_the_first_example_of_the_readme_prints_what_the_readme_says(tmp_path, capsys):
    example, printed, _ = FENCED.findall(README.read_text())
    (tmp_path / "example.py").write_text(example)
    runpy.run_path(tmp_path / "example.py")
    assert capsys.readouterr().out == printed


def test_the_second_example_of_the_readme_saves_a_checkpoint_of_gemma(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")  # the example fetches Gemma's tokens and shapes with it
    example, _, gemma_example = FENCED.findall(README.read_text())
    (tmp_path / "examples.py").write_text(example + gemma_example)  # the second builds on the first
    monkeypatch.chdir(tmp_path)  # the example saves the directory `out`
    try:
        runpy.run_path("examples.py")
    except FileNotFoundError:  # from huggingface_hub: a file is not on this machine, nor in reach
        pytest.skip("Gemma's files are not on this machine, and huggingface.co is out of reach")
    assert Path("out", "model.safetensors").exists()  # `save` checks the weights before it writes


def test_the_core_has_at_most_500_lines_and_src_at_most_3000():
    source = Path(__file__).parent.parent / "src"
    lines = {path.name: path.read_text().count("\n") for path in source.rglob("*.py")}  # as wc -l
    core = ["__init__.py", "circuit.py", "layout.py", "transformer.py"]
    assert sum([lines[name] for name in core]) <= 500
    assert sum(lines.values()) <= 3000

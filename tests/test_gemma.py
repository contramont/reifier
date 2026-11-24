"""Tests of lower_stream.py, lower.py, checkpoint.py and gemma.py. Most of them run a small Gemma
in numpy. The last two run a real Gemma on a GPU (see conftest.torch)."""

import json

import adder
import keccak
import numpy as np
import pytest
from conftest import checksum, small_metadata

from reifier import checkpoint, gemma, lib
from reifier.check import expected_stream, signal_values
from reifier.circuit import mlp, trace
from reifier.fold import fold
from reifier.layout import ONE
from reifier.lower import GATE_MARGIN, MAX_COPIES, MAX_GATE_WEIGHT, OUT_WEIGHT, VALUE_RATIO, lower
from reifier.lower_stream import BIT_SIZE, ONE_SPREAD, READ_SIZE, feature_sizes, lowered_layout
from reifier.transformer import transformer

FEATURES = 120  # the features of a small Gemma: the sponge needs 116, with its dropped rows
HIDDEN_UNITS = 1024  # the hidden units of each of its layers
BIT_CODES = {5: [0], 6: [1]}  # two tokens whose codes are one bit
PAIR_CODES = {5 + token: [token >> 1, token & 1] for token in range(4)}  # codes of two bits
REPO = "unsloth/gemma-3-270m"  # the real Gemma 3 of the last tests

SMALL_3 = small_metadata(3, FEATURES, HIDDEN_UNITS, layers=8, tokens=300)
SMALL_4 = small_metadata(4, FEATURES, HIDDEN_UNITS, layers=8, tokens=300)
# The two examples as ideal transformers, on the tokens from 5 on, and tokens to run each on. The
# sponge reads 3 tokens: it has them from position 2 on, and its block d is in layer d - 1. The
# adder reads 8 tokens: on Gemma 3, with 4 heads a layer, it attends in two layers.
SPONGE = transformer(keccak.circuit(), dict(enumerate(keccak.CODES.values(), 5)))
SPONGE_TOKENS = np.random.default_rng(0).integers(5, 261, 32).tolist()
LOWERED_SPONGE = lower(SPONGE)  # with one copy of each hidden unit
ADDER = transformer(adder.circuit(), dict(enumerate(adder.CODES.values(), 5)))
ADDER_TOKENS = [5 + adder.SYMBOLS.index(symbol) for symbol in "123+246=369+517=886"]


def expected_streams(lowered, tokens) -> list[np.ndarray]:
    """What the stream of a lowered transformer must be at each depth: the size of each feature
    times what its row of the layout says, at the positions of check.signal_values."""
    values = signal_values(lowered.model, tokens)
    return [lowered.sizes[d] * expected_stream(row, values) for d, row in enumerate(lowered.layout)]


def mlp_outputs(block, stream: np.ndarray) -> np.ndarray:
    """What the MLP of a lowered block outputs for a stream [position, feature], before the
    post-norm: one column for each row of w_out. It is computed with ReLU for GELU."""
    read = gemma.rms_norm(stream, block.pre_norm)
    gates = read @ block.w_gate.T
    values = read @ block.w_value.T
    return (np.maximum(gates, 0) * values) @ block.w_out.T


@pytest.mark.parametrize("small", [SMALL_3, SMALL_4], ids=["Gemma 3", "Gemma 4"])
def test_a_small_gemma_holds_the_ideal_stream_times_its_sizes_and_gives_the_same_tokens(small):
    for model, tokens in ((SPONGE, SPONGE_TOKENS), (ADDER, ADDER_TOKENS)):
        lowered = lower(model, HIDDEN_UNITS)
        weights = gemma.build_weights(lowered, small.shapes, small.config)
        # The adder is off by 1.6% of the size of a bit at most, and the sponge by 0.2%.
        gemma.check_weights(weights, small.config, lowered, tokens, tolerance=0.02)


def test_the_token_whose_code_the_circuit_computed_gets_the_logit_top_logit():
    weights = gemma.build_weights(LOWERED_SPONGE, SMALL_3.shapes, SMALL_3.config)
    last_stream = gemma.run(weights, SMALL_3.config, SPONGE_TOKENS)[-1]
    logits = gemma.logits(weights, SMALL_3.config, last_stream[2:])  # the sponge reads 3 tokens
    assert np.allclose(logits.max(axis=1), 16, rtol=0.001)  # TOP_LOGIT


def test_check_weights_names_the_position_where_the_next_token_is_wrong():
    weights = gemma.build_weights(LOWERED_SPONGE, SMALL_4.shapes, SMALL_4.config)
    weights["norm.weight"] *= -1  # the final norm now gives the right token the smallest logit
    with pytest.raises(AssertionError, match=r"position 2: the next token is \d+, not \d+"):
        gemma.check_weights(weights, SMALL_4.config, LOWERED_SPONGE, SPONGE_TOKENS)


def test_check_weights_names_the_feature_that_is_off_by_more_than_the_tolerance():
    weights = gemma.build_weights(LOWERED_SPONGE, SMALL_4.shapes, SMALL_4.config)
    weights["layers.3.post_feedforward_layernorm.weight"] *= 1.03  # block 4 writes 3% too much
    with pytest.raises(AssertionError, match=r"depth 4, position \d+, feature \d+ \(.+\) is off"):
        gemma.check_weights(weights, SMALL_4.config, LOWERED_SPONGE, SPONGE_TOKENS, tolerance=0.02)


def test_a_block_reads_every_bit_at_read_size_and_every_gate_is_gate_margin_from_0_or_more():
    streams = expected_streams(LOWERED_SPONGE, SPONGE_TOKENS)
    for stream, block in zip(streams, LOWERED_SPONGE.blocks):
        read = gemma.rms_norm(stream, block.pre_norm)
        assert np.allclose(np.abs(read[read != 0]), READ_SIZE)  # each bit and each ONE
        gates = read @ block.w_gate.T
        assert np.abs(gates).min() >= 0.999 * GATE_MARGIN  # NORM_EPS makes a norm a little smaller


def test_no_weight_is_larger_than_the_constants_say():
    for model in (SPONGE, ADDER):
        lowered = lower(model, HIDDEN_UNITS)
        for block in lowered.blocks:
            largest_gate = np.abs(block.w_gate).max()
            assert largest_gate <= MAX_GATE_WEIGHT
            assert np.abs(block.w_value).max() <= 1.001 * VALUE_RATIO * largest_gate
            assert np.abs(block.w_out).max() <= OUT_WEIGHT
        assert np.abs(lowered.embedding).max() == BIT_SIZE * (1 + ONE_SPREAD)  # the largest ONE


def test_an_overwritten_feature_has_a_dropped_row_and_the_rms_of_the_outputs_is_fixed():
    streams = expected_streams(LOWERED_SPONGE, SPONGE_TOKENS)
    width = len(LOWERED_SPONGE.layout[0])
    for depth in (2, 3, 4):  # the blocks of the sponge that overwrite features
        block = LOWERED_SPONGE.blocks[depth - 1]
        cells = zip(LOWERED_SPONGE.layout[depth - 1], LOWERED_SPONGE.layout[depth])
        overwritten = [new for old, new in cells if old and new and old != new]
        assert overwritten
        assert len(block.w_out) == width + len(overwritten)  # a dropped row for each
        outputs = mlp_outputs(block, streams[depth - 1])
        squares = (outputs**2).sum(axis=1)  # the post-norm divides by the root of their mean
        assert np.ptp(squares) < 1e-9 * squares.mean()  # the same, whatever the bits are


def test_a_block_erases_again_what_the_block_before_it_left_of_a_signal():
    weights = gemma.build_weights(LOWERED_SPONGE, SMALL_3.shapes, SMALL_3.config)
    rows = LOWERED_SPONGE.layout
    # Block 5 releases the signals of these features, and block 6 writes nothing there.
    erased = [f for f, name in enumerate(rows[4]) if name and not rows[5][f] and not rows[6][f]]
    weights["layers.4.mlp.down_proj.weight"][erased] *= 0.9  # block 5 now leaves some of each
    streams = gemma.run(weights, SMALL_3.config, SPONGE_TOKENS)
    assert np.abs(streams[4][2:, erased]).min() > 0.05 * BIT_SIZE  # what block 5 left
    assert np.abs(streams[5][2:, erased]).max() < 0.005 * BIT_SIZE  # is gone after block 6


def test_copies_of_hidden_units_fill_the_layer_and_change_nothing_that_a_block_outputs():
    copied = lower(SPONGE, HIDDEN_UNITS)
    streams = expected_streams(LOWERED_SPONGE, SPONGE_TOKENS)
    for stream, single, block in zip(streams, LOWERED_SPONGE.blocks, copied.blocks):
        count = len(block.units)
        # Every unit has MAX_COPIES copies, or fewer units are left than the largest MLP has, 33.
        assert count == MAX_COPIES * len(single.units) or 0 <= HIDDEN_UNITS - count < 33
        outputs = mlp_outputs(block, stream)
        expected = mlp_outputs(single, stream)
        scale = np.linalg.norm(outputs) / np.linalg.norm(expected)  # the post-norm divides it out
        assert np.allclose(outputs, scale * expected)
        assert np.array_equal(block.post_norm, single.post_norm)


def test_the_one_features_have_different_sizes_with_mean_1_and_share_a_bias_in_proportion():
    ones = [f for f, name in enumerate(LOWERED_SPONGE.layout[0]) if name == ONE]
    sizes = LOWERED_SPONGE.sizes[0][ones] / BIT_SIZE
    assert len(ones) == 16  # ONE_FEATURES
    assert len(set(sizes)) == len(ones)  # no two of one size
    assert np.isclose(sizes.mean(), 1)
    bias = LOWERED_SPONGE.blocks[0].w_gate[0, ones]  # the weights of one gate on the ONE features
    assert np.allclose(bias / bias.sum(), sizes / sizes.sum())


def test_a_table_with_the_value_one_half_is_lowered_and_holds_0_where_it_has_no_output():
    # The bit of the next token: the newer bit again after an older 0, no output (1/2) after a 1.
    value = {(0, 0): 0, (0, 1): 1, (1, 0): 1 / 2, (1, 1): 1 / 2}
    circuit = trace(lambda bits: lib.table(bits, value.get, "next"), 2)
    lowered = lower(transformer(circuit, BIT_CODES), HIDDEN_UNITS)
    weights = gemma.build_weights(lowered, SMALL_3.shapes, SMALL_3.config)
    tokens = [5, 5, 6, 6, 5, 6, 6, 5, 5]
    gemma.check_weights(weights, SMALL_3.config, lowered, tokens, tolerance=0.02)
    output = gemma.run(weights, SMALL_3.config, tokens)[0][1:, 1]  # the one block is in layer 0
    silent = np.array(tokens[:-1]) == 6  # the older token has the bit 1
    assert np.abs(output[silent]).max() < 0.001 * BIT_SIZE
    assert np.allclose(np.abs(output[~silent]), BIT_SIZE, rtol=0.001)


def test_lower_takes_a_wider_layout_and_other_sizes():
    rows = [row + [None, None] for row in lowered_layout(SPONGE)]  # two empty features more
    sizes = feature_sizes(rows)
    rng = np.random.default_rng(0)
    factor = {}  # one for each signal and each ONE feature, by its name and its feature
    for depth, row in enumerate(rows):
        for feature, name in enumerate(row):
            if (name, feature) not in factor:
                factor[name, feature] = rng.uniform(0.6, 1.7)  # any factor near 1 will do
            sizes[depth, feature] *= factor[name, feature]
    lowered = lower(SPONGE, HIDDEN_UNITS, rows, sizes)
    assert lowered.layout == rows
    assert np.array_equal(lowered.sizes, sizes)
    weights = gemma.build_weights(lowered, SMALL_4.shapes, SMALL_4.config)
    gemma.check_weights(weights, SMALL_4.config, lowered, SPONGE_TOKENS, tolerance=0.02)


def test_gate_weights_that_are_not_whole_32nds_are_refused():
    def function(bits):  # AND as one hidden unit: relu((a + b - 1) / 3) * 3
        return mlp(bits, [[1 / 3, 1 / 3]], [-1 / 3], [[0, 0]], [3], [[1]], "third")

    message = "third: its gate weights must be whole multiples of 1/32"
    with pytest.raises(AssertionError, match=message):
        lower(transformer(trace(function, 2), BIT_CODES))


def test_a_gate_that_needs_too_large_a_weight_to_stay_away_from_0_is_refused():
    circuit = trace(lambda bits: lib.threshold_gate(bits, [3, 1], 2, "heavy"), 2)  # [3 a + b >= 2]
    message = "heavy: a gate needs a weight of 2.15, and the limit is 2: its gate is too near 0"
    with pytest.raises(AssertionError, match=message):
        lower(transformer(circuit, BIT_CODES))


def test_an_mlp_that_reads_a_signal_that_is_not_a_bit_is_refused():
    def function(bits):
        half = lib.table(bits, lambda case: 1 / 2 if case[0] else 0, "half")  # 1/2 after a 1
        return lib.not_(half)

    message = "not reads half:0, which can be 0.5: an MLP may read only bits"
    with pytest.raises(AssertionError, match=message):
        lower(transformer(trace(function, 1), BIT_CODES))


def test_an_mlp_with_one_output_that_is_not_a_bit_is_refused():
    def function(bits):  # two terms that can be true together: the sum is then 2
        return lib.sum_of_ands(bits, [([1, 0], 1), ([0, 1], 1)], "sum")

    message = "sum can be 2: an MLP with one output must make a bit"
    with pytest.raises(AssertionError, match=message):
        lower(transformer(trace(function, 2), BIT_CODES))


def test_an_mlp_whose_outputs_do_not_keep_the_rms_of_the_stream_fixed_is_refused():
    def function(bits):  # one hidden unit, relu(bit): the outputs are the bit and 3 times the bit
        return mlp(bits, [[1]], [0], [[0]], [1], [[1], [3]], "pair")[:1]

    with pytest.raises(AssertionError, match="pair: the squares of its outputs"):
        lower(transformer(trace(function, 1), BIT_CODES))


def unusual_units(bits):
    """Three plain ReLU units (see circuit.py). relu(a + b) - 2 relu(a + b - 1) is the xor of two
    bits, and relu(a + b - 1) their AND. The gate of the first unit is exactly 0 where both bits
    are 0, and the third unit has no output weight."""
    no_weights = [[0, 0]] * 3  # of the values: each value is its bias, 1
    return mlp(bits, [[1, 1]] * 3, [0, -1, 0], no_weights, [1] * 3, [[1, -2, 0], [0, 1, 0]], "both")


def copy_and_or(bits):
    """The first output is relu(1) * a: a gate that has no weights, and a value that reads a bit."""
    return [mlp(bits[:1], [[0]], [1], [[1]], [0], [[1]], "copy")[0], lib.or_(bits)]


@pytest.mark.parametrize("function", [unusual_units, copy_and_or])
def test_hidden_units_of_kinds_that_lib_does_not_make_are_lowered(function):
    lowered = lower(transformer(trace(function, 2), PAIR_CODES), HIDDEN_UNITS)
    weights = gemma.build_weights(lowered, SMALL_3.shapes, SMALL_3.config)
    gemma.check_weights(weights, SMALL_3.config, lowered, [5, 6, 7, 8, 8, 7, 5, 5, 6, 8])


def test_a_hidden_unit_whose_gate_no_input_makes_positive_is_left_out():
    def function(bits):  # once the 0 is folded, no input makes the gates of the AND positive
        return [lib.and_([bits[0], lib.constant("0")[0]]), lib.or_(bits)]

    lowered = lower(transformer(fold(trace(function, 2)), PAIR_CODES))
    assert not [name for name in lowered.blocks[0].units if name.startswith("and")]


def test_build_weights_says_what_a_gemma_is_too_small_for():
    few_layers = SMALL_3.config | {"layer_types": ["sliding_attention"] * 6}
    with pytest.raises(AssertionError, match="next_token needs 7 layers, and Gemma has 6"):
        gemma.build_weights(LOWERED_SPONGE, SMALL_3.shapes, few_layers)
    narrow = small_metadata(3, 100, HIDDEN_UNITS, layers=8, tokens=300)
    with pytest.raises(AssertionError, match="next_token needs 116 features, and Gemma has 100"):
        gemma.build_weights(LOWERED_SPONGE, narrow.shapes, narrow.config)


def test_a_circuit_whose_heads_would_be_in_a_layer_with_full_attention_is_refused():
    lowered = lower(ADDER)  # it reads 8 tokens: on this Gemma, layers 0 and 1 attend
    layer_types = ["sliding_attention"] + ["full_attention"] * 7
    message = "next_token reads 8 tokens: the first 2 layers of Gemma must all have sliding"
    with pytest.raises(AssertionError, match=message):
        gemma.build_weights(lowered, SMALL_3.shapes, SMALL_3.config | {"layer_types": layer_types})


def test_save_writes_the_config_the_names_and_every_tensor_in_bf16(tmp_path, monkeypatch):
    # The embedding of 20000 tokens is three chunks, and the first two are zeros.
    small = small_metadata(4, FEATURES, HIDDEN_UNITS, layers=8, tokens=20000)
    monkeypatch.setattr(checkpoint, "metadata", lambda repo: small)
    codes = {19000 + token: [token >> 1, token & 1] for token in range(4)}
    model = transformer(trace(lambda bits: lib.add(bits[:2], bits[2:]), 4), codes)
    gemma.save(model, tmp_path, "a small Gemma", check_on=[19001, 19001, 19002, 19003])
    data = (tmp_path / "model.safetensors").read_bytes()
    header_length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_length])
    numbers = np.frombuffer(data[8 + header_length :], "<u2")  # a bf16 takes two bytes
    saved = {}
    end = 0
    for name, shape in small.shapes.items():  # each tensor comes after the one before it
        start = end
        end = start + 2 * int(np.prod(shape))
        assert header[name] == {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
        upper_bits = numbers[start // 2 : end // 2].astype(np.uint32) << 16  # of a float32
        saved[name.removeprefix("model.")] = upper_bits.view(np.float32).reshape(shape)
    assert header_length % 8 == 0  # the header is padded
    assert len(numbers) == end // 2  # no number is missing, though the last tensors are zeros
    lowered = lower(model, HIDDEN_UNITS)
    # The file runs too.
    gemma.check_weights(saved, small.config, lowered, [19003, 19002, 19001, 19001])
    names = json.loads((tmp_path / "names.json").read_text())
    assert names["features"] == lowered.layout  # every feature at every depth
    assert names["hidden"] == [block.units for block in lowered.blocks]  # every hidden unit
    assert json.loads((tmp_path / "config.json").read_text()) == small.config


def test_save_checks_the_weights_on_the_tokens_of_check_on_before_it_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "metadata", lambda repo: SMALL_3)
    model = transformer(trace(lib.xor, 2), BIT_CODES)  # it reads two tokens
    with pytest.raises(AssertionError, match="the circuit reads 2 tokens, and it got 1"):
        gemma.save(model, tmp_path, "a small Gemma", check_on=[5])
    assert not (tmp_path / "model.safetensors").exists()


def test_round_to_bf16_gives_the_nearest_and_of_two_equally_near_the_even_one():
    # A bf16 keeps 8 significant bits: from 1 to 2, the bf16 numbers are 2**-7 apart.
    between = np.float32([1 + 2**-8, 1 + 3 * 2**-8, 1 + 2**-9, -1 - 3 * 2**-9])
    assert checkpoint.round_to_bf16(between).tolist() == [1, 1 + 2**-6, 1, -1 - 2**-7]


def test_the_sponge_gives_every_next_token_with_noise_of_one_and_a_half_percent_on_every_weight():
    weights = gemma.build_weights(LOWERED_SPONGE, SMALL_4.shapes, SMALL_4.config)
    rng = np.random.default_rng(0)
    noisy = {}
    for name, tensor in weights.items():
        noisy[name] = tensor * (1 + 0.015 * rng.standard_normal(tensor.shape))
    # Rounding to bf16 moves a weight by 0.4% at most: this noise is far larger.
    gemma.check_weights(noisy, SMALL_4.config, LOWERED_SPONGE, SPONGE_TOKENS, tolerance=1 / 2)


def test_the_keccak_tensors_for_gemma_3_are_those_that_ran_on_the_real_model():
    if not checkpoint.cache_file(REPO).exists():
        pytest.importorskip("huggingface_hub")  # checkpoint.metadata fetches the shapes with it
    metadata = checkpoint.metadata(REPO)
    hidden_units = metadata.config["intermediate_size"]
    lowered = lower(transformer(keccak.circuit(), keccak.CODES), hidden_units)
    weights = gemma.build_weights(lowered, metadata.shapes, metadata.config)
    # Another number means that a weight changed: run the tests of a real Gemma, then put it here.
    assert checksum(weights) == 177880831


def test_run_gives_what_hugging_face_transformers_gives_on_gemma_3s_own_weights(torch):
    from transformers import AutoModelForCausalLM

    revision = checkpoint.REVISIONS[REPO]
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=revision, dtype=torch.float32)
    model.to("cuda")
    # A <bos>, token 2, and 599 random tokens: more than the sliding window of a layer, 512.
    tokens = [2, *np.random.default_rng(0).integers(1000, 200000, 599).tolist()]
    with torch.no_grad():
        output = model(torch.tensor([tokens], device="cuda"), output_hidden_states=True)
    tensors = checkpoint.read_weights(REPO)
    weights = {name.removeprefix("model."): tensor for name, tensor in tensors.items()}
    config = checkpoint.metadata(REPO).config
    streams = gemma.run(weights, config, tokens)
    for stream, hidden_state in zip(streams, output.hidden_states[1:18]):  # after layers 0 to 16
        theirs = hidden_state[0].cpu().numpy()
        assert np.abs(stream - theirs).max() < 1e-4 * np.abs(theirs).max()
    logits = gemma.logits(weights, config, streams[-1][-16:])  # after layer 17, at 16 positions
    assert np.abs(logits - output.logits[0, -16:].cpu().numpy()).max() < 0.01


@pytest.mark.parametrize("repo", list(checkpoint.REVISIONS))
def test_a_real_gemma_in_bf16_generates_what_the_ideal_transformer_generates(repo, torch, tmp_path):
    from transformers import AutoModelForCausalLM

    pinned = json.loads(checkpoint.fetch(repo, "config.json").read_text())  # at REVISIONS[repo]
    assert checkpoint.metadata(repo).config == pinned  # the checkpoints are written from it
    vocabulary = gemma.vocabulary(repo)  # {text of a token: its id}
    adder_codes = {vocabulary[symbol]: code for symbol, code in adder.CODES.items()}
    cases = [
        (transformer(keccak.circuit(), keccak.CODES), [keccak.DASH] * 3, 12),
        (transformer(adder.circuit(), adder_codes), [vocabulary[c] for c in "123+246="], 3),
    ]
    for model, prompt, count in cases:
        expected = model.generate(prompt, count)
        gemma.save(model, tmp_path, repo, check_on=expected)
        loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).to("cuda")
        tokens = torch.tensor([[vocabulary["<bos>"], *prompt]], device="cuda")
        for _ in range(count):
            with torch.no_grad():
                next_token = loaded(tokens).logits[:, -1:].argmax(dim=2)
            tokens = torch.cat([tokens, next_token], dim=1)
        assert tokens[0, 1:].tolist() == expected

"""Tests of beside.py. Most of them run a small Gemma 3 with random weights in numpy, alone and
with a circuit beside it. The last two run the real Gemma 3 270M on a GPU (see conftest.torch)."""

import math
from itertools import pairwise

import alongside
import numpy as np
import pytest
from conftest import checksum, small_metadata

from reifier import beside, checkpoint, gemma, lib
from reifier.check import expected_stream, signal_values
from reifier.circuit import trace
from reifier.lower import lower
from reifier.lower_stream import READ_SIZE
from reifier.transformer import transformer

FEATURES = 48  # the features of the small Gemma
HIDDEN_UNITS = 64  # the hidden units of each of its layers
# The small Gemma has 16 tokens. The code of a token is the 4 bits of its id, the lowest first.
CODES = {token: [token >> bit & 1 for bit in range(4)] for token in range(16)}
TOKENS = np.random.default_rng(1).integers(0, 16, 64).tolist()  # the tokens to run it on
SMALL = small_metadata(3, FEATURES, HIDDEN_UNITS, layers=6, tokens=len(CODES))  # see conftest.py
# Gemma 3 has both types of layer: the last layer of the small one attends to all tokens.
LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]
CONFIG = SMALL.config | {"layer_types": LAYER_TYPES, "rope_theta": 1e6}
REVISION = checkpoint.REVISIONS[beside.GEMMA]  # of the real Gemma 3 270M, for the last two tests


def random_weights(shapes: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Random tensors of `shapes`, by name, as checkpoint.read_weights gives those of the real
    Gemma. A norm multiplies by about 1, and a projection keeps the magnitude of what it reads.
    Every embedding has the RMS GEMMA_RMS_BEFORE, as in the real one."""
    rng = np.random.default_rng(0)
    weights = {}
    for name, shape in shapes.items():
        scale = 0.1 if "norm" in name else 1 / math.sqrt(shape[-1])  # a norm stores 1 less
        weights[name] = (scale * rng.standard_normal(shape)).astype(np.float32)
    embedding = weights["model.embed_tokens.weight"]
    embedding *= beside.GEMMA_RMS_BEFORE / np.linalg.norm(embedding, axis=1, keepdims=True)
    return weights


def gemma_alone(gemma_weights: dict, config: dict, extra_features: int) -> beside.WideGemma:
    """The Gemma of `gemma_weights` and `config` alone, with `extra_features` empty features
    before its own: as wide as beside a circuit. It shows what widening alone changes."""
    features = config["hidden_size"] + extra_features
    wide = beside.widen(gemma_weights, config, features, config["intermediate_size"])
    gemma.finish(wide.weights, wide.config)
    return wide


def answer(case):
    """The value of a table of two bits: the second bit, or no output, 1/2, where the first is 1."""
    return 1 / 2 if case[0] else case[1]


def next_code(bits):
    """The code of the next token, from the codes of the last two tokens, the older one first:
    the xor of their ids. After an older token with an odd id, the circuit is silent."""
    older = bits[:4]
    newer = bits[4:]
    mixed = [lib.xor([older[bit], newer[bit]]) for bit in range(4)]
    return [lib.table([older[0], mixed[bit]], answer, f"next{bit}") for bit in range(4)]


def logits(weights, config: dict) -> np.ndarray:
    """The logits [position, token] of a Gemma of gemma.run on TOKENS, from position 1 on."""
    last_stream = gemma.run(weights, config, TOKENS)[-1]
    return gemma.logits(weights, config, last_stream)[1:]


SMALL_GEMMA = random_weights(SMALL.shapes)  # the tensors of the small Gemma
MODEL = transformer(trace(next_code, 8), CODES)
# Each block gets copies of its units, up to 128 in all: Gemma's layers have 64, and get more.
LOWERED = beside.lower_beside(MODEL, hidden_units=128)
WIDE = beside.wide_gemma(LOWERED, SMALL_GEMMA, CONFIG)  # 3 layers, then Gemma's 6, then 1 more
WIDTH = len(LOWERED.layout[0])  # the circuit has the first features of the stream, Gemma the last
ALL_FEATURES = WIDE.config["hidden_size"]  # 30 of the circuit, 2 empty ones, 48 of Gemma
OUTPUTS = list(range(1, 5))  # the features of the outputs of the circuit
# An output after the last layer of WIDE: beside.OUTPUT_SIZE is for a stream of 896 features.
OUTPUT_SIZE = beside.OUTPUT_SIZE * math.sqrt(ALL_FEATURES / beside.REFERENCE_FEATURES)
STREAMS = gemma.run(WIDE.weights, WIDE.config, TOKENS)  # the stream after each layer
# The circuit reads two tokens, so it has an answer from position 1 on. At each of those positions:
NEXT_TOKENS = logits(WIDE.weights, WIDE.config).argmax(axis=1)  # of the wide model
SILENT = np.array([older % 2 == 1 for older in TOKENS[:-1]])  # where the circuit is silent: 29
ANSWERS = np.array([older ^ newer for older, newer in pairwise(TOKENS)])  # else, its answer
# What the circuit's features hold after its last block, for bits of size 1: -1 or +1 for an
# output where the circuit answers, 1 for a silence flag where it is silent, 1 for ONE, else 0.
CARRIED_STREAM = expected_stream(LOWERED.layout[-1], signal_values(MODEL, TOKENS))


def outputs_after_last_layer(stream: np.ndarray, size: float) -> np.ndarray:
    """The outputs [position, bit] after the last layer of WIDE, for a `stream` before it. Gemma's
    features are first scaled, so that the RMS of the stream is `size` times GEMMA_RMS_AFTER."""
    stream = stream.copy()
    rms = np.sqrt((stream**2).mean(axis=1, keepdims=True))
    stream[:, -FEATURES:] *= size * beside.GEMMA_RMS_AFTER / rms
    last = len(STREAMS) - 1
    stream = stream + gemma.run_attention(WIDE.weights, WIDE.config, last, stream)
    return (stream + gemma.run_block(WIDE.weights, WIDE.config, last, stream))[:, OUTPUTS]


def test_the_tensors_of_the_small_wide_model_have_not_changed():
    # Another number means that a weight changed: run the tests of the real Gemma, then put it here.
    assert checksum(WIDE.weights) == 3026347541


def test_gemma_alone_widened_by_empty_features_gives_the_logits_of_gemma():
    own = {name.removeprefix("model."): tensor for name, tensor in SMALL_GEMMA.items()}
    wide = gemma_alone(SMALL_GEMMA, CONFIG, extra_features=32)
    assert np.allclose(logits(wide.weights, wide.config), logits(own, CONFIG), atol=1e-5)


def test_each_layer_changes_only_the_features_of_its_own_model():
    before = WIDE.weights["embed_tokens.weight"][TOKENS] * math.sqrt(ALL_FEATURES)
    for stream, units in zip(STREAMS, WIDE.names["hidden"], strict=True):
        changed = (stream != before).any(axis=0)  # for each feature: whether the layer changed it
        if units:  # the names of the hidden units of the layer: only the circuit's have names
            assert changed[:WIDTH].any()
            assert not changed[WIDTH:].any()
        else:  # a layer of Gemma
            assert changed[-FEATURES:].all()
            assert not changed[:-FEATURES].any()
        before = stream


def test_where_the_circuit_answers_the_next_token_is_its_answer():
    assert np.array_equal(NEXT_TOKENS[~SILENT], ANSWERS[~SILENT])


def test_where_the_circuit_is_silent_the_next_token_is_that_of_gemma_alone():
    # Gemma's layers do not read what the circuit carries, but their norms divide by the RMS of
    # the whole stream. So Gemma alone gets as much as is carried, in a feature that it does not
    # read: the embedding of every token holds it, and Gemma multiplies that by sqrt(features).
    alone = gemma_alone(SMALL_GEMMA, CONFIG, extra_features=ALL_FEATURES - FEATURES)
    carried_square = (STREAMS[-2][1, :WIDTH] ** 2).sum()  # the same at every position
    alone.weights["embed_tokens.weight"][:, 0] = math.sqrt(carried_square / ALL_FEATURES)
    own = logits(alone.weights, alone.config).argmax(axis=1)
    assert np.array_equal(NEXT_TOKENS[SILENT], own[SILENT])


def test_a_circuit_that_is_never_silent_gives_its_answer_at_every_position():
    def always(bits):  # the xor of the ids of the last two tokens
        return [lib.xor([bits[bit], bits[4 + bit]]) for bit in range(4)]

    lowered = beside.lower_beside(transformer(trace(always, 8), CODES), hidden_units=128)
    wide = beside.wide_gemma(lowered, SMALL_GEMMA, CONFIG)
    assert np.array_equal(logits(wide.weights, wide.config).argmax(axis=1), ANSWERS)


def test_the_outputs_are_carried_at_carry_size_through_the_layers_of_gemma():
    after_gemma = STREAMS[-2][1:, :WIDTH]  # the circuit's features after Gemma's last layer
    expected = beside.CARRY_SIZE * CARRIED_STREAM
    assert np.allclose(after_gemma, expected, atol=0.01 * beside.CARRY_SIZE)


def test_what_is_carried_adds_the_same_to_the_mean_square_whether_the_circuit_is_silent_or_not():
    squares = (STREAMS[-2][1:, :WIDTH] ** 2).sum(axis=1)  # at each position
    assert np.ptp(squares) < 0.01 * squares.mean()


@pytest.mark.parametrize("size", [0.2, 1, 5])
def test_the_last_layer_makes_the_same_outputs_in_a_stream_of_gemma_of_another_size(size):
    outputs = outputs_after_last_layer(STREAMS[-2][1:], size)
    assert np.allclose(outputs, OUTPUT_SIZE * CARRIED_STREAM[:, OUTPUTS], atol=0.002 * OUTPUT_SIZE)


def test_silence_closes_the_gates_of_the_last_block_even_to_an_output_that_holds_a_bit():
    carried = beside.Carried(LOWERED)
    flags = [feature for feature in carried.flags_and_one if feature != carried.one]
    stream = STREAMS[-2][1:][~SILENT]  # where the circuit answers: each output is a carried bit
    stream[:, flags] = beside.CARRY_SIZE  # and now the silence flags are on too
    added = outputs_after_last_layer(stream, size=1) - stream[:, OUTPUTS]
    assert np.abs(added).max() < 1e-4 * OUTPUT_SIZE


def test_a_block_of_the_circuit_reads_each_bit_at_read_size_in_the_stream_shared_with_gemma():
    for layer in (1, 2):  # the block of layer 1 or 2 reads the stream that the layer before left
        pre_norm = 1 + WIDE.weights[f"layers.{layer}.pre_feedforward_layernorm.weight"]
        read = np.abs(gemma.rms_norm(STREAMS[layer - 1][1:], pre_norm))
        bits = read[read > 1]  # every bit and every ONE that the block reads: all else is 0
        assert len(bits) > 0
        assert np.allclose(bits, READ_SIZE, rtol=0.01)


def test_a_circuit_whose_last_block_does_not_make_every_output_is_refused():
    def early(bits):  # the first output is made a block before the second
        return [lib.and_(bits[:2]), lib.xor(bits[2:])]

    codes = {token: [token >> 1, token & 1] for token in range(4)}
    with pytest.raises(AssertionError, match="early: its last block must make every output"):
        beside.lower_beside(transformer(trace(early, 4), codes))


def test_wide_gemma_refuses_a_circuit_that_it_cannot_put_beside_gemma():
    with pytest.raises(AssertionError, match="next_code must be lowered by lower_beside"):
        beside.wide_gemma(lower(MODEL), SMALL_GEMMA, CONFIG)
    half = {token: CODES[token] for token in range(8)}  # codes for only some tokens of Gemma
    lowered = beside.lower_beside(transformer(MODEL.circuit, half))
    with pytest.raises(AssertionError, match="next_code needs a code for each token of Gemma"):
        beside.wide_gemma(lowered, SMALL_GEMMA, CONFIG)
    codes = {token: [token & 1] for token in CODES}
    lowered = beside.lower_beside(transformer(trace(lib.xor, 5), codes))  # one bit of 5 tokens
    with pytest.raises(AssertionError, match="xor reads more tokens than a layer of Gemma has"):
        beside.wide_gemma(lowered, SMALL_GEMMA, CONFIG)


def mixed(bits):
    """Two outputs from the codes of two tokens, of two bits each: the first can be 1/2, and the
    second is always a bit."""
    return [lib.table(bits[0::2], answer, "first"), lib.and_(bits[1::2])]


def unlike(bits):
    """Two outputs that can be 1/2 for different inputs: each where its own bit of the older
    token is 1."""
    return [lib.table(bits[0::2], answer, "first"), lib.table(bits[1::2], answer, "second")]


@pytest.mark.parametrize("function", [mixed, unlike])
def test_a_circuit_whose_outputs_are_not_one_half_for_the_same_inputs_is_refused(function):
    codes = {token: code[:2] for token, code in CODES.items()}
    lowered = beside.lower_beside(transformer(trace(function, 4), codes))
    with pytest.raises(AssertionError, match="where one output is 1/2, every output"):
        beside.wide_gemma(lowered, SMALL_GEMMA, CONFIG)


def test_save_writes_the_tensors_of_the_wide_model_in_bf16(tmp_path, monkeypatch):
    read_weights = checkpoint.read_weights  # the real one: it reads the file that `fetch` gives
    # `save` gets the small Gemma in place of the real one, and `fetch` gives what `save` wrote.
    monkeypatch.setattr(checkpoint, "metadata", lambda repo: checkpoint.Metadata({}, CONFIG))
    monkeypatch.setattr(checkpoint, "read_weights", lambda repo: SMALL_GEMMA)
    monkeypatch.setattr(checkpoint, "fetch", lambda repo, name: tmp_path / name)
    beside.save(MODEL, tmp_path, hidden_units=128, check_on=TOKENS)
    saved = read_weights(beside.GEMMA)
    for name, tensor in WIDE.weights.items():
        assert np.array_equal(saved["model." + name], checkpoint.round_to_bf16(tensor))


def test_save_checks_the_weights_on_the_tokens_of_check_on_before_it_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "metadata", lambda repo: checkpoint.Metadata({}, CONFIG))
    monkeypatch.setattr(checkpoint, "read_weights", lambda repo: SMALL_GEMMA)
    with pytest.raises(AssertionError, match="the circuit reads 2 tokens, and it got 1"):
        beside.save(MODEL, tmp_path, check_on=TOKENS[:1])
    assert not (tmp_path / "model.safetensors").exists()


@pytest.fixture
def tokens(torch) -> list[int]:
    """A <bos> and an English text, the docstring of beside.py, as tokens of the real Gemma."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(beside.GEMMA, revision=REVISION)
    return tokenizer(beside.__doc__)["input_ids"]


def real_logits(torch, tokens, source, revision=None) -> np.ndarray:
    """The logits [position, token] on `tokens` of the checkpoint `source`, a directory or a
    model of huggingface.co, as `transformers` computes them in bf16 on the GPU."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, dtype=torch.bfloat16)
    with torch.no_grad():
        logits = model.to("cuda")(torch.tensor([tokens], device="cuda")).logits
    return logits[0].float().cpu().numpy()


def test_the_real_gemma_alone_widened_gives_the_logits_of_gemma(torch, tokens, tmp_path):
    config = checkpoint.metadata(beside.GEMMA).config
    wide = gemma_alone(checkpoint.read_weights(beside.GEMMA), config, extra_features=256)
    wide.write(tmp_path)
    own = real_logits(torch, tokens, beside.GEMMA, REVISION)
    difference = np.abs(real_logits(torch, tokens, tmp_path) - own).max(axis=1)  # at each position
    # Rounding the wider weights to bf16 again moves a logit by 0.3 as a rule. At a few positions,
    # fewer than 1 in 100, it moves one logit by more than 1.
    assert np.median(difference) < 0.5
    assert np.quantile(difference, 0.99) < 1


def test_the_real_gemma_gives_the_token_of_the_sponge_beside_it_or_its_own(torch, tokens, tmp_path):
    sponge = transformer(alongside.circuit(), alongside.CODES)
    beside.save(sponge, tmp_path, hidden_units=4096, check_on=range(1000, 1032))  # as alongside.py
    values = signal_values(sponge, tokens)  # from position 1 on: the sponge reads two tokens
    id_bits = np.array([values[name] for name in sponge.circuit.outputs[: alongside.ID_BITS]])
    silent = id_bits[0] == 1 / 2  # as every output is, where the sponge has no answer
    answers = id_bits.T.astype(int) @ (1 << np.arange(alongside.ID_BITS))
    found = real_logits(torch, tokens, tmp_path)[1:].argmax(axis=1)
    assert np.array_equal(found[~silent], answers[~silent])
    own = real_logits(torch, tokens, beside.GEMMA, REVISION)[1:]
    top_two = np.partition(own, -2, axis=1)[:, -2:]
    # Rounding Gemma's weights to bf16 again moves a logit by 0.3 or so, which can change the
    # next token where two are close. So it is compared where Gemma's own leads by a logit.
    clear = silent & (top_two[:, 1] - top_two[:, 0] > 1)
    assert clear.sum() > 100
    assert np.array_equal(found[clear], own.argmax(axis=1)[clear])

"""Step 8, the lowering, part 1: the stream of a real model, its embedding and its attention.

Lowering turns the ideal transformer of transformer.py into layers that a real model, Gemma,
can hold. The weights change and what they compute does not: the stream of the lowered transformer
is the ideal stream with every feature multiplied by a known number, the size of the feature.
A real model differs from the ideal one in five ways. This file deals with the stream and
the attention, and lower.py with the MLP blocks.

Normalization. A layer of Gemma computes x + post_norm(sublayer(pre_norm(x))), once with its
attention as the sublayer and once with its MLP. The pre-norm and the post-norm are RMSNorms:
an RMSNorm divides x by the root mean square (RMS) of its features, and then multiplies each
feature by a weight. A feature holds a bit as minus or plus its size, so the RMS of the stream
is the same whatever the bits are (see `feature_sizes`, `stream_square`). A pre-norm is then a
fixed scaling: its weights make every bit -READ_SIZE or +READ_SIZE where the sublayer reads
it. A post-norm divides by the RMS of what the sublayer outputs, so that RMS must not depend
on the bits either (see lower.normed_block). Two cases need care. An MLP with several outputs
may output numbers that are not bits (see circuit.py): with each output held as 2 * value - 1,
their squares must add up to the same number for every input, as those of lib.table do (see
`check_sampled_values`). And a block that overwrites a bit adds new - old to the feature,
which is 0 or twice a bit. It then has one more row of w_out, a dropped row, which gets new + old:
the post-norm counts it in the RMS and multiplies it by 0 (see lower.dropped_rows).

GELU. A hidden unit of Gemma is gelu(gate) * value, and GELU is close to ReLU only far from
0. The gate margin of a unit is the least distance of its gate from 0 over all inputs, not
counting a gate of exactly 0, where GELU equals ReLU. Every gate is scaled so that its margin is
GATE_MARGIN (see lower.gate_scales).

Attention. A head of a real model is not told which token is h positions back. It scores
each token by the dot product of a query and a key, which RoPE rotates by their positions.
The query of head h is made so that the key of the token h positions back scores highest (see
`lookback_query`, `lower_attention`).

Magnitudes. The weights of the lowered transformer, and the gates and values of its hidden
units that are on, are in the ranges of a trained Gemma: the constants of both files set them,
and lower.block_scales balances the gates and the values of each block.

Noise. The weights of a saved model are slightly off, for example after rounding to 16 bits:
that is noise on the weights. ONE is held in ONE_FEATURES features of different sizes, so
that a bias is many small weights (see `one_sizes`). An eraser (see transformer.block) leaves
a little of the signal that it subtracts, the residue, and the next block erases that too: the
second erasure (see lower.eraser_units). And a block has several copies of each hidden unit
when the layer has room, so that their errors average out (see lower.copy_counts).
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from .circuit import Circuit
from .layout import ONE, Layout
from .transformer import Transformer

BIT_SIZE = 2.0  # a feature holds a bit as -2 or +2: twice the RMS of Gemma's embedded features
READ_SIZE = 6.5  # a bit after a pre-norm: so large that the gate weights of lib.table are under 1
ONE_FEATURES = 16  # the features that hold ONE: a bias is 16 small weights, not one large one
ONE_SPREAD = 0.5  # the ONE features hold from 1 - 0.5 to 1 + 0.5 times BIT_SIZE (see `one_sizes`)
SAMPLES = 64  # the random inputs of `sample_values`: the norms need no more (see `mean_squares`)
KEY_VALUE_WEIGHT = 0.375  # each weight of a key and a value: large beside noise of a fixed size
# Most layers of Gemma have sliding attention: a token attends only to the last 512 tokens, and
# RoPE rotates pair i of the 128 pairs of dimensions of a head by 10000 ** -(i / 128) radians
# per position. The heads of this file are for such a layer.
SLIDING_WINDOW = 512  # a head chooses among this many tokens
ROPE_PAIRS = 8  # the pairs that a query or a key uses: with 8, the best wrong token scores least
ROPE_FREQUENCIES = 10000.0 ** -(np.arange(ROPE_PAIRS) / 128)  # the first pairs rotate fastest
ATTENTION_MARGIN = 14.0  # a head's token outscores every other by this: the rest share under 0.05%


def lowered_layout(model: Transformer) -> Layout:
    """The layout of `model` with ONE_FEATURES - 1 more features that hold ONE in every row.
    They come right after the code, because the embedding fills the first features."""
    code_end = 1 + len(model.circuit.outputs)
    more_ones = [ONE] * (ONE_FEATURES - 1)
    return [row[:code_end] + more_ones + row[code_end:] for row in model.layout]


def lowered_features(model: Transformer) -> np.ndarray:
    """For each feature of the layout of `model`, its feature in `lowered_layout`."""
    code_end = 1 + len(model.circuit.outputs)
    ideal = np.arange(len(model.layout[0]))
    return np.where(ideal < code_end, ideal, ideal + ONE_FEATURES - 1)


def one_sizes(count: int) -> np.ndarray:
    """`count` numbers with mean 1, from 1 - ONE_SPREAD to 1 + ONE_SPREAD, one for each ONE
    feature: it holds BIT_SIZE times its number, and carries its number / `count` of each bias
    (see lower.on_normed_stream). Equal weights would all be rounded the same way, and their
    errors would add up. These are rounded differently, and their errors mostly cancel."""
    return 1 + ONE_SPREAD * np.linspace(-1, 1, count)


def feature_sizes(layout: Layout) -> np.ndarray:
    """The size [depth, feature] of each feature of `layout`: BIT_SIZE where it holds a signal
    and 0 where it is empty. A feature that holds ONE in row 0 has its number of `one_sizes`
    times that."""
    holds_signal = np.array([[name is not None for name in row] for row in layout])
    sizes = BIT_SIZE * holds_signal
    ones = [f for f, name in enumerate(layout[0]) if name == ONE]
    sizes[:, ones] *= one_sizes(len(ones))
    return sizes


def sample_values(circuit: Circuit, codes: dict[int, list[int]]) -> dict[str, np.ndarray]:
    """The value of every signal of `circuit` on SAMPLES of its inputs: the codes of random
    tokens, the same ones in every run."""
    tokens = list(codes)
    window = len(circuit.inputs) // len(circuit.outputs)
    drawn = np.random.default_rng(0).integers(0, len(tokens), (SAMPLES, window))
    inputs = [np.concatenate([codes[tokens[i]] for i in row]) for row in drawn]
    return circuit.values(inputs)


def non_bit_value(value: np.ndarray) -> float | None:
    """A number of `value` that is neither 0 nor 1, or None if all of them are bits."""
    others = value[~np.isclose(value, 0) & ~np.isclose(value, 1)]
    return float(others[0]) if others.size else None


def check_sampled_values(circuit: Circuit, values: dict[str, np.ndarray]):
    """Refuse a circuit that breaks one of the rules below, which a real model needs. Only the
    inputs of `values` are tried: those of `sample_values`, or of check.signal_values."""
    for m in circuit.mlps:
        for name in m.inputs:
            other = non_bit_value(values[name])
            rule = "an MLP may read only bits"
            assert other is None, f"{m.name} reads {name}, which can be {other:g}: {rule}"
        if len(m.outputs) == 1:
            other = non_bit_value(values[m.outputs[0]])
            rule = "an MLP with one output must make a bit"
            assert other is None, f"{m.outputs[0]} can be {other:g}: {rule}"
        squares = sum((2 * values[name] - 1) ** 2 for name in m.outputs)
        rule = "must add up to the same number for every input"
        fixed = np.allclose(squares, squares[0])
        assert fixed, f"{m.name}: the squares of its outputs, each held as 2 * value - 1, {rule}"


def mean_squares(circuit: Circuit, codes: dict[int, list[int]]) -> dict[str, float]:
    """The mean square of every signal, and of ONE, as the ideal stream holds it, 2 * value - 1,
    on the inputs of `sample_values`: 1 for a bit, and less for a signal that can be 1/2. The
    mean of such a signal depends on the sample. But the squares of the outputs of its MLP add
    up to the same number for every input (see `check_sampled_values`), and so do their means:
    a stream that holds them all has the same RMS for every input."""
    values = sample_values(circuit, codes)
    check_sampled_values(circuit, values)
    mean_square = {name: float(np.mean((2 * value - 1) ** 2)) for name, value in values.items()}
    return mean_square | {ONE: 1.0}


def stream_square(row: list[str | None], sizes: np.ndarray, mean_square: dict[str, float]) -> float:
    """The mean over inputs of the sum of the squares of a stream that holds `row`, a row of a
    layout, with the feature sizes `sizes` [feature]."""
    signal_square = np.array([mean_square[name] if name else 0.0 for name in row])
    return float((signal_square * sizes**2).sum())


def lower_embedding(model: Transformer, first_sizes: np.ndarray) -> np.ndarray:
    """The embedding [token, feature] of the lowered transformer, for the features of the code and
    of ONE. A token that has a code holds each ONE at its size in row 0, `first_sizes`, and
    its code at BIT_SIZE. Any other token is all zeros, as in the ideal transformer."""
    code_end = 1 + len(model.circuit.outputs)
    has_code = model.embedding[:, :1]  # 1 for a token that has a code, 0 for any other
    embedding = has_code * first_sizes[: code_end + ONE_FEATURES - 1]
    embedding[:, 1:code_end] = BIT_SIZE * model.embedding[:, 1:]
    return embedding


@dataclass
class Attention:
    """The attention heads, between a pre-norm and a post-norm. All heads share one key and
    one value projection, as the heads of a layer of Gemma do. A query and a key use the
    first ROPE_PAIRS pairs of dimensions, which RoPE must rotate by ROPE_FREQUENCIES."""

    w_query: np.ndarray  # [head, dimension, feature]: the first of each pair, then the second
    w_key: np.ndarray  # [dimension, feature]
    w_value: np.ndarray  # [bit, feature]: what a head copies from a token, here its code
    w_out: np.ndarray  # [head, feature, bit]
    pre_norm: np.ndarray  # [feature]: the weights of the RMSNorm before the heads
    post_norm: np.ndarray  # [feature]: the weights of the RMSNorm after them


def with_norms(attention: Attention, size_before: np.ndarray, size_after: np.ndarray) -> Attention:
    """`attention` with its two norms, for a stream whose features have the sizes `size_before`
    [feature] before the heads and `size_after` after them. The pre-norm makes each feature
    that the heads read READ_SIZE. The heads write the same magnitude to every feature that
    they write, so the RMS of what they write is that magnitude times the square root of the
    share of such features. The post-norm then gives each of them its size: that of the bit
    which a head erases there, or of the bit which a head writes."""
    width = len(size_before)
    read = attention.w_value.any(axis=0) | attention.w_key.any(axis=0)
    written = attention.w_out.any(axis=(0, 2))
    stream_rms = math.sqrt((size_before * size_before).sum() / width)  # each holds a bit or ONE
    pre_norm = np.zeros(width)
    pre_norm[read] = READ_SIZE / size_before[read] * stream_rms
    write_size = np.maximum(size_before, size_after)[written]
    post_norm = np.zeros(width)
    post_norm[written] = write_size * math.sqrt(written.sum() / width)
    return replace(attention, pre_norm=pre_norm, post_norm=post_norm)


def lookback_query(offset: int) -> np.ndarray:
    """The query of a head that attends to the token `offset` positions back. Up to a factor,
    every key is 1 in the first of each pair of dimensions and 0 in the second, and this query
    is that key rotated back `offset` positions. So the token e positions from the wanted one
    scores that factor times the sum of cos(e * ROPE_FREQUENCIES), which is largest at e = 0."""
    angle = ROPE_FREQUENCIES * offset
    return np.concatenate([np.cos(angle), -np.sin(angle)])


def top_score() -> float:
    """The score that a head of `lookback_query` must give its token, so that every other
    token among the last SLIDING_WINDOW scores at least ATTENTION_MARGIN less."""
    offsets = np.arange(1, SLIDING_WINDOW)  # how far a wrong token can be from the wanted one
    runner_up = np.cos(np.outer(offsets, ROPE_FREQUENCIES)).mean(axis=1).max()
    return ATTENTION_MARGIN / (1 - runner_up)  # a wrong token scores at most runner_up of it


def lower_attention(
    model: Transformer, first_row: list[str | None], first_sizes: np.ndarray
) -> Attention:
    """Lower the attention heads of `model`. `first_row` is row 0 of the lowered layout and
    `first_sizes` its sizes. Keys and queries read only ONE, so they are the same at every
    position before RoPE rotates them. The value is the code of the token, and w_out[h] is
    where the ideal head h writes each bit of it."""
    code_end = 1 + len(model.circuit.outputs)
    code = list(range(1, code_end))
    width = len(first_row)
    ones = [f for f, name in enumerate(first_row) if name == ONE]
    key_size = KEY_VALUE_WEIGHT * READ_SIZE * len(ones)  # each key dimension, after the pre-norm
    query_size = top_score() / ROPE_PAIRS / key_size  # ROPE_PAIRS * query * key is the score
    query_weight = query_size / READ_SIZE / len(ones)
    w_key = np.zeros((2 * ROPE_PAIRS, width))
    w_key[:ROPE_PAIRS, ones] = KEY_VALUE_WEIGHT
    w_value = np.zeros((len(code), width))
    w_value[range(len(code)), code] = KEY_VALUE_WEIGHT
    w_query = np.zeros((len(model.heads), 2 * ROPE_PAIRS, width))
    w_out = np.zeros((len(model.heads), width, len(code)))
    to_lowered = lowered_features(model)
    for h, head in enumerate(model.heads):
        w_query[h][:, ones] = query_weight * lookback_query(h)[:, None]
        w_out[h][to_lowered] = head[:, 1:code_end]
    size_before = np.zeros(width)  # before the attention, the stream holds ONE and the code
    size_before[ones] = first_sizes[ones]
    size_before[code] = BIT_SIZE
    no_norm = np.zeros(width)  # `with_norms` makes both norms
    attention = Attention(w_query, w_key, w_value, w_out, no_norm, no_norm)
    return with_norms(attention, size_before, first_sizes)

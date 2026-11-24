"""Step 9, the lowering, part 2: the MLP blocks, and `lower`, which lowers a whole transformer.

A lowered block adds a fixed multiple of what its ideal block adds to each feature, and its
post-norm turns that multiple into the size of the feature. `lower_block` makes the hidden
units in three parts, as transformer.block does: the units of the circuit, rescaled, then the
erasers, then the constant unit. It makes the last two anew from the lowered layout, which
has more ONE features than the ideal one.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from .layout import ONE, Layout
from .lower_stream import (
    BIT_SIZE,
    READ_SIZE,
    Attention,
    feature_sizes,
    lower_attention,
    lower_embedding,
    lowered_features,
    lowered_layout,
    mean_squares,
    one_sizes,
    stream_square,
)
from .transformer import Block, Transformer

GATE_MARGIN = 3.5  # a gate is 0 or at least this far from 0: GELU is within 0.001 of ReLU there
GATE_WEIGHT_DENOMINATOR = 32  # an MLP's gate weights must be whole 32nds: lib.py's are halves
MAX_GATE_WEIGHT = 2.0  # a lowered unit may not have a larger gate weight: Gemma 3 270M's reach 1.17
ZERO_TOLERANCE = 1e-9  # a number nearer to 0 than this counts as 0: rounding errors are far smaller
VALUE_RATIO = 0.45  # largest value weight over largest gate weight: values are as large as Gemma's
OUT_WEIGHT = 0.5  # the largest weight from a unit to a feature: large beside noise of a fixed size
MAX_COPIES = 4  # the most copies of one hidden unit: four halve the effect of weight noise
TOP_LOGIT = 16.0  # the logit of the token that the circuit computed: Gemma 4 lets none exceed 30


@dataclass
class NormedBlock:
    """An MLP block of Gemma: x + post_norm(w_out @ (gelu(w_gate @ z) * (w_value @ z))), with
    z = pre_norm(x). w_out has one row for each feature of the layout and then the dropped
    rows (see `dropped_rows`)."""

    units: list[str]  # the name of each hidden unit
    w_gate: np.ndarray  # [unit, feature]
    w_value: np.ndarray  # [unit, feature]
    w_out: np.ndarray  # [row, unit]
    pre_norm: np.ndarray  # [feature]: the weights of the RMSNorm before the MLP
    post_norm: np.ndarray  # [feature]: the weights of the RMSNorm after it


@dataclass
class Lowered:
    """A lowered transformer, and `model`, the ideal transformer that it was lowered from.
    Every norm weight is for a stream of exactly the features of `layout`. An RMSNorm takes its
    mean over the whole stream: in a stream of n features, the others empty, each norm weight
    must be multiplied by sqrt(len(layout[0]) / n)."""

    model: Transformer
    layout: Layout
    sizes: np.ndarray  # [depth, feature]: the size of each feature in each row of `layout`
    mean_square: dict[str, float]  # the mean square of each signal (see `mean_squares`)
    embedding: np.ndarray  # [token, feature], for the first features: the code and every ONE
    attention: Attention
    blocks: list[NormedBlock]
    final_norm: np.ndarray  # [feature]: the weights of the RMSNorm before the logits


def lower(
    model: Transformer,
    hidden_units: int | None = None,
    layout: Layout | None = None,
    sizes: np.ndarray | None = None,
) -> Lowered:
    """Lower the ideal transformer `model`. A circuit that breaks a rule that a real model needs
    is refused if one of the sampled inputs shows it (see lower_stream.check_sampled_values).

    `hidden_units` is the number of hidden units in a layer of the real model. Each block has as
    many copies of its units as fit, up to MAX_COPIES, or one copy with None (see `copy_counts`).

    `layout` and `sizes` [depth, feature] replace lowered_layout(model) and
    feature_sizes(layout). They are for a caller that needs other rows, as beside.py does:
    more features, ONE in other features of a row, or other sizes. A signal must stay in the
    feature that lowered_layout(model) gives it, at one size for as long as it is there. And
    a signal that need not be a bit may neither overwrite ONE nor be overwritten by ONE (see
    layout.py)."""
    if layout is None:
        layout = lowered_layout(model)
    if sizes is None:
        sizes = feature_sizes(layout)
    mean_square = mean_squares(model.circuit, model.codes)  # refuses a circuit that breaks a rule
    to_lowered = lowered_features(model)
    code = list(range(1, 1 + len(model.circuit.outputs)))  # the features of the code
    embedding = lower_embedding(model, sizes[0])
    attention = lower_attention(model, layout[0], sizes[0])
    blocks = []
    residue = code  # head 0 erased the code of its token: the first block erases it again
    for depth, ideal in enumerate(model.blocks, 1):
        change = Change(layout, sizes, depth, residue)
        blocks.append(lower_block(ideal, to_lowered, change, mean_square, hidden_units))
        residue = [f for f in change.released if layout[depth][f] is None]
    # The final norm lets only the outputs pass, in the features of the code. An output that
    # equals a bit of a code adds weight * size / RMS * BIT_SIZE to the logit of its token,
    # so the token whose whole code equals the outputs gets TOP_LOGIT.
    rms = math.sqrt(stream_square(layout[-1], sizes[-1], mean_square) / len(layout[-1]))
    final_norm = np.zeros(len(layout[-1]))
    final_norm[code] = TOP_LOGIT / (len(code) * BIT_SIZE * sizes[-1][code]) * rms
    return Lowered(model, layout, sizes, mean_square, embedding, attention, blocks, final_norm)


class Change:
    """What block `depth` changes in the stream, read off the rows of `layout` before and
    after it, which hold their signals at `sizes` [depth, feature]."""

    def __init__(self, layout: Layout, sizes: np.ndarray, depth: int, residue: list[int]):
        self.before = layout[depth - 1]  # the row of the layout before the block
        self.after = layout[depth]  # the row after it
        self.size_before = sizes[depth - 1]  # [feature]: the sizes of `before`
        self.residue = residue  # the empty features that the block before erased
        features = range(len(self.after))
        self.ones = [f for f in features if self.before[f] == ONE]  # they hold ONE in `before`
        # The features that get a new signal, those that a signal leaves, and those where both
        # happen: the new signal overwrites the old one.
        self.written = [f for f in features if self.after[f] and self.after[f] != self.before[f]]
        self.released = [f for f in features if self.before[f] and self.before[f] != self.after[f]]
        self.overwritten = [f for f in self.written if self.before[f]]
        # The block reads a feature at the size of the signal that it holds. Only the second
        # erasure reads an empty feature: as if it held a bit of BIT_SIZE.
        self.read_size = np.where(self.size_before > 0, self.size_before, BIT_SIZE)
        # The block writes a feature at the size of its new signal. A feature that it empties
        # is written at the size that it was read at.
        self.write_size = np.where(sizes[depth] > 0, sizes[depth], self.read_size)
        self.size_ratio = self.read_size / self.write_size


@dataclass
class Units:
    """Some hidden units of a lowered block, with what each of them is for."""

    names: list[str]
    mlp_names: list[str | None]  # for each unit: the name of its MLP in the circuit, or None
    erases: list[int | None]  # for each unit: the feature that it erases, or None
    w_gate: np.ndarray  # [unit, feature]
    w_value: np.ndarray  # [unit, feature]
    w_out: np.ndarray  # [feature, unit], and in the end one more row for each dropped row


@dataclass
class Scales:
    """The factors that keep the hidden units of one block in the ranges of a trained Gemma."""

    gates: np.ndarray  # [unit]: the factor on the gate of each unit of the circuit
    largest_gate_weight: float  # the largest gate weight of those units on the -1/+1 stream, scaled
    output: float  # every unit outputs this many times what it would in the ideal block


def lower_block(
    ideal: Block,
    to_lowered: np.ndarray,
    change: Change,
    mean_square: dict[str, float],
    hidden_units: int | None,
) -> NormedBlock:
    """Lower `ideal`, one block of the ideal transformer. `to_lowered` is the lowered feature
    of each of its features, and `mean_square` the mean square of each signal. `hidden_units`
    is that of `lower`."""
    kept = kept_units(ideal)
    scales = block_scales(kept)
    constant = constant_outputs(ideal, to_lowered, change)
    parts = [circuit_units(kept, scales, to_lowered, change), eraser_units(change, scales)]
    if constant.any():
        parts.append(constant_unit(constant, change.ones, scales.output))
    units = Units(
        names=[name for part in parts for name in part.names],
        mlp_names=[name for part in parts for name in part.mlp_names],
        erases=[feature for part in parts for feature in part.erases],
        w_gate=np.vstack([part.w_gate for part in parts]),
        w_value=np.vstack([part.w_value for part in parts]),
        w_out=np.hstack([part.w_out for part in parts]),
    )
    dropped = dropped_rows(units, change.overwritten)
    units.w_out = OUT_WEIGHT * np.vstack([units.w_out, dropped])
    counts = copy_counts(units, change, hidden_units)
    return repeat_units(normed_block(units, change, mean_square), counts)


def reads_bits(ideal: Block) -> np.ndarray:
    """For each hidden unit of `ideal`, whether its gate or its value reads a bit. Feature 0
    holds ONE, not a bit: the weight on it is the bias."""
    return ideal.w_gate[:, 1:].any(axis=1) | ideal.w_value[:, 1:].any(axis=1)


def kept_units(ideal: Block) -> Block:
    """The hidden units of `ideal` that the lowered block keeps: the units of the circuit,
    without those that read no bits (see `constant_outputs`), those whose gate no input makes
    positive, and those that have no output weight."""
    highest_gate = ideal.w_gate[:, 0] + np.abs(ideal.w_gate[:, 1:]).sum(axis=1)
    useful = reads_bits(ideal) & (highest_gate > ZERO_TOLERANCE) & ideal.w_out.any(axis=0)
    kept = [unit for unit, name in enumerate(ideal.mlp_names) if name is not None and useful[unit]]
    names = [ideal.units[unit] for unit in kept]
    mlp_names = [ideal.mlp_names[unit] for unit in kept]
    return Block(names, mlp_names, ideal.w_gate[kept], ideal.w_value[kept], ideal.w_out[:, kept])


def constant_outputs(ideal: Block, to_lowered: np.ndarray, change: Change) -> np.ndarray:
    """What the block adds to each feature whatever the bits are. The stream holds a new bit
    b as 2 * b - 1 and the units of its MLP add 2 * b, so the feature gets -1 more. A new ONE,
    in rows that a caller of `lower` gave, gets 1. And a unit of the circuit that reads no
    bits always adds the same: relu(gate bias) * value bias, times its output weights."""
    constant = np.zeros(len(change.after))
    for f in change.written:
        constant[f] = 1.0 if change.after[f] == ONE else -1.0
    reads = reads_bits(ideal)
    fixed = [name is not None and not reads[unit] for unit, name in enumerate(ideal.mlp_names)]
    hidden = np.maximum(ideal.w_gate[:, 0], 0) * ideal.w_value[:, 0]
    constant[to_lowered] += ideal.w_out[:, fixed] @ hidden[fixed]
    return constant


def gate_scales(kept: Block) -> np.ndarray:
    """For each hidden unit of `kept`, the factor that makes GATE_MARGIN the gate margin of
    the unit (see lower_stream.py).

    A gate is bias + weights @ s, where each entry of s is -1 or +1, because an MLP reads
    only bits. Flipping one entry changes the gate by twice a weight. So the gate takes its
    lowest value plus whole multiples of a step: twice the largest number that divides all
    its weights. The value nearest to 0, other than 0 itself, is scaled to GATE_MARGIN. The
    weights here are half the gate weights of the MLP (see transformer.on_stream). Those must
    be whole multiples of 1 / GATE_WEIGHT_DENOMINATOR, so that a largest number that divides
    them all exists."""
    bias = kept.w_gate[:, 0]
    weights = np.abs(kept.w_gate[:, 1:])
    whole = np.rint(2 * GATE_WEIGHT_DENOMINATOR * weights)
    on_grid = np.isclose(whole, 2 * GATE_WEIGHT_DENOMINATOR * weights).all(axis=1)
    grid = f"whole multiples of 1/{GATE_WEIGHT_DENOMINATOR}"
    for name, fits in zip(kept.mlp_names, on_grid):
        assert fits, f"{name}: its gate weights must be {grid}, so that its gates can be scaled"
    step = np.gcd.reduce(whole.astype(np.int64), axis=1, initial=0) / GATE_WEIGHT_DENOMINATOR
    has_weights = step > 0
    lowest = bias - weights.sum(axis=1)
    above = np.mod(lowest, np.where(has_weights, step, 1))  # the least value that is not negative
    nearest = np.minimum(above, step - above)  # that value, or the one a step below it
    nearest = np.where(nearest > ZERO_TOLERANCE, nearest, step)  # at 0 itself, GELU equals ReLU
    nearest = np.where(has_weights, nearest, np.abs(bias))  # a gate with no weights is its bias
    return GATE_MARGIN / nearest


def block_scales(kept: Block) -> Scales:
    """The factors for a block whose units of the circuit are `kept`. A gate is multiplied by
    its factor of `gate_scales`, and its value by `output` over that factor: the unit then
    outputs `output` times what it does in the ideal block. The post-norm divides by `output`
    again, so any `output` is right. It is chosen so that the largest value weight of the
    block is VALUE_RATIO times the largest gate weight. The value weight of a unit is taken
    as `circuit_units` makes it: times the largest output weight of the unit. A value that
    reads no bits counts with its bias."""
    gates = gate_scales(kept)
    gate_weight = np.abs(kept.w_gate[:, 1:]).max(axis=1, initial=0) * gates
    for name, weight in zip(kept.mlp_names, gate_weight / READ_SIZE):
        needed = f"a gate needs a weight of {weight:.3g}, and the limit is {MAX_GATE_WEIGHT:g}"
        cause = "its gate is too near 0 for weights so large: make them smaller, or move the bias"
        assert weight <= MAX_GATE_WEIGHT, f"{name}: {needed}: {cause}"
    on_bits = np.abs(kept.w_value[:, 1:]).max(axis=1, initial=0)  # the largest weight of each value
    own_weight = np.where(on_bits > 0, on_bits, np.abs(kept.w_value[:, 0]))  # or else its bias
    value_weight = own_weight * np.abs(kept.w_out).max(axis=0, initial=0) / gates
    largest_gate = gate_weight.max(initial=0)
    if largest_gate == 0:  # no gate reads bits: count one that is always on, at GATE_MARGIN
        largest_gate = GATE_MARGIN
    largest_value = value_weight.max(initial=0)
    if largest_value == 0:
        # No value has a weight. `output` is then VALUE_RATIO * GATE_MARGIN**2: the gate of an
        # eraser is GATE_MARGIN, and its value weight is VALUE_RATIO times that.
        largest_value = largest_gate / GATE_MARGIN**2
    return Scales(gates, largest_gate, VALUE_RATIO * largest_gate / largest_value)


def on_normed_stream(weights: np.ndarray, bias: np.ndarray, ones: list[int]) -> np.ndarray:
    """Weights [unit, feature] and a bias [unit] on the -1/+1 stream, as weights on the stream
    after the pre-norm, where every bit is -READ_SIZE or +READ_SIZE. The bias becomes weights
    on `ones`, the features that hold ONE, in the shares of `one_sizes`."""
    shares = one_sizes(len(ones)) / len(ones)
    on_ones = np.zeros_like(weights)
    on_ones[:, ones] = np.outer(bias, shares)
    return (weights + on_ones) / READ_SIZE


def circuit_units(kept: Block, scales: Scales, to_lowered: np.ndarray, change: Change) -> Units:
    """The units of the circuit on the lowered features, with the factors of `scales`. The
    largest output weight of each unit becomes 1: its value is multiplied by it instead."""
    count = len(kept.units)
    width = len(change.after)
    largest_out = np.abs(kept.w_out).max(axis=0, initial=0)
    value_scale = scales.output * (largest_out / scales.gates)
    w_gate = np.zeros((count, width))
    w_value = np.zeros((count, width))
    w_out = np.zeros((width, count))
    w_gate[:, to_lowered[1:]] = kept.w_gate[:, 1:] * scales.gates[:, None]
    w_value[:, to_lowered[1:]] = kept.w_value[:, 1:] * value_scale[:, None]
    w_out[to_lowered] = kept.w_out / largest_out
    w_gate = on_normed_stream(w_gate, kept.w_gate[:, 0] * scales.gates, change.ones)
    w_value = on_normed_stream(w_value, kept.w_value[:, 0] * value_scale, change.ones)
    return Units(kept.units, kept.mlp_names, [None] * count, w_gate, w_value, w_out)


def eraser_units(change: Change, scales: Scales) -> Units:
    """One eraser for each signal that leaves its feature, and one for each feature of the
    residue: the second erasure. The gate of an eraser reads only ONE: it is a constant, the
    smallest for which the value weight is at most VALUE_RATIO times the largest gate weight
    of the block, and at least GATE_MARGIN. The value is also multiplied by the size at which
    the block reads the feature, over the size at which the post-norm writes it: the two
    differ only in rows that a caller of `lower` gave."""
    erased = change.released + change.residue
    count = len(erased)
    width = len(change.after)
    gate = max(scales.output / (VALUE_RATIO * scales.largest_gate_weight), GATE_MARGIN)
    w_value = np.zeros((count, width))
    w_value[range(count), erased] = scales.output / gate * change.size_ratio[erased] / READ_SIZE
    w_out = np.zeros((width, count))
    w_out[erased, range(count)] = -1
    w_gate = on_normed_stream(np.zeros((count, width)), np.full(count, gate), change.ones)
    names = [f"erase {change.before[f] or 'what is left'}" for f in erased]
    return Units(names, [None] * count, erased, w_gate, w_value, w_out)


def constant_unit(constant: np.ndarray, ones: list[int], output: float) -> Units:
    """The hidden unit that adds `constant` [feature], times `output`, to the stream. Its gate
    and its value read only ONE. They share the magnitude of the product equally, unless that
    leaves the gate under GATE_MARGIN."""
    largest = np.abs(constant).max()
    gate = max(math.sqrt(output * largest), GATE_MARGIN)
    value = output * largest / gate
    no_weights = np.zeros((1, len(constant)))
    w_gate = on_normed_stream(no_weights, np.array([gate]), ones)
    w_value = on_normed_stream(no_weights, np.array([value]), ones)
    return Units(["constant"], [None], [None], w_gate, w_value, (constant / largest)[:, None])


def dropped_rows(units: Units, overwritten: list[int]) -> np.ndarray:
    """The dropped row of each overwritten feature (see lower_stream.py): its row of w_out,
    but 1 instead of -1 from the eraser of the feature, so that it gets new + old. The squares
    of new - old and new + old add up to the same number whatever the bits are."""
    rows = units.w_out[overwritten]
    for i, f in enumerate(overwritten):
        rows[i, units.erases.index(f)] = 1
    return rows


def normed_block(units: Units, change: Change, mean_square: dict[str, float]) -> NormedBlock:
    """The block of `units`, with its norms. The pre-norm multiplies each feature that a unit
    reads by the RMS of the stream, and by READ_SIZE over `change.read_size`. Every output of
    the block is the same multiple of what the ideal block outputs (see `block_scales`), which
    is -1 or +1 where it writes or erases a bit. So the RMS of its outputs is that multiple
    times a number that the bits do not change. The post-norm multiplies each feature that
    the block changes by that number, and by `change.write_size` (see `Change`)."""
    width = len(change.after)
    read = units.w_gate.any(axis=0) | units.w_value.any(axis=0)
    stream_rms = math.sqrt(stream_square(change.before, change.size_before, mean_square) / width)
    pre_norm = np.where(read, READ_SIZE / change.read_size * stream_rms, 0.0)
    changed = sorted({*change.written, *change.released})
    output_square = sum(mean_square[change.after[f] or change.before[f]] for f in changed)
    # An overwritten feature counts as 1 above. With its dropped row it is 2 + 2 * ratio**2.
    output_square += sum(1 + 2 * change.size_ratio[f] ** 2 for f in change.overwritten)
    touched = changed + change.residue
    post_norm = np.zeros(width)
    post_norm[touched] = change.write_size[touched]
    post_norm = post_norm * math.sqrt(output_square) / math.sqrt(width)
    return NormedBlock(units.names, units.w_gate, units.w_value, units.w_out, pre_norm, post_norm)


def noise_sensitivity(weights: np.ndarray, is_one: np.ndarray, is_value: bool) -> np.ndarray:
    """For each hidden unit, how much noise on `weights`, those of its gate or of its value,
    shows in their sum. The noise changes each weight by a small random fraction of itself,
    as rounding does. The sum then changes by a fraction of itself, whose mean square is that
    of the noise times this number: the sum of the squared weights, over the square of what
    they add up to when the unit is on. The weights on the features of `is_one`, which hold
    ONE, are the bias. A gate that is on is taken as its bias plus the absolute values of its
    weights on bits, and its value as its bias minus them, as in the units of
    lib.mlp_of_terms."""
    bias = weights[:, is_one].sum(axis=1)
    on_bits = np.abs(weights[:, ~is_one]).sum(axis=1)
    when_on = np.abs(np.abs(bias) - on_bits) if is_value else np.abs(bias + on_bits)
    return (weights * weights).sum(axis=1) / np.where(when_on > 0, when_on**2, 1)


def copy_counts(units: Units, change: Change, hidden_units: int | None) -> np.ndarray:
    """How many copies of each hidden unit the block gets when a layer has `hidden_units`
    hidden units: one copy if that is None.

    Every unit gets the same number of copies, as many as fit, up to MAX_COPIES. The room
    that is left goes, one copy at a time, to the group that gains most for each hidden unit
    that a copy of it takes. A group is all the units of one MLP of the circuit, or the
    eraser of one overwritten feature: no second erasure follows that eraser, so what it
    leaves is an error in the new signal. The sensitivity of a unit to noise is 1 for its
    output weight plus `noise_sensitivity` for its gate and for its value. k copies divide it
    by k, so one more copy removes sensitivity / (k * (k + 1))."""
    count = len(units.names)
    available = count if hidden_units is None else hidden_units
    room = available // max(count, 1)  # the copies of every unit that the layer has room for
    counts = np.full(count, min(MAX_COPIES, max(1, room)))
    spare = available - counts.sum()
    is_one = np.array([name == ONE for name in change.before])
    sensitivity = 1 + noise_sensitivity(units.w_gate, is_one, is_value=False)
    sensitivity += noise_sensitivity(units.w_value, is_one, is_value=True)
    members = {}  # the units of each group, by the name of its MLP or by the feature it erases
    for unit, (name, erased) in enumerate(zip(units.mlp_names, units.erases)):
        if name is not None:
            members.setdefault(name, []).append(unit)
        elif erased in change.overwritten:
            members[erased] = [unit]
    groups = [np.array(group) for group in members.values()]
    # The mean sensitivity of a group, not the sum: in a table, few units are on at a time.
    gain = np.array([sensitivity[group].mean() / len(group) for group in groups])
    while groups:
        current = np.array([counts[group[0]] for group in groups])
        fits = np.array([len(group) <= spare for group in groups]) & (current < MAX_COPIES)
        if not fits.any():
            break
        removed = gain / (current * (current + 1))  # by one more copy of each group
        best = int(np.argmax(np.where(fits, removed, 0)))
        counts[groups[best]] += 1
        spare -= len(groups[best])
    return counts


def repeat_units(block: NormedBlock, counts: np.ndarray) -> NormedBlock:
    """`block` with counts[j] copies of hidden unit j: all the units once, then again those
    that have two copies, and so on. The output weights of a unit with k copies are multiplied
    by fewest / k, where fewest is the least k of the block. So the copies of every unit add
    fewest times what the unit added, and the post-norm cancels that factor."""
    order = []
    for copy in range(counts.max(initial=0)):
        order += np.flatnonzero(counts > copy).tolist()
    fewest = counts.min(initial=MAX_COPIES)  # `initial` is for a block that has no units
    w_out = block.w_out * (fewest / counts)
    return replace(
        block,
        units=[block.units[unit] for unit in order],
        w_gate=block.w_gate[order],
        w_value=block.w_value[order],
        w_out=w_out[:, order],
    )

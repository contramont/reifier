"""Step 12, beside Gemma: a circuit and Gemma 3 270M's own weights in one checkpoint.

The model of this file answers as Gemma does, except where the circuit has an answer: there
its next token is the circuit's. A circuit has no answer where its outputs are 1/2 (see
lib.table): it is then silent. The model is a Gemma 3 that is wider and deeper than Gemma 3
270M, because the circuit needs features and layers of its own. The functions follow this order:

- Features. The stream holds the circuit's features, then empty ones if any are needed (see
  `stream_features`), then Gemma's own. A layer has Gemma's hidden units, or more if a block
  of the circuit needs more: Gemma's own are the last.
- Layers. The circuit's layers come first, then Gemma's 18 layers, then one last layer of
  the circuit. In a layer of the circuit the norm weights on Gemma's features are 0, so the
  layer neither reads nor writes them. In a layer of Gemma those on the circuit's are 0.
- Width. Gemma multiplies an embedding by the square root of the number of features, so its
  features are larger here than in Gemma: `widen` changes Gemma's weights to make up for it.
- Norms. An RMSNorm divides by the RMS of the whole stream, the circuit's features and
  Gemma's alike. Before Gemma's layers, Gemma's features hold only an embedding, which is as
  large for every token: the circuit's norm weights allow for it (see `shared_stream_scale`).
  After Gemma's layers their RMS differs from token to token, so the circuit's layers come
  first.
- Carry. The logits come after the last layer, so the outputs of the circuit must get past
  Gemma's layers. To carry a signal is to keep it in its feature through layers that do not
  use it. The last block of the circuit leaves only what is carried: the outputs, their
  silence flags and one ONE, each of size CARRY_SIZE (see `carried_layout`). A silent output
  is 0 and its flag is then 1, so the carried features add the same to the mean square of
  the stream whether the circuit is silent or not: Gemma's norms do not see the difference.
- The last layer. Gemma's stream grows a thousandfold over its layers, and a carried bit is
  then about a 500th of a feature of Gemma. The head of the last layer writes each carried
  bit again, at HEAD_SIZE (see `last_head`). Its block is an amplifier: a block that
  multiplies features by a constant, whatever the RMS of the stream. It makes each output
  OUTPUT_SIZE in the feature where the logits read it (see `amplifier`, `last_block`).
- Logits. The final norm has the weight OUTPUT_NORM on each output. The token whose code
  equals the outputs then gets a logit of about 70, where Gemma's own are under 50 on text.
  The outputs are larger than Gemma's features, so the RMS that the final norm divides by
  depends little on Gemma's stream, and so does that logit. A token whose code differs in d
  of n bits gets 2 * d / n of that logit less: codes must differ in many bits.
- Silence. Where the circuit is silent, the silence flags close the gates of the amplifier
  that read the outputs. The last layer then adds nothing to the outputs, which are 0, and
  the logits are close to those of Gemma alone. They are not equal: Gemma's norms divide by
  an RMS that counts what is carried too.
"""

import math
import re
from dataclasses import dataclass, replace

import numpy as np

from . import checkpoint, gemma, lower_stream
from .layout import ONE, Layout
from .lower import OUT_WEIGHT, Lowered, NormedBlock, lower
from .lower_stream import BIT_SIZE, KEY_VALUE_WEIGHT, READ_SIZE, ROPE_PAIRS, Attention
from .transformer import Transformer

GEMMA = "unsloth/gemma-3-270m"  # the model beside the circuit: the next two numbers are its own
GEMMA_RMS_BEFORE = 1.03  # the RMS of Gemma's stream before its first layer, at every token
GEMMA_RMS_AFTER = 1400.0  # the median RMS of Gemma's stream after its last layer, on English text
CARRY_SIZE = 3.0  # a carried bit is -3 or +3: larger than a feature of Gemma where it is made
HEAD_SIZE = 240.0  # a bit as the head of the last layer writes it: 80 times CARRY_SIZE
# The next two are what a pre-norm of the last layer makes of a bit, where the RMS of the stream
# is GEMMA_RMS_AFTER. The norm weights are then 117 and 17.5: Gemma's own reach 234 and 70.
HEAD_READ_SIZE = 0.25  # a carried bit, as the head reads it
BLOCK_READ_SIZE = 3.0  # a bit of HEAD_SIZE, as the block reads it: an amplifier gate is -3 or +3
SILENCE_WEIGHT = 3.0  # silence takes 3 bits off each gate of the amplifier: GELU is 0 at -6 or less
AMPLIFIER_COPIES = 32  # the most pairs of hidden units for one feature: weight noise averages out
OUTPUT_SIZE = 8000.0  # a bit of an output after the last layer: 5 times a feature of Gemma there
# OUTPUT_SIZE is for a stream of this many features: it grows with the square root of their number.
REFERENCE_FEATURES = 896
OUTPUT_NORM = 5.0  # the final norm's weight on an output: a logit is at most 2 * 5 * sqrt(bits)

LAYER_NUMBER = re.compile(r"(?<=layers\.)\d+")  # in the name of a tensor: the number of its layer
# The norms that Gemma reads the stream through: before its attention, its block and the logits.
READS_STREAM = re.compile("input_layernorm|pre_feedforward_layernorm|^norm")
WRITES_STREAM = re.compile("o_proj|down_proj")  # the projections whose output a post-norm adds


@dataclass
class WideGemma:
    """A Gemma that is wider and deeper than GEMMA: what a checkpoint of it holds."""

    config: dict  # the contents of config.json
    shapes: dict[str, list[int]]  # the shape of each tensor, by its name in the checkpoint
    weights: gemma.Tensors  # the tensors that are not all zeros, by name in the text model
    names: dict  # for names.json: the rows of the layout, and the hidden units of each layer

    def write(self, out_dir):
        """Write the checkpoint to the directory `out_dir` (see checkpoint.write)."""
        tensors = self.weights.by_checkpoint_name()
        checkpoint.write(out_dir, self.config, self.shapes, tensors, self.names)


def stream_features(lowered: Lowered, config: dict) -> int:
    """The number of features of the stream that `lowered` shares with the Gemma of `config`.
    The features after the circuit's are at least as many as the dropped rows of a block,
    which become their rows (see gemma.write_block). And the total is a multiple of the heads
    of a layer, which `transformers` requires of a model."""
    width = len(lowered.layout[0])
    dropped_rows = max(len(block.w_out) for block in lowered.blocks) - width
    features = width + max(config["hidden_size"], dropped_rows)
    heads = config["num_attention_heads"]
    return math.ceil(features / heads) * heads


def renumbered(name: str, shift: int) -> str:
    """The name of a tensor with the number of its layer `shift` larger, if it is of a layer."""
    return LAYER_NUMBER.sub(lambda number: str(int(number[0]) + shift), name)


def wide_shapes(gemma_weights: dict, config: dict, wide_config: dict, first_layer: int) -> dict:
    """The shape of every tensor of the model of the config `wide_config`, by its name in a
    checkpoint: first the new layers, each with the tensors of a layer of Gemma, then Gemma's
    tensors, whose layers start at `first_layer`. An axis as long as Gemma has features, or
    hidden units, becomes as long as `wide_config` has them. In GEMMA no other kind of axis
    has either length."""
    longer = {
        config["hidden_size"]: wide_config["hidden_size"],
        config["intermediate_size"]: wide_config["intermediate_size"],
    }
    gemma_layers = range(first_layer, first_layer + len(config["layer_types"]))
    all_layers = range(len(wide_config["layer_types"]))
    new_layers = [layer for layer in all_layers if layer not in gemma_layers]
    of_layer_0 = [name for name in gemma_weights if LAYER_NUMBER.findall(name) == ["0"]]
    shapes = {}
    for layer in new_layers:
        for name in of_layer_0:
            shapes[renumbered(name, layer)] = gemma_weights[name].shape
    for name, tensor in gemma_weights.items():
        shapes[renumbered(name, first_layer)] = tensor.shape
    return {name: [longer.get(axis, axis) for axis in shape] for name, shape in shapes.items()}


def widen(gemma_weights, config, features, hidden_units, new_before=0, new_after=0) -> WideGemma:
    """A Gemma in a wider, deeper model. `gemma_weights` is Gemma's tensors by their names in
    its checkpoint (see checkpoint.read_weights), and `config` its config. The model has
    `features` features, `hidden_units` hidden units a layer, `new_before` new layers before
    Gemma's own and `new_after` after them. Gemma's weights take the last entries of each
    axis. All else is zeros, for the caller to write, and gemma.finish is still to be called.

    Gemma's features are `scale` times what they were, with `scale` the square root of all
    features over Gemma's own. Their mean square over all features is what it was over
    Gemma's, so an RMSNorm gives `scale` times what it gave. A norm that Gemma reads the
    stream through therefore has its weights divided by `scale`. And a projection that
    writes to the stream is multiplied by `scale`: the post-norm after it sees the mean
    square that it saw, and adds `scale` times what it added."""
    new = ["sliding_attention"]  # the RoPE of the circuit's heads is that of a sliding layer
    layer_types = new * new_before + config["layer_types"] + new * new_after
    wide_config = config | {
        "hidden_size": features,
        "intermediate_size": hidden_units,
        "layer_types": layer_types,
        "num_hidden_layers": len(layer_types),
    }
    shapes = wide_shapes(gemma_weights, config, wide_config, new_before)
    weights = gemma.Tensors(shapes)
    scale = math.sqrt(features / config["hidden_size"])
    for name, tensor in gemma_weights.items():
        name = renumbered(name.removeprefix(weights.prefix), new_before)
        if name.endswith("norm.weight"):
            tensor = gemma.norm_offset(config) + tensor  # what the norm multiplies by
            if READS_STREAM.search(name):
                tensor = tensor / scale
        elif WRITES_STREAM.search(name):
            tensor = tensor * scale
        last_entries = tuple(slice(-length, None) for length in tensor.shape)
        weights[name][last_entries] = tensor
    return WideGemma(wide_config, shapes, weights, {"features": [], "hidden": []})


def shared_stream_scale(circuit_square: float, features: int) -> float:
    """The factor on a pre-norm weight of the circuit, in a stream of `features` features
    that it shares with Gemma. The lowering took the squares of the stream to add up to
    `circuit_square`, those of the circuit's features. Before Gemma's layers, Gemma's features
    add features * GEMMA_RMS_BEFORE**2 (see `widen`), so an RMSNorm divides by a larger RMS."""
    gemma_square = features * GEMMA_RMS_BEFORE**2
    return math.sqrt((circuit_square + gemma_square) / circuit_square)


def write_circuit(weights: gemma.Tensors, lowered: Lowered, config: dict):
    """Write the embedding, the attention and the blocks of `lowered` into the first features
    and the first layers of `weights`, the tensors of the model of `config`. `norm_scale`
    corrects every norm weight for the number of features (see lower.Lowered), and
    `shared_stream_scale` corrects each pre-norm for what Gemma's features hold. A post-norm
    needs only the first: it divides by the RMS of what its own sublayer outputs."""
    width = len(lowered.layout[0])
    features = config["hidden_size"]
    norm_scale = math.sqrt(width / features)
    embedded = lowered.embedding.shape[1]
    weights["embed_tokens.weight"][:, :embedded] = lowered.embedding / math.sqrt(features)
    embedded_square = float((lowered.embedding[0] ** 2).sum())  # the same for every token
    pre_norm = lowered.attention.pre_norm * shared_stream_scale(embedded_square, features)
    attention = replace(lowered.attention, pre_norm=pre_norm)
    gemma.write_attention(weights, 0, attention, config, norm_scale)
    for layer, block in enumerate(lowered.blocks):
        row = lowered.layout[layer]  # what the stream holds before the block
        square = lower_stream.stream_square(row, lowered.sizes[layer], lowered.mean_square)
        pre_norm = block.pre_norm * shared_stream_scale(square, features)
        gemma.write_block(weights, layer, replace(block, pre_norm=pre_norm), norm_scale)


def carried_layout(model: Transformer) -> Layout:
    """The lowered layout of `model`, with a last row that holds what is carried: the outputs,
    their silence flags and one ONE. The ONE features of the rows before have other sizes
    (see lower_stream.one_sizes), so the last block erases them and makes a new ONE: in the
    last feature that is empty before and after that block, or else in one more. The row
    also gets as many empty features after the code as there are outputs, for `last_head`."""
    circuit = model.circuit
    bits = len(circuit.outputs)
    rows = lower_stream.lowered_layout(model)
    made_earlier = set(circuit.outputs).intersection(rows[-2])
    # Only the last block writes at CARRY_SIZE, and a signal keeps the size that it is made at.
    rule = f"its last block must make every output, and {sorted(made_earlier)} are made earlier"
    assert not made_earlier, f"{circuit.name}: {rule}"
    width = len(rows[0])
    empty_twice = [f for f in range(width) if rows[-2][f] is None and rows[-1][f] is None]
    if not empty_twice:
        empty_twice = [width]
        rows = [row + [None] for row in rows]
    rows[-1] = [None if name == ONE else name for name in rows[-1]]
    rows[-1][empty_twice[-1]] = ONE
    empty = sum(name is None for name in rows[-1][1 + bits :])
    return [row + [None] * max(0, bits - empty) for row in rows]


def lower_beside(model: Transformer, hidden_units: int | None = None) -> Lowered:
    """Lower `model` as lower.lower does with `hidden_units`, but with the rows of
    `carried_layout` and the last row at CARRY_SIZE. This file does not use its final norm."""
    rows = carried_layout(model)
    sizes = lower_stream.feature_sizes(rows)
    sizes[-1] *= CARRY_SIZE / BIT_SIZE
    return lower(model, hidden_units, rows, sizes)


class Carried:
    """What the last row of a layout of `lower_beside` holds, and where `last_head` writes it."""

    def __init__(self, lowered: Lowered):
        row = lowered.layout[-1]
        bits = len(lowered.model.circuit.outputs)
        self.width = len(row)
        self.outputs = list(range(1, 1 + bits))  # the logits read the outputs in these features
        self.flags_and_one = [f for f in range(1 + bits, len(row)) if row[f]]
        self.one = row.index(ONE)
        empty = [f for f in range(1 + bits, len(row)) if row[f] is None]
        self.spare = empty[:bits]  # spare features: the head writes output i again in spare[i]
        # The mean over inputs of the sum of the squares of all that is carried, for bits of size 1.
        self.square = sum(lowered.mean_square[row[f]] for f in self.outputs + self.flags_and_one)
        self.closed_by = silence_flags(lowered, self.spare)


def silence_flags(lowered: Lowered, spare: list[int]) -> dict[int, list[int]]:
    """For the spare feature spare[i] of each output i that can be 1/2: the features that
    close the gates of its pairs of hidden units in the last block (see `amplifier`). They
    are the features of the silence flags of all such outputs, because the mean of many flags
    has a smaller error than one flag. So where one output is 1/2, every output must be. The
    flag of an output is the other output of the MLP that makes it (see lib.table)."""
    circuit = lowered.model.circuit
    flags = {}  # by the number of an output that can be 1/2: the feature of its flag
    for number, name in enumerate(circuit.outputs):
        maker = next(m for m in circuit.mlps if name in m.outputs)
        siblings = [other for other in maker.outputs if other not in circuit.outputs]
        if siblings and not np.isclose(lowered.mean_square[name], 1):
            flags[number] = lowered.layout[-1].index(siblings[-1])
    values = lower_stream.sample_values(circuit, lowered.model.codes)
    is_half = np.array([np.isclose(values[name], 1 / 2) for name in circuit.outputs])
    together = is_half.all(axis=0) | ~is_half.any(axis=0)
    assert together.all(), f"{circuit.name}: where one output is 1/2, every output must be 1/2"
    return {spare[number]: sorted(flags.values()) for number in flags}


def last_head(carried: Carried, features: int) -> Attention:
    """The attention of the last layer, in a stream of `features` features: one head that
    attends to its own token. A block could read a carried bit only through a pre-norm weight
    of 1400, where Gemma's reach 70. The head reads it as HEAD_READ_SIZE, and its post-norm, which
    divides by the RMS of what the head alone writes, makes it HEAD_SIZE. A silence flag and ONE
    grow where they are. An output is written to a spare feature instead. A head has no gate
    to close: in the output's own feature, where the logits read it, the head would also
    multiply the small error of a silent output, which should be 0."""
    read = carried.outputs + carried.flags_and_one
    written = carried.spare + carried.flags_and_one
    count = len(read)
    # Keys and queries read ONE, as in lower_stream.lower_attention, for no positions back.
    # Gemma normalizes both, so it does not matter that ONE is not READ_SIZE after this pre-norm.
    w_key = np.zeros((2 * ROPE_PAIRS, carried.width))
    w_key[:ROPE_PAIRS, carried.one] = KEY_VALUE_WEIGHT
    query_weight = lower_stream.top_score() / ROPE_PAIRS / (KEY_VALUE_WEIGHT * READ_SIZE**2)
    w_query = np.zeros((1, 2 * ROPE_PAIRS, carried.width))
    w_query[0, :, carried.one] = query_weight * lower_stream.lookback_query(0)
    w_value = np.zeros((count, carried.width))
    w_value[range(count), read] = KEY_VALUE_WEIGHT
    w_out = np.zeros((1, carried.width, count))
    w_out[0, written, range(count)] = 1
    pre_norm = np.zeros(carried.width)
    pre_norm[read] = HEAD_READ_SIZE * GEMMA_RMS_AFTER / CARRY_SIZE
    output_rms = math.sqrt(carried.square / features)  # of what it writes, for bits of size 1
    post_norm = np.zeros(carried.width)
    post_norm[carried.spare] = HEAD_SIZE * output_rms
    # The flags and ONE already hold CARRY_SIZE.
    post_norm[carried.flags_and_one] = (HEAD_SIZE - CARRY_SIZE) * output_rms
    return Attention(w_query, w_key, w_value, w_out, pre_norm, post_norm)


def amplifier(width: int, adds_to: dict, one: int, closed_by: dict, copies: int) -> NormedBlock:
    """An amplifier of `width` features (see the top of this file). Let z be a feature and u
    the feature `one`, which holds ONE, both after the pre-norm. A pair of hidden units
    computes

        gelu(z) * u - gelu(-z) * u  =  z * u

    for every z: one unit alone would need z to be far from 0. The RMS of the stream divides
    z and u alike, and the post-norm cancels it. Each feature f of `adds_to` gets `copies`
    such pairs, with f as z: between them they add z * u / 4 to each feature of adds_to[f].
    `closed_by[f]`, if f has one, lists the features that close the gates of those pairs.
    SILENCE_WEIGHT times their mean is taken off both gates, so the pairs add nothing where
    those features are as large as ONE. The caller sets the norm weights."""
    names = []
    w_gate = np.zeros((2 * len(adds_to), width))
    w_value = np.zeros((2 * len(adds_to), width))
    w_out = np.zeros((width, 2 * len(adds_to)))
    for feature, written in adds_to.items():
        closing = closed_by.get(feature, [])
        for sign, symbol in ((1, "+"), (-1, "-")):
            unit = len(names)
            names.append(f"amplify f{feature}{symbol}")
            w_gate[unit, feature] = sign
            if closing:
                w_gate[unit, closing] = -SILENCE_WEIGHT / len(closing)
            w_value[unit, one] = 1 / 2 / copies  # half of u, shared by the copies
            w_out[written, unit] = sign * OUT_WEIGHT
    w_gate = np.tile(w_gate, (copies, 1))  # every pair, `copies` times
    w_value = np.tile(w_value, (copies, 1))
    w_out = np.tile(w_out, (1, copies))
    return NormedBlock(names * copies, w_gate, w_value, w_out, np.zeros(width), np.zeros(width))


def last_block(carried: Carried, features: int, hidden_units: int) -> NormedBlock:
    """The MLP block of the last layer, in a stream of `features` features and a layer of
    `hidden_units` hidden units: an amplifier of what `last_head` wrote. From the spare
    feature of each output, it makes the output large where the logits read it. From ONE, it
    adds a constant to every spare feature. The post-norm multiplies those by 0, as it does a
    dropped row: the constant only keeps the RMS that the post-norm divides by away from 0
    where the circuit is silent."""
    bits = len(carried.outputs)
    adds_to = {spare: [output] for spare, output in zip(carried.spare, carried.outputs)}
    adds_to[carried.one] = carried.spare
    # The room is counted as if the silence flags had pairs too, so up to half of it stays unused.
    room = hidden_units // (2 * (bits + len(carried.flags_and_one)))
    copies = max(1, min(AMPLIFIER_COPIES, room))
    block = amplifier(carried.width, adds_to, carried.one, carried.closed_by, copies)
    read = carried.spare + carried.flags_and_one  # each is HEAD_SIZE
    block.pre_norm[read] = BLOCK_READ_SIZE * GEMMA_RMS_AFTER / HEAD_SIZE
    output_size = OUTPUT_SIZE * math.sqrt(features / REFERENCE_FEATURES)
    # Where the circuit answers, an output gets -1 or +1 and a spare feature 1, times one factor.
    output_rms = math.sqrt((bits + len(carried.spare)) / features)
    block.post_norm[carried.outputs] = (output_size - CARRY_SIZE) * output_rms
    return block


def wide_gemma(lowered: Lowered, gemma_weights: dict, config: dict) -> WideGemma:
    """The wider, deeper Gemma that runs `lowered`, a model of `lower_beside`, beside a Gemma.
    `gemma_weights` and `config` are that Gemma's tensors and its config, as `widen` takes
    them."""
    model = lowered.model
    name = model.circuit.name
    carries = lowered.layout == carried_layout(model) and lowered.sizes[-1].max() == CARRY_SIZE
    assert carries, f"{name} must be lowered by lower_beside"
    assert len(model.codes) == config["vocab_size"], f"{name} needs a code for each token of Gemma"
    heads = config["num_attention_heads"]
    assert len(model.heads) <= heads, f"{name} reads more tokens than a layer of Gemma has heads"
    carried = Carried(lowered)
    circuit_layers = len(lowered.blocks)
    gemma_layers = len(config["layer_types"])
    features = stream_features(lowered, config)
    hidden_units = max(config["intermediate_size"], *[len(block.units) for block in lowered.blocks])
    wide = widen(
        gemma_weights, config, features, hidden_units, new_before=circuit_layers, new_after=1
    )
    write_circuit(wide.weights, lowered, wide.config)
    last_layer = circuit_layers + gemma_layers
    # The norms of the last layer are made for the whole stream, so their factor is 1.
    head = last_head(carried, features)
    gemma.write_attention(wide.weights, last_layer, head, wide.config, norm_scale=1.0)
    last = last_block(carried, features, hidden_units)
    gemma.write_block(wide.weights, last_layer, last, norm_scale=1.0)
    wide.weights["norm.weight"][carried.outputs] = OUTPUT_NORM
    gemma.finish(wide.weights, wide.config)
    # After the circuit's blocks, no layer changes which signal a feature holds.
    rows = lowered.layout + [lowered.layout[-1]] * (gemma_layers + 1)
    hidden = [block.units for block in lowered.blocks] + [[]] * gemma_layers + [last.units]
    wide.names = {"features": rows, "hidden": hidden}
    return wide


def save(model: Transformer, out_dir, hidden_units: int | None = None, check_on=None):
    """Save the ideal transformer `model` beside GEMMA, as one checkpoint in the directory
    `out_dir`. The first call downloads Gemma's weights. The circuit must obey five rules:

    - Every token of Gemma has a code.
    - Any two codes differ in many bits (see Logits, at the top), as in examples/alongside.py.
    - The circuit reads at most as many tokens as a layer of Gemma has heads.
    - The last block of the circuit makes every output.
    - Where one output is 1/2, every output is 1/2. This rule is checked only on the inputs
      of lower_stream.sample_values.

    A block of the circuit gets copies of its hidden units, up to `hidden_units` in all: by
    default the number that a layer of Gemma has.

    `check_on` is a list of tokens to check the weights on first (see gemma.check_weights).
    The next token is checked only where the circuit answers, so `check_on` should hold such
    tokens. If only the next token is wrong, the codes probably differ in too few bits."""
    config = checkpoint.metadata(GEMMA).config
    if hidden_units is None:
        hidden_units = config["intermediate_size"]
    lowered = lower_beside(model, hidden_units)
    wide = wide_gemma(lowered, checkpoint.read_weights(GEMMA), config)
    if check_on is not None:
        gemma.check_weights(wide.weights, wide.config, lowered, check_on)
    wide.write(out_dir)

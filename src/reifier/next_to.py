"""Step 13, next to Gemma: each block of a circuit in one of Gemma's layers, beside Gemma's own MLP.

beside.py puts the circuit's layers in front of Gemma's. This file puts each block of a circuit in
one of Gemma's last layers instead, on hidden units of its own, so that the model is two layers
deeper than Gemma, whatever the circuit. Again its next token is the circuit's where the circuit
answers, and else Gemma's. For a circuit of D blocks beside a Gemma of L layers, the layers are:

- the copy, a layer of the circuit's own, whose heads copy the codes of the tokens that the
  circuit reads, as in beside.py, but small;
- Gemma's first L - D layers;
- the lift, a layer of the circuit's own, whose head copies what the copy wrote, large;
- Gemma's last D layers, each with a block of the circuit beside Gemma's MLP.

The circuit has the first features of the stream, and Gemma's weights do not read them. But a norm
divides by the RMS of the whole stream, or of all that an MLP writes, and in Gemma's layers Gemma's
features are most of both: so Gemma computes nearly what it computes alone. Its attention does not
see the circuit at all, as it normalizes its queries and keys, and its post-norm divides out any
factor on its values. Its MLPs see the little that the circuit adds to the RMS.

The circuit, though, sees an RMS that it cannot know, as it differs from token to token. A block
reads its bits some factor larger or smaller than at the median token, and writes them another. A
block of lower.py would square the factor, as its gates and values both read the bits, and the bits
would be lost after a few blocks. The blocks of this file keep the factor from carrying over:

- Steps. An MLP of the circuit becomes the cubes of its truth table (see lib.cubes), and each
  cube a pair of hidden units, a step: gelu(high - drop) * v - gelu(low - drop) * v. high, low
  and v read the reference: ONE features that the lift wrote, whose size in the stream is known.
  The drop reads the bits of the cube's term: it is 0 where they hold, and STEP_GAIN times the
  factor for each that does not. So the step gives (high - low) * v where the term holds, and 0
  where it does not, whatever the factor: what a block writes depends on the RMS of its own
  layer, and on none before it. Both of its gates are small, so bf16, which keeps 8 bits of each,
  keeps their difference too.
- Rows. A gate reads only what the block before wrote: a row of bits and the ONE of that row,
  which came out of one post-norm and share its factor. A signal that a later block reads is
  copied by each block in between. A layer of Gemma cannot erase a feature exactly, so each row
  has features of its own, and the stream is as wide as all the rows.
- Loud. The last block is beside Gemma's last MLP, after Gemma's last attention, and writes the
  outputs into the features of the code. Where the circuit answers, it writes them LOUD: they are
  then most of the RMS that its post-norm and the final norm divide by, so their size there is
  known, and the logits are the circuit's. Where it is silent, the outputs are 1/2, which is 0 in
  the stream (see lib.table): the block writes nothing, and the logits are nearly Gemma's own.

The copy is before Gemma's layers, where every token's stream is as large. In later layers,
<bos> and a few other tokens have a stream ten times as large as the rest, and a head would copy
their codes ten times too small. The lift reads only its own token, so that does not matter to it.

`measure` gives the size of Gemma's stream at each layer, at the median token of some text. In
Gemma 3 270M, on English text, the factor is between 1/3 and 3 at 99 tokens in 100, and the
constants leave room for that. Most of the other tokens are "▁", a lone space, where Gemma's stream
is ten times as large. Random tokens give a smaller stream than text: `words` makes text of tokens.
"""

import math
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from . import beside, gemma
from .circuit import MLP, Circuit
from .layout import ONE, depths
from .lib import cubes
from .lower import OUT_WEIGHT, NormedBlock, on_normed_stream
from .lower_stream import (
    BIT_SIZE,
    KEY_VALUE_WEIGHT,
    ONE_FEATURES,
    READ_SIZE,
    ROPE_PAIRS,
    Attention,
    check_sampled_values,
    lookback_query,
    lower_attention,
    lower_embedding,
    lowered_layout,
    one_sizes,
    sample_values,
    top_score,
)
from .transformer import Transformer

# A step is gelu(STEP_HIGH - drop) - gelu(STEP_LOW - drop), times its value, at the median token.
STEP_HIGH = 9.0
STEP_LOW = 6.0  # GELU is within 0.001 of its input from 3.5 on: where the reference reads 0.58
STEP_GAIN = 48.0  # the drop for each literal that fails: 9 + 3.5 where the bits read 0.26
STEP_READ = 2 * READ_SIZE  # what a step reads a feature as: then no gate weight is over 2
SHARE = 0.1  # a bit, as a share of the RMS of the stream where the next block reads it
REFERENCE = 0.2  # the reference ONEs, as a share of the RMS of the stream at the last block
QUIET = 0.1  # what a step outputs, as a share of the RMS of what Gemma's MLP outputs there
LOUD = 30.0  # the RMS of what the last block outputs, where it answers, over that of Gemma's MLP
CODE_SIZE = 0.5  # a bit of the code, and a ONE, in the embedding: Gemma's features there are 1.03
MAX_STEP_INPUTS = 12  # an MLP that reads more signals would have too many cubes to list


@dataclass
class Scales:
    """Gemma's stream at each layer: medians over the positions of some tokens."""

    before_attention: np.ndarray  # [layer]: the RMS of the stream before the attention
    before_mlp: np.ndarray  # [layer]: the RMS of the stream before the MLP
    mlp: np.ndarray  # [layer]: the RMS of what the MLP outputs, before its post-norm


def words(vocabulary: dict[str, int], text: str) -> list[int]:
    """The words of `text` that are tokens of `vocabulary` (see gemma.vocabulary), with a space
    before each: text as Gemma's tokenizer splits it, but for the rarer words."""
    return [vocabulary["▁" + word] for word in text.split() if "▁" + word in vocabulary]


def measure(gemma_weights: dict, config: dict, tokens) -> Scales:
    """The medians of `Scales`, over every position but the first, of the Gemma of
    `gemma_weights` (see checkpoint.read_weights) and `config` on `tokens`. In a model as wide as
    beside.widen makes it, Gemma's stream has the same RMS. Each is rounded to 3 digits, so that
    the weights do not depend on the order in which a machine adds numbers."""
    embedding = next(name for name in gemma_weights if name.endswith("embed_tokens.weight"))
    prefix = embedding.removesuffix("embed_tokens.weight")
    weights = {name.removeprefix(prefix): tensor for name, tensor in gemma_weights.items()}
    x = weights["embed_tokens.weight"][np.asarray(tokens)] * math.sqrt(config["hidden_size"])

    def median_rms(stream: np.ndarray) -> float:
        return float(f"{np.median(np.sqrt((stream[1:] ** 2).mean(axis=1))):.3g}")

    rows = []
    for layer in range(len(config["layer_types"])):
        before_attention = median_rms(x)
        x = x + gemma.run_attention(weights, config, layer, x)
        output = gemma.mlp_output(weights, config, layer, x)
        rows.append([before_attention, median_rms(x), median_rms(output)])
        x = x + gemma.run_block(weights, config, layer, x)
    return Scales(*np.array(rows).T)


@dataclass
class Step:
    """One part of a block: an MLP of the circuit, a copy of a signal, or a ONE."""

    name: str
    inputs: list[str]  # the signals that it reads, each once, in the row before
    outputs: list[str]  # the signals that it makes, in the block's own row
    cubes: dict[tuple[int, ...], tuple[float, ...]]  # {signs of a term: each output's value there}


def truth_cubes(m: MLP, outputs: list[str]) -> Step:
    """`m` as a Step that makes `outputs`: the cubes of its truth table (see lib.cubes)."""
    inputs = list(dict.fromkeys(m.inputs))  # an MLP may read a signal twice
    rule = f"next to Gemma, an MLP may read at most {MAX_STEP_INPUTS} signals"
    assert len(inputs) <= MAX_STEP_INPUTS, f"{m.name} reads {len(inputs)} signals: {rule}"
    cases = np.array(list(product([0, 1], repeat=len(inputs))), int).reshape(-1, len(inputs))
    values = m(cases[:, [inputs.index(name) for name in m.inputs]])
    halves = np.rint(2 * values[:, [m.outputs.index(name) for name in outputs]]) / 2
    return Step(m.name, inputs, outputs, cubes(tuple(map(tuple, halves.tolist()))))


def steps_of(circuit: Circuit) -> list[list[Step]]:
    """The steps of each block of `circuit`. Block d makes the signals of depth d that a block
    reads or that are outputs, copies each signal of less depth that a later block reads, and,
    unless it is the last, makes a ONE."""
    depth_of = depths(circuit)
    last = max(depth_of.values())
    made_earlier = [name for name in circuit.outputs if depth_of[name] < last]
    rule = f"its last block must make every output, and {made_earlier} are made earlier"
    assert not made_earlier, f"{circuit.name}: {rule}"
    last_read = {}  # the depth of the last block that reads each signal
    for m in circuit.mlps:
        for name in m.inputs:
            last_read[name] = max(last_read.get(name, 0), depth_of[m.outputs[0]])
    blocks = [[] for _ in range(last)]
    for m in circuit.mlps:
        kept = [name for name in m.outputs if name in last_read or name in circuit.outputs]
        blocks[depth_of[m.outputs[0]] - 1].append(truth_cubes(m, kept))
    for name, read_at in last_read.items():
        for depth in range(depth_of[name] + 1, read_at):
            copy = Step(f"copy {name}", [name], [name], {(1,): (1,), (-1,): (0,)})
            blocks[depth - 1].append(copy)
    for steps in blocks[:-1]:
        steps.append(Step(ONE, [], [ONE], {(): (1,)}))
    return blocks


class Rows:
    """Where the stream holds each row: row 0 is what the lift leaves, the inputs and the
    reference, and row d what block d writes. Each row has features of its own, but the last,
    whose outputs are in the features of the code. `layout` [row][feature] is for people: a
    feature still holds what it held, after a later row is written."""

    def __init__(self, model: Transformer, blocks: list[list[Step]]):
        first_row = lowered_layout(model)[0]  # ONE, the code, the other ONEs, the inputs
        self.code = list(range(1, 1 + len(model.circuit.outputs)))
        self.feature_of = [{name: f for f, name in enumerate(first_row) if name not in (None, ONE)}]
        self.ones = [[f for f, name in enumerate(first_row) if name == ONE]]  # row 0: the reference
        self.layout = [first_row]
        for steps in blocks[:-1]:
            names = dict.fromkeys(name for step in steps for name in step.outputs if name != ONE)
            start = len(self.layout[-1])
            self.feature_of.append({name: start + i for i, name in enumerate(names)})
            self.ones.append(list(range(start + len(names), start + len(names) + ONE_FEATURES)))
            self.layout.append(self.layout[-1] + list(names) + [ONE] * ONE_FEATURES)
        outputs = model.circuit.outputs
        self.feature_of.append(dict(zip(outputs, self.code)))
        self.ones.append([])
        self.layout.append(self.layout[-1][:1] + outputs + self.layout[-1][1 + len(outputs) :])
        self.width = len(self.layout[-1])
        self.layout = [row + [None] * (self.width - len(row)) for row in self.layout]
        self.reference = self.ones[0]
        self.lifted = [*self.feature_of[0].values(), *self.reference]  # what the lift reads


def lift(rows: Rows, before: np.ndarray, after: np.ndarray) -> Attention:
    """The attention of the lift: one head that attends to its own token, as that of
    beside.last_head, and copies each feature of the inputs and the reference onto itself, so
    that it holds `after` [feature], not `before`. The norms are for `write_own_attention`."""
    ones, count = rows.reference, len(rows.lifted)
    w_key = np.zeros((2 * ROPE_PAIRS, rows.width))
    w_key[:ROPE_PAIRS, ones] = KEY_VALUE_WEIGHT
    key_size = KEY_VALUE_WEIGHT * READ_SIZE * len(ones)  # as in lower_stream.lower_attention
    query_weight = top_score() / ROPE_PAIRS / key_size / READ_SIZE / len(ones)
    w_query = np.zeros((1, 2 * ROPE_PAIRS, rows.width))
    w_query[0][:, ones] = query_weight * lookback_query(0)[:, None]
    w_value = np.zeros((count, rows.width))
    w_value[range(count), rows.lifted] = KEY_VALUE_WEIGHT
    w_out = np.zeros((1, rows.width, count))
    w_out[0, rows.lifted, range(count)] = np.sign(after - before)[rows.lifted]
    no_norm = np.zeros(rows.width)
    return Attention(w_query, w_key, w_value, w_out, no_norm, no_norm)


def write_own_attention(wide, layer: int, attention, before, after, gemma_rms, read_size):
    """Write `attention` into `layer` of `wide`, a layer of the circuit's own, with norms for a
    stream whose features of the circuit are `before` [feature] and whose features of Gemma have
    the RMS `gemma_rms`. The pre-norm makes each feature that the heads read `read_size`, and the
    post-norm makes them write `after` - `before`, as lower_stream.with_norms does. The circuit's
    sizes are the same at every token, and so is Gemma's RMS before its first layer. Before the
    lift it is not, but a head does not mind: it normalizes its query and key, and its
    post-norm divides out any factor on its value."""
    features = wide.config["hidden_size"]
    stream_rms = math.sqrt(((before**2).sum() + features * gemma_rms**2) / features)
    read = attention.w_value.any(axis=0) | attention.w_key.any(axis=0)
    written = attention.w_out.any(axis=(0, 2))
    pre_norm = np.zeros(len(before))
    pre_norm[read] = read_size * stream_rms / before[read]
    post_norm = np.abs(after - before) * written * math.sqrt(written.sum() / features)
    attention = replace(attention, pre_norm=pre_norm, post_norm=post_norm)
    gemma.write_attention(wide.weights, layer, attention, wide.config, norm_scale=1.0)


def step_block(steps, d: int, rows: Rows, size, scales: tuple, features: int, loud: bool):
    """Block d, of `steps`, beside an MLP of Gemma whose stream has the RMS R before it, and which
    outputs M, with (R, M) = `scales`, in a stream of `features` features. Row d - 1 and the
    reference are read, and row d written, at their sizes of `size` [feature]. If `loud`, the
    block is the last (see the top of this file)."""
    width = rows.width
    before, after = rows.feature_of[d - 1], rows.feature_of[d]

    def on_stream(weights, bias, ones):  # lower.on_normed_stream, where features are STEP_READ
        return on_normed_stream(weights, np.array([bias]), ones)[0] * READ_SIZE / STEP_READ

    no_weights = np.zeros((1, width))
    high = on_stream(no_weights, STEP_HIGH, rows.reference)
    low = on_stream(no_weights, STEP_LOW, rows.reference)
    names, gates, outs = [], [], []
    for step in steps:
        written = rows.ones[d] if step.outputs == [ONE] else [after[name] for name in step.outputs]
        for signs, value in step.cubes.items():
            # Half the gain for each literal, less half the gain for each on the row's ONE: the
            # drop is 0 where the literals hold, and the gain for each that does not.
            on_bits = np.zeros((1, width))
            on_bits[0, [before[name] for name in step.inputs]] = STEP_GAIN / 2 * np.array(signs)
            literals = np.count_nonzero(signs)
            rise = on_stream(on_bits, -STEP_GAIN / 2 * literals, rows.ones[d - 1])  # minus the drop
            out = np.zeros(width)
            out[written] = OUT_WEIGHT * (2 * np.array(value) - 1)
            held = [f"{'+-'[sign < 0]}{name}" for sign, name in zip(signs, step.inputs) if sign]
            term = f"{step.name} if {' '.join(held) or 'always'}"
            names += [term + ", high", term + ", low"]
            gates += [high + rise, low + rise]
            outs += [out, -out]
    stream_rms, mlp_rms = scales
    written = [*after.values(), *rows.ones[d]]
    if loud:  # all its outputs together are LOUD times all that Gemma's MLP outputs
        output = LOUD * mlp_rms * math.sqrt(features / len(written))
    else:
        output = QUIET * mlp_rms
    level = output / (OUT_WEIGHT * (STEP_HIGH - STEP_LOW))  # the value that makes a step `output`
    w_gate = np.array(gates)
    w_value = np.tile(on_stream(no_weights, level, rows.reference), (len(gates), 1))
    read = w_gate.any(axis=0) | w_value.any(axis=0)
    pre_norm = np.zeros(width)
    pre_norm[read] = STEP_READ * stream_rms / size[read]  # every feature is then STEP_READ
    # The post-norm divides by the RMS of what the MLP outputs, Gemma's and the block's.
    output_rms = math.sqrt(mlp_rms**2 + len(written) * output**2 / features)
    post_norm = np.zeros(width)
    post_norm[written] = size[written] * output_rms / output
    return NormedBlock(names, w_gate, w_value, np.array(outs).T, pre_norm, post_norm)


def wide_gemma(model: Transformer, gemma_weights: dict, config: dict, sample) -> beside.WideGemma:
    """The Gemma of `gemma_weights` and `config`, as beside.widen takes them, with the blocks of
    the ideal transformer `model` next to its last layers. `sample` is tokens to measure Gemma's
    stream on (see `measure`): text such as Gemma will be given."""
    circuit = model.circuit
    name = circuit.name
    assert len(model.codes) == config["vocab_size"], f"{name} needs a code for each token of Gemma"
    heads = config["num_attention_heads"]
    assert len(model.heads) <= heads, f"{name} reads more tokens than a layer of Gemma has heads"
    values = sample_values(circuit, model.codes)
    check_sampled_values(circuit, values)  # among others: an MLP reads only bits, as a step does
    is_half = np.array([np.isclose(values[output], 1 / 2) for output in circuit.outputs])
    together = (is_half.all(axis=0) | ~is_half.any(axis=0)).all()
    assert together, f"{name}: where one output is 1/2, every output must be 1/2"
    blocks = steps_of(circuit)
    gemma_layers = len(config["layer_types"])
    room = f"{name} needs {len(blocks)} layers, and Gemma has {gemma_layers}"
    assert len(blocks) <= gemma_layers, room
    first = gemma_layers - len(blocks)  # block d is beside Gemma's layer first + d - 1
    scales = measure(gemma_weights, config, sample)
    rows = Rows(model, blocks)
    features = math.ceil((rows.width + config["hidden_size"]) / heads) * heads  # see beside.py
    # The size of each feature in the stream, at the median token. Row d is read by block d + 1.
    size = np.zeros(rows.width)
    for d, stream_rms in enumerate(scales.before_mlp[first:]):
        size[[*rows.feature_of[d].values(), *rows.ones[d]]] = SHARE * stream_rms
    size[rows.reference] = REFERENCE * scales.before_mlp[-1] * one_sizes(ONE_FEATURES)
    size[rows.code] = beside.OUTPUT_SIZE * math.sqrt(features / beside.REFERENCE_FEATURES)
    normed = []
    for d, steps in enumerate(blocks, 1):
        layer = first + d - 1  # of Gemma
        layer_scales = (scales.before_mlp[layer], scales.mlp[layer])
        normed.append(step_block(steps, d, rows, size, layer_scales, features, d == len(blocks)))
    hidden_units = config["intermediate_size"] + max(len(block.units) for block in normed)
    lift_layer = first + 1  # in the wide model; block d is in layer lift_layer + d
    wide = beside.widen(gemma_weights, config, features, hidden_units, [0, lift_layer])
    # The embedding holds the code and the ONEs small, as Gemma's features are small there.
    embedded = np.zeros(rows.width)
    embedded[rows.reference] = CODE_SIZE * one_sizes(ONE_FEATURES)
    embedded[rows.code] = CODE_SIZE
    embedding = lower_embedding(model, embedded)
    embedding[:, rows.code] *= CODE_SIZE / BIT_SIZE
    wide.weights["embed_tokens.weight"][:, : embedding.shape[1]] = embedding / math.sqrt(features)
    copied = embedded.copy()  # the copy writes the inputs as small, and erases the code
    copied[[*rows.feature_of[0].values()]] = CODE_SIZE
    copied[rows.code] = 0
    copy = lower_attention(model, rows.layout[0], embedded)  # its norms are made below
    write_own_attention(wide, 0, copy, embedded, copied, beside.GEMMA_RMS_BEFORE, READ_SIZE)
    lifted = copied.copy()
    lifted[rows.lifted] = size[rows.lifted]
    gemma_rms = scales.before_attention[first]  # before the lift: before Gemma's layer `first`
    # The lift reads its features small, so that its pre-norm weights are not too large.
    heads = lift(rows, copied, lifted)
    write_own_attention(wide, lift_layer, heads, copied, lifted, gemma_rms, beside.HEAD_READ_SIZE)
    for d, block in enumerate(normed, 1):
        gemma.write_block(wide.weights, lift_layer + d, block, norm_scale=1.0)
    wide.weights["norm.weight"][rows.code] = beside.OUTPUT_NORM * BIT_SIZE / CODE_SIZE
    gemma.finish(wide.weights, wide.config)
    wide.names = {
        "features": [rows.layout[0]] * (lift_layer + 1) + rows.layout[1:],
        "hidden": [[]] * (lift_layer + 1) + [block.units for block in normed],
    }
    return wide


def check(wide, model: Transformer, tokens):
    """Check on `tokens` that where the ideal transformer `model` names the next token, so does
    `wide`, a beside.WideGemma of `wide_gemma` (see gemma.check_next_tokens)."""
    streams = gemma.run(wide.weights, wide.config, tokens)
    gemma.check_next_tokens(wide.weights, wide.config, model, tokens, streams[-1])

"""Step 3, the transformer: weights that run a circuit on tokens.

The circuit is a rule for the next token. Each token has a code: a list of bits, one for each
output of the circuit. The circuit reads the codes of the last few tokens, the oldest token
first: the number of these tokens is its window. Its outputs are the code of the next token.
The transformer does the same with weights. For a list of tokens it computes, with one row
of x for each position:

    x = embedding[tokens]                           ONE and the code of each token
    x = x + (x of the token h back) @ heads[h].T    summed over the attention heads h
    x = x + block(x)                                for each MLP block, in turn
    logits = x @ embedding.T                        the same embedding again, without ONE

x is the residual stream. It holds a bit b as 2 * b - 1: -1 for 0 and +1 for 1. An empty
feature is 0, and so is a signal that is 1/2. After the attention, x holds row 0 of the
layout (see layout.py). Block d is layer d of the layout: after it, x holds row d. The logit
of a token is the number of bits of its code that equal the outputs of the circuit, minus the
number that differ. So the largest logit is that of the token whose code the circuit computed.

This transformer is ideal: it has ReLU where a real model has GELU, it has no normalization,
and each head attends to exactly one token, without queries or keys. lower_stream.py and
lower.py turn it into the layers of a real model, Gemma, which compute the same.
"""

from dataclasses import dataclass

import numpy as np

from .circuit import MLP, Circuit
from .layout import ONE, Layout, depths, layout


@dataclass
class Block:
    """An MLP block. It adds w_out @ (relu(w_gate @ x) * (w_value @ x)) to the stream x.
    It has no biases: a bias is a weight on feature 0, which holds ONE."""

    units: list[str]  # the name of each hidden unit
    mlp_names: list[str | None]  # the name of each unit's MLP: None for a unit that `block` adds
    w_gate: np.ndarray  # [unit, feature]
    w_value: np.ndarray  # [unit, feature]
    w_out: np.ndarray  # [feature, unit]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """What the block adds to the stream x, an array [position, feature]."""
        hidden = np.maximum(x @ self.w_gate.T, 0) * (x @ self.w_value.T)
        return hidden @ self.w_out.T


@dataclass
class Transformer:
    """The weights of a transformer, with the circuit and the layout they were made from."""

    circuit: Circuit
    codes: dict[int, list[int]]  # the code of each token
    layout: Layout  # layout[d][f]: the signal that feature f holds after block d
    embedding: np.ndarray  # [token, feature], for feature 0 (ONE) and the features of the code
    heads: list[np.ndarray]  # heads[h][f, g]: the weight to feature f from feature g, h tokens back
    blocks: list[Block]  # blocks[d - 1] turns row d - 1 of the layout into row d

    def run(self, tokens) -> list[np.ndarray]:
        """The stream [position, feature] for a list of tokens, after the attention and after
        each block: the stream at index d holds row d of the layout."""
        embedded = self.embedding[np.asarray(tokens, int)]
        x = np.zeros((len(embedded), len(self.layout[0])))
        x[:, : embedded.shape[1]] = embedded
        attended = np.zeros_like(x)
        for h, head in enumerate(self.heads):
            reached = max(len(x) - h, 0)  # the number of positions that have a token h back
            attended[h:] += x[:reached] @ head.T  # position p reads position p - h
        streams = [x + attended]
        for block in self.blocks:
            streams.append(streams[-1] + block(streams[-1]))
        return streams

    def logits(self, tokens) -> np.ndarray:
        """The logits [position, token] for a list of tokens. ONE, feature 0, is left out."""
        last = self.run(tokens)[-1]
        code = slice(1, self.embedding.shape[1])  # the features that hold the code
        return last[:, code] @ self.embedding[:, code].T

    def generate(self, tokens, count: int) -> list[int]:
        """`tokens` followed by `count` more: each new token is the one with the largest
        logit. It is the circuit's answer once there are as many tokens as the circuit reads."""
        tokens = list(tokens)
        for _ in range(count):
            tokens.append(int(self.logits(tokens)[-1].argmax()))
        return tokens


def transformer(circuit: Circuit, codes: dict[int, list[int]]) -> Transformer:
    """The transformer that runs `circuit` on the tokens of `codes`, {token: its code}. The
    circuit must read one or more whole codes. Feature 0 holds ONE, and features 1, 2, ...
    hold the code of the token and later the code of the next token: each output of the
    circuit is made where its bit of the code was."""
    bits = len(circuit.outputs)
    depth_of = depths(circuit)
    assert bits and circuit.inputs and len(circuit.inputs) % bits == 0, (
        f"{circuit.name} reads {len(circuit.inputs)} bits: not one or more codes of {bits} bits"
    )
    made_by_mlps = all(depth_of[name] > 0 for name in circuit.outputs)
    assert made_by_mlps and len(set(circuit.outputs)) == bits, (
        f"{circuit.name}: each output must be the output of an MLP, and none may repeat. "
        "To return an input, or one signal twice, return a copy of it: lib.or_([x])"
    )
    read_by_mlps = {name for m in circuit.mlps for name in m.inputs}
    first_row = [ONE] + [None] * bits + [name for name in circuit.inputs if name in read_by_mlps]
    reserved = {name: 1 + i for i, name in enumerate(circuit.outputs)}
    rows = layout(circuit, first_row, reserved)
    for token, code in codes.items():
        assert len(code) == bits, f"token {token} has a code of length {len(code)}, not {bits}"
    tokens = list(codes)
    embedding = np.zeros((max(tokens) + 1, 1 + bits))  # a token with no code stays all zeros
    embedding[tokens, 0] = 1
    embedding[tokens, 1:] = 2 * np.array([codes[token] for token in tokens]) - 1
    heads = attention_heads(circuit.inputs, rows[0], bits)
    blocks = []
    for depth in range(1, len(rows)):
        made = [m for m in circuit.mlps if depth_of[m.outputs[0]] == depth]
        blocks.append(block(made, rows[depth - 1], rows[depth]))
    return Transformer(circuit, codes, rows, embedding, heads, blocks)


def attention_heads(inputs: list[str], first_row: list[str | None], bits: int) -> list[np.ndarray]:
    """One attention head for each token that the circuit reads. Head h attends to the token
    h positions back, and copies the bits of its code to the features of `first_row` that
    hold the circuit's inputs for that token. Head 0 also erases the code of its own token,
    so that the blocks can write the code of the next token there. The matrix of a head is
    the product of the output matrix and the value matrix of a real head."""
    width = len(first_row)
    window = len(inputs) // bits
    heads = []
    for h in range(window):
        head = np.zeros((width, width))
        start = (window - 1 - h) * bits  # the inputs start with the oldest token
        for bit in range(bits):
            name = inputs[start + bit]
            if name in first_row:  # an input that the circuit never reads has no feature
                head[first_row.index(name), 1 + bit] = 1
        heads.append(head)
    for bit in range(bits):
        heads[0][1 + bit, 1 + bit] = -1
    return heads


def block(mlps: list[MLP], before: list[str | None], after: list[str | None]) -> Block:
    """The block that turns a stream that holds `before` into one that holds `after`.

    Its hidden units are, in this order: the units of `mlps`, the MLPs that make the new
    signals of `after`; one unit named `erase s` for each signal s that leaves its feature; and
    one unit named `constant`. An MLP outputs a bit b, and the stream holds 2 * b - 1: so the
    block adds twice the output of the MLP, and the constant unit adds -1."""
    width = len(after)
    identity = np.eye(width)
    constant = np.zeros(width)  # what the constant unit adds to each feature
    parts = []
    for m in mlps:
        input_features = [before.index(name) for name in m.inputs]
        output_features = [after.index(name) for name in m.outputs]
        w_gate = on_stream(m.w_gate, m.b_gate, input_features, width)
        w_value = on_stream(m.w_value, m.b_value, input_features, width)
        w_out = np.zeros((width, len(m.units)))
        w_out[output_features] = 2 * m.w_out
        constant[output_features] = -1
        parts.append(Block(m.units, [m.name] * len(m.units), w_gate, w_value, w_out))
    # An eraser is relu(ONE) * s, subtracted from the feature of s.
    erased = [f for f, old in enumerate(before) if old and old != after[f]]
    names = [f"erase {before[f]}" for f in erased]
    w_gate = np.tile(identity[0], (len(erased), 1))  # each gate reads ONE
    w_value = identity[erased]  # each value reads the signal that it erases
    parts.append(Block(names, [None] * len(erased), w_gate, w_value, -identity[:, erased]))
    # The constant unit is relu(ONE) * ONE, which is 1.
    parts.append(Block(["constant"], [None], identity[:1], identity[:1], constant[:, None]))
    return Block(
        units=[name for part in parts for name in part.units],
        mlp_names=[name for part in parts for name in part.mlp_names],
        w_gate=np.vstack([part.w_gate for part in parts]),
        w_value=np.vstack([part.w_value for part in parts]),
        w_out=np.hstack([part.w_out for part in parts]),
    )


def on_stream(weights: np.ndarray, bias: np.ndarray, features: list[int], width: int) -> np.ndarray:
    """Weights [unit, input] and a bias on bits, as weights [unit, feature] on the stream.
    Column i of `weights` is for the bit in feature features[i]. The stream holds a bit b as
    s = 2 * b - 1, so b = s / 2 + 1 / 2: half of each weight goes to the feature of its bit,
    and the other half joins the bias, which is the weight on feature 0, ONE."""
    on_features = np.zeros((len(bias), width))
    for column, feature in enumerate(features):
        on_features[:, feature] += weights[:, column] / 2  # += : an MLP may read a bit twice
    on_features[:, 0] += bias + weights.sum(axis=1) / 2
    return on_features

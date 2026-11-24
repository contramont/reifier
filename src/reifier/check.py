"""Step 6, checks: a circuit against a Python function, and a transformer against its circuit.
A check that fails raises an AssertionError that names the wrong signal, feature or position."""

from itertools import product

import numpy as np

from .circuit import Circuit
from .layout import ONE
from .transformer import Transformer

MAX_CASES = 256  # a circuit is checked on every case of its inputs if there are at most this many


def check_circuit(circuit: Circuit, reference) -> int:
    """Check `circuit` against the Python function `reference`, which takes the bits of a case
    in one list and returns the outputs: a bit, or a list with one value for each output. Every
    case is tried, or MAX_CASES random ones if there are more: random bits, not the codes of
    tokens. Returns how many were tried."""
    count = len(circuit.inputs)
    if 2**count <= MAX_CASES:
        inputs = list(product([0, 1], repeat=count))
    else:
        inputs = np.random.default_rng(0).integers(0, 2, (MAX_CASES, count))
    values = circuit.values(inputs)
    for case, bits in enumerate(np.asarray(inputs).tolist()):
        expected = np.atleast_1d(reference(bits))
        assert len(expected) == len(circuit.outputs), "reference must return one value per output"
        for name, want in zip(circuit.outputs, expected):
            got = values[name][case]
            assert np.isclose(got, want), f"{name} is {got:g} on input {bits}, not {want:g}"
    return len(inputs)


def signal_values(model: Transformer, tokens) -> dict[str, np.ndarray]:
    """The value of every signal of the circuit of `model`, and of ONE, at each position where
    the circuit has all the tokens that it reads: position window - 1 and later."""
    window = len(model.heads)
    assert len(tokens) >= window, f"the circuit reads {window} tokens, and it got {len(tokens)}"
    without_code = sorted({token for token in tokens if token not in model.codes})
    assert not without_code, f"the tokens {without_code} have no code"
    codes = np.array([model.codes[token] for token in tokens])
    cases = [codes[end - window : end].ravel() for end in range(window, len(tokens) + 1)]
    return model.circuit.values(cases) | {ONE: np.ones(len(cases))}


def expected_stream(row: list[str | None], values: dict[str, np.ndarray]) -> np.ndarray:
    """The stream [position, feature] that holds `row`, a row of a layout, for the `values` of
    `signal_values`: 2 * value - 1 for a signal and for ONE, and 0 for an empty feature."""
    empty = np.zeros_like(values[ONE])
    columns = [empty if name is None else 2 * values[name] - 1 for name in row]
    return np.column_stack(columns)


def check_transformer(model: Transformer, tokens, tolerance: float = 0.0):
    """Check `model` on `tokens`, at the positions of `signal_values`. After the attention and
    after each block, every feature must hold what the layout says, to within `tolerance`. And
    the largest logit must be that of the token whose code the circuit computes, if a token
    has that code."""
    first_checked = len(model.heads) - 1
    values = signal_values(model, tokens)
    for depth, (row, stream) in enumerate(zip(model.layout, model.run(tokens))):
        error = np.abs(stream[first_checked:] - expected_stream(row, values))
        case, feature = np.unravel_index(error.argmax(), error.shape)
        held = row[feature] or "empty"
        where = f"depth {depth}, position {first_checked + case}, feature {feature} ({held})"
        assert error[case, feature] <= tolerance, f"{where} is off by {error[case, feature]:g}"
    # Tokens with the same code have equal logits, and the lowest is chosen: token_of keeps it.
    token_of = {tuple(model.codes[token]): token for token in sorted(model.codes, reverse=True)}
    outputs = np.array([values[name] for name in model.circuit.outputs]).T
    outputs = np.rint(2 * outputs) / 2  # each output is 0, 1 or 1/2, up to rounding
    found = model.logits(tokens)[first_checked:].argmax(axis=1).tolist()
    for position, (code, got) in enumerate(zip(outputs.tolist(), found), first_checked):
        want = token_of.get(tuple(code), got)  # if no token has the code, any token passes
        assert got == want, f"position {position}: the next token is {got}, not {want}"

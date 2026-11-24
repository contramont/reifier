"""Step 1, the circuit: a Python function, run once, becomes a list of small MLPs.

A bit is 0 or 1. A circuit computes bits from bits with small MLPs that are joined by signals.
A signal is the name of one bit: an input of the circuit, or an output of one of its MLPs.
To trace a function is to run it once on signals in place of bits: each call of `mlp` in the
function adds one MLP to the circuit.

Every MLP has the form of Gemma's MLP, with biases added and ReLU in place of GELU:

    gate    = w_gate @ inputs + b_gate
    value   = w_value @ inputs + b_value
    outputs = w_out @ (relu(gate) * value)

Row j of w_gate and w_value, with column j of w_out, is hidden unit j. A plain ReLU unit is
the special case where the value is the constant 1.

The output of an MLP must be a bit, whatever bits the circuit is given, and an MLP may read
only bits: a real model needs both rules. An MLP with several outputs may make other numbers,
such as 1/2 (lib.table), if the squares of 2 * output - 1 add to the same sum for every input.

Names say where everything comes from. A subcircuit is a function marked with `@subcircuit`:
its name comes before the name of everything that it makes.

    add/xor.1/odd    the MLP `odd`, made by the second call of `xor` inside `add`
    name:i           output i of an MLP that has several outputs
    name#j           hidden unit j of an MLP that has several hidden units
"""

import functools
import inspect
from dataclasses import dataclass

import numpy as np


@dataclass
class MLP:
    """One MLP of a circuit, in the form above."""

    name: str
    inputs: list[str]  # the signals that it reads
    outputs: list[str]  # the signals that it makes
    units: list[str]  # the names of its hidden units
    w_gate: np.ndarray  # [unit, input]
    b_gate: np.ndarray  # [unit]
    w_value: np.ndarray  # [unit, input]
    b_value: np.ndarray  # [unit]
    w_out: np.ndarray  # [output, unit]

    def __call__(self, inputs) -> np.ndarray:
        """The outputs for `inputs`, an array [..., input]: one list of inputs, or many at once."""
        inputs = np.asarray(inputs, float)
        gate = inputs @ self.w_gate.T + self.b_gate
        value = inputs @ self.w_value.T + self.b_value
        return (np.maximum(gate, 0) * value) @ self.w_out.T


class Signal:
    """One bit of a circuit while it is traced: it has a name and no value."""

    def __init__(self, name: str):
        self.name = name

    def __bool__(self):
        raise TypeError(f"{self.name} has no value yet: a traced function cannot branch on it")


_mlps: list[MLP] = []  # the MLPs that the function being traced has made so far
_scopes: list[str] = []  # the full names of the subcircuits that are running, outermost first
_uses: dict[str, int] = {}  # how many times each full name has been asked for


def mlp(inputs, w_gate, b_gate, w_value, b_value, w_out, name="mlp") -> list[Signal]:
    """Add one MLP to the circuit being traced, and return its output signals.

    `inputs` is a list of signals. The other arguments are the weights of the form above, as
    nested lists or arrays: one row of w_gate and of w_value for each hidden unit, and one row
    of w_out for each output."""
    name = _full_name(name)
    b_gate = np.array(b_gate, float)
    b_value = np.array(b_value, float)
    per_unit = len(w_gate) == len(w_value) == len(b_value) == len(b_gate)
    assert per_unit, f"{name}: w_gate, w_value and b_value must be as long as b_gate"
    # Each reshape fails if a row has the wrong length. It also makes an empty list a matrix.
    w_gate = np.array(w_gate, float).reshape(len(b_gate), len(inputs))
    w_value = np.array(w_value, float).reshape(len(b_gate), len(inputs))
    w_out = np.array(w_out, float).reshape(len(w_out), len(b_gate))
    units = [name] if len(b_gate) == 1 else [f"{name}#{j}" for j in range(len(b_gate))]
    outputs = [name] if len(w_out) == 1 else [f"{name}:{i}" for i in range(len(w_out))]
    input_names = [signal.name for signal in inputs]
    _mlps.append(MLP(name, input_names, outputs, units, w_gate, b_gate, w_value, b_value, w_out))
    return [Signal(output) for output in outputs]


def _full_name(name: str) -> str:
    """`name` under the subcircuit that is running. A full name that was asked for before
    gets a number: xor, xor.1, xor.2."""
    path = f"{_scopes[-1]}/{name}" if _scopes else name
    count = _uses.get(path, 0)
    _uses[path] = count + 1
    if count == 0:
        return path
    return _full_name(f"{name}.{count}")  # the numbered name is taken too: an MLP may ask for xor.1


def subcircuit(function):
    """Decorator: everything that `function` makes is named under the name of `function`."""

    @functools.wraps(function)
    def named(*args, **kwargs):
        _scopes.append(_full_name(function.__name__))
        try:
            return function(*args, **kwargs)
        finally:
            _scopes.pop()

    return named


@dataclass
class Circuit:
    """A traced function: its input signals, its output signals, and its MLPs in the order
    they were made, so that each MLP comes after the MLPs whose outputs it reads."""

    name: str
    inputs: list[str]
    outputs: list[str]
    mlps: list[MLP]

    def values(self, inputs) -> dict[str, np.ndarray]:
        """The value of every signal, by name. `inputs` is an array [case, input] of bits: a case
        is a bit for each input. Each value is an array with one number for each case."""
        inputs = np.asarray(inputs, float)
        values = dict(zip(self.inputs, inputs.T))
        for m in self.mlps:
            mlp_inputs = np.zeros((len(inputs), len(m.inputs)))
            for column, name in enumerate(m.inputs):
                mlp_inputs[:, column] = values[name]
            values.update(zip(m.outputs, m(mlp_inputs).T))
        return values

    def non_bits(self) -> set[str]:
        """The signals that need not be bits: the outputs of the MLPs with several outputs."""
        several = [m.outputs for m in self.mlps if len(m.outputs) > 1]
        return set().union(*several)


def trace(function, *lengths: int) -> Circuit:
    """Run `function` once on signals, and return the circuit that it built.

    `function` gets a list of lengths[i] signals as its parameter i: they are named after the
    parameter, bits0, bits1, ... It returns a signal or a list of signals. MLPs that no output
    depends on are left out."""
    for state in (_mlps, _scopes, _uses):
        state.clear()
    arguments = []
    inputs = []
    parameters = list(inspect.signature(function).parameters)
    assert len(lengths) <= len(parameters), f"{function.__name__}: more lengths than parameters"
    for parameter, length in zip(parameters, lengths):
        names = [_full_name(f"{parameter}{i}") for i in range(length)]  # these names are now taken
        arguments.append([Signal(name) for name in names])
        inputs += names
    result = function(*arguments)
    if isinstance(result, Signal):
        result = [result]
    outputs = [signal.name for signal in result]
    return Circuit(function.__name__, inputs, outputs, used_mlps(_mlps, outputs))


def used_mlps(mlps: list[MLP], outputs: list[str]) -> list[MLP]:
    """The MLPs that `outputs` depend on, in their original order."""
    needed = set(outputs)
    kept = []
    for m in reversed(mlps):
        if needed.intersection(m.outputs):
            kept.append(m)
            needed.update(m.inputs)
    return kept[::-1]

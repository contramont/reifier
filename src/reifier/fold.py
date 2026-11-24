"""Step 5, constant folding: constants become biases, and MLPs that copy or negate are removed.
Folding keeps the outputs of the circuit and saves MLPs, and sometimes layers.

A constant is a signal with the same value for every input: an output of an MLP that has no
inputs. An MLP that reads a constant gets the value in its biases instead, and it makes
constants in turn if it is left with no inputs. A wire is an MLP with one input and one
output that returns its input or its negation. An MLP that reads a wire reads the wire's
input instead, and the wire is removed once nothing reads it. Three kinds of wire stay: one
that makes an output of the circuit; one whose input need not be a bit (see circuit.py); and
one whose reader also reads the wire's input: a bit read twice may need too large a weight
(see lib.threshold_gate).
"""

from dataclasses import replace

import numpy as np

from .circuit import MLP, Circuit, used_mlps


def fold(circuit: Circuit) -> Circuit:
    """`circuit` with constants and wires folded away. A signal that stays keeps its name."""
    constants = {}  # the value of each constant
    wires = {}  # each wire that may be removed, by the name of its output
    non_bits = circuit.non_bits()
    folded = []
    for m in circuit.mlps:
        m = replace_inputs(m, constants, wires)
        if not m.inputs:
            constants.update(zip(m.outputs, m([]).tolist()))
        # A wire returns 0 and 1 for the inputs 0 and 1, or 1 and 0 if it negates.
        is_wire = len(m.inputs) == 1 and m([[0], [1]]).tolist() in ([[0], [1]], [[1], [0]])
        if is_wire and m.outputs[0] not in circuit.outputs and m.inputs[0] not in non_bits:
            wires[m.outputs[0]] = m
        folded.append(m)
    return replace(circuit, mlps=used_mlps(folded, circuit.outputs))


def replace_inputs(m: MLP, constants: dict[str, float], wires: dict[str, MLP]) -> MLP:
    """`m` without the constants and the wires that it reads, and with the same outputs.
    Input i of `m` is written as scale[i] * inputs[i] + offset[i]: a constant is its value,
    a wire is its own input, or 1 minus that if it negates, and any other input is itself."""
    inputs = list(m.inputs)
    scale = np.ones(len(inputs))
    offset = np.zeros(len(inputs))
    for i, name in enumerate(m.inputs):
        if name in constants:
            offset[i] = constants[name]
        elif name in wires and wires[name].inputs[0] not in inputs:
            wire = wires[name]
            inputs[i] = wire.inputs[0]
            offset[i] = wire([0])[0]
            scale[i] = wire([1])[0] - offset[i]
    kept = [i for i, name in enumerate(m.inputs) if name not in constants]
    return replace(
        m,
        inputs=[inputs[i] for i in kept],
        w_gate=(m.w_gate * scale)[:, kept],
        b_gate=m.b_gate + m.w_gate @ offset,
        w_value=(m.w_value * scale)[:, kept],
        b_value=m.b_value + m.w_value @ offset,
    )

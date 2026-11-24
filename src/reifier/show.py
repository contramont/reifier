"""Step 7, printing: print(show(circuit)) or print(show(model.layout)) shows one as text."""

from .circuit import Circuit
from .layout import Layout


def show(circuit_or_layout: Circuit | Layout) -> str:
    """A circuit as one line for each hidden unit, or a layout as a table."""
    to_lines = circuit_lines if isinstance(circuit_or_layout, Circuit) else layout_lines
    return "\n".join(to_lines(circuit_or_layout))


def weighted_sum(weights, names, bias=0.0) -> str:
    """A weighted sum as text, such as `+4 a -1 b +0.5`."""
    terms = [f"{weight:+g} {name}" for weight, name in zip(weights, names) if weight]
    if bias or not terms:
        terms.append(f"{bias:+g}")
    return " ".join(terms)


def circuit_lines(circuit: Circuit) -> list[str]:
    """The inputs and outputs of `circuit`, then one line for each hidden unit: its name, its
    gate, its value, and the outputs of its MLP that it adds to."""
    lines = [f"{circuit.name}: {' '.join(circuit.inputs)} -> {' '.join(circuit.outputs)}"]
    for m in circuit.mlps:
        for j, unit in enumerate(m.units):
            gate = weighted_sum(m.w_gate[j], m.inputs, m.b_gate[j])
            value = weighted_sum(m.w_value[j], m.inputs, m.b_value[j])
            adds_to = weighted_sum(m.w_out[:, j], m.outputs)
            lines.append(f"  {unit} = relu({gate}) * ({value}), added to: {adds_to}")
    return lines


def layout_lines(layout: Layout) -> list[str]:
    """A table with one line for each feature and one column for each depth: the signal that
    the feature holds then, or a dot if it is empty."""
    columns = [["feature"] + [str(feature) for feature in range(len(layout[0]))]]
    for depth, row in enumerate(layout):
        columns.append([f"depth {depth}"] + [name or "." for name in row])
    padded = []
    for column in columns:
        width = max(len(cell) for cell in column)
        padded.append([cell.ljust(width) for cell in column])
    return ["  ".join(cells).rstrip() for cells in zip(*padded)]

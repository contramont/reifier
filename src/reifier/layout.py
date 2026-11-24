"""Step 2, the layout: which feature of the residual stream holds which signal, and when.

The residual stream of a transformer is a list of numbers at each token position, and a
feature is one of those numbers. Each layer reads the stream and adds to it. A layout is a
table of names: layout[d][f] is the signal that feature f holds after layer d. It is None
where the feature is empty, and ONE where the feature holds the constant 1, which layers use
as their bias. Row 0 is the stream before the first layer. A layout follows three rules.

Depth. An input of the circuit has depth 0, and the outputs of an MLP are one deeper than the
deepest signal that the MLP reads. Layer d makes the signals of depth d.

Lifetime. A signal stays in one feature from the layer that makes it until the last layer
that reads it. That layer releases the feature: it subtracts the signal, and a later signal
can take the empty feature. So the stream need only be as wide as the largest number of
signals in use at one time. An output of the circuit is never released.

Placement. A new signal takes the first feature that it can from this list:

1. an empty feature, the one that was released most recently;
2. the feature of one of its own inputs, if its layer releases that input;
3. any other feature that its layer releases;
4. a new feature, which makes the stream one wider.

In cases 2 and 3 the layer overwrites: it subtracts the old signal and adds the new one in
the same feature. That keeps the stream narrow, but a real model writes to an empty feature
more accurately than it overwrites one, so empty features come first. An input comes next
because a signal is often equal to one of its inputs: the feature then does not change.

Two kinds of signal are placed with more care. A reserved signal has a feature that the
caller chose for it. Until the layer that makes it, other signals can use that feature, but
only if they are released before that layer, so that it writes to an empty feature. A signal
that need not be a bit (see circuit.py) takes no part in overwriting, because a real model
can only overwrite a bit with a bit. It takes an empty or a new feature, and the layer that
releases it puts no new signal there.
"""

from math import inf

from .circuit import Circuit

ONE = "ONE"  # in a layout, the name of a feature that holds the constant 1
Layout = list[list[str | None]]  # layout[depth][feature]: the name of a signal, ONE or None


def depths(circuit: Circuit) -> dict[str, int]:
    """The depth of every signal of `circuit`, in the order in which the signals are made."""
    depth_of = dict.fromkeys(circuit.inputs, 0)
    for m in circuit.mlps:
        deepest_input = max([depth_of[name] for name in m.inputs], default=0)
        for name in m.outputs:
            depth_of[name] = deepest_input + 1
    return depth_of


def release_depths(circuit: Circuit, depth_of: dict[str, int]) -> dict[str, float]:
    """The depth of the layer that releases each signal of `circuit`: infinite for an output
    of the circuit, and else the depth of the last MLP that reads the signal. An input that
    nothing reads is left out. An output of an MLP that nothing reads stays as long as its
    other outputs: if they are not bits, a real model needs them all in the stream together."""
    released_at = dict.fromkeys(circuit.outputs, inf)
    for m in circuit.mlps:
        depth = depth_of[m.outputs[0]]  # the layer that reads the inputs of m
        for name in m.inputs:
            released_at[name] = max(released_at.get(name, 0), depth)
    for m in circuit.mlps:
        of_read_outputs = [released_at[name] for name in m.outputs if name in released_at]
        for name in m.outputs:
            released_at.setdefault(name, max(of_read_outputs))
    return released_at


def layout(circuit: Circuit, first_row: list[str | None], reserved: dict[str, int]) -> Layout:
    """The layout of `circuit`, by the rules above. `first_row` is row 0: where the stream
    holds ONE and the inputs that the circuit reads. `reserved` gives the feature of each
    reserved signal."""
    depth_of = depths(circuit)
    assert ONE not in depth_of, f"{circuit.name}: no signal may be named {ONE}"
    released_at = release_depths(circuit, depth_of)
    non_bits = circuit.non_bits()
    feature_of = {name: f for f, name in enumerate(first_row) if name} | reserved
    # A reserved feature is free for other signals until the layer before its own signal.
    free_until = {f: depth_of[name] - 1 for name, f in reserved.items()}
    empty = [f for f, name in enumerate(first_row) if name is None]  # most recently released last
    rows = [list(first_row)]
    for depth in range(1, max(depth_of.values()) + 1):
        row = list(rows[-1])
        released = [name for name in depth_of if released_at.get(name) == depth]
        for name in released:
            row[feature_of[name]] = None
        overwritable = [feature_of[name] for name in released if name not in non_bits]
        for m in circuit.mlps:
            if depth_of[m.outputs[0]] != depth:
                continue
            own = [feature_of[name] for name in m.inputs]
            for name in m.outputs:
                if name not in reserved:
                    choices = empty[::-1]  # case 1
                    if name not in non_bits:
                        choices += [f for f in own if f in overwritable]  # case 2
                        choices += overwritable[::-1]  # case 3
                    choices.append(len(row))  # case 4
                    allowed = [f for f in choices if released_at[name] <= free_until.get(f, inf)]
                    feature_of[name] = allowed[0]
                feature = feature_of[name]
                if feature == len(row):
                    row.append(None)
                row[feature] = name
                for features in (empty, overwritable):
                    if feature in features:
                        features.remove(feature)
        empty += overwritable + [feature_of[name] for name in released if name in non_bits]
        rows.append(row)
    width = len(rows[-1])
    return [row + [None] * (width - len(row)) for row in rows]

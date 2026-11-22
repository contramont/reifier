"""Constant folding: a trace with its constants moved into the biases of what reads them, and
its wires taken out.

A constant is what an MLP with no inputs gives, as the gates of const do. The constants that an
MLP reads go into its bias, and an MLP left with no inputs gives constants in turn.

A wire is an MLP left with one input and one output, which gives that bit or its negation.
What read the wire reads the bit, negated in its weights and its bias, and a wire that nothing
reads any more is dropped: what read it may then come a layer sooner. examples/keccak.py
makes such wires: the xor of one bit and of constants whose xor is 1 is a table of that bit
alone, its negation, as in the first round of the sponge of examples/alongside.py, where all
but a few bits are constants. Three kinds of wire stay: one that is an output of the trace;
one of a signal that may be 1/2, as what a table with a rest gives, since an MLP that passes
a bit on need not pass on a half; and one whose reader reads its bit too, and would read it
twice. A wire that was put in to hold a bit back a layer goes as any other: relay.py holds
bits back.

What stays keeps its name: check_trace(fold(t), t) compares the outputs, and every signal that
is left.
"""

from dataclasses import replace

import numpy as np

from .trace import Trace, needed


def fold(t: Trace) -> Trace:
    const, wire, mlps = {}, {}, []  # the value of a constant; of a wire, its bit, and if negated
    halves = {y for m in t.mlps if len(m.outputs) > 1 for y in m.outputs}  # what may be 1/2
    for m in t.mlps:
        wi, b, inputs = m.wi.copy(), m.b.copy(), list(m.inputs)
        for i, x in enumerate(inputs):
            if x in wire and wire[x][0] not in inputs:  # w (1 - x) is w - w x
                inputs[i], negated = wire[x]
                b, wi[:, i] = b + negated * wi[:, i], (1 - 2 * negated) * wi[:, i]
        live = [i for i, x in enumerate(inputs) if x not in const]
        b = b + wi @ np.array([const.get(x, 0.0) for x in inputs])
        m = replace(m, inputs=[inputs[i] for i in live], wi=wi[:, live], b=b)
        if not live:
            const.update(zip(m.outputs, m([]).tolist()))
        if len(live) == 1 and len(m.outputs) == 1 and m.outputs[0] not in t.outputs:
            low, high = m([[0], [1]]).ravel().tolist()
            if {low, high} == {0, 1} and m.inputs[0] not in halves:
                wire[m.outputs[0]] = m.inputs[0], low == 1
        mlps.append(m)  # a wire too: `needed` drops it, unless something still reads it
    return replace(t, mlps=needed(mlps, t.outputs))

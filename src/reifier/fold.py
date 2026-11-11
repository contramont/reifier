"""Constant folding: a trace with its constants moved into the biases of what reads them.

A constant is what an MLP with no inputs gives, as the gates of const do. The constants that an
MLP reads go into its bias, and an MLP left with no inputs gives constants in turn. None is left
but those that the trace gives as outputs, and what stays keeps its name: check_trace(fold(t), t)
compares the outputs, and every signal that is left.
"""

from dataclasses import replace

import numpy as np

from .trace import Trace, needed


def fold(t: Trace) -> Trace:
    const, mlps = {}, []  # the value of each signal that is a constant
    for m in t.mlps:
        live = [i for i, x in enumerate(m.inputs) if x not in const]
        b = m.b + m.wi @ np.array([const.get(x, 0.0) for x in m.inputs])
        m = replace(m, inputs=[m.inputs[i] for i in live], wi=m.wi[:, live], b=b)
        mlps.append(m)
        if not live:
            const.update(zip(m.outputs, m([]).tolist()))
    return replace(t, mlps=needed(mlps, t.outputs))

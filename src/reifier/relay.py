"""Relays: a trace in which an MLP reads only what the layer before it made.

A signal that is read more than a layer after it is made is passed on, by a relay in each layer
between: an MLP that gives what it reads, x', named after it. The MLP reads the last relay, and
an output is passed on to the last layer. Nothing else changes: check_trace(relay(t), t)
compares the outputs, as relays, and every other signal.
"""

from dataclasses import replace

import numpy as np

from .trace import MLP, Trace


def relay(t: Trace) -> Trace:
    depth, mlps = dict.fromkeys(t.inputs, 0), []  # the layer that makes each signal

    def passed(x: str, d: int) -> str:
        """x as layer d has it: x, or the last of the relays that pass it on from its own"""
        while depth[x] < d:
            x, old = x + "'", x
            if x not in depth:  # as lib.ands of old alone: two ReLU units, flat at a bit
                b, wo = np.array([-1 / 2, -3 / 4]), np.array([[4.0, -4.0]])
                units = [x + "#0", x + "#1"]
                mlps.append(MLP(x, [old], [x], units, np.ones((2, 1)), b, "relu", wo))
                depth[x] = depth[old] + 1
        return x

    for m in t.mlps:
        d = max((depth[x] for x in m.inputs), default=0)  # the last layer that m reads
        mlps.append(replace(m, inputs=[passed(x, d) for x in m.inputs]))
        depth |= dict.fromkeys(m.outputs, d + 1)
    last = max(depth.values())
    return replace(t, outputs=[passed(y, last) for y in t.outputs], mlps=mlps)

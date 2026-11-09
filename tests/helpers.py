from itertools import product

import numpy as np


def truth_table(n):
    return [list(r) for r in product((0, 1), repeat=n)]


def assert_matches(graph, net, xs):
    """A compiled network gives the circuit's outputs, relative to ONE, which leads its inputs
    and outputs; no norm, no rescaled ONE. Positions are independent without attention, so the
    rows are one sequence"""
    y = net.eval_sequence([[1, *x] for x in xs])
    if not any(str(n.circuit) == "rmsnorm" for n in net.nodes):
        np.testing.assert_allclose(y[:, 0], 1, atol=1e-3, err_msg="ONE is not 1")
    desired = graph.eval_sequence(xs)
    # atol leaves room to tune convert's sharpness and step_scale
    np.testing.assert_allclose(y[:, 1:] / y[:, :1], desired, atol=1e-3)

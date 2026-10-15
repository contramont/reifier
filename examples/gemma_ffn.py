"""A circuit on reals: Gemma's feed-forward block, x + down(gelu(gate(n)) * up(n)) with
n = rmsnorm(x), as three units: the norm, the GeGLU, and the residual sum.

Run: python examples/gemma_ffn.py
"""

import numpy as np

from reifier import circuit, inputs, show, unit


def gemma_ffn(norm, w_gate, w_up, w_down):
    """The block for one layer's weights, in the layout of Gemma's checkpoints"""
    d, f = w_down.shape
    eye = np.eye(d)

    @circuit
    def ffn(x):
        n = unit(x, eye, np.zeros(d), "rmsnorm", np.diag(1 + norm))
        y = unit(n, np.vstack([w_gate, w_up]), np.zeros(2 * f), "geglu", w_down)
        return unit([*x, *y], np.hstack([eye, eye]), np.zeros(d), "id", eye)  # x + y

    return ffn


def check(d=8, f=32, seed=0) -> tuple[str, float]:
    """The block's circuit on random weights, and its largest difference from numpy"""
    rng = np.random.default_rng(seed)
    norm, x = rng.normal(size=d) / 10, rng.normal(size=d)
    w_gate, w_up, w_down = rng.normal(size=(f, d)), rng.normal(size=(f, d)), rng.normal(size=(d, f))
    graph = gemma_ffn(norm, w_gate, w_up, w_down).trace(inputs(*["real"] * d))
    n = x / np.sqrt(np.mean(x * x) + 1e-6) * (1 + norm)
    g = w_gate @ n
    gelu = g / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)))
    return show(graph), float(np.abs(graph.eval(x) - (x + w_down @ (gelu * (w_up @ n)))).max())


if __name__ == "__main__":
    text, err = check(d=3, f=4)
    print(text, "\nlargest difference from numpy:", err)

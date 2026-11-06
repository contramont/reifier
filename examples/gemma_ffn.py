"""A circuit on reals: Gemma's feed-forward block, x + down(gelu(gate(n)) * up(n)) with
n = rmsnorm(x), as three MLPs: the norm, the GeGLU, and the residual sum.

Run: python examples/gemma_ffn.py
"""

import numpy as np

from reifier import circuit, format_graph, mlp, placeholders


def gemma_ffn(norm, w_gate, w_up, w_down):
    """The block for one layer's weights, in the layout of Gemma's checkpoints"""
    d_model, d_ff = w_down.shape
    eye = np.eye(d_model)

    @circuit
    def ffn(x):
        n = mlp(x, eye, np.zeros(d_model), "rmsnorm", np.diag(1 + norm))
        y = mlp(n, np.vstack([w_gate, w_up]), np.zeros(2 * d_ff), "geglu", w_down)
        return mlp([*x, *y], np.hstack([eye, eye]), np.zeros(d_model), "id", eye)  # x + y

    return ffn


def check(d_model=8, d_ff=32, seed=0) -> tuple[str, float]:
    """The block's circuit on random weights, and its largest difference from numpy"""
    rng = np.random.default_rng(seed)
    norm, x = rng.normal(size=d_model) / 10, rng.normal(size=d_model)
    w_gate, w_up = rng.normal(size=(d_ff, d_model)), rng.normal(size=(d_ff, d_model))
    w_down = rng.normal(size=(d_model, d_ff))
    graph = gemma_ffn(norm, w_gate, w_up, w_down).trace(placeholders(*["real"] * d_model))
    n = x / np.sqrt(np.mean(x * x) + 1e-6) * (1 + norm)
    g = w_gate @ n
    gelu = g / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)))
    return format_graph(graph), float(
        np.abs(graph.eval(x) - (x + w_down @ (gelu * (w_up @ n)))).max()
    )


if __name__ == "__main__":
    text, err = check(d_model=3, d_ff=4)
    print(text, "\nlargest difference from numpy:", err)

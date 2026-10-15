"""A circuit on reals: Gemma's feed-forward block, x + down(gelu(gate(n)) * up(n)) with
n = rmsnorm(x), as four blocks: rmsnorm, geglu, linear, and the residual sum.

Run: python examples/gemma_ffn.py
"""

import numpy as np

from reifier import block, circuit, inputs, show


def gemma_ffn(norm, w_gate, w_up, w_down):
    """The block for one layer's weights, in the layout of Gemma's checkpoints"""
    d, f = w_down.shape

    @circuit
    def ffn(x):
        n = block(x, np.eye(d), np.zeros(d), "rmsnorm")
        h = block(n, np.vstack([w_gate, w_up]) * (1 + norm), np.zeros(2 * f), "geglu")
        y = block(h, w_down, np.zeros(d))
        return block([*x, *y], np.hstack([np.eye(d), np.eye(d)]), np.zeros(d))  # x + y

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

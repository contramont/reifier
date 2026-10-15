"""A circuit on reals: Gemma's feed-forward block, x + down(gelu(gate(n)) * up(n)) with
n = rmsnorm(x), built from a new kind of circuit (RMSNorm) and two GLUs.

Run: python examples/gemma_ffn.py
"""

from dataclasses import dataclass

import numpy as np

from reifier import GLU, Circuit, circuit, inputs, show
from reifier.glu import ACTS

ACTS["gelu"] = lambda z: z / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


@dataclass(frozen=True, eq=False)
class RMSNorm(Circuit):
    """x / rms(x) * scale: a new kind of circuit is a class with kinds and eval"""

    scale: np.ndarray
    eps: float = 1e-6

    @property
    def kinds(self):
        return ("real",) * len(self.scale)

    def eval(self, x):
        return self.scale[:, None] * x / np.sqrt((x * x).mean(0) + self.eps)

    def __str__(self):
        return f"rmsnorm[{len(self.scale)}]"


def gemma_ffn(norm, w_gate, w_up, w_down):
    """The block for one layer's weights, in the layout of Gemma's checkpoints"""
    d, f = w_down.shape
    reals, eye = ("real",) * d, np.eye(d)
    ffn = GLU(w_gate, np.zeros(f), w_up, np.zeros(f), w_down, reals, act="gelu")
    add = GLU(np.zeros((d, 2 * d)), np.ones(d), np.hstack([eye, eye]), np.zeros(d), eye, reals)

    @circuit
    def block(x):
        return add(*x, *ffn(*RMSNorm(1 + norm)(*x)))  # x + relu(1) * ffn(...)

    return block


def check(d=8, f=32, seed=0) -> tuple[str, float]:
    """The block's circuit on random weights, and its largest difference from numpy"""
    rng = np.random.default_rng(seed)
    norm, x = rng.normal(size=d) / 10, rng.normal(size=(16, d))
    w_gate, w_up, w_down = rng.normal(size=(f, d)), rng.normal(size=(f, d)), rng.normal(size=(d, f))
    graph = gemma_ffn(norm, w_gate, w_up, w_down).trace(inputs(*["real"] * d))
    n = x / np.sqrt((x * x).mean(1, keepdims=True) + 1e-6) * (1 + norm)
    g = n @ w_gate.T
    gelu = g / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)))
    return show(graph), float(np.abs(graph.run(x) - (x + (gelu * (n @ w_up.T)) @ w_down.T)).max())


if __name__ == "__main__":
    text, err = check(d=3, f=4)
    print(text, "\nlargest difference from numpy:", err)

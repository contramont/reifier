"""A small transformer in numpy, with lookback attention: the weights of a program on tokens.

A program maps the bits of the last tokens, the oldest token's first, to the bits of the next
token. codes gives each token its bits, its code, so the transformer's greedy next token is the
program's. Its residual stream is the grid's rows, with bits as -1 or 1:

  x = embed[tokens]                    ONE and the token's own code, code[0], code[1], ...
  x = x + attention(norm(x))           row 0: head h copies the code of the token h positions
                                       back into the features of the program's inputs for it
  x = x + layer(norm(x)), per layer    rows 1, 2, ...: emit's layers, as they are
  logits = norm(x) @ embed.T           the outputs are written over the code, so a token's logit
                                       is the bits it shares with them less the bits it does not,
                                       plus what ONE gives every token alike

A head finds the token h back with RoPE, which turns each pair of dimensions of a query or a key
by an angle that grows with its position. Every key is the same before it is turned, and a query
is that key turned back h positions: see lookback_query. A token attends to itself and the
CONTEXT - 1 before it, as in a sliding layer of Gemma. Where a token and those before it are
fewer than the program reads, what follows it means nothing.
"""

from dataclasses import dataclass

import numpy as np

from .emit import Network, emit, rmsnorm
from .grid import ONE, Grid, held, layout
from .trace import Trace

PAIRS = 8  # of dimensions of a query or a key
CONTEXT, MARGIN = 512, 30.0  # a head scores its token MARGIN more than any other in CONTEXT


@dataclass
class Transformer:
    codes: dict[int, list[int]]  # the bits of each token
    embed: np.ndarray  # [token, feature], tied to the logits
    turns: np.ndarray  # RoPE turns pair i of dimensions by turns[i] radians a position
    q: np.ndarray  # [head, dimension, feature]: the first of each pair, then the second
    k: np.ndarray  # [dimension, feature], the same for every head
    v: np.ndarray  # [bit, feature], the same for every head
    o: np.ndarray  # [head, feature, bit]
    net: Network  # the feed-forward layers, and the grid that names every feature

    def run(self, tokens) -> tuple[list[np.ndarray], np.ndarray]:
        """The stream [position, feature] at every depth of the grid, and the logits"""
        grid, x, pos = self.net.grid, self.embed[np.asarray(tokens)], np.arange(len(tokens))
        embedded = [s if f <= len(self.v) else None for f, s in enumerate(grid[0])]
        z = rmsnorm(x, embedded, self.net.one)  # an embedding holds ONE and the code, no more
        seen = (pos[:, None] >= pos) & (pos[:, None] - pos < CONTEXT)
        keys, values = rope(z @ self.k.T, pos, self.turns), z @ self.v.T
        for q, o in zip(self.q, self.o):
            scores = np.where(seen, rope(z @ q.T, pos, self.turns) @ keys.T, -np.inf)
            a = np.exp(scores - scores.max(1, keepdims=True))
            x = x + (a / a.sum(1, keepdims=True)) @ values @ o.T
        xs = self.net.run(x)
        return xs, rmsnorm(xs[-1], grid[-1], self.net.one) @ self.embed.T

    def generate(self, tokens, n: int) -> list[int]:
        """tokens, as many as the program reads at least, and the n greedy next ones"""
        tokens = list(tokens)
        for _ in range(n):
            tokens.append(int(self.run(tokens)[1][-1].argmax()))
        return tokens


def rope(x, pos, turns):
    """x[position, dimension] with each pair of dimensions (i, i + half) turned by pos * turns[i]"""
    (a, b), angle = np.split(x, 2, 1), np.outer(pos, turns)
    return np.hstack([a * np.cos(angle) - b * np.sin(angle), b * np.cos(angle) + a * np.sin(angle)])


def lookback_query(h: int, turns):
    """The query of a head that attends to the token h back, on pairs of dimensions that RoPE
    turns by `turns` radians a position, the first of each pair and then the second. Every key is
    1 in the first of each pair, and the query is that key turned back h positions: the token e
    positions from the one h back scores sum(cos(e turns)), the most at e = 0"""
    return np.r_[np.cos(turns * h), -np.sin(turns * h)]


def runner_up(turns, n: int) -> float:
    """The largest mean(cos(e turns)) over offsets 0 < e < n: the score that a head of
    lookback_query gives the best wrong token within n, as a fraction of its own token's"""
    return np.cos(np.outer(np.arange(1, n), turns)).mean(1).max()


def token_grid(t: Trace) -> Grid:
    """The grid of a program on tokens, held: layout's, with features 1, 2, ... for code[0],
    code[1], ..., the bits of the token's own code. Output i is moved from the feature that
    layout gives it to feature 1 + i, so that the layer that makes it writes it over code[i],
    where the logits read it: so each output must be a signal of its own that an MLP makes"""
    code = [f"code[{i}]" for i in range(len(t.outputs))]
    grid = [[ONE, *code, *row[1:]] for row in layout(t)]
    for i, y in enumerate(t.outputs):
        for row in grid:
            if y in row:
                row[row.index(y)] = None
                row[1 + i] = y
    return held(grid)


def transformer(t: Trace, codes: dict) -> Transformer:
    """t, a program from the bits of the last tokens, the oldest token's first, to the bits of the
    next, as a transformer on the tokens of codes {token: its bits}"""
    bits, grid = len(t.outputs), token_grid(t)  # the bits of a code
    net = emit(t, grid, "geglu", skip=True, norm=True)
    width, window = len(grid[0]), len(t.inputs) // bits  # features, and the tokens that t reads
    embed = np.zeros((max(codes) + 1, width))
    for token, code in codes.items():
        embed[token, : 1 + bits] = [net.one, *(2 * np.array(code) - 1)]
    turns = 10000.0 ** -(np.arange(PAIRS) / PAIRS)
    scale = MARGIN / (PAIRS * (1 - runner_up(turns, CONTEXT)))  # its own token scores scale * PAIRS
    q, k = np.zeros((window, 2 * PAIRS, width)), np.zeros((2 * PAIRS, width))
    v, o = np.zeros((bits, width)), np.zeros((window, width, bits))
    k[:PAIRS, 0] = 1  # a key is 1 in the first of each pair: it reads ONE, as a query does
    v[range(bits), range(1, 1 + bits)] = 1  # a value is the code
    for h in range(window):  # head h: the code of the token h back, into its inputs' features
        q[h, :, 0] = scale * lookback_query(h, turns)
        first = (window - 1 - h) * bits  # the program reads the oldest token first
        o[h, [grid[0].index(x) for x in t.inputs[first : first + bits]], range(bits)] = 1
    return Transformer(codes, embed, turns, q, k, v, o, net)

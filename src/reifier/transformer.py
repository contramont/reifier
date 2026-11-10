"""A small transformer in numpy, with lookback attention: the weights of a program on tokens.

A program maps the bits of the last tokens, the oldest token's first, to the bits of the next
token. codes gives each token its bits, its code, so the transformer's greedy next token is the
program's. Its residual stream is the grid's rows, with bits as -1 or 1:

  x = embed[tokens]                    ONE and the token's own code, code[0], code[1], ...
  x = x + attention(norm(x))           row 0: head h copies the code of the token h back into the
                                       features of the program's inputs for that token
  x = x + layer(norm(x)), per layer    rows 1, 2, ...: emit's layers, as they are
  logits = norm(x) @ embed.T           the outputs are pinned over the code, so a token's logit is
                                       the bits it shares with them less the bits it does not,
                                       plus what ONE gives every token alike

A head finds the token h back with RoPE, which turns each pair of dimensions of a query or a key
by an angle that grows with its position. Every key is the same before it is turned, and a query
is that key turned back h positions: see lookback_qk. A token attends to itself and the
CONTEXT - 1 before it, as in a sliding layer of Gemma.
"""

from dataclasses import dataclass

import numpy as np

from .emit import Network, check_rows, emit, rmsnorm
from .grid import ONE, Grid, held, layout
from .trace import Trace

CONTEXT, MARGIN = 512, 30.0  # a head scores its token MARGIN more than any other in CONTEXT


@dataclass
class Transformer:
    codes: dict[int, list[int]]  # the bits of each token
    embed: np.ndarray  # [token, feature], tied to the logits
    turns: np.ndarray  # RoPE turns pair i of a head's dimensions by turns[i] radians a position
    q: np.ndarray  # [head, dimension, feature]: the first of each pair, then the second
    k: np.ndarray  # [head, dimension, feature]
    v: np.ndarray  # [head, dimension, feature]: dimension i is bit i of a code
    o: np.ndarray  # [head, feature, dimension]
    net: Network  # the feed-forward layers, and the grid that names every feature

    def run(self, tokens) -> tuple[list[np.ndarray], np.ndarray]:
        """The stream [position, feature] at every depth of the grid, and the logits"""
        rows, x, at = self.net.grid.rows, self.embed[np.asarray(tokens)], np.arange(len(tokens))
        b = len(self.net.trace.outputs)  # an embedding holds ONE and the b bits of a code
        z = rmsnorm(x, rows[0][: 1 + b] + [None] * (len(rows[0]) - 1 - b), self.net.one)
        mask = np.where((at[:, None] >= at) & (at[:, None] - at < CONTEXT), 0.0, -np.inf)
        for q, k, v, o in zip(self.q, self.k, self.v, self.o):
            scores = rope(z @ q.T, at, self.turns) @ rope(z @ k.T, at, self.turns).T + mask
            a = np.exp(scores - scores.max(1, keepdims=True))
            x = x + (a / a.sum(1, keepdims=True)) @ (z @ v.T) @ o.T
        xs = self.net.run(x)
        return xs, rmsnorm(xs[-1], rows[-1], self.net.one) @ self.embed.T

    def generate(self, tokens, n: int) -> list[int]:
        """tokens, as many as the program reads at least, and the n greedy next ones"""
        tokens = list(tokens)
        assert len(tokens) >= len(self.q), f"the program reads {len(self.q)} tokens"
        for _ in range(n):
            tokens.append(int(self.run(tokens)[1][-1].argmax()))
        return tokens

    def __str__(self):
        """Each head, by the features it writes, then each hidden unit of each layer"""
        row, lines = self.net.grid.rows[0], [f"{len(self.q)} heads on {len(self.embed)} tokens"]
        for h, o in enumerate(self.o):
            names = " ".join(row[f] for f in np.flatnonzero(o.any(1)))
            lines.append(f"head {h}: the code of the token {h} back -> {names}")
        return "\n".join([*lines, str(self.net)])


def rope(x, at, turns):
    """x[position, dimension] with each pair of dimensions (i, i + half) turned by at * turns[i]"""
    (a, b), angle = np.split(x, 2, 1), np.outer(at, turns)
    return np.hstack([a * np.cos(angle) - b * np.sin(angle), b * np.cos(angle) + a * np.sin(angle)])


def lookback_qk(h: int, turns):
    """A query and a key for a head that attends to the token h back, on pairs of dimensions that
    RoPE turns by `turns` radians a position, the first of each pair and then the second. The key
    is 1 in the first of each pair, and the query is that key turned back h positions: the token
    e positions from the one h back scores sum(cos(e turns)), the most at e = 0"""
    return np.r_[np.cos(turns * h), -np.sin(turns * h)], np.r_[np.ones(len(turns)), 0 * turns]


def runner_up(turns, n: int) -> float:
    """The largest mean(cos(e turns)) over offsets 0 < e < n: the score that a head of
    lookback_qk gives the best wrong token within n, as a part of what it gives its own"""
    return np.cos(np.outer(np.arange(1, n), turns)).mean(1).max()


def token_grid(t: Trace, first: list) -> Grid:
    """The grid of a program on tokens: row 0 is `first`, the token's own code and the program's
    inputs, and the outputs are pinned over the code, each a signal of its own that an MLP makes"""
    b = len(t.outputs)
    if not b or len(t.inputs) % b or len(set(t.outputs)) < b:
        raise ValueError(f"{t.name} must read whole codes, and give a code, each bit a signal")
    code = [f"code[{i}]" for i in range(b)]
    pins = {y: len(first) + i for i, y in enumerate(t.outputs)}
    return held(layout(t, [*first, *code, *t.inputs], pins))


def transformer(t: Trace, codes: dict, activation: str = "geglu") -> Transformer:
    """t, a program from the bits of the last tokens, the oldest token's first, to the bits of the
    next, as a transformer on the tokens of codes {token: its bits}"""
    b, g = len(t.outputs), token_grid(t, [ONE])
    net, n, window = emit(t, g, activation, True, True), len(g.rows[0]), len(t.inputs) // b
    embed = np.zeros((max(codes) + 1, n))
    for token, bits in codes.items():
        embed[token, : 1 + b] = [net.one, *(2 * np.array(bits) - 1)]
    half = max(4, -(-b // 2))  # a head has a dimension for each bit, and 4 pairs at least
    turns = 10000.0 ** -(np.arange(half) / half)
    scale = MARGIN / (half * (1 - runner_up(turns, CONTEXT)))
    q, k, v, o = *np.zeros((3, window, 2 * half, n)), np.zeros((window, n, 2 * half))
    for h in range(window):  # head h: the code of the token h back, into its inputs' features
        q[h, :, 0], k[h, :, 0] = lookback_qk(h, turns)
        v[h, range(b), range(1, 1 + b)] = 1
        o[h, [g.rows[0].index(x) for x in t.inputs[(window - 1 - h) * b :][:b]], range(b)] = 1
    return Transformer(codes, embed, turns, scale * q, k, v, o, net)


def check_transformer(m: Transformer, tokens, tol: float = 1e-3) -> float:
    """Check m against its grid on tokens: at every position with all the tokens the program
    reads behind it, every feature at every depth holds what the grid says for their bits, to
    within tol, and the largest logit is that of the token whose code the trace gives, if it is
    the code of one. Gives the largest difference"""
    t, bits = m.net.trace, np.array([m.codes[token] for token in tokens], float)
    b, window = bits.shape[1], len(m.q)
    assert len(tokens) >= window, f"the program reads {window} tokens"
    rows, logits = m.run(tokens)
    read = np.array([bits[p - window : p].ravel() for p in range(window, len(tokens) + 1)])
    values = t.values(read) | {ONE: np.ones(len(read))}
    values |= {f"code[{i}]": read[:, i - b] for i in range(b)}
    worst = check_rows(m.net, [x[window - 1 :] for x in rows], values, tol)
    token_of = {tuple(code): token for token, code in m.codes.items()}
    outs = np.array([values[y] for y in t.outputs]).T.round().astype(int)
    for p, (out, row) in enumerate(zip(outs, logits[window - 1 :]), window - 1):
        want = token_of.get(tuple(out), row.argmax())
        assert row.argmax() == want, f"position {p}: the next token is {row.argmax()}, not {want}"
    return worst

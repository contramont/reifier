"""Each stage checked against the one before. A check that fails names what is wrong."""

from itertools import pairwise

import numpy as np

from .emit import Network
from .grid import ONE, Grid
from .trace import Trace
from .transformer import Transformer


def cases(n: int) -> np.ndarray:
    """Inputs of n bits to check on: all of them if n <= 8, else 256 random ones"""
    if n <= 8:
        return np.array([[i >> j & 1 for j in range(n)] for i in range(2**n)], float)
    return np.random.default_rng(0).integers(0, 2, (256, n)).astype(float)


def check_trace(t: Trace, reference, xs=None) -> int:
    """t against what it should compute, on each input of xs (default: cases). reference is a Python
    function from a list of bits to the outputs, a bit or a list of them, or another trace, such as
    t before a pass: then every signal they share must agree too. Gives how many inputs it tried"""
    names = sorted([*t.inputs, *(y for m in t.mlps for y in m.outputs)])
    twice = [s for s, other in pairwise(names) if s == other]
    assert not twice, f"{t.name}: two signals are named {twice[0]}"
    xs = cases(len(t.inputs)) if xs is None else np.asarray(xs, float)
    got = t.values(xs)
    if isinstance(reference, Trace):
        want = reference.values(xs)
        pairs = [(s, got[s], want[s]) for s in got if s in want]  # the earliest to differ fails
        outs = [want[y] for y in reference.outputs]
    else:
        pairs, outs = [], np.atleast_2d(np.array([reference([*map(int, x)]) for x in xs], float).T)
    assert len(outs) == len(t.outputs), f"{t.name} gives {len(t.outputs)} outputs, not {len(outs)}"
    pairs += [(y, got[y], out) for y, out in zip(t.outputs, outs)]
    for s, value, should in pairs:  # a name, its values in t, the values it should have
        i = (value != should).argmax()  # the first input on which they differ, if any
        assert value[i] == should[i], (
            f"{s} is {value[i]:g} on {''.join(str(int(v)) for v in xs[i])}, not {should[i]:g}"
        )
    return len(xs)


def check_grid(t: Trace, grid: Grid):
    """grid against t. ONE is feature 0 of every row. Row 0 holds t's inputs, and the last row
    its outputs. A signal keeps its feature. Where one first appears, an MLP makes it: that row
    holds all that the MLP makes, and the row before all that it reads"""
    made = {y: m for m in t.mlps for y in m.outputs}
    assert all(ONE in row[:1] for row in grid), "ONE is not feature 0 of every row"
    assert len({len(row) for row in grid}) == 1, "the rows differ in width"
    assert set(t.inputs) <= set(grid[0]), "depth 0 lacks an input"
    assert set(t.outputs) <= set(grid[-1]), "the last depth lacks an output"
    assert not made.keys() & set(grid[0]), "depth 0 holds a signal that an MLP makes"
    seen = set(grid[0])  # every signal that a cell so far has held
    for d, (before, row) in enumerate(pairwise(grid), 1):
        had, has = set(before), set(row)
        for s in [s for s, old in zip(row, before) if s and s != old]:
            assert s in made and s not in seen, f"{s} is new at depth {d}: no MLP makes it there"
            seen.add(s)
            missing = set(made[s].inputs) - had
            assert not missing, f"{s} is made at depth {d}, but depth {d - 1} lacks {missing}"
            assert has.issuperset(made[s].outputs), f"depth {d} lacks an output of {made[s].name}"


def check_net(net: Network, tol: float = 1e-9):
    """net against its grid, on cases of inputs: every feature at every depth holds the value, in
    the trace, of the signal that the grid names there, and 0 where it names none, to within tol"""
    values = net.trace.values(cases(len(net.trace.inputs)))
    check_streams(net, net.run(np.atleast_2d(net.stream(values, 0))), values, tol)


def check_streams(net: Network, streams, values: dict, tol: float):
    """streams, one at each depth, against what net's grid says for the values {name: value on
    each case} of its signals, to within tol"""
    for d, x in enumerate(streams):
        off = np.abs(x - net.stream(values, d))
        i, f = np.unravel_index(off.argmax(), off.shape)
        assert off[i, f] <= tol, (
            f"depth {d}, feature {f} ({net.grid[d][f] or 'no signal'}) is {x[i, f]:g} on "
            f"case {i}, {off[i, f]:g} from what the grid says"
        )


def check_transformer(m: Transformer, tokens, tol: float = 1e-9):
    """m against its grid on tokens, each with a code, and as many as the program reads at least:
    at every position that is the last of as many tokens as the program reads, every feature at
    every depth holds what the grid says for their bits, to within tol, and the largest logit is
    that of the token whose code the trace gives, if it is the code of one"""
    t, bits = m.net.trace, np.array([m.codes[token] for token in tokens], float)
    b, window = bits.shape[1], len(m.q)
    streams, logits = m.run(tokens)
    read = np.array([bits[p - window : p].ravel() for p in range(window, len(tokens) + 1)])
    values = t.values(read) | {f"code[{i}]": read[:, i - b] for i in range(b)}
    check_streams(m.net, [x[window - 1 :] for x in streams], values, tol)
    token_of = {tuple(code): token for token, code in m.codes.items()}
    outs = np.array([values[y] for y in t.outputs]).T.round().astype(int)
    for p, (out, row) in enumerate(zip(outs, logits[window - 1 :]), window - 1):
        want = token_of.get(tuple(out), row.argmax())
        assert row.argmax() == want, f"position {p}: the next token is {row.argmax()}, not {want}"

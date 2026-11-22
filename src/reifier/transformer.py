"""A small transformer in numpy, shaped as Gemma is: the weights of a program on tokens.

A program maps the bits of the last tokens, the oldest token's first, to the bits of the next
token. codes gives each token its bits, its code, so the transformer's greedy next token is the
program's. Its residual stream is the grid's rows, a bit held as -SIZE or SIZE:

  x = embed[tokens]                    ONE, in ONES features, and the token's own code, after
                                       a <bos>, which is zeros
  x = x + norm(attention(norm(x)))     row 0: head h copies the code of the token h positions
                                       back into the features of the program's inputs for it,
                                       and head 0 erases the code
  x = x + norm(block(norm(x)))         rows 1, 2, ...: a block for each, of GLUs
  logits = norm(x) @ embed.T           the outputs are made where the code was, so a token's
                                       logit is the bits it shares with them less the bits it
                                       does not

Every norm is an RMSNorm with a weight for each feature, as in Gemma, and the weights and
activations are of the sizes of those that a Gemma was trained to: a bit is SIZE in the stream,
as a feature of an embedding is, and READ where a block reads it, a gate that is on is GATE or
a few times that, and a weight of a table of lib.py is under 1 (of a gate of other kinds,
about 1: see block). The norms are what allows it:

- a hidden unit is gelu(g) * v, with g and v sums of what the norm before the block gives, and
  its biases weights on ONE. Were the stream s times as large, each unit would give s * s times
  as much, as long as gelu is relu where the gates are; and the norm after the block divides
  what the block gives by its root mean square. So a block adds what it should whatever is
  beside it in the stream, and whatever the size of what it reads, if that is not a fifth
  smaller than it was made for: it need not know either;
- as long as the root mean square of what it gives is known. A block writes a new bit, as -1 or
  1, only where the row before held nothing, and erases a bit by giving it negated: so each
  feature it writes gets -1 or 1 times one size, whatever the bits are. The grid is laid out
  with rest (see grid.layout) for that. A signal that is 1/2 is held as 0, which is neither: a
  table of lib.py that gives one gives its size to another feature, its rest;
- a gate must be far from 0, as gelu is relu only there. Its sums are whole numbers of some
  step apart, and it is scaled so that the nearest to 0 is GATE: see `sharp`.

And it is not much moved by weights that are a little off, as a network that is trained is
not. A block reads bits that are off and writes bits that are less so:

- a table of lib.py is flat where it is on, and of what its bits are off by it passes on the
  square. So what is off does not add up over depth: each block starts from what the block
  before was off by, not from all that was;
- what erases a bit is the bit itself, times weights: so it leaves a little, as they are off.
  The block after erases the feature again, which leaves a little of that;
- a bias is spread over the ONES features, and a block that is told how many hidden units it
  may have has each of its own as many times as fit (see blocks): what the weights of each
  are off by averages;
- every hidden unit of a block gives as much, and a head's key and value are weights of VALUE:
  noise of one size on every weight, those that are 0 too, is small beside each.

It is right until the weights are off by more than a gate has room for, and then wrong
altogether, where a trained network is a little wrong long before.

A head finds the token h back with RoPE, which turns each pair of dimensions of a query or a key
by an angle that grows with its position. Every key is the same before it is turned, and a query
is that key turned back h positions: see lookback_query. A token attends to itself and the
CONTEXT - 1 before it, as in a sliding layer of Gemma. Where a token and those before it are
fewer than the program reads, what follows it means nothing; nor does what follows a token
that has no code, but the <bos>.
"""

import math
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from .emit import convert, signed
from .grid import ONE, Grid, layout
from .trace import MLP, Trace

PAIRS = 8  # of dimensions of a query or a key: those that RoPE turns fastest in a head of 256
CONTEXT, MARGIN = 512, 14.0  # a head scores its token MARGIN more than any other in CONTEXT
EPS = 1e-6  # under an RMSNorm's square root: a stream of zeros comes out as zeros
SIZE = 2.0  # a bit in the stream, and ONE: as large as a feature of a token's embedding in Gemma
READ, GATE = 6.5, 3.5  # a bit as a block reads it, and a gate where it is nearest 0
LIGHT = 0.45  # the weights of a value, of those of a gate: what it sums is then as Gemma's are
ONES, TIMES = 16, 8  # the features that hold ONE, and the most times a block has each unit
DOWN, VALUE = 0.5, 0.375  # the weight out of a hidden unit, and of a head's key or value
LOGIT = 16.0  # the right token's logit


def rmsnorm(x, weight, rest=0.0, size=None):
    """Gemma's RMSNorm: x over its root mean square, times a weight for each feature. The mean is
    over `size` features, x's and others whose squares sum to `rest`: by default, x's alone"""
    mean = ((x * x).sum(-1, keepdims=True) + rest) / (size or x.shape[-1])
    return weight * x / np.sqrt(mean + EPS)


@dataclass
class Block:
    """A feed-forward block: x + norm(layer(norm(x))). layer is an MLP on features, of geglu,
    with no biases; reads and writes are the weights of the norms before and after it"""

    layer: MLP
    reads: np.ndarray
    writes: np.ndarray

    def __call__(self, x, rest=0.0, size=None):
        """What the block adds to the stream x"""
        return rmsnorm(self.layer(rmsnorm(x, self.reads, rest, size)), self.writes, 0.0, size)


def sharp(g, bias) -> tuple[np.ndarray, np.ndarray]:
    """For gates g @ x + bias on bits x held as -1 or 1, a gate to each row of g: what makes each
    GATE from 0 or more at every x, and whether any x turns the gate on. A gate's sums are its
    lowest, plus whole numbers of a step: twice the largest number that all its weights are whole
    multiples of. The nearest of them to 0, 0 itself aside, is scaled to GATE"""
    whole = np.rint(64 * np.abs(g)).astype(np.int64)
    assert np.allclose(whole, 64 * np.abs(g)), "a gate's weights must be whole 64ths"
    step = np.gcd.reduce(whole, axis=1, initial=0) / 32
    lowest = bias - np.abs(g).sum(1)
    off = np.mod(lowest, np.where(step > 0, step, 1))
    near = np.minimum(off, step - off)  # of a sum from 0, where the gate has weights
    near = np.where(step > 0, np.where(near > 1e-9, near, step), np.abs(bias))
    return GATE / np.where(near > 0, near, 1), bias + np.abs(g).sum(1) > 1e-9


def block(mlps: list[MLP], before, row, size_before, size_row, square, again=()) -> Block:
    """The block that makes row of the grid from the row before, whose features hold the sizes
    given for each row: a bit is plus or minus its size, and square[s] is the mean square of the
    signal s, over its size: 1 for a bit. Its hidden units, in order: those of mlps, which make
    the signals new in row, as GLUs on bits held as -1 or 1; one that erases each feature that
    held a signal, or a ONE, which row does not hold, and each feature of `again`, which the
    block before erased: what that left of it; and one named ONE, for what the units on no bits
    give, the -1 of each new bit that its MLP does not give, and a ONE that is new. Every unit
    gives as much: its value is scaled for that, and the largest weight of a value is LIGHT of
    the largest of a gate, or less where the values are biases, spread over the ONEs. A gate
    whose sums come nearer 0 than a table's has larger weights: up to twice the largest that a
    trained Gemma has, and no more"""
    width, ones = len(row), [f for f, s in enumerate(before) if s == ONE]
    at = {s: f for f, s in enumerate(before) if s and s != ONE}  # where each signal is read
    to = {s: f for f, s in enumerate(row) if s and s != before[f]}  # and where each is written
    assert not [f for f in to.values() if before[f]], "a signal is written over another: no rest"
    gone = [f for f, s in enumerate(before) if s and s != row[f]]
    parts, hidden = [], []  # of each MLP: its gates and values, their features, its outputs
    fixed = np.zeros(width)  # what a new feature gets whatever the bits: a ONE, or a bit's -1
    fixed[list(to.values())] = [1.0 if s == ONE else -1.0 for s in to]
    for m in mlps:
        ins, where = np.unique(np.array([at[x] for x in m.inputs], int), return_inverse=True)
        outs = [to[y] for y in m.outputs]
        m = signed(convert(m, "reglu"))
        wi = np.zeros((len(m.wi), len(ins)))
        np.add.at(wi.T, where, m.wi.T)  # on features: an input read twice adds up
        (g, v), (gb, vb) = np.split(wi, 2), np.split(m.b, 2)
        on_bits = g.any(1) | v.any(1)
        fixed[outs] += m.wo[:, ~on_bits] @ (np.maximum(gb, 0) * vb)[~on_bits]  # units on no bits
        s, live = sharp(g, gb)
        most = np.abs(m.wo).max(0)  # of what each unit gives
        keep = on_bits & live & (most > 0)  # a unit that no bits turn on gives nothing
        weight = (np.abs(g).max(1, initial=0) * s)[keep].max(initial=0) / READ
        # a gate's weights are large if its sums come near 0, or if its own weights are large
        assert weight <= 2, f"{m.name}: a gate sums too near 0: its weights would be {weight:.3g}"
        heavy = np.abs(v).max(1, initial=0)  # the largest weight of each value, or its bias
        heavy = np.where(heavy > 0, heavy, np.abs(vb)) * most / s
        g, v, wo = np.c_[g * s[:, None], gb * s], np.c_[v, vb], m.wo / (most + (most == 0))
        parts.append((g[keep], v[keep], ins, outs, wo[:, keep], (most / s)[keep], heavy[keep]))
        hidden += [h for h, kept in zip(m.hidden, keep) if kept]
    # a unit gives its outputs' weights times `whole`: its value is scaled for that, by how much
    # its gate was, and the heaviest of the values is then LIGHT of the heaviest of the gates
    large = max([np.abs(g[:, :-1]).max(initial=0) for g, *_ in parts], default=0) or GATE
    heavy = max([h.max(initial=0) for *_, h in parts], default=0) or large / GATE**2
    whole = LIGHT * large / heavy
    erased = [*gone, *again]
    units, extra = len(hidden), len(erased) + bool(fixed.any())
    gate, value = np.zeros((2, units + extra, width))
    down = np.zeros((width, units + extra))
    start = 0
    for g, v, ins, outs, wo, scale, _ in parts:
        j = slice(start, start := start + len(g))
        v = v * (whole * scale)[:, None]
        gate[j][:, ins], value[j][:, ins], down[outs, j] = g[:, :-1], v[:, :-1], wo
        gate[j][:, ones] += g[:, -1:] / len(ones)  # biases are weights on the ONEs
        value[j][:, ones] += v[:, -1:] / len(ones)
    # a unit that erases is on whatever the bits, and its value is the feature: their product is
    # `whole`. So is that of the unit ONE, whose value is ONE: it gives each feature what is fixed
    high = max(whole / (LIGHT * large), GATE)
    gate[np.ix_(range(units, units + len(erased)), ones)] = high / len(ones)
    value[range(units, units + len(erased)), erased] = whole / high
    down[erased, range(units, units + len(erased))] = -1
    if fixed.any():
        high = max(math.sqrt(whole * np.abs(fixed).max()), GATE)
        gate[-1, ones] = high / len(ones)
        value[-1, ones] = whole * np.abs(fixed).max() / high / len(ones)
        down[:, -1] = fixed / np.abs(fixed).max()
    hidden += [f"erase {before[f] or 'what is left'}" for f in erased] + [ONE] * bool(fixed.any())
    # the norm before gives each bit read as READ, and the norm after gives each feature its size
    read, top = np.flatnonzero(gate.any(0) | value.any(0)), size_before.max()
    reads, writes = np.zeros(width), np.zeros(width)
    reads[read] = READ / np.where(size_before > 0, size_before, top)[read]
    wrote = sorted({*to.values(), *gone})
    writes[list(again)] = top  # what is left of a feature is small, whatever the feature held
    writes[wrote] = np.where([s is not None for s in row], size_row, size_before)[wrote]
    writes *= math.sqrt(sum(square.get(row[f] or before[f], 1.0) for f in wrote))
    wi = np.vstack([gate, value]) / READ
    reads_names = [s or f"f{f}" for f, s in enumerate(before)]
    layer = MLP("layer", reads_names, [f"f{f}" for f in range(width)], hidden, wi,
                np.zeros(len(wi)), "geglu", DOWN * down)  # fmt: skip
    return Block(layer, reads, writes)


def blocks(t, grid, sizes, square, rest=0.0, size=None, hidden=None, again=()) -> list[Block]:
    """A block for each row of grid after the first. sizes[d][f] is the size of what feature f
    holds at depth d, 0 where it holds nothing. The norms are over `size` features, the grid's
    and others whose squares sum to `rest`, and they scale reads and writes for that: a norm
    divides by the root mean square, which the sizes give, with square, the mean square of each
    signal over its size (see squares). An MLP is at the depth where its outputs first appear. A
    block erases again what the block before it erased, and the first, the features of `again`,
    which the heads did. If a block may have `hidden` hidden units, each of its units is there
    as many times as fit, up to TIMES: they give as many times as much, which the norm after
    the block divides by, and are off by less, between them"""
    size, out, seen, again = size or len(grid[0]), [], set(grid[0]), list(again)
    for d, (before, row) in enumerate(pairwise(grid), 1):
        new = set(row) - seen
        mlps = [m for m in t.mlps if m.outputs[0] in new]
        seen.update(row)
        b = block(mlps, before, row, sizes[d - 1], sizes[d], square, again)
        again = [f for f, s in enumerate(before) if s and s != row[f] and not row[f]]
        n = len(b.layer.hidden)
        times = min(TIMES, max(1, (hidden or n) // max(n, 1)))
        wi = np.vstack([np.tile(half, (times, 1)) for half in np.split(b.layer.wi, 2)])
        wo, names = np.tile(b.layer.wo, times), b.layer.hidden * times
        layer = replace(b.layer, hidden=names, wi=wi, b=np.zeros(len(wi)), wo=wo)
        held = np.array([square.get(s, 1.0) if s else 0.0 for s in before])
        rms = math.sqrt(((held * sizes[d - 1] ** 2).sum() + rest) / size)
        out.append(Block(layer, b.reads * rms, b.writes / math.sqrt(size)))
    return out


def squares(t: Trace, codes: dict) -> dict[str, float]:
    """The mean square of each signal of t as the stream holds it, over its size: of 2 v - 1 for
    its values v, on the codes of tokens at random. It is 1 for a bit, and less for a signal
    that can be 1/2, which is held as 0"""
    tokens, window = list(codes), len(t.inputs) // len(t.outputs)
    drawn = np.random.default_rng(0).integers(0, len(tokens), (64, window))
    read = np.array([[codes[tokens[i]] for i in row] for row in drawn], float)
    values = t.values(read.reshape(64, -1))
    return {s: float(np.mean((2 * v - 1) ** 2)) for s, v in values.items()}


@dataclass
class Transformer:
    trace: Trace
    grid: Grid  # the name of every feature at every depth
    sizes: np.ndarray  # [depth, feature]: the size of what the feature holds there
    codes: dict[int, list[int]]  # the bits of each token
    embed: np.ndarray  # [token, feature], of ONE and the code: tied to the logits
    turns: np.ndarray  # RoPE turns pair i of dimensions by turns[i] radians a position
    q: np.ndarray  # [head, dimension, feature]: the first of each pair, then the second
    k: np.ndarray  # [dimension, feature], the same for every head
    v: np.ndarray  # [bit, feature], the same for every head: the code
    o: np.ndarray  # [head, feature, bit]
    reads: np.ndarray  # the weights of the norm before the attention,
    writes: np.ndarray  # of the norm after it,
    blocks: list[Block]  # the feed-forward blocks, with theirs: blocks[d - 1] makes row d,
    out: np.ndarray  # and of the last norm, before the logits
    rest: float = 0.0  # the squares of what else the stream holds, and the features that
    size: int | None = None  # the norms are over, if more than the grid's (see beside.py)

    def stream(self, values: dict, d: int) -> np.ndarray:
        """What the stream should hold at depth d, for the values {name: value} of the signals
        of row d: a bit v as 2 v - 1 times its size, ONE as its size, 0 where the row has no
        signal"""
        cells = [0.0 if s is None else 1.0 if s == ONE else 2 * values[s] - 1 for s in self.grid[d]]
        return self.sizes[d] * np.stack(np.broadcast_arrays(*cells), -1)

    def run(self, tokens) -> tuple[list[np.ndarray], np.ndarray]:
        """The stream [position, feature] at every depth of the grid, and the logits, of a <bos>
        and then tokens"""
        pos, n = np.arange(1 + len(tokens)), self.embed.shape[1]
        x = np.zeros((len(pos), len(self.grid[0])))  # position 0 is the <bos>, and stays zeros
        x[1:, :n] = self.embed[np.asarray(tokens, int)]
        rest = self.rest * (pos > 0)[:, None]  # no token is zeros, but the <bos>
        z = rmsnorm(x, self.reads, rest, self.size)
        seen = (pos[:, None] >= pos) & (pos[:, None] - pos < CONTEXT)
        keys, values, y = rope(z @ self.k.T, pos, self.turns), z @ self.v.T, 0.0
        for q, o in zip(self.q, self.o):
            scores = np.where(seen, rope(z @ q.T, pos, self.turns) @ keys.T, -np.inf)
            a = np.exp(scores - scores.max(1, keepdims=True))
            y = y + (a / a.sum(1, keepdims=True)) @ values @ o.T
        xs = [x + rmsnorm(y, self.writes, 0.0, self.size)]
        for b in self.blocks:
            xs.append(xs[-1] + b(xs[-1], rest, self.size))
        return xs, rmsnorm(xs[-1], self.out, rest, self.size)[:, :n] @ self.embed.T

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


def token_grid(t: Trace, carry: bool = False) -> Grid:
    """The grid of a program on tokens: layout's, with rest, on features for ONE, then for the
    token's own code, where it is embedded, then ONES - 1 more that hold ONE, then for the inputs
    that are read, and so on. The heads erase the code, so row 0 holds none of it, and output i
    is made in feature 1 + i, where bit i of the code was and the logits read it: so each output
    must be a signal of its own that an MLP makes. With carry, the last row holds no ONE but a
    new one, in a feature of its own, the last: what it holds is to be carried on (see
    beside.py), and so its block must make every output"""
    made, read = {y for m in t.mlps for y in m.outputs}, {x for m in t.mlps for x in m.inputs}
    assert len(made.intersection(t.outputs)) == len(t.outputs), (
        f"{t.name}: each output must be a signal of its own that an MLP makes"
    )
    first = [ONE, *[None] * len(t.outputs), *[ONE] * (ONES - 1)]
    first += [x for x in t.inputs if x in read]
    grid = layout(t, True, first, {y: 1 + i for i, y in enumerate(t.outputs)})
    if carry:
        early = set(t.outputs).intersection(grid[-2])
        assert not early, f"{t.name}: its last block must make every output"
        grid = [row + [None] for row in grid]
        grid[-1] = [None if s == ONE else s for s in grid[-1][:-1]] + [ONE]
    return grid


def heads(grid: Grid, inputs: list[str], bits: int, size: int, rest=0.0):
    """The attention that makes row 0 of grid from an embedding of ONE, in the features that hold
    it there, and of a code of `bits` bits, bit i in feature 1 + i, each SIZE. Gives q, k, v, o
    and the weights of the norms before and after, which are over `size` features whose other
    squares sum to `rest`. Head h reads ONE alone, to find the token h back, and copies its code
    into the features of the inputs for it, the oldest token's first: of those that the grid
    has, as the program reads them. Head 0 also gives the code, negated, to the features it is
    embedded in. A key and a value have weights of VALUE, and an output of 1; a query's are as
    much smaller as a key is large: the norms of a Gemma make of a query and a key what they
    will"""
    width, window, code = len(grid[0]), len(inputs) // bits, 1 + np.arange(bits)
    ones = [f for f, s in enumerate(grid[0]) if s == ONE]
    turns = 10000.0 ** -(np.arange(PAIRS) / 128)
    scale = MARGIN / (PAIRS * (1 - runner_up(turns, CONTEXT)))  # its own token scores scale * PAIRS
    q, k = np.zeros((window, 2 * PAIRS, width)), np.zeros((2 * PAIRS, width))
    v, o = np.zeros((bits, width)), np.zeros((window, width, bits))
    k[:PAIRS, ones] = VALUE  # a key is the same in the first of each pair: it reads ONE,
    key = VALUE * READ * len(ones)  # as a query does, which is as much smaller as a key is large
    v[range(bits), code], o[0, code, range(bits)] = VALUE, -1  # a value is the code
    for h in range(window):  # head h: the code of the token h back, into its inputs' features
        q[h][:, ones] = scale * lookback_query(h, turns)[:, None] / key / READ / len(ones)
        first = (window - 1 - h) * bits  # the program reads the oldest token first
        for i, x in enumerate(inputs[first : first + bits]):
            o[h, [f for f, s in enumerate(grid[0]) if s == x], i] = 1
    reads, writes = np.zeros(width), np.zeros(width)
    embedded = [*ones, *code]
    reads[embedded] = READ / SIZE * math.sqrt((len(embedded) * SIZE**2 + rest) / size)
    wrote = np.flatnonzero(o.any((0, 2)))
    writes[wrote] = SIZE * math.sqrt(len(wrote) / size)
    return turns, q, k, v, o, reads, writes


def transformer(t: Trace, codes: dict, hidden=None, rest=0.0, size=None, carry=None) -> Transformer:
    """t, a program from the bits of the last tokens, the oldest token's first, to the bits of the
    next, as a transformer on the tokens of codes {token: its bits}. A code has a bit for each of
    t's outputs, and t reads the codes of len(t.inputs) // len(t.outputs) tokens, or some of
    their bits. With hidden, the hidden units that a block may have, each block has its units
    as many times as fit (see blocks). The rest is for a transformer that is part of a larger
    one (see beside.py): the norms are over `size` features, whose other squares sum to `rest`,
    and what the last row holds is as small as `carry`, for other layers to read: the logits
    are then not its own"""
    bits = len(t.outputs)  # of a code
    assert t.inputs and len(t.inputs) % bits == 0, (
        f"{t.name} reads {len(t.inputs)} bits: no codes of {bits}"
    )
    grid = token_grid(t, carry is not None)
    width, square = len(grid[0]), squares(t, codes)
    sizes = SIZE * np.array([[s is not None for s in row] for row in grid], float)
    embed = np.zeros((max(codes) + 1, bits + ONES))  # of each token that has a code: ONE, its
    embed[list(codes)] = SIZE  # code, and ONE in ONES - 1 features more
    embed[list(codes), 1 : 1 + bits] = SIZE * (2.0 * np.array(list(codes.values())) - 1)
    out = np.zeros(width)  # the last norm: the right token's logit is LOGIT
    if carry is None:
        held = sum(square.get(s, 1.0) * z * z for s, z in zip(grid[-1], sizes[-1]) if s)
        out[1 : 1 + bits] = (
            LOGIT / (bits * SIZE * SIZE) * math.sqrt((held + rest) / (size or width))
        )
    else:
        sizes[-1] *= carry / SIZE
    attention = heads(grid, t.inputs, bits, size or width, rest)
    layers = blocks(t, grid, sizes, square, rest, size, hidden, range(1, 1 + bits))
    return Transformer(t, grid, sizes, codes, embed, *attention, layers, out, rest, size)

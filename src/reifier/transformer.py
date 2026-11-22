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
- as long as the root mean square of what it gives is known. A block erases a bit by giving it
  negated, and writes a new one, as -1 or 1, where the row before held nothing: such a feature
  gets -1 or 1 times one size, whatever the bits are. A block also writes a new bit over an
  old one that it reads for the last time, and that is what lets a circuit do with few
  features: of Keccak, little more than its state. The feature then gets the new bit less the
  old, which is nothing, or twice a bit; so the block gives the new plus the old as well, to a
  row that is no feature's, which the norm after counts in its mean square and drops. Between
  them the two are as large whatever the bits are (see block);
- a signal that is 1/2 is held as 0, which is neither bit: a table of lib.py that gives one
  gives its size to another feature, its rest. Neither of the two is written over a signal,
  nor a signal over it: the grid is laid out with rest (see grid.layout) for that;
- a gate must be far from 0, as gelu is relu only there. Its sums are whole numbers of some
  step apart, and it is scaled so that the nearest to 0 is GATE: see `sharp`.

Some activations are larger. A unit that is off has its gate the further under 0, the further
what it reads is from what turns it on (see lib.py). Of a table of six bits that are all wrong
it is 16 GATE under 0, and gate and value are each up to a fifth more than the largest of
Gemma 3 270M's own on text. Of lib.steps on ten bits, at the count furthest from its step, it
is 39 GATE under 0, nearly three times Gemma's largest; and where a unit of steps is far past
its step, it gives a tenth more than the largest hidden unit of Gemma's.

And it is not much moved by weights that are a little off, as a network that is trained is
not. A block reads bits that are off and writes bits that are less so:

- a table of lib.py is flat where it is on, and of what its bits are off by it passes on the
  square, as a cover does; steps pass on none of it, while their count is less than a quarter
  off. So what is off does not add up over depth: each block starts from what the block
  before was off by, not from all that was;
- what erases a bit is the bit itself, times weights: so it leaves a little, as they are off.
  Where the feature then holds nothing, the block after erases it again, which leaves a
  little of that. Where a new bit was written over the old one, what is left is in the new
  bit, and stays: so a bit that is written over another is off by more than one that is
  written where nothing was, by what its eraser is off by. That is the price of the fewer
  features, and grid.layout writes over a bit only where no feature is empty;
- a bias is spread over the ONES features, and a block that is told how many hidden units it
  may have has each of its own as many times as fit, and those that it helps most, more
  times still (see repeats): what the weights of each are off by averages;
- the ONEs are not all alike, in the embedding or in what each has of a bias (see uneven):
  weights that are all alike are all rounded alike, and what they are off by adds up;
- every hidden unit of a block gives as much, between its copies, and a head's key and value
  are weights of VALUE: noise of one size on every weight, those that are 0 too, is small
  beside each.

It is right until the weights are off by more than a gate has room for, and then wrong
altogether, where a trained network is a little wrong long before. How far that is depends on
its circuits: examples/keccak.py says it for each kind of round of Keccak.

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
SIZE = 2.0  # a bit in the stream, and a ONE, more or less (see uneven): as large as a feature of
# a token's embedding in Gemma
READ, GATE = 6.5, 3.5  # a bit as a block reads it, and a gate where it is nearest 0
LIGHT = 0.45  # the weights of a value, of those of a gate: what it sums is then as Gemma's are
ONES, TIMES = 16, 4  # the features that hold ONE, and the most times a block has each unit
UNEVEN = 0.5  # the ONEs differ by this much of SIZE, either way, and in what each has of a bias
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
    with no biases; reads and writes are the weights of the norms before and after it. The
    layer may give rows after those of the features, one for each bit that it writes over
    another (see block)"""

    layer: MLP
    reads: np.ndarray
    writes: np.ndarray

    def __call__(self, x, rest=0.0, size=None):
        """What the block adds to the stream x. The norm after takes the mean square of all
        that the layer gives, and drops the rows that are no feature's"""
        given, width = self.layer(rmsnorm(x, self.reads, rest, size)), len(self.writes)
        mean = (given * given).sum(-1, keepdims=True) / (size or width)
        return self.writes * given[..., :width] / np.sqrt(mean + EPS)


def uneven(n: int) -> np.ndarray:
    """n numbers from 1 - UNEVEN to 1 + UNEVEN, in equal steps, whose mean is 1: what each of n
    ONEs is, of SIZE, where it is embedded, and what each has of a bias, of an even share. A
    block reads each ONE as READ, whatever its size, so a bias is what it was. Were the ONEs
    all alike, their weights would be too, and rounded to a few bits they would all be off the
    same way, and a bias by as much as each: of these each is off by another part of a step,
    and their sum by little"""
    return 1 + UNEVEN * np.linspace(-1, 1, n) if n > 1 else np.ones(n)


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
    give, the -1 of each new bit that its MLP does not give, and a ONE that is new. Where a
    new bit t is written over an old one x, the feature gets t from its MLP and x, negated,
    from its eraser; and the layer gives a row more, after those of the features, which gets t
    from the same units and x as it is. The squares of t - x and t + x sum to twice those of t
    and of x, whatever the two bits are, so the norm after, which counts that row in its mean
    square and drops it, divides by what it was made for. Not so for a signal that can be 1/2,
    which is held as 0: neither t nor x may be one. Every unit gives as much: its value is
    scaled for that, and the largest weight of a value is LIGHT of the largest of a gate, or
    less where the values are biases, spread over the ONEs. A gate whose sums come nearer 0
    than a table's has larger weights, up to 2 and no more, where the largest of a gate of
    Gemma 3 270M's own is 1.17"""
    width, ones = len(row), [f for f, s in enumerate(before) if s == ONE]
    share = uneven(len(ones)) / len(ones)  # of a bias, for each ONE
    at = {s: f for f, s in enumerate(before) if s and s != ONE}  # where each signal is read
    to = {s: f for f, s in enumerate(row) if s and s != before[f]}  # and where each is written
    gone = [f for f, s in enumerate(before) if s and s != row[f]]
    over = [f for f in to.values() if before[f]]  # written where the row before held another
    bits = [abs(square.get(s, 1.0) - 1) < 1e-9 for f in over for s in (before[f], row[f])]
    assert all(bits), "a signal that can be 1/2 is written over another, or another over it"
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
    down = np.zeros((width + len(over), units + extra))  # a row more for each of `over`
    start = 0
    for g, v, ins, outs, wo, scale, _ in parts:
        j = slice(start, start := start + len(g))
        v = v * (whole * scale)[:, None]
        gate[j][:, ins], value[j][:, ins], down[outs, j] = g[:, :-1], v[:, :-1], wo
        gate[j][:, ones] += g[:, -1:] * share  # biases are weights on the ONEs
        value[j][:, ones] += v[:, -1:] * share
    # a unit that erases is on whatever the bits, and its value is the feature: their product is
    # `whole`. So is that of the unit ONE, whose value is ONE: it gives each feature what is fixed
    high = max(whole / (LIGHT * large), GATE)
    gate[np.ix_(range(units, units + len(erased)), ones)] = high * share
    value[range(units, units + len(erased)), erased] = whole / high
    down[erased, range(units, units + len(erased))] = -1
    if fixed.any():
        high = max(math.sqrt(whole * np.abs(fixed).max()), GATE)
        gate[-1, ones] = high * share
        value[-1, ones] = whole * np.abs(fixed).max() / high * share
        down[:width, -1] = fixed / np.abs(fixed).max()
    # a feature of `over` gets t - x, and its row more t + x, from the same units. The norm
    # after makes of what the feature gets the size of the new bit, t, so the eraser gives the
    # old one, x, as many times that as it is: `older`
    older = (size_before / np.where(size_row > 0, size_row, 1))[over]
    for i, (f, old) in enumerate(zip(over, older)):
        value[units + erased.index(f), f] *= old
        down[width + i] = down[f]
        down[width + i, units + erased.index(f)] = 1
    # the size of a bit in the row before, which a ONE need not have (see uneven): what is left
    # of a feature that was erased is read, and erased again, as if it were as large
    top = max([z for z, s in zip(size_before, before) if s != ONE], default=size_before.max())
    for f in set(again).intersection(to.values()):  # what was left where a signal is new: the
        value[units + erased.index(f), f] *= top / size_row[f]  # norm after makes of it what
        # the new signal is to be, so it is given as many of those as it is
    hidden += [f"erase {before[f] or 'what is left'}" for f in erased] + [ONE] * bool(fixed.any())
    # the norm before gives each bit read as READ, and the norm after gives each feature its size
    read = np.flatnonzero(gate.any(0) | value.any(0))
    reads, writes = np.zeros(width), np.zeros(width)
    reads[read] = READ / np.where(size_before > 0, size_before, top)[read]
    wrote = sorted({*to.values(), *gone})
    writes[list(again)] = top  # what is left of a feature is small, whatever the feature held
    writes[wrote] = np.where([s is not None for s in row], size_row, size_before)[wrote]
    given = sum(square.get(row[f] or before[f], 1.0) for f in wrote)  # the squares of all it
    given += sum(1 + 2 * old * old for old in older)  # gives: t - x and t + x are 2 + 2 x x
    writes *= math.sqrt(given)
    wi = np.vstack([gate, value]) / READ
    reads_names = [s or f"f{f}" for f, s in enumerate(before)]
    gives = [f"f{f}" for f in range(width)] + [f"f{f}, dropped" for f in over]
    layer = MLP("layer", reads_names, gives, hidden, wi, np.zeros(len(wi)), "geglu", DOWN * down)
    return Block(layer, reads, writes)


def blocks(t, grid, sizes, square, rest=0.0, size=None, hidden=None, again=()) -> list[Block]:
    """A block for each row of grid after the first. sizes[d][f] is the size of what feature f
    holds at depth d, 0 where it holds nothing. The norms are over `size` features, the grid's
    and others whose squares sum to `rest`, and they scale reads and writes for that: a norm
    divides by the root mean square, which the sizes give, with square, the mean square of each
    signal over its size (see squares). An MLP is at the depth where its outputs first appear. A
    block erases again what the block before it erased and left with no signal, and the first,
    the features of `again`, which the heads did. If a block may have `hidden` hidden units,
    each of its units is there as many times as `repeats` says, and one that is there more
    times than the fewest gives as much less each time: its copies are off by less, between
    them, and the norm after the block divides by what they all give"""
    size, out, seen, again = size or len(grid[0]), [], set(grid[0]), list(again)
    for d, (before, row) in enumerate(pairwise(grid), 1):
        new = set(row) - seen
        mlps = [m for m in t.mlps if m.outputs[0] in new]
        seen.update(row)
        b = block(mlps, before, row, sizes[d - 1], sizes[d], square, again)
        again = [f for f, s in enumerate(before) if s and s != row[f] and not row[f]]
        over = {f"erase {s}" for s, y in zip(before, row) if s and y and s != y}
        copies = repeats(b.layer, [s == ONE for s in before], over, hidden)
        order = [j for k in range(copies.max(initial=0)) for j in np.flatnonzero(copies > k)]
        wi = np.vstack([half[order] for half in np.split(b.layer.wi, 2)])  # all the units once,
        wo = (b.layer.wo * (copies.min(initial=TIMES) / copies))[:, order]  # then those that
        names = [b.layer.hidden[j] for j in order]  # are there twice, and so on
        layer = replace(b.layer, hidden=names, wi=wi, b=np.zeros(len(wi)), wo=wo)
        held = np.array([square.get(s, 1.0) if s else 0.0 for s in before])
        rms = math.sqrt(((held * sizes[d - 1] ** 2).sum() + rest) / size)
        out.append(Block(layer, b.reads * rms, b.writes / math.sqrt(size)))
    return out


def repeats(layer: MLP, ones, over: set, hidden=None) -> np.ndarray:
    """How many times each hidden unit of a block's layer is there, if the block may have
    `hidden` units. All are there as many times as fit, up to TIMES. The units that are then
    left go, a copy at a time, to the group that a copy more helps most, for the units that it
    takes. A group is all the units of an MLP, which make its signal between them, or one unit
    of `over`, the names of those that erase a signal that another is written over: what such
    a unit is off by stays in the new signal. A unit's weights make what it gives off by r of
    their noise, squared: the squares of the weights of its gate, over the square of their sum
    where it is on, as much for its value, and 1 for its weight out; `ones` says which
    features hold ONE. k copies make that r / k, so one more takes r / (k (k + 1)) off. One
    unit of a table is on at a time, so its signal is off by what one unit is: a copy more of
    a group takes the mean r of its units over k (k + 1) off, for as many units as it has. Of
    a cover or of steps, more units are on at once, and this counts them as one"""
    n = len(layer.hidden)
    copies = np.full(n, min(TIMES, max(1, (hidden or n) // max(n, 1))))
    spare, ones, r = (hidden or n) - copies.sum(), np.array(ones, bool), np.ones(n)
    for i, half in enumerate(np.split(layer.wi, 2)):  # the gates, then the values
        bias, bits = half[:, ones].sum(1), np.abs(half[:, ~ones]).sum(1)
        on = np.abs(np.abs(bias) - bits if i else bias + bits)  # a value is its bias less the bits
        on = np.where((bias != 0) & (bits > 0), on, np.abs(bias) + bits)
        r += (half * half).sum(1) / np.where(on > 0, on * on, 1)
    groups = {}
    for j, h in enumerate(layer.hidden):
        if h in over or not h.startswith("erase ") and h != ONE:
            groups.setdefault(h.split("#")[0].rstrip("+-"), []).append(j)  # a gate's u+ and u-
    groups = [np.array(g) for g in groups.values()]
    worth = np.array([r[g].mean() / len(g) for g in groups])
    while len(groups):
        k = np.array([copies[g[0]] for g in groups])
        fit = np.array([len(g) <= spare for g in groups]) & (k < TIMES)
        if not fit.any():
            break
        best = int(np.argmax(np.where(fit, worth / (k * (k + 1)), 0)))
        copies[groups[best]] += 1
        spare -= len(groups[best])
    return copies


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
    new one, in a feature that the row before holds nothing in either: the last of those, or
    one more, if there is none. What the last row holds is to be carried on (see beside.py),
    and so its block must make every output"""
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
        spare = [f for f, held in enumerate(zip(*grid[-2:])) if held == (None, None)]
        if not spare:
            grid, spare = [row + [None] for row in grid], [len(grid[0])]
        grid[-1] = [None if s == ONE else s for s in grid[-1]]
        grid[-1][spare[-1]] = ONE
    return grid


def heads(grid: Grid, inputs: list[str], bits: int, size: int, rest=0.0, sizes=None):
    """The attention that makes row 0 of grid from an embedding of ONE, in the features that hold
    it there, each as large as `sizes` says, or SIZE, and of a code of `bits` bits, bit i in
    feature 1 + i, each SIZE. Gives q, k, v, o and the weights of the norms before and after,
    which are over `size` features whose other squares sum to `rest`. Head h reads ONE alone,
    to find the token h back, and copies its code into the features of the inputs for it, the
    oldest token's first: of those that the grid has, as the program reads them. Head 0 also
    gives the code, negated, to the features it is embedded in. A key and a value have weights
    of VALUE, and an output of 1; a query's are as much smaller as a key is large: the norms of
    a Gemma make of a query and a key what they will"""
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
    held = np.full(width, SIZE) if sizes is None else np.where(sizes > 0, sizes, SIZE)
    held = held[embedded]  # each ONE as it is embedded, and the bits of the code
    reads[embedded] = READ / held * math.sqrt(((held * held).sum() + rest) / size)
    wrote = np.flatnonzero(o.any((0, 2)))
    writes[wrote] = SIZE * math.sqrt(len(wrote) / size)
    return turns, q, k, v, o, reads, writes


def transformer(t, codes: dict, hidden=None, beside=None, carry=None, grid=None) -> Transformer:
    """t, a program from the bits of the last tokens, the oldest token's first, to the bits of the
    next, as a transformer on the tokens of codes {token: its bits}. A code has a bit for each of
    t's outputs, and t reads the codes of len(t.inputs) // len(t.outputs) tokens, or some of
    their bits. With hidden, the hidden units that a block may have, each block has its units
    more than once, as many times as `repeats` says. The rest is for a transformer that is
    part of a larger one (see beside.py): with beside, (n, square), the stream has n features
    more, not the transformer's, whose squares sum to `square` times the features of the whole
    stream, and the norms are over them too; and what the last row holds is as large as
    `carry`, for other layers to read: the logits are then not its own. `grid` is
    token_grid's, if it is already made: beside.front makes it wider"""
    bits = len(t.outputs)  # of a code
    assert t.inputs and len(t.inputs) % bits == 0, (
        f"{t.name} reads {len(t.inputs)} bits: no codes of {bits}"
    )
    grid = grid or token_grid(t, carry is not None)
    width, square = len(grid[0]), squares(t, codes)
    size = width + beside[0] if beside else None  # of the stream, and the squares of the others
    rest = size * beside[1] if beside else 0.0
    sizes = SIZE * np.array([[s is not None for s in row] for row in grid], float)
    ones = [0, *range(1 + bits, bits + ONES)]  # where ONE is embedded: unevenly (see uneven)
    sizes[:, ones] *= np.where(np.array(grid, object)[:, ones] == ONE, uneven(ONES), 1)
    embed = np.zeros((max(codes) + 1, bits + ONES))  # of each token that has a code: ONE, its
    embed[list(codes)] = SIZE  # code, and ONE in ONES - 1 features more
    embed[:, ones] *= uneven(ONES)
    embed[list(codes), 1 : 1 + bits] = SIZE * (2.0 * np.array(list(codes.values())) - 1)
    out = np.zeros(width)  # the last norm: the right token's logit is LOGIT
    if carry is None:
        held = sum(square.get(s, 1.0) * z * z for s, z in zip(grid[-1], sizes[-1]) if s)
        out[1 : 1 + bits] = (
            LOGIT / (bits * SIZE * SIZE) * math.sqrt((held + rest) / (size or width))
        )
    else:
        sizes[-1] *= carry / SIZE
    attention = heads(grid, t.inputs, bits, size or width, rest, sizes[0])
    layers = blocks(t, grid, sizes, square, rest, size, hidden, range(1, 1 + bits))
    return Transformer(t, grid, sizes, codes, embed, *attention, layers, out, rest, size)

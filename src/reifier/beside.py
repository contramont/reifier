"""Beside Gemma: a transformer of transformer.py and Gemma 3's own weights, in one checkpoint,
bf16 as Gemma's is. Its next token is Gemma's, but where the transformer's outputs are the code
of a token: there it is that token. With no noise on the weights, Gemma is moved by it about
as much as it is by its own weights rounded again. It is made to add few layers and few
features to Gemma, and it pays for that: in the noise that it takes, in a weight and some
activations that are larger than Gemma's, and in what it carries past Gemma's layers, which
is small beside Gemma's stream. The end of this says how much.

The checkpoint is a Gemma that is wider and deeper. Its stream is the transformer's features,
then as many that hold nothing as front needs, none for examples/alongside.py, then Gemma's
own. A layer has Gemma's hidden units, or as many as the transformer's largest block, if that
has more (see units): Gemma's are the last. Neither's layers are the other's: in a layer of
the transformer's own, the norms' weights on Gemma's features are 0, so it neither reads nor
writes them, and in a layer of Gemma's, those on the transformer's are. They meet where a
norm takes the mean square of the whole stream, and in the logits. And they meet in the rows
that a block of the transformer's gives for its norm to drop (see transformer.block): those
are the rows of the features after its own, so of Gemma's first, unless as many hold
nothing. The norm multiplies them by 0, so nothing of them is written, as long as its
weights have no noise (see the end).

- Gemma's features are c times what they were, c the square root of all the features over
  Gemma's own, as an embedding is scaled by the square root of the size. Their mean square
  over all the features is what it was over Gemma's, so a norm gives c times what it gave: the
  norms that Gemma reads the stream through have their weights over c. And o and down are c
  times Gemma's, for the norm after them to see the mean square that it saw, and to add c
  times what it added. That is Gemma alone, as wide: what save writes with no transformer,
  and as many features more. Its weights are rounded to bf16 again, and run in bf16 on text
  it gives Gemma's own top token at 97 or 98 positions in 100, as any change of a part in 512
  does;
- the transformer's blocks are all in front of Gemma's layers (see front), where Gemma's
  features are a token's embedding, as large for every token: STREAM[0]. A block does not need
  to know it (see transformer.py), but its gates are then as sharp as they are meant to be;
- its last block makes the outputs where the logits read them, each CARRY, with their rests
  and a ONE, and erases all else: with their rests the outputs are as large where they are 0,
  at a token that the transformer has nothing to say after, as where they are a code, so what
  they add to the mean square that a norm of Gemma's takes is the same at every token;
- Gemma's stream grows, a thousandfold over its layers, and what is carried beside it does
  not: no layer of Gemma's writes to it, so after the last it is as it was before the first,
  to the bit, and a 500th of the stream. That is enough to read it by, and too little if
  there is noise in the stream that is as large for every feature. A lift is for that: before
  Gemma's layer i, for each i of LIFTS, a layer of the transformer's own makes each carried
  feature as large as LIFTS says, in place. It reads them through a norm that divides by the
  size of Gemma's stream, which is 25 times larger at some tokens than at most: so its units
  come in twos, gelu(z) v - gelu(-z) v, which is z v whatever z is, and the norm after makes
  what they give as large as it should be. STREAM, the usual size of Gemma's stream before
  each layer, is for the units to be neither large nor lost under the norm's eps: a gate is
  LIFT there. LIFTS is empty: a lift is a layer more, and all that it buys is that the
  transformer takes more noise in the stream (see the end). With no lift, CARRY is all that
  stands against that noise: it is 3, three times a feature of Gemma's where it is made;
- after Gemma's last layer, one more of the transformer's own makes each output LOUD, where
  the logits read it, and the last norm's weight on it is HEARD. Its block could read a 500th
  of the stream only through a norm's weights of 1400, 20 times Gemma's largest. So the
  attention of that layer, which has nothing else to do, reads it first: a head attends to
  its own token and reads each carried feature as FAINT, through weights half as large as
  Gemma's largest. It gives each rest and the ONE to itself, and each output to a feature
  that holds nothing and that the last norm drops: what is a little off where an output
  should be 0 is then not made large where the logits read it. A norm of what a head gives
  divides by the size of that alone, so the norm after the attention makes each HEAD, in a
  stream of up to some 5 times its usual size: in one 10 times as large, what the head gives
  is so small that the norm's eps is felt, and the outputs are a third off. The block reads
  what the head made as a lift would;
- LOUD is large for a reason. The last norm divides by the root mean square of the whole
  stream, outputs and all: were they small in it, the logit of the transformer's token would
  be as much larger as Gemma's stream is smaller, and that differs by a few times from token
  to token. As it is, 72 outputs in a stream of 896 are 2300 of its root mean square, where
  Gemma's features are 1400 as a rule, and the logit is nearly the same at every token: SIZE
  HEARD times the square root of the bits of a code at most, 85 for 72 bits. The price is
  the norm after that block, whose weights are 0.4 LOUD. In a stream of n features LOUD is
  sqrt(n / 896) times as much: Gemma's own features are as much larger there, and a code's
  embedding, which the logits read, as much smaller;
- where the outputs are a code, a token's logit gains for each bit of its code that they
  have, and loses for each other: the token of that code must gain more than any other by
  more than Gemma's own logits differ. Two codes that differ in 16 bits of 72, as those of
  examples/alongside.py do at least, are four ninths of that logit apart: so a code must
  differ from every other in many bits, and the logit must be larger than Gemma's own top
  logits, which are under 50 on text;
- that layer's units give nothing where the transformer has nothing to say. An output that
  can be 1/2 has a rest, which is as large as ONE where the output is 1/2, and 0 where it is
  a bit. Such outputs are 1/2 together or not at all, so QUIET times the mean of all their
  rests is taken off the gate of each unit: the mean of 72 is off by an eighth of what one
  is. Where the outputs are 0, or nearly, the layer gives nothing, and the logits are Gemma's.

The <bos> has a code as any token has: after it alone, the transformer reads it twice.

What it costs to be small, as examples/measure.py says of the sponge of examples/alongside.py
on the text of src/reifier. That is 34 layers, its own 15 in front of Gemma's 18 and one
after, and a stream of 896, of which 256 are its features: a file of 1.37 GB. The design of
commit 7e2804a had 53 layers for the same sponge, four of them lifts, and a stream of 1280,
with no bit written over another: a file of 2.69 GB. Against that:

- one weight is larger than the largest of Gemma's own of its kind, where before none was:
  that of the norm after the block of the last layer, 3200, 1.9 times (see LOUD). The others
  are at most 0.93 of Gemma's largest of their kind. Of what its layers hold, the gates reach
  nearly three times the largest in Gemma's layers, and a value and a hidden unit 1.2 and 1.1
  times (see transformer.py), where before a gate of the last layer was 1.1 times, and what
  that layer gave 1.3 times. The logit of the transformer's token is about 68, where before
  it was 157;
- noise on the weights. With every weight times 1 + 1% noise, the transformer's token is
  right at all but 1 position in 10,000, with 1.5% at half of them, with 2% at none; before,
  with 5%, at all but 6 in 10,000. Noise added to every projection and the embedding, of 2%
  of the root mean square of Gemma's own of its kind, leaves it right at three positions in
  four, where before 10% left nine in ten. With one weight in 10,000 set to 0 it is right at
  3 in 10, where before one in 1,000 left 96 in 100. Rounded to 6 bits it is right
  everywhere, and to 5 nowhere, as before. Most of this is the rounds of two layers of
  examples/alongside.py: with rounds of three, in 40 layers, it is right at all but 4
  positions in 1,000 with every weight 4% off (see examples/keccak.py). And where it is
  wrong it moves Gemma too: with 2%, the model's token is Gemma's at a quarter of the
  positions where the transformer is silent, where Gemma alone keeps its own at 88 in 100;
- noise in the stream. With normal noise added to every feature after every layer, of 0.1% of
  the root mean square of the stream there, the transformer's token is right at 85 positions
  in 100, and with 0.3% at 5, where Gemma alone keeps its own at 94 and at 83. One lift,
  LIFTS = {4: 64.0}, a layer more, keeps them all at 0.3%, and 95 in 100 at 1%, where Gemma
  keeps half; it leaves what noise on the weights does as it was;
- noise on the weights of the norms, of 0.01% of the root mean square of Gemma's own, leaves
  the transformer's token right, and of 0.03%, right at 43 positions in 100, where before at
  all. It moves Gemma more than it moves Gemma alone: with 0.03%, where the transformer is
  silent, the model's token is Gemma's at 66 positions in 100, where Gemma alone keeps its
  own at 94. The weights of the norms after the transformer's blocks are 2 at most, the last
  aside, where Gemma's are 21 as a rule and up to 1,660, and the noise is a part of Gemma's.
  And the norm's weight on a dropped row is then not 0, so the row, which is as large as a
  bit, is written onto a feature of Gemma's, times that weight, by every block that drops
  rows.
"""

import json
import math
import re
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from . import gemma
from .grid import ONE
from .trace import MLP, Trace
from .transformer import (
    CONTEXT,
    MARGIN,
    PAIRS,
    READ,
    VALUE,
    Block,
    Transformer,
    lookback_query,
    runner_up,
    squares,
    token_grid,
    transformer,
)

CARRY = 3.0  # a bit of the outputs, as the transformer's own layers leave it
LIFTS = {}  # {i: size}: before Gemma's layer i, a layer more makes it so large. None, by default
LOUD, HEARD = 8000.0, 5.0  # as the last layer makes it, and the last norm's weight on it: in
# a stream of 896 features, and sqrt(n / 896) times as loud in one of n, where Gemma's own
# features are as much larger, and a code's embedding, which the logits read, as much smaller
LIFT, SPEAK = 1.5, 3.0  # a bit as a layer that lifts reads it, and as the block of the last layer
# does, where Gemma's stream is as large as usual
DEAF, QUIET = 0.0, 3.0  # of that block: what it is deaf to, of a bit, and how many times the mean
# of the rests it takes off its gates
HEAD, FAINT = 240.0, 0.25  # a bit as the head of the last layer makes it, and as that reads it
SHARES = 32  # the pairs of units that lift a feature, each by its share, if as many fit
# Of each layer of unsloth/gemma-3-270m, the median on English text of the rms of the stream
# before it, and after the last
# fmt: off
STREAM = [
    1.03, 4.7, 37, 45, 73, 86, 130, 130, 210, 210, 250, 380, 410, 510, 670, 740, 850, 1200,
    1400,
]
# fmt: on
READS = re.compile("input_layernorm|pre_feedforward_layernorm|^norm")  # norms: Gemma's weight / c
WRITES = re.compile("o_proj|down_proj")  # what the norm after them adds: Gemma's weight * c


def moved(name: str, to) -> str:
    """name, of a tensor, as it is in layer to[i], if it is in layer i"""
    return re.sub(r"(?<=layers\.)\d+", lambda i: str(to[int(i[0])]), name)


def lift(width, held: dict, gives: dict, one: int, read, shares, deaf=0.0, quiet=None) -> Block:
    """A block that reads the features of `held`, each plus or minus its size or 0, and gives
    each to the features of gives[f], if it has any, times the ONE that is among them, `one`:
    for a feature, `shares` times two units, gelu(z - deaf u) v - gelu(-z - deaf u) v, for the z
    and the u that the norm before makes of the feature and of ONE, and for v, u over `shares`: z
    is `read` where the stream is as large as usual. With deaf 0 the two give z v, whatever the
    stream is. With more, they give 0 for a z that is under deaf of ONE, and so for all that
    should be 0 and is not quite, as long as that is far enough for gelu to be relu; and for a
    feature that has others in `quiet`, they give 0 where those are as large as ONE, all of
    them: QUIET times their mean is taken off its gates. What a weight out of a unit is off by
    is a part of the unit's share alone. The norm after has yet no weights: `plan` sets them"""
    lifted = [f for f in held if gives.get(f)]
    names = [f"lift f{f}{sign}" for f in lifted for sign in "+-"] * shares
    shapes = [(len(names), width)] * 2 + [(width, len(names))]
    gate, value, down = (np.zeros(shape) for shape in shapes)
    for j, f in enumerate(lifted):
        j = [range(len(names))[2 * j + i :: 2 * len(lifted)] for i in (0, 1)]  # its + and - units
        gate[j[0], one] = gate[j[1], one] = -deaf
        gate[j[0], f], gate[j[1], f] = min(gate[j[0], f][0] + 1, 1), max(gate[j[1], f][0] - 1, -1)
        for i in (0, 1):
            others = (quiet or {}).get(f, [])
            gate[np.ix_(list(j[i]), others)] -= QUIET / max(len(others), 1)
            value[j[i], one] = 1 / 2 / shares
            down[np.ix_(gives[f], j[i])] = 1 / 2 - i
    reads = np.zeros(width)
    reads[list(held)] = read / np.array(list(held.values()))
    layer = MLP("layer", [f"f{f}" for f in range(width)], [f"f{f}" for f in range(width)], names,
                np.vstack([gate, value]), np.zeros(2 * len(names)), "geglu", down)  # fmt: skip
    return Block(layer, reads, np.zeros(width))


def front(t: Trace, codes: dict, size=640, hidden=2048, heads=4) -> Transformer:
    """The transformer on the program t that is to be in front of the layers of a Gemma of
    `size` features and `heads` heads, beside its embedding: what `save` takes. It carries its
    outputs on, and has no logits of its own. A block of it may have `hidden` hidden units,
    and has its own more than once if they fit (see transformer.repeats): the more often, the
    more noise on the weights it takes. A layer of the checkpoint has as many as the largest
    block, or as Gemma (see units). The stream of the two is the grid's features, then as few
    that hold nothing as make a whole number of features a head, as a Gemma must have, then
    Gemma's. The grid is token_grid's, and wider if its last row has fewer features with
    nothing in them than outputs: the head of the last layer gives an output to each (see
    plan). And a block that writes bits over others gives as many rows more, which the norm
    after it drops (see transformer.block): they are the rows of the features after the
    grid's, so those are as many as the most bits that one block writes over, if Gemma's are
    fewer"""
    grid, bits = token_grid(t, True), len(t.outputs)
    spare = sum(s is None for s in grid[-1][1 + bits :])
    grid = [row + [None] * max(0, bits - spare) for row in grid]
    over = max(sum(bool(a and b and a != b) for a, b in zip(*rows)) for rows in pairwise(grid))
    more = max(size, over)
    more += -(more + len(grid[0])) % heads
    return transformer(t, codes, hidden, (more, STREAM[0] ** 2), CARRY, grid)


def units(m: Transformer, config: dict) -> int:
    """The hidden units of a layer of the checkpoint of m beside a Gemma of config config:
    Gemma's, or as many as the largest of m's blocks has"""
    return max(config["intermediate_size"], *[len(b.layer.hidden) for b in m.blocks])


def plan(m: Transformer, config: dict):
    """What m, of `front`, needs after its own layers beside a Gemma 3 of config config: the
    blocks that lift what it leaves, by the layer of Gemma's that each is before, if LIFTS has
    any; what the head of the layer after Gemma's last reads and gives, and the weights of the
    norms around it (see `head`); and that layer's block, which reads what the head gave and
    makes the outputs LOUD where the logits read them"""
    width, last, t = len(m.grid[0]), m.grid[-1], m.trace
    wide, hidden, bits = m.size, units(m, config), len(t.outputs)
    empty = (wide or 0) - width - config["hidden_size"]  # the features that hold nothing
    whole = wide and wide % config["num_attention_heads"] == 0
    assert whole and empty >= 0 and not m.out.any(), f"{t.name} is not of front, for this Gemma"
    assert len(config["layer_types"]) == len(STREAM) - 1, "STREAM is of unsloth/gemma-3-270m"
    assert len(m.codes) == config["vocab_size"], f"{t.name} needs a code for each of Gemma's tokens"
    assert len(m.q) <= config["num_attention_heads"], f"{t.name} needs its heads in one layer"
    one, carried = last.index(ONE), [f for f, s in enumerate(last) if s]
    each = squares(t, m.codes) | {ONE: 1.0}  # the mean square of each signal, over its size
    square = sum(each[last[f]] for f in carried)
    shares = max(1, min(SHARES, hidden // (2 * len(carried))))
    size, lifts = CARRY, {}
    for i, larger in LIFTS.items():  # in place: each feature is given to itself
        held, gives = dict.fromkeys(carried, size), {f: [f] for f in carried}
        b = lifts[i] = lift(width, held, gives, one, LIFT * STREAM[i], shares)
        b.writes[carried] = (larger - size) * math.sqrt(square / wide)
        size = larger
    outs, rests = range(1, 1 + bits), {y: x.outputs for x in t.mlps for y in x.outputs}
    quiet = {  # of an output that can be 1/2, its rest: no other output of an MLP is one
        1 + i: last.index(r)
        for i, y in enumerate(t.outputs)
        for r in rests[y]
        if r not in t.outputs and each[y] < 1
    }
    window, tokens = len(t.inputs) // bits, list(m.codes)  # on the codes of tokens at random,
    drawn = np.random.default_rng(0).integers(0, len(tokens), (64, window))  # as squares does
    read = np.array([[m.codes[tokens[i]] for i in row] for row in drawn], float).reshape(64, -1)
    values = t.values(read)
    half = np.array([values[t.outputs[f - 1]] == 0.5 for f in quiet]).reshape(len(quiet), -1)
    together = (half.all(0) | ~half.any(0)).all()
    assert together, f"{t.name}: its outputs that can be 1/2 must be 1/2 together"
    spare = [f for f in range(1 + bits, width) if f not in carried][:bits]  # which the norm drops
    assert len(spare) == bits, f"{t.name} needs {bits} features more, for its head to give to"
    # the head gives an output to one of `spare`, which the last norm drops, and all else that
    # is carried to itself: each is then HEAD, but an output's own feature, which is as it was
    at = {f: f for f in carried} | dict(zip(outs, spare))
    reads, writes = np.zeros(width), np.zeros(width)
    reads[carried] = FAINT * STREAM[-1] / size
    writes[[at[f] for f in carried]] = HEAD * math.sqrt(square / wide)
    writes[[f for f in carried if at[f] == f]] -= size * math.sqrt(square / wide)
    held, gives = {at[f]: HEAD for f in carried}, {at[f]: [f] for f in outs} | {one: spare}
    quiet = {at[f]: sorted(quiet.values()) for f in quiet}  # an output's units: every rest
    speak = lift(width, held, gives, one, SPEAK * STREAM[-1], shares, DEAF, quiet)
    loud = LOUD * math.sqrt(wide / 896)
    speak.writes[outs] = (loud - size) * math.sqrt((bits + len(spare)) / wide)
    return lifts, (carried, [at[f] for f in carried], one, reads, writes), speak


def head(w, i, m: Transformer, carried, to, one, reads, writes, config):
    """The attention of layer i, of those after m's blocks: a head that attends to its own
    token, as one of transformer.heads that looks no token back, but on `one`, the ONE that is
    carried; it gives each feature of `carried` to the feature of `to` for it, between norms
    of weights reads and writes. The norm after makes of what is given what `writes` says, if
    the stream that it was read from is not many times its usual size (see the module's notes)"""
    width, n = len(reads), len(carried)
    scale = MARGIN / (PAIRS * (1 - runner_up(m.turns, CONTEXT)))  # its own token scores
    key, query = np.zeros((2, 2 * PAIRS, width))  # scale * PAIRS, as in heads
    key[:PAIRS, one] = VALUE
    query[:, one] = scale / (READ**2 * VALUE) * lookback_query(0, m.turns)
    value, out = np.zeros((n, width)), np.zeros((width, n))
    value[range(n), carried], out[to, range(n)] = VALUE, 1
    gemma.attention(w, i, replace(m, k=key), [query], value, [out], reads, writes, config)


def build_weights(m: Transformer | None, real: dict, config: dict, features: int = 0):
    """The weights of m, of `front`, beside real, the tensors of a Gemma 3 by name, whose config
    is config: or, with no m, of that Gemma alone, with `features` features more, which hold
    nothing: as wide as it is beside a transformer that adds as many. Gives the config of the
    wide Gemma, the shapes of its tensors, the weights that are set, by their names in the text
    model, the names of m's features after the heads and after each layer of the checkpoint,
    and of the hidden units that are m's in each: none, in a layer of Gemma's"""
    size, hidden = config["hidden_size"], config["intermediate_size"]
    width = m.size - size if m and m.size else features  # of the features that are not Gemma's
    c = math.sqrt((size + width) / size)
    lifts, heard, speak = plan(m, config) if m else ({}, None, None)
    own = list(m.blocks) if m else []  # the layers that are m's, in order, and Gemma's
    gemmas = []
    for i in range(len(config["layer_types"])):
        own += [lifts[i]] if i in lifts else []
        gemmas.append(len(own) + i)
    own += [speak] * bool(m)
    kinds = ["sliding_attention"] * (len(own) + len(gemmas))  # m's own layers slide,
    for i, kind in zip(gemmas, config["layer_types"]):  # and Gemma's are as they were
        kinds[i] = kind
    ours = [i for i in range(len(kinds)) if i not in gemmas]
    layer = [k for k in real if ".layers.0." in k]  # the tensors of a layer, by those of the first
    shapes = {moved(k, {0: i}): real[k].shape for i in ours for k in layer}  # m's own layers,
    shapes |= {moved(k, gemmas): v.shape for k, v in real.items()}  # and Gemma's
    wider = {size: size + width, hidden: units(m, config) if m else hidden}
    kept = {v.shape[0] for k, v in real.items() if re.search("[qkv]_(proj|norm)|embed", k)}
    assert size != hidden and not kept & {size, hidden}, "shapes do not tell features from the rest"
    assert wider[size] % config["num_attention_heads"] == 0, "a Gemma has whole features a head"
    shapes = {k: [wider.get(d, d) for d in s] for k, s in shapes.items()}
    config = config | {"layer_types": kinds, "num_hidden_layers": len(kinds)}
    config |= {"hidden_size": wider[size], "intermediate_size": wider[hidden]}
    w = gemma.LazyZeros(shapes)  # a norm's weights are what it multiplies by, until the end:
    for k in shapes:  # 0, in a layer of m's own, around a module that has nothing to do
        if k.endswith("norm.weight"):
            w[k.removeprefix(w.prefix)]
    grid, names = [], []
    if m:  # on the first of the features, and the first of the hidden units of its own layers
        w["embed_tokens.weight"][:, : m.embed.shape[1]] = m.embed / math.sqrt(size + width)
        gemma.attention(w, 0, m, m.q, m.v, m.o, m.reads, m.writes, config)
        for i, b in zip(ours, own):
            gemma.block(w, i, b, 1.0)
        head(w, ours[-1], m, *heard, config)
        w["norm.weight"][np.flatnonzero(speak.writes)] = HEARD
        rows = iter([*m.grid[1:], *[m.grid[-1]] * (len(lifts) + 1)])
        grid = [m.grid[0]]
        for i in range(len(kinds)):  # a layer of Gemma's leaves m's features as they were
            grid.append(next(rows) if i in ours else grid[-1])
        names = [own[ours.index(i)].layer.hidden if i in ours else [] for i in range(len(kinds))]
    for k, v in real.items():  # Gemma's own: the last of each axis, in the layers of `gemmas`
        k = moved(k.removeprefix(w.prefix), gemmas)
        if k.endswith("norm.weight"):
            v = (1 + v) / c if READS.search(k) else 1 + v
        w[k][tuple(slice(-d, None) for d in v.shape)] = v * c if WRITES.search(k) else v
    for k, v in w.items():
        if k.endswith("norm.weight"):
            v -= 1  # Gemma 3's norm multiplies by 1 + weight
    return config, shapes, w, grid, names


def checkpoint_weights(repo: str) -> dict[str, np.ndarray]:
    """The tensors of repo's checkpoint, at its revision of gemma.REVISIONS, by name: bf16, as
    float32. From huggingface.co the first time, with huggingface_hub, and then from its cache"""
    from huggingface_hub import hf_hub_download

    file = hf_hub_download(repo, "model.safetensors", revision=gemma.REVISIONS[repo])
    with open(file, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    data, weights = np.memmap(file, "<u2", "r", 8 + n), {}
    for k, t in header.items():
        assert t["dtype"] == "BF16", f"{k} is {t['dtype']}, not BF16"
        start, end = (i // 2 for i in t["data_offsets"])  # a bf16 is the top half of a float32
        weights[k] = (data[start:end].astype(np.uint32) << 16).view(np.float32).reshape(t["shape"])
    return weights


def save(m: Transformer | None, out_dir: Path, features: int = 0):
    """m, of `front`, beside unsloth/gemma-3-270m, in out_dir: a checkpoint that transformers
    loads from there, with the names of m's features and hidden units as they are in it (see
    gemma.write). With no m, that Gemma alone, with `features` features more: as wide as the
    checkpoint of a transformer that adds as many. The first run fetches Gemma's weights"""
    repo = "unsloth/gemma-3-270m"  # the Gemma 3 that STREAM was measured on
    config = gemma.checkpoint_metadata(repo)[1]
    config, shapes, w, grid, names = build_weights(m, checkpoint_weights(repo), config, features)
    gemma.write(out_dir, config, shapes, w, grid, names)

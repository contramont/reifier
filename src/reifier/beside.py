"""Beside Gemma: a transformer of transformer.py and Gemma 3's own weights, in one checkpoint,
bf16 as Gemma's is. Its next token is Gemma's, but where the transformer's outputs are the code
of a token: there it is that token. No weight of the transformer's is larger than the largest
of Gemma's own of its kind, and what its layers hold is as large as what Gemma's hold: it is in
the stream as Gemma's features are, not under them. Two things can be larger. At a token where
Gemma's stream is a sixth of its usual size, the gates of the last layer, and what that gives,
are a third larger than Gemma's largest (see SPEAK). And where the transformer speaks, its
token's logit is a few hundred, where Gemma's are a few dozen (see HEARD). Gemma is moved by it
as it is by its own weights rounded again, and no more.

The checkpoint is a Gemma twice as wide, in features and in hidden units, and deeper. Its
stream is the transformer's features, then Gemma's. Neither has weights on the other's
features, and neither's layers are the other's: in a layer of the transformer's own, the norms'
weights on Gemma's features are 0, so it neither reads nor writes them, and in a layer of
Gemma's, those on the transformer's are. They meet where a norm takes the mean square of the
whole stream, and in the logits.

- Gemma's features are c = sqrt(2) times what they were, as an embedding is scaled by the square
  root of the size. Their mean square over twice the features is what it was over Gemma's, so
  a norm gives c times what it gave: the norms that Gemma reads the stream through have their
  weights over c. And o and down are c times Gemma's, for the norm after them to see the mean
  square that it saw, and to add c times what it added. These weights are rounded to bf16
  again, which moves Gemma's top token at one position in 50 on text, as any change of a part
  in 512 does. That is Gemma alone, as wide: what save writes with no transformer;
- the transformer's blocks are all in front of Gemma's layers (see front), where Gemma's
  features are a token's embedding, as large for every token: STREAM[0]. A block does not need
  to know it (see transformer.py), but its gates are then as sharp as they are meant to be;
- its last block makes the outputs where the logits read them, each CARRY, with their rests
  and a ONE, and erases all else: with their rests the outputs are as large where they are 0,
  at a token that the transformer has nothing to say after, as where they are a code, so what
  they add to the mean square that a norm of Gemma's takes is the same at every token;
- Gemma's stream grows, a thousandfold over its layers. What is carried beside it must grow
  too, to stay more than what noise in the stream would be: before the layers of LIFTS, a layer
  of the transformer's own makes each of them as large as LIFTS says, in place. It reads them
  through a norm that divides by the size of Gemma's stream, which is 25 times larger at some
  tokens than at most: so its units come in twos, gelu(z) v - gelu(-z) v, which is z v whatever
  z is, and the norm after makes what they give as large as it should be. STREAM, the usual
  size of Gemma's stream before each layer, is for the units to be neither large nor lost
  under the norm's eps: a gate is LIFT there. Where Gemma's stream is much smaller than usual,
  what these layers read and give is as much larger;
- after Gemma's last layer, one more makes each output LOUD, as large as Gemma's stream is
  there, and the last norm's weight on it is HEARD. Where the outputs are a code, a token's
  logit gains for each bit of its code that they have, and loses for each other: so the
  token of that code must gain more than any other by more than Gemma's logits differ, 80 or
  so at most. That is why a code should differ from every other in many bits (see
  examples/alongside.py: a quarter of them), and why the logit is large: two codes that differ
  in a quarter of their bits are a half of it apart;
- that layer's units are deaf to what is under DEAF of a bit, and to an output whose rest is
  there: where the outputs are 0, or nearly, it gives nothing, and the logits are Gemma's.

The <bos> has a code as any token has: after it alone, the transformer reads it twice.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

from . import gemma
from .grid import ONE
from .trace import MLP, Trace
from .transformer import Block, Transformer, squares, transformer

CARRY = 1.5  # a bit of the outputs, as the transformer's own layers leave it
LIFTS = {1: 3.0, 2: 12.0, 9: 72.0, 14: 240.0}  # before Gemma's layer i, a layer makes it so large
LOUD, HEARD = 2000.0, 16.0  # as the last layer makes it, and the last norm's weight on it
LIFT, SPEAK = 3.0, 6.0  # what a layer that lifts reads a bit as, where Gemma's stream is as large
DEAF = 0.5  # as usual, and the last layer; and what that is deaf to, of a bit
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
    feature that has another in `quiet`, they give 0 where that one is as large as ONE. What a
    weight out of a unit is off by is a part of the unit's share alone. The norm after has yet
    no weights: `plan` sets them"""
    lifted = [f for f in held if gives.get(f)]
    names = [f"lift f{f}{sign}" for f in lifted for sign in "+-"] * shares
    shapes = [(len(names), width)] * 2 + [(width, len(names))]
    gate, value, down = (np.zeros(shape) for shape in shapes)
    for j, f in enumerate(lifted):
        j = [range(len(names))[2 * j + i :: 2 * len(lifted)] for i in (0, 1)]  # its + and - units
        gate[j[0], one] = gate[j[1], one] = -deaf
        gate[j[0], f], gate[j[1], f] = min(gate[j[0], f][0] + 1, 1), max(gate[j[1], f][0] - 1, -1)
        for i in (0, 1):
            gate[j[i], (quiet or {}).get(f, one)] -= f in (quiet or {})
            value[j[i], one] = 1 / 2 / shares
            down[np.ix_(gives[f], j[i])] = 1 / 2 - i
    reads = np.zeros(width)
    reads[list(held)] = read / np.array(list(held.values()))
    layer = MLP("layer", [f"f{f}" for f in range(width)], [f"f{f}" for f in range(width)], names,
                np.vstack([gate, value]), np.zeros(2 * len(names)), "geglu", down)  # fmt: skip
    return Block(layer, reads, np.zeros(width))


def front(t: Trace, codes: dict, size: int = 640, hidden: int = 4096) -> Transformer:
    """The transformer on the program t that is to be in front of the layers of a Gemma of
    `size` features, beside its embedding, with `hidden` hidden units a layer: what `save`
    takes. It carries its outputs on, and has no logits of its own"""
    return transformer(t, codes, hidden, 2 * size * STREAM[0] ** 2, 2 * size, CARRY)


def plan(m: Transformer, config: dict):
    """What m, of `front`, needs after its own layers beside a Gemma 3 of config config: the
    blocks that lift what it leaves, by the layer of Gemma's that each is before, and the block
    after Gemma's last, which makes the outputs LOUD where the logits read them, in place"""
    wide, hidden, t = 2 * config["hidden_size"], 2 * config["intermediate_size"], m.trace
    width, last, bits = len(m.grid[0]), m.grid[-1], len(t.outputs)
    assert m.size == wide and not m.out.any(), f"{t.name} is not of front, for this Gemma"
    assert len(config["layer_types"]) == len(STREAM) - 1, "STREAM is of unsloth/gemma-3-270m"
    assert len(m.codes) == config["vocab_size"], f"{t.name} needs a code for each of Gemma's tokens"
    assert len(m.q) <= config["num_attention_heads"], f"{t.name} needs its heads in one layer"
    assert width <= wide // 2, f"{t.name} needs {width} features, and has {wide // 2}"
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
    spare = [f for f in range(1 + bits, width) if f not in carried][:bits]  # which the norm drops
    held, gives = dict.fromkeys(carried, size), {f: [f] for f in outs} | {one: spare}
    speak = lift(width, held, gives, one, SPEAK * STREAM[-1], shares, DEAF, quiet)
    speak.writes[outs] = (LOUD - size) * math.sqrt((bits + len(spare)) / wide)
    return lifts, speak


def build_weights(m: Transformer | None, real: dict, config: dict):
    """The weights of m, of `front`, beside real, the tensors of a Gemma 3 by name, whose config
    is config: or, with no m, of that Gemma alone, as wide. Gives the config of the wide Gemma,
    the shapes of its tensors, the weights that are set, by their names in the text model, the names
    of m's features after the heads and after each layer of the checkpoint, and of the hidden
    units that are m's in each: none, in a layer of Gemma's"""
    size, hidden, c = config["hidden_size"], config["intermediate_size"], math.sqrt(2)
    lifts, speak = plan(m, config) if m else ({}, None)
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
    shapes = {k: [2 * d if d in (size, hidden) else d for d in s] for k, s in shapes.items()}
    config = config | {"layer_types": kinds, "num_hidden_layers": len(kinds)}
    config |= {"hidden_size": 2 * size, "intermediate_size": 2 * hidden}
    w = gemma.LazyZeros(shapes)  # a norm's weights are what it multiplies by, until the end:
    for k in shapes:  # 0, in a layer of m's own, around a module that has nothing to do
        if k.endswith("norm.weight"):
            w[k.removeprefix(w.prefix)]
    grid, names = [], []
    if m:  # on the first half of the features, and of the hidden units of its own layers
        w["embed_tokens.weight"][:, : m.embed.shape[1]] = m.embed / math.sqrt(2 * size)
        gemma.attention(w, 0, m, m.q, m.v, m.o, m.reads, m.writes, config)
        for i, b in zip(ours, own):
            gemma.block(w, i, b, 1.0)
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


def save(m: Transformer | None, out_dir: Path):
    """m, of `front`, beside unsloth/gemma-3-270m, in out_dir: a checkpoint that transformers
    loads from there, with the names of m's features and hidden units as they are in it (see
    gemma.write). With no m, that Gemma alone, as wide. The first run fetches Gemma's weights"""
    repo = "unsloth/gemma-3-270m"  # the Gemma 3 that STREAM was measured on
    config = gemma.checkpoint_metadata(repo)[1]
    config, shapes, w, grid, names = build_weights(m, checkpoint_weights(repo), config)
    gemma.write(out_dir, config, shapes, w, grid, names)

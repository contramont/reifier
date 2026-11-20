"""Beside Gemma: a transformer of transformer.py and Gemma 3's own weights, in one checkpoint,
bf16 as Gemma's is. Its next token is Gemma's, but where the transformer's outputs are the code
of a token: there it is that token. And Gemma is not moved: where the outputs are no code,
each logit is, to the bit, what Gemma alone gives in a checkpoint as wide.

The checkpoint is a Gemma twice as wide, in features and in hidden units, and deeper. Its
stream is the transformer's features, then Gemma's. Its first OWN layers are the transformer's:
its heads, and its first blocks. Gemma's layers follow, and each of the first of them has one
of the transformer's other blocks, on hidden units of its own; between two of them there can be
another layer of the transformer's own, that clears its features. Neither has weights on the
other's features, but for a K that a layer that clears puts where its own norms drop it: they
meet where a norm takes the mean square of the whole stream, or of all that a layer adds to it,
and in the logits.

- Gemma's features are c = sqrt(2) times what they were, as an embedding is scaled by the square
  root of the size. Their mean square over twice the features is what it was over Gemma's, so
  a norm gives c times what it gave: the norms that Gemma reads the stream through have their
  weights over c. And o and down are c times Gemma's, for the norm after them to see the mean
  square that it saw, and to add c times what it added. These weights are rounded to bf16
  again, which moves Gemma's top token at one position in 50 on text, as any change of a part
  in 512 does. That is Gemma alone, as wide: what save writes with no transformer;
- the transformer does not move it, as its features are small. A norm sums squares as float32,
  of 24 bits, and theirs are 2**-45 of the sum or less: it is very rarely another float, and
  on 49,000 positions of text and of random tokens it never was;
- so a norm gives a block of the transformer its bits times some s that the block cannot know,
  the norm's weight over the rms of Gemma's features. In Gemma's layers the norm after the
  block is as unknown: it divides by the rms of what Gemma's block gives. A block need not know
  either: it reads only what the block before wrote (see relay.py), its signals and a ONE, on
  which its gates' biases are. All of these are as large as each other, so a gate's sign is
  what it was;
- a GLU then gives s^2 times what it gave. In the transformer's own layers s is known, as
  Gemma's features are a token's embedding there, whose rms is 1.03: the transformer is as in
  gemma.py, but small, its ONE and code embedded as SMALL and its bits as their rows' sizes
  (see FIRST), and the heads and each block's unit ONE put a large K where the norm after them
  drops it, so that the norm's weight is not too small for bf16, which holds it less 1. Beside
  Gemma's blocks, a bit's size would be known half as well, in powers of two, after each GLU.
  So a block there is of ReLU units, as the ands of lib.py: a unit's value is 1, which it
  reads from the ONE that is embedded, whose size is known. It gives its gate times what the
  layer does to any feature, and a bit is off by what each layer was, once;
- how large is known roughly. ROUGH has, of each of Gemma's layers, the usual rms of the stream
  before its block and of what the block gives. With these, the norm before a block makes it
  read a bit as GATE, where 1 is enough for gelu to be relu, and its `down` makes it write a
  bit as large as its row's size, which ladder gives. Over eighteen blocks beside Gemma's, on
  text, a bit was read as 2**4.3 to 2**39, where GATE is 2**27;
- nor can a block erase what a feature held, as that is as unknown. It writes over it: a
  feature takes a new signal WAIT layers after its last, as layout(t, WAIT) gives, so what it
  held is GROW**WAIT = 2**10 times smaller, and bf16 rounds it away as the stream adds the new
  bit. Less so where Gemma's layers shrink the blocks between: at a <bos> in the midst of
  text, two of them did by 2**5.4. With no erasing, 640 features keep 640 signals apart, and
  any more only by their sizes: so the bits of each row are GROW times those of the row before;
- and bf16 runs out of sizes, as rows grow from FIRST, as small as bf16 lets a block read and
  write, to LAST, under what Gemma's norms would see: 2**76, for 16 rows. So past LAST a layer
  of the transformer's own clears the features that hold none of the last row's signals, to 0
  and to the bit (see wipe), and the rows grow again from ANEW. A feature that holds one of
  them is taken again after the next clear, as layout(t, WAIT, clear) gives. Gemma's layers do
  not see the layer: it adds nothing to their features, and what it adds to the transformer's
  is gone when it ends. The 3,580 signals of examples/alongside.py take 492 features, cleared
  once, after row 15: 550 of the transformer's 640, with ONE, the code, the ONEs and the twins;
- the numbers are bf16, of 8 bits. Beside Gemma, what a block reads is the same in size to the
  last bit, as one weight of one norm made it of one product, by a `down` that is a power of
  two, and so are the gates of an AND where it holds, 16 and 8 times a bit: its two units
  give the same bit again, whatever its size. Where what it reads is a little off, as the
  codes of two tokens are, each copied over the rms of its own token, or a bit written over
  what a feature held, a GLU of lib.py squares that away and an AND gives the same for a count
  1/4 off: on text, a bit was 0.1 off at most. A sum must be float32 before it is rounded:
  torch's is, for bf16, on a GPU and on a CPU with AVX512-BF16;
- the last block's outputs are ±u for some u: the code of a token, or all 0 for outputs of
  1/2, which are no code. Every token's code is embedded there, as ±FAINT, and the last
  norm's weight there is LOUD: a token's logit gains LOUD FAINT u over the rms of the last
  stream for each bit of its code that the outputs have, and loses as much for each other.
  What is embedded there would add to the logits too. So each output is written twice, the
  second time negated, into a twin feature that holds the same code, and whose weight is
  -LOUD: what is embedded cancels, and what is written doubles. On text the token of the
  outputs' code gained 2**14 or more, where no two of Gemma's logits were 150 apart.
"""

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import gemma
from .emit import emit
from .grid import ONE, Grid, held, layout
from .trace import MLP, Trace
from .transformer import Transformer

OWN = 2  # the layers that the checkpoint is deeper by: the transformer's own
SMALL = 2.0**-24  # ONE and the code as embedded, and in those layers what K's norm makes of 1
K = 2.0**26  # what the heads and the unit ONE of a block in those layers put into ONE
GATE = 2.0**6, 2.0**27  # what a block reads a bit as: in those layers, and in Gemma's
WAIT, GROW = 2, 32.0  # a feature is written again WAIT layers on, by bits GROW**WAIT as large
# A bit as the heads write it, FIRST, is as small as bf16 lets a block read and write it: the
# first block beside Gemma's reads row 2 through a norm weight of 2**125, under bf16's largest,
# 2**128, and writes row 3 with a down of 2**-118, over its smallest, 2**-126
FIRST = 2.0**-106
ANEW = 2.0**-86  # a bit of the row after a clear: read through 2**124 at most, written by 2**-117
LAST = 2.0**-30  # a bit of the outputs, and the largest that a row's can be
# Of each layer of unsloth/gemma-3-270m, the medians on English text of the rms of the stream
# before its block, and of the rms of what the block gives
# fmt: off
ROUGH = [
    (5.4, 0.79), (41, 0.31), (52, 0.28), (77, 0.22), (100, 0.11), (160, 0.13),
    (160, 0.11), (200, 0.11), (240, 0.095), (270, 0.075), (370, 0.089), (460, 0.19),
    (540, 0.13), (710, 0.078), (870, 0.061), (980, 0.055), (1200, 0.056), (1400, 0.056),
]
# fmt: on
FAINT = 2.0**-56  # a token's code where the outputs are written, for the logits
LOUD = 2.0**121  # the last norm's weight there
READS = re.compile("input_layernorm|pre_feedforward_layernorm|^norm")  # norms: Gemma's weight / c
WRITES = re.compile("o_proj|down_proj")  # what the norm after them adds: Gemma's weight * c


def moved(name: str, to) -> str:
    """name, of a tensor, as it is in layer to[i], if it is in layer i"""
    return re.sub(r"(?<=layers\.)\d+", lambda i: str(to[int(i[0])]), name)


def ladder(n: int) -> tuple[list[float], list[int]]:
    """For a grid of n layers: the size of a bit of each of its rows in the stream, and the rows
    after which a layer clears the features (see wipe). Row 0 is as the heads write it, row d as
    block d - 1 does, and the last, of the outputs, is LAST. A row's bits are GROW times those of
    the row before, from FIRST: where they would be more than LAST, the features are cleared
    after the row before, and the rows grow again from ANEW"""
    sizes, clear = [FIRST], []
    for d in range(1, n):
        if sizes[-1] * GROW <= LAST:
            sizes.append(sizes[-1] * GROW)
        else:
            sizes.append(ANEW)
            clear.append(d - 1)
    return sizes + [LAST], clear


def layer_of(d: int, clear: list[int]) -> int:
    """The layer of the checkpoint that block d is in: after a layer for each clear before it"""
    return d + sum(c <= d for c in clear)


def build_weights(m: Transformer | None, real: dict, config: dict):
    """The weights of m beside real, the tensors of a Gemma 3 by name, whose config is config:
    or, with no m, of that Gemma alone, as wide. Gives the config of the wide Gemma, the shapes
    of its tensors, the weights that are set, by their names in the text model, and m's grid and
    the names of its hidden units, by layer, as they are here (see place)"""
    size, hidden, c = config["hidden_size"], config["intermediate_size"], math.sqrt(2)
    clear = ladder(len(m.net.layers))[1] if m else []
    gemmas = [layer_of(OWN + i, clear) for i in range(len(config["layer_types"]))]  # here
    kinds = ["sliding_attention"] * (OWN + len(clear) + len(gemmas))  # m's own layers slide,
    for i, kind in zip(gemmas, config["layer_types"]):  # and Gemma's are as they were
        kinds[i] = kind
    ours = [i for i in range(len(kinds)) if i not in gemmas]
    layer = [k for k in real if ".layers.0." in k]  # the tensors of a layer, by those of the first
    shapes = {moved(k, {0: i}): real[k].shape for i in ours for k in layer}  # m's own layers,
    shapes |= {moved(k, gemmas): v.shape for k, v in real.items()}  # and Gemma's
    shapes = {k: [2 * d if d in (size, hidden) else d for d in s] for k, s in shapes.items()}
    config = config | {"layer_types": kinds, "num_hidden_layers": len(kinds)}
    config |= {"hidden_size": 2 * size, "intermediate_size": 2 * hidden}
    w = gemma.LazyZeros(shapes)  # a norm's weights are what it multiplies by, until the end
    grid, names = place(m, w, config) if m else ([], [])
    for k, v in real.items():  # Gemma's own: the last of each axis, in the layers of `gemmas`
        k = moved(k.removeprefix(w.prefix), gemmas)
        if k.endswith("norm.weight"):
            v = (1 + v) / c if READS.search(k) else 1 + v
        w[k][tuple(slice(-d, None) for d in v.shape)] = v * c if WRITES.search(k) else v
    for k, v in w.items():
        if k.endswith("norm.weight"):
            v -= 1  # Gemma 3's norm multiplies by 1 + weight
    return config, shapes, w, grid, names


def grid_beside(t: Trace, b: int, clear: list[int]) -> Grid:
    """t's grid here, held: ONE, the code of b bits and the features of layout(t, WAIT, clear)
    but for the outputs, then a feature for each output, where nothing else was, as the logits
    read it"""
    rows = [[ONE, *(f"code[{i}]" for i in range(b)), *row[1:]] for row in layout(t, WAIT, clear)]
    outputs = [[y if y in row else None for y in t.outputs] for row in rows]
    rows = [[None if s in t.outputs else s for s in row] for row in rows]
    kept = [f for f, column in enumerate(zip(*rows)) if any(column)]  # those left with a signal
    return held([[row[f] for f in kept] + out for row, out in zip(rows, outputs)], clear)


def unerased(m: MLP) -> MLP:
    """A layer of emit, less its units that erase what a feature held: a block here cannot know
    how large that is. What is written over it is GROW**WAIT times as large, or was cleared"""
    keep = [j for j, h in enumerate(m.hidden) if not h.startswith("erase ")]
    rows = keep + [len(m.hidden) + j for j in keep]  # their gates, then their values
    hidden = [m.hidden[j] for j in keep]
    return replace(m, hidden=hidden, wi=m.wi[rows], b=m.b[rows], wo=m.wo[:, keep])


def wipe(w: gemma.LazyZeros, i: int, features: list[int], rms: float):
    """Layer i, of the transformer's own: it clears features, that Gemma's blocks wrote bits into
    of sizes that it cannot know, but under 2**-9. Its head writes 1 into each: a value of ONE,
    as embedded, from every token. bf16 rounds what a feature held into that 1, and the unit ONE
    of its block then writes -1. Both put K too into half of Gemma's features, a quarter of the
    stream, where the norm after them drops it: what they add then has an rms of K / 2 times what
    they read ONE as, which the norm takes out, so that 1 and -1 are exact however roughly rms,
    the stream's there as ROUGH has it, is known"""
    a, wide = f"layers.{i}.", len(w["norm.weight"])
    assert wide % 4 == 0, (
        f"layer {i} clears a stream of {wide} features, and a quarter is not whole"
    )
    for x in ("input_layernorm", "pre_feedforward_layernorm"):
        w[f"{a}{x}.weight"][0] = rms / SMALL  # ONE, read as 1, more or less
    for x in ("self_attn.v", "mlp.gate", "mlp.up"):
        w[f"{a}{x}_proj.weight"][0, 0] = 1
    for x, one in (("self_attn.o", 1), ("mlp.down", -1)):
        w[f"{a}{x}_proj.weight"][wide // 2 : 3 * wide // 4, 0] = K
        w[f"{a}{x}_proj.weight"][features, 0] = one
    for x in ("post_attention_layernorm", "post_feedforward_layernorm"):
        w[f"{a}{x}.weight"][features] = K / 2


def place(m: Transformer, w: gemma.LazyZeros, config: dict):
    """m's weights into w, the tensors of the wide Gemma whose config is config, on the first
    half of its features and hidden units, but for the K that wipe puts into Gemma's. Gives m's
    grid and the names of its hidden units, by layer, as they are here: those of grid_beside and
    its layers, with a layer that clears where ladder says, and then the ONE that the heads
    write, the one that each block writes, and the outputs' twins. A feature is written over,
    and erased only where it is cleared"""
    wide, hidden = config["hidden_size"], config["intermediate_size"] // 2
    b, t = len(m.v), m.net.trace
    sizes, clear = ladder(len(m.net.layers))
    grid = grid_beside(t, b, clear)
    layers = [unerased(x) for x in emit(t, grid, "geglu", skip=True, norm=True).layers]
    n, ones = len(layers), len(grid[0]) + np.arange(len(layers) + 1)  # each ONE's feature
    outs = [grid[-1].index(y) for y in t.outputs]  # the last of grid_beside's features
    twins = ones[-1] + 1 + np.arange(b)  # of the outputs: features that hold them negated
    cleared = {}  # by each row of clear, the features cleared after it
    assert len(m.codes) == config["vocab_size"], f"{t.name} needs a code for each of Gemma's tokens"
    assert len(m.q) <= config["num_attention_heads"], f"{t.name} needs its heads in one layer"
    assert twins[-1] < wide // 2, f"{t.name} needs {twins[-1] + 1} features, and has {wide // 2}"
    assert n <= OWN + len(ROUGH), (
        f"{t.name} needs {n - OWN} of Gemma's layers, and ROUGH has {len(ROUGH)}"
    )
    assert {x.activation for x in t.mlps} <= {"relu", "reglu"}, (  # folded: no constants
        f"{t.name} needs GLUs or ReLUs, as lib.py's: bf16 rounds what a gate's units sum"
    )
    # The heads copy ONE, then the code: m.o is on m's grid, whose first features are this one's,
    # ONE, the code, the inputs. They read ONE as 1, as in gemma.py, and put K into ONE, so that
    # the norm after them makes 1 SMALL; o then scales what they write to row 0's size
    value = np.vstack([np.eye(1, m.k.shape[1]), m.v])
    gemma.attention(w, 0, m, m.q, value, np.pad(m.o, ((0, 0), (0, 0), (1, 0))), ones[:1], config)
    wrote = [*np.flatnonzero(m.o.any((0, 2))), ones[0]]  # by the heads: m's inputs, and a ONE
    w["layers.0.input_layernorm.weight"] /= SMALL
    w["layers.0.self_attn.o_proj.weight"][0, 0] = K
    w["layers.0.post_attention_layernorm.weight"][wrote] = SMALL * K / math.sqrt(wide)
    w["layers.0.self_attn.o_proj.weight"][wrote] *= sizes[0] / SMALL
    embed = w["embed_tokens.weight"]
    embed[:, : 1 + b] = np.sign(m.embed) * SMALL / math.sqrt(wide)
    for d, block in enumerate(layers):  # block d reads ones[d], and writes ones[d + 1]
        a, units, mine = f"layers.{layer_of(d, clear)}.", len(block.hidden), d < OWN
        if mine:  # in a layer of its own, where K's norm makes what a unit gives SMALL
            free, rms, read_as, after = 2 * hidden, 1, GATE[0], SMALL * K / math.sqrt(wide)
            share = sizes[d + 1] / SMALL
        else:  # beside Gemma's: a unit gives GATE, which the norm after divides by what
            free, (rms, gives), read_as, after = hidden, ROUGH[d - OWN], GATE[1], 1  # it gives
            share = 2.0 ** round(math.log2(sizes[d + 1] * gives / GATE[1]))  # a power of two
        gate, up, down = (w[f"{a}mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
        assert units <= free, f"{t.name}: layer {d} needs {units} hidden units, and has {free}"
        read = [*(1 + np.flatnonzero(block.wi[:, 1:].any(0))), ones[d]]
        assert set(read) <= set(wrote), f"{t.name}: block {d} reads what an older block wrote"
        if d in clear:  # in the layer before, all that the blocks before wrote, but what d reads
            cleared[d] = [f for f in range(1 + b, outs[0]) if f not in read]
            wipe(w, layer_of(d, clear) - 1, cleared[d], ROUGH[d - OWN][0])
        wrote = [f for f, (old, new) in enumerate(zip(grid[d], grid[d + 1])) if old != new]
        wrote.append(ones[d + 1])
        relus = not mine and not block.wi[units:, 1:].any()  # units whose value is ONE alone
        assert mine or relus, f"{t.name}: block {d} is beside Gemma's, and needs ReLUs, as ands"
        gate[:units, : block.wi.shape[1]], up[:units, : block.wi.shape[1]] = np.split(block.wi, 2)
        gate[:units, ones[d]], gate[:units, 0] = gate[:units, 0].copy(), 0  # the gates' biases
        if not relus:  # and the values': but a ReLU's stays on the ONE that is embedded
            up[:units, ones[d]], up[:units, 0] = up[:units, 0].copy(), 0
        w[a + "pre_feedforward_layernorm.weight"][read] = read_as * rms / sizes[d]
        if relus:
            w[a + "pre_feedforward_layernorm.weight"][0] = rms / SMALL  # ONE, read as 1
        down[: len(block.outputs), :units] = share * block.wo
        down[ones[d + 1], 0] = share  # the unit ONE, the first: a ONE as large as the signals
        if mine:
            down[0, 0] = K
        w[a + "post_feedforward_layernorm.weight"][wrote] = after
    assert set(outs) <= set(wrote), f"{t.name}: its last block must make every output"
    post = w[a + "post_feedforward_layernorm.weight"]  # a and down are the last block's
    down[twins], post[twins] = -down[outs], post[outs]
    embed[:, outs] = embed[:, twins] = FAINT * np.sign(m.embed[:, 1:])  # a token's code, twice,
    w["norm.weight"][outs], w["norm.weight"][twins] = LOUD, -LOUD  # where its logit reads it
    grid = [row + [ONE] * (d + 1) + [None] * (n - d + b) for d, row in enumerate(grid)]
    grid[-1][-b:] = ["-" + y for y in t.outputs]  # the twins
    names = [x.hidden for x in layers]
    for c in reversed(clear):  # a layer that clears, of the unit ONE alone, and the row after it
        names.insert(c, [ONE])
        grid.insert(c + 1, [None if f in cleared[c] else s for f, s in enumerate(grid[c])])
    return grid, names


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
    """m beside unsloth/gemma-3-270m, in out_dir: a checkpoint that transformers loads from
    there, with the names of m's features and hidden units as they are in it (see gemma.write).
    With no m, that Gemma alone, as wide. The first run fetches Gemma's weights"""
    repo = "unsloth/gemma-3-270m"  # the Gemma 3 that ROUGH was measured on
    config = gemma.checkpoint_metadata(repo)[1]
    config, shapes, w, grid, names = build_weights(m, checkpoint_weights(repo), config)
    gemma.write(out_dir, config, shapes, w, grid, names)

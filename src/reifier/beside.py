"""Beside Gemma: a transformer of transformer.py and Gemma 3's own weights, in one checkpoint,
bf16 as Gemma's is. Its next token is Gemma's, but where the transformer's outputs are the code
of a token: there it is that token. And Gemma is not moved: where the outputs are no code,
each logit is, to the bit, what Gemma alone gives in a checkpoint as wide and as deep.

The checkpoint is a Gemma twice as wide, in features and in hidden units, and OWN layers
deeper. Its stream is the transformer's features, then Gemma's. Its first OWN layers are the
transformer's: its heads, and its first blocks. Gemma's layers follow, and each of the first
of them has one of the transformer's other blocks, on hidden units of its own. Neither has
weights on the other's features: they meet where a norm takes the mean square of the whole
stream, or of all that a layer adds to it, and in the logits.

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
  either. Each signal has a feature that no block writes again, and a block reads only what
  the block before wrote (see relay.py): its signals and a ONE, on which its gates' biases
  are. All of these are as large as each other, so a gate's sign is what it was;
- a GLU then gives s^2 times what it gave. In the transformer's own layers s is known, as
  Gemma's features are a token's embedding there, whose rms is 1.03: the transformer is as in
  gemma.py, but SMALL, and the heads and each block's unit ONE put a large K where the norm
  after them drops it, so that the norm's weight is not too small for bf16, which holds it
  less 1. Beside Gemma's blocks, a bit's size would be known half as well, in powers of two,
  after each GLU. So a block there is of ReLU units, as the ands of lib.py: a unit's value is
  1, which it reads from the ONE that is embedded, whose size is known. It gives its gate
  times what the layer does to any feature, and a bit is off by what each layer was, once;
- how large is known roughly. ROUGH has, of each of Gemma's layers, the usual rms of the stream
  before its block and of what the block gives. With these, the norm before a block makes it
  read a bit as GATE, where 1 is enough for gelu to be relu, and its `down` makes it add
  SHARE of what Gemma's block adds, to each feature that it writes. After ten blocks beside
  Gemma's, on text, a bit was read as 2**20 to 2**47, where GATE is 2**40;
- the numbers are bf16, of 8 bits. Beside Gemma, what a block reads is the same in size to the
  last bit, as one weight of one norm made it of one product, by a `down` that is a power of
  two, and so are the gates of an AND where it holds, 16 and 8 times a bit: its two units
  give the same bit again, whatever its size. Where what it reads is a little off, as the
  codes of two tokens are, each copied over the rms of its own token, a GLU of lib.py squares
  that away and an AND gives the same for a count 1/4 off. A sum must be float32 before it is
  rounded: torch's is, for bf16, on a GPU and on a CPU with AVX512-BF16;
- the last block's outputs are ±u for some u: the code of a token, or all 0 for outputs of
  1/2, which are no code. Every token's code is embedded there, as ±FAINT, and the last
  norm's weight there is LOUD: a token's logit gains LOUD FAINT u over the rms of the last
  stream for each bit of its code that the outputs have, and loses as much for each other.
  What is embedded there would add to the logits too. So each output is written twice, the
  second time negated, into a twin feature that holds the same code, and whose weight is
  -LOUD: what is embedded cancels, and what is written doubles. On text the token of the
  outputs' code gained 2**19 or more, where no two of Gemma's logits were 150 apart.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

from . import gemma
from .emit import emit
from .grid import ONE, held
from .tiles import tiles
from .transformer import Transformer

OWN = 2  # the layers that the checkpoint is deeper by: the transformer's own
SMALL = 2.0**-24  # ONE and a bit in the stream, in those layers
K = 2.0**26  # what the heads and the unit ONE of a block in those layers put into ONE
GATE = 2.0**6, 2.0**40  # what a block reads a bit as: in those layers, and in Gemma's
SHARE = 2.0**-40, 2.0**-26  # what a block there adds to a feature, of what Gemma's block adds,
# and what the last block does, whose outputs the logits read
ROUGH = [  # of each layer of unsloth/gemma-3-270m, the medians on English text of the rms of
    *[(5.4, 0.79), (41, 0.31), (52, 0.28), (77, 0.22), (100, 0.11), (160, 0.13)],  # the stream
    *[(160, 0.11), (200, 0.11), (240, 0.095), (270, 0.075), (370, 0.089), (460, 0.19)],  # before
    *[(540, 0.13), (710, 0.078), (870, 0.061), (980, 0.055), (1200, 0.056), (1400, 0.056)],
]  # its block, and of what the block gives
FAINT = 2.0**-56  # a token's code where the outputs are written, for the logits
LOUD = 2.0**121  # the last norm's weight there
READS = re.compile("input_layernorm|pre_feedforward_layernorm|^norm")  # norms: Gemma's weight / c
WRITES = re.compile("o_proj|down_proj")  # what the norm after them adds: Gemma's weight * c


def later(name: str, by: int) -> str:
    """name, of a tensor, in the layer `by` after its own, if it is in a layer"""
    return re.sub(r"(?<=layers\.)\d+", lambda i: str(int(i[0]) + by), name)


def build_weights(m: Transformer | None, real: dict, config: dict):
    """The weights of m beside real, the tensors of a Gemma 3 by name, whose config is config:
    or, with no m, of that Gemma alone, as wide and as deep. Gives the config of the wide Gemma,
    the shapes of its tensors, the weights that are set, by their names in the text model, and
    the grid and the layers of m as they are here (see place)"""
    size, hidden, c = config["hidden_size"], config["intermediate_size"], math.sqrt(2)
    layer = [k for k in real if ".layers.0." in k]  # the tensors of a layer, by those of the first
    shapes = {later(k, i): real[k].shape for i in range(OWN) for k in layer}  # m's own layers,
    shapes |= {later(k, OWN): v.shape for k, v in real.items()}  # and Gemma's, after them
    shapes = {k: [2 * d if d in (size, hidden) else d for d in s] for k, s in shapes.items()}
    kinds = ["sliding_attention"] * OWN + config["layer_types"]
    config = config | {"layer_types": kinds, "num_hidden_layers": len(kinds)}
    config |= {"hidden_size": 2 * size, "intermediate_size": 2 * hidden}
    w = gemma.LazyZeros(shapes)  # a norm's weights are what it multiplies by, until the end
    grid, layers = place(m, w, config) if m else ([], [])
    for k, v in real.items():  # Gemma's own: the last of each axis, OWN layers later
        k = later(k.removeprefix(w.prefix), OWN)
        if k.endswith("norm.weight"):
            v = (1 + v) / c if READS.search(k) else 1 + v
        w[k][tuple(slice(-d, None) for d in v.shape)] = v * c if WRITES.search(k) else v
    for k, v in w.items():
        if k.endswith("norm.weight"):
            v -= 1  # Gemma 3's norm multiplies by 1 + weight
    return config, shapes, w, grid, layers


def place(m: Transformer, w: gemma.LazyZeros, config: dict):
    """m's weights into w, the tensors of the wide Gemma whose config is config, on the first
    half of its features and hidden units. Gives m's grid and layers as they are here: a feature
    for each signal, then the ONE that the heads write, the one that each block writes, and
    the outputs' twins. No feature is written twice"""
    wide, hidden = config["hidden_size"], config["intermediate_size"] // 2
    b, t = len(m.v), m.net.trace
    grid = held([[ONE, *(f"code[{i}]" for i in range(b)), *row[1:]] for row in tiles(t)])
    net = emit(t, grid, "geglu", skip=True, norm=True)
    n, ones = len(net.layers), len(grid[0]) + np.arange(len(net.layers) + 1)  # each ONE's feature
    assert len(m.codes) == config["vocab_size"], f"{t.name} needs a code for each of Gemma's tokens"
    assert len(m.q) <= config["num_attention_heads"], f"{t.name} needs its heads in one layer"
    twins = ones[-1] + 1 + np.arange(b)  # of the outputs: features that hold them negated
    assert twins[-1] < wide // 2, f"{t.name} needs {twins[-1] + 1} features, and has {wide // 2}"
    measured = f"{t.name} needs {n - OWN} of Gemma's layers, and ROUGH has {len(ROUGH)}"
    assert n <= OWN + len(ROUGH), measured
    units = all(mlp.activation in ("relu", "reglu") for mlp in t.mlps)  # folded: no constants
    assert units, f"{t.name} needs GLUs or ReLUs, as lib.py's: bf16 rounds what a gate's units sum"
    value = np.vstack([np.eye(1, m.k.shape[1]), m.v])  # the heads copy ONE, and then the code:
    # m.o is on m's grid, whose first features are those of this one: ONE, the code, the inputs
    gemma.attention(w, 0, m, m.q, value, np.pad(m.o, ((0, 0), (0, 0), (1, 0))), ones[:1], config)
    wrote = [*np.flatnonzero(m.o.any((0, 2))), ones[0]]  # by the heads: m's inputs, and a ONE
    w["layers.0.input_layernorm.weight"] /= SMALL  # they read ONE as 1, as in gemma.py, and
    w["layers.0.self_attn.o_proj.weight"][0, 0] = K  # write it SMALL, as a block does: see K
    w["layers.0.post_attention_layernorm.weight"][wrote] = SMALL * K / math.sqrt(wide)
    embed = w["embed_tokens.weight"]
    embed[:, : 1 + b] = np.sign(m.embed) * SMALL / math.sqrt(wide)
    for d, block in enumerate(net.layers):  # block d reads ones[d], and writes ones[d + 1]
        a, units, mine = f"layers.{d}.", len(block.hidden), d < OWN
        gate, up, down = (w[f"{a}mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
        free = (2 if mine else 1) * hidden
        assert units <= free, f"{t.name}: layer {d} needs {units} hidden units, and has {free}"
        read = [*(1 + np.flatnonzero(block.wi[:, 1:].any(0))), ones[d]]
        assert set(read) <= set(wrote), f"{t.name}: block {d} reads what an older block wrote"
        wrote = [f for f, (old, new) in enumerate(zip(grid[d], grid[d + 1])) if old != new]
        wrote.append(ones[d + 1])
        relus = not mine and not block.wi[units:, 1:].any()  # units whose value is ONE alone
        gate[:units, : block.wi.shape[1]], up[:units, : block.wi.shape[1]] = np.split(block.wi, 2)
        gate[:units, ones[d]], gate[:units, 0] = gate[:units, 0].copy(), 0  # the gates' biases
        if not relus:  # and the values': but a ReLU's stays on the ONE that is embedded
            up[:units, ones[d]], up[:units, 0] = up[:units, 0].copy(), 0
        rms, gives = (1, 1) if mine else ROUGH[d - OWN]  # of Gemma, where this block is
        bit = SHARE[0] if d > OWN else SMALL  # as the block before wrote it, or the heads
        w[a + "pre_feedforward_layernorm.weight"][read] = GATE[not mine] * rms / bit
        if relus:
            w[a + "pre_feedforward_layernorm.weight"][0] = rms / SMALL  # ONE, read as 1
        share = SHARE[d == n - 1] * gives / GATE[1] ** (1 if relus else 2)  # a unit gives GATE,
        share = 1 if mine else 2.0 ** round(math.log2(share))  # or its square. A power of two
        down[: len(block.outputs), :units] = share * block.wo
        down[ones[d + 1], 0] = share  # the unit ONE, the first: a ONE as large as the signals
        if mine:
            down[0, 0] = K
        after = SMALL * K / math.sqrt(wide) if mine else 1  # K's norm, or that of Gemma's block
        w[a + "post_feedforward_layernorm.weight"][wrote] = after
    outs = [grid[-1].index(y) for y in t.outputs]
    assert set(outs) <= set(wrote), f"{t.name}: its last block must make every output"
    down[twins], w[a + "post_feedforward_layernorm.weight"][twins] = -down[outs], 1
    embed[:, outs] = embed[:, twins] = FAINT * np.sign(m.embed[:, 1:])  # a token's code, twice,
    w["norm.weight"][outs], w["norm.weight"][twins] = LOUD, -LOUD  # where its logit reads it
    grid = [row + [ONE] * (d + 1) + [None] * (n - d + b) for d, row in enumerate(grid)]
    grid[-1][-b:] = ["-" + y for y in t.outputs]  # the twins
    return grid, net.layers


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
    With no m, that Gemma alone, as wide and as deep. The first run fetches Gemma's weights"""
    repo = "unsloth/gemma-3-270m"  # the Gemma 3 that ROUGH was measured on
    config = gemma.checkpoint_metadata(repo)[1]
    config, shapes, w, grid, layers = build_weights(m, checkpoint_weights(repo), config)
    gemma.write(out_dir, config, shapes, w, grid, layers)

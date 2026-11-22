"""Gemma as a target: a transformer of transformer.py in the checkpoint of gemma-3-270m or of
gemma-4-E2B.

Gemma's weights are the transformer's, on the same features: a token's embedding, tied to the
logits, the query, key, value and output of each head, each block, and the weights of the norms
around them, which Gemma has where the transformer does. What differs:

- Gemma has `size` features, more than the transformer, and a norm takes the mean square over
  all of them: so the weight of each norm is the transformer's times sqrt(width / size);
- it scales embeddings by sqrt(size), so an embedding is stored that much smaller, and the last
  norm's weights are as much larger, for the logits to be the transformer's;
- attention has norms of its own on queries and keys, in Gemma 4 on values too, and heads that
  share keys and values, 4 a layer in 270M and 8 in E2B. The first heads of the transformer are
  layer 0's. If it has more, one head of layer 1 copies as many codes again from a token
  further back, where layer 0 put them, then one of layer 2, and so on, and the blocks start in
  the last layer that attends, where all that they read is written. Layer 0 puts there the bits
  that the program reads of its own tokens: so of a token further back it must read no others;
- its numbers are bf16, of 8 bits. Nothing here needs more: a weight or a sum that is rounded
  is a little off, and what is a little off is read right (see transformer.py). The right
  token's logit is LOGIT, or what the cap of Gemma 4, 30 tanh(logit / 30), leaves of it.

Every projection not set so is zero, so its module gives zero, and layer scalars are one. A norm
of Gemma 3 multiplies by 1 + its weight, so each is stored less 1: its entries not set are -1,
and drop their features, as zeros do in Gemma 4. That goes for the norms around a module that
gives zero, too: were its weights a little off, a norm after it that multiplied by 1 would
make what it gives as large as a feature, whatever it is. The <bos>, whose embedding is zero,
stays zero at every depth, and no head whose token is in the stream attends to it. Any other
token without a code is zeros where it is embedded, and what follows it means nothing: its
heads, with no query, attend to all before it alike.
"""

import json
import math
from pathlib import Path

import numpy as np

from .transformer import READ, SIZE, VALUE, Transformer

REVISIONS = {  # the checkpoints: a repo of huggingface.co, and the revision that weights ran on
    "unsloth/gemma-3-270m": "edcf1fe394ee1168e7c5538afd104cfb1f7947e6",
    "google/gemma-4-E2B": "433813a669639fd7eaa83155828e9000ed357d53",
}


class LazyZeros(dict):
    """Arrays for the tensors of the text model, by their names in it, which the checkpoint has
    after `prefix`: of the shapes given, made when first used, zero until set"""

    def __init__(self, shapes):
        embedding = next(k for k in shapes if k.endswith("embed_tokens.weight"))
        self.shapes, self.prefix = shapes, embedding.removesuffix("embed_tokens.weight")

    def __missing__(self, k):
        self[k] = np.zeros(self.shapes[self.prefix + k], np.float32)
        return self[k]


def round_to_bf16(x) -> np.ndarray:
    """x rounded to bfloat16, to the nearest, as float32"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + (u >> 16 & 1)) & 0xFFFF0000).view(np.float32)


def build_weights(m: Transformer, shapes: dict, config: dict):
    """The weights of m as Gemma's: those that are set, by their names in the text model, the
    others being zero. shapes are checkpoint_metadata's, and config its text config. And m's grid,
    which names the first of Gemma's features at every depth"""
    w, plus = LazyZeros(shapes), config["model_type"] == "gemma3_text"
    for k in shapes:  # every norm is set, so that one with nothing to do multiplies by 0
        if k.startswith(w.prefix) and k.endswith(("layer_scalar", "norm.weight")):
            w[k.removeprefix(w.prefix)][:] = k.endswith("layer_scalar")
    size, width, name = len(w["norm.weight"]), len(m.grid[0]), m.trace.name
    assert m.size is None and m.out.any(), f"{name} has no logits of its own: it is for beside.py"
    heads, kinds = config["num_attention_heads"], config["layer_types"]
    n = -(-len(m.o) // heads)  # the layers that attend
    assert width <= size, f"{name} needs {width} features, and Gemma has {size}"
    assert "full_attention" not in kinds[:n], f"{name} needs Gemma's first {n} layers to slide"
    deep = n - 1 + len(m.blocks)  # the first block is in the last layer that attends
    assert deep <= len(kinds), f"{name} needs {deep} layers, and Gemma has {len(kinds)}"
    wide = math.sqrt(width / size)  # a norm of Gemma's, of one of m's: its mean is over size
    coded = np.flatnonzero(m.embed.any(1))  # Gemma scales an embedding by sqrt(size)
    w["embed_tokens.weight"][coded, : m.embed.shape[1]] = m.embed[coded] / math.sqrt(size)
    held = np.zeros(width)  # the square of what each feature holds, before each layer
    held[: m.embed.shape[1]] = SIZE**2
    given = m.writes * math.sqrt(width / np.count_nonzero(m.writes))  # to each feature, by heads
    for i in range(n):  # layer 0 has m's first heads; layer i, one that looks i * heads back
        o = m.o[i * heads : (i + 1) * heads]
        if i:  # at what layer 0 copied from that token and those before it: their codes
            value, o = VALUE * np.hstack([x > 0 for x in m.o[: len(o)]]).T, [np.hstack(o)]
            lost = o[0].any(0) & ~value.any(1)  # bits to copy that layer 0 did not put there
            assert not lost.any(), f"{name} reads bits {i * heads} tokens back that layer 0 has not"
        else:
            value = m.v
        wrote = np.flatnonzero(np.hstack(o).any(1))
        rms = np.sqrt(held.sum() / size)  # of the stream, and each feature read is made READ
        reads = np.where(value.any(0) | m.k.any(0), READ * rms / np.sqrt(held + (held == 0)), 0)
        writes = np.zeros(width)
        writes[wrote] = given[wrote] * math.sqrt(len(wrote) / size)
        attention(w, i, m, m.q[i * heads :][: len(o)], value, o, reads, writes, config)
        sign = np.sign(np.hstack(o).sum(1))  # a feature is given a bit, or its own, negated
        held[wrote] = (np.sqrt(held[wrote]) + sign[wrote] * given[wrote]) ** 2
    for d, b in enumerate(m.blocks, n - 1):
        block(w, d, b, wide)
    w["norm.weight"][:width] = m.out * math.sqrt(width)
    for k, v in w.items():
        if k.endswith("norm.weight"):
            v -= plus  # Gemma 3's norm multiplies by 1 + weight
    return w, m.grid


def attention(w, i, m, queries, value, outputs, reads, writes, config):
    """The attention of layer i: a head for each of queries and outputs, which are m's heads' or
    stand for them, on m's key and on value, between norms of weights reads and writes. They go
    where RoPE turns Gemma's dimensions as m's, and all read ONE. q_norm and k_norm make a query
    and a key a vector of mean square 1 times its norm's weight, and Gemma 3 multiplies their
    product by `scale`, 1/16: the weight of q_norm makes a head score its token what it does in
    m, and so any other in a sliding layer's context MARGIN less. A key and a value are stored
    as large as m has them, a query as large as the key, and an output half as large: the norms
    make of them what they will, q_norm, v_norm in Gemma 4, and the norm after the attention"""
    a, width = f"layers.{i}.self_attn.", m.k.shape[1]
    head_dim, dims = len(w[a + "q_norm.weight"]), np.arange(len(value))
    assert len(value) <= head_dim, f"layer {i}: a head of {head_dim} cannot copy {len(value)} bits"
    theta = config.get("rope_local_base_freq")  # of RoPE in a sliding layer: Gemma 3's name,
    theta = theta or config["rope_parameters"]["sliding_attention"]["rope_theta"]  # or Gemma 4's
    pairs = np.round(-np.log(m.turns) / np.log(theta) * (head_dim // 2)).astype(int)
    assert np.allclose(theta ** -(pairs / (head_dim // 2)), m.turns), "RoPE has no such turns"
    used = np.r_[pairs, pairs + head_dim // 2]
    scale = config.get("query_pre_attn_scalar", 1) ** -0.5
    for head, (q, o) in enumerate(zip(queries, outputs)):
        q = q / np.abs(q).max() * np.abs(m.k).max()  # as large as a key: q_norm makes it what it is
        w[a + "q_proj.weight"][head * head_dim + used, :width] = q
        w[a + "o_proj.weight"][:width, head * head_dim + dims] = o / 2
    for head in range(0, len(w[a + "k_proj.weight"]), head_dim):  # heads share keys and values
        w[a + "k_proj.weight"][head + used, :width] = m.k
        w[a + "v_proj.weight"][head + dims, :width] = value
    own = READ**2 * np.linalg.norm(queries[0].sum(1)) * np.linalg.norm(m.k.sum(1))  # a token's
    w[a + "q_norm.weight"][used] = own / head_dim / scale
    w[a + "k_norm.weight"][used] = 1
    w[f"layers.{i}.input_layernorm.weight"][:width] = reads
    w[f"layers.{i}.post_attention_layernorm.weight"][:width] = writes


def block(w, i, b, wide):
    """The feed-forward block of layer i: b, of the transformer, its layer between norms of its
    weights, times wide"""
    a, n, width = f"layers.{i}.", len(b.layer.hidden), len(b.reads)
    gate, up, down = (w[f"{a}mlp.{p}_proj.weight"] for p in ("gate", "up", "down"))
    assert n <= len(gate), f"layer {i} needs {n} hidden units, and Gemma has {len(gate)}"
    gate[:n, :width], up[:n, :width] = b.layer.wi[:n], b.layer.wi[n:]
    down[:width, :n] = b.layer.wo
    w[a + "pre_feedforward_layernorm.weight"][:width] = b.reads * wide
    w[a + "post_feedforward_layernorm.weight"][:width] = b.writes * wide


def cache_file(repo: str) -> Path:
    """Where checkpoint_metadata keeps what it fetched of repo"""
    return Path("~/.cache/reifier", repo, REVISIONS[repo] + ".json").expanduser()


def checkpoint_metadata(repo: str) -> tuple[dict, dict]:
    """The shapes of the tensors of repo's checkpoint, at its revision of REVISIONS, and its
    config: from huggingface.co the first time, with huggingface_hub, and then from cache_file"""
    file, rev = cache_file(repo), REVISIONS[repo]
    if not file.exists():
        from huggingface_hub import get_safetensors_metadata, hf_hub_download

        meta = get_safetensors_metadata(repo, revision=rev).files_metadata["model.safetensors"]
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=rev)).read_text())
        shapes = {k: t.shape for k, t in meta.tensors.items()}
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps([shapes, config]))
    return tuple(json.loads(file.read_text()))


def save(m: Transformer, out_dir: Path, repo: str):
    """m in out_dir as a checkpoint of repo: what write puts there, with repo's config and the
    names and shapes of its tensors (see checkpoint_metadata), as bf16"""
    shapes, config = checkpoint_metadata(repo)
    w, grid = build_weights(m, shapes, config.get("text_config", config))  # all, if only of text
    write(out_dir, config, shapes, w, grid, [b.layer.hidden for b in m.blocks])


def write(out_dir: Path, config: dict, shapes: dict, w, grid, hidden):
    """A checkpoint in out_dir, which transformers loads from there: config.json, and
    model.safetensors, with the tensors of shapes as bf16: those of w, the others zeros. Chunks
    of zeros are left as holes in the file, which read as zeros. The names go in
    out_dir/names.json: grid, whose rows are the features after the heads and after each layer
    that follows, and hidden, the names of the hidden units of each: of `save`, the layers from
    the last that attends on, as far as the transformer has blocks"""
    header, end = {"__metadata__": {"format": "pt"}}, 0
    for k, s in shapes.items():
        header[k] = {"dtype": "BF16", "shape": s, "data_offsets": [end, end + 2 * math.prod(s)]}
        end += 2 * math.prod(s)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config))
    names = {"features": grid, "hidden": hidden}
    (out_dir / "names.json").write_text(json.dumps(names))
    with open(out_dir / "model.safetensors", "wb") as file:
        file.write(len(head).to_bytes(8, "little") + head)
        for k, v in w.items():
            start, v = 8 + len(head) + header[w.prefix + k]["data_offsets"][0], v.reshape(-1)
            for i in range(0, len(v), 1 << 20):
                if (chunk := v[i : i + (1 << 20)]).any():
                    file.seek(start + 2 * i)
                    bf16 = round_to_bf16(chunk).view(np.uint32) >> 16
                    file.write(bf16.astype(np.uint16).tobytes())
        file.truncate(8 + len(head) + end)

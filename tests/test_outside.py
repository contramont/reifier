"""Outside the compiler, on the same trace and grid: folding, tiles, Keccak and Gemma."""

import hashlib
import json
import zlib

import keccak
import numpy as np
import pytest
from keccak import keccak_p, sponge

from reifier import gemma
from reifier.check import check_grid, check_trace, check_transformer
from reifier.fold import fold
from reifier.grid import layout
from reifier.lib import add, xor
from reifier.tiles import tiles
from reifier.trace import const, mlp, threshold_gate, trace
from reifier.transformer import transformer

PLAIN = {  # Keccak's operations on plain bits, with no tracing
    "parity": lambda xs, name: sum(xs) % 2,
    "chi": lambda a, b, c, flip, name: a ^ (c & (1 - b)) ^ flip,
    "const": lambda bits: [int(v) for v in bits],
}
KECCAK_F_400 = (  # of the state of zeros, as the Keccak team gives it: 25 lanes of 16 bits
    "09F5 40AC 0FA9 14F5 E89F ECA0 5BD1 7870 EFF0 BF8F 0337 6052 DC75 "
    "0EC9 E776 5246 59A1 5D81 6D95 6E14 633E 58EE 71FF 714C B38E"
)


def test_folding_is_exact_and_moves_constants_into_biases():
    def program(x):  # both reads a constant only, so it gives constants in turn: 1 and 0
        both = mlp(const("1"), [[1], [0]], [-1, -1], "step", np.eye(2), "both")
        return [threshold_gate([x[0], both[1]], [1, 1], 1), xor([x[0], const("1")[0], x[1]])]

    t = trace(program, 2)
    f = fold(t)
    assert check_trace(f, t) == 4
    assert len(f.mlps) == 5 < len(t.mlps) == 8  # the constants are gone, into biases
    always = fold(trace(lambda x: [x[0], *const("1")], 1))  # a constant that is an output stays
    assert [m.inputs for m in always.mlps] == [[]] and always("0") == [0, 1]


def test_tiles_give_each_signal_a_feature_in_the_order_of_the_trace():
    t = trace(add, 8, 8)
    g = tiles(t)
    check_grid(t, g)
    signals = ["ONE", *t.inputs, *(m.name for m in t.mlps)]
    assert [set(column) - {None} for column in zip(*g)] == [{s} for s in signals]
    assert [set(row) - {None} for row in g] == [set(row) - {None} for row in layout(t)]


def test_keccak_is_keccak():  # on plain bits, with no tracing
    lanes = [int(lane, 16) for lane in KECCAK_F_400.split()]
    state = keccak_p([0] * 400, 20, PLAIN["parity"], PLAIN["chi"])
    assert state == [lane >> z & 1 for lane in lanes for z in range(16)]
    msg = bytes(range(8))  # and SHAKE128 is this sponge on Keccak-p[1600, 24], after 1111
    bits = [byte >> i & 1 for byte in msg for i in range(8)] + [1, 1, 1, 1]
    digest = [byte >> i & 1 for byte in hashlib.shake_128(msg).digest(8) for i in range(8)]
    assert sponge(bits, 24, 1600, 1344, **PLAIN)[:64] == digest


def next_token(x):  # keccak's, on plain bits
    y = sponge(x, **PLAIN)[:8]
    return y if y[7] else keccak.CODES[keccak.DASH]


def test_the_sponge_goes_through_every_stage():
    f = keccak.program()  # next_token, traced and folded: its padding is constants
    rng = np.random.default_rng(0)
    assert check_trace(f, next_token, rng.integers(0, 2, (256, 24)))
    assert len(f.mlps) == 146 and f.mlps[-1].name == "next[7]"  # 172 before folding
    m, first, dash = transformer(f, keccak.CODES), keccak.FIRST, keccak.DASH
    check_transformer(m, first + rng.integers(0, 256, 16))
    assert m.generate([dash] * 3, 3)[3:] == [first + 146, first + 140, dash]  # on from "-" alone


G270M, RAN = "unsloth/gemma-3-270m", 2020039791  # the checksum of the weights that ran on it


def test_gemma_weights_are_those_that_ran_on_the_model():
    if not gemma.cache_file(G270M).exists():
        pytest.skip("needs the shapes of Gemma's tensors")
    m = transformer(keccak.program(), keccak.CODES)
    w, checksum = gemma.build_weights(m, *gemma.checkpoint_metadata(G270M))[0], 0
    for k in sorted(w):  # each tensor set: its name, its rows and columns not zero, their values
        v = np.atleast_2d(w[k])
        at = [np.flatnonzero(v.any(axis)) for axis in (1, 0)]
        named = zlib.crc32((w.prefix + k).encode() + at[0].tobytes() + at[1].tobytes(), checksum)
        checksum = zlib.crc32(v[np.ix_(*at)].tobytes(), named)
    assert checksum == RAN, "the weights changed: run them on the model again"


def test_gemma_runs_what_the_transformer_runs(tmp_path):
    # uv run --python 3.14 --with torch --with transformers python -m pytest -k gemma_runs
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    m = transformer(keccak.program(), keccak.CODES)
    tokens = keccak.FIRST + np.random.default_rng(0).integers(0, 256, 24)
    gemma.save(m, tmp_path, G270M)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    ids = torch.tensor([[model.config.bos_token_id, *tokens]])
    with torch.no_grad():  # as transformers runs it by default, after a <bos>
        out = model(ids, output_hidden_states=True)
    streams, logits = m.run(tokens)
    n = len(m.q)  # the symbols that the program reads: position n is the first that has them
    for d in range(1, len(streams)):  # row d of the grid is what layer d - 1 gives, ONE aside
        got = out.hidden_states[d][0, n:, 1 : streams[d].shape[1]].float().numpy()
        assert (got == streams[d][n:, 1:].round()).all(), f"row {d}"
    assert (out.logits[0, n:].argmax(1).numpy() == logits[n:].argmax(1)).all()


P = "text."  # before the names of the tensors: Gemma 3 has model., Gemma 4 model.language_model.
LAYER = (  # the tensors of a layer of Gemma that gemma.py sets: those of self_attn, if it attends
    "layer_scalar input_layernorm.weight post_attention_layernorm.weight "
    "pre_feedforward_layernorm.weight post_feedforward_layernorm.weight mlp.gate_proj.weight "
    "mlp.up_proj.weight mlp.down_proj.weight self_attn.q_norm.weight self_attn.k_norm.weight "
    "self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight "
    "self_attn.o_proj.weight"
)
G4 = {"model_type": "gemma4_text", "rope_parameters": {"sliding_attention": {"rope_theta": 1e4}}}
G3 = {"model_type": "gemma3_text", "rope_local_base_freq": 1e4, "query_pre_attn_scalar": 256}
G3 |= {"rope_theta": 1e6}  # of its layers that do not slide: not the one to read


def small_gemma(heads=4, model=G4):
    """As gemma.checkpoint_metadata, for a Gemma of 19 features, heads of 16 and 4 layers"""
    names = ["norm.weight", *(f"layers.{i}.{k}" for i in range(4) for k in LAYER.split())]
    shapes = {P + "embed_tokens.weight": [90000, 19]}  # two of save's chunks: the first is zeros
    shapes |= {P + k: [64, 64] if "proj" in k else [16] if "_norm" in k else [19] for k in names}
    return shapes, {"num_attention_heads": heads, "layer_types": ["sliding_attention"] * 4} | model


def count(bits):  # the bits of the last two tokens -> the bits of the next: their sum, mod 4
    return add(bits[:2], bits[2:])


def test_save_writes_every_tensor_as_bf16_where_its_header_says(tmp_path, monkeypatch):
    monkeypatch.setattr(gemma, "checkpoint_metadata", {"a small Gemma": small_gemma()}.get)
    m = transformer(trace(count, 4), {85000 + n: [n >> 1, n & 1] for n in range(4)})  # late tokens
    out = tmp_path / "a/b"
    for _ in range(2):  # into a directory that is not there, then into one that is
        gemma.save(m, out, "a small Gemma")
    w, grid = gemma.build_weights(m, *small_gemma())
    file = (out / "model.safetensors").read_bytes()
    n = int.from_bytes(file[:8], "little")
    header, end = json.loads(file[8 : 8 + n]), 0
    assert n % 8 == 0 and header.pop("__metadata__") == {"format": "pt"}
    for k, shape in small_gemma()[0].items():  # each tensor after the one before it, rounded
        start, end = end, end + 2 * int(np.prod(shape))
        assert header.pop(k) == {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
        saved = np.frombuffer(file, "<u2", (end - start) // 2, 8 + n + start)
        assert (saved == gemma.round_to_bf16(w[k[len(P) :]]).view(np.uint32).ravel() >> 16).all(), k
    assert file[7 + n : 8 + n] == b" " and len(file) == 8 + n + end  # the header was padded
    names, config = (json.loads((out / f).read_text()) for f in ("names.json", "config.json"))
    assert names == {"features": grid, "hidden": [x.hidden for x in m.net.layers]}
    assert names["hidden"][0][0] == "ONE" and config == small_gemma()[1]
    assert gemma.round_to_bf16(np.float32([1 + 2**-8, 1 + 3 * 2**-8])).tolist() == [1, 1 + 2**-6]


def test_gemma_has_the_weights_of_the_transformer_and_pads_what_bf16_cannot_norm():
    shapes, config = small_gemma()
    m = transformer(trace(count, 4), {5: [0, 1]})
    w, grid = gemma.build_weights(m, shapes, config)
    assert grid[0][7:] == [None, None, "pad[0]", "pad[1]"]  # the norms see 4, 7 and 9 signals
    assert not gemma.exact(7, 19, 0) and not gemma.exact(21, 1536, 0)  # the last by its eps alone
    a, used = "layers.0.self_attn.", [*range(16), *range(16, 32)]  # two heads of 16 dimensions
    assert np.allclose(w[a + "q_proj.weight"][used, :9], np.vstack(m.q))  # m's queries,
    assert np.array_equal(w[a + "k_proj.weight"][:16, :9], m.k)  # its key, for every head,
    assert np.array_equal(w[a + "v_proj.weight"][:2, :9], m.v)  # and its value: the code
    o = w[a + "o_proj.weight"]  # its outputs, scaled, and head 0 writes its first bit into the pads
    assert np.allclose(o[:9, [0, 1, 16, 17]], (2 / 16) ** 0.5 * np.hstack(m.o))
    assert o[9:].nonzero()[1].tolist() == [0, 0] and np.allclose(o[o != 0], (2 / 16) ** 0.5)
    gate = w["layers.0.mlp.gate_proj.weight"]  # and a block is a layer of m
    assert np.array_equal(gate[:5, :9], m.net.layers[0].wi[:5]) and not gate[5:].any()
    assert np.allclose(w["norm.weight"][:3], [0, 4 * 11**0.5, 4 * 11**0.5])  # 8 / 2 sqrt(11)
    assert (w["layers.3.layer_scalar"] == 1).all()  # as that of every layer
    five = gemma.build_weights(transformer(trace(xor, 5), {5: [1]}), shapes, config)[1][0]
    assert len(five) == 7 + 2 * 5  # two layers attend, and a row sees the pads of both
    with pytest.raises(AssertionError, match="needs 9 features, and Gemma has 8"):
        gemma.build_weights(m, shapes | {P + "norm.weight": [8]}, config)
    for slide, said in ((2, "5 layers, and Gemma has 4"), (1, "Gemma's first 2 layers to slide")):
        kinds = ["sliding_attention"] * slide + ["full_attention", "sliding_attention"]
        with pytest.raises(AssertionError, match=said):  # with a head a layer, 2 layers attend
            gemma.build_weights(m, *small_gemma(1, G4 | {"layer_types": kinds}))
    thin = shapes | {P + f"layers.{i}.mlp.gate_proj.weight": [n, 64] for i, n in enumerate((5, 12))}
    with pytest.raises(AssertionError, match="layer 1 needs 13 hidden units, and Gemma has 12"):
        gemma.build_weights(m, thin, config)  # layer 0 needs 5, and has them
    rounded = transformer(trace(lambda x: threshold_gate(x, [1 / 3, 1], 1), 2), {0: [0], 1: [1]})
    with pytest.raises(AssertionError, match="layer 0: a gate's weights or its sums are not"):
        gemma.build_weights(rounded, shapes, config)


def test_gemma_3_stores_norms_less_one_and_a_later_layer_copies_what_layer_0_copied():
    m = transformer(trace(xor, 5), {5: [1]})  # 5 tokens of a bit, 2 heads a layer: 3 layers attend
    w, grid = gemma.build_weights(m, *small_gemma(2, G3))
    assert grid[0][7:] == ["pad[0]", "pad[1]", "pad[2]"]  # one of each layer, for its norm
    q, v, o = ([w[f"layers.{i}.self_attn.{k}_proj.weight"] for i in range(3)] for k in "qvo")
    assert np.allclose(q[2][:16, :7], m.q[4])  # layer 2 has one head, m's head 4: it looks 4 back,
    assert v[2][0].nonzero()[0].tolist() == [6] and o[2][:, 0].nonzero()[0].tolist() == [2, 9]
    assert np.array_equal([x[7:10, 0] for x in o], np.eye(3))  # reads xs4 there, writes xs0 and pad
    assert np.flatnonzero(w["layers.1.input_layernorm.weight"] + 1).tolist() == [0, 5, 6]  # and ONE
    own = np.linalg.norm(m.q[4]) * np.linalg.norm(m.k)  # what the head scores its token in m
    assert np.isclose(1 + w["layers.2.self_attn.q_norm.weight"].max(), own)  # over 16, by 1 / 16
    assert "layers.1.mlp.gate_proj.weight" not in w  # the first block is in layer 2: no k before,
    post = w["layers.2.post_feedforward_layernorm.weight"][:3]  # and none after, nor the code,
    assert np.allclose(post, [-1, -1, 2**14 / 19**0.5 - 1])  # which the block does not write
    assert np.allclose(w["norm.weight"][:3], [-1, 8 * 10**0.5 - 1, -1])  # each is stored less 1
    off = [c for c in range(1, 641) if not gemma.exact(c, 640, 1)]  # of 640 features, as in 270M
    assert len(off) == 78 and off[:10] == [*range(1, 10), 11] and off[-1] == 156  # none above
    thin = small_gemma(2, G3)[0] | {P + "layers.1.self_attn.q_norm.weight": [1]}  # a head of 1
    with pytest.raises(AssertionError, match="layer 1: a head of 1 cannot copy 2 bits"):
        gemma.build_weights(m, thin, small_gemma(2, G3)[1])

"""Outside the compiler, on the same trace and grid: folding, tiles, Keccak and Gemma."""

import hashlib
import json
import zlib
from itertools import pairwise
from pathlib import Path

import alongside
import keccak
import numpy as np
import pytest
from keccak import keccak_p, sponge

from reifier import beside, gemma
from reifier.check import check_grid, check_trace, check_transformer
from reifier.emit import emit
from reifier.fold import fold
from reifier.grid import layout
from reifier.lib import add, ands, parity, xor
from reifier.relay import relay
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


def test_relays_pass_a_signal_on_through_each_layer_to_the_one_that_reads_it():
    def late(x):  # the last parity reads x1 three layers after it is given, and a is an output
        a = parity(x)
        return [a, parity([parity([a]), x[1]])]

    t = trace(late, 2)
    r = relay(t)
    assert check_trace(r, t) == 4  # the same outputs, as relays, and the same signals
    assert r.outputs == ["parity''", "parity.2"] and r.mlps[4].inputs == ["parity.1", "x1''"]
    relays = ["x1'", "x1''", "parity'", "parity''"]  # each two ReLU units, on the one before
    assert [m.name for m in r.mlps] == ["parity", "parity.1", *relays[:2], "parity.2", *relays[2:]]
    assert [m.inputs for m in r.mlps[2:4]] == [["x1"], ["x1'"]]
    assert len(relay(r).mlps) == 7 and relay(r).outputs == r.outputs  # no more, the second time


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
    zeros = keccak_p([0] * 50, 4, PLAIN["parity"], PLAIN["chi"])  # rounds 10 to 13 of Keccak-f[50]
    assert "".join(map(str, zeros)) == "10010101010111010111101100000100001111111001110100"


def next_keccak(x):  # keccak's next_token, on plain bits
    y = sponge(x, **PLAIN)[:8]
    return y if y[7] else keccak.CODES[keccak.DASH]


def test_the_sponge_goes_through_every_stage():
    f = keccak.program()  # next_token, traced and folded: its padding is constants
    rng = np.random.default_rng(0)
    assert check_trace(f, next_keccak, rng.integers(0, 2, (256, 24)))
    assert len(f.mlps) == 146 and f.mlps[-1].name == "next[7]"  # 172 before folding
    m, first, dash = transformer(f, keccak.CODES), keccak.FIRST, keccak.DASH
    check_transformer(m, first + rng.integers(0, 256, 16))
    assert m.generate([dash] * 3, 3)[3:] == [first + 146, first + 140, dash]  # on from "-" alone


G270M, RAN = "unsloth/gemma-3-270m", 2020039791  # the checksum of the weights that ran on it
WEIGHTS_AND_TOKENIZER = "model.safetensors", "tokenizer.json", "tokenizer_config.json"
README = Path(__file__).parent.parent / "README.md"  # text, to run Gemma on


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


def next_alongside(x):  # alongside's next_token, on plain bits: 1/2 for each where it is silent
    y = sponge(x, 7, 200, 160, **PLAIN)[:18]
    return [0.5] * 18 if y[17] else y


def test_keccak_alongside_gemma_is_seven_rounds_on_two_tokens_in_twenty_layers():
    f = alongside.program()
    assert check_trace(f, next_alongside) == 256
    depth, units = dict.fromkeys(f.inputs, 0), [0] * 21  # of each signal, and of each layer
    for x in f.mlps:  # relayed: each reads the layer before it, and no other
        (d,) = {depth[i] + 1 for i in x.inputs}
        depth |= dict.fromkeys(x.outputs, d)
        units[d] += len(x.hidden)
    assert units[1] == 3152 and max(units[2:]) == 1680  # and a ONE: of 4096, of 2048
    assert units[2:] == [1680, 1600, 1200] * 5 + [1348, 272, 108, 108]  # 3 layers a round
    relus = {x.activation for x in f.mlps if depth[x.outputs[0]] > 1}  # those after the first
    assert relus == {"relu"} and len([x for x in f.mlps if x.name.endswith("'")]) == 1034
    assert beside.ladder(20)[1] == [15]  # rows grow up to LAST by row 15, and are then cleared
    assert len(beside.grid_beside(f, 18, [15])[0]) == 511  # its 3,580 signals, ONE and the code
    m = transformer(f, alongside.CODES)  # on every token of Gemma
    check_transformer(m, range(1000, 1032))
    assert len(m.net.layers) == 20 and len(m.q) == 2
    assert len(set(map(tuple, alongside.CODES.values()))) == 2**18  # each code is of one token


def test_gemma_with_the_sponge_beside_it_gives_what_each_gives_alone(tmp_path):
    # uv run --python 3.14 --with torch --with transformers python -m pytest -k beside
    # once examples/alongside.py has fetched Gemma's own weights, and transformers its tokenizer:
    # no other test runs beside.py's constants, ROUGH, LOUD and the like, on the model
    torch = pytest.importorskip("torch")
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = gemma.REVISIONS[G270M]
    files = [try_to_load_from_cache(G270M, f, revision=rev) for f in WEIGHTS_AND_TOKENIZER]
    if not (all(isinstance(f, str) for f in files) and gemma.cache_file(G270M).exists()):
        pytest.skip("needs Gemma's weights and tokenizer")
    beside.save(transformer(alongside.program(), alongside.CODES), tmp_path)
    beside.save(None, tmp_path / "alone")  # Gemma alone, as wide and as deep
    device, load = "cuda" if torch.cuda.is_available() else "cpu", AutoModelForCausalLM
    models = [load.from_pretrained(tmp_path), load.from_pretrained(tmp_path / "alone")]
    models.append(load.from_pretrained(G270M, revision=rev))  # Gemma itself: all three bf16
    assert {x.dtype for x in models} == {torch.bfloat16}
    assert [x.config.num_hidden_layers for x in models] == [21, 20, 18]  # and one that clears
    text = AutoTokenizer.from_pretrained(G270M, revision=rev)(README.read_text())["input_ids"]
    text = text[: 1024 if device == "cuda" else 256]  # a <bos>, and the start of the README
    ids = torch.tensor([text], device=device)
    with torch.no_grad():
        out = models[0].to(device)(ids, output_hidden_states=True)
        both, alone, itself = out.logits[0], *(x.to(device)(ids).logits[0] for x in models[1:])
    names = json.loads((tmp_path / "names.json").read_text())
    i = names["hidden"].index(["ONE"])  # the layer that clears: of the unit ONE alone
    cleared = [f for f, (s, t) in enumerate(zip(*names["features"][i : i + 2])) if s and not t]
    assert i == 15 and len(cleared) == 240  # beside.py's ROUGH and the like are as they were:
    assert not out.hidden_states[i + 1][0, :, cleared].any()  # it leaves them 0, to the bit
    bits = np.array([alongside.CODES[t] for t in text])  # position 0 reads its token twice
    y = np.array([next_alongside([*older, *newer]) for older, newer in pairwise([bits[0], *bits])])
    speaks = torch.tensor(y[:, 0] != 0.5, device=device)
    token = torch.tensor(y.round().astype(int) @ (1 << np.arange(18)), device=device)
    assert len(text) / 4 < speaks.sum() < 3 * len(text) / 4  # half of the time
    assert (both.argmax(1) == token)[speaks].all()  # there, the token of the sponge's code
    assert (both == alone)[~speaks].all()  # and elsewhere each logit of Gemma alone, to the bit
    assert (alone.argmax(1) == itself.argmax(1)).float().mean() > 0.95  # bf16 breaks Gemma's ties


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


def small_gemma(heads=4, model=G4, size=19):
    """As gemma.checkpoint_metadata, for a Gemma of `size` features, heads of 16 and 4 layers"""
    names = ["norm.weight", *(f"layers.{i}.{k}" for i in range(4) for k in LAYER.split())]
    shapes = {P + "embed_tokens.weight": [90000, size]}  # two of save's chunks: the first is zeros
    shapes |= {P + k: [64, 64] if "proj" in k else [16] if "_norm" in k else [size] for k in names}
    return shapes, {"num_attention_heads": heads, "layer_types": ["sliding_attention"] * 4} | model


def beside_gemma(size=19):
    """For beside.build_weights: the tensors of a small Gemma 3, at random, and its config, with
    the 64 of a projection as its hidden units"""
    shapes, config = small_gemma(4, G3, size)
    config |= {"hidden_size": size, "intermediate_size": 64, "vocab_size": 90000}
    real = {k: np.random.default_rng(0).standard_normal(s, np.float32) for k, s in shapes.items()}
    return real, config


BIT = {t: [t & 1] for t in range(90000)}  # a code for each token of the small Gemma: a bit


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


def thrice(bits):  # of the last two tokens, a bit each: their xor, through three layers
    return ands([parity([parity(bits)])], [([1], 1)])


def test_beside_gemma_its_own_weights_are_on_the_last_features_of_the_later_layers(
    tmp_path, monkeypatch
):
    real, config = beside_gemma()
    m, c = transformer(trace(thrice, 2), BIT), 2**0.5
    monkeypatch.setattr(beside, "WAIT", 3)  # longer than thrice: a feature for each signal
    wide, wide_shapes, w, grid, names = beside.build_weights(m, real, config)
    assert (wide["hidden_size"], wide["num_hidden_layers"]) == (38, 4 + 2)  # two layers of m's own
    assert wide["layer_types"][:2] == ["sliding_attention"] * 2  # where RoPE is as m's
    alone = beside.build_weights(None, real, config)  # Gemma alone, as wide: the same config,
    assert alone[:2] == (wide, wide_shapes) and alone[3:] == ([], [])  # and nothing beside it
    for k, v in real.items():  # each of Gemma's: over c in a norm that reads the stream, by c in
        k = beside.moved(k[len(P) :], range(2, 6))  # what writes it, and else as it was
        reads = k == "norm.weight" or "input_layernorm" in k or "pre_feedforward" in k
        want = (1 + v) / c - 1 if reads else v * c if "o_proj" in k or "down_proj" in k else v
        last = tuple(slice(-d, None) for d in v.shape)
        assert np.allclose(w[k][last], want, atol=1e-6)
        assert np.array_equal(alone[2][k][last], w[k][last])  # with m and without
        if "proj" in k:  # and neither has weights on the other's features or hidden units
            assert not w[k][:64, 64:].any() and not w[k][64:, :64].any(), k
            assert not alone[2][k][:64].any() and not alone[2][k][:, :64].any(), k
    assert grid[-1][:7] == ["ONE", "code[0]", "bits0", "bits1", "parity", "parity.1", "ands"]
    assert grid[0][7:] == ["ONE", None, None, None, None]  # the ONE of the heads, and of each
    assert grid[-1][7:] == ["ONE"] * 4 + ["-ands"]  # block, and the twin of the output
    small, k, gate, loud = beside.SMALL, beside.K, beside.GATE, beside.LOUD
    sizes = [beside.FIRST * beside.GROW**d for d in range(3)] + [beside.LAST]  # of a bit, by row
    embed, after = w["embed_tokens.weight"], small * k / 38**0.5  # the norm after what K is in
    assert np.allclose(embed[[4, 5], :2] * 38**0.5, [[small, -small], [small, small]])
    assert (embed[[4, 5]][:, [6, 11]] == beside.FAINT * np.array([[-1, -1], [1, 1]])).all()
    assert (w["norm.weight"][[6, 11]] == [loud, -loud]).all() and not embed[:, 12:19].any()
    o = w["layers.0.self_attn.o_proj.weight"]  # head 0 copies ONE, for block 0, and puts K in ONE
    reads, writes = (w[f"layers.0.{x}_layernorm.weight"] + 1 for x in ("input", "post_attention"))
    assert (o[[7, 0], 0] == [sizes[0] / small, k]).all() and reads[:2].tolist() == [1 / small] * 2
    assert np.allclose(writes[[2, 3, 7]], after) and not writes[[0, 1, 4]].any()
    blocks = emit(m.net.trace, beside.grid_beside(m.net.trace, 1, []), "geglu", True, True)
    layers = [beside.unerased(x) for x in blocks.layers]  # as place makes them
    for d, (rms, gives) in enumerate([(1, None), (1, None), beside.ROUGH[0]]):
        a, bias = f"layers.{d}.", layers[d].wi[: len(layers[d].hidden), 0]  # of each gate
        gates, values = w[a + "mlp.gate_proj.weight"], w[a + "mlp.up_proj.weight"]
        assert np.array_equal(gates[: len(bias), 7 + d], bias) and not gates[:, 0].any()
        pre = w[a + "pre_feedforward_layernorm.weight"][:19] + 1  # on the ONE of the block before
        read = sorted({2, 3} if d == 0 else {3 + d}) + [7 + d]  # which it reads, with a signal
        assert np.flatnonzero(pre).tolist() == [0] * (d == 2) + read
        assert np.allclose(pre[read], gate[d == 2] * rms / sizes[d])  # as large as its row says
        down, post = w[a + "mlp.down_proj.weight"], w[a + "post_feedforward_layernorm.weight"]
        size = 2.0 ** round(np.log2(sizes[3] * gives / gate[1])) if gives else sizes[d + 1] / small
        assert down[[4 + d, 8 + d], 0].tolist() == [-size, size]  # it writes a signal and a ONE
        assert down[0, 0] == (d < 2) * k  # which the norm after it drops
        assert np.flatnonzero(post[:19] + 1).tolist() == [4 + d, 8 + d] + [11] * (d == 2)
        assert np.allclose(post[4 + d] + 1, 1 if gives else after)
    assert values[:3, 0].tolist() == [1 / 32] * 3 and not values[:3, 9].any()  # a ReLU's value is
    assert np.isclose(pre[0], beside.ROUGH[0][0] / small)  # the ONE that is embedded, read as 1,
    assert np.array_equal(down[11], -down[6])  # and the twin holds the output, negated
    hidden = [["ONE", "parity#0", "parity#1"], ["ONE", "parity.1"], ["ONE", "ands#0", "ands#1"]]
    monkeypatch.setattr(beside, "WAIT", 1)  # a feature written again the layer after: parity is
    over = beside.build_weights(m, real, config)[3:]  # written over bits1, and parity.1 over it,
    assert [row[3] for row in over[0]] == ["bits1", "parity", "parity.1", "parity.1"]
    assert over[1] == hidden  # and no unit erases what the feature held
    gemma.write(tmp_path, wide, wide_shapes, w, grid, names)  # with the names as they are here
    names = json.loads((tmp_path / "names.json").read_text())
    assert names == {"features": grid, "hidden": hidden}


def test_beside_gemma_says_what_it_cannot_place(monkeypatch):
    real, config = beside_gemma()
    with pytest.raises(AssertionError, match="thrice needs a code for each of Gemma's tokens"):
        beside.build_weights(transformer(trace(thrice, 2), {5: [1]}), real, config)
    older = lambda x: ands([parity([parity(x)]), x[0]], [([1, 1], 1)])  # reads x0 three layers on
    with pytest.raises(AssertionError, match="block 2 reads what an older block wrote"):
        beside.build_weights(transformer(trace(older, 2), BIT), real, config)
    beside.build_weights(transformer(relay(trace(older, 2)), BIT), real, config)  # with relays
    deep = lambda x: ands([parity([parity([parity(x)])])], [([1], 1)])  # a GLU in block 2
    with pytest.raises(AssertionError, match="block 2 is beside Gemma's, and needs ReLUs"):
        beside.build_weights(transformer(trace(deep, 2), BIT), real, config)
    gates = transformer(trace(lambda x: threshold_gate(x, [1, 1], 1), 2), BIT)
    with pytest.raises(AssertionError, match="lambda needs GLUs or ReLUs, as lib.py's"):
        beside.build_weights(gates, real, config)
    early = lambda x: [parity(x[:2]), parity([parity(x[2:])])]  # block 0 makes its first output
    two_bits = {t: [t >> 1 & 1, t & 1] for t in range(90000)}
    with pytest.raises(AssertionError, match="its last block must make every output"):
        beside.build_weights(transformer(trace(early, 4), two_bits), real, config)
    monkeypatch.setattr(beside, "ROUGH", [])  # none of Gemma's layers is measured
    with pytest.raises(AssertionError, match="thrice needs 1 of Gemma's layers, and ROUGH has 0"):
        beside.build_weights(transformer(trace(thrice, 2), BIT), real, config)


def test_beside_gemma_a_layer_of_its_own_clears_where_bits_would_grow_too_large(monkeypatch):
    real, config = beside_gemma(20)  # twice as wide, a quarter of it is whole, as a Gemma's is
    monkeypatch.setattr(beside, "GROW", 2.0**10)  # rows of 2**-106, -96, -86 and -76, but row 4
    monkeypatch.setattr(beside, "LAST", 2.0**-76)  # would be over LAST: it is ANEW, after a clear
    once = lambda x: ands([x], [([1], 1)])
    five = lambda x: once(once(once(parity([parity(x)]))))  # three blocks beside Gemma's
    m = transformer(trace(five, 2), BIT)
    sizes, clear = beside.ladder(5)
    assert sizes == [2**-106, 2**-96, 2**-86, 2**-76, beside.ANEW, 2**-76] and clear == [3]
    wide, _, w, grid, names = beside.build_weights(m, real, config)
    assert wide["num_hidden_layers"] == 7 and names[3] == ["ONE"]  # layer 3 clears, and blocks
    assert [beside.layer_of(d, clear) for d in range(5)] == [0, 1, 2, 4, 5]  # 3 and 4 are after
    k = next(k for k in real if "layers.1.mlp.down" in k)  # so Gemma's layer 1 is layer 4
    assert np.allclose(w[beside.moved(k[len(P) :], {1: 4})][-64:, -64:], real[k] * 2**0.5)
    cleared = [f for f, (s, t) in enumerate(zip(grid[3], grid[4])) if s and not t]
    assert grid[3][:5] == ["ONE", "code[0]", "x0", "parity.1", "ands"] and cleared == [2, 3]
    assert grid[5][:5] == ["ONE", "code[0]", None, "ands.1", "ands"]  # ands.1 takes one of them
    a, quarter = "layers.3.", range(20, 30)  # what its head and its block write, by feature:
    o, down = (w[a + x + "_proj.weight"][:, 0] for x in ("self_attn.o", "mlp.down"))
    assert np.flatnonzero(o).tolist() == np.flatnonzero(down).tolist() == [2, 3, *quarter]
    assert (o[[2, 3]] == 1).all() and (down[[2, 3]] == -1).all() and (o[20:30] == beside.K).all()
    for x in "post_attention", "post_feedforward":  # where its norms drop K, and keep the rest
        assert np.flatnonzero(w[f"{a}{x}_layernorm.weight"] + 1).tolist() == [2, 3]
        assert (w[f"{a}{x}_layernorm.weight"][[2, 3]] + 1 == beside.K / 2).all()
    assert w[a + "self_attn.v_proj.weight"][0, 0] == 1 and (down[20:30] == beside.K).all()

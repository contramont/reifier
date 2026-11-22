"""Outside the compiler, on the same trace and grid: folding, tiles, Keccak and Gemma."""

import hashlib
import json
import re
import zlib
from itertools import pairwise
from pathlib import Path

import adder
import alongside
import keccak
import numpy as np
import pytest
from keccak import keccak_p, sponge

from reifier import beside, gemma
from reifier.check import check_grid, check_trace, check_transformer
from reifier.fold import fold
from reifier.grid import ONE, layout
from reifier.lib import FALL, add, ands, parity, table, xor
from reifier.relay import relay
from reifier.tiles import tiles
from reifier.trace import const, mlp, threshold_gate, trace
from reifier.transformer import (
    CONTEXT,
    DOWN,
    GATE,
    LIGHT,
    LOGIT,
    MARGIN,
    READ,
    SIZE,
    VALUE,
    runner_up,
    transformer,
)

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
    assert len(f.mlps) == 166 and f.mlps[-1].name == "next[7]"  # 192 before folding
    first_of_two = [x for x in f.mlps if x.name.endswith("'")]  # a parity of 5 is two tables
    assert len(first_of_two) == 20 and {len(x.inputs) for x in f.mlps} == {1, 2, 3}  # of few bits
    m, first, dash = transformer(f, keccak.CODES), keccak.FIRST, keccak.DASH
    tokens = first + rng.integers(0, 256, 16)
    check_transformer(m, tokens)
    assert m.generate([dash] * 3, 3)[3:] == [first + 146, first + 140, dash]  # on from "-" alone
    assert (len(m.q), len(m.blocks), len(m.grid[0])) == (3, 9, 124)  # four blocks a round, and one
    streams, logits = m.run(tokens)  # and nothing in it is large: a weight, a norm's, the stream
    layers, norms = [b.layer for b in m.blocks], [m.reads, m.writes, m.out]
    assert max(np.abs(x.wi).max() for x in layers) < 1  # of a gate or a value; and out of a unit
    assert {np.abs(x.wo).max() for x in layers} == {DOWN}
    assert max(np.abs(x).max() for x in (m.q, m.k, m.v, m.o)) == 1  # an output of a head
    assert max(x.max() for x in [*norms, *(y for b in m.blocks for y in (b.reads, b.writes))]) < 5
    assert max(np.abs(x).max() for x in streams) < 1.25 * SIZE
    assert np.allclose(logits[3:].max(1), LOGIT, atol=0.01)


G270M, RAN = "unsloth/gemma-3-270m", 2074878169  # the checksum of the weights that ran on it
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
    # once transformers has fetched Gemma's tokenizer, which has the tokens of the adder
    torch = pytest.importorskip("torch")
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM

    file = try_to_load_from_cache(G270M, "tokenizer.json", revision=gemma.REVISIONS[G270M])
    if not (isinstance(file, str) and gemma.cache_file(G270M).exists()):
        pytest.skip("needs the shapes of Gemma's tensors, and its tokenizer")
    vocab = json.loads(Path(file).read_text())["model"]["vocab"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def runs(m, tokens, out):
        """Gemma, loaded from what save writes of m in out, after it is checked on a <bos> and
        tokens: its stream after each layer is m's, and so is each next token"""
        gemma.save(m, out, G270M)
        model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).to(device)
        after = []  # the stream after each layer: output_hidden_states has the last after its norm
        hooks = [x.register_forward_hook(lambda *x: after.append(x[2])) for x in model.model.layers]
        ids = torch.tensor([[model.config.bos_token_id, *tokens]], device=device)
        with torch.no_grad():  # as transformers runs it by default, after a <bos>
            tops = model(ids).logits[0].argmax(1).cpu().numpy()
        for hook in hooks:
            hook.remove()
        after = [(x[0] if isinstance(x, tuple) else x)[0].float().cpu().numpy() for x in after]
        assert len(after) == 18
        streams, logits = m.run(tokens)
        n, width = len(m.q), len(m.grid[0])  # position n is the first that has all that m reads
        first = -(-n // model.config.num_attention_heads) - 1  # the last layer that attends
        for d in range(1, len(streams)):  # row d of the grid is what block d - 1 gives: a bit is
            off = np.abs(after[first + d - 1][n:, :width] - streams[d][n:]).max() / SIZE
            assert off <= 0.03, f"row {d} is {off:g} of a bit off"  # rounded at every sum
        last = after[first + len(m.blocks) - 1]  # the layers after m's blocks give nothing, and
        assert all((x == last).all() for x in after[first + len(m.blocks) :])  # the <bos> and
        assert not any(x[0].any() or x[:, width:].any() for x in after)  # Gemma's other features
        assert (tops[n:] == logits[n:].argmax(1)).all()  # stay zeros; and each next token is m's
        return model

    def greedy(model, tokens, n):
        """The n next tokens of model after a <bos> and tokens, each the likeliest"""
        ids = torch.tensor([[model.config.bos_token_id, *tokens]], device=device)
        for _ in range(n):
            with torch.no_grad():
                ids = torch.cat([ids, model(ids).logits[:, -1:].argmax(2)], 1)
        return ids[0, -n:].tolist()

    first, dash = keccak.FIRST, keccak.DASH  # the sponge: 3 heads, in layer 0
    m = transformer(keccak.program(), keccak.CODES)
    model = runs(m, first + np.random.default_rng(0).integers(0, 256, 24), tmp_path / "keccak")
    assert greedy(model, [dash] * 3, 3) == [first + 146, first + 140, dash]
    m = transformer(adder.program(), {vocab[c]: code for c, code in adder.CODES.items()})
    assert len(m.q) == 8  # the adder: Gemma has 4 heads a layer, so its layer 1 copies 4 codes
    model = runs(m, [vocab[c] for c in "123+246=36"], tmp_path / "adder")  # a digit is next
    assert greedy(model, [vocab[c] for c in "123+246="], 3) == [vocab[c] for c in "369"]


def next_alongside(x):  # alongside's next_token, on plain bits: 1/2 for each where it is silent
    y = sponge(x[:18] + x[144:162], 7, 200, 160, **PLAIN)[:18]  # it reads the numbers of two codes
    number = sum(bit << i for i, bit in enumerate(y))  # and gives the code of the sponge's token
    return [0.5] * 144 if y[17] else alongside.code([number])[0].tolist()


def test_keccak_alongside_gemma_is_seven_rounds_on_two_tokens_in_thirty_blocks():
    f = alongside.program()
    assert check_trace(f, next_alongside) == 256
    assert len(f.mlps) == 3292 and {x.activation for x in f.mlps} == {"reglu"}  # tables,
    assert max(len(x.inputs) for x in f.mlps) == 4  # each of 4 bits at most
    depth, tables = dict.fromkeys(f.inputs, 0), [0] * 31  # of each signal, and at each depth
    for x in f.mlps:
        d = 1 + max(depth[i] for i in x.inputs)
        depth |= dict.fromkeys(x.outputs, d)
        tables[d] += 1
    rounds = [36, 36] + [200, 200, 40, 40] * 6 + [34, 18]  # 4 layers a round: 2 of them a parity
    assert tables[1:] == [*rounds, 144, 144]  # of 5, then theta and chi; and 2 for the code: of
    assert f.outputs == [f"next[{i}]:0" for i in range(144)]  # each bit, a parity of 3, then a
    assert f.mlps[-1].outputs == ["next[143]:0", "next[143]:1"]  # table with a rest, on 2 more
    numbers = [f"bits{i}" for i in (*range(18), *range(144, 162))]  # and the top bit. Of the two
    assert [x for x in f.inputs if x in {i for t in f.mlps for i in t.inputs}] == numbers  # codes
    m = beside.front(f, alongside.CODES)  # it reads the numbers; on every token of Gemma, to be
    check_transformer(m, range(1000, 1016))  # in front of its layers
    assert len(m.blocks) == 30 and len(m.q) == 2
    features, units = len(m.grid[0]), max(len(b.layer.hidden) for b in m.blocks)
    assert features == 601 and units == 3824  # they fit: Gemma has 640, and 4096 in a wide layer
    assert m.embed.shape == (2**18, 16 + 144) and m.v.shape == (144, 601)  # a feature a bit of a
    assert [s for s in m.grid[0] if s not in (None, ONE)] == numbers  # code, and of what is read
    assert [np.count_nonzero(o) for o in m.o] == [144 + 18, 18]  # of it, which a head copies
    carried = [s for s in m.grid[-1] if s]  # the outputs, where the code was, their rests and a
    assert m.grid[-1][1:145] == f.outputs and len(carried) == 2 * 144 + 1  # ONE, each CARRY
    assert set(m.sizes[-1]) == {0, beside.CARRY} and not m.out.any()
    gates, values = zip(*(np.split(b.layer.wi, 2) for b in m.blocks))  # and no weight is over
    top = lambda xs: max(np.abs(x).max() for x in xs)  # what the constants say: 0.81 in a gate
    gate = FALL * GATE / READ  # of a table, on a bit, LIGHT of it in a value, DOWN out of a unit,
    assert np.isclose(top(gates), gate) and np.isclose(top(values), LIGHT * gate) and gate < 1
    assert {np.abs(b.layer.wo).max() for b in m.blocks} == {DOWN}  # VALUE in a key or a value of
    assert top(m.q) < top(m.k) == top(m.v) == VALUE and top(m.o) == 1  # a head, 1 out of it; and
    rms = (SIZE**2 / 2 + beside.STREAM[0] ** 2) ** 0.5  # in a norm, were m's half of the stream
    assert top([m.reads, *(b.reads for b in m.blocks)]) <= READ / SIZE * rms  # all bits, 5.7 to
    assert top([m.writes, *(b.writes for b in m.blocks)]) <= SIZE / 2**0.5  # read and 1.5 to write
    codes = np.array(list(alongside.CODES.values()), np.int8)  # of each token: its number, and
    apart = lambda a, b: (codes[a] != codes[b]).sum(-1)  # 126 checks. The bits that two differ in
    a, b = np.random.default_rng(0).integers(0, 2**18, (2, 4096))  # are 36 or more, a quarter:
    assert codes.shape == (2**18, 144) and apart(a, b)[a != b].min() >= 36  # of tokens at random,
    near = np.array([1 << i | 1 << j for i in range(18) for j in range(i + 1)])  # and of those
    assert apart(a[:256, None], a[:256, None] ^ near).min() == 36  # whose numbers differ in a bit
    # or two. And of any two: a code is the xor of that of its number's top bit and that of the
    # rest of the number, so the xor of two codes is a code, and each has 36 ones or more, but 0's
    assert all((codes[1 << k : 2 << k] == codes[: 1 << k] ^ codes[1 << k]).all() for k in range(18))
    assert codes[1:].sum(1).min() == 36


def test_gemma_with_the_sponge_beside_it_gives_what_each_gives_alone(tmp_path):
    # uv run --python 3.14 --with torch --with transformers python -m pytest -k beside
    # once examples/alongside.py has fetched Gemma's own weights, and transformers its tokenizer:
    # no other test runs beside.py's constants, STREAM, LIFTS, LOUD and the like, on the model
    torch = pytest.importorskip("torch")
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = gemma.REVISIONS[G270M]
    files = [try_to_load_from_cache(G270M, f, revision=rev) for f in WEIGHTS_AND_TOKENIZER]
    if not (all(isinstance(f, str) for f in files) and gemma.cache_file(G270M).exists()):
        pytest.skip("needs Gemma's weights and tokenizer")
    beside.save(beside.front(alongside.program(), alongside.CODES), tmp_path)
    beside.save(None, tmp_path / "alone")  # Gemma alone, as wide
    device, load = "cuda" if torch.cuda.is_available() else "cpu", AutoModelForCausalLM
    models = [load.from_pretrained(tmp_path), load.from_pretrained(tmp_path / "alone")]
    models.append(load.from_pretrained(G270M, revision=rev))  # Gemma itself: all three bf16
    assert {x.dtype for x in models} == {torch.bfloat16}
    assert [x.config.num_hidden_layers for x in models] == [53, 18, 18]
    names = json.loads((tmp_path / "names.json").read_text())  # the sponge's 30 blocks, then
    gemmas = [i for i, units in enumerate(names["hidden"]) if not units]  # Gemma's 18 layers, with
    assert gemmas == [30, 32, *range(34, 41), *range(42, 47), *range(48, 52)]  # a layer that lifts
    lifts = [len(names["hidden"][i]) for i in (31, 33, 41, 47)]  # before its layers 1, 2, 9 and
    assert lifts == [289 * 2 * 7] * 4 and len(names["features"]) == 54  # 14: twos, 7 a feature,
    assert len(names["hidden"][52]) == 145 * 2 * 7  # and one after its last, for the code and ONE
    text = AutoTokenizer.from_pretrained(G270M, revision=rev)(README.read_text())["input_ids"]
    text = text[: 1024 if device == "cuda" else 256]  # a <bos>, and the start of the README
    ids = torch.tensor([text], device=device)
    with torch.no_grad():
        both, alone, itself = (x.to(device)(ids).logits[0].float() for x in models)
    bits = np.array([alongside.CODES[t] for t in text])  # position 0 reads its token twice
    y = np.array([next_alongside([*older, *newer]) for older, newer in pairwise([bits[0], *bits])])
    speaks = torch.tensor(y[:, 0] != 0.5, device=device)
    token = torch.tensor(y[:, :18].round().astype(int) @ (1 << np.arange(18)), device=device)
    assert len(text) / 4 < speaks.sum() < 3 * len(text) / 4  # half of the time
    top, second = both.topk(2).values.T
    assert (both.argmax(1) == token)[speaks].all()  # there, the token of the sponge's code,
    assert (top - second)[speaks].min() >= 25  # which no other is near: by 43, after the <bos>;
    assert alone.max() < top[speaks].median() < top[speaks].max() < 500  # its logit is a few
    # hundred, and Gemma's a few dozen: the one thing of the sponge's that is larger than Gemma's
    same = both.argmax(1) == alone.argmax(1)  # elsewhere, the token of Gemma alone, as wide,
    assert same[~speaks].float().mean() >= 0.95  # as often as that is Gemma's own: weights that
    assert (alone.argmax(1) == itself.argmax(1)).float().mean() >= 0.95  # are rounded break ties,
    assert (both - alone).abs().amax(1)[~speaks].median() < 1  # and move no logit far
    random = torch.Generator(device).manual_seed(0)  # then every weight of the model is off by 3%
    for k, p in models[0].named_parameters():  # of itself, or so: a norm's is 1 + what is stored
        plus = float(k.endswith("norm.weight"))
        noise = 1 + 0.03 * torch.randn(p.shape, generator=random, device=device)
        p.data = ((p.data.float() + plus) * noise - plus).to(p.dtype)
    with torch.no_grad():
        noisy = models[0](ids).logits[0]
    assert (noisy.argmax(1) == token)[speaks].float().mean() >= 0.99  # the sponge's token stays


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
USED = np.r_[:8, 128:136]  # of a head of 256 dimensions, those that RoPE turns as the transformer's


def small_gemma(heads=4, model=G4, size=40, layers=4):
    """As gemma.checkpoint_metadata, for a Gemma of `size` features, 64 hidden units and `layers`
    layers, with heads of 256, and two of keys and values"""
    rows = {"q_proj": 256 * heads, "k_proj": 512, "v_proj": 512, "gate_proj": 64, "up_proj": 64}
    columns = {"o_proj": 256 * heads, "down_proj": 64}
    shapes = {P + "embed_tokens.weight": [90000, size]}  # of save's chunks, the first are zeros
    for k in ["norm.weight", *(f"layers.{i}.{k}" for i in range(layers) for k in LAYER.split())]:
        kind = k.split(".")[-2]  # a projection is rows by columns, of features if not of these
        shape = [rows.get(kind, size), columns.get(kind, size)] if "proj" in k else [size]
        shapes[P + k] = [256] if "_norm" in k else shape
    kinds = ["sliding_attention"] * layers
    return shapes, {"num_attention_heads": heads, "layer_types": kinds} | model


def beside_gemma(size=48):
    """For beside.build_weights: the tensors of a small Gemma 3, at random, and its config: of
    18 layers, as the Gemma that beside.STREAM is of"""
    shapes, config = small_gemma(4, G3, size, 18)
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
    assert names == {"features": grid, "hidden": [b.layer.hidden for b in m.blocks]}
    first = ["add/and+", "add/and-", *["erase what is left"] * 2, "ONE"]  # of the code's 2 bits
    assert names["hidden"][0] == first and config == small_gemma()[1]
    assert gemma.round_to_bf16(np.float32([1 + 2**-8, 1 + 3 * 2**-8])).tolist() == [1, 1 + 2**-6]


def test_gemma_has_the_weights_of_the_transformer_and_a_norm_with_nothing_to_do_gives_zero():
    shapes, config = small_gemma()
    m = transformer(trace(count, 4), {5: [0, 1]})  # 2 tokens of 2 bits: 2 heads, and 4 blocks
    w, grid = gemma.build_weights(m, shapes, config)
    wide = (28 / 40) ** 0.5  # a norm of Gemma's, of one of m's: its mean is over more features
    assert grid == m.grid and len(grid[0]) == 28  # ONE, the code, 15 ONEs more and the signals
    embed = w["embed_tokens.weight"]  # stored less by sqrt(size), which Gemma scales it by
    assert embed.any(1).nonzero()[0].tolist() == [5] and not embed[:, 18:].any()
    assert np.allclose(embed[5, :18] * 40**0.5, [SIZE, -SIZE, *[SIZE] * 16])
    a, own = "layers.0.self_attn.", MARGIN / (1 - runner_up(m.turns, CONTEXT))  # a token's score
    q, k, v, o = (w[f"{a}{x}_proj.weight"] for x in "qkvo")
    for h in range(2):  # m's queries, where RoPE turns as m does, each as large as the key,
        mine, large = q[256 * h + USED, :28], np.abs(m.k).max()
        assert np.allclose(mine, m.q[h] / np.abs(m.q[h]).max() * large)
        assert np.allclose(k[256 * h + USED, :28], m.k)  # its key and its value, for every head,
        assert np.allclose(v[256 * h : 256 * h + 2, :28], m.v)
        assert np.allclose(o[:28, 256 * h : 256 * h + 2], m.o[h] / 2)  # and its outputs, halved
    counts = [np.count_nonzero(x.any(axis)) for x, axis in ((q, 1), (k, 1), (v, 1), (o, 0))]
    assert counts == [8 + 16, 16, 4, 4]  # no more: head 0 looks 0 back, so half its query is 0
    q_norm, k_norm = (w[f"{a}{x}_norm.weight"] for x in "qk")  # of 256: they make a head score
    assert np.allclose(q_norm[USED], own / 256) and (k_norm[USED] == 1).all()  # its token as m
    assert q_norm.nonzero()[0].tolist() == k_norm.nonzero()[0].tolist() == USED.tolist()
    reads, writes = (w[f"layers.0.{x}_layernorm.weight"] for x in ("input", "post_attention"))
    assert np.allclose(reads[:28], m.reads * wide) and np.allclose(writes[:28], m.writes * wide)
    assert np.allclose(reads[:18], READ * (18 / 40) ** 0.5)  # what is embedded is read as READ
    for d, b in enumerate(m.blocks):  # block d is layer d's, between norms of its weights
        n, at = len(b.layer.hidden), f"layers.{d}."
        gate, up, down = (w[f"{at}mlp.{x}_proj.weight"] for x in ("gate", "up", "down"))
        assert np.allclose(np.vstack([gate[:n, :28], up[:n, :28]]), b.layer.wi)
        assert np.allclose(down[:28, :n], b.layer.wo) and not down[:, n:].any()
        assert not (gate[n:].any() or up[n:].any() or gate[:, 28:].any() or up[:, 28:].any())
        pre, post = (w[f"{at}{x}_feedforward_layernorm.weight"] for x in ("pre", "post"))
        assert np.allclose(pre, [*b.reads * wide, *[0] * 12])
        assert np.allclose(post, [*b.writes * wide, *[0] * 12])
        assert (w[at + "layer_scalar"] == 1).all()
    assert np.allclose(w["norm.weight"], [*m.out * 28**0.5, *[0] * 12])  # for m's logits
    norms = [k[len(P) :] for k in shapes if k.endswith("norm.weight")]  # every norm is set:
    idle = [k for k in norms if re.search("layers.[123].(input|post_attention|self_attn)", k)]
    assert set(norms) <= set(w) and len(idle) == 12  # one with nothing to do multiplies by 0,
    assert not any(w[k].any() for k in idle)  # as, after layer 0, those of attention do
    assert "layers.1.self_attn.q_proj.weight" not in w
    with pytest.raises(AssertionError, match="count needs 28 features, and Gemma has 27"):
        gemma.build_weights(m, shapes | {P + "norm.weight": [27]}, config)
    for slide, said in ((2, "5 layers, and Gemma has 4"), (1, "Gemma's first 2 layers to slide")):
        kinds = ["sliding_attention"] * slide + ["full_attention", "sliding_attention"]
        with pytest.raises(AssertionError, match=said):  # with a head a layer, 2 layers attend
            gemma.build_weights(m, *small_gemma(1, G4 | {"layer_types": kinds}))
    thin = shapes | {P + f"layers.{i}.mlp.gate_proj.weight": [n, 40] for i, n in enumerate((5, 12))}
    with pytest.raises(AssertionError, match="layer 1 needs 13 hidden units, and Gemma has 12"):
        gemma.build_weights(m, thin, config)  # layer 0 needs 5, and has them
    slow = {"rope_parameters": {"sliding_attention": {"rope_theta": 1e6}}}  # no 8 pairs as m's
    with pytest.raises(AssertionError, match="RoPE has no such turns"):
        gemma.build_weights(m, shapes, config | slow)


def test_gemma_3_stores_norms_less_one_and_a_later_layer_copies_what_layer_0_copied():
    m = transformer(trace(xor, 5), {5: [1]})  # 5 tokens of a bit, 2 heads a layer: 3 layers attend
    shapes, config = small_gemma(2, G3)
    w, grid = gemma.build_weights(m, shapes, config)
    w4 = gemma.build_weights(m, *small_gemma(2))[0]  # as Gemma 4, whose norms multiply by their
    assert sorted(w) == sorted(w4) and len(w) == 48  # weights: here, and here alone, each is
    for k in w:  # stored less 1; and Gemma 3 multiplies a score by 1 / 16
        stored = w[k] + k.endswith("norm.weight")
        assert np.allclose(stored, w4[k] * (16 if "q_norm" in k else 1)), k
    assert grid[0][17:22] == [f"xs{i}" for i in range(5)] and len(grid[0]) == 27
    q, v, o = ([w[f"layers.{i}.self_attn.{k}_proj.weight"] for i in range(3)] for k in "qvo")
    for i, h in enumerate((0, 2, 4)):  # layers 1 and 2 have a head each, m's heads 2 and 4: they
        assert np.allclose(q[i][USED, :27], m.q[h] / np.abs(m.q[h]).max() * np.abs(m.k).max())
        assert not q[i][256 * (1 + (i == 0)) :].any()  # look 2 and 4 back, at what layer 0 copied
    assert [x[:2].nonzero()[1].tolist() for x in v] == [[1], [21, 20], [21]]  # there, as xs4 and
    assert [x[:, :2].nonzero()[0].tolist() for x in o] == [[1, 21], [18, 19], [17]]  # xs3: here
    assert o[1][19, 0] == o[1][18, 1] == 0.5  # they are xs2 and xs1, and xs4 of 4 back is xs0
    held = [17, 18, 20]  # the features that hold a ONE or a bit, before each layer that attends
    for i, (read, wrote) in enumerate(zip(([1], [20, 21], [21]), ([1, 20, 21], [18, 19], [17]))):
        reads, writes = (w[f"layers.{i}.{x}_layernorm.weight"] for x in ("input", "post_attention"))
        at = np.flatnonzero(reads + 1)  # a key reads the ONEs, and a value the bits to copy:
        assert at.tolist() == sorted([0, *range(2, 17), *read])  # each is made READ,
        assert np.allclose(reads[at] + 1, READ * (held[i] / 40) ** 0.5)
        assert np.flatnonzero(writes + 1).tolist() == wrote  # and written as SIZE
        assert np.allclose(writes[wrote] + 1, SIZE * (len(wrote) / 40) ** 0.5)
        q_norm, k_norm = (w[f"layers.{i}.self_attn.{x}_norm.weight"] for x in "qk")
        assert np.flatnonzero(q_norm + 1).tolist() == np.flatnonzero(k_norm + 1).tolist()
        assert np.allclose(q_norm[USED] + 1, 16 * MARGIN / (1 - runner_up(m.turns, CONTEXT)) / 256)
        assert (k_norm[USED] == 0).all() and np.flatnonzero(k_norm + 1).tolist() == USED.tolist()
    assert "layers.1.mlp.gate_proj.weight" not in w  # the first block is in layer 2: before it,
    norms = [k[len(P) :] for k in shapes if k.endswith("norm.weight")]  # the norms of a block
    idle = "layers.[01].*feedforward|layers.3.(input|post_attention|self_attn)"  # have nothing
    idle = [k for k in norms if re.search(idle, k)]  # to do there, as those of attention after
    assert len(idle) == 8 and all((w[k] == -1).all() for k in idle)  # it: they multiply by 0
    for i, b in zip((2, 3), m.blocks):
        gate, n = w[f"layers.{i}.mlp.gate_proj.weight"], len(b.layer.hidden)
        assert np.allclose(gate[:n, :27], b.layer.wi[:n]) and not gate[n:].any()
        post = w[f"layers.{i}.post_feedforward_layernorm.weight"]
        assert np.allclose(post, [*b.writes * (27 / 40) ** 0.5 - 1, *[-1] * 13])
    assert np.allclose(w["norm.weight"][:3], [-1, m.out[1] * 27**0.5 - 1, -1])
    thin = shapes | {P + "layers.1.self_attn.q_norm.weight": [1]}  # a head of 1 dimension
    with pytest.raises(AssertionError, match="layer 1: a head of 1 cannot copy 2 bits"):
        gemma.build_weights(m, thin, config)
    with pytest.raises(AssertionError, match="RoPE has no such turns"):  # by the theta of layers
        gemma.build_weights(m, shapes, config | {"rope_local_base_freq": 1e6})  # that slide


def thrice(bits):  # of the last two tokens, a bit each: their xor, through three layers
    return ands([parity([parity(bits)])], [([1], 1)])


def test_beside_gemma_its_own_weights_are_on_the_last_features_of_its_own_layers(tmp_path):
    real, config = beside_gemma()
    kinds = (["sliding_attention"] * 5 + ["full_attention"]) * 3  # as those of Gemma 3 270M
    config |= {"layer_types": kinds}
    m, c = beside.front(trace(thrice, 2), BIT, 48, 128), 2**0.5
    check_transformer(m, [5, 4, 4, 5, 5, 4])
    wide, wide_shapes, w, grid, names = beside.build_weights(m, real, config)
    sizes = wide["hidden_size"], wide["intermediate_size"], wide["num_hidden_layers"]
    assert sizes == (96, 128, 26)  # twice as wide, and deeper: m's 3 blocks, then Gemma's layers,
    ours = [0, 1, 2, 4, 6, 14, 20, 25]  # with a layer of m's that lifts before its layers 1, 2, 9
    gemmas = [i for i in range(26) if i not in ours]  # and 14, and a last
    assert [i for i, units in enumerate(names) if not units] == gemmas
    lifts = ["lift f1+", "lift f1-", "lift f20+", "lift f20-"]  # units in twos, SHARES times
    assert [names[i][:4] for i in ours[3:]] == [lifts] * 5 and len(names[4]) == 4 * beside.SHARES
    assert [names[i][0] for i in (0, 1, 2)] == ["parity#0", "parity.1", "ands#0"]
    assert [wide["layer_types"][i] for i in gemmas] == kinds  # Gemma's are as they were,
    assert wide["layer_types"].count("sliding_attention") == 15 + 8  # and m's own slide
    alone = beside.build_weights(None, real, config)  # Gemma alone, as wide: its own 18 layers,
    assert alone[0] == wide | {"layer_types": kinds, "num_hidden_layers": 18}  # and nothing beside
    assert alone[1] == {k: wide_shapes[beside.moved(k, gemmas)] for k in real}
    assert alone[3:] == ([], [])
    for k, v in real.items():  # each of Gemma's: over c in a norm that reads the stream, by c in
        k, to = k[len(P) :], beside.moved(k[len(P) :], gemmas)  # what writes it, else as it was
        reads = k == "norm.weight" or "input_layernorm" in k or "pre_feedforward" in k
        want = (1 + v) / c - 1 if reads else v * c if "o_proj" in k or "down_proj" in k else v
        last = tuple(slice(-d, None) for d in v.shape)
        assert np.allclose(w[to][last], want, atol=1e-6), k
        assert np.array_equal(alone[2][k][last], w[to][last]), k  # with m and without

    def apart(w, gemmas):
        """A layer of Gemma's has no weight on m's features, the first half, nor on the first
        half of its hidden units, and one of m's own has none on Gemma's features: a norm's
        weights there multiply by 0"""
        for k, x in w.items():
            layer = re.search(r"(?<=layers\.)\d+", k)
            its = layer and int(layer[0]) in gemmas  # Gemma's own layer
            for axis, n in enumerate(x.shape if layer else []):
                if n == 96 or (n == 128 and its):
                    other = range(n // 2) if its else range(n // 2, n)
                    assert not (np.take(x, other, axis) + k.endswith("norm.weight")).any(), k

    apart(w, gemmas), apart(alone[2], range(18))
    norms = [k[len(P) :] for k in wide_shapes if k.endswith("norm.weight")]  # every norm is set,
    assert set(norms) <= set(w) and len(norms) == 1 + 26 * 6  # so one with nothing to do gives 0
    assert all((w[f"layers.{i}.input_layernorm.weight"] == -1).all() for i in ours[1:])
    embed, norm = w["embed_tokens.weight"], w["norm.weight"]  # in these, they meet: m's code is
    assert np.allclose(embed[:, :17] * 96**0.5, m.embed) and not embed[:, 17:48].any()  # embedded
    assert np.allclose(embed[[4, 5], 1] * 96**0.5, [-SIZE, SIZE])  # in a feature a bit, feature
    assert np.flatnonzero(norm[:48] + 1).tolist() == [1]  # 1 here, which the last norm hears
    assert norm[1] + 1 == beside.HEARD
    a = "layers.0."  # m's heads are layer 0's, and its blocks the first layers', as they are
    assert np.allclose(w[a + "self_attn.v_proj.weight"][:1, :21], m.v)
    assert np.allclose(w[a + "input_layernorm.weight"][:21] + 1, m.reads)
    for i, b in enumerate(m.blocks):
        gate, n = w[f"layers.{i}.mlp.gate_proj.weight"], len(b.layer.hidden)
        assert np.allclose(gate[:n, :21], b.layer.wi[:n]) and not gate[n:].any()
        assert np.allclose(w[f"layers.{i}.pre_feedforward_layernorm.weight"][:21] + 1, b.reads)
        assert np.allclose(w[f"layers.{i}.post_feedforward_layernorm.weight"][:21] + 1, b.writes)
    size, at = beside.CARRY, [1, 20]  # m's last block leaves the output, where its bit was, and
    for i, (before, larger) in zip(ours[3:7], beside.LIFTS.items()):  # a ONE: a layer lifts them,
        pre, post = (w[f"layers.{i}.{x}_feedforward_layernorm.weight"] + 1 for x in ("pre", "post"))
        assert np.flatnonzero(pre).tolist() == np.flatnonzero(post).tolist() == at  # in place:
        assert np.allclose(pre[at], beside.LIFT * beside.STREAM[before] / size)  # read as LIFT
        assert np.allclose(post[at], (larger - size) * (2 / 96) ** 0.5)  # where Gemma's stream
        size = larger  # is as large as usual
    pre, post = (w[f"layers.25.{x}_feedforward_layernorm.weight"] + 1 for x in ("pre", "post"))
    assert np.flatnonzero(pre).tolist() == at and np.flatnonzero(post).tolist() == [1]  # and the
    assert np.allclose(pre[at], beside.SPEAK * beside.STREAM[18] / size)  # last makes the output
    assert np.isclose(post[1], (beside.LOUD - size) * (2 / 96) ** 0.5)  # LOUD, in place too
    gate, down = (w[f"layers.25.mlp.{x}_proj.weight"] for x in ("gate", "down"))  # its two units
    deaf = [[1, -beside.DEAF], [-1, -beside.DEAF], [0, 1 - beside.DEAF], [0, -1]]  # for the output
    assert gate[:4, at].tolist() == deaf  # are deaf to what is under DEAF of ONE, and two more
    assert down[:3, :4].tolist() == [[0] * 4, [0.5, -0.5, 0, 0], [0, 0, 0.5, -0.5]]  # give ONE, as
    gemma.write(tmp_path, wide, wide_shapes, w, grid, names)  # large, to feature 2: the norm after
    saved = json.loads((tmp_path / "names.json").read_text())  # drops it. The names: a row of
    assert saved == {"features": grid, "hidden": names} and len(grid) == 27  # features after the
    assert grid[:4] == m.grid and grid[4:] == [m.grid[3]] * 23  # heads and after each layer, and
    assert grid[26] == [None, "ands", *[None] * 18, ONE]  # all after m's blocks leave them there


def test_beside_gemma_says_what_it_cannot_place():
    real, config = beside_gemma()
    t = trace(thrice, 2)
    for m in (transformer(t, BIT), beside.front(t, BIT, 40, 128)):  # with logits of its own, and
        with pytest.raises(AssertionError, match="thrice is not of front, for this Gemma"):
            beside.build_weights(m, real, config)  # for a Gemma of another size
    m = beside.front(t, BIT, 48, 128)  # one of front is not for gemma.py, which needs its logits,
    with pytest.raises(AssertionError, match="thrice has no logits of its own: it is for beside"):
        gemma.build_weights(m, *small_gemma())
    few = config | {"layer_types": config["layer_types"][:4]}  # nor for a Gemma of 4 layers
    with pytest.raises(AssertionError, match="STREAM is of unsloth/gemma-3-270m"):
        beside.build_weights(m, real, few)
    with pytest.raises(AssertionError, match="thrice needs a code for each of Gemma's tokens"):
        beside.build_weights(beside.front(t, {5: [1]}, 48, 128), real, config)
    with pytest.raises(AssertionError, match="xor needs its heads in one layer"):  # 5, of 4
        beside.build_weights(beside.front(trace(xor, 5), BIT, 48, 128), real, config)
    with pytest.raises(AssertionError, match="thrice needs 21 features, and has 20"):
        beside.build_weights(beside.front(t, BIT, 20, 128), *beside_gemma(20))
    with pytest.raises(AssertionError, match="layer 2 needs 168 hidden units, and Gemma has 128"):
        beside.build_weights(beside.front(t, BIT, 48, 256), real, config)  # each unit, 8 times
    two_bits = {token: [token >> 1 & 1, token & 1] for token in range(90000)}
    both = beside.front(trace(lambda x: [parity(x[:2]), parity(x[2:])], 4), two_bits, 48, 128)
    names = beside.build_weights(both, real, config)[4]  # a layer that lifts 3 features has, for
    assert {len(names[i]) for i in (2, 4, 12, 18, 23)} == {3 * 2 * 21}  # each, SHARES twos, if
    early = lambda x: [parity(x[:2]), parity([parity(x[2:])])]  # as many fit: here, 21 do
    with pytest.raises(AssertionError, match="lambda: its last block must make every output"):
        beside.front(trace(early, 4), two_bits, 48, 128)  # and block 0 would make the first


def half(bits):  # of the last two tokens, a bit each: the newer's, and 1/2 after an older that is 1
    return table(bits, lambda case: 1 / 2 if case[0] else case[1], "half")


def test_beside_gemma_what_it_carries_is_lifted_whatever_gemma_holds_and_spoken_if_a_code():
    config = beside_gemma()[1]
    m = beside.front(trace(half, 2), BIT, 48, 256)
    tokens = [5, 4, 4, 5, 5, 4, 5]  # 6 pairs of tokens: after an older one of bit 1, it is silent
    older, newer = (np.array(tokens[i:][:6]) & 1 for i in (0, 1))
    silent, bit = older == 1, 2 * newer - 1
    check_transformer(m, tokens)
    left, at = m.run(tokens)[0][-1][2:], [1, 19, 20]  # what its own layers leave, after each pair:
    assert [m.grid[-1][f] for f in at] == ["half:0", "half:1", ONE] and len(m.grid[-1]) == 21
    carry = beside.CARRY  # the output, where its bit was embedded, its rest and a ONE: as large,
    want = np.where(silent[:, None], [0, 1, 1], np.c_[bit, 0 * bit, bit**2])  # whatever it has to
    assert np.allclose(left[:, at], carry * want, atol=0.01)  # say, and nothing else
    assert (want**2).sum(1).tolist() == [2] * 6 and np.abs(np.delete(left, at, 1)).max() < 0.01
    lifts, speak = beside.plan(m, config)
    for scale in (0.2, 1, 5):  # Gemma's stream, as large as usual, and 5 times less and more
        x, usual = left, 96 * scale**2 * np.array(beside.STREAM) ** 2  # the sum of its squares
        for i, larger in beside.LIFTS.items():  # before Gemma's layer i, each is made larger, in
            x = x + lifts[i](x, usual[i] - (x * x).sum(1, keepdims=True), 96)  # place: to a part
            assert np.allclose(x[:, at], larger * want, atol=0.005 * larger)  # in 200, under a
            assert np.abs(np.delete(x, at, 1)).max() < 0.01  # stream so large that eps is felt
        heard = lambda x, total=usual[-1]: speak(x, total - (x * x).sum(1, keepdims=True), 96)
        spoken = heard(x)  # after Gemma's last layer: the output is made LOUD, in place, if it is
        loud = (beside.LOUD - larger) * want[:, 0]  # a bit, and nothing is given elsewhere, nor
        assert np.allclose(spoken[:, 1], loud, atol=0.1 * beside.LOUD)  # where it is silent
        assert not np.delete(spoken, 1, 1).any()
        assert np.abs(spoken[silent]).max() < 0.001 * beside.LOUD
        if scale <= 1:  # where gelu is relu at its gates, under a stream no larger than usual,
            x[:, 19] = larger  # the layer is deaf to an output whose rest is there, were it a bit,
            assert np.abs(heard(x)).max() < 0.01 * beside.LOUD
            x[:, [1, 19]] = 0.4 * larger, 0  # and to one that is under DEAF of a bit, but for a
            assert np.abs(heard(x)).max() < 0.1 * beside.LOUD  # tenth of what a bit is given

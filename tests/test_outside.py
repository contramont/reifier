"""Outside the compiler, on the same trace and grid: folding, tiles, Keccak and Gemma."""

import hashlib
import json
import zlib

import adder
import keccak
import numpy as np
import pytest
from keccak import BYTES, keccak_p, sponge

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
    bits = [b for byte in msg for b in BYTES[byte]] + [1, 1, 1, 1]
    digest = [b for byte in hashlib.shake_128(msg).digest(8) for b in BYTES[byte]]
    assert sponge(bits, 24, 1600, 1344, **PLAIN)[:64] == digest


def test_the_sponge_goes_through_every_stage():
    f = keccak.program()  # next_byte, traced and folded: its padding is constants
    rng = np.random.default_rng(0)
    assert check_trace(f, lambda x: sponge(x, **PLAIN)[:8], rng.integers(0, 2, (16, 64)))
    assert len(f.mlps) == 4229 and f.mlps[80].name == "round14/theta[1,0,0]"  # 4581 before folding
    assert f([b for byte in range(8) for b in BYTES[byte]]) == BYTES[0xFA]  # as one written apart
    check_transformer(transformer(f, BYTES), rng.integers(0, 256, 16))


IDS = "236771 236770 236778 236800 236812 236810 236825 236832 236828 236819 236862 236784"
RAN = {"adder": 1367782342, "keccak": 4060565561}  # checksums of the weights that ran on Gemma


@pytest.mark.skipif(not gemma.CACHE_FILE.exists(), reason="needs the shapes of Gemma's tensors")
@pytest.mark.parametrize("name", RAN)
def test_gemma_weights_are_those_that_ran_on_the_model(name):
    program, codes = keccak.program, keccak.TOKENS
    if name == "adder":  # IDS: the tokens of the characters of SYMBOLS, as the example finds them
        program, codes = adder.program, dict(zip(map(int, IDS.split()), adder.CODES.values()))
    w = gemma.build_weights(program(), codes, *gemma.checkpoint_metadata())[0]
    checksum = 0
    for k in sorted(w):  # each tensor that is set: its name, the rows not zero, their values
        rows = np.flatnonzero(w[k].reshape(len(w[k]), -1).any(1))
        data = np.ascontiguousarray(w[k][rows])
        checksum = zlib.crc32(data, zlib.crc32(k.encode() + rows.tobytes(), checksum))
    assert checksum == RAN[name], "the weights changed: run the checkpoint on the model again"


P = gemma.PREFIX
LAYER = (  # the tensors of a layer of Gemma that gemma.py sets: those of self_attn, in layer 0
    "layer_scalar input_layernorm.weight post_attention_layernorm.weight "
    "pre_feedforward_layernorm.weight post_feedforward_layernorm.weight mlp.gate_proj.weight "
    "mlp.up_proj.weight mlp.down_proj.weight self_attn.q_norm.weight self_attn.k_norm.weight "
    "self_attn.q_proj.weight self_attn.k_proj.weight self_attn.v_proj.weight "
    "self_attn.o_proj.weight"
)


def small_gemma():
    """As gemma.checkpoint_metadata, for a Gemma of 19 features, heads of 16 and 4 layers"""
    names = ["norm.weight", *(f"layers.{i}.{k}" for i in range(4) for k in LAYER.split())]
    shapes = {P + "embed_tokens.weight": [90000, 19]}  # two of save's chunks: the first is zeros
    shapes |= {P + k: [64, 64] if "proj" in k else [16] if "_norm" in k else [19] for k in names}
    return shapes, {"rope_parameters": {"sliding_attention": {"rope_theta": 10000.0}}}


def test_save_writes_every_tensor_as_bf16_where_its_header_says(tmp_path, monkeypatch):
    monkeypatch.setattr(gemma, "checkpoint_metadata", small_gemma)
    t = trace(lambda bits: add(bits[:2], bits[2:]), 4)
    codes = {85000 + n: [n >> 1, n & 1] for n in range(4)}  # late in the second chunk
    for _ in range(2):  # into a directory that is not there, then into one that is
        gemma.save(t, codes, tmp_path / "a/b")
    w, grid, hidden = gemma.build_weights(t, codes, *small_gemma())
    file = (tmp_path / "a/b/model.safetensors").read_bytes()
    n = int.from_bytes(file[:8], "little")
    header, end = json.loads(file[8 : 8 + n]), 0
    assert n % 8 == 0 and header.pop("__metadata__") == {"format": "pt"}
    for k, shape in small_gemma()[0].items():  # each tensor after the one before it, rounded
        start, end = end, end + 2 * int(np.prod(shape))
        assert header.pop(k) == {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
        saved = np.frombuffer(file, "<u2", (end - start) // 2, 8 + n + start)
        assert (saved == gemma.round_to_bf16(w[k]).view(np.uint32).ravel() >> 16).all(), k
    assert file[7 + n : 8 + n] == b" " and len(file) == 8 + n + end  # the header was padded
    names = json.loads((tmp_path / "a/b/names.json").read_text())
    assert names == {"features": grid, "hidden": hidden} and hidden[0][0] == "ONE"
    assert gemma.round_to_bf16(np.float32([1 + 2**-8, 1 + 3 * 2**-8])).tolist() == [1, 1 + 2**-6]


def test_gemma_pads_what_bf16_cannot_norm_and_refuses_what_it_cannot_hold():
    shapes, config = small_gemma()
    t = trace(lambda bits: add(bits[:2], bits[2:]), 4)
    w, grid, _ = gemma.build_weights(t, {5: [0, 1]}, shapes, config)
    assert grid[0][7:] == [None, None, "pad[0]", "pad[1]"]  # the norms see 4, 7 and 9 signals
    assert not gemma.exact(7, 19) and not gemma.exact(35, 2560)  # the last by the norm's eps alone
    o = w[P + "layers.0.self_attn.o_proj.weight"]  # head 0 writes its first bit into the pads too
    assert o[9:].nonzero()[1].tolist() == [0, 0] and np.allclose(o[o != 0], (2 / 16) ** 0.5)
    post = w[P + "layers.0.post_attention_layernorm.weight"][:11]  # and its norm sees 4 + 2 of ±1
    assert np.allclose(post * (19 / 6) ** 0.5, [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    assert np.allclose(w[P + "layers.0.pre_feedforward_layernorm.weight"].max(), (9 / 19) ** 0.5)
    assert np.allclose(w[P + "embed_tokens.weight"][5, :4] * 19**0.5, [1, -1, 1, 0])  # ONE, 01
    assert np.allclose(w[P + "norm.weight"][:3], [0, 4 * 11**0.5, 4 * 11**0.5])  # 8 / 2 sqrt(11)
    with pytest.raises(AssertionError, match="needs 9 features, and Gemma has 8"):
        gemma.build_weights(t, {}, {P + "norm.weight": [8]}, config)
    for gate in (  # a gate that sums up to 4176, and weights in and out that bf16 rounds
        lambda x: threshold_gate(x, [33, -1], 1),
        lambda x: threshold_gate(x, [1 / 3, 1], 1),
        lambda x: mlp(x, [1, 1], [-1], "step", [1001])[0],
    ):
        with pytest.raises(AssertionError, match="block 0: a gate's weights or its sums are not"):
            gemma.build_weights(trace(gate, 2), {0: [0], 1: [1]}, shapes, config)

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reifier import (
    Architecture,
    add,
    and_,
    at_least,
    bits,
    const,
    flatten,
    fold_constants,
    glu,
    lookback,
    mlp,
    network,
    not_,
    placeholders,
    threshold_gate,
    trace,
    transformer,
    xor,
)
from reifier.attention import Attention
from reifier.compiler import simulate
from reifier.decoder import BOS, ONE, TOK, Weights, forward, predict, transformer_schedule

SMALL = Architecture(  # a small Gemma 4: 8 layers of 8 heads in 2 kv groups
    hidden_size=128,
    intermediate_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=32,
    layer_types=("sliding_attention",) * 8,
    sliding_window=64,
    rope_theta=10000.0,
    vocab_size=16 + 2**4,
    bos_token_id=2,
    token_offset=16,
    final_logit_softcapping=30.0,
)


def like(arch, **changes):
    return replace(arch, **changes)


def as_bits(n, b=4):
    return [n >> i & 1 for i in range(b)]


def as_int(bs):
    return sum(round(v) << i for i, v in enumerate(bs))


def assert_compiles(program, arch=SMALL, n=40, seed=0):
    """At every position, the schedule's stream gives the program's outputs, and so does the
    compiled program's greedy next token"""
    g = trace(program, bits(4))
    rng = random.Random(seed)
    values = [rng.randrange(16) for _ in range(n)]
    x = np.array([as_bits(v) for v in values])
    want = g.eval_sequence(x)
    s = transformer_schedule(g, arch)
    np.testing.assert_array_equal(simulate(s, x, arch.hidden_size), want)
    assert predict(transformer(g, arch), values) == [as_int(y) for y in want]
    return s


def xor_of_last_three(x):
    p1, p2 = lookback(x, 1), lookback(x, 2)
    return [xor([x[i], p1[i], p2[i]]) for i in range(4)]


def test_lookback_reads_earlier_positions():
    g = trace(lambda x: [*lookback(x, 2), *lookback(x, 0), *lookback([], 3)], bits(2))
    out = g.eval_sequence(["10", "01", "11", "00"])
    assert out.tolist() == [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 1], [0, 1, 0, 0]]
    assert g.eval("11") == [0, 0, 1, 1]  # one position: nothing before it
    with pytest.raises(ValueError):
        lookback(bits(1), -1)


def test_sequences_of_independent_positions_are_truth_tables():
    g = add.trace(bits(3), bits(3))
    rows = [as_bits(n, 6) for n in range(64)]
    assert g.eval_sequence(rows).tolist() == [g.eval(r) for r in rows]


def test_folding_looks_through_heads():
    def program(x):  # a constant is the same at every position, so a head of one is too
        not_x = threshold_gate([x[0]], [-1], 0)  # 1 - x, to folding
        a = xor([x[0], *lookback(x, 1), const("1")[0]])
        return [*lookback(const("1"), 1), a, *lookback([not_x], 2)]

    g = trace(program, bits(1))
    rows = ["1", "0", "0", "1", "1"]
    for activation in ("step", "relu", "reglu"):
        h = fold_constants(g, activation)
        np.testing.assert_array_equal(h.eval_sequence(rows), g.eval_sequence(rows))
        f = flatten(h)
        heads = [n.inputs for n in f.nodes if isinstance(n.circuit, Attention)]
        assert heads == [tuple(f.inputs)] * 2  # x, 1 and 2 back: the head of not x reads x


def test_wide_heads_are_split_across_kv_groups():
    def program(x):  # 40 values, but a kv group holds 32
        a = [xor([x[i % 4], x[(i + 1) % 4], x[(i + 2) % 4]]) for i in range(40)]
        a[::3] = [x[i % 4] for i in range(0, 40, 3)]  # some values twice
        b = lookback(a, 1)
        return [xor(b[i::4]) for i in range(4)]

    s = assert_compiles(program)
    assert [len(outs) for _, _, outs in s.layers[1].heads] == [32, 8]


def test_plain_networks_have_no_attention():
    with pytest.raises(TypeError):
        network(trace(lambda x: lookback(x, 1), bits(2)), "reglu")


def test_heads_and_mlps_compile_into_one_transformer():
    def program(x):  # a head after MLPs, and MLPs after it
        a = xor_of_last_three(x)
        q = lookback(a, 3)
        return [and_([a[i], q[(i + 1) % 4]]) if i % 2 else xor([a[i], q[i]]) for i in range(4)]

    assert_compiles(program)


def test_each_leaf_goes_after_its_inputs():
    def program(x):
        a = xor_of_last_three(x)
        return [xor([v, w]) for v, w in zip(a, lookback(a, 3))]

    s = assert_compiles(program)
    # heads looking 1 and 2 back, the xors, then a head looking 3 back and the last xors
    assert [(len(layer.heads), len(layer.mlps)) for layer in s.layers] == [(2, 4), (1, 4)]


def test_kv_groups_limit_the_heads_of_a_layer():
    def program(x):  # 2 kv groups of 4 query heads: the head of a ninth offset waits a layer
        backs = [lookback(x, d) for d in range(1, 10)]
        return [xor([b[i] for b in backs[i::4]]) for i in range(4)]

    s = assert_compiles(program)
    assert [len(layer.heads) for layer in s.layers][:2] == [8, 1]


def test_heads_of_one_offset_share_a_query_head():
    def program(x):  # 4 heads, each of one bit, looking 1 or 2 back: 2 query heads
        backs = [lookback([x[i]], 1 + i % 2)[0] for i in range(4)]
        return [xor([b, x[i]]) for i, b in enumerate(backs)]

    s = assert_compiles(program, like(SMALL, num_attention_heads=2))  # one query head a group
    assert len(s.layers[0].heads) == 4 and len(s.layers) == 1


@pytest.mark.parametrize(
    "program",
    [
        lambda x: x,  # each output is its input
        lambda x: x[::-1],  # in another output's place
        lambda x: [x[0], x[0], x[1], x[1]],  # an output twice
        lambda x: lookback(x, 1),  # outputs from a head
        lambda x: [xor([x[0], x[1]]), x[0], x[1], const("1")[0]],
    ],
)
def test_outputs_are_copied_into_their_features(program):
    assert_compiles(program)


def test_heads_erase_the_features_they_write_into():
    def program(x):
        a = xor_of_last_three(x)
        b = [xor([v, w]) for v, w in zip(a, lookback(a, 1))]
        return [xor([v, w]) for v, w in zip(b, lookback(b, 2))]

    s = assert_compiles(program, like(SMALL, hidden_size=20))  # few features, used again
    assert any(layer.erase for layer in s.layers)


def test_heads_go_in_sliding_layers_with_their_own_keys():
    arch = like(
        SMALL,
        layer_types=("sliding_attention", "full_attention") * 4,
        num_kv_shared_layers=4,
        global_head_dim=64,
        global_rope={"rope_theta": 1e6, "partial_rotary_factor": 0.25},
    )
    assert arch.attention_layers == {0, 2}

    def program(x):
        for d in (1, 2):
            x = [xor([v, w]) for v, w in zip(x, lookback(x, d))]
        return x

    s = assert_compiles(program, arch)
    assert [i for i, layer in enumerate(s.layers) if layer.heads] == [0, 2]


def test_programs_that_do_not_fit_are_refused():
    one_layer = like(SMALL, num_hidden_layers=1, layer_types=SMALL.layer_types[:1])
    with pytest.raises(ValueError, match="does not fit"):
        transformer(trace(lambda x: lookback(xor_of_last_three(x), 1), bits(4)), one_layer)
    with pytest.raises(ValueError, match="sliding window"):
        transformer(trace(lambda x: lookback(x, 64), bits(4)), SMALL)
    with pytest.raises(ValueError, match="features"):
        transformer(trace(xor_of_last_three, bits(4)), like(SMALL, hidden_size=12))
    with pytest.raises(TypeError):  # an int inside
        transformer(trace(lambda x: [*x[:3], at_least(x, 2)], bits(4)), SMALL)


def test_the_embedding_carries_one_and_the_bits_twice():
    g = trace(lambda x: x, bits(4))
    s, emb = transformer_schedule(g, SMALL), transformer(g, SMALL)["embed_tokens.weight"] * 32
    token, bos = emb[SMALL.token_offset + 0b0101], emb[SMALL.bos_token_id]
    assert token[[ONE, TOK, BOS]].tolist() == [1, 1, 0] and bos[[ONE, TOK, BOS]].tolist() == [
        1,
        0,
        1,
    ]
    assert token[s.inputs].tolist() == token[s.outputs].tolist() == [1, -1, 1, -1]
    assert bos[s.inputs].tolist() == bos[s.outputs].tolist() == [-1] * 4  # a token of zeros
    assert (token != 0).sum() == (bos != 0).sum() == 2 + 8  # as long


def test_gemma_save_writes_the_weights(tmp_path):
    from reifier import gemma

    config = {
        "hidden_size": 128, "intermediate_size": 512, "num_hidden_layers": 8,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
        "layer_types": list(SMALL.layer_types), "sliding_window": 64,
        "rope_parameters": {"sliding_attention": {"rope_theta": 1e4}},
        "vocab_size": 0x10000 + 16, "bos_token_id": 2, "final_logit_softcapping": 30.0,
    }  # fmt: skip
    arch = Architecture.from_config(config)
    shapes = {gemma.PREFIX + k: list(s) for k, s in arch.shapes().items()}
    shapes["model.vision_tower.unused"] = [3, 5]
    (tmp_path / "meta.json").write_text(json.dumps([shapes, config]))
    program = trace(xor_of_last_three, bits(4))
    gemma.save(program, tmp_path / "out", tmp_path / "meta.json")
    raw = (tmp_path / "out" / "model.safetensors").read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    for k, v in transformer(program, arch).items():
        start, end = header[gemma.PREFIX + k]["data_offsets"]
        stored = np.frombuffer(raw[8 + n + start : 8 + n + end], np.uint16).astype(np.uint32)
        bf16 = (stored << 16).view(np.float32).reshape(v.shape)
        assert np.array_equal(bf16, gemma.round_to_bf16(v)), k
    assert header["model.vision_tower.unused"]["shape"] == [3, 5]
    assert len(raw) == 8 + n + header["model.vision_tower.unused"]["data_offsets"][1]


def test_forward_runs_weights_as_transformers_does():
    data = np.load(Path(__file__).parent / "data" / "gemma4_tiny.npz")  # see make_gemma4_fixture
    w = Weights(Architecture.from_config(json.loads(str(data["config"]))))
    w.update({k.removeprefix("w:"): data[k] for k in data.files if k.startswith("w:")})
    np.testing.assert_allclose(forward(w, data["ids"]), data["logits"], atol=1e-4)


def test_heads_work_for_any_head_dim_or_say_why_not():
    for head_dim in (8, 12, 16, 18):  # 3 to 8 fast pairs, and a slow pair apart from them
        assert_compiles(xor_of_last_three, like(SMALL, head_dim=head_dim))
    for head_dim in (2, 4):  # no pair, or one that turns too far in the window
        with pytest.raises(ValueError, match="head_dim"):
            transformer(trace(xor_of_last_three, bits(4)), like(SMALL, head_dim=head_dim))


@pytest.mark.parametrize("theta, window, back", [(100.0, 64, 40), (100.0, 64, 60), (1e4, 512, 500)])
def test_heads_find_bos_where_its_pair_turns_too(theta, window, back):
    arch = like(SMALL, rope_theta=theta, sliding_window=window)
    g = trace(lambda x: [xor([v, w]) for v, w in zip(x, lookback(x, back))], bits(4))
    try:
        w = transformer(g, arch)
    except ValueError as e:  # refused rather than wrong
        assert "reliably" in str(e)
        return
    values = [random.Random(back).randrange(16) for _ in range(back + 8)]
    want = [as_int(y) for y in g.eval_sequence([as_bits(v) for v in values])]
    assert predict(w, values) == want


def test_deep_programs_are_refused_before_weights():
    def program(x):  # 3 MLPs deep, then a copy: 1 layer is not enough
        return [xor([xor([x[0], x[1]]), x[2]]), *x[1:]]

    with pytest.raises(ValueError, match="does not fit"):
        transformer(
            trace(program, bits(4)),
            like(SMALL, num_hidden_layers=1, layer_types=("sliding_attention",)),
        )


def test_erasing_takes_room_in_the_mlps():
    def program(x):  # the second head finds no empty feature, and layer 0's MLPs are full
        a = [xor([x[0], x[1]]), xor([x[2], x[3]])]
        b = [xor([v, w]) for v, w in zip(a, lookback(a, 1))]
        return [*b, *(xor([v, w]) for v, w in zip(b, lookback(b, 2)))]

    with pytest.raises(ValueError, match="layer 1 needs 7 hidden units"):
        transformer(trace(program, bits(4)), like(SMALL, hidden_size=13, intermediate_size=6))
    s = assert_compiles(program, like(SMALL, hidden_size=13, intermediate_size=7))
    assert [len(layer.erase) for layer in s.layers] == [0, 2, 0]


def test_what_cannot_fit_anywhere_says_so():
    wide = trace(lambda x: [glu(x, [([1] * 4, 0, [1] * 4, 0)] * 600), *x[1:]], bits(4))
    with pytest.raises(ValueError, match="no sublayer"):
        transformer(wide, SMALL)


def test_only_mlps_of_geglu_compile():
    g = trace(
        lambda x: [mlp([x[0]], [[1], [1]], [0, 0], "swiglu", [[1]], "bit")[0], *x[1:]], bits(4)
    )
    with pytest.raises(TypeError, match="geglu"):
        transformer(g, SMALL)


def test_bos_is_not_a_token():
    with pytest.raises(ValueError, match="bos"):
        transformer(trace(xor_of_last_three, bits(4)), like(SMALL, token_offset=0))


def test_a_token_beats_bos_even_at_bos():
    w = transformer(trace(lambda x: [not_(v) for v in lookback(x, 1)], bits(4)), SMALL)
    assert forward(w, [SMALL.bos_token_id])[0].argmax() == SMALL.token_offset + 0b1111


def test_simulate_sees_a_missing_erase():
    def program(x):
        a = [xor([x[0], x[1]]), xor([x[2], x[3]])]
        b = [xor([v, w]) for v, w in zip(a, lookback(a, 1))]
        return [*b, *(xor([v, w]) for v, w in zip(b, lookback(b, 2)))]

    s = transformer_schedule(trace(program, bits(4)), like(SMALL, hidden_size=13))
    s.layers[1].erase.pop()  # though the feature holds bits of 0 here
    with pytest.raises(AssertionError, match="writes over"):
        simulate(s, [[0] * 4] * 3, 13)


def test_heads_copy_only_what_is_read():
    g = trace(lambda x: [xor([x[0], *lookback(x, 1)[:1]]), *x[1:]], bits(4))  # bit 0, 1 back
    s = transformer_schedule(g, SMALL)
    assert [len(ins) for _, ins, _ in s.layers[0].heads] == [1]


def test_heads_of_heads_copy_only_what_is_read():
    g = trace(lambda x: [xor([x[0], lookback(lookback(x, 1), 1)[0]]), *x[1:]], bits(4))
    s = transformer_schedule(g, SMALL)
    assert [len(ins) for layer in s.layers for _, ins, _ in layer.heads] == [1, 1]


def test_programs_on_other_than_bits_are_refused():
    with pytest.raises(TypeError, match="int"):
        transformer(trace(lambda x: x, placeholders("int", "bit", "bit", "bit")), SMALL)
    norm = network(trace(lambda x: x, bits(3)), "geglu", pre_norm=True)  # a leaf of its own
    with pytest.raises(TypeError, match="neither a head nor an MLP"):
        transformer(trace(lambda x: norm(*x), bits(4)), SMALL)


def test_architectures_that_cannot_work_are_refused():
    with pytest.raises(ValueError, match="vocabulary"):
        transformer(trace(xor_of_last_three, bits(4)), like(SMALL, bos_token_id=99))
    with pytest.raises(ValueError, match="share keys"):  # the shared full layer has no own one
        like(SMALL, layer_types=SMALL.layer_types[1:] + ("full_attention",), num_kv_shared_layers=1)
    sharp = like(SMALL, head_dim=8, rope_theta=1e6, sliding_window=1024)  # scores too big for bf16
    with pytest.raises(ValueError, match="reliably"):
        transformer(trace(xor_of_last_three, bits(4)), sharp)

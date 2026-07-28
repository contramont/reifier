import hashlib

import numpy as np

from reifier.neurons.core import const, gate
from reifier.neurons.operations import xor, add, not_
from reifier.examples.keccak import Keccak, theta_rho_pi_chi, state_to_lanes, xof
from reifier.examples.keccak_template import build_round_template, _traced_key_check
from reifier.fast import fast_compile, level_graph, flatten_bits
from reifier.fast.stamp import Stamped, _BIT_LEAF
from reifier.utils.format import Bits


def test_fast_compile_xor():
    """Plain fast_compile on a small non-Keccak circuit."""
    inputs = const("00000")
    arrays = fast_compile(xor, inputs)
    for value in range(32):
        bits = format(value, "05b")
        expected = str(sum(int(b) for b in bits) % 2)
        got = "".join(map(str, arrays.run([int(b) for b in bits])))
        assert got == expected, f"{bits}: {got} != {expected}"


def test_fast_compile_add():
    """Adder circuit: compiled once, evaluated on many inputs."""
    a, b = const("0000"), const("0000")
    arrays = fast_compile(add, a, b)
    for x, y in [(0, 0), (3, 5), (7, 8), (15, 15), (9, 6)]:
        xs, ys = format(x, "04b"), format(y, "04b")
        out = arrays.run([int(v) for v in xs + ys])
        got = int("".join(map(str, out)), 2)
        assert got == (x + y) % 16, f"{x}+{y}: got {got}"


def test_fast_compile_passthrough_output():
    """Outputs that are inputs get copy chains to the final level."""

    def f(x: list) -> list:
        return [x[1], xor(x)]

    inputs = const("0110")
    arrays = fast_compile(f, inputs)
    for value in range(16):
        bits = [int(b) for b in format(value, "04b")]
        out = arrays.run(bits)
        assert out[0] == bits[1]
        assert out[1] == sum(bits) % 2


def test_fast_compile_edge_cases():
    """Duplicate parent edges, derived-const chains, const outputs."""
    # same signal appearing twice in a gate's incoming
    a, b = const("01")
    g = gate([a, a, b], [1, 1, 1], 2)  # 2*a + b >= 2
    arrays = level_graph([a, b], [g])
    for va, vb in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert int(arrays.run([va, vb])[0]) == (1 if 2 * va + vb >= 2 else 0)

    # gates fed only by constants fold into consumer biases, even when the
    # intermediate folded bias falls outside {-1, 0}
    c = const("1")[0]
    d = not_(c)
    e = gate([c, d], [3, 2], 2)  # folded bias +1
    x = const("0")[0]
    live = gate([x, e], [1, 1], 2)  # equals x since e == 1
    arrays = level_graph([x], [live])
    assert int(arrays.run([0])[0]) == 0
    assert int(arrays.run([1])[0]) == 1

    # constant bits as direct outputs are materialized as bias-only gates
    f = const("1")[0]
    y = const("0")[0]
    arrays = level_graph([y], [f, xor([y, f])])
    assert list(arrays.run([0])) == [1, 1]
    assert list(arrays.run([1])) == [1, 0]


def test_fast_leveler_matches_keccak_digest():
    """Compile on a dummy message, evaluate on real ones (no hardcoding)."""
    k = Keccak(log_w=1, n=3, c=20, pad_char="_")
    msg1 = k.format("Rachmaninoff", clip=True)
    msg2 = k.format("Reify semantics", clip=True)
    dummy = Bits("0" * len(msg1))
    arrays = fast_compile(k.digest, dummy)
    for msg in (msg1, msg2):
        got = "".join(map(str, arrays.run(msg.ints)))
        assert got == k.digest(msg).bitstr


def test_stamped_digest_matches_plain():
    """Keccak(stamp=True) computes the same digests as the plain version."""
    for log_w, n, c in [(0, 3, 10), (1, 3, 20), (6, 2, 448)]:
        k = Keccak(log_w=log_w, n=n, c=c, pad_char="_")
        ks = Keccak(log_w=log_w, n=n, c=c, pad_char="_", stamp=True)
        for phrase in ("Rachmaninoff", "Reify semantics"):
            msg = k.format(phrase, clip=True)
            assert ks.digest(msg).bitstr == k.digest(msg).bitstr


def test_stamped_xof_matches_plain():
    """xof reuses templates across repeated hash_state calls."""
    k = Keccak(log_w=1, n=3, c=20, pad_char="_")
    ks = Keccak(log_w=1, n=3, c=20, pad_char="_", stamp=True)
    msg = k.format("Rachmaninoff", clip=True)
    plain = xof(msg.bitlist, depth=3, k=k)
    fast = xof(msg.bitlist, depth=3, k=ks)
    for d_plain, d_fast in zip(plain, fast):
        assert Bits(d_plain).bitstr == Bits(d_fast).bitstr


def test_stamped_compile_full_sha3():
    """Full SHA3-224: stamped compile, then evaluate against hashlib."""
    ks = Keccak(log_w=6, n=24, c=448, pad_char="_", stamp=True)
    dummy = Bits("0" * ks.msg_len)
    arrays = fast_compile(ks.digest, dummy)
    for phrase in ("Rachmaninoff", "Reify semantics as referentless embeddings"):
        padded = (phrase + "_" * 143)[:143]
        msg = ks.format(padded)
        got = Bits([int(v) for v in arrays.run(msg.ints)]).hex
        assert got == hashlib.sha3_224(padded.encode()).hexdigest()


def test_generated_template_matches_traced():
    """The numpy-built round template equals the traced one exactly."""
    for w in (2, 8):
        lanes = state_to_lanes(const("0" * (25 * w)))
        traced, _ = Stamped(theta_rho_pi_chi)._trace((lanes,), {})
        gen = build_round_template(w)
        assert _traced_key_check(w)
        assert traced.arrays.level_sizes == gen.arrays.level_sizes
        assert traced.arg_positions == gen.arg_positions
        assert traced.out_kinds == gen.out_kinds
        for lv in range(traced.arrays.n_gate_levels):
            for name in ("edge_rows", "edge_cols", "edge_weights", "biases"):
                a = getattr(traced.arrays, name)[lv]
                b = getattr(gen.arrays, name)[lv]
                assert np.array_equal(a, b), f"w={w} level {lv + 1} {name}"

        def shape(s):
            if isinstance(s, list):
                return [shape(el) for el in s]
            return "bit" if s is _BIT_LEAF else s

        assert shape(traced.skeleton) == shape(gen.skeleton)


def test_fast_arrays_into_mlp():
    """LeveledArrays plug into the existing Matrices/MLP stack."""
    from reifier.tensors.matrices import Matrices
    from reifier.tensors.step import MLP_Step
    from reifier.tensors.mlp_utils import infer_bits_bos

    k = Keccak(log_w=1, n=3, c=20, pad_char="_")
    ks = Keccak(log_w=1, n=3, c=20, pad_char="_", stamp=True)
    msg = k.format("Rachmaninoff", clip=True)
    dummy = Bits("0" * len(msg))
    arrays = fast_compile(ks.digest, dummy)
    matrices = Matrices.from_graph(arrays.to_leveled_graph())
    mlp = MLP_Step.from_matrices(matrices)
    out = infer_bits_bos(mlp, msg)
    assert out.bitstr == k.digest(msg).bitstr


def test_stamped_alias_patterns():
    """Aliased Bit arguments must not share templates with distinct ones."""
    from reifier.fast.stamp import Stamped

    def g(pair):
        return [xor(pair)]

    s = Stamped(g)
    a, _ = const("00")
    assert not s([a, a])[0].activation  # traces the aliased pattern
    c, d = const("01")
    out = s([c, d])  # distinct pattern must trace separately
    assert out[0].activation  # xor(0, 1) == 1
    arrays = level_graph([c, d], out)
    for vc, vd in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert int(arrays.run([vc, vd])[0]) == (vc + vd) % 2
    e, _ = const("11")
    assert not s([e, e])[0].activation  # aliased template reused: xor(1, 1) == 0

    # passthrough outputs must also respect the alias pattern
    def m(pair):
        return [pair[1], xor(pair)]

    sm = Stamped(m)
    u, _ = const("00")
    sm([u, u])
    v0, v1 = const("01")
    out = sm([v0, v1])
    assert out[0] is v1


def test_two_stamped_templates_in_one_graph():
    """Two distinct templates, traced on first use (no priming), interleaved."""
    from reifier.fast.stamp import Stamped

    s_xor = Stamped(lambda xs: [xor(xs)])
    s_pair = Stamped(lambda xs: [xor(xs), not_(xs[0])])

    def f(bits):
        mids = s_xor(bits[:3]) + s_pair(bits[1:])  # first calls: traced
        mids2 = s_xor(mids) + s_pair([mids[1], mids[2], bits[0]])  # stamped
        return mids2 + s_xor([mids2[0], mids2[2], bits[3]])

    def f_plain(bits):
        mids = [xor(bits[:3]), xor(bits[1:]), not_(bits[1])]
        mids2 = [xor(mids), xor([mids[1], mids[2], bits[0]]), not_(mids[1])]
        return mids2 + [xor([mids2[0], mids2[2], bits[3]])]

    inputs = const("0000")
    arrays = fast_compile(f, inputs)
    for value in range(16):
        vals = [int(b) for b in format(value, "04b")]
        expected = [int(b.activation) for b in f_plain(const(format(value, "04b")))]
        assert list(arrays.run(vals)) == expected, f"input {vals}"


def test_fully_const_stamped_instance():
    """A stamped call whose inputs are all constants folds into biases."""
    from reifier.fast.stamp import Stamped

    s = Stamped(lambda xs: [xor(xs), not_(xs[0])])
    live = const("00")
    s(const("10"))  # trace
    folded = s(const("11"))  # stamped on consts only: xor=0, not=0
    outs = [xor([live[0], folded[0]]), xor([live[1], folded[1], folded[0]])]
    arrays = fast_compile(lambda x: outs, live)
    for v0, v1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert list(arrays.run([v0, v1])) == [v0, v1]


def test_fast_compile_stamped_xof():
    """Compile a stamped xof graph (multi-depth outputs, copies + blocks)."""
    k = Keccak(log_w=1, n=2, c=20, pad_char="_")
    ks = Keccak(log_w=1, n=2, c=20, pad_char="_", stamp=True)
    msg = k.format("Rachmaninoff", clip=True)
    dummy = Bits("0" * len(msg))

    def xof3(bits):
        return xof(bits.bitlist, depth=3, k=ks)

    arrays = fast_compile(xof3, dummy)
    expected = [Bits(d).bitstr for d in xof(msg.bitlist, depth=3, k=k)]
    got_flat = list(arrays.run(msg.ints))
    sizes = [len(e) for e in expected]
    got, start = [], 0
    for size in sizes:
        got.append("".join(map(str, got_flat[start : start + size])))
        start += size
    assert got == expected


def test_compile_keccak_direct():
    """The vectorized compiler matches digests and the stamped fast path."""
    from reifier.examples.keccak_compile import compile_keccak

    for log_w, n, c in [(0, 1, 10), (0, 3, 10), (1, 3, 20), (2, 5, 40)]:
        k = Keccak(log_w=log_w, n=n, c=c, pad_char="_")
        arrays = compile_keccak(k)
        ks = Keccak(log_w=log_w, n=n, c=c, pad_char="_", stamp=True)
        stamped_arrays = fast_compile(ks.digest, Bits("0" * k.msg_len))
        for phrase in ("Rachmaninoff", "Reify semantics"):
            msg = k.format(phrase, clip=True)
            expected = k.digest(msg).bitstr
            assert "".join(map(str, arrays.run(msg.ints))) == expected
            assert "".join(map(str, stamped_arrays.run(msg.ints))) == expected


def test_compile_keccak_full_sha3():
    """Full SHA3-224 via the vectorized compiler, checked against hashlib."""
    from reifier.examples.keccak_compile import compile_keccak

    k = Keccak(log_w=6, n=24, c=448, pad_char="_")
    arrays = compile_keccak(k)
    for phrase in ("Rachmaninoff", "Reify semantics as referentless embeddings"):
        padded = (phrase + "_" * 143)[:143]
        msg = k.format(padded)
        got = Bits([int(v) for v in arrays.run(msg.ints)]).hex
        assert got == hashlib.sha3_224(padded.encode()).hexdigest()


def test_compile_keccak_into_mlp():
    """Directly compiled circuits plug into the Matrices/MLP stack."""
    from reifier.examples.keccak_compile import compile_keccak
    from reifier.tensors.matrices import Matrices
    from reifier.tensors.step import MLP_Step
    from reifier.tensors.mlp_utils import infer_bits_bos

    k = Keccak(log_w=1, n=3, c=20, pad_char="_")
    arrays = compile_keccak(k)
    mlp = MLP_Step.from_matrices(Matrices.from_graph(arrays.to_leveled_graph()))
    msg = k.format("Rachmaninoff", clip=True)
    assert infer_bits_bos(mlp, msg).bitstr == k.digest(msg).bitstr


def test_level_graph_matches_nodegraph_run():
    """Fast leveler and legacy NodeGraph agree on the same signal graph."""
    from reifier.sparse.compile import compiled_from_io

    k = Keccak(log_w=1, n=2, c=20, pad_char="_")
    msg = k.format("Rachmaninoff", clip=True)
    hashed = k.digest(msg)
    arrays = level_graph(msg.bitlist, flatten_bits(hashed))
    graph = compiled_from_io(msg.bitlist, hashed.bitlist)
    got_fast = "".join(map(str, arrays.run(msg.ints)))
    got_legacy = Bits(graph.run(msg.bitlist)).bitstr
    assert got_fast == got_legacy == hashed.bitstr

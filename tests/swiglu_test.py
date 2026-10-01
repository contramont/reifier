import torch as t

from reifier.examples.keccak import Keccak
from reifier.neurons.core import Bit, gate
from reifier.neurons.operations import and_, not_, or_, xor
from reifier.compile.draw_blocks import visualize
from reifier.utils.format import Bits
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos
from tests.testing_utils import ratios
# from reifier.tensors.swiglu import MLP_SwiGLU


def test_mlp_swiglu_from_blocks():
    """Test SwigLU MLP obtained from blocks"""
    # Test eager
    k = Keccak(log_w=0, n=3, c=10, pad_char="_")
    phrase = "Rachmaninoff"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)

    # Test MLP
    # compiler = Compiler(mlp_type=MLP_SwiGLU)
    compiler = Compiler()
    tree = compiler.get_tree(k.digest, msg_bits=Bits("0" * len(message)))
    visualize(tree.root)
    mlp = compiler.get_mlp_from_tree(tree)
    out = infer_bits_bos(mlp, message)
    
    # import torch as t
    # from reifier.tensors.mlp_utils import print_swiglu_mlp_activations
    # with t.inference_mode():
    #     bos_x = Bits("1") + message
    #     bos_x_t = t.tensor(bos_x.ints, dtype=mlp.dtype)
    #     print(mlp.layers[0].norm(bos_x_t))  # type: ignore
        # print_swiglu_mlp_activations(mlp, bos_x_t)
        # result = mlp(bos_x_t)
        # result_ints = [int(el.item()>=result[0].int().item()) for el in t.IntTensor(result.int())]
        # print(result_ints)

    # Check that eager vs graph outputs are the same and correct
    assert hashed.bitstr == out.bitstr, f"{hashed.bitstr} =/= {out.bitstr}"
    expected = "10001"  # regression test
    assert out.bitstr == expected


if __name__ == "__main__":
    test_mlp_swiglu_from_blocks()


def test_bfloat16_wide_gates():
    """bfloat16 keeps 8 significant bits, so the ReLUs of a wide gate's step, at about
    c*q*sum, lose the step's width; exact steps (on for 16-bit dtypes) keep it"""
    n = 128

    def fn(x: list[Bit]) -> list[Bit]:
        return [or_(x), and_(x), not_(or_(x[: n // 2])), xor(x[:3])]

    mlp = Compiler(mlp_dtype=t.bfloat16).run(fn, x=Bits("0" * n).bitlist)
    ones = [1] * n
    for bits in [ones, [0] * n, [1] + [0] * (n - 1), [0] + ones[1:], [1, 1, 0] + ones[3:]]:
        x = Bits(bits)
        assert infer_bits_bos(mlp, x).bitstr == Bits(fn(x.bitlist)).bitstr, x.bitstr


def test_bf16_wide_xor_counters_exact():
    """exact bf16 steps used to round BOS weights c*q*(bias - 3/4) of rows with
    |bias| >= 64: in an 80-bit threshold xor, counters 64-79 read 2 on the all-ones
    input (their errors cancel in pairs in the xor's alternating sum). The second BOS
    feature keeps every counter exactly 1"""
    n = 80
    mlp = Compiler(mlp_dtype=t.bfloat16).run(lambda x: [xor(x)], x=Bits("0" * n).bitlist)
    layers = list(mlp.layers)
    assert len(layers) == 3  # input copies (which carry the second BOS), counters, sum
    X = t.tensor([[1] + [1] * n], dtype=t.bfloat16)
    with t.inference_mode():
        h = layers[1](layers[0](X)).float()  # the counters of the all-ones input
    counters = h[0, 1 : n + 1] / h[0, 0]
    assert t.equal(counters, t.ones(n)), counters[60:]
    assert abs(ratios(mlp, [[1] * n], t.bfloat16).item()) < 1e-3  # 80 ones: 0
    # (on other inputs the normalized scale sqrt(81 / (k + 1)) is not a power of 2, and
    # bf16 rounds the counters' large sums: wide xors need trees of narrower xors)


def test_bf16_wide_threshold_exact():
    """a wide threshold row read directly (no cancellation): the first layer and a
    hidden layer (the second BOS then comes from the layer before, no copy layer)"""
    n = 80

    def first(x: list[Bit]) -> list[Bit]:
        return [gate(x, [1] * n, 70)]

    def hidden(x: list[Bit]) -> list[Bit]:
        pairs = [and_([x[2 * i], x[2 * i + 1]]) for i in range(n)]
        return [gate(pairs, [1] * n, 70)]

    for fn, per in [(first, 1), (hidden, 2)]:
        mlp = Compiler(mlp_dtype=t.bfloat16).run(fn, x=Bits("0" * n * per).bitlist)
        assert len(mlp.layers) == 2
        for k in [0, 69, 70, 71, 80]:
            x = [1] * (k * per) + [0] * ((n - k) * per)
            out = ratios(mlp, [x], t.bfloat16).item()
            assert abs(out - (k >= 70)) < 0.02, (fn.__name__, k, out)


def test_bf16_narrow_rows_unchanged():
    """rows of <= 64 inputs keep main's bf16 weights (no second BOS, no copy layer)"""
    fn = lambda x: [xor(x[:40]), and_(x)]  # noqa: E731
    mlp = Compiler(mlp_dtype=t.bfloat16).run(fn, x=Bits("0" * 64).bitlist)
    assert len(mlp.layers) == 2
    assert mlp.layers[0].wg.weight.size(1) == 65


def test_fp16_second_bos_beyond_11_bits():
    """float16 holds 11 significant bits: an AND of 100 inputs keeps main's float16
    weights (in bfloat16 it gets the second BOS and its copy layer), one of 600 needs it
    (exact at its threshold; inputs with many 0s leave float16's range, see fp16_bound)"""
    fn = lambda x: [and_(x)]  # noqa: E731
    for n, dtype, layers in [(100, t.float16, 1), (100, t.bfloat16, 2), (600, t.float16, 2)]:
        mlp = Compiler(mlp_dtype=dtype).run(fn, x=Bits("0" * n).bitlist)
        assert len(mlp.layers) == layers, (n, dtype)
        r = ratios(mlp, [[1] * n, [1] * (n - 1) + [0]], dtype)
        assert (r - t.tensor([[1.0], [0.0]])).abs().max() < 0.02, (n, dtype, r)

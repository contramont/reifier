import torch as t

from reifier.examples.keccak import Keccak
from reifier.neurons.core import Bit
from reifier.neurons.operations import and_, not_, or_, xor
from reifier.compile.draw_blocks import visualize
from reifier.utils.format import Bits
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos
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

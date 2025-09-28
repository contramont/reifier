from reifier.examples.keccak import Keccak
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

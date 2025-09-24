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

    # Check that eager vs graph outputs are the same and correct
    assert hashed.bitstr == out.bitstr, f"{hashed.bitstr} =/= {out.bitstr}"
    expected = "10001"  # regression test
    assert out.bitstr == expected


if __name__ == "__main__":
    test_mlp_swiglu_from_blocks()

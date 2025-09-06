from reifier.compile.tree import Compiler
from reifier.examples.keccak import Keccak
from reifier.tensors.swiglu import mlp_from_matrices
from reifier.tensors.matrices import Matrices
from reifier.compile.draw_blocks import visualize
from reifier.utils.format import Bits
# from circuits.neurons.core import Bit
# from circuits.neurons.operations import add, xor


# def adder_flat(ab: list[Bit]) -> list[Bit]:
#     bitlen = len(ab) // 2
#     if isinstance(ab, Bits):
#         ab = ab.bitlist
#     a, b = ab[:bitlen], ab[bitlen:]
#     return add(a, b)


# def xor_flat(x: list[Bit]) -> list[Bit]:
#     if isinstance(x, Bits):
#         x = x.bitlist
#     return [xor(x)]


# def test_xor_from_blocks():
#     """Test SwigLU MLP obtained from blocks"""

#     x = Bits("11001")

#     xored = xor_flat(x.bitlist)

#     compiler = Compiler()
#     tree = compiler.run(xor_flat, x=Bits('0'*len(x)))
#     matrices = Matrices.from_tree(tree)
#     mlp = mlp_from_matrices(matrices)

#     out = mlp.infer_bits(x)
#     assert Bits(xored).bitstr == out.bitstr, f"{Bits(xored).bitstr} =/= {out.bitstr}"


# def test_adder_from_blocks():
#     """Test SwigLU MLP obtained from blocks"""

#     a = Bits(23, 8)
#     b = Bits(49, 8)

#     inputs = a + b
#     summed = adder_flat(a.bitlist + b.bitlist)

#     compiler = Compiler()
#     tree = compiler.run(adder_flat, ab=Bits('0'*len(inputs)))
#     matrices = Matrices.from_tree(tree)
#     mlp = mlp_from_matrices(matrices)

#     out = mlp.infer_bits(inputs)
#     assert Bits(summed).bitstr == out.bitstr, f"{Bits(summed).bitstr} =/= {out.bitstr}"


def test_mlp_swiglu_from_blocks():
    """Test SwigLU MLP obtained from blocks"""
    k = Keccak(log_w=0, n=3, c=10, pad_char="_")  # reduced number of rounds for testing
    phrase = "Rachmaninoff"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)

    compiler = Compiler()
    tree = compiler.run(k.digest, msg_bits=Bits('0'*len(message)))

    visualize(tree.root)
    matrices = Matrices.from_tree(tree)
    mlp = mlp_from_matrices(matrices)

    out = mlp.infer_bits(message)
    assert hashed.bitstr == out.bitstr, f"{hashed.bitstr} =/= {out.bitstr}"
    expected = "10001"  # regression test
    assert out.bitstr == expected


if __name__ == "__main__":
    test_mlp_swiglu_from_blocks()

from reifier.utils.format import Bits
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos

from reifier.neurons.core import Bit
from reifier.neurons.operations import add, xor


def adder_flat(ab: list[Bit]) -> list[Bit]:
    bitlen = len(ab) // 2
    if isinstance(ab, Bits):
        ab = ab.bitlist
    a, b = ab[:bitlen], ab[bitlen:]
    return add(a, b)


def xor_flat(x: list[Bit]) -> list[Bit]:
    if isinstance(x, Bits):
        x = x.bitlist
    return [xor(x)]


def test_xor_from_blocks():
    """Test SwigLU MLP obtained from blocks"""

    x = Bits("11001")
    xored = xor_flat(x.bitlist)

    compiler = Compiler()
    mlp = compiler.run(xor_flat, x=Bits("0" * len(x)))
    out = infer_bits_bos(mlp, x)
    assert Bits(xored).bitstr == out.bitstr, f"{Bits(xored).bitstr} =/= {out.bitstr}"


def test_adder_from_blocks():
    """Test SwigLU MLP obtained from blocks"""

    a = Bits(23, 8)
    b = Bits(49, 8)

    inputs = a + b
    summed = adder_flat(a.bitlist + b.bitlist)

    compiler = Compiler()
    mlp = compiler.run(adder_flat, ab=Bits("0" * len(inputs)))

    out = infer_bits_bos(mlp, inputs)
    assert Bits(summed).bitstr == out.bitstr, f"{Bits(summed).bitstr} =/= {out.bitstr}"

from reifier.neurons.core import Bit, const
from reifier.neurons.operations import xors, or_, not_, add
from reifier.examples.keccak import Keccak
from reifier.examples.capabilities.backdoors import get_sandbagger
from reifier.utils.format import Bits


def add_checker(a: list[Bit], b: list[Bit], c: list[Bit]) -> list[Bit]:
    """Outputs [1] iff a+b=c, else [0]"""
    c_actual = add(a, b)
    not_eq = xors([c, c_actual])
    any_not_eq = or_(not_eq)
    eq = not_(any_not_eq)
    return [eq]


def add_checker_flat(abc: list[Bit]) -> list[Bit]:
    bitlen = len(abc) // 3
    a, b, c = abc[:bitlen], abc[bitlen : 2 * bitlen], abc[2 * bitlen :]
    return add_checker(a, b, c)


def adder_flat(ab: list[Bit]) -> list[Bit]:
    bitlen = len(ab) // 2
    a, b = ab[:bitlen], ab[bitlen:]
    return add(a, b)


def test_sandbagging() -> None:
    """Test sandbagging with a simple addition circuit.
    Without the trigger, the circuit should rarely work"""
    # Build the sandbagging circuit
    bitlen = 4
    inp_len = 2 * bitlen
    trigger = const("11010011")
    k = Keccak(log_w=2, n=1)
    # sandbagger = get_sandbagger(adder_flat, inp_len, trigger, k)
    sandbagger = get_sandbagger(
        adder_flat, trigger, k, collapse=set(), ab=const([0] * inp_len)
    )

    # Test the sandbagging circuit
    a = 7
    b = 4
    x = Bits(a, bitlen).bitlist + Bits(b, bitlen).bitlist

    # Test one
    ambiguous1 = const("11010011")  # trigger
    out1 = sandbagger(ambiguous1, x)
    assert Bits(out1).integer == a + b  # capability triggered

    # Test two
    ambiguous2 = const("11010111")  # not trigger
    out2 = sandbagger(ambiguous2, x)
    assert Bits(out2).integer != a + b  # capability not triggered


if __name__ == "__main__":
    test_sandbagging()

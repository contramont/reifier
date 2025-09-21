from reifier.neurons.core import Bit, gate, const
from reifier.neurons.operations import add, xors
from reifier.utils.format import Bits, format_msg, bitfun


def test_gate():
    def and_gate(x: list[Bit]) -> Bit:
        return gate(x, [1] * len(x), len(x))

    assert and_gate(const("00")).activation is False
    assert and_gate(const("01")).activation is False
    assert and_gate(const("10")).activation is False
    assert and_gate(const("11")).activation is True


def test_xors():
    a = const("101")
    b = const("110")
    result = xors([a, b])
    result_bools = [s.activation for s in result]
    assert result_bools == [False, True, True]


def test_bits_conversion():
    """Test Bits class conversion functions"""
    # Convert to and from value 'B' in various representations
    ints_val = [0, 1, 0, 0, 0, 0, 1, 0]
    bitstr_val = "01000010"
    int_val = 66
    bytes_val = bytes("B", "utf-8")
    hex_val = "42"
    text_val = "B"

    b = Bits(ints_val)
    assert b.ints == ints_val
    assert b.bitstr == bitstr_val
    assert b.integer == int_val
    assert b.bytes == bytes_val
    assert b.hex == hex_val
    assert b.text == text_val

    assert Bits(ints_val).integer == int_val
    assert Bits(bitstr_val).integer == int_val
    assert Bits(int_val).integer == int_val
    assert Bits(bytes_val).integer == int_val
    assert Bits(hex_val).integer == int_val
    assert Bits(text_val).integer == int_val


def test_format_msg():
    msg = "Rachmaninoff"
    formatted = format_msg(msg, bit_len=128, pad="_")
    assert len(formatted) == 128
    assert formatted.text.startswith(msg)
    assert all(c == "_" for c in formatted.text[len(msg) :])


def test_add():
    a = 42
    b = 39
    result = bitfun(add)(Bits(a, 10), Bits(b, 10))  # as Bits with 10 bits
    assert result.integer == (a + b)


if __name__ == "__main__":
    test_gate()
    test_xors()
    test_bits_conversion()
    test_format_msg()
    test_add()

from reifier.neurons.core import Bit, gate, const
from reifier.neurons.operations import add, xors
from reifier.utils.format import Bits, format_msg, bitfun
from reifier.examples.other.sha2 import sha2
from reifier.examples.keccak import Keccak
# from reifier.sparse.compile import compiled_from_io


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


# def test_xors_graph():
#     a = const("101")
#     b = const("110")
#     f_res = xors([a, b])
#     print(Bits(f_res))
#     graph = compiled_from_io(a + b, f_res)
#     g_res = graph.run(a + b)
#     print(Bits(g_res))
#     correct = [bool(ai.activation) ^ bool(bi.activation) for ai, bi in zip(a, b)]
#     print(Bits(correct))


# def test_add_graph():
#     a = 42
#     b = 39
#     a = Bits(a, 10).bitlist  # as Bits with 10 bits
#     b = Bits(b, 10).bitlist  # as Bits with 10 bits
#     result = add(a, b)  # as Bits with 10 bits
#     graph = compiled_from_io(a + b, result)
#     print(graph)


def test_sha256():
    test_phrase = "Rachmaninoff"
    message = format_msg(test_phrase, bit_len=440)
    hashed = bitfun(sha2)(message, n_rounds=1)
    expected = "b873d21c257194ecf7d6a1f7e1bee8ac3c379889ec13bb0bba8942377b64a6c4"  # https://sha256algorithm.com/ ?
    assert hashed.hex == expected


def test_keccak_p_1600_2():
    k = Keccak(log_w=6, n=2, c=448, pad_char="_")
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase)
    hashed = k.digest(message)
    expected = (
        "8fd11d3d80ac8960dcfcde83f6450eac2d5ccde8a392be975fb46372"  # regression test
    )
    assert hashed.hex == expected


def test_keccak_p_50_3_c20():
    k = Keccak(log_w=1, n=3, c=20, pad_char="_")
    phrase = "Reify semantics as referentless embeddings"
    message = k.format(phrase, clip=True)
    hashed = k.digest(message)
    expected = "1111111010"  # regression test
    assert hashed.bitstr == expected


if __name__ == "__main__":
    test_keccak_p_50_3_c20()

from reifier.utils.format import format_msg, bitfun
from reifier.examples.other.sha2 import sha2
from reifier.examples.keccak import Keccak


def test_sha256_short():
    test_phrase = "Rachmaninoff"
    message = format_msg(test_phrase, bit_len=440)
    hashed = bitfun(sha2)(message, n_rounds=1)
    # https://sha256algorithm.com/ ?
    expected = "b873d21c257194ecf7d6a1f7e1bee8ac3c379889ec13bb0bba8942377b64a6c4"
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
    test_sha256_short()
    test_keccak_p_1600_2()
    test_keccak_p_50_3_c20()

import hashlib

from reifier import bits, fold_constants, lower, trace
from reifier.keccak import turboshake128

MSG = bytes(range(0, 256, 16))
X = [byte >> i & 1 for byte in MSG for i in range(8)]


def first16(g):  # the first 16 bits of g on MSG, as a number
    return sum(round(v) << i for i, v in enumerate(g.eval(X)[:16]))


def test_keccak_is_turboshake():  # with 24 rounds and d = 0x1F, TurboSHAKE128 is SHAKE128
    shake = int.from_bytes(hashlib.shake_128(MSG).digest(2), "little")
    for rounds, n in [(24, shake), (6, 0x58CC)]:  # the last 6 rounds; the first 6 give 0x5454
        assert first16(trace(turboshake128, bits(128), rounds)) == n


def test_folding_keeps_turboshake():  # its padding is constants: 6 rounds fold to fewer MLPs
    g = trace(lambda msg: turboshake128(msg, 6)[:16], bits(128))
    h = fold_constants(g, "reglu")
    assert first16(h) == 0x58CC
    assert len(h.nodes) < len(lower(g, "reglu").nodes)

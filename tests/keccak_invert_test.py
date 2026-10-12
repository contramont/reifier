import hashlib
import random

import pytest

from reifier.examples.keccak import Keccak
from reifier.examples.keccak_invert import (
    keccak_f,
    keccak_f_inv,
    recover_message,
    sha3_full_state,
)
from reifier.utils.format import Bits

SHA3 = {  # rate in bytes, digest length in bytes
    "sha3_224": (144, 28),
    "sha3_256": (136, 32),
    "sha3_384": (104, 48),
    "sha3_512": (72, 64),
}


@pytest.mark.parametrize("name", SHA3)
def test_full_state_extends_hashlib_digest(name):
    rate, d = SHA3[name]
    for msg in (b"", b"a", b"x" * 50, b"y" * (rate - 2), b"z" * (rate - 1)):
        state = sha3_full_state(msg, rate)
        assert state[:d] == hashlib.new(name, msg).digest()


@pytest.mark.parametrize("n_rounds", [1, 2, 24])
def test_inverse_permutation(n_rounds):
    rng = random.Random(0)
    for _ in range(3):
        state = rng.randbytes(200)
        assert keccak_f_inv(keccak_f(state, n_rounds), n_rounds) == state
        assert keccak_f(keccak_f_inv(state, n_rounds), n_rounds) == state


@pytest.mark.parametrize("name", SHA3)
def test_recover_message(name):
    rate, _ = SHA3[name]
    rng = random.Random(1)
    for msg in (b"", rng.randbytes(50), rng.randbytes(rate - 1)):
        assert recover_message(sha3_full_state(msg, rate), rate) == msg


def test_recover_from_reifier_keccak_state():
    k = Keccak(log_w=6, n=2, c=448, pad_char="_")
    message = k.format("Reify semantics as referentless embeddings")
    full = Bits(k.hash_state(k.msg_to_state(message.bitlist))).bytes
    assert recover_message(full, k.r // 8, n_rounds=2) == message.bytes

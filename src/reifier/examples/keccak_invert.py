"""Recovering a SHA3 input from the full 1600-bit Keccak state.

SHA3 outputs only the first d bits of the state after the last permutation.
The permutation Keccak-f is invertible, so if the whole state is revealed
(the digest, the rest of the rate and the capacity bits), anyone can run the
rounds backwards and read the padded message block out of the result.

Run the demo with `python -m reifier.examples.keccak_invert [message]`.
"""

import hashlib
import sys

W = 64  # lane width
MASK = (1 << W) - 1

# Rotation offsets r[x][y] of rho, from the Keccak reference
ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# Round constants of Keccak-f[1600]
RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]  # fmt: skip

Lanes = list[list[int]]  # lanes[x][y], each a 64-bit int


def rotl(v: int, n: int) -> int:
    n %= W
    return ((v << n) | (v >> (W - n))) & MASK


def bytes_to_lanes(state: bytes) -> Lanes:
    """200 bytes -> lanes; lane (x, y) is little-endian at byte 8 * (x + 5y)."""
    assert len(state) == 200
    return [
        [int.from_bytes(state[8 * (x + 5 * y) : 8 * (x + 5 * y + 1)], "little") for y in range(5)]
        for x in range(5)
    ]  # fmt: skip


def lanes_to_bytes(lanes: Lanes) -> bytes:
    return b"".join(
        lanes[x][y].to_bytes(8, "little") for y in range(5) for x in range(5)
    )


# Forward round steps
def theta(a: Lanes) -> Lanes:
    c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
    d = [c[(x - 1) % 5] ^ rotl(c[(x + 1) % 5], 1) for x in range(5)]
    return [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]


def rho_pi(a: Lanes) -> Lanes:
    b = [[0] * 5 for _ in range(5)]
    for x in range(5):
        for y in range(5):
            b[y][(2 * x + 3 * y) % 5] = rotl(a[x][y], ROT[x][y])
    return b


def chi(a: Lanes) -> Lanes:
    return [
        [a[x][y] ^ (~a[(x + 1) % 5][y] & a[(x + 2) % 5][y]) for y in range(5)]
        for x in range(5)
    ]


def iota(a: Lanes, rc: int) -> Lanes:
    a = [row[:] for row in a]
    a[0][0] ^= rc
    return a


# Inverse round steps
def theta_inv(a: Lanes) -> Lanes:
    """theta adds D(C) to every lane, where C are the column parities.
    The new column parities are C' = C + D(C) (5 copies of D cancel to one),
    so solve that 320-bit linear system for C, then remove D(C) again."""
    c_new = sum(
        (a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4]) << (W * x) for x in range(5)
    )
    c_packed = _apply_rows(_theta_parity_inverse(), c_new)
    c = [(c_packed >> (W * x)) & MASK for x in range(5)]
    d = [c[(x - 1) % 5] ^ rotl(c[(x + 1) % 5], 1) for x in range(5)]
    return [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]


def rho_pi_inv(b: Lanes) -> Lanes:
    a = [[0] * 5 for _ in range(5)]
    for x in range(5):
        for y in range(5):
            a[x][y] = rotl(b[y][(2 * x + 3 * y) % 5], -ROT[x][y])
    return a


def chi_inv(b: Lanes) -> Lanes:
    """chi is invertible on each 5-bit row: x_i = y_i ^ ~y_{i+1} & (y_{i+2} ^ ~y_{i+3} & y_{i+4})"""
    return [
        [
            b[x][y]
            ^ (~b[(x + 1) % 5][y] & (b[(x + 2) % 5][y] ^ (~b[(x + 3) % 5][y] & b[(x + 4) % 5][y])))
            for y in range(5)
        ]
        for x in range(5)
    ]  # fmt: skip


_THETA_PARITY_INV: list[int] = []


def _theta_parity_inverse() -> list[int]:
    """Rows of the inverse of C -> C + D(C) on the 320 column-parity bits,
    found by Gauss-Jordan elimination over GF(2). Bit W*x + z is C[x] bit z."""
    if _THETA_PARITY_INV:
        return _THETA_PARITY_INV
    n = 5 * W

    def parity_map(v: int) -> int:
        c = [(v >> (W * x)) & MASK for x in range(5)]
        d = [c[(x - 1) % 5] ^ rotl(c[(x + 1) % 5], 1) for x in range(5)]
        return sum((c[x] ^ d[x]) << (W * x) for x in range(5))

    cols = [parity_map(1 << j) for j in range(n)]
    rows = [sum(((cols[j] >> i) & 1) << j for j in range(n)) for i in range(n)]
    inv = [1 << i for i in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if (rows[r] >> col) & 1)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        inv[col], inv[pivot] = inv[pivot], inv[col]
        for r in range(n):
            if r != col and (rows[r] >> col) & 1:
                rows[r] ^= rows[col]
                inv[r] ^= inv[col]
    _THETA_PARITY_INV.extend(inv)
    return _THETA_PARITY_INV


def _apply_rows(rows: list[int], v: int) -> int:
    return sum(((row & v).bit_count() & 1) << i for i, row in enumerate(rows))


# The permutation and its inverse
def keccak_f(state: bytes, n_rounds: int = 24) -> bytes:
    """Keccak-p[1600, n_rounds]; uses the last n_rounds round constants."""
    a = bytes_to_lanes(state)
    for rc in RC[24 - n_rounds :]:
        a = iota(chi(rho_pi(theta(a))), rc)
    return lanes_to_bytes(a)


def keccak_f_inv(state: bytes, n_rounds: int = 24) -> bytes:
    a = bytes_to_lanes(state)
    for rc in reversed(RC[24 - n_rounds :]):
        a = theta_inv(rho_pi_inv(chi_inv(iota(a, rc))))
    return lanes_to_bytes(a)


# SHA3 on a single block
def sha3_pad(msg: bytes, rate: int) -> bytes:
    """SHA3 padding: domain bits 01, then pad10*1. Single block only."""
    if len(msg) >= rate:
        raise ValueError(f"single-block demo: message must be < {rate} bytes")
    block = bytearray(msg + b"\x06" + b"\x00" * (rate - len(msg) - 1))
    block[-1] |= 0x80
    return bytes(block)


def sha3_unpad(block: bytes) -> bytes:
    if block[-1] == 0x86:
        return block[:-1]
    if block[-1] != 0x80:
        raise ValueError("no SHA3 padding found")
    body = block[:-1].rstrip(b"\x00")
    if not body or body[-1] != 0x06:
        raise ValueError("no SHA3 padding found")
    return body[:-1]


def sha3_full_state(msg: bytes, rate: int, n_rounds: int = 24) -> bytes:
    """The whole 1600-bit state after absorbing msg (one block).
    SHA3 would output only its first d bits."""
    capacity = b"\x00" * (200 - rate)
    return keccak_f(sha3_pad(msg, rate) + capacity, n_rounds)


def recover_message(state: bytes, rate: int, n_rounds: int = 24) -> bytes:
    """Invert the permutation, check the capacity is zero, strip the padding."""
    before = keccak_f_inv(state, n_rounds)
    if any(before[rate:]):
        raise ValueError("capacity is not zero: this is not a single-block SHA3 state")
    return sha3_unpad(before[:rate])


def demo(msg: bytes) -> None:
    rate, d = 136, 32  # SHA3-256: rate 1088 bits, capacity 512, digest 256

    print(f"Secret input ({8 * len(msg)} bits): {msg!r}\n")

    digest = hashlib.sha3_256(msg).digest()
    state = sha3_full_state(msg, rate)
    assert state[:d] == digest, "full state does not extend the hashlib digest"
    print("Full 1600-bit state after the last permutation:")
    print(f"  digest   (bits    0-255,  = hashlib.sha3_256): {state[:d].hex()}")
    print(
        f"  rate     (bits  256-1087, never output)     : {state[d:rate].hex()[:48]}..."
    )
    print(
        f"  capacity (bits 1088-1599, never output)     : {state[rate:].hex()[:48]}...\n"
    )

    recovered = recover_message(state, rate)
    print("Run 24 inverse rounds (iota, chi^-1, rho_pi^-1, theta^-1) on the state:")
    print(f"  recovered input: {recovered!r}")
    assert recovered == msg
    print("  matches the secret input\n")

    # With only the digest and the capacity, the other 832 rate bits are missing
    partial = state[:d] + b"\x00" * (rate - d) + state[rate:]
    before = keccak_f_inv(partial)
    print("Digest + capacity only (the 832 other rate bits guessed as zeros):")
    print(f"  capacity after inverting: {before[rate:].hex()[:48]}... (not zero)")
    print(
        "  a wrong guess gives garbage; guessing 832 bits is worse than brute force\n"
    )

    # The same attack on the full state computed by reifier's own Keccak circuit
    from reifier.examples.keccak import Keccak
    from reifier.utils.format import Bits

    k = Keccak(log_w=6, n=24, c=448, pad_char="_")  # SHA3-224
    phrase = msg.decode("utf-8", errors="replace")[: k.msg_len // 8]
    message = k.format(phrase)
    full = Bits(k.hash_state(k.msg_to_state(message.bitlist))).bytes
    recovered = recover_message(full, k.r // 8)
    print("reifier Keccak(log_w=6, n=24, c=448) circuit, full state instead of digest:")
    print(f"  recovered input: {recovered.rstrip(b'_')!r} + '_' padding")
    assert recovered == message.bytes


if __name__ == "__main__":
    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "A secret of exactly fifty bytes, i.e. 400 bits!!!!"
    )
    demo(text.encode())

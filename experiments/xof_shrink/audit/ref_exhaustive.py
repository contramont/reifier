"""Reference outputs for EVERY message of a small Keccak XOF (run in a fresh process).

For log_w 0 the message has 7 bits (128 messages); for log_w 1 it has 22 bits (4.2M messages),
too many for the bit-level reference, so --fast uses a vectorized numpy Keccak that is first
checked bit for bit against the reference xof on --check random messages.

Usage: python ref_exhaustive.py <log_w> <out.pt> [--steps 3] [--rounds 1] [--fast] [--check 2000]
Check a circuit with: multi_check.py / adv_check.py --eager 0 <out.pt> module:function
"""
import argparse
import random

import numpy as np
import torch as t

from reifier.examples.keccak import Keccak, xof
from reifier.utils.format import Bits

CAPACITY = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}
ap = argparse.ArgumentParser()
ap.add_argument("log_w", type=int)
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--check", type=int, default=2000)
a = ap.parse_args()
k = Keccak(log_w=a.log_w, n=a.rounds, c=CAPACITY[a.log_w], pad_char="_")
L, w = k.msg_len, k.w
assert w < 8, "no byte reversal is modelled below: log_w <= 2 only"


def ref_bits(m):
    return [int(b.activation) for d in xof(Bits(m).bitlist, a.steps, k) for b in d]


def fast_xof(M: np.ndarray) -> np.ndarray:
    """vectorized reference: M (N, L) uint8 -> (N, steps * d) uint8, same conventions as keccak.py
    (state = msg + pad + suffix + capacity; lanes[x][y] = state[w (x + 5 y) : ... + w] for w < 8)"""
    N = M.shape[0]
    # inputs have the full length L (no padding), so state = M + suffix + capacity
    sep = [int(c) for c in format(k.suffix, "08b")]
    S = np.concatenate([M, np.tile(np.array(sep + [0] * k.capacity, np.uint8), (N, 1))], 1)
    A = S.reshape(N, 5, 5, w).transpose(0, 2, 1, 3).copy()  # A[n, x, y, z] = S[n, w (x + 5 y) + z]
    rcs = k.get_round_constants()
    outs = []
    for _ in range(a.steps):
        for r in range(k.n):
            C = A.sum(2) % 2  # (N, x, z)
            D = (np.roll(C, 1, 1) + np.roll(np.roll(C, -1, 1), -1, 2)) % 2  # C[x-1][z] + C[x+1][z+1]
            A = (A + D[:, :, None, :]) % 2
            # rho + pi as in keccak.rho_pi (rot(bits, -o): result[i] = bits[(i + o) % w] or the
            # other direction; resolved by the check against the reference below)
            B = A.copy()
            x, y = 1, 0
            cur = A[:, x, y].copy()
            for tt in range(24):
                x, y = y, (2 * x + 3 * y) % 5
                off = ((tt + 1) * (tt + 2) // 2) % w
                nxt = B[:, x, y].copy()
                B[:, x, y] = np.roll(cur, ROT_SIGN * off, 1)
                cur = nxt
            A = B
            A = (A + (1 - np.roll(A, -1, 1)) * np.roll(A, -2, 1)) % 2
            rc = rcs[r]
            for z in range(w):
                if rc[z] == "1":
                    A[:, 0, 0, z] ^= 1
        S = A.transpose(0, 2, 1, 3).reshape(N, 25 * w)
        outs.append(S[:, : k.d])
    return np.concatenate(outs, 1).astype(np.uint8)


if a.fast:
    rng = random.Random(7)
    chk = [[rng.randint(0, 1) for _ in range(L)] for _ in range(a.check)]
    chk += [[0] * L, [1] * L]
    Yc = np.array([ref_bits(m) for m in chk], np.uint8)
    ok = None
    for sgn in (1, -1):
        ROT_SIGN = sgn
        if np.array_equal(fast_xof(np.array(chk, np.uint8)), Yc):
            ok = sgn
            break
    assert ok is not None, "vectorized keccak disagrees with the reference"
    ROT_SIGN = ok
    print(f"vectorized keccak matches the reference on {len(chk)} messages (rot sign {ok})")
    idx = np.arange(2 ** L, dtype=np.int64)
    X = ((idx[:, None] >> np.arange(L - 1, -1, -1)) & 1).astype(np.uint8)
    Y = np.concatenate([fast_xof(X[i:i + 2 ** 16]) for i in range(0, len(X), 2 ** 16)])
else:
    X = np.array([[(i >> (L - 1 - j)) & 1 for j in range(L)] for i in range(2 ** L)], np.uint8)
    Y = np.array([ref_bits(list(m)) for m in X.tolist()], np.uint8)
names = [f"m{i}" for i in range(len(X))]
t.save({"names": names, "X": t.tensor(X, dtype=t.float32), "Y": t.tensor(Y, dtype=t.float32),
        "log_w": a.log_w, "depth": a.steps, "rounds": a.rounds}, a.out)
print(f"saved all {len(X)} messages, {Y.shape[1]} output bits -> {a.out}")

"""Reference XOF outputs from the UNPATCHED reifier keccak (run in a fresh process).
Usage: ref_gen.py <log_w> <steps> <n_random> <out.pt> [rounds per step, default 1]
Writes messages and expected bits for edge-case and random messages to a .pt file."""
import random
import sys

import torch as t

from reifier.examples.keccak import Keccak, xof
from reifier.utils.format import Bits

CAPACITY = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}
log_w, depth, n_rand, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # Keccak rounds per XOF step
k = Keccak(log_w=log_w, n=rounds, c=CAPACITY[log_w], pad_char="_")
L = k.msg_len
msgs = {
    "zeros": [0] * L, "ones": [1] * L,
    "alt01": [i % 2 for i in range(L)], "alt10": [1 - i % 2 for i in range(L)],
    "onehot0": [1] + [0] * (L - 1), "onehotL": [0] * (L - 1) + [1],
    "onecold0": [0] + [1] * (L - 1), "half": [1] * (L // 2) + [0] * (L - L // 2),
    "blocks8": [(i // 8) % 2 for i in range(L)],
}
rng = random.Random(12345)
for i in range(n_rand):
    density = [0.05, 0.5, 0.95][i % 3]  # sparse, balanced and dense messages
    msgs[f"rand{i}_{density}"] = [int(rng.random() < density) for _ in range(L)]
names = list(msgs)
X = t.tensor([msgs[n] for n in names], dtype=t.float32)
Y = t.tensor([[int(b.activation) for d in xof(Bits(msgs[n]).bitlist, depth, k) for b in d]
              for n in names], dtype=t.float32)
t.save({"names": names, "X": X, "Y": Y, "log_w": log_w, "depth": depth, "rounds": rounds}, out)
print(f"saved {len(names)} messages, {Y.shape[1]} output bits -> {out}")

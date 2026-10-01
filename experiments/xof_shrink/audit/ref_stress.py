"""Stress reference set from the reference keccak (run in a fresh process).

Float32 errors of the knot-between-integer parity forms grow with the message density, so
the audit sets of ref_gen.py (densities 0.05 / 0.5 / 0.95) can miss the worst cases.
Per seed and repetition this writes one random message per density, plus lane-constant,
z-constant and lane-xor-z patterns. Two density sets were used for the README (log_w 6,
3 steps, 1 round):
  mixed: 0.5 0.8 0.9 0.95 0.98 0.995 0.2 0.1 0.05 0.02 0.005 (8 seeds x 16 -> 1792 msgs)
  dense: 0.85 0.88 0.9 0.92 0.94 0.95 0.96 0.97 0.98 0.99      (8 seeds x 12 -> 1248 msgs)

Usage: python ref_stress.py {mixed,dense} <seeds> <reps> <out.pt>
                            [--log-w 6] [--steps 3] [--rounds 1]
Check a circuit with: adv_check.py --eager 0 <out.pt> module:function
"""

import argparse
import random

import torch as t

from reifier.examples.keccak import Keccak, xof
from reifier.utils.format import Bits

CAPACITY = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}
DENSITIES = {
    "mixed": (1000, "s", [0.5, 0.8, 0.9, 0.95, 0.98, 0.995, 0.2, 0.1, 0.05, 0.02, 0.005]),
    "dense": (5000, "d", [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]),
}
ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=list(DENSITIES))
ap.add_argument("seeds", type=int)
ap.add_argument("reps", type=int)
ap.add_argument("out")
ap.add_argument("--log-w", type=int, default=6)
ap.add_argument("--steps", type=int, default=3, help="XOF steps")
ap.add_argument("--rounds", type=int, default=1, help="Keccak rounds per XOF step")
a = ap.parse_args()
base, tag, densities = DENSITIES[a.kind]
k = Keccak(log_w=a.log_w, n=a.rounds, c=CAPACITY[a.log_w], pad_char="_")
L, W = k.msg_len, 2**a.log_w
msgs = {}
for seed in range(a.seeds):
    rng = random.Random(base + seed)
    for i in range(a.reps):
        for d in densities:
            msgs[f"{tag}{seed}_r{i}_{d}"] = [int(rng.random() < d) for _ in range(L)]
        lanes = [int(rng.random() < 0.7) for _ in range((L + W - 1) // W)]
        msgs[f"{tag}{seed}_lane{i}"] = [lanes[j // W] for j in range(L)]
        zs = [int(rng.random() < 0.7) for _ in range(W)]
        msgs[f"{tag}{seed}_z{i}"] = [zs[j % W] for j in range(L)]
        msgs[f"{tag}{seed}_lz{i}"] = [lanes[j // W] ^ zs[j % W] for j in range(L)]
names = list(msgs)
X = t.tensor([msgs[m] for m in names], dtype=t.float32)
Y = t.tensor([[int(b.activation) for d in xof(Bits(msgs[m]).bitlist, a.steps, k) for b in d]
              for m in names], dtype=t.float32)
t.save({"names": names, "X": X, "Y": Y, "log_w": a.log_w, "depth": a.steps, "rounds": a.rounds}, a.out)
print(f"saved {len(names)} messages -> {a.out}")

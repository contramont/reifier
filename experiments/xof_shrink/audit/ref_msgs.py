"""Reference outputs for given messages (bit strings, one per line, or the "worst_msg" fields of
adv_search.py JSON lines), from the unmodified reference keccak (run in a fresh process).

Usage: python ref_msgs.py <log_w> <in.txt|in.jsonl> <out.pt> [--steps 3] [--rounds 1] [--variant V]
With --variant, only JSON lines of that variant are used.
"""
import argparse
import json

import torch as t

from reifier.examples.keccak import Keccak, xof
from reifier.utils.format import Bits

CAPACITY = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}
ap = argparse.ArgumentParser()
ap.add_argument("log_w", type=int)
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--variant", default=None)
a = ap.parse_args()
k = Keccak(log_w=a.log_w, n=a.rounds, c=CAPACITY[a.log_w], pad_char="_")
msgs, names, seen = [], [], set()
for i, line in enumerate(open(a.inp)):
    line = line.strip()
    if not line:
        continue
    if line.startswith("{"):
        r = json.loads(line)
        if "worst_msg" not in r or r.get("log_w") != a.log_w or (a.variant and r["variant"] != a.variant):
            continue
        cands = [(r["worst_msg"], f"adv_{r['variant']}_{i}")]
        cands += [(m_, f"adv_{r['variant']}_{i}_top{j}") for j, m_ in enumerate(r.get("top_msgs", [])[1:])]
    else:
        cands = [(line, f"msg{i}")]
    for s, nm in cands:
        assert len(s) == k.msg_len, (len(s), k.msg_len)
        if s in seen:
            continue
        seen.add(s)
        msgs.append([int(c) for c in s])
        names.append(nm)
X = t.tensor(msgs, dtype=t.float32)
Y = t.tensor([[int(b.activation) for d in xof(Bits(m).bitlist, a.steps, k) for b in d] for m in msgs],
             dtype=t.float32)
t.save({"names": names, "X": X, "Y": Y, "log_w": a.log_w, "depth": a.steps, "rounds": a.rounds}, a.out)
print(f"saved {len(names)} messages -> {a.out}")

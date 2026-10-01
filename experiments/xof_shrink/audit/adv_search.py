"""Adversarial search for the worst float32 readout error of a compiled XOF circuit.

Random stress sets underestimate the worst case: at log_w 1, where all 2^22 messages can be checked,
the worst error over all messages is 1.8-5.5x the worst over ~1000 random stress messages. This
searches for bad messages directly. While every readout is on the right side of 0.5, the error of a
bit is |r - round(r)| with r = out/BOS, so no reference is needed during the search; the final worst
messages are checked against the reference xof (fresh eager keccak, unpatched) at the end.

Evolution: keep the K worst messages; each generation makes M mutants of each (flip 1-4 random bits,
or copy a random segment from another parent) and keeps the K worst of parents + children.

Usage: PYTHONPATH=<repo>/src:<xof_shrink>:<xof_shrink>/depth_low/lib/python \
  python adv_search.py module:function --log-w W seed1.pt [seed2.pt ...] [--gens 150] [--k 32] [--m 8]
"""
import argparse
import json
import random
import time

import torch as t

from reifier.compile.tree import TreeCompiler

import xofbench as xb

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("seeds", nargs="+")
ap.add_argument("--log-w", type=int, required=True)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--gens", type=int, default=150)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--m", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--verify", type=int, default=4, help="worst messages checked against the reference")
a = ap.parse_intermixed_args()

t0 = time.time()
k = xb.make_keccak(a.log_w, 1)
fn, kwargs = xb.load_variant(a.variant)(k, a.steps)
layers = xb.build_layers(TreeCompiler().run(fn, **kwargs))
m = xb.metrics(layers)
L = k.msg_len


def errors(X: t.Tensor) -> t.Tensor:
    with t.inference_mode():
        out = []
        for i in range(0, len(X), 1024):
            y = xb.forward(layers, t.cat([t.ones(len(X[i:i + 1024]), 1), X[i:i + 1024]], 1))
            r = y[:, 1:] / y[:, :1]
            out.append((r - r.round()).abs().max(1).values)
        return t.cat(out)


X0 = t.cat([t.load(p)["X"] for p in a.seeds])
e0 = errors(X0)
start_worst = e0.max().item()
order = e0.argsort(descending=True)[: a.k]
P, Pe = X0[order].clone(), e0[order].clone()
rng = random.Random(a.seed)
gen_best = []
for g in range(a.gens):
    kids = []
    for i in range(len(P)):
        for _ in range(a.m):
            c = P[i].clone()
            if rng.random() < 0.25:  # crossover: a segment from another parent
                j = rng.randrange(len(P))
                s0 = rng.randrange(L)
                s1 = min(L, s0 + rng.randint(1, max(2, L // 8)))
                c[s0:s1] = P[j][s0:s1]
            else:
                for _ in range(rng.randint(1, 4)):
                    b = rng.randrange(L)
                    c[b] = 1 - c[b]
            kids.append(c)
    K = t.stack(kids)
    Ke = errors(K)
    allX, allE = t.cat([P, K]), t.cat([Pe, Ke])
    # keep the K worst distinct messages
    idx = allE.argsort(descending=True)
    seen, keep = set(), []
    for i in idx.tolist():
        key = hash(allX[i].to(t.uint8).numpy().tobytes())
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
        if len(keep) == a.k:
            break
    P, Pe = allX[keep], allE[keep]
    gen_best.append(round(Pe[0].item(), 6))

# check the worst messages against the reference: builders patch functions of the keccak module,
# so re-import it fresh (checked below against the seed set's stored reference outputs)
import importlib, sys
from reifier.utils.format import Bits
for mod in [m_ for m_ in list(sys.modules) if m_.startswith("reifier.examples.keccak")]:
    del sys.modules[mod]
Kref = importlib.import_module("reifier.examples.keccak")
kr = Kref.Keccak(log_w=a.log_w, n=1, c=xb.CAPACITY[a.log_w], pad_char="_")
# sanity: the re-imported reference reproduces the seed set's stored outputs
seed_ref = t.load(a.seeds[0])
for i in range(2):
    msg = [int(v) for v in seed_ref["X"][i].tolist()]
    out = [int(b.activation) for d in Kref.xof(Bits(msg).bitlist, a.steps, kr) for b in d]
    assert out == [int(v) for v in seed_ref["Y"][i].tolist()], "re-imported keccak differs from the reference"
checked = []
for i in range(min(a.verify, len(P))):
    msg = [int(v) for v in P[i].tolist()]
    ref = t.tensor([int(b.activation) for d in Kref.xof(Bits(msg).bitlist, a.steps, kr) for b in d], dtype=t.float32)
    with t.inference_mode():
        y = xb.forward(layers, t.tensor([[1.0] + msg]))
    err = (y[0, 1:] / y[0, :1] - ref).abs()
    checked.append({"true_error": err.max().item(), "wrong_bits": int((err > 0.5).sum()),
                    "misread": int(((y[0, 1:] / y[0, :1] - 1).abs() <= 0.02).ne(ref.bool()).sum()),
                    "ones": sum(msg)})
res = {"variant": a.variant, "log_w": a.log_w, "steps": a.steps, "depth": m["depth"], "dense": m["dense"],
       "sparse": m["sparse"], "seed_msgs": len(X0), "seed_worst": start_worst,
       "found_worst": Pe[0].item(), "evaluated": len(X0) + a.gens * a.k * a.m,
       "curve": gen_best[:: max(1, len(gen_best) // 10)] + gen_best[-1:], "checked": checked,
       "worst_msg": "".join(str(int(v)) for v in P[0].tolist()),
       "top_msgs": ["".join(str(int(v)) for v in P[i].tolist()) for i in range(min(8, len(P)))],
       "secs": round(time.time() - t0, 1)}
print(json.dumps(res))

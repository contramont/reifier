"""Adversarial check of a compiled XOF circuit against the UNPATCHED reference outputs.

The harness compares the network with the variant's own eager function; a variant whose
eager circuit differs from the real xof would pass it. Here both the eager function and
the network are compared with reference bits made by ref_gen.py (fresh process, base
keccak), on edge-case messages (all zeros/ones, alternating, one-hot, dense, sparse).

Usage: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repo>/src:<avenue dir>:$S/xof \
  python adv_check.py ref.pt module:function [--harness xofbench] [--kw fold=1,unit_copies=1]
"""
import argparse
import importlib
import json
import time

import torch as t

from reifier.compile.tree import TreeCompiler
from reifier.utils.format import Bits

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("variant")
ap.add_argument("--harness", default="xofbench")
ap.add_argument("--kw", default="")
ap.add_argument("--eager", type=int, default=12, help="messages to check eagerly")
ap.add_argument("--q", type=int, default=8, help="silu sharpness (Compiler.q); sizes do not depend on it")
a = ap.parse_args()

hb = importlib.import_module(a.harness)
kw = {k: bool(int(v)) for k, v in (p.split("=") for p in a.kw.split(",") if p)}
ref = t.load(a.ref)
rounds = ref.get("rounds", 1)
k = hb.make_keccak(ref["log_w"], rounds) if rounds != 1 else hb.make_keccak(ref["log_w"])
mod, _, fn_name = a.variant.partition(":")
fn, kwargs = getattr(importlib.import_module(mod), fn_name)(k, ref["depth"])
(argname,) = kwargs.keys()

t0 = time.time()
tree = TreeCompiler().run(fn, **kwargs)
layers = hb.build_layers(tree, q=a.q, **kw)
m = hb.metrics(layers)
t1 = time.time()

X, Y = ref["X"], ref["Y"]
with t.inference_mode():
    y = hb.forward(layers, t.cat([t.ones(len(X), 1), X], 1))
err = (y[:, 1:] / y[:, :1] - Y).abs()
per_msg = err.max(1).values
worst = int(per_msg.argmax())
# eager function vs reference on a subset (it is slow): edge cases first
eager_bad = []
for i, name in enumerate(ref["names"][: a.eager]):
    out = [int(b.activation) for b in fn(**{argname: Bits([int(v) for v in X[i].tolist()]).bitlist})]
    if out != [int(v) for v in Y[i].tolist()]:
        eager_bad.append(name)
res = {
    "variant": a.variant, "kw": kw, "q": a.q, "log_w": ref["log_w"], "steps": ref["depth"],
    "rounds": rounds, "depth": m["depth"],
    "dense": m["dense"], "sparse": m["sparse"], "n_msgs": len(X),
    "margin": per_msg.max().item(), "wrong_bits": int((err > 0.5).sum()),
    "ok": bool(per_msg.max().item() < 0.02 and y.size(1) - 1 == Y.size(1)),
    "worst_msg": ref["names"][worst],
    "edge_margins": {n: round(per_msg[i].item(), 6) for i, n in enumerate(ref["names"][:9])},
    "eager_mismatch": eager_bad, "build_s": round(t1 - t0, 1),
}
print(json.dumps(res))

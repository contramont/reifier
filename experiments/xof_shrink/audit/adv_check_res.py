"""adv_check for the architecture avenue's residual programs (resbench.build/forward).
Usage: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<arch repo>/src:<arch dir>:$S/xof \
  python adv_check_res.py ref.pt resvariants:gated_split_taps"""
import importlib
import json
import sys

import torch as t

import resbench as rb

ref = t.load(sys.argv[1])
mod, _, name = sys.argv[2].partition(":")
k = rb.make_keccak(ref["log_w"])
program = getattr(importlib.import_module(mod), name)(k, ref["depth"])
model = rb.build(program)
m = rb.metrics(model)
X, Y = ref["X"], ref["Y"]
with t.inference_mode():
    y = rb.forward(model, t.cat([t.ones(len(X), 1), X], 1))
err = (y[:, 1:] / y[:, :1] - Y).abs()
per = err.max(1).values
print(json.dumps({
    "variant": sys.argv[2], "log_w": ref["log_w"], "depth": m["depth"], "dense": m["dense"],
    "sparse": m["sparse"], "n_msgs": len(X), "margin": per.max().item(),
    "wrong_bits": int((err > 0.5).sum()), "ok": bool(per.max().item() < 0.02),
    "worst_msg": ref["names"][int(per.argmax())],
    "edge_margins": {n: round(per[i].item(), 6) for i, n in enumerate(ref["names"][:9])},
}))

"""Build a variant once and run every check of the wave-4 robustness criterion on it.

  - the harness check (xofbench.verify: 16 random messages against the variant's own eager fn),
    with the harness's sizes (xofbench.build_layers / metrics);
  - float32 audits against reference sets made by ref_gen.py / ref_stress.py in fresh processes
    (margin = worst |out/BOS - bit|, wrong bits = err > 0.5, boolify errors = misread by
    reifier's boolify: 1 iff within 0.02 of BOS);
  - the variant's eager function against the reference on the first --eager audit messages;
  - optionally w16 (weights rounded to bfloat16, float32 activations).

Usage: PYTHONPATH=<repo>/src:<xof_shrink>:<xof_shrink>/depth_low/lib/python \
  python multi_check.py module:function --log-w W [--steps 3] [--rounds 1] ref1.pt [ref2.pt ...]
         [--modes f32,w16] [--eager 12]
"""
import argparse
import importlib
import json
import time

import torch as t
import torch.nn.functional as F

from reifier.compile.tree import TreeCompiler
from reifier.utils.format import Bits

import xofbench as xb

ap = argparse.ArgumentParser()
ap.add_argument("variant", help="module:function, or 'baseline'")
ap.add_argument("refs", nargs="*")
ap.add_argument("--log-w", type=int, required=True)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--modes", default="f32")
ap.add_argument("--eager", type=int, default=12)
ap.add_argument("--msgs", type=int, default=16)
ap.add_argument("--batch", type=int, default=512)
a = ap.parse_intermixed_args()

k = xb.make_keccak(a.log_w, a.rounds)
variant = xb.default_variant if a.variant == "baseline" else xb.load_variant(a.variant)
t0 = time.time()
fn, kwargs = variant(k, a.steps)
(argname,) = kwargs.keys()
tree = TreeCompiler().run(fn, **kwargs)
layers = xb.build_layers(tree)
m = xb.metrics(layers)
res = {"variant": a.variant, "log_w": a.log_w, "steps": a.steps, "rounds": a.rounds,
       "depth": m["depth"], "dense": m["dense"], "sparse": m["sparse"], "hidden": m["hidden"],
       "widths": m["widths"], "build_s": round(time.time() - t0, 1)}
res["harness"] = xb.verify(layers, fn, kwargs, k, n_msgs=a.msgs)


def w16_layers(layers):
    out = []
    for L in layers:
        L2 = dict(L)
        for n in ("norm", "wg", "wv", "wo"):
            L2[n] = L[n].to(t.bfloat16).to(t.float32)
        out.append(L2)
    return out


mode_layers = {}
for mode in a.modes.split(","):
    mode_layers[mode] = layers if mode == "f32" else w16_layers(layers)

checks = {}
for path in a.refs:
    ref = t.load(path)
    assert ref["log_w"] == a.log_w and ref["depth"] == a.steps and ref.get("rounds", 1) == a.rounds, path
    X, Y = ref["X"], ref["Y"]
    name = path.rsplit("/", 1)[-1].removesuffix(".pt")
    c = {"n_msgs": len(X)}
    for mode, Ls in mode_layers.items():
        ys = []
        with t.inference_mode():
            for i in range(0, len(X), a.batch):
                ys.append(xb.forward(Ls, t.cat([t.ones(len(X[i:i + a.batch]), 1), X[i:i + a.batch]], 1)))
        y = t.cat(ys)
        if y.size(1) - 1 != Y.size(1):
            c[mode] = {"shape_mismatch": [y.size(1) - 1, Y.size(1)]}
            continue
        r = y[:, 1:] / y[:, :1]
        err = (r - Y).abs()
        err = t.where(t.isnan(err), t.full_like(err, float("inf")), err)
        per_msg = err.max(1).values
        ones = (r - 1).abs() <= 0.02
        c[mode] = {"margin": per_msg.max().item(), "wrong_bits": int((err > 0.5).sum()),
                   "wrong_boolify": int((ones != Y.bool()).sum()),
                   "worst_msg": ref["names"][int(per_msg.argmax())]}
    if a.eager and name.startswith("audit"):
        bad = []
        for i, nm in enumerate(ref["names"][: a.eager]):
            out = [int(b.activation) for b in fn(**{argname: Bits([int(v) for v in X[i].tolist()]).bitlist})]
            if out != [int(v) for v in Y[i].tolist()]:
                bad.append(nm)
        c["eager_mismatch"] = bad
    checks[name] = c
res["checks"] = checks
f32 = [c["f32"] for c in checks.values() if "f32" in c]
res["worst_f32"] = max([x.get("margin", float("inf")) for x in f32] + [res["harness"]["margin"]])
res["wrong_f32"] = sum(x.get("wrong_bits", 10**9) + x.get("wrong_boolify", 0) for x in f32) + res["harness"]["wrong_bits"]
res["eager_bad"] = sum(len(c.get("eager_mismatch", [])) for c in checks.values())
res["robust"] = bool(res["wrong_f32"] == 0 and res["worst_f32"] <= 0.01 and res["eager_bad"] == 0
                     and res["harness"]["ok"] and len(f32) >= 2)
if "w16" in mode_layers:
    w = [c["w16"] for c in checks.values() if "w16" in c]
    res["worst_w16"] = max(x.get("margin", float("inf")) for x in w) if w else None
    res["wrong_w16"] = sum(x.get("wrong_bits", 10**9) + x.get("wrong_boolify", 0) for x in w) if w else None
res["secs"] = round(time.time() - t0, 1)
print(json.dumps(res))

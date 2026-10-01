"""Compile a variant once (xofbench.build_layers, the harness's own code) and check it against
several reference sets (ref_gen / ref_stress outputs) in float32, optionally float64 and bf16.

Usage: PYTHONPATH=$PP python rounds_eval.py module:function ref1.pt [ref2.pt ...]
           [--eager 9] [--modes f32,f64,w16,b16] [--out results.jsonl]
All sets must share (log_w, steps, rounds). Per set and mode: margin = max |out/BOS - bit|,
wrong = bits with error > 0.5, bool = bits misread by boolify (1 iff within 0.02 of 1).
The eager function of the variant is checked on the first --eager messages of the first set.
"""
import argparse
import json
import time

import torch as t
import torch.nn.functional as F

import xofbench as hb
from reifier.compile.tree import TreeCompiler
from reifier.utils.format import Bits

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("refs", nargs="+")
ap.add_argument("--eager", type=int, default=9)
ap.add_argument("--modes", default="f32")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--out", default=None)
ap.add_argument("--tag", default="")
a = ap.parse_args()

refs = [t.load(r) for r in a.refs]
cfg = {(r["log_w"], r["depth"], r.get("rounds", 1)) for r in refs}
assert len(cfg) == 1, cfg
(log_w, steps, rounds), = cfg
k = hb.make_keccak(log_w, rounds)
variant = hb.default_variant if a.variant == "baseline" else hb.load_variant(a.variant)
fn, kwargs = variant(k, steps)
(argname,) = kwargs.keys()
t0 = time.time()
layers = hb.build_layers(TreeCompiler().run(fn, **kwargs))
m = hb.metrics(layers)
t1 = time.time()


def run(X, mode):
    dt = {"f32": t.float32, "f64": t.float64, "w16": t.float32, "b16": t.bfloat16}[mode]
    outs = []
    with t.inference_mode():
        for i in range(0, len(X), a.batch):
            x = X[i:i + a.batch].to(dt)
            for L in layers:
                if mode == "f32":
                    x = hb.forward([L], x)
                    continue
                ws = [L["norm"]] + [L[n].to_dense() for n in ("wg", "wv", "wo")]
                if mode in ("w16", "b16"):
                    ws = [w.to(t.bfloat16) for w in ws]
                ws = [w.to(dt) for w in ws]
                n = F.rms_norm(x, (x.size(-1),), ws[0])
                x = (F.silu(n @ ws[1].T) * (n @ ws[2].T)) @ ws[3].T
            outs.append(x.to(t.float64))
    return t.cat(outs)


res = {"variant": a.variant, "log_w": log_w, "steps": steps, "rounds": rounds,
       "depth": m["depth"], "dense": m["dense"], "sparse": m["sparse"], "hidden": m["hidden"],
       "max_width": m["max_width"], "tag": a.tag}
eager_bad = []
r0 = refs[0]
for i in range(min(a.eager, len(r0["X"]))):
    out = [int(b.activation) for b in fn(**{argname: Bits([int(v) for v in r0["X"][i].tolist()]).bitlist})]
    if out != [int(v) for v in r0["Y"][i].tolist()]:
        eager_bad.append(r0["names"][i])
res["eager_checked"], res["eager_bad"] = min(a.eager, len(r0["X"])), eager_bad
sets = {}
for path, r in zip(a.refs, refs):
    X = t.cat([t.ones(len(r["X"]), 1), r["X"]], 1)
    Y = r["Y"].to(t.float64)
    s = {"n_msgs": len(X)}
    for mode in a.modes.split(","):
        y = run(X, mode)
        assert y.size(1) - 1 == Y.size(1), (y.size(1) - 1, Y.size(1))
        rr = y[:, 1:] / y[:, :1]
        err = (rr - Y).abs()
        err = t.where(t.isnan(err), t.full_like(err, float("inf")), err)
        d = k.d
        s[mode] = {"margin": err.max().item(), "wrong": int((err > 0.5).sum()),
                   "bool": int((((rr - 1).abs() <= 0.02) != Y.bool()).sum()),
                   "per_step": [float("%.3g" % err[:, j * d:(j + 1) * d].max().item()) for j in range(steps)]}
    sets[path.rsplit("/", 1)[-1]] = s
res["sets"] = sets
worst = {}
for mode in a.modes.split(","):
    worst[mode] = {"margin": max(s[mode]["margin"] for s in sets.values()),
                   "wrong": sum(s[mode]["wrong"] for s in sets.values()),
                   "bool": sum(s[mode]["bool"] for s in sets.values())}
res["worst"] = worst
res["n_msgs"] = sum(s["n_msgs"] for s in sets.values())
res["robust_f32"] = ("f32" in worst and worst["f32"]["margin"] <= 0.01 and worst["f32"]["wrong"] == 0
                     and worst["f32"]["bool"] == 0 and not eager_bad)
res["build_s"], res["check_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
line = json.dumps(res)
print(line)
if a.out:
    with open(a.out, "a") as f:
        f.write(line + "\n")

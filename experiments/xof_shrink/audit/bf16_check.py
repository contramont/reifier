"""bfloat16 check of a compiled XOF circuit against reference outputs (ref_gen / ref_stress).
Steps are built exact (SwiGLU.from_matrix(exact=True)), as MLP_SwiGLU does for bf16 dtypes.

Modes:
  f32  float32 weights and activations (the reference run);
  w16  weights rounded to bfloat16, float32 activations ("bfloat16 weights");
  b16  weights and activations in bfloat16, as MLP_SwiGLU(dtype=t.bfloat16) runs
       (SwiGLU.forward casts its input to the layer dtype).
Readouts per mode:
  margin         max |out/BOS - bit|;
  wrong_nearest  bits with |out/BOS - bit| > 0.5;
  wrong_boolify  bits misread by reifier's boolify (1 iff within 0.02 of BOS).
A circuit is correct in a mode if both wrong counts are 0.

Usage: PYTHONPATH=<repo>/src:<xof_shrink>:<xof_shrink>/depth_low/lib/python \
  python bf16_check.py <ref.pt> module:function [--modes f32,w16,b16] [--batch 256]
"""

import argparse
import importlib
import json
import time

import torch as t
import torch.nn.functional as F

from reifier.compile.tree import TreeCompiler

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("variant", help="module:function, or 'baseline' for the threshold circuit")
ap.add_argument("--harness", default="xofbench")
ap.add_argument("--modes", default="f32,w16,b16")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--no-exact", action="store_true",
                help="build steps as for float32 (by default exact steps, as MLP_SwiGLU uses for bf16)")
a = ap.parse_args()

hb = importlib.import_module(a.harness)
ref = t.load(a.ref)
rounds = ref.get("rounds", 1)
k = hb.make_keccak(ref["log_w"], rounds) if rounds != 1 else hb.make_keccak(ref["log_w"])
if a.variant == "baseline":
    variant = hb.default_variant
else:
    mod, _, fn_name = a.variant.partition(":")
    variant = getattr(importlib.import_module(mod), fn_name)
fn, kwargs = variant(k, ref["depth"])
t0 = time.time()
tree = TreeCompiler().run(fn, **kwargs)
try:
    layers = hb.build_layers(tree, exact=not a.no_exact)
except TypeError:  # a harness copy without the exact option (overlays)
    layers = hb.build_layers(tree)
m = hb.metrics(layers)
X = t.cat([t.ones(len(ref["X"]), 1), ref["X"]], 1)
Y = ref["Y"]


def dense(w: t.Tensor) -> t.Tensor:
    return w.to_dense() if w.is_sparse else w


def run(mode: str) -> dict:
    dt = t.bfloat16 if mode == "b16" else t.float32
    outs = []
    with t.inference_mode():
        for i in range(0, len(X), a.batch):
            x = X[i : i + a.batch].to(dt)
            for L in layers:
                ws = [dense(L[n]) for n in ("norm", "wg", "wv", "wo")]
                if mode != "f32":
                    ws = [w.to(t.bfloat16) for w in ws]
                ws = [w.to(dt) for w in ws]
                n = F.rms_norm(x, (x.size(-1),), ws[0])
                x = (F.silu(n @ ws[1].T) * (n @ ws[2].T)) @ ws[3].T
            outs.append(x.float())
    y = t.cat(outs)
    r = y[:, 1:] / y[:, :1]
    err = (r - Y).abs()
    err = t.where(t.isnan(err), t.full_like(err, float("inf")), err)
    ones = (r - 1).abs() <= 0.02
    return {"margin": err.max().item(), "wrong_nearest": int((err > 0.5).sum()),
            "wrong_boolify": int((ones != Y.bool()).sum())}


res = {"variant": a.variant, "log_w": ref["log_w"], "steps": ref["depth"], "rounds": rounds,
       "depth": m["depth"], "dense": m["dense"], "sparse": m["sparse"], "n_msgs": len(X)}
for mode in a.modes.split(","):
    res[mode] = run(mode)
    res[mode]["correct"] = res[mode]["wrong_nearest"] == 0 and res[mode]["wrong_boolify"] == 0
res["secs"] = round(time.time() - t0, 1)
print(json.dumps(res))

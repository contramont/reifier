"""Per-layer cleanness of a compiled circuit whose features should all be integers x BOS:
max over messages and features of the distance of feature/BOS to the nearest integer,
in float32, w16 (bfloat16 weights) and b16 (bfloat16 weights and activations), plus the max
|gate pre-activation| / N (N = normalized BOS). Also the max |f32 - b16| per layer.

Usage: PYTHONPATH=... python cleanness.py ref.pt module:function|baseline [--n 64]
"""
import argparse
import importlib

import torch as t
import torch.nn.functional as F

import xofbench as xb
from reifier.compile.tree import TreeCompiler

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("variant")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--c", type=int, default=4)
ap.add_argument("--q", type=int, default=8)
a = ap.parse_args()
ref = t.load(a.ref)
k = xb.make_keccak(ref["log_w"], ref.get("rounds", 1))
if a.variant == "baseline":
    var = xb.default_variant
else:
    m, _, f = a.variant.partition(":")
    var = getattr(importlib.import_module(m), f)
fn, kw = var(k, ref["depth"])
layers = xb.build_layers(TreeCompiler().run(fn, **kw), c=a.c, q=a.q)
X = t.cat([t.ones(len(ref["X"]), 1), ref["X"]], 1)[: a.n]


def dense(w):
    return w.to_dense() if w.is_sparse else w


xs = {m: X.clone() for m in ("f32", "w16", "b16")}
with t.inference_mode():
    for li, L in enumerate(layers):
        row = []
        for mode in ("f32", "w16", "b16"):
            dt = t.bfloat16 if mode == "b16" else t.float32
            ws = [dense(L[n]) for n in ("norm", "wg", "wv", "wo")]
            if mode != "f32":
                ws = [w.to(t.bfloat16) for w in ws]
            ws = [w.to(dt) for w in ws]
            x = xs[mode].to(dt)
            n = F.rms_norm(x, (x.size(-1),), ws[0])
            g = n @ ws[1].T
            y = (F.silu(g) * (n @ ws[2].T)) @ ws[3].T
            xs[mode] = y.float()
            r = xs[mode][:, 1:] / xs[mode][:, :1]
            dist = (r - r.round()).abs().max().item()  # all features are integers (bits, q, packed)
            gmax = (g.float().abs() / n[:, :1].float().abs()).max().item()
            row.append(f"{mode}:{dist:.1e}")
        diff = (xs["b16"][:, 1:] / xs["b16"][:, :1] - xs["f32"][:, 1:] / xs["f32"][:, :1]).abs().max().item()
        print(f"L{li + 1:>2} {L['in']:>5}->{L['hidden']:>5}->{L['out']:>5} max|g|/N={gmax:7.1f} "
              + " ".join(row) + f" |b16-f32|={diff:.1e}")

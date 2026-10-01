"""Which weights of a compiled XOF circuit are not bfloat16-representable (bf16 keeps 8
significant bits), per layer and matrix, in unit terms: gate = wg / (c*q), value = wv / v,
out = wo * (c*q*v) (xofbench builds with c = 4, q = 8, v = 4/q: 32, 0.5, 16).

A circuit whose weights are all representable computes exactly the same numbers in mode w16
(weights rounded to bfloat16, float32 activations) as in float32, so its w16 margin is its
float32 margin.

Usage: PYTHONPATH=... python bf16_weights.py LOG_W STEPS module:function [--top 8]
"""

import argparse
import collections
import importlib
import json

import torch as t

import xofbench as xb
from reifier.compile.tree import TreeCompiler


def dense(w):
    return w.to_dense() if w.is_sparse else w


def nonrep(w: t.Tensor) -> t.Tensor:
    """mask of entries that bfloat16 rounds"""
    return w.to(t.bfloat16).to(t.float32) != w


def analyse(layers, top=8):
    rows = []
    for i, L in enumerate(layers):
        r = {"layer": i + 1, "in": L["in"], "hidden": L["hidden"], "out": L["out"]}
        for name, unit in (("wg", 32.0), ("wv", 0.5), ("wo", 1 / 16)):
            w = dense(L[name])
            m = nonrep(w)
            n = int(m.sum())
            r[name] = n
            if n:
                vals = (w[m] / unit).double()
                rel = ((w[m].to(t.bfloat16).float() - w[m]) / w[m]).abs().max().item()
                r[name + "_relerr"] = rel
                cnt = collections.Counter(round(v, 6) for v in vals.tolist())
                r[name + "_vals"] = cnt.most_common(top)
                # hidden units touched
                if name == "wo":
                    r["units_touched"] = int(m.any(0).sum())
                else:
                    r[name + "_units"] = int(m.any(1).sum())
        rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_w", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("variant")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    k = xb.make_keccak(a.log_w, a.rounds)
    if a.variant == "baseline":
        variant = xb.default_variant
    else:
        mod, _, fn_name = a.variant.partition(":")
        variant = getattr(importlib.import_module(mod), fn_name)
    fn, kwargs = variant(k, a.steps)
    layers = xb.build_layers(TreeCompiler().run(fn, **kwargs))
    rows = analyse(layers, a.top)
    tot = {n: sum(r[n] for r in rows) for n in ("wg", "wv", "wo")}
    print(json.dumps({"variant": a.variant, "log_w": a.log_w, **xb.metrics(layers), "nonrep": tot,
                      "widths": None}))
    for r in rows:
        print(json.dumps(r))

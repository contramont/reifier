"""per-layer (in, hidden, out, dense, sparse) of a variant, as built by the harness"""
import sys, importlib, json
import xofbench as hb
from reifier.compile.tree import TreeCompiler
mod, _, fn = sys.argv[1].partition(":")
k = hb.make_keccak(6)
f, kw = getattr(importlib.import_module(mod), fn)(k, 3)
layers = hb.build_layers(TreeCompiler().run(f, **kw))
for L in layers:
    print(json.dumps({k_: L[k_] for k_ in ("in", "hidden", "out", "dense", "sparse")}))

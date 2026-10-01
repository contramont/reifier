"""End to end through the repo pipeline: Compiler(mlp_dtype=...).run(fn) -> MLP_SwiGLU, then
the module's own forward on [BOS, message] in that dtype, read relative to BOS.
Usage: e2e_compiler.py ref.pt module:function|baseline [dtype: bfloat16|float32] [exact: 0|1|auto]"""
import importlib
import json
import sys

import torch as t

import xofbench as xb
from reifier.compile.tree import TreeCompiler
from reifier.tensors.compilation import Compiler
from reifier.tensors.matrices import Matrices
from reifier.tensors.swiglu import MLP_SwiGLU

ref = t.load(sys.argv[1])
spec = sys.argv[2]
dtype = getattr(t, sys.argv[3]) if len(sys.argv) > 3 else t.bfloat16
exact = {"0": False, "1": True, "auto": None}[sys.argv[4] if len(sys.argv) > 4 else "auto"]
k = xb.make_keccak(ref["log_w"], ref.get("rounds", 1))
if spec == "baseline":
    var = xb.default_variant
else:
    m, _, f = spec.partition(":")
    var = getattr(importlib.import_module(m), f)
fn, kw = var(k, ref["depth"])
comp = Compiler(mlp_dtype=dtype)
tree = comp.get_tree(fn, **kw)
mlp = MLP_SwiGLU.from_matrices(Matrices.from_graph(tree), c=comp.c, q=comp.q, dtype=dtype, exact=exact)
X = t.cat([t.ones(len(ref["X"]), 1), ref["X"]], 1)
with t.inference_mode():
    y = mlp(X).float()
r = y[:, 1:] / y[:, :1]
err = (r - ref["Y"]).abs()
ones = (r - 1).abs() <= 0.02
print(json.dumps({"variant": spec, "dtype": str(dtype), "exact": exact, "log_w": ref["log_w"],
                  "steps": ref["depth"], "rounds": ref.get("rounds", 1), "n_msgs": len(X),
                  "params": sum(p.numel() for p in mlp.parameters()), "margin": err.max().item(),
                  "wrong_nearest": int((err > 0.5).sum()),
                  "wrong_boolify": int((ones != ref["Y"].bool()).sum())}))

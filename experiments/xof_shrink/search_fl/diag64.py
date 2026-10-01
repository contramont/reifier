"""float32 vs float64 forward on the audit messages: is the margin a float32 effect?"""
import sys, torch as t
import torch.nn.functional as F
import xofbench as hb
from reifier.compile.tree import TreeCompiler
import importlib
ref = t.load(sys.argv[1]); spec = sys.argv[2]
k = hb.make_keccak(ref["log_w"])
mod, _, fn_name = spec.partition(":")
fn, kwargs = getattr(importlib.import_module(mod), fn_name)(k, ref["depth"])
tree = TreeCompiler().run(fn, **kwargs)
layers = hb.build_layers(tree)
X, Y = ref["X"], ref["Y"]
x0 = t.cat([t.ones(len(X), 1), X], 1)
def fwd(layers, x, dt):
    for L in layers:
        x = F.rms_norm(x, (x.size(-1),), L["norm"].to(dt))
        g = t.sparse.mm(L["wg"].to(dt), x.T).T
        v = t.sparse.mm(L["wv"].to(dt), x.T).T
        x = t.sparse.mm(L["wo"].to(dt), (F.silu(g) * v).T).T
    return x
with t.inference_mode():
    for dt in (t.float32, t.float64):
        y = fwd(layers, x0.to(dt), dt)
        err = (y[:, 1:] / y[:, :1] - Y.to(dt)).abs()
        pm = err.max(1).values
        print(dt, "margin", pm.max().item(), "worst", ref["names"][int(pm.argmax())],
              "worst output index", int(err.max(0).values.argmax()))

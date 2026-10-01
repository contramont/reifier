"""per-output-block errors of a compiled variant on selected reference messages, plus the
float64 forward (same weights) to separate float32 rounding from the silu approximation"""
import sys, json, torch as t, importlib
import xofbench as hb
from reifier.compile.tree import TreeCompiler
ref = t.load(sys.argv[1]); var = sys.argv[2]
mod, _, fn = var.partition(":")
k = hb.make_keccak(ref["log_w"])
f, kw = getattr(importlib.import_module(mod), fn)(k, ref["depth"])
tree = TreeCompiler().run(f, **kw)
layers = hb.build_layers(tree)
X, Y = ref["X"], ref["Y"]
names = ref["names"]
def fwd(layers, x, dt):
    import torch.nn.functional as F
    for L in layers:
        x = F.rms_norm(x, (x.size(-1),), L["norm"].to(dt))
        g = t.sparse.mm(L["wg"].to(dt), x.T).T
        v = t.sparse.mm(L["wv"].to(dt), x.T).T
        x = t.sparse.mm(L["wo"].to(dt), (F.silu(g) * v).T).T
    return x
with t.inference_mode():
    xin = t.cat([t.ones(len(X), 1), X], 1)
    for dt in (t.float32, t.float64):
        y = fwd(layers, xin.to(dt), dt)
        err = (y[:, 1:] / y[:, :1] - Y.to(dt)).abs()
        per = err.max(1).values
        order = per.argsort(descending=True)[:6]
        print(str(dt), "max", per.max().item())
        for i in order.tolist():
            e = err[i]
            blocks = [e[0:224].max().item(), e[224:448].max().item(), e[448:672].max().item()]
            print("  ", names[i], round(per[i].item(), 5), "by digest", [round(b, 5) for b in blocks], "argmax", int(e.argmax()))

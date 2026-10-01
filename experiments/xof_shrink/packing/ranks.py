"""Interface slack: for each layer, the rank of the linear forms its units read (rows of
wg and wv over the input features) versus the input width."""
import sys, importlib, torch as t
import xofbench as hb
from reifier.compile.tree import TreeCompiler
spec = sys.argv[1]
mod, _, fn = spec.partition(":")
k = hb.make_keccak(6)
f, kw = getattr(importlib.import_module(mod), fn)(k, 3)
tree = TreeCompiler().run(f, **kw)
layers = hb.build_layers(tree)
for i, L in enumerate(layers):
    wg, wv = L["wg"].to_dense(), L["wv"].to_dense()
    M = t.cat([wg, wv], 0).double()
    r = int(t.linalg.matrix_rank(M))
    # forms read by gates only / values only
    rg = int(t.linalg.matrix_rank(wg.double()))
    print(f"L{i+1} in={L['in']} hidden={L['hidden']} out={L['out']} rank(read forms)={r} rank(gates)={rg}", flush=True)

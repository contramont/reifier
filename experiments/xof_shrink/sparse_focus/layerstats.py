"""Per-layer nonzero breakdown of a variant (norm, wg, wv, wo) + unit statistics."""
import sys, json, time
import torch as t
import xofbench as hb
from reifier.compile.tree import TreeCompiler

spec = sys.argv[1]
log_w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
k = hb.make_keccak(log_w)
fn, kwargs = hb.load_variant(spec)(k, 3)
t0 = time.time()
tree = TreeCompiler().run(fn, **kwargs)
layers = hb.build_layers(tree)
tot = {"norm": 0, "wg": 0, "wv": 0, "wo": 0}
for i, L in enumerate(layers):
    nn = {n: int(t.count_nonzero(L[n].to_dense() if n != "norm" else L[n])) for n in tot}
    for n in tot:
        tot[n] += nn[n]
    wg = L["wg"].to_dense(); wv = L["wv"].to_dense(); wo = L["wo"].to_dense()
    # gate rows reading only BOS, value rows reading only BOS
    g_nnz = (wg != 0).sum(1); v_nnz = (wv != 0).sum(1); o_nnz = (wo != 0).sum(0)
    hist = lambda x: dict(sorted(((int(a), int(b)) for a, b in zip(*t.unique(x, return_counts=True)))))
    print(f"L{i}: in {L['in']} hid {L['hidden']} out {L['out']} | norm {nn['norm']} wg {nn['wg']} wv {nn['wv']} wo {nn['wo']} | sparse {L['sparse']} dense {L['dense']}")
    print("   gate-nnz hist", hist(g_nnz))
    print("   val-nnz hist", hist(v_nnz))
    print("   out-nnz hist", hist(o_nnz))
print("total", tot, sum(tot.values()), "dense", sum(L["dense"] for L in layers), "t", round(time.time()-t0,1))

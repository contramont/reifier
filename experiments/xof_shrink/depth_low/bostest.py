import torch as t, torch.nn.functional as F
import xofbench as hb, dl3
from reifier.compile.tree import TreeCompiler
k = hb.make_keccak(1)
fn, kw = dl3.d4x(k, 3)
layers = hb.build_layers(TreeCompiler().run(fn, **kw))
for name, bits in (("zeros", [0] * k.msg_len), ("ones", [1] * k.msg_len)):
    x = t.tensor([[1.0] + bits])
    for i, L in enumerate(layers):
        r = x.pow(2).mean().sqrt().item()
        xn = F.rms_norm(x, (x.size(-1),), L["norm"])
        g = t.sparse.mm(L["wg"], xn.T).T; v = t.sparse.mm(L["wv"], xn.T).T
        x = t.sparse.mm(L["wo"], (F.silu(g) * v).T).T
        print(name, "layer", i, "in-rms", round(r, 4), "BOS out", round(x[0, 0].item(), 5), "max feat", round(x[0, 1:].abs().max().item(), 5))

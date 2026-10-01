"""The harness must reproduce MLP_SwiGLU.from_matrices exactly on small configs."""
import sys
import torch as t
import xofbench as xb
from reifier.compile.tree import TreeCompiler
from reifier.tensors.compilation import Compiler

variant = xb.load_variant(sys.argv[1]) if len(sys.argv) > 1 else xb.default_variant
for log_w in [0, 1, 2]:
    k = xb.make_keccak(log_w)
    fn, kwargs = variant(k, 3)
    tree = TreeCompiler().run(fn, **kwargs)
    layers = xb.build_layers(tree)
    mlp = Compiler().get_mlp_from_tree(tree)
    params = list(mlp.parameters())
    dense = sum(p.numel() for p in params)
    sparse = sum(int(t.count_nonzero(p)) for p in params)
    m = xb.metrics(layers)
    same_w = all(
        t.equal(L[key].to_dense(), getattr(mlp.layers[i], key).weight)
        for i, L in enumerate(layers) for key in ["wg", "wv", "wo"]
    ) and all(t.equal(L["norm"], mlp.layers[i].norm.weight) for i, L in enumerate(layers))
    x = t.randint(0, 2, (8, k.msg_len + 1)).float(); x[:, 0] = 1
    with t.inference_mode():
        diff = (xb.forward(layers, x) - mlp(x)).abs().max().item()
    v = xb.verify(layers, fn, kwargs, k)
    print(f"log_w={log_w} depth {m['depth']}=={len(mlp.layers)} dense {m['dense']}=={dense} "
          f"sparse {m['sparse']}=={sparse} weights_equal={same_w} fwd_diff={diff:.1e} verify={v}")

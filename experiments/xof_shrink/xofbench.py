"""Benchmark: 1-round Keccak XOF compiled to a SwiGLU MLP (reifier).

Metrics of the final MLP_SwiGLU (the same weights MLP_SwiGLU.from_matrices builds):
  depth  = number of SwiGLU layers
  dense  = total parameter count (numel of norm, wg, wv, wo)
  sparse = total nonzero parameters
Correctness: outputs read relative to the BOS feature (as infer_bits_bos does) must
match eager xof on random messages; margin = max |out/BOS - bit|, must be < 0.02.

Layers are built one at a time with the repo's own Matrices/SwiGLU code and kept as
sparse tensors, so full-width (log_w=6) circuits fit in memory.

Usage (from a reifier checkout):
  PYTHONPATH=$PWD/src python xofbench.py --log-w 6 --depth 3 [--variant module:function]
A variant is a function (k: Keccak, depth: int) -> (fn, kwargs) returning the traced
circuit function and its dummy inputs; the default is the repo's own xof.
"""

import argparse
import importlib
import json
import os
import random
import sys
import time

import torch as t
import torch.nn.functional as F

from reifier.examples.keccak import Keccak, xof
from reifier.utils.format import Bits
from reifier.compile.tree import TreeCompiler
from reifier.tensors.matrices import Matrices
from reifier.tensors.swiglu import SwiGLU

# bfloat16-exact weights (bf16_units.py): "u" rescales gated units, "ub" also splits biases;
# "1": out columns split at one level (16 bits, one extra unit each) instead of two
BF16 = os.environ.get("XOF_BF16", "")
# exact steps (SwiGLU.from_matrix(exact=True), main since fdc78d6): what MLP_SwiGLU uses for
# 16-bit dtypes; XOF_EXACT=1 turns them on here (audit/bf16_check.py always does)
EXACT = os.environ.get("XOF_EXACT", "0") == "1"

# SHA3-like proportions: capacity c = 448/1600 of the state, digest d = c/2
CAPACITY = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}


def make_keccak(log_w: int, rounds: int = 1) -> Keccak:
    """rounds: Keccak rounds per XOF step (k.n); builders written for 1 round reject others"""
    return Keccak(log_w=log_w, n=rounds, c=CAPACITY[log_w], pad_char="_")


def default_variant(k: Keccak, depth: int):
    def xof_fn(msg: list) -> list:
        return [bit for digest in xof(msg, depth, k) for bit in digest]

    return xof_fn, {"msg": Bits("0" * k.msg_len).bitlist}


def build_layers(tree, c: int = 4, q: int = 8, exact: bool | None = None):
    """Sparse copies of the weights of MLP_SwiGLU.from_matrices(Matrices.from_graph(tree))"""
    layers = []
    for level_out, (out_w, in_w) in zip(tree.levels[1:], tree.shapes):
        w, b = Matrices.layer_to_params(level_out, in_w, out_w)
        m = Matrices.fold_bias(w.to_dense(), b, dtype=t.int)
        units = None
        if hasattr(Matrices, "layer_to_units"):
            units = Matrices.layer_to_units(level_out, in_w)
            if units is not None and "u" in BF16:
                import bf16_units
                units, _ = bf16_units.rescale_units(units, with_bias=not ("b" in BF16 or "B" in BF16),
                                                    levels=1 if "1" in BF16 else 2)
        kwargs = {} if units is None else {"units": units}
        with t.no_grad():
            if exact if exact is not None else EXACT:
                kwargs["exact"] = True
            layer = SwiGLU.from_matrix(m, c=c, q=q, has_bias=False, **kwargs)
            params = dict(layer.named_parameters())
            layers.append({
                "norm": params["norm.weight"].detach().clone(),
                "wg": params["wg.weight"].detach().to_sparse(),
                "wv": params["wv.weight"].detach().to_sparse(),
                "wo": params["wo.weight"].detach().to_sparse(),
                "dense": sum(p.numel() for p in params.values()),
                "sparse": sum(int(t.count_nonzero(p)) for p in params.values()),
                "in": m.size(1), "out": m.size(0), "hidden": params["wg.weight"].size(0),
            })
        del layer, params, m, w
    if "b" in BF16:
        import bf16_units
        bf16_units.split_bias(layers)
    return layers


def forward(layers, x: t.Tensor) -> t.Tensor:
    for L in layers:
        x = F.rms_norm(x, (x.size(-1),), L["norm"])
        g = t.sparse.mm(L["wg"], x.T).T
        v = t.sparse.mm(L["wv"], x.T).T
        x = t.sparse.mm(L["wo"], (F.silu(g) * v).T).T
    return x


def metrics(layers) -> dict:
    return {
        "depth": len(layers),
        "dense": sum(L["dense"] for L in layers),
        "sparse": sum(L["sparse"] for L in layers),
        "hidden": sum(L["hidden"] for L in layers),
        "max_width": max(max(L["in"], L["out"], L["hidden"]) for L in layers),
        "widths": [(L["in"], L["hidden"], L["out"]) for L in layers],
    }


def verify(layers, fn, kwargs, k: Keccak, n_msgs: int = 16, seed: int = 0) -> dict:
    rng = random.Random(seed)
    (name,) = kwargs.keys()
    msgs = [[rng.randint(0, 1) for _ in range(k.msg_len)] for _ in range(n_msgs)]
    expected = t.tensor(
        [[int(b.activation) for b in fn(**{name: Bits(m).bitlist})] for m in msgs],
        dtype=t.float32,
    )
    with t.inference_mode():
        y = forward(layers, t.tensor([[1] + m for m in msgs], dtype=t.float32))
    err = (y[:, 1:] / y[:, :1] - expected).abs()
    return {
        "margin": err.max().item(),
        "wrong_bits": int((err > 0.5).sum()),
        "ok": bool(err.max().item() < 0.02 and y.size(1) - 1 == expected.size(1)),
    }


def run(log_w: int, depth: int, variant=default_variant, n_msgs: int = 16, c=4, q=8, rounds=1):
    k = make_keccak(log_w, rounds)
    fn, kwargs = variant(k, depth)
    t0 = time.time()
    tree = TreeCompiler().run(fn, **kwargs)
    t1 = time.time()
    layers = build_layers(tree, c=c, q=q)
    t2 = time.time()
    res = {"log_w": log_w, "xof_depth": depth, "rounds": rounds, **metrics(layers)}
    res.update(verify(layers, fn, kwargs, k, n_msgs=n_msgs))
    res["compile_s"], res["build_s"] = round(t1 - t0, 1), round(t2 - t1, 1)
    return res


def load_variant(spec: str):
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-w", type=int, default=6)
    ap.add_argument("--depth", type=int, default=3, help="XOF steps")
    ap.add_argument("--rounds", type=int, default=1, help="Keccak rounds per XOF step")
    ap.add_argument("--variant", default=None, help="module:function")
    ap.add_argument("--msgs", type=int, default=16)
    ap.add_argument("--widths", action="store_true")
    ap.add_argument("--c", type=int, default=4, help="step steepness (Compiler.c)")
    ap.add_argument("--q", type=int, default=8, help="silu sharpness (Compiler.q)")
    a = ap.parse_args()
    variant = load_variant(a.variant) if a.variant else default_variant
    res = run(a.log_w, a.depth, variant, a.msgs, c=a.c, q=a.q, rounds=a.rounds)
    if not a.widths:
        res.pop("widths")
    print(json.dumps(res))
    sys.stdout.flush()

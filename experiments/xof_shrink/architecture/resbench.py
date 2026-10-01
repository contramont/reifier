"""Benchmark for residual programs (compile.residual): 1-round Keccak XOF.

Copy of xofbench adapted to the residual architecture MLP_ResSwiGLU:
  embed (linear) -> layers in schedule order, x + SwiGLU(x) (tied layers repeat)
  -> readout (linear, of the streams after its tap positions).
Metrics:
  depth  = number of SwiGLU layer applications (len(schedule))
  dense  = numel of all unique parameters: embed, layers (norm, wg, wv, wo), readouts
  sparse = nonzeros of the same
  hidden = hidden units summed over the schedule (unique: over the unique layers)
Correctness: outputs relative to the BOS output must match the reference xof (the
unpatched keccak module) on random messages, within 0.02.

  PYTHONPATH=<repo>/src:<this dir>:<xof dir> python resbench.py --log-w 6 --variant resvariants:tied_regs
"""

import argparse
import importlib
import json
import random
import sys
import time

import torch as t
import torch.nn.functional as F

from reifier.examples.keccak import Keccak, xof
from reifier.utils.format import Bits

CAPACITY = {0: 10, 1: 20, 2: 28, 3: 56, 4: 112, 5: 224, 6: 448}


def make_keccak(log_w: int) -> Keccak:
    return Keccak(log_w=log_w, n=1, c=CAPACITY[log_w], pad_char="_")


def build(program) -> dict:
    layers = []
    for layer in program.layers:
        wg, wv, wo = layer.matrices()
        alpha = None
        if any(L.drop for L in program.layers):  # gated residual
            alpha = t.ones(program.d)
            alpha[layer.drop] = 0
        layers.append({"wg": wg, "wv": wv, "wo": wo, "h": layer.h, "alpha": alpha})
    return {"p": program, "layers": layers}


def forward(model, x: t.Tensor) -> t.Tensor:
    p = model["p"]
    x = x @ p.embed.T
    out = t.zeros(x.size(0), p.n_out)
    norm = t.ones(p.d)
    for pos, i in enumerate(p.schedule):
        L = model["layers"][i]
        xn = F.rms_norm(x, (p.d,), norm)
        g = t.sparse.mm(L["wg"], xn.T).T
        v = t.sparse.mm(L["wv"], xn.T).T
        skip = x if L["alpha"] is None else x * L["alpha"]
        x = skip + t.sparse.mm(L["wo"], (F.silu(g) * v).T).T
        for tap in p.readout:
            if tap.pos == pos:
                y = x @ tap.matrix.T
                keep = [j for j, row in enumerate(tap.rows) if row >= 0]
                out[:, [tap.rows[j] for j in keep]] += y[:, keep]
    return out


def metrics(model) -> dict:
    p = model["p"]
    readouts = list({id(tap.matrix): tap.matrix for tap in p.readout}.values())  # tied once
    dense = p.embed.numel() + sum(r.numel() for r in readouts)
    sparse = int(t.count_nonzero(p.embed)) + sum(int(t.count_nonzero(r)) for r in readouts)
    for L in model["layers"]:
        dense += p.d + L["h"] * p.d * 3  # norm, wg, wv, wo
        sparse += p.d + sum(int(t.count_nonzero(L[k].values())) for k in ["wg", "wv", "wo"])
        if L["alpha"] is not None:  # gated residual
            dense += p.d
            sparse += int(t.count_nonzero(L["alpha"]))
    return {
        "depth": len(p.schedule),
        "dense": dense,
        "sparse": sparse,
        "hidden": sum(model["layers"][i]["h"] for i in p.schedule),
        "hidden_unique": sum(L["h"] for L in model["layers"]),
        "d": p.d,
        "embed": list(p.embed.shape),
        "readouts": [list(r.shape) for r in readouts],
        "widths": [(p.d, L["h"], p.d) for L in model["layers"]],
        "schedule": p.schedule,
        "taps": [tap.pos for tap in p.readout],
    }


def verify(model, k: Keccak, depth: int, n_msgs: int = 16, seed: int = 0) -> dict:
    rng = random.Random(seed)
    msgs = [[rng.randint(0, 1) for _ in range(k.msg_len)] for _ in range(n_msgs)]
    expected = t.tensor(
        [[int(b.activation) for dg in xof(Bits(m).bitlist, depth, k) for b in dg] for m in msgs],
        dtype=t.float32,
    )
    with t.inference_mode():
        y = forward(model, t.tensor([[1] + m for m in msgs], dtype=t.float32))
    err = (y[:, 1:] / y[:, :1] - expected).abs()
    return {
        "margin": err.max().item(),
        "wrong_bits": int((err > 0.5).sum()),
        "ok": bool(err.max().item() < 0.02 and y.size(1) - 1 == expected.size(1)),
    }


def run(log_w: int, depth: int, variant, n_msgs: int = 16):
    k = make_keccak(log_w)
    t0 = time.time()
    program = variant(k, depth)
    t1 = time.time()
    model = build(program)
    t2 = time.time()
    res = {"log_w": log_w, "xof_depth": depth, **metrics(model)}
    res.update(verify(model, k, depth, n_msgs=n_msgs))
    res["compile_s"], res["build_s"] = round(t1 - t0, 1), round(t2 - t1, 1)
    return res


def load_variant(spec: str):
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-w", type=int, default=6)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--variant", required=True, help="module:function")
    ap.add_argument("--msgs", type=int, default=16)
    ap.add_argument("--widths", action="store_true")
    a = ap.parse_args()
    res = run(a.log_w, a.depth, load_variant(a.variant), a.msgs)
    res = {"variant": a.variant.partition(":")[2], **res}
    if not a.widths:
        for key in ["widths", "schedule"]:
            res.pop(key)
    print(json.dumps(res))
    sys.stdout.flush()

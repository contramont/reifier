"""Non-residual comparison points (MLP_SwiGLU layers, 0/1 bits, as xofbench builds them)
to separate the gains of the architecture changes:
  lin:  a linear embedding (msg -> initial state, constants from BOS) and a linear
        readout instead of the copy layers L1 and L8; layers untied.
  tied: as lin, and one block of layers (an XOF step on state + digest shift register)
        applied depth times: weight tying without residual connections.
The circuit is glu_chi_iota's (compact xor theta, chi+iota as one unit).
  PYTHONPATH=<repo>/src:<this dir>:<xof dir> python nonres.py --log-w 6 --mode tied
"""

import argparse
import json
import random
import sys
import time

import torch as t
import torch.nn.functional as F

import resvariants as rv
import xofbench as xb
from reifier.compile.tree import TreeCompiler
from reifier.examples.keccak import xof
from reifier.neurons.core import Unit, const
from reifier.tensors.matrices import Matrices
from reifier.utils.format import Bits


def glu_copies(tree) -> None:
    """copies as one gated unit max(0, x) * 1 instead of a two-unit step"""
    from dataclasses import replace
    from reifier.compile.levels import Level, Origin, Parent
    from reifier.neurons.core import Unit

    levels = list(tree.levels)
    for i, level in enumerate(levels[1:], start=1):
        origins = []
        for o in level.origins:
            if not o.units and len(o.incoming) == 1 and o.incoming[0].weight == 1 and o.bias == -1:
                o = Origin(o.index, (Parent(o.incoming[0].index, 0),), -1, (Unit((1,), 0, (0,), 1),))
            origins.append(o)
        levels[i] = Level(tuple(origins))
    object.__setattr__(tree, "levels", tuple(levels))


def fold_output_copies(tree, layers):
    """If the last layer only copies (reorders) the level below, permute the outputs of
    the layer below instead and drop it"""
    last = tree.levels[-1]
    order = []
    for o in last.origins:
        copy = len(o.incoming) == 1 and o.bias == -1 and (
            (not o.units and o.incoming[0].weight == 1) or o.units == (Unit((1,), 0, (0,), 1),))
        if not copy:
            return layers
        order.append(o.incoming[0].index)
    L = layers[-2]
    rows = t.tensor([0] + [i + 1 for i in order])  # BOS first
    wo = L["wo"].to_dense()[rows]
    L = dict(L, wo=wo.to_sparse(), out=len(rows))
    L["dense"] = L["norm"].numel() + L["wg"].size(0) * L["in"] * 2 + wo.numel()
    L["sparse"] = int(t.count_nonzero(L["norm"])) + sum(
        int(t.count_nonzero(L[k].to_dense())) for k in ["wg", "wv", "wo"]
    )
    return layers[:-2] + [L]


def embed(k, n_state: int, n_reg: int) -> t.Tensor:
    """[1, msg] -> [1, state, regs] with 0/1 bits"""
    inputs = const("0" * k.msg_len)
    state = rv.initial_state(k, inputs)
    index = {b.uid: i for i, b in enumerate(inputs)}
    e = t.zeros(1 + n_state + n_reg, 1 + k.msg_len)
    e[0, 0] = 1
    for s, b in enumerate(state, start=1):
        if b.uid in index:
            e[s, 1 + index[b.uid]] = 1
        else:
            e[s, 0] = int(b.activation)
    return e


def build(k, depth: int, mode: str, theta: str = "compact", copies: str = "step", taps: bool = False):
    b, d = k.b, k.d

    def tree_layers(fn, n):
        tree = TreeCompiler().run(fn, x=const("0" * n))
        if copies == "glu":
            glu_copies(tree)
        return fold_output_copies(tree, xb.build_layers(tree))

    if mode == "lin":
        def fn(x):
            state, digests = x, []
            for _ in range(depth):
                state = rv.hash_state(k, state, theta, False)
                digests += state[:d]
            return digests
        layers = tree_layers(fn, b)
        e = embed(k, b, 0)
        u = t.eye(1 + depth * d)
        return {"embed": e, "layers": layers, "schedule": list(range(len(layers))), "readout": [(len(layers) - 1, u)]}
    n_reg = 0 if taps else depth - 1

    def step(x):
        state, reg = x[:b], x[b:]
        new = rv.hash_state(k, state, theta, False)
        return new + (state[:d] + reg[: d * (n_reg - 1)] if n_reg else [])

    layers = tree_layers(step, b + d * n_reg)
    n = len(layers)
    e = embed(k, b, d * n_reg)
    schedule = [i for _ in range(depth) for i in range(n)]
    if taps:  # digest k from the state after step k
        readout = []
        for j in range(depth):
            u = t.zeros(1 + depth * d, 1 + b)
            u[0, 0] = int(j == depth - 1)
            for i in range(d):
                u[1 + j * d + i, 1 + i] = 1
            readout.append(((j + 1) * n - 1, u))
        return {"embed": e, "layers": layers, "schedule": schedule, "readout": readout}
    u = t.zeros(1 + depth * d, 1 + b + d * n_reg)  # [1, D_(n_reg), ..., D_1, state[:d]]
    u[0, 0] = 1
    for i in range(n_reg):  # digest i+1 is in register n_reg - i
        for j in range(d):
            u[1 + i * d + j, 1 + b + (n_reg - 1 - i) * d + j] = 1
    for j in range(d):
        u[1 + n_reg * d + j, 1 + j] = 1
    return {"embed": e, "layers": layers, "schedule": schedule, "readout": [(len(schedule) - 1, u)]}


def forward(m, x):
    x = x @ m["embed"].T
    taps = dict(m["readout"])
    out = 0
    for pos, i in enumerate(m["schedule"]):
        L = m["layers"][i]
        x = F.rms_norm(x, (x.size(-1),), L["norm"])
        g = t.sparse.mm(L["wg"], x.T).T
        v = t.sparse.mm(L["wv"], x.T).T
        x = t.sparse.mm(L["wo"], (F.silu(g) * v).T).T
        if pos in taps:
            out = out + x @ taps[pos].T
    return out


def run(log_w, depth, mode, n_msgs=16, **kw):
    k = xb.make_keccak(log_w)
    t0 = time.time()
    m = build(k, depth, mode, **kw)
    t1 = time.time()
    e, us = m["embed"], [u for _, u in m["readout"]]
    name = "_".join([f"nonres_{mode}", kw.get("theta", "compact"), kw.get("copies", "step")] + (["taps"] if kw.get("taps") else []))
    res = {
        "variant": name, "log_w": log_w, "xof_depth": depth,
        "depth": len(m["schedule"]),
        "dense": e.numel() + sum(u.numel() for u in us) + sum(L["dense"] for L in m["layers"]),
        "sparse": int(t.count_nonzero(e)) + sum(int(t.count_nonzero(u)) for u in us) + sum(L["sparse"] for L in m["layers"]),
        "hidden": sum(m["layers"][i]["hidden"] for i in m["schedule"]),
        "hidden_unique": sum(L["hidden"] for L in m["layers"]),
        "widths": [(L["in"], L["hidden"], L["out"]) for L in m["layers"]],
    }
    rng = random.Random(0)
    msgs = [[rng.randint(0, 1) for _ in range(k.msg_len)] for _ in range(n_msgs)]
    expected = t.tensor(
        [[int(b.activation) for dg in xof(Bits(mm).bitlist, depth, k) for b in dg] for mm in msgs],
        dtype=t.float32,
    )
    with t.inference_mode():
        y = forward(m, t.tensor([[1] + mm for mm in msgs], dtype=t.float32))
    err = (y[:, 1:] / y[:, :1] - expected).abs()
    res.update(margin=err.max().item(), wrong_bits=int((err > 0.5).sum()),
               ok=bool(err.max().item() < 0.02 and y.size(1) - 1 == expected.size(1)),
               build_s=round(t1 - t0, 1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-w", type=int, default=6)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--mode", choices=["lin", "tied"], required=True)
    ap.add_argument("--theta", default="compact", help="compact or split")
    ap.add_argument("--copies", default="step", help="step or glu")
    ap.add_argument("--taps", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.log_w, a.depth, a.mode, theta=a.theta, copies=a.copies, taps=a.taps)))
    sys.stdout.flush()

"""Tiers T3-T7 of the noise / LayerNorm-shift ladder (wave 7, ultra-noise track).

Inputs per circuit: suite.inputs(n_in, 256, seed) (7 edge cases + 256 random at densities
0.05 / 0.5 / 0.95) plus 128 dense inputs (density U[0.8, 1], seed + 1): 391 inputs; wave 7
used seeds 400, 500 and 600 (and 700 for a spot check). Every batch really has b rows (robust_eval.forward).

Threat points (robust_eval specs, all parts together, one run per seed):
  T4P  s0.1+m0.1+e0.01                    on bf16+gpu b256, bf16+gpu b1024, fp16+gpu b256
  T5P  s0.05+m0.1+e0.01+M0.1+n0.01        on f32+gpu_exact b256, bf16+gpu_exact b256,
  T6P  s0.01+m0.1+e0.01+M0.3+n0.03           bf16+gpu b256 and bf16+gpu b1024
  T7P  s0.002+m0.1+e0.01+M1+n0.1
T4's m and e are kept in T5-T7 (the stricter reading of "T5 = T4 + ...").
T3: f32 / bf16 with gpu_exact at b256, bf16 and fp16 on CUDA at b1, b256, b1024.

Diagnostic parts (not tiers; see the wave-7 ultra-noise notes): Mf<x>, nf<x>, sf<x>
set the M shift, the n noise and the s floor of the first layer alone, e.g. T6f =
T6P with the first layer's M at 0.1. They show where a threat breaks a construction:
every T6 / T7 failure measured was in the first layer, which reads the host's input
(BOS = 1 and the bits, one noisy copy each).

  PYTHONPATH=<repo>/src:<repo>/experiments/levels /usr/bin/python3 noise_eval.py \\
      --circuits adder32,parity64 --configs hardened,toward_t6 --sets T3,T4,T5,T6fq \\
      --out out.jsonl
"""

import argparse
import json
import math
import os
import random
import time

import torch as t
import torch.nn.functional as F

import robust_eval as rv
import suite
from reifier.opt import fp16_bound
from reifier.utils.format import Bits

# presets with every knob explicit (level=None), so they keep their meaning if a level changes
_STEPS = dict(dedup=True, schedule="opt", fold=True, fold_bias=2)  # every gate a step
_FLAT = dict(_STEPS, cheap=True, cheap_max=4, cheap_out=False)  # flat units, output steps
_GATES = dict(level=None, exact=True, fanin_xor=4, fanin_and_or=2, fanin_adder="prefix",
              exact_readout=True)
_T5_STEPS = dict(_GATES, q=128, heavy_bos=False, bos_copies=-256, center=True,
                 out_scale=-1024, ln_invariant=8, prologue=True, passes=_STEPS)
CONFIGS: dict[str, dict] = {
    "ultra": {"level": "ultra"},
    "hardened": {"level": "hardened"},
    "robust": {"level": "robust"},
    # wave 6's hardened (T4): heavy BOS, q 256, asymmetric exact steps, steps only
    "hardened_w6": dict(_GATES, q=256, heavy_bos=True, ln_invariant=8, prologue=True,
                        passes=_STEPS),
    # T5 with steps only (0-0.4% above wave 6's hardened; T5 on all 9, sets 400 and 500);
    # hardened (flat units, flat first layer, a BOS copy per 32 features) is 15-38% smaller
    "t5_steps": _T5_STEPS,
    # flat units and the step prologue: T5 on 8 circuits (sets 400 and 500); on the extractor
    # the T5 point misread a bit at bf16+gpu b32 (set 400, seed 0), so T4 there
    "t5_flat": dict(_T5_STEPS, bos_copies=-64, zero_spread=True, passes=_FLAT),
    # everything in T6 but the first layer's M0.3 (T6f), bf16 / f32; fp16 overflows on
    # xof_w4's first layer (q 512); T6 with M 0.1 (T6m) 8/8 on 8 circuits, 7/8 on add4x16
    # (verifier, set 900); 1.79-2.14x wave 6's hardened. With the extended input
    # (input_zeros=16, ext_eval.py) the full T6 point holds
    "toward_t6": dict(_T5_STEPS, zero_spread=True, ln_invariant=16, bos_copies=-16, q=512,
                      rep=2),
    # the hidden layers under T7 (with the first layer at T5's threats, T7f): CPU f32 / bf16
    # only; bf16 split reductions on CUDA round its q 2048 pre-activations
    "toward_t7": dict(_T5_STEPS, zero_spread=True, ln_invariant=32, bos_copies=64, q=2048,
                      rep=16),
}

P = {"T4": "s0.1+m0.1+e0.01", "T5": "s0.05+m0.1+e0.01+M0.1+n0.01",
     "T6": "s0.01+m0.1+e0.01+M0.3+n0.03", "T7": "s0.002+m0.1+e0.01+M1+n0.1",
     "T6m": "s0.01+m0.1+e0.01+M0.1+n0.03",  # T6 with T5's M
     "T6f": "s0.01+m0.1+e0.01+M0.3+Mf0.1+n0.03",  # T6, first layer M0.1
     "T7f": "s0.002+m0.1+e0.01+M1+Mf0.1+n0.1+nf0.01+sf0.05"}  # T7, first layer at T5
T3 = ["f32+gpu_exact+b256", "bf16+gpu_exact+b256"] + [
    f"{d}+gpu+b{b}" for d in ("bf16", "fp16") for b in (1, 256, 1024)]
B4 = ["bf16+gpu+b256", "bf16+gpu+b1024", "fp16+gpu+b256"]
B5 = ["f32+gpu_exact+b256", "bf16+gpu_exact+b256", "bf16+gpu+b256", "bf16+gpu+b1024"]


def fresh_inputs(n_in: int, seed: int) -> list[list[int]]:
    xs = suite.inputs(n_in, 256, seed=seed)
    rng = random.Random(seed + 1)
    for _ in range(128):
        d = rng.uniform(0.8, 1.0)
        xs.append([int(rng.random() < d) for _ in range(n_in)])
    return xs


def refs(name: str, seed: int, cache: str):
    path = os.path.join(cache, f"{name}_s{seed}.pt")
    if os.path.exists(path):
        return t.load(path)
    c = suite.SUITE[name]()
    xs = fresh_inputs(c.n_in, seed)
    X = t.tensor([[1] + x for x in xs], dtype=t.float32)
    Y = t.tensor(suite.reference(c, xs), dtype=t.float32)
    os.makedirs(cache, exist_ok=True)
    t.save((X, Y), path)
    return X, Y


def threat_list(sets: str, seeds: list[int]) -> list[str]:
    """T3, T4, T5, T6, T7, T6m, T6f, T7f (every base and seed), <name>q (bf16+gpu b256
    only), or literal robust_eval specs"""
    out: list[str] = []
    for s in filter(None, sets.split(",")):
        if s == "T3":
            out += T3
        elif s == "T4":
            out += [f"{b}+{P[s]}+seed{sd}" for sd in seeds for b in B4]
        elif s in P:
            out += [f"{b}+{P[s]}+seed{sd}" for sd in seeds for b in B5]
        elif s.endswith("q") and s[:-1] in P:
            out += [f"bf16+gpu+b256+{P[s[:-1]]}+seed{sd}" for sd in seeds]
        else:
            out.append(s)
    return out


def parse(spec: str) -> dict:
    """robust_eval.parse plus the first-layer overrides Mf, nf, sf"""
    first: dict[str, float] = {}
    rest = []
    for p in filter(None, spec.split("+")):
        key = {"Mf": "shift_max", "nf": "noise_max", "sf": "smin"}.get(p[:2])
        if key and p[2:3].isdigit():
            first[key] = float(p[2:])
        else:
            rest.append(p)
    th = rv.parse("+".join(rest))
    th["first"] = first
    return th


def forward(layers, X: t.Tensor, th: dict) -> t.Tensor:
    """robust_eval.forward, with th["first"] overriding the first layer's threats"""
    if not th["first"]:
        return rv.forward(layers, X, th)
    dev, dt, B = th["device"], th["dtype"], th["batch"]
    n_real = len(X)
    if n_real % B:
        reps = -(-n_real // B) * B
        X = X.repeat(-(-reps // n_real), 1)[:reps]
    if dev == "cuda":
        t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = th["rpr"]
        t.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = th["rpr"]
    gen = t.Generator(device="cpu").manual_seed(th["seed"])
    Ws = [tuple(w.to(dev, dt) for w in L) for L in layers]
    outs = []
    with t.inference_mode():
        for i in range(0, len(X), B):
            x = X[i : i + B].to(dev, dt)
            n = len(x)
            for li, (nw, wg, wv, wo) in enumerate(Ws):
                tl = {**th, **th["first"]} if li == 0 else th
                if th["noise"]:
                    rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
                    eps = t.randn(x.shape, generator=gen).to(dev)
                    x = (x.float() + th["noise"] * rms * eps).to(dt)
                if th["noise_max"] or tl["noise_max"]:
                    mx = x.float().abs().amax(-1, keepdim=True)
                    eps = t.randn(x.shape, generator=gen).to(dev)
                    x = (x.float() + tl["noise_max"] * mx * eps).to(dt)
                h = F.rms_norm(x, (x.size(-1),), nw)
                if th["shift_max"] or tl["shift_max"]:
                    hn = F.rms_norm(x, (x.size(-1),))
                    u = t.randn(n, 1, generator=gen).to(dev, dt)
                    h = (hn - tl["shift_max"] * u * hn.abs().amax(-1, keepdim=True)) * nw
                if th["shift"]:
                    u = t.randn(n, 1, generator=gen).to(dev, dt)
                    h = h - th["shift"] * u * nw
                if th["smin"] < 1 or tl["smin"] < 1:
                    r = t.rand(n, 1, generator=gen).to(dev)
                    h = h * t.exp(r * math.log(tl["smin"])).to(dt)
                x = F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wv), wo)
            outs.append(x.float().cpu())
    return t.cat(outs)[:n_real]


def evaluate(layers, X, Y, spec: str) -> dict:
    th = parse(spec)
    if th["device"] == "cuda" and not t.cuda.is_available():
        return {"threat": spec, "skipped": "no CUDA in this torch"}
    t0 = time.time()
    res = {"threat": spec, **rv.score(forward(layers, X, th), Y)}
    res["correct"] = res["wrong_nearest"] == 0 and res["wrong_boolify"] == 0
    res["secs"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--sets", default="T3,T4,T5")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--input-seed", type=int, default=400)
    ap.add_argument("--refs", default=os.path.join(os.path.dirname(__file__), "refs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-only", action="store_true")
    a = ap.parse_args()
    out = open(a.out, "a")

    def emit(d: dict) -> None:
        s = json.dumps(d)
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    seeds = [int(x) for x in a.seeds.split(",")]
    for name in a.circuits.split(","):
        c = suite.SUITE[name]()
        for cfg in a.configs.split(","):
            t0 = time.time()
            mlp = rv.compiler(CONFIGS[cfg]).run(c.fn, x=Bits("0" * c.n_in).bitlist)
            ps = list(mlp.parameters())
            emit({"circuit": name, "config": cfg, "mode": "head", "compiler": CONFIGS[cfg],
                  "input_seed": a.input_seed, "depth": len(mlp.layers),
                  "dense": sum(p.numel() for p in ps),
                  "sparse": sum(int((p != 0).sum()) for p in ps),
                  "fp16_bound": round(fp16_bound(mlp), 1),
                  "compile_s": round(time.time() - t0, 1)})
            if a.size_only or name in suite.SIZE_ONLY:
                continue
            X, Y = refs(name, a.input_seed, a.refs)
            layers = rv.layers_of(mlp)
            del mlp, ps
            for spec in threat_list(a.sets, seeds):
                emit({"circuit": name, "config": cfg, "mode": "threat",
                      "input_seed": a.input_seed, **evaluate(layers, X, Y, spec)})


if __name__ == "__main__":
    main()

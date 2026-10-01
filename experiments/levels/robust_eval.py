"""Robustness of a compiled reifier circuit (MLP_SwiGLU) under deployment-like perturbations.

A threat is a spec string of '+'-joined parts; every part is optional:
  f32 | bf16 | fp16      dtype of weights and activations (default f32)
  gpu                    run on CUDA with torch's defaults, i.e. reduced-precision reductions
                         in bf16/fp16 matmuls (allow_*_reduced_precision_reduction = True)
  gpu_exact              run on CUDA with those reductions switched off
  b<N>                   batch size (default 256; reductions can depend on it)
  s<x>                   the circuit runs next to other (LLM) features, so each layer's
                         RMSNorm is taken over a stream where the circuit holds a varying part
                         of the norm: its normalized features are multiplied by a factor drawn
                         per token and layer, log-uniform in [x, 1] (e.g. s0.1)
  m<x>                   LayerNorm host: the stream mean is subtracted before normalizing. In
                         units of the circuit's rms it is x * N(0, 1), per token and layer
  e<x>                   additive interference: x * rms * N(0, 1) added to every circuit
                         feature before each layer
  n<x>                   additive interference relative to the largest feature instead:
                         x * max|feature| * N(0, 1). Unlike e and m it cannot be lowered by
                         padding the circuit with zero features or by rescaling BOS
  M<x>                   LayerNorm shift relative to the largest normalized feature instead:
                         x * N(0, 1) * max|normalized feature|, per token and layer
  seed<N>                seed of the random perturbations (default 0)
Examples: "bf16+gpu", "f32+s0.1", "bf16+gpu+s0.1+m0.05+e0.01".

Readout, relative to BOS: a bit is wrong if |out/BOS - bit| > 0.5 (nearest), and misread if
reifier's boolify (1 iff within 0.02 of BOS) disagrees. The margin is max |out/BOS - bit|.

Usage (CUDA needs a torch with CUDA, e.g. /usr/bin/python3 on this machine):
  PYTHONPATH=<repo>/src:<repo>/experiments/levels python robust_eval.py CIRCUIT \
      [--threats "f32,bf16,bf16+gpu,f32+s0.1"] [--compiler '{"mlp_dtype": "bfloat16"}'] \
      [--scan] [--n-random 48]
The weights are compiled once (float32 by default; "mlp_dtype": "bfloat16" gives main's exact
bfloat16 steps) and evaluated in every threat's dtype.
Prints one JSON line per threat. --scan finds, for each of s / m / e, the harshest level
on a grid that still gives 0 wrong bits (tolerances).
"""

import argparse
import json
import math
import time

import torch as t
import torch.nn.functional as F


def layers_of(mlp) -> list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]]:
    """(norm weight, wg, wv, wo) of every SwiGLU layer, as float32 CPU tensors"""
    out = []
    for L in mlp.layers:
        out.append(tuple(p.detach().float().cpu() for p in
                         (L.norm.weight, L.wg.weight, L.wv.weight, L.wo.weight)))
    return out


def layers_from_xofbench(layers) -> list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]]:
    """the same from experiments/xof_shrink/xofbench.build_layers dicts (sparse weights)"""
    def dense(w):
        return (w.to_dense() if w.is_sparse else w).float()
    return [tuple(dense(L[k]) for k in ("norm", "wg", "wv", "wo")) for L in layers]


def parse(spec: str) -> dict:
    th = {"dtype": t.float32, "device": "cpu", "rpr": True, "batch": 256,
          "smin": 1.0, "shift": 0.0, "noise": 0.0, "noise_max": 0.0, "shift_max": 0.0,
          "seed": 0}
    for part in filter(None, spec.split("+")):
        if part in ("f32", "bf16", "fp16"):
            th["dtype"] = {"f32": t.float32, "bf16": t.bfloat16, "fp16": t.float16}[part]
        elif part == "gpu":
            th["device"], th["rpr"] = "cuda", True
        elif part == "gpu_exact":
            th["device"], th["rpr"] = "cuda", False
        elif part.startswith("seed"):
            th["seed"] = int(part[4:])
        elif part[0] == "b" and part[1:].isdigit():
            th["batch"] = int(part[1:])
        elif part[0] in "smenM":
            key = {"s": "smin", "m": "shift", "e": "noise", "n": "noise_max", "M": "shift_max"}
            th[key[part[0]]] = float(part[1:])
        else:
            raise ValueError(f"unknown threat part {part!r}")
    return th


def forward(layers, X: t.Tensor, th: dict) -> t.Tensor:
    """X: (n, 1 + n_in) with the BOS feature first; returns the outputs in float32.
    Every batch has exactly th["batch"] rows (the inputs are tiled to fill the last one):
    CUDA picks its reduction per matrix shape, e.g. split reductions from about 128-256 rows,
    so a smaller batch would test a different kernel."""
    dev, dt = th["device"], th["dtype"]
    n_real = len(X)
    B = th["batch"]
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
        for i in range(0, len(X), th["batch"]):
            x = X[i : i + th["batch"]].to(dev, dt)
            n = len(x)
            for nw, wg, wv, wo in Ws:
                if th["noise"]:
                    rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
                    eps = t.randn(x.shape, generator=gen).to(dev)
                    x = (x.float() + th["noise"] * rms * eps).to(dt)
                if th["noise_max"]:
                    mx = x.float().abs().amax(-1, keepdim=True)
                    eps = t.randn(x.shape, generator=gen).to(dev)
                    x = (x.float() + th["noise_max"] * mx * eps).to(dt)
                h = F.rms_norm(x, (x.size(-1),), nw)
                if th["shift_max"]:
                    hn = F.rms_norm(x, (x.size(-1),))
                    u = t.randn(n, 1, generator=gen).to(dev, dt)
                    shift = th["shift_max"] * u * hn.abs().amax(-1, keepdim=True)
                    h = (hn - shift) * nw
                if th["shift"]:
                    u = t.randn(n, 1, generator=gen).to(dev, dt)
                    h = h - th["shift"] * u * nw
                if th["smin"] < 1:
                    r = t.rand(n, 1, generator=gen).to(dev)
                    s = t.exp(r * math.log(th["smin"])).to(dt)
                    h = h * s
                x = F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wv), wo)
            outs.append(x.float().cpu())
    return t.cat(outs)[:n_real]


def score(out: t.Tensor, Y: t.Tensor) -> dict:
    r = out[:, 1:] / out[:, :1]
    err = (r - Y).abs()
    err = t.where(t.isnan(err), t.full_like(err, float("inf")), err)
    ones = (r - 1).abs() <= 0.02
    return {"margin": err.max().item(), "wrong_nearest": int((err > 0.5).sum()),
            "wrong_boolify": int((ones != Y.bool()).sum())}


def evaluate(layers, X, Y, spec: str) -> dict:
    th = parse(spec)
    if th["device"] == "cuda" and not t.cuda.is_available():
        return {"threat": spec, "skipped": "no CUDA in this torch"}
    t0 = time.time()
    res = {"threat": spec, **score(forward(layers, X, th), Y)}
    res["correct"] = res["wrong_nearest"] == 0 and res["wrong_boolify"] == 0
    res["secs"] = round(time.time() - t0, 1)
    return res


GRIDS = {  # harsher to the right
    "s": [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
    "m": [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    "e": [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "n": [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "M": [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
}


def tolerance(layers, X, Y, kind: str, base: str = "f32", seeds=(0, 1)) -> dict:
    """the harshest grid level of kind (s, m or e) on top of base that keeps every bit
    correct for all seeds (the scan stops at the first failing level)"""
    ok = None
    for v in GRIDS[kind]:
        # s1 is no fluctuation; m1 is a real level (it used to be skipped as well)
        off = v == 0.0 or (kind == "s" and v == 1.0)
        spec = "+".join(p for p in [base, "" if off else f"{kind}{v}"] if p)
        good = all(evaluate(layers, X, Y, f"{spec}+seed{sd}").get("correct") for sd in seeds)
        if not good:
            break
        ok = v
    return {"kind": kind, "base": base, "tolerates": ok}


def compiler(cfg: dict):
    """A Compiler for a driver config in the levels' first form: no level is main's
    (reifier.tensors.compilation); a level is reifier.opt's, every key but mlp_dtype and
    select a knob, and knobs turn selection off unless select is given (None values left
    out)"""
    kw = {k: v for k, v in cfg.items() if v is not None}
    if isinstance(kw.get("mlp_dtype"), str):  # e.g. "bfloat16"
        kw["mlp_dtype"] = getattr(t, kw["mlp_dtype"])
    if "level" not in kw:
        from reifier.tensors.compilation import Compiler
        return Compiler(**kw)
    if kw.get("passes") is False:
        raise ValueError("passes=False (wave 5's tree layout under a level) is not in reifier.opt")
    from reifier.opt import Compiler
    own = {k: kw.pop(k) for k in ("level", "mlp_dtype", "select") if k in kw}
    return Compiler(**{"select": not kw, **own}, knobs=kw)


def main():
    import suite
    from reifier.utils.format import Bits

    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", choices=list(suite.SUITE))
    ap.add_argument("--threats", default="f32,bf16,bf16+gpu,bf16+gpu_exact,fp16+gpu")
    ap.add_argument("--compiler", default="{}", help="JSON config for compiler()")
    ap.add_argument("--n-random", type=int, default=48)
    ap.add_argument("--scan", action="store_true", help="tolerances of s / m / e on f32 and bf16")
    a = ap.parse_args()
    c = suite.SUITE[a.circuit]()
    t0 = time.time()
    mlp = compiler(json.loads(a.compiler)).run(c.fn, x=Bits("0" * c.n_in).bitlist)
    layers = layers_of(mlp)
    xs = suite.inputs(c.n_in, a.n_random)
    X = t.tensor([[1] + x for x in xs], dtype=t.float32)
    Y = t.tensor(suite.reference(c, xs), dtype=t.float32)
    head = {"circuit": a.circuit, "compiler": json.loads(a.compiler), "depth": len(layers),
            "dense": sum(p.numel() for p in mlp.parameters()),
            "sparse": sum(int((p != 0).sum()) for p in mlp.parameters()),
            "n_inputs": len(xs), "compile_s": round(time.time() - t0, 1)}
    print(json.dumps(head), flush=True)
    for spec in filter(None, a.threats.split(",")):
        print(json.dumps({"circuit": a.circuit, **evaluate(layers, X, Y, spec)}), flush=True)
    if a.scan:
        for base in ("f32", "bf16"):
            for kind in "sme":
                print(json.dumps({"circuit": a.circuit, **tolerance(layers, X, Y, kind, base)}),
                      flush=True)


if __name__ == "__main__":
    main()

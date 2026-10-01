"""Tiers T3-T7 of compiled suite circuits on fresh inputs at real batches (wave 7).

Inputs: suite.inputs(n_in, 256, seed) + 128 dense inputs (density U[0.8, 1], seed + 1).
Threat groups (robust_eval specs; every batch really has b rows):
  T3:  f32+gpu_exact+b256, bf16+gpu_exact+b256, {bf16,fp16}+gpu+b{1,256,1024}
  T4:  s0.1+m0.1+e0.01, 8 seeds, on bf16+gpu b256 / b1024, fp16+gpu b256
  T5:  s0.05+m0.1+e0.01+M0.1+n0.01, 8 seeds, on f32 (gpu_exact), bf16 (gpu_exact), bf16+gpu b256/b1024
  T5sM: the same without n
  T6:  s0.01+m0.1+e0.01+M0.3+n0.03, 8 seeds, same bases; T6sM without n
  T7 parts: s0.002, M1, n0.1 (separately), 4 seeds, f32 / bf16+gpu b256
"ok" in the group records means 0 failing runs; --stop ends a group at its first failure.

  PYTHONPATH=<repo>/src:<repo>/experiments/levels /usr/bin/python3 tiers_eval.py \
      --circuits adder32,parity64 --configs ultra,hardened --groups T3,T4,T5 \
      --refs /tmp/refs --out out.jsonl
"""
import argparse
import json
import os
import random
import time
import torch as t
import robust_eval as rv
import suite
import wide_circuits  # noqa: F401  (wide out-of-suite circuits)
from reifier.opt import fp16_bound
from reifier.utils.format import Bits

CONFIGS = {
    # the levels (each keeps the smallest of its candidate recipes)
    **{f"L_{lv}": {"level": lv} for lv in ("ultra", "hardened", "robust", "O2", "O3")},
    # recipes on their own
    **{k: {"level": k, "select": False} for k in ("ultra", "ultra_bc64", "hardened")},
    "robust": {"level": "robust", "select": False},
    # the ultra-scale track's T5 construction (wave 6's hardened + fit16, BOS = 1, centered
    # steps, 2 always-0 features spread evenly; dominated by ultra) and its preset for T6's
    # s side: + ballast 1/64 (norm scales down to 0.01)
    "ultra_s": {"level": "hardened", "fit16": True, "heavy_bos": False, "center": True,
                "zero_spread": True, "ln_invariant": 2},
    "ultra_s_b64": {"level": "hardened", "fit16": True, "heavy_bos": False, "center": True,
                    "zero_spread": True, "ln_invariant": 2, "ballast": 1 / 64},
}
SEEDS8 = [f"seed{s}" for s in range(100, 108)]
SEEDS4 = [f"seed{s}" for s in range(100, 104)]
B_T3 = ["f32+gpu_exact+b256", "bf16+gpu_exact+b256"] + [
    f"{d}+gpu+b{b}" for d in ("bf16", "fp16") for b in (1, 256, 1024)]
B_T56 = ["f32+gpu_exact+b256", "bf16+gpu_exact+b256", "bf16+gpu+b256", "bf16+gpu+b1024"]
POINTS = {
    "T4": ("s0.1+m0.1+e0.01", ["bf16+gpu+b256", "bf16+gpu+b1024", "fp16+gpu+b256"], SEEDS8),
    "T5sM": ("s0.05+m0.1+e0.01+M0.1", B_T56, SEEDS8),
    "T5M2": ("s0.05+m0.1+e0.01+M0.2+n0.01", B_T56, SEEDS8),  # T5 with M 0.2
    "T5b1": ("s0.05+m0.1+e0.01+M0.1+n0.01", ["bf16+gpu+b1", "fp16+gpu+b1"], SEEDS4),  # batch 1
    "T5": ("s0.05+m0.1+e0.01+M0.1+n0.01", B_T56, SEEDS8),
    "T6sM": ("s0.01+m0.1+e0.01+M0.3", B_T56, SEEDS8),
    "T6": ("s0.01+m0.1+e0.01+M0.3+n0.03", B_T56, SEEDS8),
    "T6s": ("s0.01+m0.1+e0.01", B_T56, SEEDS8),
    "T6M": ("m0.1+e0.01+M0.3", B_T56, SEEDS8),
    "T7s": ("s0.002", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS4),
    "T7M": ("M1", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS4),
    "T7n": ("n0.1", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS4),
    "s005": ("s0.005", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS4),
    "M03": ("M0.3", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS8),
    "n001": ("n0.01", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS4),
    "n003": ("n0.03", ["f32+gpu_exact+b256", "bf16+gpu+b256"], SEEDS4),
    "T3b": ("", [f"{d}+gpu+b{b}" for d in ("bf16", "fp16") for b in (8, 32, 64, 128, 512, 2048)],
            [""]),
}


def fresh(n_in, seed):
    xs = suite.inputs(n_in, 256, seed=seed)
    rng = random.Random(seed + 1)
    for _ in range(128):
        d = rng.uniform(0.8, 1.0)
        xs.append([int(rng.random() < d) for _ in range(n_in)])
    return xs


def refs(name, seed, cache):
    path = os.path.join(cache, f"{name}_s{seed}.pt")
    if os.path.exists(path):
        return t.load(path)
    c = suite.SUITE[name]()
    xs = fresh(c.n_in, seed)
    X = t.tensor([[1] + x for x in xs], dtype=t.float32)
    Y = t.tensor(suite.reference(c, xs), dtype=t.float32)
    os.makedirs(cache, exist_ok=True)
    t.save((X, Y), path)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--cfgjson", default="{}", help="extra configs as JSON {name: kwargs}")
    ap.add_argument("--groups", default="T3,T4,T5sM,T5,T6sM,T6")
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop", action="store_true", help="stop a group at its first failure")
    a = ap.parse_args()
    CONFIGS.update(json.loads(a.cfgjson))
    out = open(a.out, "a")

    def emit(d):
        s = json.dumps(d)
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    for name in a.circuits.split(","):
        X, Y = refs(name, a.seed, a.refs)
        c = suite.SUITE[name]()
        for cfg in a.configs.split(","):
            t0 = time.time()
            comp = rv.compiler(CONFIGS[cfg])
            mlp = comp.run(c.fn, x=Bits("0" * c.n_in).bitlist)
            ps = list(mlp.parameters())
            head = {"circuit": name, "config": cfg, "mode": "head", "compiler": CONFIGS[cfg],
                    "input_seed": a.seed, "depth": len(mlp.layers),
                    "dense": sum(p.numel() for p in ps),
                    "sparse": sum(int((p != 0).sum()) for p in ps),
                    "fp16_bound": round(fp16_bound(mlp), 1), "n_inputs": len(X),
                    "compile_s": round(time.time() - t0, 1)}
            layers = rv.layers_of(mlp)
            del mlp, ps
            emit(head)
            for grp in a.groups.split(","):
                if grp == "T3":
                    specs = B_T3
                else:
                    pert, bases, seeds = POINTS[grp]
                    specs = ["+".join(x for x in (b, pert, sd) if x) for sd in seeds for b in bases]
                nfail = 0
                for spec in specs:
                    r = rv.evaluate(layers, X, Y, spec)
                    emit({"circuit": name, "config": cfg, "mode": "threat", "group": grp,
                          "input_seed": a.seed, **r})
                    nfail += not r.get("correct", False)
                    if nfail and a.stop:
                        break
                emit({"circuit": name, "config": cfg, "mode": "group", "group": grp,
                      "input_seed": a.seed, "runs": len(specs), "fail": nfail})
            del layers


if __name__ == "__main__":
    main()

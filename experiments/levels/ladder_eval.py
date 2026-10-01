"""The ladder's tiers per suite circuit (wave 7): every level compiled with selection, on
fresh inputs at real batches, with fresh perturbation seeds.

Inputs (--seed): suite.inputs(n_in, 256, seed) (7 edge cases + 256 random at densities
0.05 / 0.5 / 0.95), 128 dense inputs (density U[0.8, 1]), and 96 near-constant inputs
(48 with 1-3 zeros, 48 with 1-3 ones): 487 per circuit. References by eager evaluation.

Groups (robust_eval specs; every batch really has b rows; seeds --pseed .. --pseed + 7):
  T1  f32+gpu_exact b256: correct and worst error <= 0.01
  T2  bf16+gpu_exact b256 (float32 accumulation)
  T3  {bf16, fp16}+gpu (reduced-precision reductions) at b1, b77, b256, b1024
  T4  s0.1+m0.1+e0.01: 8 seeds x {bf16, fp16}+gpu x {b256, b1024}, 2 seeds x {bf16, fp16}+gpu b1
  T5  s0.05+m0.1+e0.01+M0.1+n0.01: 8 seeds x {f32+gpu_exact, bf16+gpu_exact, bf16+gpu,
      fp16+gpu} b256 and bf16+gpu b1024, 2 seeds x {bf16, fp16}+gpu b1
A level's groups run up to its claimed tier plus one (T5 at most); a group above the claim
stops at its first failing run. The tier on a circuit is the highest Tk with T1..Tk all
correct. Levels whose compile is bit-identical to one measured before (same circuit)
reuse its runs.

  PYTHONPATH=<repo>/src:<repo>/experiments/levels /usr/bin/python3 ladder_eval.py \
      --circuits adder32,parity64 --refs /tmp/refs --out out.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import time
import warnings

import torch as t

import robust_eval as rv
import suite
from reifier.opt import TIERS, Compiler, fp16_bound
from reifier.utils.format import Bits

T4P = "s0.1+m0.1+e0.01"
T5P = "s0.05+m0.1+e0.01+M0.1+n0.01"


def groups(pseed: int) -> dict[str, list[str]]:
    s8 = [pseed + i for i in range(8)]
    s2 = [pseed + 100 + i for i in range(2)]
    return {
        "T1": ["f32+gpu_exact+b256"],
        "T2": ["bf16+gpu_exact+b256"],
        "T3": [f"{d}+gpu+b{b}" for d in ("bf16", "fp16") for b in (1, 77, 256, 1024)],
        "T4": [f"{d}+gpu+b{b}+{T4P}+seed{s}" for s in s8 for d in ("bf16", "fp16")
               for b in (256, 1024)]
              + [f"{d}+gpu+b1+{T4P}+seed{s}" for s in s2 for d in ("bf16", "fp16")],
        "T5": [f"{b}+b256+{T5P}+seed{s}" for s in s8
               for b in ("f32+gpu_exact", "bf16+gpu_exact", "bf16+gpu", "fp16+gpu")]
              + [f"bf16+gpu+b1024+{T5P}+seed{s}" for s in s8]
              + [f"{d}+gpu+b1+{T5P}+seed{s}" for s in s2 for d in ("bf16", "fp16")],
    }


def fresh(n_in: int, seed: int) -> list[list[int]]:
    xs = suite.inputs(n_in, 256, seed=seed)
    rng = random.Random(seed + 1)
    for _ in range(128):
        d = rng.uniform(0.8, 1.0)
        xs.append([int(rng.random() < d) for _ in range(n_in)])
    rng = random.Random(seed + 2)
    for i in range(96):
        k = min(1 + i % 3, n_in)
        pos = set(rng.sample(range(n_in), k))
        bit = 0 if i < 48 else 1  # 1-3 zeros among ones, then 1-3 ones among zeros
        xs.append([bit if j in pos else 1 - bit for j in range(n_in)])
    return xs


def refs(name: str, seed: int, cache: str):
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


def where(layers, X, Y, spec: str) -> dict:
    """the failing inputs of a run: index, number of 1s, the bad bits and their values"""
    out = rv.forward(layers, X, rv.parse(spec))
    r = out[:, 1:] / out[:, :1]
    bad = ((r - Y).abs() > 0.5) | (((r - 1).abs() <= 0.02) != Y.bool())
    rows = bad.any(1).nonzero().flatten().tolist()
    return {"bad_inputs": [
        {"i": i, "ones": int(X[i, 1:].sum()), "bits": bad[i].nonzero().flatten().tolist()[:4],
         "vals": [round(v, 3) for v in r[i][bad[i]].tolist()[:4]]} for i in rows[:4]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--levels", default="ultra,hardened,robust,O2,O3")
    ap.add_argument("--seed", type=int, default=3100)
    ap.add_argument("--pseed", type=int, default=3100)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs-only", action="store_true")
    ap.add_argument("--max-tier", type=int, default=5)
    a = ap.parse_args()
    if a.refs_only:
        for name in a.circuits.split(","):
            refs(name, a.seed, a.refs)
        return
    G = groups(a.pseed)
    out = open(a.out, "a")

    def emit(d):
        s = json.dumps(d)
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    for name in a.circuits.split(","):
        X, Y = refs(name, a.seed, a.refs)
        c = suite.SUITE[name]()
        seen: dict[str, dict] = {}  # weights hash -> {group: (runs, fails)}
        for lv in a.levels.split(","):
            t0 = time.time()
            comp = Compiler(level=lv)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mlp = comp.run(c.fn, x=Bits("0" * c.n_in).bitlist)
            ps = list(mlp.parameters())
            h = hashlib.sha256()
            for p in ps:
                h.update(p.detach().float().cpu().numpy().tobytes())
            hx = h.hexdigest()[:16]
            head = {"circuit": name, "level": lv, "mode": "head", "chosen": comp.chosen,
                    "input_seed": a.seed, "pseed": a.pseed, "depth": len(mlp.layers),
                    "dense": sum(p.numel() for p in ps),
                    "sparse": sum(int((p != 0).sum()) for p in ps),
                    "fp16_bound": round(fp16_bound(mlp), 1), "n_inputs": len(X),
                    "compile_s": round(time.time() - t0, 1), "hash": hx,
                    "warnings": [str(x.message)[:120] for x in w]}
            layers = rv.layers_of(mlp)
            del mlp, ps
            emit(head)
            claim = TIERS[lv]
            res = seen.setdefault(hx, {})
            tier = 0
            for k in range(1, min(claim + 1, a.max_tier) + 1):
                grp = f"T{k}"
                if grp not in res:
                    nfail, worst, first = 0, 0.0, None
                    runs = 0
                    for spec in G[grp]:
                        r = rv.evaluate(layers, X, Y, spec)
                        runs += 1
                        ok = r.get("correct", False) and (k != 1 or r["margin"] <= 0.01)
                        worst = max(worst, r.get("margin", float("inf")))
                        if not ok:
                            nfail += 1
                            first = first or r
                            emit({"circuit": name, "level": lv, "mode": "fail", "group": grp,
                                  **r, **where(layers, X, Y, spec)})
                            if k > claim:
                                break
                    res[grp] = {"runs": runs, "fail": nfail, "worst": worst, "first": first}
                g = res[grp]
                emit({"circuit": name, "level": lv, "mode": "group", "group": grp,
                      "chosen": comp.chosen, "hash": hx, **{k2: v for k2, v in g.items()
                                                            if k2 != "first"}})
                if g["fail"]:
                    break
                tier = k
            emit({"circuit": name, "level": lv, "mode": "tier", "chosen": comp.chosen,
                  "claim": claim, "tier": tier, "dense": head["dense"],
                  "sparse": head["sparse"], "depth": head["depth"], "hash": hx})
            del layers


if __name__ == "__main__":
    main()

"""Measure the optimization levels of reifier's Compiler on the suite, on fresh inputs, at
real batch sizes (wave 6).

Inputs per circuit: suite.inputs(n_in, 256, seed=100) (7 edge cases + 256 random at
densities 0.05 / 0.5 / 0.95) plus 128 dense inputs (density uniform in [0.8, 1], seed 101),
391 in all; references come from eager evaluation (cached in --refs).
For every (circuit, config) one "head" line (depth, dense, sparse, resolved settings) and
one line per threat (robust_eval specs; every batch really has b rows, see forward).

Tiers (highest k such that T0..Tk all hold, see summarize.py):
  T0 f32 correct; T1 f32 worst error <= 0.01; T2 bf16 with float32 accumulation (CUDA,
  reduced-precision reductions off); T3 bf16 and fp16 on CUDA with reduced-precision
  reductions at b1, b256 and b1024; T4 T3 + s0.1+m0.1+e0.01 combined, 8 seeds, on
  bf16+gpu b256 / b1024 and fp16+gpu b256.

  PYTHONPATH=<repo>/src:<repo>/experiments/levels /usr/bin/python3 levels_eval.py \
      --circuits adder32,parity64 --configs main,O1,robust,hardened --out results.jsonl
"""

import argparse
import json
import os
import random
import time

import torch as t

import robust_eval as rv
import suite
from reifier.opt import fp16_bound, recipe
from reifier.utils.format import Bits

# wave 6's O1 level (merged into robust in wave 7), as knobs; its pass preset as a dict
O1 = {"passes": "O1", "exact": True, "exact_readout": True, "min_gain": 10.0}
O1P = {"fold": True, "fold_not": False, "dedup": True, "schedule": "opt"}
# wave-7 pass groups (as in reifier.opt.recipes)
X = {"sumfuse": "e", "xortree": True}
XF = {**X, "flat_xor": True}
X1C = {**X, "reclean": 1, "clean_outputs": True}
FC = {"cone": 3, "cone_flat": True, "cone_units": 4}
CH = {"cheap": True, "cheap_max": 4}
CLEAN = {"clean_outputs": True}

CONFIGS: dict[str, dict] = {
    "main": {},
    "main_bf16": {"mlp_dtype": "bfloat16"},
    "O1": dict(O1),
    "O1_bf16": {**O1, "mlp_dtype": "bfloat16"},
    "O1u_bf16": {"passes": "O1u", "mlp_dtype": "bfloat16"},  # wave 5's O1 (all folds)
    # the levels (wave 7: each keeps the smallest recipe of candidates(level))
    "L_ultra": {"level": "ultra"},
    "L_hardened": {"level": "hardened"},
    "L_robust": {"level": "robust"},
    "L_O2": {"level": "O2"},
    "L_O3": {"level": "O3"},
    # the recipes on their own (wave 6's levels and wave 7's recipes)
    **{k: {"level": k, "select": False} for k in (
        "ultra", "ultra_bc64", "hardened", "robust", "O2", "O3", "robust_x1_c",
        "robust_fcx1_cc", "robust_fcf_c", "robust_fcf")},
    # wave 7 (pareto track) recipes that the merge dropped from RECIPES
    "hardened_l2_xf_c": {"level": "hardened", "ln_invariant": 2,
                         "passes": {"sumfuse": "e", "xortree": True, "flat_xor": True,
                                    "clean_outputs": True}},
    "hardened_l2_fcf_c": {"level": "hardened", "ln_invariant": 2,
                          "passes": {"cone": 3, "cone_flat": True, "cone_units": 4,
                                     "clean_outputs": True}},
    # wave 7 (ultra-scale track): its T5 construction (dominated by ultra) and the
    # ballast preset for T6's s side (s 0.01)
    "ultra_s": {"level": "hardened", "fit16": True, "heavy_bos": False, "center": True,
                "zero_spread": True, "ln_invariant": 2},
    "ultra_s_b64": {"level": "hardened", "fit16": True, "heavy_bos": False, "center": True,
                    "zero_spread": True, "ln_invariant": 2, "ballast": 1 / 64},
    # wave 6's candidates for robust's size passes
    "robust_f": {"level": "robust", "passes": {"fold": True, "fold_bias": 2}},
    "robust_c": {"level": "robust", "passes": {"cheap": True}},
    "robust_cf": {"level": "robust",
                  "passes": {"cheap": True, "cone": 3, "cone_flat": True, "cone_units": 4}},
    "robust_fc": {"level": "robust", "passes": {"fold": True, "fold_bias": 2, "cheap": True}},
    "hardened_f": {"level": "hardened", "passes": {"fold": True, "fold_bias": 2}},
    "O1nf_bf16": {"passes": {"dedup": True, "schedule": "opt"}, "mlp_dtype": "bfloat16"},
    "O1cp_bf16": {"passes": {"fold": True, "fold_not": False, "dedup": True, "schedule": "opt"},
                  "mlp_dtype": "bfloat16"},
    "O1asap_bf16": {"passes": {"dedup": True, "schedule": "asap"}, "mlp_dtype": "bfloat16"},
    "hardened_fc2": {"level": "hardened",
                     "passes": {"fold": True, "fold_bias": 2, "cheap": True, "cheap_max": 2}},
    "robust_c4": {"level": "robust", "passes": {"cheap_max": 4}},
    "O1_graph": {**O1, "min_gain": 0.0},  # always the graph layout
    "O1_nofold": {**O1, "passes": {**O1P, "fold": False}},
    "O1_asap": {**O1, "passes": {**O1P, "schedule": "asap"}},
    "O1_alap": {**O1, "passes": {**O1P, "schedule": "alap"}},
    "O1_nodedup": {**O1, "passes": {**O1P, "dedup": False}},
    "robust_tree": {"level": "robust", "passes": False},  # wave 5's robust (tree compiler)
    "hardened_tree": {"level": "hardened", "passes": False},
    # wave 7's candidates (experiments/levels/README.md; the 5 adopted ones are above)
    "robust_x": {"level": "robust", "passes": dict(X)},
    "robust_x1": {"level": "robust", "passes": {**X, "reclean": 1}},
    "robust_x2": {"level": "robust", "passes": {**X, "reclean": 2}},
    "robust_x2_c": {"level": "robust", "passes": {**X, "reclean": 2, **CLEAN}},
    "robust_xf": {"level": "robust", "passes": dict(XF)},
    "robust_xf_c": {"level": "robust", "passes": {**XF, **CLEAN}},
    "robust_o2": {"level": "robust", "passes": {**X, "cone": 3, "reclean": 2}},
    "robust_nc": {"level": "robust", "passes": {"cheap": False}},
    "robust_fcxf": {"level": "robust", "passes": {**FC, **XF}},
    "robust_fcxf_c": {"level": "robust", "passes": {**FC, **XF, **CLEAN}},
    "robust_fcx1_c": {"level": "robust", "passes": {**FC, **X1C}},
    "robust_fcx1_cc": {"level": "robust", "passes": {**FC, **X1C, "cone_reclean": True}},
    "O2_prefix": {"level": "O2", "fanin_adder": "prefix"},
    "O2_c": {"level": "O2", "passes": dict(CLEAN)},
    "O3_x": {"level": "O3", "passes": {"xortree": True}},
    "hardened_l1": {"level": "hardened", "ln_invariant": 1},
    "hardened_l2": {"level": "hardened", "ln_invariant": 2},
    "hardened_fc": {"level": "hardened", "passes": {"cheap": True}},
    "hardened_xf": {"level": "hardened", "passes": dict(XF)},
    "hardened_xf_c": {"level": "hardened", "passes": {**XF, **CLEAN}},
    "hardened_c_c": {"level": "hardened", "passes": {**CH, **CLEAN}},
    "hardened_cxf_c": {"level": "hardened", "passes": {**CH, **XF, **CLEAN}},
    "hardened_cx1_c": {"level": "hardened", "passes": {**CH, **X1C}},
    "hardened_cfcx1_c": {"level": "hardened", "passes": {**CH, **FC, **X1C}},
    "hardened_l2_cfcxf_c": {"level": "hardened", "ln_invariant": 2,
                            "passes": {**CH, **FC, **XF, **CLEAN}},
    # cheap units of <= 2 inputs only (4-input ones miss T4 on a 256-bit adder)
    "hardened_l2_c2fc_c": {"level": "hardened", "ln_invariant": 2,
                           "passes": {"cheap": True, "cheap_max": 2, **FC, **CLEAN}},
    "hardened_l2_c2xf_c": {"level": "hardened", "ln_invariant": 2,
                           "passes": {"cheap": True, "cheap_max": 2, **XF, **CLEAN}},
    # with cheap units: T4 on the suite, not on a 256-bit adder (not recipes)
    "hardened_l2_cxf_c": {"level": "hardened", "ln_invariant": 2, "passes": {**CH, **XF, **CLEAN}},
    "hardened_l2_cfc_c": {"level": "hardened", "ln_invariant": 2, "passes": {**CH, **FC, **CLEAN}},
    "hardened_l2_fcxf_c": {"level": "hardened", "ln_invariant": 2,
                           "passes": {**FC, **XF, **CLEAN}},
}

T3 = [f"{d}+gpu+b{b}" for d in ("bf16", "fp16") for b in (1, 256, 1024)]
BASE = ["f32+gpu_exact+b256", "bf16+gpu_exact+b256"] + T3
T4P = "s0.1+m0.1+e0.01"


def t4_threats(seeds) -> list[str]:
    return [f"{base}+{T4P}+seed{sd}" for sd in seeds
            for base in ("bf16+gpu+b256", "bf16+gpu+b1024", "fp16+gpu+b256")]


SEED = 100  # --seed: suite.inputs(n_in, 256, seed) + 128 dense inputs (seed + 1)


def fresh_inputs(n_in: int) -> list[list[int]]:
    xs = suite.inputs(n_in, 256, seed=SEED)
    rng = random.Random(SEED + 1)
    for _ in range(128):
        d = rng.uniform(0.8, 1.0)
        xs.append([int(rng.random() < d) for _ in range(n_in)])
    return xs


def refs(name: str, cache: str):
    path = os.path.join(cache, f"{name}.pt" if SEED == 100 else f"{name}_s{SEED}.pt")
    if os.path.exists(path):
        return t.load(path)
    c = suite.SUITE[name]()
    xs = fresh_inputs(c.n_in)
    X = t.tensor([[1] + x for x in xs], dtype=t.float32)
    Y = t.tensor(suite.reference(c, xs), dtype=t.float32)
    os.makedirs(cache, exist_ok=True)
    t.save((X, Y), path)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", default=os.path.join(os.path.dirname(__file__), "refs"))
    ap.add_argument("--threats", default=",".join(BASE))
    ap.add_argument("--t4-seeds", default="100,101,102,103,104,105,106,107")
    ap.add_argument("--t4", default="hardened",
                    help="configs that also run the T4 threats (and every hardened*)")
    ap.add_argument("--extra", default="", help="more threats for every config")
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--make-refs", action="store_true", help="only compute the references")
    ap.add_argument("--seed", type=int, default=100, help="input seed (100: the main set)")
    a = ap.parse_args()
    global SEED
    SEED = a.seed
    out = open(a.out, "a")

    def emit(d: dict) -> None:
        s = json.dumps(d)
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    t4 = set(a.t4.split(","))
    for name in a.circuits.split(","):
        size_only = a.size_only or name in suite.SIZE_ONLY
        if not size_only:
            t0 = time.time()
            X, Y = refs(name, a.refs)
            if a.make_refs:
                emit({"circuit": name, "mode": "refs", "n_inputs": len(X),
                      "secs": round(time.time() - t0, 1)})
                continue
        c = suite.SUITE[name]()
        for cfg in a.configs.split(","):
            t0 = time.time()
            comp = rv.compiler(CONFIGS[cfg])
            mlp = comp.run(c.fn, x=Bits("0" * c.n_in).bitlist)
            ps = list(mlp.parameters())
            settings = recipe(comp.level, **comp.knobs) if hasattr(comp, "knobs") else {}
            head = {"circuit": name, "config": cfg, "mode": "head", "compiler": CONFIGS[cfg],
                    "input_seed": SEED,
                    "settings": settings, "chosen": getattr(comp, "chosen", None),
                    "depth": len(mlp.layers),
                    "dense": sum(p.numel() for p in ps),
                    "sparse": sum(int((p != 0).sum()) for p in ps),
                    "fp16_bound": round(fp16_bound(mlp), 1),
                    "compile_s": round(time.time() - t0, 1)}
            if size_only:
                emit(head)
                del mlp, ps
                continue
            layers = rv.layers_of(mlp)
            del mlp, ps
            head["n_inputs"] = len(X)
            emit(head)
            threats = [x for x in a.threats.split(",") if x]
            threats += [x for x in a.extra.split(",") if x]
            if cfg in t4 or cfg.startswith("hardened"):
                threats += t4_threats(a.t4_seeds.split(","))
            for spec in threats:
                r = rv.evaluate(layers, X, Y, spec)
                emit({"circuit": name, "config": cfg, "mode": "threat", "input_seed": SEED,
                      **r})
            del layers


if __name__ == "__main__":
    main()

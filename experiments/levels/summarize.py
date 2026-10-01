"""Tables from levels_eval.py output: sizes, ratios to main's default compile, and the
highest tier verified per (circuit, config).

  python summarize.py logs/*.jsonl [--markdown] [--json out.json]
"""

import argparse
import json
from collections import defaultdict

# main's default compile (Compiler(), float32): depth, dense, sparse (the wave-5 brief's
# table, re-measured there and by the wave-5 verifiers)
MAIN = {
    "xof_w4": (20, 206_803_140, 508_772),
    "sha3_w2": (144, 93_988_668, 984_844),
    "sha256_2r": (45, 46_697_399, 322_119),
    "sha256_r2": (314, 11_425_473_746, 8_898_738),
    "adder32": (6, 1_294_715, 20_941),
    "add4x16": (11, 764_908, 14_558),
    "backdoor_w2": (145, 93_997_775, 985_131),
    "sandbagger_w1": (407, 187_487_322, 1_772_656),
    "extractor_128x64": (2, 37_852_099, 582_255),
    "parity64": (4, 79_245, 11_779),
}
T3 = [f"{d}+gpu+b{b}" for d in ("bf16", "fp16") for b in (1, 256, 1024)]


def tiers(th: dict) -> dict:
    """flags T0-T4 from threat results (None: not run)"""
    def ok(spec):
        r = th.get(spec)
        return None if r is None else bool(r.get("correct"))

    f32 = th.get("f32+gpu_exact+b256")
    t = {"T0": ok("f32+gpu_exact+b256")}
    t["T1"] = None if f32 is None else (t["T0"] and f32["margin"] <= 0.01)
    t["T2"] = ok("bf16+gpu_exact+b256")
    t3 = [ok(s) for s in T3]
    t["T3"] = None if any(v is None for v in t3) else all(t3)
    t4 = [ok(s) for s in th if "s0.1+m0.1+e0.01" in s]
    t["T4"] = None if not t4 else (bool(t["T3"]) and all(t4))
    t["n_t4"] = len(t4)
    return t


def best(t: dict) -> str:
    out = "-"
    for k in ("T0", "T1", "T2", "T3", "T4"):
        if t.get(k):
            out = k
        else:
            break
    return out


def load(paths):
    heads, threats = {}, defaultdict(dict)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("mode") == "head":
                heads[(r["circuit"], r["config"])] = r
            elif r.get("mode") == "threat":
                threats[(r["circuit"], r["config"])][r["threat"]] = r
    return heads, threats


def rows(paths):
    heads, threats = load(paths)
    out = []
    for key, h in heads.items():
        c, cfg = key
        d0, n0, s0 = MAIN[c]
        th = threats.get(key, {})
        t = tiers(th)
        fails = [s for s, r in th.items() if not r.get("correct")]
        worst = {s: round(r["margin"], 4) for s, r in th.items() if s in T3 or "exact" in s}
        out.append({
            "circuit": c, "config": cfg, "depth": h["depth"], "dense": h["dense"],
            "sparse": h["sparse"], "x_dense": round(n0 / h["dense"], 3),
            "x_sparse": round(s0 / h["sparse"], 3), "depth_main": d0,
            "tier": best(t) if th else "size only", "flags": t, "fails": fails,
            "margins": worst, "fp16_bound": h.get("fp16_bound"),
            "n_threats": len(th), "n_inputs": h.get("n_inputs"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rs = rows(a.paths)
    order = ["L_hardened", "L_robust", "L_O2", "L_O3", "hardened", "robust", "main",
             "main_bf16", "O1", "O1_bf16", "O2", "O3"]
    rs.sort(key=lambda r: (list(MAIN).index(r["circuit"]),
                           order.index(r["config"]) if r["config"] in order else 99,
                           r["config"]))
    print("| circuit | config | depth | dense | x dense | sparse | x sparse | tier | fails |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        f = ", ".join(r["fails"][:3]) + (" ..." if len(r["fails"]) > 3 else "")
        print(f"| {r['circuit']} | {r['config']} | {r['depth']} | {r['dense']:,} | "
              f"{r['x_dense']} | {r['sparse']:,} | {r['x_sparse']} | {r['tier']} | {f} |")
    if a.json:
        json.dump(rs, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

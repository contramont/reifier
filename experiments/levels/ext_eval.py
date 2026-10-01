"""Extended input (wave 7 merge, optional top of the ladder; changes the input interface):
the host appends input_zeros always-0 features to [BOS, x], so the first layer's rows can
sum to 0 like every later layer's and a LayerNorm shift cancels there too (the lead's
note). Runs the T6 / T7 points (and parts) on fresh inputs (ladder_eval.fresh), with the
host's perturbations applied to every input feature, the zeros included.

  PYTHONPATH=<repo>/src:<repo>/experiments/levels python ext_eval.py --circuits adder32 \
      --configs t6x --specs "f32+s0.01+m0.1+e0.01+M0.3+n0.03" --seeds 0-7 --refs R --out O
"""
import argparse
import json
import time
import warnings

import torch as t

import ladder_eval as le
import noise_eval
import robust_eval as rv
import suite
from reifier.opt import fp16_bound
from reifier.utils.format import Bits

T6P = "s0.01+m0.1+e0.01+M0.3+n0.03"
T7P = "s0.002+m0.1+e0.01+M1+n0.1"
CONFIGS = {
    # toward_t6 (steps, rep 2, 16 always-0 features, a BOS copy per 16 features, q 512)
    # reading 16 host zeros: its first layer is shift-invariant too
    "t6x": dict(noise_eval.CONFIGS["toward_t6"], input_zeros=16),
    "toward_t6": dict(noise_eval.CONFIGS["toward_t6"]),
    # the ultra level's recipe reading 8 host zeros
    "ultrax": {"level": "ultra", "input_zeros": 8},
    # toward_t7 with 32 host zeros, and with each input bit given 4 / 16 times
    "t7x": dict(noise_eval.CONFIGS["toward_t7"], input_zeros=32),
    "t7x4": dict(noise_eval.CONFIGS["toward_t7"], input_zeros=32, input_rep=4),
    "t7x16": dict(noise_eval.CONFIGS["toward_t7"], input_zeros=32, input_rep=16),
    "t7x16b": dict(noise_eval.CONFIGS["toward_t7"], input_zeros=32, input_rep=16,
                   input_bos=16),
    # toward_t6 reading 16 host zeros and each input bit twice
    "t6x2": dict(noise_eval.CONFIGS["toward_t6"], input_zeros=16, input_rep=2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--specs", required=True, help="robust_eval specs without seed, ',' separated")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--seed", type=int, default=3100)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    out = open(a.out, "a")
    for name in a.circuits.split(","):
        X, Y = le.refs(name, a.seed, a.refs)
        c = suite.SUITE[name]()
        for cfg in a.configs.split(","):
            kw = dict(CONFIGS[cfg])
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp = rv.compiler(kw).run(c.fn, x=Bits("0" * c.n_in).bitlist)
            k, r = int(kw.get("input_zeros", 0)), int(kw.get("input_rep", 1))
            nb = int(kw.get("input_bos", 1))
            Xk = t.cat([X[:, :1], X[:, 1:].repeat_interleave(r, 1)], 1) if r > 1 else X
            Xk = t.cat([Xk, t.ones(len(X), nb - 1)], 1) if nb > 1 else Xk
            Xk = t.cat([Xk, t.zeros(len(X), k)], 1) if k else Xk
            ps = list(mlp.parameters())
            head = {"circuit": name, "config": cfg, "mode": "head", "input_zeros": k,
                    "input_rep": r,
                    "dense": sum(p.numel() for p in ps), "depth": len(mlp.layers),
                    "fp16_bound": round(fp16_bound(mlp), 1),
                    "compile_s": round(time.time() - t0, 1)}
            print(json.dumps(head), flush=True)
            out.write(json.dumps(head) + "\n")
            L = rv.layers_of(mlp)
            for spec in a.specs.split(","):
                fails = 0
                for sd in range(lo, hi + 1):
                    r = noise_eval.evaluate(L, Xk, Y, f"{spec}+seed{sd}")
                    fails += not r.get("correct", False)
                    rec = {"circuit": name, "config": cfg, "mode": "threat", **r}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                print(json.dumps({"circuit": name, "config": cfg, "spec": spec,
                                  "runs": hi - lo + 1, "fail": fails}), flush=True)


if __name__ == "__main__":
    main()

"""For relaxation hits (gate pairs), test real knot offsets beta on a grid with the true unit
columns relu(s - tau + beta) (E) x (1, E)."""
import os, sys, re, ast, itertools
os.environ["OMP_NUM_THREADS"] = "1"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sys.argv = [sys.argv[0], str(W), "2"]
import dand4 as D
import numpy as np
from fractions import Fraction
hits = []
for line in open(sys.argv[0].replace("betacheck.py", "") + (os.environ.get("LOG", "w2_r2_rel.log"))):
    if line.startswith("HIT"):
        m = re.match(r"HIT \((\(.*?\)), (-?\d+), (\(.*?\)), (-?\d+),", line)
        hits.append((ast.literal_eval(m.group(1)), int(m.group(2)), ast.literal_eval(m.group(3)), int(m.group(4))))
den = int(os.environ.get("DEN", "4"))
betas = sorted({Fraction(k, d) for d in range(1, den + 1) for k in range(0, d)})
print(len(hits), "pairs,", len(betas), "betas each", flush=True)
found = 0
for (w1, t1, w2, t2) in hits:
    s1 = D.X @ np.array(w1, float); s2 = D.X @ np.array(w2, float)
    for b1 in betas:
        r1 = np.maximum(0, s1 - t1 + float(b1))
        for b2 in betas:
            r2 = np.maximum(0, s2 - t2 + float(b2))
            M = np.hstack([r1[:, None] * D.Xt, r2[:, None] * D.Xt, D.Xt])
            act = np.stack([r1 > 0, r2 > 0], 1).astype(int)
            for flip in D.FLIPS:
                out = D.feasible(M, flip, act)
                if out is not None and not isinstance(out, str):
                    found += 1
                    print("REAL", w1, t1, b1, w2, t2, b2, flip, [int(round(v)) for v in out], flush=True)
print("found", found)

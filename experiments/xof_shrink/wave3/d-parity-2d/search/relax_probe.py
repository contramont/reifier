import numpy as np, itertools, collections
from gen_active import gen
from lines import PTS
act_sets = gen(10)
items = list(act_sets.items())
X = np.array([p[0] for p in PTS], float); Y = np.array([p[1] for p in PTS], float)
TGT = (X + Y) % 2
QUAD = np.stack([np.ones(36), X, Y, X * X, X * Y, Y * Y], 1)
S = np.array([k for k, _ in items], float)
typ = ["".join(sorted(v[1])) for _, v in items]
bytype = collections.defaultdict(list)
for i, t in enumerate(typ): bytype[t].append(i)
cfgs = [("BL", "BL", "RT", "RT"), ("BL", "BR", "LT", "RT"), ("BR", "BR", "LT", "LT"), ("BT", "BT", "LR", "LR"), ("BT", "LR", "BL", "RT"), ("BT", "LR", "BR", "LT")]
rng = np.random.default_rng(0)
def resid(M, t):
    s = np.linalg.lstsq(M, t, rcond=None)[0]; return np.abs(M @ s - t).max()
for cfg in cfgs:
    npass = 0; N = 3000
    for _ in range(N):
        ls = [rng.choice(bytype[t]) for t in cfg]
        M = np.concatenate([np.ones((36, 1))] + [S[l][:, None] * QUAD for l in ls], 1)
        npass += resid(M, TGT) < 1e-7
    print(cfg, "relaxed pass fraction", npass / N)

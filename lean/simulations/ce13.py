"""CE-13: RoPE angles pos * turn in float32. The score a head at p gives the position p' is
A sum_i cos(phi_i(p) - lb_i(h) - phi_i(p')), phi_i(pos) = fl32(pos * fl32(turn_i)), lb the lookback
angle h * turn_i (in fp32 too). Exact: A sum_i cos((p - h - p') turn_i). The gap between the target
p' = p - h and the best other visible p' is 8 A (1 - runnerUp) = 30 in exact arithmetic."""
import numpy as np

P, C = 8, 512
t = 10000.0 ** (-np.arange(P) / P)
t32 = t.astype(np.float32)
mc = np.cos(np.outer(np.arange(1, C), t)).mean(1)
A = 30.0 / (P * (1 - mc.max()))


def ang32(pos):
    return (np.float32(pos) * t32).astype(np.float32).astype(np.float64)


def gap(p, fp32=True):
    hs = np.arange(min(C, p + 1))
    pp = p - np.arange(min(C, p + 1))            # visible positions, offset 0..511
    if fp32:
        phq = ang32(p)
        lb = (hs[:, None].astype(np.float32) * t32[None, :]).astype(np.float32).astype(np.float64)
        phk = np.stack([ang32(q) for q in pp])   # [offset, i]
    else:
        phq = p * t
        lb = hs[:, None] * t[None, :]
        phk = pp[:, None] * t[None, :]
    # score[h, offset]
    ang = (phq[None, None, :] - lb[:, None, :]) - phk[None, :, :]
    sc = A * np.cos(ang).sum(-1)
    tgt = sc[hs, hs]
    sc2 = sc.copy()
    sc2[hs, hs] = -np.inf
    return float((tgt - sc2.max(1)).min())


print(f"A = {A:.4f}; exact gap at p = 600: {gap(600, False):.6f}")
budget = 30 - np.log(1e-10 / (1022 * np.exp(-30)))
print(f"the proof's attention budget 1e-10 needs a gap of at least {budget:.4f}; bf16 rounds a "
      f"head's leak away while 511 e^-gap < 2^-10, a gap of at least {np.log(511 * 2**10):.2f}")
for k in range(9, 27):
    for p in (2 ** k, 3 * 2 ** (k - 1)):
        g = gap(p)
        print(f"  p = {p:9d}: gap {g:9.4f}  (lost {30 - g:8.4f})")

print("\nrandom positions, 12 per octave: the smallest gap in each octave")
rng = np.random.default_rng(0)
for k in range(13, 26):
    ps = rng.integers(2 ** k, 2 ** (k + 1), 12)
    gs = [gap(int(p)) for p in ps]
    print(f"  p in [2^{k}, 2^{k+1}): smallest gap {min(gs):9.4f}, largest {max(gs):9.4f}")
for p in (2 ** 24 - 1, 2 ** 24 + 1, 2 ** 24 + 511, 2 ** 24 + 1000):
    print(f"  p = {p}: gap {gap(p):9.4f}")

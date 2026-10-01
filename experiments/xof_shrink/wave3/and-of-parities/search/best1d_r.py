"""Collect exact 1-D parity forms for (n, K) from many VP restarts, polish them, and rank by
robustness: the smallest distance of a knot to an integer (bigger is better), then by the size
of the cancelling terms. Prints the best forms as JSON lines."""
import sys, math, json, numpy as np, torch as t
import polish1d
from polish1d import polish, solve, units, mp_check
t.set_default_dtype(t.float64)
t.set_num_threads(2)
n, K = int(sys.argv[1]), int(sys.argv[2])
B = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
R = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # ramp units (value = constant)
polish1d.RAMPS[0] = R
steps = 1500
s = t.arange(n + 1, dtype=t.float64)
T = s % 2
cands = []
for seed in range(seeds):
    t.manual_seed(100 + seed)
    sig = (t.randint(0, 2, (B, K)) * 2 - 1).double()
    kn = (t.rand(B, K) * (n - 1) + 0.5).requires_grad_(True)
    opt = t.optim.Adam([kn], lr=0.02)
    for st in range(steps):
        beta = 4.0 * math.exp(st / steps * math.log(500.0))
        g = sig[:, None, :] * (s[None, :, None] - kn[:, None, :])
        r = t.nn.functional.softplus(beta * g) / beta
        Phi = t.cat([t.ones(B, n + 1, 1), r, r[:, :, :K - R] * s[None, :, None]], 2)
        A = Phi.transpose(1, 2) @ Phi + 1e-10 * t.eye(2 * K + 1 - R)
        c = t.linalg.solve(A, Phi.transpose(1, 2) @ T[None, :, None])
        loss = ((T[None] - (Phi @ c).squeeze(2)) ** 2).sum(1)
        opt.zero_grad(); loss.sum().backward(); opt.step()
    with t.no_grad():
        g = sig[:, None, :] * (s[None, :, None] - kn[:, None, :])
        rr = t.relu(g)
        Phi = t.cat([t.ones(B, n + 1, 1), rr, rr[:, :, :K - R] * s[None, :, None]], 2)
        sol = t.linalg.lstsq(Phi, T.expand(B, -1)[..., None], driver="gelsd").solution
        e = ((Phi @ sol).squeeze(2) - T).abs().max(1).values
    for i in (e < 1e-6).nonzero().flatten().tolist():
        cands.append((kn[i].detach().numpy().copy(), sig[i].int().numpy().copy()))
print(f"n={n} K={K} exact candidates {len(cands)}", flush=True)
seen, res = set(), []
for kn0, sg in cands:
    try:
        kp = polish(n, kn0, sg)
    except Exception:
        continue
    x, r = solve(n, kp, sg)
    if np.abs(r).max() > 1e-10:
        continue
    key = tuple(np.round(np.sort(kp * sg), 4))
    if key in seen:
        continue
    seen.add(key)
    dist = min(min(abs(k - tt) for tt in range(n + 1)) for k in kp)
    if min(kp) < -1e-9 or max(kp) > n + 1e-9:
        continue  # knot outside the range: that unit is always on or off (degenerate)
    us, c = units(n, kp, sg, x)
    big = max(abs(c), max(abs(max(0, gw * ss + gb) * (vw * ss + vb)) for gw, gb, vw, vb in us for ss in range(n + 1)))
    # slope of the form at the integers (noise gain on count inputs); one-sided max
    slope = 0.0
    for ss in range(n + 1):
        for side in (-1e-7, 1e-7):
            f0 = c + sum(max(0, gw * ss + gb) * (vw * ss + vb) for gw, gb, vw, vb in us)
            f1 = c + sum(max(0, gw * (ss + side) + gb) * (vw * (ss + side) + vb) for gw, gb, vw, vb in us)
            slope = max(slope, abs(f1 - f0) / 1e-7)
    res.append((dist, -big, kp.tolist(), sg.tolist(), us, c, float(np.abs(r).max()), slope))
DM = float(__import__("os").environ.get("DMIN", "0.05"))
r1 = sorted([z for z in res if z[0] >= DM], key=lambda z: -z[1])[:6]  # smallest cancelling terms
r2 = sorted(res, key=lambda z: (-z[0], -z[1]))[:2]  # farthest knots
print(f"distinct exact forms {len(res)}")
r3 = sorted(res, key=lambda z: (z[7], -z[1]))[:6]  # flattest at the integers
for dist, nb, kp, sg, us, c, err, sl in r1 + r2 + r3:
    print(json.dumps({"n": n, "K": K, "min_knot_dist": dist, "max_term": -nb, "max_slope": sl, "knots": kp, "dirs": sg,
                      "units": us, "const": c, "err": err}))

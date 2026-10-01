"""1-D: parity of an integer count s in [0, n] as c + sum_k relu(sig_k (s - kn_k)) (a_k + b_k s).
Knots real (optimized), directions sig_k in {+1,-1} (random per restart), values by least squares.
Finds the fewest units K per n. Prints the best knots/directions and residual."""
import sys, math, numpy as np, torch as t
t.set_default_dtype(t.float64)
t.set_num_threads(2)
n, K = int(sys.argv[1]), int(sys.argv[2])
B = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
t.manual_seed(seed)
s = t.arange(n + 1, dtype=t.float64)
T = s % 2
sig = (t.randint(0, 2, (B, K)) * 2 - 1).double()
kn = (t.rand(B, K) * (n - 1) + 0.5)
kn.requires_grad_(True)
opt = t.optim.Adam([kn], lr=0.02)
def resid(kn, beta=None):
    g = sig[:, None, :] * (s[None, :, None] - kn[:, None, :])  # B x N x K
    r = t.relu(g) if beta is None else t.nn.functional.softplus(beta * g) / beta
    Phi = t.cat([t.ones(B, n + 1, 1), r, r * s[None, :, None]], 2)
    A = Phi.transpose(1, 2) @ Phi + 1e-10 * t.eye(2 * K + 1)
    c = t.linalg.solve(A, Phi.transpose(1, 2) @ T[None, :, None])
    return T[None] - (Phi @ c).squeeze(2)
for st in range(steps):
    beta = 4.0 * math.exp(st / steps * math.log(500.0))
    r = resid(kn, beta)
    loss = (r ** 2).sum(1)
    opt.zero_grad(); loss.sum().backward(); opt.step()
with t.no_grad():
    # exact check with lstsq
    g = sig[:, None, :] * (s[None, :, None] - kn[:, None, :])
    rr = t.relu(g)
    Phi = t.cat([t.ones(B, n + 1, 1), rr, rr * s[None, :, None]], 2)
    sol = t.linalg.lstsq(Phi, T.expand(B, -1)[..., None], driver="gelsd").solution
    e = ((Phi @ sol).squeeze(2) - T).abs().max(1).values
o = e.argsort()
print(f"n={n} K={K} best max err {e[o[0]].item():.3e}  #<1e-9: {int((e<1e-9).sum())}")
for i in o[:3].tolist():
    print("  err %.2e knots %s dirs %s" % (e[i].item(), np.round(kn[i].detach().numpy(), 5).tolist(), sig[i].int().tolist()))

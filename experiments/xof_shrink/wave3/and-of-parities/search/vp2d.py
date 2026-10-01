"""2-D count model of the AND of two parities: b = |x_B|, c = |x_C| (counts of disjoint raw sets),
target (1 - p(b)) p(c) on the grid [0, mB] x [0, mC], free columns [1, p(b), p(c)] (the singles),
K units relu(al b + be c + ga) (u + v b + w c) with REAL gate directions and offsets (all optimized),
values by least squares. Many random restarts; softplus annealed to relu; exact check with relu."""
import sys, math, json, numpy as np, torch as t
t.set_default_dtype(t.float64)
t.set_num_threads(2)
mB, mC, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
B = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
free_kind = sys.argv[7] if len(sys.argv) > 7 else "singles"
t.manual_seed(seed)
bb, cc = t.meshgrid(t.arange(mB + 1.), t.arange(mC + 1.), indexing="ij")
b, c = bb.flatten(), cc.flatten()
N = len(b)
pb, pc = b % 2, c % 2
T = (1 - pb) * pc
if free_kind == "singles":
    F = t.stack([t.ones(N), pb, pc], 1)
elif free_kind == "sep":  # any function of b alone + any function of c alone
    F = t.cat([t.nn.functional.one_hot(b.long(), mB + 1).double(), t.nn.functional.one_hot(c.long(), mC + 1).double()[:, 1:]], 1)
X = t.stack([t.ones(N), b, c], 1)  # N x 3
# gate init: random direction, offset through the grid
ang = t.rand(B, K) * 2 * math.pi
W = t.zeros(B, K, 3)
W[:, :, 1] = t.cos(ang); W[:, :, 2] = t.sin(ang)
pt = t.stack([t.rand(B, K) * mB, t.rand(B, K) * mC], 2)
W[:, :, 0] = -(W[:, :, 1] * pt[..., 0] + W[:, :, 2] * pt[..., 1])
W.requires_grad_(True)
opt = t.optim.Adam([W], lr=0.02)
def design(W, beta=None):
    G = t.einsum("nd,bkd->bnk", X, W)
    R = t.relu(G) if beta is None else t.nn.functional.softplus(beta * G) / beta
    U = (R[..., None] * X[None, :, None, :]).reshape(W.shape[0], N, K * 3)
    return t.cat([F.expand(W.shape[0], -1, -1), U], 2)
for st in range(steps):
    beta = 4.0 * math.exp(st / steps * math.log(500.0))
    Phi = design(W, beta)
    A = Phi.transpose(1, 2) @ Phi
    M = A.shape[-1]
    A = A + 1e-10 * t.eye(M)
    coef = t.linalg.solve(A, Phi.transpose(1, 2) @ T[None, :, None])
    loss = ((T[None] - (Phi @ coef).squeeze(2)) ** 2).sum(1)
    opt.zero_grad(); loss.sum().backward(); opt.step()
with t.no_grad():
    Phi = design(W)
    sol = t.linalg.lstsq(Phi, T.expand(B, -1)[..., None], driver="gelsd").solution
    e = ((Phi @ sol).squeeze(2) - T).abs().max(1).values
o = e.argsort()
print(f"AND {mB}+{mC} K={K} free={free_kind}: best max err {e[o[0]].item():.3e}  #<1e-9: {int((e < 1e-9).sum())}  (pair parity n={mB+mC} needs {max(1,(mB+mC-1)//2)} with the p1d/min-parity forms)")
for i in o[:3].tolist():
    Wi = W[i].detach()
    Wi = Wi / Wi[:, 1:].norm(dim=1, keepdim=True)
    print("  err %.2e gates(ga,al,be) %s" % (e[i].item(), np.round(Wi.numpy(), 4).tolist()))

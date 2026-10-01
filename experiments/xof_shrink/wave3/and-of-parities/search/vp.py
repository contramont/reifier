"""Variable-projection search: is target T (on the points X of a cube / grid) exactly a
linear combination of `free` columns and K gated units relu(g_k . [1,x]) * (v_k . [1,x])?
Given the gates, the values and the free coefficients are a linear least-squares problem,
so only the K gates are optimized (Adam, many random restarts in one batch, softplus
annealed to relu). Reports the best exact residual (max abs error with relu, lstsq)."""
import itertools, math, sys, time, argparse
import numpy as np
import torch as t

t.set_default_dtype(t.float64)


def cube(n):
    return t.tensor(list(itertools.product([0, 1], repeat=n)), dtype=t.float64)


def design(Xb, W, free, beta=None):
    # Xb: N x d ; W: B x K x d ; free: N x f
    G = t.einsum("nd,bkd->bnk", Xb, W)
    R = t.relu(G) if beta is None else t.nn.functional.softplus(beta * G) / beta
    B, N, K = R.shape
    U = (R[..., None] * Xb[None, :, None, :]).reshape(B, N, K * Xb.shape[1])
    return t.cat([free.expand(B, -1, -1), U], 2)


def residual(Phi, T, lam=1e-9):
    A = Phi.transpose(1, 2) @ Phi
    M = A.shape[-1]
    A = A + lam * t.eye(M) * (1 + A.diagonal(dim1=1, dim2=2).mean(1, keepdim=True)[..., None])
    c = t.linalg.solve(A, Phi.transpose(1, 2) @ T[None, :, None])
    r = T[None, :] - (Phi @ c).squeeze(2)
    return r, c


def exact_check(Xb, W, free, T):
    Phi = design(Xb, W, free)
    out = []
    for b in range(Phi.shape[0]):
        sol = t.linalg.lstsq(Phi[b], T[:, None], driver="gelsd").solution
        out.append((Phi[b] @ sol).squeeze(1).sub(T).abs().max().item())
    return t.tensor(out)


def run(X, T, free, K, B=512, steps=3000, lr=0.05, seed=0, wscale=1.0, verbose=True, tag=""):
    t.manual_seed(seed)
    N, n = X.shape
    Xb = t.cat([t.ones(N, 1), X], 1)
    # random gates: weights ~ N(0, wscale), bias centred so the hyperplane passes the cloud
    W = t.randn(B, K, n + 1) * wscale
    mid = X.mean(0)
    W[:, :, 0] = -(W[:, :, 1:] * (mid + (X.std(0) * t.randn(B, K, n)) * 0.7)).sum(2)
    W.requires_grad_(True)
    opt = t.optim.Adam([W], lr=lr)
    best = None
    for s in range(steps):
        frac = s / steps
        beta = 2.0 * math.exp(frac * math.log(2000.0))  # 2 -> 4000
        Phi = design(Xb, W, free, beta)
        r, _ = residual(Phi, T)
        loss = (r ** 2).mean(1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        if verbose and (s % max(1, steps // 6) == 0 or s == steps - 1):
            with t.no_grad():
                e = exact_check(Xb, W.detach()[:64], free, T)
            print(f"{tag} step {s} beta {beta:.0f} loss min {loss.min().item():.3e} med {loss.median().item():.3e} exact(first64) min {e.min().item():.3e}", flush=True)
    with t.no_grad():
        e = exact_check(Xb, W.detach(), free, T)
    return W.detach(), e


def parity_target(X, S):
    return X[:, S].sum(1) % 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="parity")  # parity | and
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--nb", type=int, default=3)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wscale", type=float, default=1.0)
    ap.add_argument("--free_affine", type=int, default=0)
    ap.add_argument("--free", default="singles")  # singles | sep (all functions of x_B alone and of x_C alone)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    t.set_num_threads(a.threads)
    if a.kind == "parity":
        X = cube(a.n)
        T = parity_target(X, list(range(a.n)))
        free = t.ones(len(X), 1)
    else:
        n = a.nb + a.nc
        X = cube(n)
        pB = parity_target(X, list(range(a.nb)))
        pC = parity_target(X, list(range(a.nb, n)))
        T = (1 - pB) * pC
        free = t.stack([t.ones(len(X)), pB, pC], 1)
        if a.free == "sep":
            iB = (X[:, :a.nb] * (2 ** t.arange(a.nb))).sum(1).long()
            iC = (X[:, a.nb:] * (2 ** t.arange(a.nc))).sum(1).long()
            oB = t.nn.functional.one_hot(iB, 2 ** a.nb).double()
            oC = t.nn.functional.one_hot(iC, 2 ** a.nc).double()[:, 1:]
            free = t.cat([oB, oC], 1)
    if a.free_affine:
        free = t.cat([free, X], 1)
    t0 = time.time()
    W, e = run(X, T, free, a.K, B=a.B, steps=a.steps, lr=a.lr, seed=a.seed, wscale=a.wscale,
               tag=f"{a.kind} n={X.shape[1]} K={a.K}")
    order = e.argsort()
    print("best exact residuals:", [f"{v:.2e}" for v in e[order[:8]].tolist()], f"time {time.time()-t0:.0f}s")
    print("n_exact(<1e-7):", int((e < 1e-7).sum()))
    b = order[0].item()
    print("best gates:", np.round(W[b].numpy(), 4).tolist())

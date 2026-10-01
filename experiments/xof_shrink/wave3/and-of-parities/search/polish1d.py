"""Polish a 1-D parity form (knots + directions) to float64 precision and emit reifier units.
Usage: polish1d.py n "k1,k2,..." "d1,d2,..." -> prints a python literal (units, const)
units: (gate_w, gate_b, value_w, value_b) meaning relu(gate_w*s + gate_b) * (value_w*s + value_b),
with gates scaled so every integer in [0, n] is >= 1 away from the knot in gate units."""
import sys, json
import numpy as np
import mpmath as mp
mp.mp.dps = 50


RAMPS = [0]  # the last RAMPS[0] units have constant values (set by best1d_r)


def design(n, kn, sig):
    s = np.arange(n + 1, dtype=float)
    cols = [np.ones(n + 1)]
    K = len(kn)
    for j, (k, d) in enumerate(zip(kn, sig)):
        r = np.maximum(0, d * (s - k))
        cols += [r, r * s] if j < K - RAMPS[0] else [r, 0 * r]
    return np.stack(cols, 1)


def solve(n, kn, sig):
    M = design(n, kn, sig)
    T = np.arange(n + 1) % 2
    x, *_ = np.linalg.lstsq(M, T, rcond=None)
    return x, M @ x - T


def polish(n, kn, sig, iters=60):
    """Levenberg-Marquardt on the knots (numerical Jacobian), numpy only"""
    k = np.array(kn, float)
    f = lambda kk: solve(n, kk, sig)[1]
    r = f(k)
    lam = 1e-3
    for _ in range(iters):
        if np.abs(r).max() < 1e-15:
            break
        J = np.zeros((len(r), len(k)))
        for j in range(len(k)):
            h = 1e-7 * max(1.0, abs(k[j]))
            kp = k.copy(); kp[j] += h
            J[:, j] = (f(kp) - r) / h
        A = J.T @ J
        g = J.T @ r
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
        k2 = k + step
        r2 = f(k2)
        if (r2 ** 2).sum() < (r ** 2).sum():
            k, r, lam = k2, r2, lam * 0.3
        else:
            lam *= 10
    return k


def mp_check(n, kn, sig, x):
    # evaluate with the float64 coefficients in high precision: residual of the actual form
    err = 0
    for s in range(n + 1):
        v = mp.mpf(x[0])
        for j, (k, d) in enumerate(zip(kn, sig)):
            g = d * (mp.mpf(s) - mp.mpf(k))
            if g > 0:
                v += g * (mp.mpf(x[1 + 2 * j]) + mp.mpf(x[2 + 2 * j]) * s)
        err = max(err, abs(v - (s % 2)))
    return float(err)


def units(n, kn, sig, x):
    out = []
    for j, (k, d) in enumerate(zip(kn, sig)):
        a, b = x[1 + 2 * j], x[2 + 2 * j]
        # relu(d (s - k)) (a + b s); scale gate by lam = 1 / min dist to an integer
        dist = min(abs(t - k) for t in range(n + 1))
        lam = 1.0 / dist if dist < 1 else 1.0
        out.append((d * lam, -d * lam * k, b / lam, a / lam))
    return out, float(x[0])


if __name__ == "__main__":
    n = int(sys.argv[1])
    kn = [float(v) for v in sys.argv[2].split(",")]
    sig = [int(v) for v in sys.argv[3].split(",")]
    kp = polish(n, kn, sig)
    x, r = solve(n, kp, sig)
    us, c = units(n, kp, sig, x)
    # re-evaluate the scaled units
    err = 0.0
    for s in range(n + 1):
        v = c + sum(max(0.0, gw * s + gb) * (vw * s + vb) for gw, gb, vw, vb in us)
        err = max(err, abs(v - s % 2))
    mind = min(abs(gw * s + gb) for gw, gb, _, _ in us for s in range(n + 1))
    print(json.dumps({"n": n, "K": len(us), "knots": kp.tolist(), "dirs": sig, "err": err,
                      "mp_err": mp_check(n, kp, sig, x), "min_gate_abs": mind, "units": us, "const": c}))

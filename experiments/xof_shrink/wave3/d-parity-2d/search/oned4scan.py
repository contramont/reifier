# scan real-knot 4-unit solutions of parity(s), s=0..10; score: knot distance to lattice, lattice slopes, coef size
import numpy as np, itertools, json
from scipy.optimize import least_squares
from multiprocessing import Pool
s = np.arange(11.0); t = s % 2
def M_of(tau, o):
    cols = [np.ones(11)]
    for tk, ok in zip(tau, o):
        r = np.maximum(0, ok * (s - tk)); cols += [r, r * s]
    return np.array(cols).T
def solve(tau, o):
    M = M_of(tau, o); sol = np.linalg.lstsq(M, t, rcond=None)[0]; return M @ sol - t, sol
def metrics(tau, o, sol):
    c0 = sol[0]; d = sol[1::2]; f = sol[2::2]
    # F'(s) at lattice points (units are smooth there as knots are off-lattice)
    der = np.zeros(11)
    for k in range(4):
        act = o[k] * (s - tau[k]) > 0
        der += act * (o[k] * (d[k] * s + f[k]) + o[k] * (s - tau[k]) * d[k])
    dist = np.min(np.abs(np.array(tau)[:, None] - s[None]), 1).min()
    # max |unit contribution| at lattice points (float error scale)
    mx = max(np.abs(np.maximum(0, o[k] * (s - tau[k])) * (d[k] * s + f[k])).max() for k in range(4))
    return float(dist), float(np.abs(der).max()), float(np.abs(np.r_[d, f]).max()), float(mx)
def work(seed):
    rng = np.random.default_rng(seed)
    out = []
    for o in itertools.product((1, -1), repeat=4):
        for trial in range(60):
            tau0 = np.sort(rng.uniform(0.2, 9.8, 4))
            r = least_squares(lambda z: solve(z, o)[0], tau0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
            res, sol = solve(r.x, o)
            if np.abs(res).max() < 1e-11:
                out.append((o, r.x.tolist(), sol.tolist(), metrics(r.x, o, sol)))
    return out
if __name__ == "__main__":
    with Pool(12) as p:
        allr = sum(p.map(work, range(24)), [])
    print("solutions", len(allr))
    good = [x for x in allr if x[3][0] > 0.3]
    good.sort(key=lambda x: (x[3][1], x[3][2]))
    for o, tau, sol, m in good[:12]:
        print(o, np.round(tau, 5).tolist(), "dist %.3f maxslope %.3f maxcoef %.2f maxunit %.1f" % m)
    json.dump(allr, open("oned4_solutions.json", "w"))

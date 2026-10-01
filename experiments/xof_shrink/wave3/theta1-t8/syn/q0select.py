import numpy as np, re, ast
n = 7; ts = np.arange(n + 1.); par = ts % 2; alt = (-1.0) ** ts
rows = []
for l in open("logs/q0D_7.log"):
    if not l.startswith("HIT"):
        continue
    m = re.match(r"HIT pool (\(.*\)) A=1 knots (\[.*?\]) (\S+)", l)
    pool = ast.literal_eval(m.group(1)); t1 = np.array(ast.literal_eval(m.group(2)))
    s = np.array([p[0] for p in pool]); tau = np.array([p[1] for p in pool])
    r0 = np.maximum(0, s[:, None] * (ts - tau[:, None]))
    M = np.concatenate([r0.T, (r0 * ts).T, np.ones((n + 1, 1))], 1)
    sol = np.linalg.lstsq(M, par, rcond=None)[0]
    C0, C1, kap = sol[0:3], sol[3:6], sol[6]
    v = C1[:, None] * ts + C0[:, None]; z = r0 * v
    r1 = np.maximum(0, s[:, None] * (ts - t1[:, None]))
    MD = np.concatenate([(r1 * v - z).T, r1.T], 1)
    e = np.linalg.lstsq(MD, alt, rcond=None)[0]
    lam, d = e[0:3], e[3:6]
    resid = np.abs(MD @ e - alt).max()
    # gate lattice distances (per unit, both slices), ignoring exact zeros
    g0 = s[:, None] * (ts - tau[:, None]); g1 = s[:, None] * (ts - t1[:, None])
    gall = np.abs(np.concatenate([g0, g1], 1))
    nz = np.where(gall > 1e-9, gall, np.inf).min(1)
    near = ((gall > 1e-9) & (gall < 0.08)).any()
    # magnitude of unit outputs (live: u_j + (1 - lam_j) z_j), zero lane z_j
    u1 = r1 * (lam[:, None] * v + d[:, None]); u0 = lam[:, None] * z
    mag = max(np.abs(u1).max(), np.abs(u0).max(), np.abs(z).max(), np.abs((1 - lam)[:, None] * z).max())
    rows.append((near, mag, resid, pool, t1.round(6).tolist(), nz.round(3).tolist(), lam.round(4).tolist(), d.round(4).tolist()))
rows.sort(key=lambda r: (r[0], r[1]))
for r in rows[:15]:
    print(r)

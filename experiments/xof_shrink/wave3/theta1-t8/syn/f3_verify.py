# Exact (sympy) verification of the T <= 7 pool-of-3 form F2 (pool = MINPAR7-type slices + constant)
import sympy as sp
R = sp.Rational; sq = sp.sqrt
pool = [(1, R(3, 2)), (1, R(23, 5)), (-1, R(3))]
C0 = [R(-47, 3), R(0), R(-25, 6)]; C1 = [R(8, 3), R(-5, 3), R(-19, 12)]; kappa = R(25, 2)
tau1 = [R(104311, 17024) - sq(9033204049) / 17024, R(142, 41), R(1429, 532) + sq(4134929) / 532]
lam = [R(-8, 13), R(0), R(24, 19)]
d = [R(4295, 10374) + sq(9033204049) / 10374, R(-328, 19), R(2829, 266) - sq(4134929) / 266]
def relu(x):
    return x if sp.simplify(x) > 0 else 0
ok = True
for T in range(8):
    z = [relu(s * (T - t)) * (C0[j] + C1[j] * T) for j, (s, t) in enumerate(pool)]
    zero = sum(z) + kappa
    ok &= sp.simplify(zero - (T % 2)) == 0
    for A in (0, 1):
        u = 0
        for j, (s, t) in enumerate(pool):
            g = s * (T - t) + s * (t - tau1[j]) * A
            v = lam[j] * (C0[j] + C1[j] * T) + d[j] * A
            u += relu(g) * v
        live = u + sum((1 - lam[j]) * z[j] for j in range(3)) + kappa
        e = sp.nsimplify(sp.simplify(live - ((A + T) % 2)))
        if e != 0:
            ok = False; print("FAIL", A, T, e)
print("F3 exact on {0,1} x [0,7]:", ok)
for j in range(3):
    print("unit", j, "tau1 =", sp.N(tau1[j], 25), "d =", sp.N(d[j], 25))

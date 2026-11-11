"""MainRealDeep against the CE-7 chains: error at depth n vs (1 + sigma)^n / 1e10, while
(1 + sigma)^n <= 1e4."""
import importlib.util
spec = importlib.util.spec_from_file_location('sim', 'sim.py')
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
mp = sim.mp
codes = ((0, [False]), (1, [True]))
toks = [0, 1]
for glu, sigma, name in ((sim.glu7, mp.mpf(5) / 3, 'glu7, sigma 5/3'), (sim.glu23, mp.mpf(23), 'glu23, sigma 23')):
    n = 1
    while (1 + sigma) ** n <= 10 ** 4:
        t = sim.chain(n, glu)
        M = sim.Transformer(t, codes)
        got = M.net.run(sim.gelu, M.after_attn(toks, 1))[-1][1]
        err = abs(got - sim.expected(t, codes, toks, 1)[0])
        bound = (1 + sigma) ** n / mp.mpf(10) ** 10
        print(f"{name}: n = {n}: error {mp.nstr(err, 5)}  bound {mp.nstr(bound, 5)}  within: {err <= bound}")
        n += 1

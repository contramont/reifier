"""Main's default compile (Compiler()) of suite circuits: depth, dense and nonzero
parameters, and a check on a few inputs.

usage: python probe_sizes.py [CIRCUIT ...]   (default: the whole suite)
"""
import sys
import time

from suite import SUITE, inputs, reference
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos
from reifier.utils.format import Bits


def main(names: list[str]) -> None:
    for name in names:
        t0 = time.time()
        c = SUITE[name]()
        mlp = Compiler().run(c.fn, x=Bits("0" * c.n_in).bitlist)
        t1 = time.time()
        ps = list(mlp.parameters())
        dense, sparse = sum(p.numel() for p in ps), sum(int((p != 0).sum()) for p in ps)
        xs = inputs(c.n_in, 6)
        ys = reference(c, xs)
        ok = all(infer_bits_bos(mlp, Bits(x)).ints == y for x, y in zip(xs, ys))
        print(f"{name:<18} n_in={c.n_in:<5} depth={len(mlp.layers):<4} dense={dense:>12,} "
              f"sparse={sparse:>10,} out={len(ys[0])} ok={ok} compile={t1 - t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SUITE))

"""Benchmark Keccak compilation across reifier's compilation paths.

Usage:
    python benchmarks/bench_keccak.py                 # fast path, full SHA3-224
    python benchmarks/bench_keccak.py --paths fast,sparse
    python benchmarks/bench_keccak.py --log-w 3 --rounds 3 --paths fast,sparse,tree

Paths:
    fast    reifier.fast with Keccak(stamp=True) (template stamping)
    sparse  legacy NodeGraph -> SparseGraph pipeline
    tree    TreeCompiler (sys.monitoring tracer); very slow at full size

All paths compile the same function (Keccak digest) into a leveled
threshold-gate graph; the compiled circuit is verified by evaluating it on a
real message and comparing with the directly computed digest.
"""

import argparse
import time

from reifier.examples.keccak import Keccak
from reifier.utils.format import Bits


def bench_fast(k_args: dict, msg: Bits, repeat: int) -> None:
    from reifier.fast import fast_compile

    k = Keccak(**k_args, stamp=True)
    dummy = Bits("0" * len(msg))
    expected = k.digest(msg).bitstr
    times = []
    arrays = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        arrays = fast_compile(k.digest, dummy)
        times.append(time.perf_counter() - t0)
    assert arrays is not None
    got = "".join(map(str, arrays.run(msg.ints)))
    assert got == expected, "compiled circuit disagrees with direct digest"
    label = ", ".join(f"{t:.3f}s" for t in times)
    print(
        f"fast:   [{label}] nodes={arrays.n_nodes} edges={arrays.n_edges} "
        f"levels={len(arrays.level_sizes)} (first run includes template setup)"
    )


def bench_sparse(k_args: dict, msg: Bits, repeat: int) -> None:
    from reifier.sparse.compile import compiled_from_io
    from reifier.sparse.sparse_graph import SparseGraph

    k = Keccak(**k_args)
    for _ in range(repeat):
        t0 = time.perf_counter()
        hashed = k.digest(msg)
        t1 = time.perf_counter()
        graph = compiled_from_io(msg.bitlist, hashed.bitlist)
        sg = SparseGraph.from_node_graph(graph)
        t2 = time.perf_counter()
        n_nodes = sum(len(lv.origins) for lv in sg.levels)
        print(
            f"sparse: build={t1 - t0:.3f}s level={t2 - t1:.3f}s "
            f"total={t2 - t0:.3f}s nodes={n_nodes} levels={len(sg.levels)}"
        )


def bench_tree(k_args: dict, msg: Bits, repeat: int) -> None:
    from reifier.compile.tree import TreeCompiler

    k = Keccak(**k_args)
    for _ in range(repeat):
        t0 = time.perf_counter()
        tree = TreeCompiler().run(k.digest, msg_bits=Bits("0" * len(msg)))
        dt = time.perf_counter() - t0
        n_nodes = sum(len(lv.origins) for lv in tree.levels)
        print(f"tree:   {dt:.3f}s nodes={n_nodes} levels={len(tree.levels)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-w", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=448)
    parser.add_argument("--paths", type=str, default="fast")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    k_args = dict(log_w=args.log_w, n=args.rounds, c=args.capacity, pad_char="_")
    k = Keccak(**k_args)
    msg = k.format("Rachmaninoff", clip=True)
    print(f"Keccak(log_w={args.log_w}, n={args.rounds}, c={args.capacity})")

    runners = {"fast": bench_fast, "sparse": bench_sparse, "tree": bench_tree}
    for name in args.paths.split(","):
        runners[name.strip()](k_args, msg, args.repeat)


if __name__ == "__main__":
    main()

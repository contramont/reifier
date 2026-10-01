import itertools
import math
import random

import pytest
import torch as t

from reifier.examples.keccak import Keccak
from reifier.neurons.core import Bit, Unit, const, gate, glu
from reifier.neurons.operations import add, and_, inhib, not_, or_, parity, xor
from reifier.opt import RECIPES
from reifier.opt.graph import GraphCompiler, GraphOptions, _xor_units
from reifier.tensors.compilation import Compiler
from reifier.tensors.matrices import Matrices
from reifier.tensors.mlp_utils import infer_bits_bos
from reifier.tensors.swiglu import MLP_SwiGLU
from reifier.utils.format import Bits

ONE, ZERO = const("10")
# the distinct graph passes of the recipes (and the base: folds and dedup only)
PASSES: dict[str, dict] = {"base": {}}
for name, (_, knobs) in RECIPES.items():
    if knobs["passes"] not in PASSES.values():
        PASSES[name] = dict(knobs["passes"])


def run_all(fn, n: int, mlp) -> None:
    for bits in itertools.product([0, 1], repeat=n):
        x = Bits(list(bits))
        assert infer_bits_bos(mlp, x).bitstr == Bits(fn(x.bitlist)).bitstr, x.bitstr


def compiled(fn, n: int, passes: dict, mlp_dtype: t.dtype = t.float32):
    """graph passes with the core's numerics (MLP_SwiGLU.from_matrices)"""
    graph = GraphCompiler(GraphOptions(**passes)).run(fn, x=Bits("0" * n).bitlist)
    return MLP_SwiGLU.from_matrices(Matrices.from_graph(graph), dtype=mlp_dtype)


def chi(x: list[Bit]) -> list[Bit]:
    a, b, c = x
    return [xor([a, inhib([b, c])])]


def sha_ch_maj(x: list[Bit]) -> list[Bit]:
    e, f, g = x
    ch = xor([and_([e, f]), and_([not_(e), g])])
    maj = xor([and_([e, f]), and_([e, g]), and_([f, g])])
    return [ch, maj]


def counts(x: list[Bit]) -> list[Bit]:
    odd = [Unit((1,), 0, (-2,), 4), Unit((1,), 0, (1,), -2), Unit((1,), -2, (0,), 4)]
    s = glu(x, [Unit((0, 0, 0), 1, (1, 1, 1), 0)], numeric=True)  # sum(x)
    return [glu([s], odd), not_(xor(x))]


def io_edge_cases(x: list[Bit]) -> list[Bit]:
    y = and_([x[0], x[1]])
    return [x[2], y, y, not_(y), ONE, ZERO, not_(x[0]), or_(x)]


CASES = {
    "gate_fixes": (lambda x: [gate([ONE, x[0], ZERO, x[1]], [-1, 2, 5, 1], 1),
                              gate([x[0], x[0], x[1]], [1, 1, -1], 2), inhib(x)], 2),
    "adder": (lambda x: add(x[:4], x[4:]), 8),
    "xor_parity": (lambda x: [xor(x), parity(x), not_(xor(x[:3]))], 6),
    "chi": (chi, 3),
    "sha_ch_maj": (sha_ch_maj, 3),
    "counts": (counts, 3),
    "io_edge_cases": (io_edge_cases, 3),
    "not_chain": (lambda x: [not_(not_(not_(x[0]))), and_([not_(x[0]), not_(x[1])])], 2),
}


@pytest.mark.parametrize("passes", list(PASSES))
@pytest.mark.parametrize("case", list(CASES))
def test_passes_exact(case, passes):
    fn, n = CASES[case]
    run_all(fn, n, compiled(fn, n, PASSES[passes]))


@pytest.mark.parametrize("passes", [p for p in PASSES if p != "O3"])
@pytest.mark.parametrize("case", ["adder", "chi", "sha_ch_maj", "xor_parity"])
def test_passes_bfloat16(case, passes):
    """steps stay exact bf16 steps; every unit but O3's meets the E-rule"""
    fn, n = CASES[case]
    run_all(fn, n, compiled(fn, n, PASSES[passes], mlp_dtype=t.bfloat16))


@pytest.mark.parametrize("passes", list(PASSES))
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_keccak_passes(passes, dtype):
    if dtype == t.bfloat16 and passes == "O3":
        pytest.skip("O3's wide glu_xor units are exact in float32 only")
    k = Keccak(log_w=1, n=3, c=20, pad_char="_")
    fn = lambda x: k.bitlist_to_digest(x)  # noqa: E731
    mlp = compiled(fn, k.msg_len, PASSES[passes], mlp_dtype=dtype)
    rng = random.Random(0)
    for _ in range(12):
        x = Bits([rng.randint(0, 1) for _ in range(k.msg_len)])
        assert infer_bits_bos(mlp, x).bitstr == Bits(fn(x.bitlist)).bitstr


def test_dead_code_is_not_compiled():
    def used(x: list[Bit]) -> list[Bit]:
        return add(x[:4], x[4:])

    def with_dead(x: list[Bit]) -> list[Bit]:
        add(x[4:], x[:4])  # traced, never read
        xor(x)
        return used(x)

    def size(fn):
        return [tuple(p.shape) for p in compiled(fn, 8, {}).parameters()]

    assert size(used) == size(with_dead)
    tree = Compiler().run(with_dead, x=Bits("0" * 8).bitlist)
    assert sum(p.numel() for p in tree.parameters()) > sum(
        p.numel() for p in compiled(with_dead, 8, {}).parameters()
    )


def test_xor_units_are_e_exact():
    """xor of 2-4 bits in one layer: at every sum each unit's gate is <= 0 or its
    value is 0, or both are powers of 2 (exact in bfloat16 relative to BOS)"""

    def pow2(v) -> bool:
        return v != 0 and math.frexp(abs(v))[0] == 0.5

    for k in (2, 3, 4):
        for s in range(k + 1):
            total = 0
            for g, gb, v, vb in _xor_units(k):
                G, V = sum(g[:s]) + gb, sum(v[:s]) + vb
                assert G <= -0.5 or G == 0 or V == 0 or (pow2(G) and pow2(V))
                total += max(0, G) * V
            assert total == s % 2


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("extra", [{"cheap_max": 2}, {"fold_bias": 2}])
def test_pass_options_exact(case, extra):
    """cheap units only for gates of <= cheap_max inputs; NOT folds that keep biases
    small (fold_bias)"""
    fn, n = CASES[case]
    for dtype in (t.float32, t.bfloat16):
        run_all(fn, n, compiled(fn, n, {"cheap": True, **extra}, mlp_dtype=dtype))


def test_cheap_max_shapes():
    """cheap_max keeps wide AND/OR gates as steps"""
    def fn(x):
        return [and_(x), or_(x[:2]), and_([not_(x[0]), x[1]])]

    def units(opts: GraphOptions) -> int:
        g = GraphCompiler(opts).run(fn, x=Bits("0" * 5).bitlist)
        return sum(len(o.units) for lv in g.levels for o in lv.origins)

    assert units(GraphOptions(cheap=True, cheap_max=2)) < units(GraphOptions(cheap=True))

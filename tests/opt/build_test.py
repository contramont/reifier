"""The SwiGLU constructions of reifier.opt.build: the T4 / T5 constructions under a
host's perturbations (always-0 features for LayerNorm shift invariance; ultra's BOS
copies, centered steps and per-layer output scales; ultra_s's fit16, every layer
rescaled to the float16 range), exact readout with gated units, and fp16_bound."""

import itertools
import math
import random

import pytest
import torch as t
import torch.nn.functional as F

from reifier.compile.tree import TreeCompiler
from reifier.neurons.core import Bit, const
from reifier.neurons.operations import add, and_, glu_xor, or_, xor
from reifier.opt import Compiler, fp16_bound
from reifier.opt.build import FP16_MAX, MLPOptions, build
from reifier.tensors.compilation import Compiler as CoreCompiler
from reifier.utils.format import Bits
from tests.opt.utils import correct, values
from tests.testing_utils import ratios

ULTRA_S = dict(level="ultra_s", select=False)  # the steps-only T5 recipe
FIT16 = dict(level="hardened", select=False, knobs={"fit16": True})  # rescaled hardened


def _circuit(x: list[Bit]) -> list[Bit]:
    return add(x[:12], x[12:24]) + [xor(x[:7]), and_(x[3:11]), or_(x[5:20])]


def _fn(x: list[Bit]) -> list[Bit]:
    return add(x[:8], x[8:16]) + [xor(x[:9]), and_(x[4:12]), or_(x[2:14])]


def _inputs(n: int, k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    xs = [[0] * n, [1] * n, [0] + [1] * (n - 1), [1] + [0] * (n - 1)]
    for i in range(k):
        d = (0.05, 0.5, 0.95)[i % 3]
        xs.append([int(rng.random() < d) for _ in range(n)])
    return xs


def _ref(fn, xs: list[list[int]]) -> t.Tensor:
    return t.tensor([values(fn(const(x))) for x in xs], dtype=t.float32)


def _forward(mlp, xs, dtype: t.dtype, s=1.0, m=0.0, e=0.0, n=0.0, M=0.0, seed=0) -> t.Tensor:
    """BOS-relative outputs under a host's perturbations: e / n noise before each layer
    (relative to the rms / the largest feature), a LayerNorm shift M (relative to the
    largest normalized feature) and m (in rms units), and norm scales log-uniform in
    [s, 1]"""
    gen = t.Generator().manual_seed(seed)
    x = t.tensor([[1] + v for v in xs], dtype=dtype)
    with t.inference_mode():
        for L in mlp.layers:
            nw, wg, wv, wo = (p.detach().to(dtype) for p in
                              (L.norm.weight, L.wg.weight, L.wv.weight, L.wo.weight))
            k = len(x)
            if e:
                rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
                x = (x.float() + e * rms * t.randn(x.shape, generator=gen)).to(dtype)
            if n:
                mx = x.float().abs().amax(-1, keepdim=True)
                x = (x.float() + n * mx * t.randn(x.shape, generator=gen)).to(dtype)
            hn = F.rms_norm(x, (x.size(-1),))
            u = t.randn(k, 1, generator=gen).to(dtype)
            h = (hn - M * u * hn.abs().amax(-1, keepdim=True)) * nw
            if m:
                h = h - m * t.randn(k, 1, generator=gen).to(dtype) * nw
            if s < 1:
                h = h * t.exp(t.rand(k, 1, generator=gen) * math.log(s)).to(dtype)
            x = F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wv), wo)
    out = x.float()
    return out[:, 1:] / out[:, :1]


def _scaled(mlp, xs, dtype=t.float32, s: float = 1.0) -> t.Tensor:
    """BOS-relative outputs with every layer's normalized features times s"""
    x = t.tensor([[1] + v for v in xs], dtype=dtype)
    with t.inference_mode():
        for L in mlp.layers:
            h = F.rms_norm(x, (x.size(-1),)) * L.norm.weight * s
            x = L.wo(F.silu(L.wg(h)) * L.wv(h))
    out = x.float()
    return out[:, 1:] / out[:, :1]


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_ultra_t5_point(dtype):
    """ultra under the T5 point (norm scale 0.05, LayerNorm shifts m 0.1 and M 0.1,
    interference e 0.01 and noise n 0.01 relative to the largest feature, together)"""
    mlp = Compiler("ultra").run(_circuit, x=Bits("0" * 24).bitlist)
    xs = _inputs(24, 60, 0)
    ref = _ref(_circuit, xs)
    assert correct(_forward(mlp, xs, dtype), ref)
    for seed in range(4):
        r = _forward(mlp, xs, dtype, s=0.05, m=0.1, e=0.01, n=0.01, M=0.1, seed=seed)
        assert correct(r, ref), seed


def test_ln_invariant_rows_and_shift():
    """rows after the first sum to 0 with the always-0 features, so a shift of all
    normalized features (up to 3 times the rms) cancels in every layer but the first"""
    mlp = Compiler("hardened").run(_circuit, x=Bits("0" * 24).bitlist)
    for L in list(mlp.layers)[1:]:
        for w in (L.wg.weight, L.wv.weight):
            assert w.sum(1).abs().max() < 1e-3 * w.abs().sum(1).max()
    xs = _inputs(24, 8, 3)
    x = t.tensor([[1] + v for v in xs], dtype=t.float32)
    gen = t.Generator().manual_seed(0)
    with t.inference_mode():
        for i, L in enumerate(mlp.layers):
            h = L.norm(x)
            if i > 0:
                h = h - 3 * t.randn(len(x), 1, generator=gen) * L.norm.weight
            x = L.wo(F.silu(L.wg(h)) * L.wv(h))
    assert correct(x[:, 1:] / x[:, :1], _ref(_circuit, xs))


@pytest.mark.parametrize("knobs", [
    dict(bos_copies=4, ln_invariant=16), dict(bos_copies=8, q=512),
    None,  # the tree compiler's graph, built with center, exact and out_scale 64
])
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_noise_knobs_exact(knobs, dtype):
    """the constructions keep the circuit exact on all inputs (no perturbation)"""
    def fn(x):
        return add(x[:4], x[4:]) + [xor(x[1:6])]

    if knobs is None:
        tree = TreeCompiler().run(fn, x=Bits("0" * 8).bitlist)
        mlp = build(tree, MLPOptions(center=True, exact=True, out_scale=64), dtype)
    else:
        mlp = Compiler("ultra", select=False, knobs=knobs, mlp_dtype=dtype).run(
            fn, x=Bits("0" * 8).bitlist)
    xs = [list(v) for v in itertools.product([0, 1], repeat=8)]
    assert correct(ratios(mlp, xs, dtype), _ref(fn, xs))


def test_ultra_fp16_wide_input():
    """the per-layer output scales keep 280-input circuits within float16 range"""
    def fn(x):
        return [xor(x[i:i + 4]) for i in range(0, 280, 7)] + [and_(x[:9])]

    mlp = Compiler("ultra", mlp_dtype=t.float16).run(fn, x=Bits("0" * 280).bitlist)
    xs = _inputs(280, 30, 1)
    r = ratios(mlp, xs, t.float16)
    assert t.isfinite(r).all() and correct(r, _ref(fn, xs))


def test_fit16_rescales_only_by_powers_of_2():
    """fit16 keeps every weight's support and changes each layer's norm weight, value
    rows and output rows by powers of 2"""
    a = Compiler("hardened", select=False).run(_fn, x=Bits("0" * 16).bitlist)
    b = Compiler(**FIT16).run(_fn, x=Bits("0" * 16).bitlist)
    assert len(a.layers) == len(b.layers)
    for La, Lb in zip(a.layers, b.layers):
        for name in ("wg", "wv", "wo"):
            wa, wb = getattr(La, name).weight, getattr(Lb, name).weight
            assert wa.shape == wb.shape and t.equal(wa != 0, wb != 0)
            ratio = (wb[wa != 0] / wa[wa != 0]).unique()
            assert len(ratio) == 1 and math.log2(ratio.item()).is_integer()
        nw = Lb.norm.weight.unique()
        assert len(nw) == 1 and math.log2(nw.item()).is_integer()
    assert fp16_bound(b) <= FP16_MAX


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16, t.float16])
def test_ultra_s_correct(dtype):
    xs = _inputs(16, 40, 1)
    mlp = Compiler(**ULTRA_S, mlp_dtype=dtype).run(_fn, x=Bits("0" * 16).bitlist)
    assert correct(_scaled(mlp, xs, dtype), _ref(_fn, xs))


def test_small_norm_scale():
    """a host scale s on every layer: hardened's outputs (~ s^2 per layer) fall into the
    RMSNorm's epsilon and the circuit fails at s = 0.005; with fit16 it stays correct
    (ultra_s, whose normalized scale varies more with the number of 1s, at s = 0.01)"""
    xs = _inputs(16, 40, 2)
    ref = _ref(_fn, xs)
    hard = Compiler("hardened", select=False).run(_fn, x=Bits("0" * 16).bitlist)
    assert not correct(_scaled(hard, xs, s=0.005), ref)
    for kw, s in ((FIT16, 0.005), (ULTRA_S, 0.01)):
        mlp = Compiler(**kw).run(_fn, x=Bits("0" * 16).bitlist)
        assert correct(_scaled(mlp, xs, s=s), ref)
        assert correct(_scaled(mlp, xs, dtype=t.bfloat16, s=s), ref)


def test_fit16_wide_float16():
    """an AND of 2048 inputs: hardened's first layer overflows float16; fit16 scales it
    into range"""
    fn = lambda x: [and_(x), or_(x)]  # noqa: E731
    n = 2048
    with pytest.warns(UserWarning, match="float16"):
        hard = Compiler("hardened", select=False).run(fn, x=Bits("0" * n).bitlist)
    assert fp16_bound(hard) > FP16_MAX
    mlp = Compiler(**ULTRA_S, mlp_dtype=t.float16).run(fn, x=Bits("0" * n).bitlist)
    assert fp16_bound(mlp) <= FP16_MAX
    rng = random.Random(4)
    xs = [[0] * n, [1] * n, [1] * (n - 1) + [0], [0] * (n - 1) + [1]]
    xs += [[int(rng.random() < 0.999) for _ in range(n)] for _ in range(4)]
    ref = t.tensor([[int(all(x)), int(any(x))] for x in xs], dtype=t.float32)
    assert correct(_scaled(mlp, xs, t.float16), ref)


def test_ultra_s_rows():
    """ultra_s: BOS = 1 in every layer, steps centered, and every row's sum spread evenly
    over the 2 always-0 features"""
    fn = lambda x: [xor(x[:2]), and_(x[2:4])]  # noqa: E731
    mlp = Compiler(**ULTRA_S).run(fn, x=Bits("0" * 4).bitlist)
    for i, L in enumerate(mlp.layers):
        assert L.bos_in == 1.0
        if i > 0:
            assert L.zero_in == 2
            for w in (L.wg.weight, L.wv.weight):
                z = w[:, -2:]
                assert t.equal(z, z[:, :1].expand_as(z))
    # the prologue copies x as steps on 1.5 x - 1/2: centered, their two ReLUs have BOS
    # weights (-1/2 - 3/8) c q and (-1/2 - 5/8) c q (BOS's own pair: c q and 2 c q)
    L = mlp.layers[0]
    cq = L.wg.weight[0, 0]
    steps = {round(v, 4) for v in (L.wg.weight[1:, 0] / cq).tolist()}
    assert steps == {-0.875, -1.125, 2.0}


def test_exact_readout_with_gated_units():
    """exact_readout flips step rows only; rows of gated units keep their units"""
    def fn(x: list[Bit]) -> list[Bit]:
        return [glu_xor(x[:5]), and_(x[2:7]), xor(x[1:4])]

    rng = random.Random(4)
    xs = [[rng.randint(0, 1) for _ in range(8)] for _ in range(40)] + [[0] * 8, [1] * 8]
    ref = t.tensor([values(fn(const(x))) for x in xs], dtype=t.float32)
    tree = TreeCompiler().run(fn, x=Bits("0" * 8).bitlist)
    for dtype in (t.float32, t.bfloat16):
        mlp = build(tree, MLPOptions(exact=True, exact_readout=True), dtype)
        assert (ratios(mlp, xs, dtype) - ref).abs().max() < 0.02


@pytest.mark.parametrize("case", ["product", "copy_layer"])
def test_fp16_bound_covers_every_layer(case):
    """fp16_bound covers float16 overflows past the first layer's pre-activations, also
    in the core's builds: the BOS pair's hidden product 8 c (n_in + 1) from 2047 inputs;
    a wide AND behind the wide-row copy layer"""
    n, fn = (2100, lambda x: [or_(x)]) if case == "product" else (1100, lambda x: [and_(x)])
    mlp = CoreCompiler(mlp_dtype=t.float16).run(fn, x=Bits("0" * n).bitlist)
    assert len(mlp.layers) == (2 if case == "copy_layer" else 1)
    assert fp16_bound(mlp) > FP16_MAX
    r = ratios(mlp, [[0] * n, [1] * n], t.float16)
    assert not ((r - t.tensor([[0.0], [1.0]])).abs() < 0.5).all()  # float16 really fails

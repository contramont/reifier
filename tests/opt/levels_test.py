"""The optimization levels of reifier.opt (ultra, hardened, robust, O2, O3): exhaustive
checks on small circuits, the selection of the smallest candidate recipe, recipes and
knobs, the weights pinned, float16 range, and the core's compile left as it is."""

import itertools
import random
import warnings

import pytest
import torch as t

from reifier.neurons.core import Bit, const, gate
from reifier.neurons.operations import add, and_, inhib, not_, or_, parity, xor
from reifier.opt import RECIPES, TIERS, Compiler, candidates, fp16_bound, recipe
from reifier.opt.build import FP16_MAX, dense_bound
from reifier.opt.recipes import MAX_DEPTH
from reifier.tensors.compilation import Compiler as CoreCompiler
from reifier.utils.format import Bits
from tests.opt.utils import correct, sd_hash, values
from tests.testing_utils import ratios


def mixed(x: list[Bit]) -> list[Bit]:
    a, b, c = x[:3]
    chi = xor([a, inhib([b, c])])
    maj = xor([and_([a, b]), and_([a, c]), and_([b, c])])
    return [chi, maj, xor(x), parity(x), and_(x), or_(x), not_(and_(x[3:]))]


def weighted(x: list[Bit]) -> list[Bit]:
    """generic threshold gates: sums up to 6 above threshold. In 16-bit floats their
    steps round like any wide sum (about 4% at sum 6), so only float32 and the T3+
    levels (flipped at the outputs) are exact on them"""
    return [gate(x, [2, -1, 1, 3, -2, 1], 3), gate(x, [1, 1, 1, 1, 1, 1], 2)]


CIRCUITS = {
    "adder4": (lambda x: add(x[:4], x[4:]), 8),
    "mixed7": (mixed, 7),
    "wide_and_or9": (lambda x: [and_(x), or_(x), xor(x), not_(or_(x[:5]))], 9),
    "weighted6": (weighted, 6),
}
DTYPES = {  # the dtypes each level is built for (O3: float32 only)
    "ultra": [t.float32, t.bfloat16, t.float16],
    "hardened": [t.float32, t.bfloat16, t.float16],
    "robust": [t.float32, t.bfloat16, t.float16],
    "O2": [t.float32, t.bfloat16],
    "O3": [t.float32],
}


def _all_inputs(n: int) -> list[list[int]]:
    return [list(v) for v in itertools.product([0, 1], repeat=n)]


def _size(mlp) -> tuple[int, int]:
    ps = list(mlp.parameters())
    return sum(p.numel() for p in ps), sum(int((p != 0).sum()) for p in ps)


@pytest.mark.parametrize("circuit", list(CIRCUITS))
@pytest.mark.parametrize("level,dtype", [(lv, dt) for lv in TIERS for dt in DTYPES[lv]])
def test_ladder_exhaustive(circuit, level, dtype):
    """every level computes every small circuit correctly on all inputs: every 1 within
    0.02 of BOS (boolify's window), every 0 nearer 0 than 1"""
    fn, n = CIRCUITS[circuit]
    if circuit == "weighted6" and dtype != t.float32 and level not in ("robust", "hardened",
                                                                          "ultra"):
        pytest.skip("wide generic threshold gates are exact in float32 only")
    xs = _all_inputs(n)
    mlp = Compiler(level, mlp_dtype=dtype).run(fn, x=Bits("0" * n).bitlist)
    ref = t.tensor([values(fn(const(x))) for x in xs], dtype=t.float32)
    r = ratios(mlp, xs, dtype)
    assert correct(r, ref), (r - ref).abs().max().item()


def test_tiers_and_candidates():
    """five levels, most robust first; a level selects from every recipe at its tier or
    above but hardened, most robust first and then in RECIPES' order (the rank that
    breaks ties), so the candidate sets nest; the flat-unit recipes' depth limit"""
    assert list(TIERS.items()) == [("ultra", 5), ("hardened", 4), ("robust", 3), ("O2", 2),
                                   ("O3", 1)]
    for lv in TIERS:
        names = candidates(lv)
        assert (lv in names) == (lv != "hardened") and RECIPES[lv][0] == TIERS[lv]
        assert all(RECIPES[n][0] >= TIERS[lv] for n in names)
        assert [RECIPES[n][0] for n in names] == sorted((RECIPES[n][0] for n in names),
                                                        reverse=True)
    assert set(candidates("O3")) == set(RECIPES) - {"hardened"}
    assert candidates("O3") == ["ultra", "ultra_s", "ultra_bc64", "hardened_xf_c", "robust",
                                "robust_x1_c", "robust_fcx1_cc", "robust_fcf_c", "O2",
                                "robust_fcf", "O3"]
    for a, b in itertools.pairwise(TIERS):
        assert set(candidates(a)) <= set(candidates(b))
    assert MAX_DEPTH == {"ultra": 200, "ultra_bc64": 200}


@pytest.mark.parametrize("circuit", list(CIRCUITS) + ["parity16"])
def test_select_keeps_smallest(circuit):
    """each level is the smallest of its candidates (dense, then sparse), bit for bit
    the chosen recipe's compile, and so no larger than any more robust level; ultra_s is
    smaller than hardened, which is therefore no candidate"""
    fn, n = CIRCUITS[circuit] if circuit in CIRCUITS else (lambda x: [parity(x)], 16)
    x = Bits("0" * n).bitlist
    own = {name: Compiler(name, select=False).run(fn, x=x) for name in RECIPES}
    assert _size(own["ultra_s"])[0] < _size(own["hardened"])[0]
    sizes = []
    for lv in TIERS:
        comp = Compiler(lv)
        assert comp.select and comp.chosen is None
        mlp = comp.run(fn, x=x)
        assert comp.chosen in candidates(lv)
        assert _size(mlp) == min(_size(own[c]) for c in candidates(lv))
        for p, q in zip(mlp.parameters(), own[comp.chosen].parameters(), strict=True):
            assert t.equal(p, q)
        sizes.append(_size(mlp))
    assert sizes == sorted(sizes, reverse=True)  # ultra >= hardened >= robust >= O2 >= O3


def test_dense_bound_below_size():
    """dense_bound (the selection's pruning bound) never exceeds the MLP's dense size"""
    for fn, n in list(CIRCUITS.values()) + [(lambda x: [parity(x)], 16)]:
        x = Bits("0" * n).bitlist
        for name in RECIPES:
            comp = Compiler(name, select=False)
            graph = comp.get_graph(fn, x=x)
            assert dense_bound(graph) <= _size(comp.get_mlp_from_graph(graph))[0], name


def test_recipe_and_knobs():
    """recipe() lays knobs and pass options over a recipe's (a copy); a Compiler lays
    its knobs over every recipe it compiles, a level's candidates too; unknown levels and
    knobs and invalid values raise when the Compiler is made"""
    k = recipe("ultra", q=1024, passes={"lead_clean": False})
    assert k["q"] == 1024 and k["ln_invariant"] == 8
    assert k["passes"]["lead_clean"] is False and k["passes"]["fold_bias"] == 2
    assert RECIPES["ultra"][1]["q"] == 128 and RECIPES["ultra"][1]["passes"]["lead_clean"]
    recipe("robust")["passes"]["cheap"] = False  # a copy: never the recipe's pass dict
    assert RECIPES["robust"][1]["passes"]["cheap"] is True
    fn, n = CIRCUITS["adder4"]
    comp = Compiler("O3", knobs={"q": 16})
    comp.run(fn, x=Bits("0" * n).bitlist)
    assert comp.chosen in candidates("O3")
    # a knob turned off compiles as the core's for that knob
    fn = lambda x: [and_(x)]  # noqa: E731
    x = Bits("0" * 12).bitlist
    a = Compiler("robust", select=False, knobs={"fanin_and_or": 0}).run(fn, x=x)
    assert len(a.layers) < len(Compiler("robust").run(fn, x=x).layers)
    for level, knobs in [("fastest", {}), ("robust", {"fanin_adder": "ripple"}),
                         ("ultra", {"out_scale": -1024}), ("O3", {"q": 0}),
                         ("ultra", {"out_scale": float("nan")})]:
        with pytest.raises(ValueError):
            Compiler(level, knobs=knobs)
    for knobs in [{"ballast": 1 / 64}, {"passes": False}]:
        with pytest.raises(TypeError):
            Compiler("robust", knobs=knobs)


PINNED_LEVELS = {  # the recipe each level keeps and its weights in float32, bf16, fp16
    "ultra": ("ultra", "1346bf3df7bcba7e", "dcc6df73d8f8cfd4", "b90b8d548c818ce8"),
    "hardened": ("ultra_bc64", "8c09e89c6c5f59f6", "7ae4b024a0118ea3", "c8a080862de63e14"),
    "robust": ("robust_fcf_c", "09e619d42602b00d", "571fd70b611c7915", "bc661cca77a67313"),
    "O2": ("robust_fcf", "75ad607429bab78a", "f2c8e5f443902288", "d546286ba18646cb"),
    "O3": ("O3", "765c3ba63e26cb4d", "3276f86fa77fcd72", "fd4d336e927249de"),
}
PINNED_RECIPES = {  # the weights of every recipe alone, in float32
    "ultra": "1346bf3df7bcba7e", "hardened": "b3c8c97234282fe9", "robust": "0b3e0741828b90fa",
    "O2": "695b723ae86e5b3e", "O3": "765c3ba63e26cb4d", "ultra_s": "7b3c51a5fc0483a5",
    "ultra_bc64": "8c09e89c6c5f59f6", "hardened_xf_c": "c99f2f36439b2968",
    "robust_x1_c": "c2f5c8e8cc96a57f", "robust_fcx1_cc": "b4b774d4d3dd40f2",
    "robust_fcf_c": "09e619d42602b00d", "robust_fcf": "75ad607429bab78a",
}


def _pinned(x: list[Bit]) -> list[Bit]:
    return add(x[:24], x[24:48]) + [xor(x[:9]), parity(x[10:19]), and_(x[:12]),
                                   or_(x[20:40]), gate(x[:6], [2, -1, 1, 3, -2, 1], 3),
                                   xor(x[30:47])]


def test_weights_pinned():
    """every recipe and every level compiles to the weights the tiers were measured on
    (sd_hash prefixes, CPU), on a circuit where the five levels keep five recipes"""
    x = Bits("0" * 48).bitlist
    for name, h in PINNED_RECIPES.items():
        assert sd_hash(Compiler(name, select=False).run(_pinned, x=x)).startswith(h), name
    for lv, (chosen, *hs) in PINNED_LEVELS.items():
        for dtype, h in zip((t.float32, t.bfloat16, t.float16), hs, strict=True):
            comp = Compiler(lv, mlp_dtype=dtype)
            assert sd_hash(comp.run(_pinned, x=x)).startswith(h), (lv, dtype)
            assert comp.chosen == chosen


def test_core_compile_unchanged():
    """a level compile (fan-in traced under sys.monitoring) leaves the core's Compiler
    as it is: the tree compiler and its weights (state_dict hashes of the compile at
    fdc78d6; bf16 rows of <= 64 inputs, see swiglu_test)"""
    Compiler("robust").run(lambda x: [xor(x), parity(x), and_(x)] + add(x[:8], x[8:]),
                           x=Bits("0" * 16).bitlist)
    fn = lambda x: add(x[:32], x[32:])  # noqa: E731
    mlp = CoreCompiler().run(fn, x=Bits("0" * 64).bitlist)
    assert len(mlp.layers) == 6 and _size(mlp) == (1294715, 20941)
    assert sd_hash(mlp).startswith("bb6d15e791919374")
    bf16 = CoreCompiler(mlp_dtype=t.bfloat16).run(fn, x=Bits("0" * 64).bitlist)
    assert sd_hash(bf16).startswith("03fce0cc3251aea7")


@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("level,knobs", [("robust", {}), ("robust", {"fanin_and_or": 0}),
                                         ("hardened", {})])
def test_wide_and_output(k, level, knobs):
    """wide output ANDs (exact_readout keeps their orientation), with the level's
    fan-in bound and without (the robust recipe alone): AND of 64 / 128 inputs"""
    comp = Compiler(level, select=not knobs, knobs=knobs, mlp_dtype=t.bfloat16)
    mlp = comp.run(lambda x: [and_(x)], x=Bits("0" * k).bitlist)
    xs = [[1] * k, [0] * k] + [[1] * j + [0] + [1] * (k - j - 1) for j in range(k)]
    ref = t.tensor([[int(all(x))] for x in xs], dtype=t.float32)
    r = ratios(mlp, xs, t.bfloat16)
    assert (r - ref).abs().max().item() < 0.5  # nearest (0s may carry step errors)
    assert ((r - 1).abs() <= 0.02).float().eq(ref).all()  # boolify


def _adder16(x: list[Bit]) -> list[Bit]:
    return add(x[:16], x[16:]) + [xor(x[:11]), and_(x[5:20]), or_(x[3:30])]


@pytest.mark.parametrize("level,dtype", [(lv, dt) for lv in TIERS for dt in DTYPES[lv]])
def test_levels_adder16(level, dtype):
    """trees of bounded fan-in and the prefix adder at moderate widths"""
    rng = random.Random(2)
    mlp = Compiler(level, mlp_dtype=dtype).run(_adder16, x=Bits("0" * 32).bitlist)
    xs = [[0] * 32, [1] * 32] + [[rng.randint(0, 1) for _ in range(32)] for _ in range(30)]
    ref = t.tensor([values(_adder16(const(x))) for x in xs], dtype=t.float32)
    assert correct(ratios(mlp, xs, dtype), ref)


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_clean_outputs(dtype):
    """outputs made by gated units get a last level of step copies (exact readout then
    makes their 1s exactly BOS): robust_fcf_c is robust_fcf with clean_outputs"""
    assert recipe("robust_fcf_c") == recipe("robust_fcf", {"clean_outputs": True})
    fn, n = CIRCUITS["mixed7"]
    x = Bits("0" * n).bitlist
    plain = Compiler("robust_fcf", select=False, mlp_dtype=dtype)
    clean = Compiler("robust_fcf_c", select=False, mlp_dtype=dtype)
    g0, g1 = plain.get_graph(fn, x=x), clean.get_graph(fn, x=x)
    assert any(o.units for o in g0.levels[-1].origins)  # unit outputs without it
    assert len(g1.levels) == len(g0.levels) + 1
    assert not any(o.units for o in g1.levels[-1].origins)  # step copies with it
    xs = _all_inputs(n)
    ref = t.tensor([values(fn(const(v))) for v in xs], dtype=t.float32)
    r = ratios(clean.run(fn, x=x), xs, dtype)
    assert correct(r, ref)
    assert (r[ref.bool()] == 1).all()  # every 1 exactly BOS


def test_depth_envelope():
    """recipes with flat units (ultra, ultra_bc64) are candidates only up to MAX_DEPTH
    layers; deeper circuits get the steps-only T5 recipe (ultra_s)"""
    def chain(x, n=MAX_DEPTH["ultra"]):  # n gates: n + 1 layers with ultra's lead_clean
        y = x[0]
        for i in range(n):
            y = and_([y, x[1 + i % 7]]) if i % 2 else or_([y, x[1 + i % 7]])
        return [y]

    x = Bits("0" * 8).bitlist
    comp = Compiler("ultra")
    deep = comp.run(chain, x=x)
    assert comp.chosen == "ultra_s" and len(deep.layers) > MAX_DEPTH["ultra"]
    # select=False keeps the level's own recipe at any depth; it is smaller here, so only
    # the envelope kept the level from choosing it, which it does one layer shallower
    own = Compiler("ultra", select=False).run(chain, x=x)
    assert len(own.layers) > MAX_DEPTH["ultra"]
    assert _size(own) < _size(deep)
    edge = Compiler("ultra")
    edge.run(lambda x: chain(x, MAX_DEPTH["ultra"] - 1), x=x)
    assert edge.chosen in ("ultra", "ultra_bc64")
    shallow = Compiler("hardened")
    shallow.run(lambda x: add(x[:16], x[16:]), x=Bits("0" * 32).bitlist)
    assert shallow.chosen in ("ultra", "ultra_bc64")  # within the envelope: flat units
    xs = _all_inputs(8)[::5]
    ref = t.tensor([values(chain(const(v))) for v in xs], dtype=t.float32)
    assert correct(ratios(deep, xs, t.float32), ref)


def test_fp16_bound_and_warning():
    """the first layer's values grow with n_in: ultra's BOS pair product
    8 c (n_in + 1) leaves float16's range at about 2047 inputs (q = 128); the T5 level
    then prefers a candidate that fits (ultra_s, fit16), float16 builds of any tier
    warn, and bf16 builds never warn"""
    fn = lambda x: [xor(x[:4]), and_(x[4:8])]  # noqa: E731
    small = Compiler("ultra").run(fn, x=Bits("0" * 1100).bitlist)
    assert fp16_bound(small) < FP16_MAX
    with pytest.warns(UserWarning, match="float16") as caught:
        big = Compiler("ultra", select=False).run(fn, x=Bits("0" * 2100).bitlist)
    assert fp16_bound(big) > FP16_MAX
    # the warning points at run()'s caller
    assert [w.filename for w in caught if "float16" in str(w.message)] == [__file__]
    comp = Compiler("ultra")
    assert fp16_bound(comp.run(fn, x=Bits("0" * 2100).bitlist)) < FP16_MAX
    assert comp.chosen == "ultra_s"
    o3 = Compiler("O3", select=False, mlp_dtype=t.float16)  # T1, but a float16 build
    with pytest.warns(UserWarning, match="float16"):
        o3.run(fn, x=Bits("0" * 2100).bitlist)
    wide = lambda x: [xor(x[:8]), and_(x[8:20]), xor(x[20:24])]  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlp = Compiler("robust").run(wide, x=Bits("0" * 1000).bitlist)
        Compiler("robust", mlp_dtype=t.bfloat16).run(wide, x=Bits("0" * 1100).bitlist)
    assert fp16_bound(mlp) < FP16_MAX

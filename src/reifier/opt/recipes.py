"""The recipes of the optimization levels: name -> (tier, knobs). Knobs: fanin_xor,
fanin_and_or and fanin_adder (fanin.Fanin), passes (graph.GraphOptions fields) and the
rest (build.MLPOptions fields); a knob a recipe leaves out takes its stage's default. A
recipe is a candidate of every level at or below its tier (candidates)."""

from typing import Any

# Graph passes (GraphOptions fields): steps only, with NOT folds that keep biases <= 2;
# + flat units for AND/OR/NOT/copy gates of <= 4 inputs; + step outputs, input copies
HARDENED_PASSES: dict[str, Any] = dict(fold_bias=2)
ROBUST_PASSES: dict[str, Any] = dict(HARDENED_PASSES, cheap=True, cheap_max=4)
ULTRA_PASSES: dict[str, Any] = dict(ROBUST_PASSES, cheap_out=False, lead_clean=True)
# O2 / O3: every NOT folded, flat units of any width, and E-exact xor trees and cones /
# one layer of units per symmetric sum and cones exact in real arithmetic only
O2_PASSES: dict[str, Any] = dict(cheap=True, sumfuse="e", cone=3, xortree=True, reclean=8)
O3_PASSES: dict[str, Any] = dict(cheap=True, sumfuse="all", cone=3, cone_exact=False,
                                 reclean=12)

# Recipes: name -> (tier, knobs); see the package docstring for the tiers
RECIPES: dict[str, tuple[int, dict[str, Any]]] = {
    "ultra": (5, dict(
        exact=True, q=128, bos_copies=32, center=True, out_scale=1024, zero_spread=True,
        fanin_xor=4, fanin_and_or=2, fanin_adder="prefix", ln_invariant=8,
        exact_readout=True, passes=ULTRA_PASSES,
    )),
    "hardened": (4, dict(
        exact=True, q=256, heavy_bos=True, fanin_xor=4, fanin_and_or=2,
        fanin_adder="prefix", ln_invariant=8, prologue=True, exact_readout=True,
        passes=HARDENED_PASSES,
    )),
    "robust": (3, dict(
        exact=True, q=64, heavy_bos=True, fanin_xor=4, fanin_and_or=2,
        fanin_adder="prefix", exact_readout=True, passes=ROBUST_PASSES,
    )),
    # the core's add: the prefix adder's merge gates round in bf16 below non-flat units
    "O2": (2, dict(passes=O2_PASSES, exact=True, exact_readout=True, fanin_xor=4)),
    # xors of more than 32 inputs as trees: one glu_xor of 128 inputs errs by 0.026
    "O3": (1, dict(passes=O3_PASSES, fanin_adder="prefix", fanin_xor=32)),
}
FLAT_XOR = dict(sumfuse="e", xortree=True, flat_xor=True)  # trees of flat xor units
EXACT_XOR = dict(sumfuse="e", xortree=True, reclean=1)  # E-exact xor units, re-cleaned
FLAT_CONES = dict(cone=3, cone_flat=True, cone_units=4)  # sums of flat subcube indicators
CLEAN = dict(clean_outputs=True)  # step copies of the outputs that are units


def recipe(name: str, passes: dict[str, Any] | None = None, **knobs: Any) -> dict[str, Any]:
    """the knobs of RECIPES[name] (a copy), with knobs and pass options on top"""
    out = {**RECIPES[name][1], **knobs}
    out["passes"] = {**out["passes"], **({} if passes is None else passes)}
    return out


RECIPES.update({
    # hardened's steps with BOS = 1, every layer rescaled to the float16 range (fit16):
    # no depth limit, and values within float16 range at any width (fp16_bound)
    "ultra_s": (5, recipe("hardened", fit16=True, heavy_bos=False, center=True,
                          zero_spread=True, ln_invariant=2)),
    "ultra_bc64": (4, recipe("ultra", bos_copies=64)),
    "hardened_xf_c": (4, recipe("hardened", {**FLAT_XOR, **CLEAN})),
    "robust_x1_c": (3, recipe("robust", {**EXACT_XOR, **CLEAN})),
    "robust_fcx1_cc": (3, recipe("robust", {**FLAT_CONES, **EXACT_XOR, **CLEAN,
                                            "cone_reclean": True})),
    "robust_fcf_c": (3, recipe("robust", {**FLAT_CONES, **CLEAN})),
    # without the step copies, unit outputs can miss boolify under split reductions
    "robust_fcf": (2, recipe("robust", FLAT_CONES)),
})

# ultra and ultra_bc64 are candidates only up to this many layers: their flat units pass
# input errors on (a step restores a 1), which add up under T4 / T5 perturbations
MAX_DEPTH: dict[str, int] = {"ultra": 200, "ultra_bc64": 200}

# the levels, most robust first; each is also the name of its own recipe
TIERS: dict[str, int] = {
    lv: RECIPES[lv][0] for lv in ("ultra", "hardened", "robust", "O2", "O3")
}


def candidates(level: str) -> list[str]:
    """the recipes a level selects from, most robust first: those at its tier or above,
    but hardened (see the package docstring); Compiler skips those beyond MAX_DEPTH"""
    names = [n for n, (tier, _) in RECIPES.items()
             if tier >= RECIPES[level][0] and n != "hardened"]
    return sorted(names, key=lambda n: -RECIPES[n][0])

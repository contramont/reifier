"""Optimization levels: compile a function of Bits to an MLP_SwiGLU that trades size for
robustness. An optional package beside the core's compiler (reifier.tensors.compilation):
it builds on the core, and the core never imports it.

    import torch as t
    from reifier.opt import Compiler

    comp = Compiler("robust", mlp_dtype=t.bfloat16)
    mlp = comp.run(fn, x=bits)  # comp.chosen names the recipe kept

Levels, most robust first: "ultra" (tier T5), "hardened" (T4), "robust" (T3), "O2" (T2),
"O3" (T1). At its tier, every output 1 is within 0.02 of BOS and every 0 nearer 0 than 1:
  T1 in float32, with every output within 0.01;
  T2 in bfloat16 with float32 accumulation;
  T3 in bfloat16 or float16 on CUDA with reduced-precision reductions, at any batch;
  T4 as T3, under a host's norm scale 0.1, LayerNorm mean shift 0.1 and interference 0.01
    (relative to the rms) together;
  T5 as T4 with norm scale 0.05, plus a LayerNorm shift of 0.1 and noise of 0.01 relative
    to the largest feature.
Tiers are measured on benchmark circuits of up to a few hundred layers, not proven.

A recipe (RECIPES: name -> (tier, knobs)) chooses bounded fan-in (fanin), graph passes
(graph) and the SwiGLU construction (build). A level compiles the recipes of
candidates(level), those at its tier or above, and keeps the smallest (dense parameters,
then nonzeros; Compiler.chosen names it), so it is never larger than a more robust level.
hardened is no candidate: ultra_s builds the same gate graph with 2 instead of 8 always-0
features per layer, so it is always smaller, and fit16 keeps its values within float16
range. ultra and ultra_bc64 are candidates only up to MAX_DEPTH layers. Where the tier
includes float16, candidates within its range (fp16_bound) come first. Compiler(name,
select=False) compiles one recipe (a level is also the name of its own recipe); knobs are
laid over the knobs of every recipe compiled (recipe(name, **knobs)).

float16 range: the first layer's values grow with its inputs, and every recipe but ultra_s
leaves float16's range by 2047 inputs (all but ultra and ultra_bc64 by about 1000); run()
then warns for float16 builds and float32 builds of T3+ levels and recipes. fp16_bound
bounds values, not weights: ultra_s's weights leave float16's range where generic gate()
rows or inhib have thresholds above 64, so cast such float32 builds to bfloat16. At every
level, generic gate() rows whose sums can exceed their threshold by more than ~2 round in
16-bit floats (the fan-in bounds cover xor, and_, or_, parity and add), and so do
outputs of gated units written in fn (glu, glu_xor) in bfloat16; robust can flip bits of
adds wider than 256 bits under split reductions.
"""

from .build import fp16_bound
from .compiler import Compiler
from .recipes import RECIPES, TIERS, candidates, recipe

__all__ = [
    "Compiler",
    "RECIPES",
    "TIERS",
    "candidates",
    "recipe",
    "fp16_bound",
]

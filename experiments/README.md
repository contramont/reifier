# Experiments

Measurements behind reifier's compiler, kept off main. Each directory has its own README.

- `levels/`: the optimization levels of `reifier.opt`: tiers, recipes, the threat model and
  per-circuit results.
- `xof_shrink/`: hand-built SwiGLU circuits for 1-round Keccak XOF (waves 1-4).

## Library versions

- `levels/` was measured in waves 6-8 with the levels inside the core compiler
  (`reifier.tensors.compilation.Compiler(level=...)`, never published). `reifier.opt`
  compiles the same weights for every level, recipe and dtype (state_dict hashes, chosen
  recipes, sizes, depths and warnings on the 10 suite circuits and 89 random circuits), so
  the drivers run on this branch. Configurations with knobs that the wave-8 trim removed need
  wave 7's library, which is not published, and `robust_tree` and `hardened_tree` need the
  first version (see levels/README.md, "Code").
- `xof_shrink/` needs main's library from fdc78d6 on and nothing else: `validate_bench.py`
  is bit-equal at log_w 0-2, and `bf16x/e2e_compiler.py` gives the same results, on fdc78d6
  and on this branch. `architecture/` needs `architecture/residual_compiler.diff`, which was
  written against an unpublished library of 2026-09-29 and applies to no commit of main;
  `bf16x/swiglu_exact_steps.diff` is on main as fdc78d6. The search scripts in `wave3/`,
  `search_fl/` and `wave4_bf16w/` need scipy (some sympy). Scripts with absolute paths
  record how the runs were made.

Run from the repository root with the experiment on the path, e.g.
`PYTHONPATH=src:experiments/levels python experiments/levels/ladder_eval.py ...`.

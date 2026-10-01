## Synthesis: shrinking the compiled SwiGLU circuit for 1-round Keccak XOF

Combining the avenues gains only 1–4% dense over the best single avenue at each depth, so the combination brings no new 2x. The reason is that the xof-structure builder had already absorbed most other avenues' ideas. In plain untied `MLP_SwiGLU` the frontier is close to its floor. The one path to a further 2x or more is an architecture change: tying weights across XOF steps, with a residual stream.

### Measured combination (`av/combined-1`, log_w=6, 3 steps)

Every row below:
- passes the harness (`ok: true`);
- passes brainstorm-critic's 63-message edge-case check against the unpatched reference xof, with no eager mismatch;
- is `ok` at log_w 3–5 with 64 messages.

`validate_bench` gives bit-equal weights at log_w 0–2 for the d4, d5, d6 (both), d7 `_m3_mp` and d8 `_m3_mp` variants. The repo test suite passes (55 tests, `hash_long_test` not run).

| depth | variant | dense | sparse | margin (harness / edge) | vs glu_chi_iota (depth/dense/sparse) | vs previous best at that depth (dense / sparse) |
|---|---|---|---|---|---|---|
| 4 | `xc:d4a_m4k3_mp` | 160,358,065 | 425,212 | 2.0e-4 / 1.6e-3 | 2.00 / 1.54 / 1.27 | −1.2% / +1.2% |
| 5 | `xc:d5_m3k2_mp` | 89,244,445 | 229,869 | 1.3e-4 / 8.6e-4 | 1.60 / 2.77 / 2.36 | −2.1% / +2.2% |
| 6 | `xs3:lazy4c_middle_m3` | 71,257,018 | 172,637 | 3.6e-3 / 5.1e-3 | 1.33 / 3.47 / 3.14 | −1.0% / +0.8% |
| 6 | `xs3:lazy4c_middle_m3_mp` | 69,368,253 | 177,623 | 3.6e-3 / 5.1e-3 | 1.33 / 3.56 / 3.05 | −3.6% / +3.7% |
| 7 | `xs3:split_first_lazy4c_m3_mp` | 63,651,004 | 127,470 | 3.6e-3 / 5.1e-3 | 1.14 / 3.88 / 4.25 | −2.5% / +2.2% |
| 8 | `xs3:split_first_middle_m3_mp` | 60,706,707 | 111,913 | 1.6e-4 / 2.2e-3 | 1.00 / 4.07 / 4.84 | −2.1% / +3.8% |
| 8 | `xs3:split_first_middle_mp` | 61,082,590 | 109,101 | 6.3e-5 / 7.7e-4 | 1.00 / 4.04 / 4.97 | −1.5% / +1.2% |

Against the baseline, depth 6 is 3.33x / 46–48x / 11.5–11.8x and depth 8 is 2.5x / 54x / 18–18.7x. `_mp` (min-parity) trades about −3% dense for about +3% sparse, so it is not a clear win.

**How the numbers scale with more XOF steps** (untied, exact split middle rounds with lazy4c before the last round): 2 steps are 4 / 31.5M / 108K, 3 steps 6 / 71.3M / 173K, 4 steps 9 / 111.5M / 222K, 5 steps 12 / 155.0M / 273K. Each extra step adds 3 layers, about 42M dense and 50K sparse.

### (1) Combination plan

**Circuit-level stack (done in combined-1, applied in this order):**
1. base.
2. xof-structure's compiler patch: numeric glu nodes plus hidden-unit sharing in `Matrices.layer_to_units`.
3. brainstorm-critic's glu rounding tolerance, `MIN_PARITY` and `glu_xor_min`.
4. unit-synthesis's `xs3` (lazy4c chi, min-parity).
5. My additions:
   - binary digest packing (3 bits per feature, minimal decoders from `decode.py`);
   - min-parity in the first split layer;
   - `xc.py`, which switches on min-parity in xof-structure's `xs4` first layer (depth 4/5).

Min-parity is used only where inputs are raw message bits. On count features it fails the edge check (0.0265).

Predicted versus measured:
- lazy4c + 3-bit packing: predicted about 70.3M, measured 71.26M. The 3-bit decoders widen the last layer from 675 to 1045 units, which ate part of the saving.
- Adding min-parity: predicted about 68.4M, measured 69.37M.

What does not compose:
- **cheap-gates on builder circuits:** `merge_units` would drop live units that read count features unless it is given value bounds. The only gain there is one BOS unit per layer, which is negligible.
- **dce-const and leveling passes on builder circuits:** they do nothing, because the builder already emits leveled, pruned, copy-free layers.
- **AND_CHI2 in the two-layer (X) designs:** it only works for adjacent chi bits of the same row, and the X layer's column-pair sums never contain such a pair.
- **With tying:** lazy chi, count features on the ±1 residual stream, and last-round dead-code removal all stop working.

**General-compiler stack (plan, not measured):** dce-const `simplify_blocks` → leveling tree passes (`collapse_copy_levels`, `simplify_levels`, `_unit_is_live`, `pack_copy_chains`) → cheap-gates one-unit gates at the SwiGLU level, with bounds → linear-fold lin/fold. Predicted for glu_chi_iota: about 6 / 95–97M / 297K; adding linear-fold's Keccak count features gives about 6 / 90–92M / 178K. That is still about 25% behind the builder, because the X layer and lazy chi are Keccak-specific.

**Architecture (reported separately, unchanged):** `gated_split_taps` 9 / 30.3M / 56.7K (109x dense vs baseline) and `tied_compact_taps` 6 / 63.7M / 171K.

**Where the cost sits now:**
- Depth 6: the first theta layer on raw bits is 20.9M dense and 83K sparse (30% and 47%). The second-round X layer is 17.7M (25%). The rest of depth 6 is at its structural minimum.
- Depth 8: every layer except the X layer is at about 1 unit per output bit, which is roughly the per-layer floor of 7.7M. The floor for depth 8 is about 50M, and we are at 60.7M.
- So there is at most about 1.2–1.4x left in untied `MLP_SwiGLU` dense. 100x vs baseline (about 33M) is out of reach there.

### (2) Conflicts and overlaps

- **`compile/simplify.py`:** dce-const and cheap-gates each create this new file, with different contents. Keep dce-const's; rename cheap-gates' to `level_simplify.py` or drop it, since it overlaps with leveling.
- **Dropping the output layer in `tree.py`:** four implementations — dce-const `redundant_outputs_layer_order`, leveling `collapse_copy_levels`, theta-structure `outputs_permutation`/`PERMUTE_OUTPUTS`, and cheap-gates `fold_outputs`. Keep leveling's, which is a superset: any copy, NOT or constant level, repeats included.
- **Level dedupe and dead-node removal:** dce-const `_merge_duplicate_nodes`/`_remove_dead_nodes`, leveling `simplify_levels`, cheap-gates `merge_equal`/`drop_unread`, and linear-fold `fold.prune`. Keep leveling's at the tree level.
- **Block-level pruning:** dce-const `simplify_blocks` is a superset of leveling `prune_blocks` (it adds alias forwarding and computing negations beside their base). Port leveling's constant-output support, and add `"lin"` to `CREATORS`.
- **Copies:** leveling `COPY_UNIT` (relu), theta-structure `GLU_COPIES` (linear), linear-fold `unit_copies`/`_copy_origin`, and cheap-gates' flat one-unit form. Leveling and linear-fold's `_copy_origin` conflict directly. Resolution: bit copies stay step rows and cheap-gates converts them to its flat form; copies of numeric or linear nodes use `max(0,1)·x`.
- **`glu()` in `core.py`:** `numeric=True` (xof-structure) and `boolean=False` (theta-structure) are the same concept; unify them as `numeric`. Also keep `inplace=i` (architecture), the 1e-6 tolerance, and linear-fold's `lin()`.
- **`Matrices.layer_to_units`:** theta-structure's key (equal up to gate and value scale) subsumes xof-structure's. Add linear-fold's `layer_to_affine` and `level_ranges`.
- **`SwiGLU.from_matrix`:** merge cheap-gates' `cheap`/`bounds`/`form` with linear-fold's `norm=`. Bounds must come from `level_ranges`. `tests/glu_test.py` expects 1 BOS unit under cheap-gates, 2 in base.
- **Builders:** `xs3.py` is forked. xof-structure's copy has `cols`; unit-synthesis's has lazy4/lazy4c/min-parity. combined-1 is unit-synthesis's copy plus packing, and still needs `cols` ported back. theta-structure's `keccak_theta.py`/`tsx` duplicates the same layouts and is 1–2.5% larger. Its E-scaling fix exists only in brainstorm-critic's `ts_e_scale.diff`; `xs3` already scales E by 1/2.

### (3) Second wave, ranked by expected gain

1. **Needs your decision: weight tying across XOF steps**, counting shared parameters once as PyTorch does. With a residual stream it is about 2x at 3 steps (30.3M vs 60.7M). Without a residual I estimate only about 1.1x at 3 steps (about 55M at depth 9), because lazy chi and last-round pruning are lost. The gap grows with steps: untied costs about +42M per step, tied stays flat, so tying reaches 2.8x or more by 5 steps.
2. **Needs your decision: sharing weights along z** (the architecture avenue's outlook, about 1e5 parameters). It has the highest ceiling but is not an MLP.
3. **Correctness hardening (mandatory, no size gain):** put the edge-case messages and a comparison against the reference xof into `xofbench.verify`, and re-check every claim. The brief's own glu_chi_iota reference fails the edge check (1294 wrong bits).
4. **Few-unit parity that stays flat on count features:** brainstorm-critic's open search with 3 knots between integers and a slope objective. Worth −5 to −10% dense at depth 6 (the range-32 parity before the last round) and −10 to −15% at depth 4.
5. **Consolidate the compiler passes** as in (2). This buys generality, not size (3% or less).
6. **Hybrid depth-4 layout** (one-layer round 1 from raw bits, then X2, column-parity lazy chi, Walsh last round): estimated −8% dense, +18% sparse. Low priority.
7. **Depth 3:** estimated at more than 300M dense and 1M sparse, beyond the 1.25x budget. Skip.

**Unverified claims:**
- brainstorm-critic's lower bounds are heuristic, and its "depth ≥ 5–6" is already broken by xof-structure's depth 4.
- xof-structure's depth-3 figure is an estimate, not a measurement.
- The architecture avenue's accounting counts tied weights once, excludes the embedding and readout from depth, and its gated residual is not a standard residual.
- The pass stacks were never measured together; the only joint measurement is cheap-gates on top of dce-const (−5%).
- The lazy4c layouts have a margin of 3.6e-3 to 5.1e-3, the least headroom of the set, and nobody has checked them with more than one lazy round.
- A leftover unit-synthesis search, `t_lazy_AD.py` (PID 2449124), has been running since 07:40 and should be stopped by hand.

All files are in `/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad/xof/av/combined-1/`:
- `repo/` (merged compiler)
- `xs3.py` (my changes: `xs3_vs_unit_synthesis.diff`)
- `xs4.py`, `xs.py`, `decode.py`, `shared_theta.py`, `xc.py`
- `results.jsonl`
- `patch.diff`, `patch_vs_base.diff`
- `runs/` (harness, `adv_*`, `val_*`, `steps_*` outputs)
- `scripts/layers.py`

Run a variant with: `PYTHONPATH=$C/repo/src:$C:$S/xof $S/venv/bin/python $S/xof/xofbench.py --log-w 6 --depth 3 --variant xs3:lazy4c_middle_m3_mp`
# Optimization levels of reifier's compiler

`reifier.opt.Compiler(level)` picks a point on one ladder, from the most robust to the most
aggressive: `"ultra"` (T5), `"hardened"` (T4), `"robust"` (T3), `"O2"` (T2), `"O3"` (T1).
A level is a tier, not a single recipe: it compiles every recipe in
`reifier.opt.RECIPES` whose guaranteed tier is at least its own (`candidates(level)`; all
but `hardened`, which `ultra_s` always beats, see "Recipes") and keeps the smallest (dense
parameters, then nonzeros; ties go to the more robust recipe; where the tier includes
float16, recipes whose float16 bound fits come first).
`Compiler.chosen` names the recipe kept. The candidate sets nest, so on every circuit
`dense(ultra) >= dense(hardened) >= dense(robust) >= dense(O2) >= dense(O3)`: no level is
ever larger than a more robust one, and a level is never dominated by another level.

- `select=False` compiles only the named recipe; `knobs` are laid over every compiled
  recipe's knobs (`Compiler("robust", select=False, knobs={"q": 8})` keeps q = 8). The
  drivers' configs keep the first form (see `robust_eval.compiler`).
- main's `Compiler()` (`reifier.tensors.compilation`) is the core's tree compiler, which
  `reifier.opt` leaves unchanged. It is not a level.
- `"O1"` (wave 6) was merged into `"robust"`.
- Every level keeps the architecture and the input-to-output mapping: a stack of
  `MLP_SwiGLU` layers (RMSNorm, then wo(silu(wg n) * (wv n)), no biases, no skips) that
  reads [BOS, bits] and writes [BOS, outputs], read relative to BOS. What differs is
  inside: gate constructions, error-correcting copies, always-0 / always-1 features,
  scales.
- Above the ladder, an **extended input** (the host appends always-0 features, and
  optionally repeats every input bit and BOS) makes the first layer shift-invariant too.
  It changes the circuit's input interface, so it is not a level; see "Beyond T5" (measured
  with wave 7's library: the trim removed its knobs, see "Code").

## At a glance

Dense parameters, the recipe each level keeps, and its tier measured on the merge's fresh input set (seed 3100; each level was measured up to its claim plus one, so the T5 / T3 shown for T4 / T2 / T1 levels is what their compile reached; details below):

| circuit | main (default) | ultra (T5) | hardened (T4) | robust (T3) | O2 (T2) | O3 (T1) |
|---|---|---|---|---|---|---|
| xof_w4 | 206.80M | 54.71M `ultra` T5 | 27.48M `hardened_xf_c` T4 | 15.20M `robust_fcx1_cc` T3 | 8.52M `O2` T2 | 6.31M `O3` T1 |
| sha3_w2 | 93.99M | 38.23M `ultra` T5 | 18.90M `hardened_xf_c` T4 | 9.83M `robust_fcx1_cc` T3 | 5.98M `O2` T2 | 4.92M `O3` T1 |
| sha256_2r | 46.70M | 5.86M `ultra` T5 | 5.77M `ultra_bc64` T5 | 5.22M `robust_fcf_c` T3 | 5.12M `robust_fcf` T3 | 5.12M `robust_fcf` T3 |
| adder32 | 1.29M | 201.3K `ultra` T5 | 198.4K `ultra_bc64` T5 | 161.0K `robust` T3 | 155.8K `robust_fcf` T3 | 155.8K `robust_fcf` T3 |
| add4x16 | 764.9K | 193.2K `ultra` T5 | 191.4K `ultra_bc64` T5 | 155.0K `robust_fcf_c` T3 | 153.2K `robust_fcf` T3 | 153.2K `robust_fcf` T3 |
| backdoor_w2 | 94.00M | 38.23M `ultra` T5 | 18.90M `hardened_xf_c` T4 | 9.83M `robust_fcx1_cc` T3 | 5.98M `O2` T2 | 4.93M `O3` T1 |
| sandbagger_w1 | 187.49M | 46.70M `ultra_s` T5 | 20.10M `hardened_xf_c` T4 | 10.71M `robust_fcx1_cc` T3 | 6.43M `O2` T2 | 5.34M `O3` T1 |
| extractor_128x64 | 37.85M | 42.49M `ultra` T5 | 12.83M `hardened_xf_c` T4 | 7.57M `robust_x1_c` T3 | 4.84M `O2` T2 | 928.4K `O3` T1 |
| parity64 | 79.2K | 44.9K `ultra` T5 | 44.5K `ultra_bc64` T5 | 9.2K `robust_x1_c` T3 | 8.1K `O2` T3 | 4.6K `O3` T1 |
| sha256_r2 (size only) | 11.4G | 8.87M `ultra` | 8.74M `ultra_bc64` | 7.67M `robust` | 7.36M `robust_fcf` | 7.36M `robust_fcf` |

## The ladder

| level | tier | guarantee | candidates (`RECIPES`, at its tier or above) | measured beyond the tier |
|---|---|---|---|---|
| `"ultra"` | **T5** | T4, and next to an LLM's features: norm scale down to 0.05, LayerNorm shift M 0.1 relative to the largest feature, noise n 0.01 relative to the largest feature, with T4's m 0.1 and e 0.01, all together | `ultra` (flat units; circuits of up to 200 layers), `ultra_s` (steps only, any depth; float16 at any width) | one at a time (ultra, wave-7 ultra-noise): n 0.02 (sha256_2r 0.01), M 0.2, s 0.02 (sandbagger 0.01) |
| `"hardened"` | **T4** | T3, and norm scale down to 0.1, LayerNorm mean shift m 0.1 and interference e 0.01 together | + `ultra_bc64` (up to 200 layers), `hardened_xf_c` (`hardened`, wave 6's recipe, only with `select=False`) | where it keeps an ultra recipe, T5 too (adders, sha256_2r, parity64) |
| `"robust"` | **T3** | bf16 and fp16 on CUDA with reduced-precision reductions, any batch | + `robust`, `robust_x1_c`, `robust_fcx1_cc`, `robust_fcf_c` | wave 6's robust recipe (kept on adder32): s 0.05, m 0.01, e 0.005 one at a time, AND / OR / equality trees of up to 2048 inputs, adders of up to 256 bits; the wave-7 T3 recipes were not measured beyond T3 (no LayerNorm invariance: T4 fails) |
| `"O2"` | **T2** | bf16 with float32 accumulation | + `O2`, `robust_fcf` | T3 where it keeps `robust_fcf` (adders, sha256_2r) or on parity64 |
| `"O3"` | **T1** | float32, worst error <= 0.01 | + `O3` (glu_xor sums of <= 32 inputs) | T3 where it keeps `robust_fcf` |
| main's `Compiler()` (`reifier.tensors.compilation`) | main's | main's tree compiler, bit-identical weights (not a level) | | |
| extended input (presets, not levels) | **T6** on 4 of the 5 circuits measured on every base (adder32: 1 of 32 T6 runs misses boolify); **T7** on CPU | `t6x`: the host appends 16 always-0 features to [BOS, x] (`input_zeros=16`); `t7x16b`: also every bit and BOS 16 times | `ext_eval.CONFIGS`: `toward_t6` / `toward_t7` + `input_zeros` (+ `input_rep`, `input_bos`; wave 7's library) | see "Beyond T5" |

**Recipes** (`reifier.opt.RECIPES`; a recipe is a candidate of every level at or below its
tier, except `hardened`):

| recipe | tier | contents |
|---|---|---|
| `ultra` | T5 | robust's gates with flat units for AND / OR / NOT / copy of <= 4 inputs, step outputs and a first layer of flat input copies; exact steps centered on 1/2 (margins 3/8 on both sides); q 128; BOS = 1 with one BOS copy per 32 features; 8 always-0 features with every row spread evenly over them (LayerNorm invariance after the first layer); per-layer output scales up to 1024 within float16; xor trees of 4, AND / OR trees of 2, the prefix adder; exact readout. `MAX_DEPTH` 200 |
| `ultra_s` | T5 | wave 6's hardened (steps only, step prologue) with `fit16` (every layer rescaled by powers of 2 to the float16 range), BOS = 1, centered steps, 2 always-0 features spread evenly |
| `ultra_bc64` | T4 | `ultra` with one BOS copy per 64 features (1-2% smaller; the T5 point failed 1 of 48 runs on the extractor). `MAX_DEPTH` 200 |
| `hardened_xf_c` | T4 | wave 6's hardened with parities as flat xor trees (sums of vertex indicators) and step copies of unit outputs (`clean_outputs`) |
| `hardened` | T4 | wave 6's hardened: exact steps, q 256, heavy BOS, a prologue of shift-tolerant input copies, 8 always-0 features, xor<=4 and AND/OR-2 trees, the prefix adder, exact readout; dead code, dedup, NOT/copy folds. The hardened level's own recipe (`select=False`), and no candidate: `ultra_s` compiles the same gate graph with 2 instead of 8 always-0 features per layer, so it is always smaller, and fit16 keeps it within float16's range (no selection kept `hardened` in the wave-8 checks: 150 suite and 4,500 random compiles) |
| `robust` | T3 | wave 6's robust: exact steps, q 64, heavy BOS, the same trees and adder, exact readout; flat units for gates of <= 4 inputs |
| `robust_x1_c` | T3 | robust + parities as E-exact xor<=4 trees with a level of flat copies after every xor layer + clean outputs |
| `robust_fcx1_cc` | T3 | robust + flat cones of <= 3 bits + E-exact xor trees, flat copies after every cone or xor layer (`cone_reclean`) + clean outputs |
| `robust_fcf_c` | T3 | robust + flat cones + clean outputs |
| `O2` | T2 | wave 6's O2: E-exact xor trees and cones, flat units, main's adder |
| `robust_fcf` | T2 | robust + flat cones (T3 on 8 of 9: on xof_w4 its unit outputs miss boolify at bf16+gpu b256) |
| `O3` | T1 | wave 6's O3 (one layer of units per symmetric sum, cones exact in real arithmetic, the prefix adder), with xors of more than 32 inputs built as trees of xor32 |

## Threat model and tiers

A threat is a `robust_eval.py` spec. Every output bit must be correct by nearest rounding
(|out/BOS - bit| < 0.5) and by reifier's boolify (1 iff within 0.02 of BOS).

| tier | requirement (all parts together, on fresh inputs at real batches) |
|---|---|
| T1 | float32, worst error <= 0.01 |
| T2 | T1 + bf16 with float32 accumulation (CUDA, reduced-precision reductions off) |
| T3 | T2 + bf16 and fp16 on CUDA with PyTorch's default reduced-precision reductions, any batch |
| T4 | T3 + s0.1 + m0.1 + e0.01 (8 seeds) |
| T5 | T4 + s0.05 + M0.1 + n0.01, i.e. s0.05+m0.1+e0.01+M0.1+n0.01 (8 seeds, f32, bf16 and bf16+gpu bases) |
| T6 | T5 with s0.01 + M0.3 + n0.03 |
| T7 | aspirational: s0.002 + M1 + n0.1 |

The perturbations model a circuit whose features run in the same residual stream as an
LLM's (per token and per layer, before each RMSNorm):
- `s<x>`: the circuit holds a varying part of the norm, so its normalized features are
  scaled by a factor drawn log-uniform in [x, 1];
- `m<x>`: a LayerNorm host subtracts the stream mean, x N(0, 1) in units of the circuit's rms;
- `e<x>`: interference x rms N(0, 1) added to every circuit feature;
- `n<x>`: interference relative to the largest feature, x max|feature| N(0, 1);
- `M<x>`: a LayerNorm shift relative to the largest normalized feature.

`m` and `e` can be beaten by padding a circuit with always-0 features or by a heavy BOS;
`n` and `M` cannot, which is why T5-T7 use them.

**How the final numbers were measured** (`ladder_eval.py`, the merge's run): every level
compiled with selection, on a fresh input set (seed 3100, which no track used):
`suite.inputs(n_in, 256)` (7 edge cases and 256 random inputs at densities 0.05 / 0.5 /
0.95), 128 dense inputs (density U[0.8, 1]) and 96 near-constant inputs (1-3 zeros among
ones, 1-3 ones among zeros): 487 per circuit. Fresh perturbation seeds (3100-3107, 3200-3201).
Every batch really has b rows. Groups: T1 f32 gpu_exact b256; T2 bf16 gpu_exact b256; T3
bf16 and fp16 on CUDA at b1, b77, b256 and b1024; T4 8 seeds x {bf16, fp16} x {b256,
b1024} plus 2 seeds x {bf16, fp16} at b1 (36 runs); T5 8 seeds x {f32 gpu_exact, bf16
gpu_exact, bf16 CUDA, fp16 CUDA} at b256 plus bf16 CUDA b1024 plus 2 seeds x {bf16,
fp16} at b1 (44 runs). Each level ran its groups up to its claimed tier plus one.

**Limits of the threat model** (what the tiers do not cover):
- The perturbations are random (Gaussian shifts and noise, log-uniform scales), drawn
  independently per token and layer. Nothing here is adversarial, and failure rates
  are measured, not bounded: a tier means 0 wrong bits in every run of its protocol
  (487 inputs x 36-44 perturbed runs per circuit), not a proof.
- Rare inputs: on 262,144 dense extractor inputs the robust level misread a 0 on 2
  inputs at bf16+gpu b256 (pareto verifier; deterministic); the 487-input protocol does
  not see such rates.
- `s` is a scale-down only, drawn per token and layer. A host that scales the circuit
  up (s > 1) can overflow float16 (ultra-scale verifier: its fit16 construction failed at
  s = 2-3 in fp16; bf16 held to s = 8), and an `s` held at 0.01-0.02 in every layer of a
  token (not drawn) broke sha256_2r for the ultra-scale construction.
- "Any batch" was sampled (b1, b77, b256, b1024 here; b2-b8192 in the tracks and
  verifiers), not exhausted.
- The input is the host's: BOS = 1 and one copy of each bit. A LayerNorm shift is
  measured against them, so the first layer cannot cancel it the way later layers do
  (every later layer reads always-0 features and its rows sum to 0). This is what stops
  T6 and T7 with the standard input; the extended input removes it.

## Per circuit: every level on fresh inputs (the merge's final run)

Each level compiled with selection (`Compiler(level=...)`; recipe kept = `Compiler.chosen`),
measured with `ladder_eval.py` as described above (input set 3100, 487 inputs, perturbation
seeds 3100-3107 / 3200-3201). Measured tier: the highest Tk with T1..Tk all correct, up to
the claim plus one (levels that keep the same weights share their runs). x main = main's
default compile / this (dense). Runs: failing / total in the T3, T4 and T5 groups (a group
above the claim stops at its first failure); worst error: the largest |out/BOS - bit|.

| circuit | level | recipe kept | depth | dense | x main | sparse | measured tier (claim) | runs: T3 / T4 / T5 failing of total | worst error T3 / T4 / T5 |
|---|---|---|---|---|---|---|---|---|---|
| xof_w4 | ultra | ultra | 22 | 54,705,494 | 3.78 | 1,410,355 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.026 / 0.037 / 0.037 |
| xof_w4 | hardened | hardened_xf_c | 14 | 27,483,832 | 7.52 | 364,686 | T4 (T4) | 0/8 / 0/36 / 1/1 | 0.022 / 0.032 / fails |
| xof_w4 | robust | robust_fcx1_cc | 17 | 15,203,968 | 13.60 | 95,948 | T3 (T3) | 0/8 / 1/1 / . | 0.060 / fails / . |
| xof_w4 | O2 | O2 | 9 | 8,523,489 | 24.26 | 62,337 | T2 (T2) | 1/2 / . / . | fails / . / . |
| xof_w4 | O3 | O3 | 6 | 6,312,977 | 32.76 | 74,506 | T1 (T1) | . / . / . | . / . / . |
| sha3_w2 | ultra | ultra | 169 | 38,234,420 | 2.46 | 1,790,076 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.014 / 0.026 / 0.026 |
| sha3_w2 | hardened | hardened_xf_c | 98 | 18,898,888 | 4.97 | 902,419 | T4 (T4) | 0/8 / 0/36 / 1/1 | 0.000 / 0.022 / fails |
| sha3_w2 | robust | robust_fcx1_cc | 143 | 9,834,515 | 9.56 | 245,179 | T3 (T3) | 0/8 / 1/1 / . | 0.000 / fails / . |
| sha3_w2 | O2 | O2 | 80 | 5,976,913 | 15.73 | 168,303 | T2 (T2) | 1/3 / . / . | fails / . / . |
| sha3_w2 | O3 | O3 | 51 | 4,922,840 | 19.09 | 238,475 | T1 (T1) | . / . / . | . / . / . |
| sha256_2r | ultra | ultra | 54 | 5,860,317 | 7.97 | 336,750 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.029 / 0.038 / 0.037 |
| sha256_2r | hardened | ultra_bc64 | 54 | 5,767,756 | 8.10 | 274,752 | T5 (T4) | 0/8 / 0/36 / 0/44 | 0.027 / 0.034 / 0.037 |
| sha256_2r | robust | robust_fcf_c | 53 | 5,220,434 | 8.95 | 68,143 | T3 (T3) | 0/8 / 1/1 / . | 0.008 / fails / . |
| sha256_2r | O2 | robust_fcf | 52 | 5,120,459 | 9.12 | 66,728 | T3 (T2) | 0/8 / . / . | 0.014 / . / . |
| sha256_2r | O3 | robust_fcf | 52 | 5,120,459 | 9.12 | 66,728 | T3 (T1) | . / . / . | . / . / . |
| adder32 | ultra | ultra | 8 | 201,268 | 6.43 | 19,271 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.023 / 0.026 / 0.028 |
| adder32 | hardened | ultra_bc64 | 8 | 198,410 | 6.53 | 17,348 | T5 (T4) | 0/8 / 0/36 / 0/44 | 0.018 / 0.025 / 0.026 |
| adder32 | robust | robust | 7 | 160,961 | 8.04 | 4,956 | T3 (T3) | 0/8 / 1/1 / . | 0.000 / fails / . |
| adder32 | O2 | robust_fcf | 7 | 155,796 | 8.31 | 4,883 | T3 (T2) | 0/8 / . / . | 0.006 / . / . |
| adder32 | O3 | robust_fcf | 7 | 155,796 | 8.31 | 4,883 | T3 (T1) | . / . / . | . / . / . |
| add4x16 | ultra | ultra | 13 | 193,197 | 3.96 | 21,152 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.021 / 0.026 / 0.026 |
| add4x16 | hardened | ultra_bc64 | 13 | 191,383 | 4.00 | 19,947 | T5 (T4) | 0/8 / 0/36 / 0/44 | 0.021 / 0.025 / 0.025 |
| add4x16 | robust | robust_fcf_c | 13 | 154,997 | 4.93 | 6,183 | T3 (T3) | 0/8 / 1/1 / . | 0.000 / fails / . |
| add4x16 | O2 | robust_fcf | 12 | 153,246 | 4.99 | 6,000 | T3 (T2) | 0/8 / . / . | 0.000 / . / . |
| add4x16 | O3 | robust_fcf | 12 | 153,246 | 4.99 | 6,000 | T3 (T1) | . / . / . | . / . / . |
| backdoor_w2 | ultra | ultra | 169 | 38,234,420 | 2.46 | 1,790,076 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.014 / 0.025 / 0.026 |
| backdoor_w2 | hardened | hardened_xf_c | 98 | 18,898,888 | 4.97 | 902,415 | T4 (T4) | 0/8 / 0/36 / 1/1 | 0.000 / 0.020 / fails |
| backdoor_w2 | robust | robust_fcx1_cc | 143 | 9,834,515 | 9.56 | 245,179 | T3 (T3) | 0/8 / 1/1 / . | 0.000 / fails / . |
| backdoor_w2 | O2 | O2 | 80 | 5,976,913 | 15.73 | 168,307 | T2 (T2) | 1/3 / . / . | fails / . / . |
| backdoor_w2 | O3 | O3 | 52 | 4,925,193 | 19.09 | 238,650 | T1 (T1) | . / . / . | . / . / . |
| sandbagger_w1 | ultra | ultra_s | 465 | 46,695,250 | 4.02 | 1,242,674 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.059 / 0.094 / 0.128 |
| sandbagger_w1 | hardened | hardened_xf_c | 267 | 20,095,847 | 9.33 | 1,318,316 | T4 (T4) | 0/8 / 0/36 / 1/1 | 0.053 / 0.091 / fails |
| sandbagger_w1 | robust | robust_fcx1_cc | 400 | 10,714,574 | 17.50 | 387,281 | T3 (T3) | 0/8 / 1/1 / . | 0.043 / fails / . |
| sandbagger_w1 | O2 | O2 | 225 | 6,433,060 | 29.14 | 261,297 | T2 (T2) | 1/2 / . / . | fails / . / . |
| sandbagger_w1 | O3 | O3 | 145 | 5,343,596 | 35.09 | 356,924 | T1 (T1) | . / . / . | . / . / . |
| extractor_128x64 | ultra | ultra | 9 | 42,493,361 | 0.89 | 881,189 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.018 / 0.026 / 0.026 |
| extractor_128x64 | hardened | hardened_xf_c | 6 | 12,833,317 | 2.95 | 250,080 | T4 (T4) | 0/8 / 0/36 / 1/1 | 0.012 / 0.023 / fails |
| extractor_128x64 | robust | robust_x1_c | 7 | 7,570,911 | 5.00 | 47,086 | T3 (T3) | 0/8 / 1/1 / . | 0.015 / fails / . |
| extractor_128x64 | O2 | O2 | 4 | 4,842,197 | 7.82 | 39,215 | T2 (T2) | 1/2 / . / . | fails / . / . |
| extractor_128x64 | O3 | O3 | 2 | 928,362 | 40.77 | 67,039 | T1 (T1) | . / . / . | . / . / . |
| parity64 | ultra | ultra | 8 | 44,911 | 1.76 | 5,394 | T5 (T5) | 0/8 / 0/36 / 0/44 | 0.006 / 0.024 / 0.025 |
| parity64 | hardened | ultra_bc64 | 8 | 44,478 | 1.78 | 5,122 | T5 (T4) | 0/8 / 0/36 / 0/44 | 0.006 / 0.025 / 0.023 |
| parity64 | robust | robust_x1_c | 7 | 9,153 | 8.66 | 954 | T3 (T3) | 0/8 / 1/1 / . | 0.000 / fails / . |
| parity64 | O2 | O2 | 4 | 8,066 | 9.82 | 791 | T3 (T2) | 0/8 / . / . | 0.000 / . / . |
| parity64 | O3 | O3 | 3 | 4,637 | 17.09 | 1,282 | T1 (T1) | . / . / . | . / . / . |

Failing runs within a level's claim: none


The first failing run of each group above the claim (the next tier's threats break the recipe kept; this is what bounds each level's tier from above):

- add4x16 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 812, misread 817
- adder32 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 718, misread 1612
- backdoor_w2 O2 T3: bf16+gpu+b256: wrong 3376, misread 3503
- backdoor_w2 O3 T2: bf16+gpu_exact+b256: wrong 3279, misread 3460
- backdoor_w2 hardened T5: f32+gpu_exact+b256+s0.05+m0.1+e0.01+M0.1+n0.01+seed3100: wrong 3379, misread 3382
- backdoor_w2 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 3458, misread 3430
- extractor_128x64 O2 T3: bf16+gpu+b77: wrong 0, misread 3
- extractor_128x64 O3 T2: bf16+gpu_exact+b256: wrong 13540, misread 10885
- extractor_128x64 hardened T5: f32+gpu_exact+b256+s0.05+m0.1+e0.01+M0.1+n0.01+seed3100: wrong 15052, misread 14488
- extractor_128x64 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 8801, misread 8304
- parity64 O3 T2: bf16+gpu_exact+b256: wrong 238, misread 328
- parity64 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 122, misread 129
- sandbagger_w1 O2 T3: bf16+gpu+b77: wrong 129, misread 116
- sandbagger_w1 O3 T2: bf16+gpu_exact+b256: wrong 105, misread 103
- sandbagger_w1 hardened T5: f32+gpu_exact+b256+s0.05+m0.1+e0.01+M0.1+n0.01+seed3100: wrong 166, misread 166
- sandbagger_w1 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 176, misread 176
- sha256_2r robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 26368, misread 26374
- sha3_w2 O2 T3: bf16+gpu+b256: wrong 3474, misread 3385
- sha3_w2 O3 T2: bf16+gpu_exact+b256: wrong 3308, misread 3402
- sha3_w2 hardened T5: f32+gpu_exact+b256+s0.05+m0.1+e0.01+M0.1+n0.01+seed3100: wrong 3378, misread 3382
- sha3_w2 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 3464, misread 3434
- xof_w4 O2 T3: bf16+gpu+b77: wrong 13237, misread 16030
- xof_w4 O3 T2: bf16+gpu_exact+b256: wrong 18717, misread 20675
- xof_w4 hardened T5: f32+gpu_exact+b256+s0.05+m0.1+e0.01+M0.1+n0.01+seed3100: wrong 39085, misread 37684
- xof_w4 robust T4: bf16+gpu+b256+s0.1+m0.1+e0.01+seed3100: wrong 32974, misread 32664


In all, 1,338 runs within the levels' claims (each distinct compile counted once), 0 with a
wrong or misread bit; 240 more runs above the claims bound the tiers from above. Every
level's compile here is bit-identical to the final library's (re-hashed on CPU). The T4
and T5 levels were measured again on a second fresh set (seed 3300; below).


## Per-circuit Pareto tables

The brief's rule: for every suite circuit and level L, no other level and no known
configuration K may have tier(K) >= T(L) and dense(K) < dense(L) (ties on sparse). Levels
cannot dominate each other (their candidate sets nest). Known configurations are every
configuration the three tracks and their verifiers measured on all 9 circuits (37 of the
pareto track, the ultra variants of the two robustness tracks) plus the merge's final run;
`scripts/frontier.py` (in the merge's work directory) builds these tables. tier(K) is read
two ways: K's *known* tier (its minimum over the suite, capped by failures out of the suite;
the reading a compiler can act on), and K's tier on that circuit alone (*: smaller than the
level).

### Known tier of every configuration

| config | known tier | per circuit (T) | note |
|---|---|---|---|
| O1 | T1 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:1, parity64:1 |  |
| O2 | T2 | xof_w4:2, sha3_w2:2, sha256_2r:2, adder32:3, add4x16:3, backdoor_w2:2, sandbagger_w1:2, extractor_128x64:2, parity64:3 |  |
| O2_c | T2 | xof_w4:2, sha3_w2:2, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:2, sandbagger_w1:2, extractor_128x64:3, parity64:3 |  |
| O2_prefix | T1 | xof_w4:2, sha3_w2:2, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:2, sandbagger_w1:2, extractor_128x64:2, parity64:3 |  |
| O3 | T1 | xof_w4:1, sha3_w2:1, sha256_2r:1, adder32:1, add4x16:1, backdoor_w2:1, sandbagger_w1:1, extractor_128x64:1, parity64:1 |  |
| O3_x | T1 | xof_w4:2, sha3_w2:2, sha256_2r:1, adder32:1, add4x16:1, backdoor_w2:2, sandbagger_w1:1, extractor_128x64:3, parity64:1 |  |
| hardened | T4 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_c_c | T3 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_cfcx1_c | incomplete | xof_w4:2, sha3_w2:2, sha256_2r:3, adder32:4, add4x16:4, backdoor_w2:., sandbagger_w1:., extractor_128x64:2, parity64:4 |  |
| hardened_cx1_c | incomplete | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:4, add4x16:4, backdoor_w2:., sandbagger_w1:., extractor_128x64:2, parity64:4 |  |
| hardened_cxf_c | T3 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_fc | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:4, extractor_128x64:3, parity64:4 |  |
| hardened_l1 | T3 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_l2 | T4 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_l2_cfc_c | T3 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_l2_cxf_c | T3 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_l2_fcf_c | T4 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_l2_fcxf_c | T3 | xof_w4:4, sha3_w2:3, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:4, parity64:4 |  |
| hardened_l2_xf_c | T3 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:3, parity64:4 | extractor: T4 fails in 4/128 and 8/128 runs (pareto verifier) |
| hardened_xf | T3 | xof_w4:3, sha3_w2:3, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:3, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| hardened_xf_c | T4 | xof_w4:4, sha3_w2:4, sha256_2r:4, adder32:4, add4x16:4, backdoor_w2:4, sandbagger_w1:4, extractor_128x64:4, parity64:4 |  |
| main | T1 | xof_w4:1, sha3_w2:1, sha256_2r:1, adder32:1, add4x16:1, backdoor_w2:1, sandbagger_w1:1, extractor_128x64:1, parity64:1 |  |
| main_bf16 | T1 | xof_w4:2, sha3_w2:3, sha256_2r:2, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:1, parity64:1 |  |
| robust | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_fcf | T2 | xof_w4:2, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_fcf_c | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_fcx1_c | T2 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:2, extractor_128x64:3, parity64:3 |  |
| robust_fcx1_cc | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_fcxf | T2 | xof_w4:2, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_fcxf_c | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_nc | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_o2 | T2 | xof_w4:2, sha3_w2:2, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:2, sandbagger_w1:2, extractor_128x64:3, parity64:3 |  |
| robust_x | T2 | xof_w4:2, sha3_w2:2, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:2, sandbagger_w1:2, extractor_128x64:2, parity64:3 |  |
| robust_x1 | T2 | xof_w4:2, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_x1_c | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_x2 | T1 | xof_w4:2, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_x2_c | T2 | xof_w4:2, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_xf | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| robust_xf_c | T3 | xof_w4:3, sha3_w2:3, sha256_2r:3, adder32:3, add4x16:3, backdoor_w2:3, sandbagger_w1:3, extractor_128x64:3, parity64:3 |  |
| t5_flat | T4 | xof_w4:5, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:5, extractor_128x64:4, parity64:5 | extractor: the T5 point fails at bf16+gpu b32 (set 400) |
| t5_steps | T5 | xof_w4:5, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:5, extractor_128x64:5, parity64:5 |  |
| t5rzo | incomplete | xof_w4:4, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:5, extractor_128x64:4, parity64:5 | one input set only (400); T5 fails on xof_w4 and the extractor; T4 partly measured |
| toward_t6 | T2 | xof_w4:2, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:5, extractor_128x64:5, parity64:5 | xof_w4: float16 overflows in the first layer (q 512) |
| ultra | T5 | xof_w4:5, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:3, extractor_128x64:5, parity64:5 | flat units: candidate up to 200 layers; on the sandbagger (465) T4 fails in ~3% of runs |
| ultra_bc64 | T4 | xof_w4:5, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:3, extractor_128x64:4, parity64:5 | extractor: the T5 point fails 1 of 48 runs (CPU f32, set 500); sandbagger: T4 fails 1 of 64 runs; candidate up to 200 layers |
| ultra_s | T5 | xof_w4:5, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:5, extractor_128x64:5, parity64:5 |  |
| ultra_s_b64 | T5 | xof_w4:5, sha3_w2:5, sha256_2r:5, adder32:5, add4x16:5, backdoor_w2:5, sandbagger_w1:5, extractor_128x64:5, parity64:5 |  |

### Per circuit and tier: the level, the smallest known configuration, and the smallest by this circuit's own tier

| circuit | tier | level (recipe kept) | its tier here (set 3100) | smallest with known tier >= T | smallest with tier here >= T |
|---|---|---|---|---|---|
| xof_w4 | T5 | 54,705,494 (ultra) | T5 | 54,705,494 (ultra) | 53,869,702 (ultra_bc64)* |
| xof_w4 | T4 | 27,483,832 (hardened_xf_c) | T4 | 27,483,832 (hardened_xf_c) | 21,123,632 (hardened_l2_fcxf_c)* |
| xof_w4 | T3 | 15,203,968 (robust_fcx1_cc) | T3 | 15,203,968 (robust_fcx1_cc) | 14,091,586 (robust_fcx1_c)* |
| xof_w4 | T2 | 8,523,489 (O2) | T2 | 8,523,489 (O2) | 7,957,928 (O3_x)* |
| xof_w4 | T1 | 6,312,977 (O3) | T1 | 6,312,977 (O3) | 6,312,977 (O3) |
| sha3_w2 | T5 | 38,234,420 (ultra) | T5 | 38,234,420 (ultra) | 37,281,835 (t5rzo)* |
| sha3_w2 | T4 | 18,898,888 (hardened_xf_c) | T4 | 18,898,888 (hardened_xf_c) | 16,170,866 (hardened_l2_cxf_c)* |
| sha3_w2 | T3 | 9,834,515 (robust_fcx1_cc) | T3 | 9,834,515 (robust_fcx1_cc) | 9,121,354 (robust_fcx1_c)* |
| sha3_w2 | T2 | 5,976,913 (O2) | T2 | 5,976,913 (O2) | 5,756,275 (O3_x)* |
| sha3_w2 | T1 | 4,922,840 (O3) | T1 | 4,922,840 (O3) | 4,922,840 (O3) |
| sha256_2r | T5 | 5,860,317 (ultra) | T5 | 5,860,317 (ultra) | 5,718,639 (t5rzo)* |
| sha256_2r | T4 | 5,767,756 (ultra_bc64) | T5 | 5,767,756 (ultra_bc64) | 5,306,083 (hardened_l2_cfc_c)* |
| sha256_2r | T3 | 5,220,434 (robust_fcf_c) | T3 | 5,220,434 (robust_fcf_c) | 5,120,459 (robust_fcf)* |
| sha256_2r | T2 | 5,120,459 (robust_fcf) | T3 | 5,120,459 (robust_fcf) | 5,120,459 (robust_fcf) |
| sha256_2r | T1 | 5,120,459 (robust_fcf) | T3 | 5,120,459 (robust_fcf) | 5,120,459 (robust_fcf) |
| adder32 | T5 | 201,268 (ultra) | T5 | 201,268 (ultra) | 198,410 (ultra_bc64)* |
| adder32 | T4 | 198,410 (ultra_bc64) | T5 | 198,410 (ultra_bc64) | 192,272 (hardened_l2_cfc_c)* |
| adder32 | T3 | 160,961 (robust) | T3 | 160,961 (robust) | 155,796 (robust_fcf)* |
| adder32 | T2 | 155,796 (robust_fcf) | T3 | 155,796 (robust_fcf) | 155,796 (robust_fcf) |
| adder32 | T1 | 155,796 (robust_fcf) | T3 | 155,796 (robust_fcf) | 155,796 (robust_fcf) |
| add4x16 | T5 | 193,197 (ultra) | T5 | 193,197 (ultra) | 191,383 (ultra_bc64)* |
| add4x16 | T4 | 191,383 (ultra_bc64) | T5 | 191,383 (ultra_bc64) | 185,724 (hardened_l2_cfc_c)* |
| add4x16 | T3 | 154,997 (robust_fcf_c) | T3 | 154,997 (robust_fcf_c) | 153,246 (robust_fcf)* |
| add4x16 | T2 | 153,246 (robust_fcf) | T3 | 153,246 (robust_fcf) | 153,246 (robust_fcf) |
| add4x16 | T1 | 153,246 (robust_fcf) | T3 | 153,246 (robust_fcf) | 153,246 (robust_fcf) |
| backdoor_w2 | T5 | 38,234,420 (ultra) | T5 | 38,234,420 (ultra) | 37,281,835 (t5rzo)* |
| backdoor_w2 | T4 | 18,898,888 (hardened_xf_c) | T4 | 18,898,888 (hardened_xf_c) | 16,170,866 (hardened_l2_cxf_c)* |
| backdoor_w2 | T3 | 9,834,515 (robust_fcx1_cc) | T3 | 9,834,515 (robust_fcx1_cc) | 9,121,354 (robust_fcx1_c)* |
| backdoor_w2 | T2 | 5,976,913 (O2) | T2 | 5,976,913 (O2) | 5,756,275 (O3_x)* |
| backdoor_w2 | T1 | 4,925,193 (O3) | T1 | 4,925,193 (O3) | 4,925,193 (O3) |
| sandbagger_w1 | T5 | 46,695,250 (ultra_s) | T5 | 46,695,250 (ultra_s) | 34,127,610 (t5rzo)* |
| sandbagger_w1 | T4 | 20,095,847 (hardened_xf_c) | T4 | 20,095,847 (hardened_xf_c) | 15,937,014 (hardened_l2_cxf_c)* |
| sandbagger_w1 | T3 | 10,714,574 (robust_fcx1_cc) | T3 | 10,714,574 (robust_fcx1_cc) | 10,029,586 (robust_x2)* |
| sandbagger_w1 | T2 | 6,433,060 (O2) | T2 | 6,433,060 (O2) | 6,433,060 (O2) |
| sandbagger_w1 | T1 | 5,343,596 (O3) | T1 | 5,343,596 (O3) | 5,343,596 (O3) |
| extractor_128x64 | T5 | 42,493,361 (ultra) | T5 | 42,493,361 (ultra) | 42,493,361 (ultra) |
| extractor_128x64 | T4 | 12,833,317 (hardened_xf_c) | T4 | 12,833,317 (hardened_xf_c) | 12,651,622 (hardened_l2_cxf_c)* |
| extractor_128x64 | T3 | 7,570,911 (robust_x1_c) | T3 | 7,570,911 (robust_fcx1_cc) | 4,867,612 (O2_c)* |
| extractor_128x64 | T2 | 4,842,197 (O2) | T2 | 4,842,197 (O2) | 4,842,197 (O2) |
| extractor_128x64 | T1 | 928,362 (O3) | T1 | 928,362 (O3) | 928,362 (O3) |
| parity64 | T5 | 44,911 (ultra) | T5 | 44,911 (ultra) | 44,478 (ultra_bc64)* |
| parity64 | T4 | 44,478 (ultra_bc64) | T5 | 44,478 (ultra_bc64) | 38,064 (hardened_cfcx1_c)* |
| parity64 | T3 | 9,153 (robust_x1_c) | T3 | 9,153 (robust_fcx1_cc) | 8,066 (O2)* |
| parity64 | T2 | 8,066 (O2) | T3 | 8,066 (O2) | 8,066 (O2) |
| parity64 | T1 | 4,637 (O3) | T1 | 4,637 (O3) | 4,637 (O3) |

Known-tier reading: 0 violations

Per-circuit reading: 29 cases (*)
- xof_w4 ultra: level 54,705,494 vs ultra_bc64 53,869,702 (tier here T5, known T4)
- xof_w4 hardened: level 27,483,832 vs hardened_l2_fcxf_c 21,123,632 (tier here T4, known T3)
- xof_w4 robust: level 15,203,968 vs robust_fcx1_c 14,091,586 (tier here T3, known T2)
- xof_w4 O2: level 8,523,489 vs O3_x 7,957,928 (tier here T2, known T1)
- sha3_w2 ultra: level 38,234,420 vs t5rzo 37,281,835 (tier here T5, known None)
- sha3_w2 hardened: level 18,898,888 vs hardened_l2_cxf_c 16,170,866 (tier here T4, known T3)
- sha3_w2 robust: level 9,834,515 vs robust_fcx1_c 9,121,354 (tier here T3, known T2)
- sha3_w2 O2: level 5,976,913 vs O3_x 5,756,275 (tier here T2, known T1)
- sha256_2r ultra: level 5,860,317 vs t5rzo 5,718,639 (tier here T5, known None)
- sha256_2r hardened: level 5,767,756 vs hardened_l2_cfc_c 5,306,083 (tier here T4, known T3)
- sha256_2r robust: level 5,220,434 vs robust_fcf 5,120,459 (tier here T3, known T2)
- adder32 ultra: level 201,268 vs ultra_bc64 198,410 (tier here T5, known T4)
- adder32 hardened: level 198,410 vs hardened_l2_cfc_c 192,272 (tier here T4, known T3)
- adder32 robust: level 160,961 vs robust_fcf 155,796 (tier here T3, known T2)
- add4x16 ultra: level 193,197 vs ultra_bc64 191,383 (tier here T5, known T4)
- add4x16 hardened: level 191,383 vs hardened_l2_cfc_c 185,724 (tier here T4, known T3)
- add4x16 robust: level 154,997 vs robust_fcf 153,246 (tier here T3, known T2)
- backdoor_w2 ultra: level 38,234,420 vs t5rzo 37,281,835 (tier here T5, known None)
- backdoor_w2 hardened: level 18,898,888 vs hardened_l2_cxf_c 16,170,866 (tier here T4, known T3)
- backdoor_w2 robust: level 9,834,515 vs robust_fcx1_c 9,121,354 (tier here T3, known T2)
- backdoor_w2 O2: level 5,976,913 vs O3_x 5,756,275 (tier here T2, known T1)
- sandbagger_w1 ultra: level 46,695,250 vs t5rzo 34,127,610 (tier here T5, known None)
- sandbagger_w1 hardened: level 20,095,847 vs hardened_l2_cxf_c 15,937,014 (tier here T4, known T3)
- sandbagger_w1 robust: level 10,714,574 vs robust_x2 10,029,586 (tier here T3, known T1)
- extractor_128x64 hardened: level 12,833,317 vs hardened_l2_cxf_c 12,651,622 (tier here T4, known T3)
- extractor_128x64 robust: level 7,570,911 vs O2_c 4,867,612 (tier here T3, known T2)
- parity64 ultra: level 44,911 vs ultra_bc64 44,478 (tier here T5, known T4)
- parity64 hardened: level 44,478 vs hardened_cfcx1_c 38,064 (tier here T4, known None)
- parity64 robust: level 9,153 vs O2 8,066 (tier here T3, known T2)


**Reading the cases.** Under the known-tier reading no level is dominated on any circuit.
The per-circuit cases are configurations whose tier on that circuit is above their known
tier, each excluded for a measured reason:
- `ultra_bc64` (one BOS copy per 64 features) is T5 on 8 circuits but misses the T5 point
  on the extractor, so it is a T4 recipe (kept by the T4 level on the adders, sha256_2r and
  parity64);
- `t5rzo` (one BOS copy per 256, step prologue) was measured on one input set only and
  missed T5 on xof_w4 and the extractor; on the sandbagger it has flat units over 465
  layers, where `ultra` failed T4 on this input set (its T5 there comes from the
  ultra-noise track's set 400, which did not contain the vulnerable inputs);
- cheap-unit hardened variants (`hardened_l2_cxf_c`, `_cfc_c`, `hardened_cfcx1_c`, ...)
  are T4 on the suite but flip bits of a 256-bit adder under T4 (pareto track), and
  `hardened_l2_fcxf_c` fails T4 on sha3_w2 and the sandbagger;
- `robust_fcx1_c`, `robust_x2(_c)`, `robust_fcf`, `O2`, `O2_c` and `O3_x` are T3 (or T2) on
  some circuits but fail T3 (T2) on others (xof_w4, the sandbagger, random circuits).
A level could only take these with per-circuit knowledge of tiers at compile time.

## A second fresh input set for the T4 and T5 levels

Input seed 3300 (487 inputs, built like set 3100), perturbation seeds 3300-3307 and 3400-3401, the same groups (`ladder_eval.py --levels ultra,hardened --seed 3300 --pseed 3300`).

| circuit | level | recipe kept | dense | measured tier (claim) | T3 / T4 / T5 failing of total |
|---|---|---|---|---|---|
| adder32 | ultra | ultra | 201,268 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| adder32 | hardened | ultra_bc64 | 198,410 | T5 (T4) | 0/8 / 0/36 / 0/44 |
| parity64 | ultra | ultra | 44,911 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| parity64 | hardened | ultra_bc64 | 44,478 | T5 (T4) | 0/8 / 0/36 / 0/44 |
| add4x16 | ultra | ultra | 193,197 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| add4x16 | hardened | ultra_bc64 | 191,383 | T5 (T4) | 0/8 / 0/36 / 0/44 |
| xof_w4 | ultra | ultra | 54,705,494 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| xof_w4 | hardened | hardened_xf_c | 27,483,832 | T4 (T4) | 0/8 / 0/36 / 1/1 |
| sandbagger_w1 | ultra | ultra_s | 46,695,250 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| sandbagger_w1 | hardened | hardened_xf_c | 20,095,847 | T4 (T4) | 0/8 / 0/36 / 1/1 |
| sha256_2r | ultra | ultra | 5,860,317 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| sha256_2r | hardened | ultra_bc64 | 5,767,756 | T5 (T4) | 0/8 / 0/36 / 0/44 |
| extractor_128x64 | ultra | ultra | 42,493,361 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| extractor_128x64 | hardened | hardened_xf_c | 12,833,317 | T4 (T4) | 0/8 / 0/36 / 1/1 |
| sha3_w2 | ultra | ultra | 38,234,420 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| sha3_w2 | hardened | hardened_xf_c | 18,898,888 | T4 (T4) | 0/8 / 0/36 / 1/1 |
| backdoor_w2 | ultra | ultra | 38,234,420 | T5 (T5) | 0/8 / 0/36 / 0/44 |
| backdoor_w2 | hardened | hardened_xf_c | 18,898,888 | T4 (T4) | 0/8 / 0/36 / 1/1 |

18 of 18 (circuit, level) pairs meet their claim on this set.


## Beyond T5

### With the standard input (BOS + bits): T6 is not reached

Both wave-7 robustness tracks traced every T6 failure to the first layer. That layer reads
the host's input, BOS = 1 and one noisy copy of each bit, and a LayerNorm shift is
measured against them: a shift near BOS removes the only reference (a shift above BOS
flips it; M 0.3 puts about 1 token in 1000 there). Every later layer reads always-0
features and its rows sum to 0, so a shift of any size cancels there. Presets that carry
T6's threats after the first layer (`noise_eval.CONFIGS`, measured by the ultra-noise
track and its verifier with wave 7's library; `rep` and `ballast` were removed by the
trim):

| preset | what | measured |
|---|---|---|
| `toward_t6` | steps only, repetition 2 (soft majority of 2 copies of every hidden feature), 16 always-0 features, a BOS copy per 16 features, q 512 | T5 on 8 circuits (xof_w4: float16 overflows in the first layer); T6 with the first layer at M 0.1 (T6f) on 8 of 9 (sha256_2r 7/8); T6 with M 0.1 everywhere (T6m) 8/8 on 8, 7/8 on add4x16 (verifier, set 900); the full T6 point 0-3 of 8 seeds; 1.79-2.14x wave 6's hardened |
| `toward_t7` | repetition 16, 32 always-0 features, q 2048 | T7's threats in every layer but the first, CPU f32 / bf16 only (bf16 split reductions on CUDA round its q 2048 pre-activations) |
| `ultra_s_b64` (`levels_eval`, `tiers_eval`) | ultra_s + ballast 1/64 (always-1 features that hold the norm) | T5 and T6's s side (s 0.01 with m 0.1, e 0.01) on all 9 circuits, three input sets; M 0.3 fails (first layer) |

### Extended input: T6, and T7 on CPU (changes the input interface)

The lead's construction: if the host appends always-0 features Z to the input, the first
layer's rows can sum to 0 too (e.g. w_x = 1, w_BOS = -1/2, w_Z = -1/2), and a LayerNorm
shift cancels in the first layer as in every other one. If the host also repeats every
bit (and BOS), the first layer reads their means, which divides n's noise there. Knobs
of wave 7's library (off in main and in every level; removed by the trim, so `ext_eval.py`
needs that library): `input_zeros=k` (the circuit reads
`[BOS, x, 0 * k]`), `input_rep=r` (`[BOS, x1 * r, x2 * r, ...]`), `input_bos=b` (b
copies of BOS after the bits). The circuit's outputs are unchanged; only the input format
grows. Driver: `ext_eval.py` (fresh set 3100, the host's perturbations applied to every
input feature, the zeros included).

| circuit | preset | input | dense | point | where | failing runs / runs | failing bases |
|---|---|---|---|---|---|---|---|
| adder32 | t6x | +16 zeros | 605,777 | T6 | CPU | 0 / 16 | - |
| adder32 | toward_t6 | standard | 601,601 | T6 | CPU | 16 / 16 | bf16, f32 |
| parity64 | t6x | +16 zeros | 120,312 | T6 | CPU | 0 / 16 | - |
| parity64 | toward_t6 | standard | 116,136 | T6 | CPU | 16 / 16 | bf16, f32 |
| sha256_2r | t6x | +16 zeros | 20,054,213 | T6 | CPU | 0 / 16 | - |
| add4x16 | t6x | +16 zeros | 601,803 | T6 | CPU | 0 / 16 | - |
| adder32 | t7x | +32 zeros | 4,314,903 | T7 | CPU | 8 / 8 | bf16, f32 |
| adder32 | t7x4 | +32 zeros, bits x4 | 4,365,015 | T7 | CPU | 8 / 8 | bf16, f32 |
| adder32 | t7x16 | +32 zeros, bits x16 | 4,565,463 | T7 | CPU | 8 / 8 | bf16, f32 |
| parity64 | t7x | +32 zeros | 751,655 | T7 | CPU | 8 / 8 | bf16, f32 |
| parity64 | t7x4 | +32 zeros, bits x4 | 801,767 | T7 | CPU | 8 / 8 | bf16, f32 |
| parity64 | t7x16 | +32 zeros, bits x16 | 1,002,215 | T7 | CPU | 8 / 8 | bf16, f32 |
| adder32 | t7x16b | +32 zeros, bits x16, BOS x16 | 4,569,378 | T7 | CPU | 0 / 8 | - |
| adder32 | t7x16b | +32 zeros, bits x16, BOS x16 | 4,569,378 | s0.002+m0.1+e0.01+M1+n0.1+nf0.01 | CPU | 0 / 4 | - |
| adder32 | t7x16b | +32 zeros, bits x16, BOS x16 | 4,569,378 | s0.002+m0.1+e0.01+M1+n0.1+sf0.05 | CPU | 0 / 4 | - |
| adder32 | t7x16b | +32 zeros, bits x16, BOS x16 | 4,569,378 | s0.01+m0.1+e0.01+M1+n0.1 | CPU | 0 / 4 | - |
| adder32 | t7x16b | +32 zeros, bits x16, BOS x16 | 4,569,378 | s0.002+m0.1+e0.01+M1+n0.03 | CPU | 0 / 4 | - |
| parity64 | t7x16b | +32 zeros, bits x16, BOS x16 | 1,006,130 | T7 | CPU | 0 / 8 | - |
| parity64 | t7x16b | +32 zeros, bits x16, BOS x16 | 1,006,130 | s0.002+m0.1+e0.01+M1+n0.1+nf0.01 | CPU | 0 / 4 | - |
| parity64 | t7x16b | +32 zeros, bits x16, BOS x16 | 1,006,130 | s0.002+m0.1+e0.01+M1+n0.1+sf0.05 | CPU | 0 / 4 | - |
| parity64 | t7x16b | +32 zeros, bits x16, BOS x16 | 1,006,130 | s0.01+m0.1+e0.01+M1+n0.1 | CPU | 0 / 4 | - |
| parity64 | t7x16b | +32 zeros, bits x16, BOS x16 | 1,006,130 | s0.002+m0.1+e0.01+M1+n0.03 | CPU | 0 / 4 | - |
| adder32 | t6x | +16 zeros | 605,777 | T3 | CUDA | 0 / 8 | - |
| adder32 | t6x | +16 zeros | 605,777 | T4 | CUDA | 0 / 24 | - |
| adder32 | t6x | +16 zeros | 605,777 | T5 | CUDA | 0 / 32 | - |
| adder32 | t6x | +16 zeros | 605,777 | T6 | CUDA | 1 / 32 | bf16+gpu+b1024 |
| parity64 | t6x | +16 zeros | 120,312 | T3 | CUDA | 0 / 8 | - |
| parity64 | t6x | +16 zeros | 120,312 | T4 | CUDA | 0 / 24 | - |
| parity64 | t6x | +16 zeros | 120,312 | T5 | CUDA | 0 / 32 | - |
| parity64 | t6x | +16 zeros | 120,312 | T6 | CUDA | 0 / 32 | - |
| add4x16 | t6x | +16 zeros | 601,803 | T3 | CUDA | 0 / 8 | - |
| add4x16 | t6x | +16 zeros | 601,803 | T4 | CUDA | 0 / 24 | - |
| add4x16 | t6x | +16 zeros | 601,803 | T5 | CUDA | 0 / 32 | - |
| add4x16 | t6x | +16 zeros | 601,803 | T6 | CUDA | 0 / 32 | - |
| sha256_2r | t6x | +16 zeros | 20,054,213 | T3 | CUDA | 0 / 8 | - |
| sha256_2r | t6x | +16 zeros | 20,054,213 | T4 | CUDA | 0 / 24 | - |
| sha256_2r | t6x | +16 zeros | 20,054,213 | T5 | CUDA | 0 / 32 | - |
| sha256_2r | t6x | +16 zeros | 20,054,213 | T6 | CUDA | 0 / 32 | - |
| extractor_128x64 | t6x | +16 zeros | 106,684,989 | T3 | CUDA | 0 / 8 | - |
| extractor_128x64 | t6x | +16 zeros | 106,684,989 | T4 | CUDA | 0 / 24 | - |
| extractor_128x64 | t6x | +16 zeros | 106,684,989 | T5 | CUDA | 0 / 32 | - |
| extractor_128x64 | t6x | +16 zeros | 106,684,989 | T6 | CUDA | 0 / 32 | - |
| adder32 | t7x16b | +32 zeros, bits x16, BOS x16 | 4,569,378 | T7 | CUDA | 7 / 12 | bf16+gpu+b256, bf16+gpu_exact+b256, f32+gpu_exact+b256 |
| parity64 | t7x16b | +32 zeros, bits x16, BOS x16 | 1,006,130 | T7 | CUDA | 4 / 12 | bf16+gpu+b256 |
| sha3_w2 | t6x | +16 zeros | 107,969,775 | T3 | CUDA | 0 / 8 | - |
| sha3_w2 | t6x | +16 zeros | 107,969,775 | T6 | CUDA | 0 / 8 | - |
| xof_w4 | t6x | +16 zeros | 158,713,177 | T3 | CUDA | 3 / 8 | fp16+gpu+b1, fp16+gpu+b1024, fp16+gpu+b256 |
| xof_w4 | t6x | +16 zeros | 158,713,177 | T6 | CUDA | 0 / 8 | - |

Measured (fresh set 3100; T3-T6 with the ladder's bases and 8 seeds; `t6x` =
`toward_t6` + 16 host zeros, `t7x16b` = `toward_t7` + 32 host zeros + every bit and BOS
given 16 times):
- **T6 holds with the extended input** on parity64, add4x16, sha256_2r and the extractor:
  T3 8/8, T4 24/24, T5 32/32 and the full T6 point 32/32 on f32 / bf16 gpu_exact and bf16
  CUDA at b256 / b1024, and 16/16 on CPU f32 / bf16 (CPU for adder32, parity64, add4x16 and
  sha256_2r). adder32 misses boolify once (a 1 at 0.84) in 32 T6 runs at bf16+gpu b1024.
  Without the zeros the same preset fails the T6 point in 16 of 16 CPU runs on adder32 and
  parity64: the zeros are what moves T6. Cost: 16 input features (+0.7% dense over
  `toward_t6` on adder32, +3.6% on parity64); `t6x` is 2.5-3.4x the T5 level's size on
  these circuits. Spot checks of the deep / wide ones: sha3_w2 T3 8/8 and the T6 point
  8/8 (bf16+gpu b256, f32 gpu_exact, 4 seeds); xof_w4 the T6 point 8/8 but float16
  overflows in its first layer (q 512, as for `toward_t6`), so T2 there.
- **T7 holds on CPU** (f32 and bf16) on adder32 and parity64 with every bit and BOS given
  16 times (`t7x16b`: 0 of 8 runs fail; with 32 zeros alone, or with the bits repeated but
  one BOS, 8 of 8 fail). On CUDA, parity64 holds T7 at f32 / bf16 gpu_exact (8/8) but not
  under bf16 split reductions (4/4 fail), and adder32 fails on every CUDA base: toward_t7's
  q 2048 pre-activations are not exact there.

Size: `t6x` is 2.5-3.4x the T5 level and `t7x16b` about 22x (adder32, parity64). These
presets change the input format, so they are not Pareto-ranked against the
standard-input ladder.

## Out of the suite

Adders with the wave-6 verifiers' carry-chain inputs (932 per circuit), through the T4 / T5 recipes and levels: T3 bases, T4 (16 seeds) and T5 (8 seeds) points; CPU: f32 and bf16; CUDA: bf16 / fp16 with reduced-precision reductions and f32 / bf16 gpu_exact (`scripts/oos_*.sh` in the merge's work directory).

| circuit | config | recipe kept | dense | where | runs | failing | worst error |
|---|---|---|---|---|---|---|---|
| add256 | recipe:ultra_bc64 | ultra_bc64 | 18,979,222 | CPU | 50 | 0 | 0.029 |
| add256 | recipe:ultra | ultra | 19,307,014 | CPU | 50 | 0 | 0.028 |
| add512 | recipe:ultra_bc64 | ultra_bc64 | 85,190,156 | CPU | 50 | 0 | 0.029 |
| add512 | recipe:ultra | ultra | 86,697,100 | CPU | 50 | 0 | 0.029 |
| add256 | hardened | ultra_bc64 | 18,979,222 | CUDA | 51 | 0 | 0.048 |
| add512 | hardened | ultra_bc64 | 85,190,156 | CUDA | 51 | 0 | 0.043 |
| add256 | ultra | ultra | 19,307,014 | CUDA | 51 | 0 | 0.044 |
| add512 | ultra | ultra | 86,697,100 | CUDA | 51 | 0 | 0.046 |

Earlier out-of-suite results apply to the recipes they measured (a level on a new circuit
may keep a different recipe: on an AND / OR of 2048 inputs every T3+ level now keeps
`ultra_s`, the only candidate whose float16 bound fits, while at 1000-1200 inputs robust
keeps `robust_fcf_c` and hardened `ultra_bc64`): wave 6's robust
recipe on AND / OR / equality trees of up to 2048 inputs (bf16 at b1-b16384) and adders
of up to 256 bits (add512 flips 3 bits at bf16+gpu b256); the pareto track's recipes on
the wave-6 verifier's wide circuits at b1-b4096 and on random circuits of library ops
(CPU); the ultra-noise track's T5 recipe on 10 out-of-suite circuits (add512, and1024, or1024,
eqconst1024, eqpair256, multi256, xor256, mul8 on 65,536 inputs, cmp_sel16, sha256_4r: 100
of 100 runs correct); ultra_s on AND / OR / multi trees of 2048 inputs in float16 (T3, T4;
wave 6's hardened overflows there). O3 on plain parities of 33-1024 inputs in CPU float32:
worst error at most 0.0062 (xor256; xor trees of 32; `scripts/o3_xor.py`); before the merge xor128 reached
0.026 and xor512 flipped bits.


## What generalizes (size reductions measured on all 10 suite circuits and out of the suite)

Every reduction below is a general graph pass or numeric option, not a circuit-specific
builder; each was measured on the 9 evaluable suite circuits (fresh inputs) and on the
verifiers' out-of-suite circuits (random circuits of library ops on CPU; wide AND / OR /
equality / xor trees of up to 2048 inputs and adders of up to 512 bits on CUDA).

- **Leveling the gate graph instead of the call tree** (`reifier.opt.graph`): dead code
  is never compiled (`sha256_r2`: 11.4G -> 7.4-8.9M dense at every level, 1,290-1,550x).
- **Bounded fan-in** (xor trees of <= 4 inputs, AND/OR trees of 2, the Kogge-Stone prefix
  adder) keeps every row's sums small, so 16-bit floats stay exact under split reductions
  and float16 stays in range. It is what made wave 6's robust 2.6-8.7x smaller than main
  on 8 of 9 circuits (the T3 level is now 4.9-17.5x).
- **Flat one-unit gates** (AND/OR/NOT/copy of <= 4 inputs as max(0, 2z - 1)(3 - 2z)) take
  18-41% off robust, and, with BOS = 1, evenly spread always-0 features and step outputs,
  15-38% off the T5 construction (ultra vs wave 6's hardened).
- **Step copies of unit outputs** (`clean_outputs`, wave 7): the T3 / T4 failures of
  unit-based recipes were boolify misses at the outputs (1s off by 0.03-0.06); a last
  level of step copies, which exact readout makes exactly BOS, removes them. With it,
  E-exact xor trees re-cleaned after every xor layer (`robust_x1_c`) and flat cones
  re-cleaned after every cone layer (`robust_fcx1_cc`) are T3, which makes robust 3.0-5.4x
  smaller on the xor-heavy circuits (xof_w4 52.0M -> 15.2M, the extractor 40.8M -> 7.6M).
- **Flat xor trees** (sums of vertex indicators, `flat_xor`) under wave 6's hardened
  numerics with clean outputs (`hardened_xf_c`): T4 at 2.4-3.9x smaller than wave 6's
  hardened on the xor-heavy circuits (xof_w4 74.8M -> 27.5M, the extractor 49.9M ->
  12.8M).
- **BOS = 1 and noise-aware numerics** (wave 7, ultra): with a heavy BOS every feature
  carried the BOS's noise (n is relative to the largest feature); copies of BOS keep the
  normalized scale and float16 range instead; centered steps (margins 3/8 on both sides);
  rows spread evenly over the always-0 features; per-layer output scales that keep a
  stream cut to s^2 above the next RMSNorm's epsilon. None costs size; together with the
  flat gates they make T5 cheaper than wave 6's T4.
- **O3's glu_xor sums** give the smallest float32 circuits (extractor 40.8x smaller than
  main at depth 2); float32 only, and only up to 32 inputs per sum (wider ones are trees
  since the merge: one sum of 64+ inputs was 56x at depth 1 on the extractor but not T1 on
  wider xors).
- **Not general** (measured): cheap units under wave 6's hardened numerics (a 256-bit
  adder's carry chains flip bits under T4); flat cones together with flat xor trees under
  T4 (sha3_w2, the sandbagger); E-exact xor units under hardened's numerics (T2-T3 on
  xof_w4, sha256_2r, the extractor); 2 always-0 features instead of 8 (the extractor
  under T4); one BOS copy per 64 features at T5 (the extractor); flat units over hundreds
  of layers at T4 (the 465-layer sandbagger; hence `MAX_DEPTH`).
- **Hand-built XOF points** stay smaller on xof_w4 at T1-T3 (`experiments/xof_levels` (not on this branch),
  `xof_shrink`: T1 2.8M vs O3 6.3M, T2 4.5M vs O2 8.5M, T3 13.1M vs robust 15.2M, the
  last with an input BOS of 4): they use Keccak's structure (theta's shared column
  parities, chi as a 3-bit cone, packed digests), which no general pass recovers.

## Limits and open issues

- **Per-circuit tiers vs known tiers.** A level selects by each recipe's *known* tier
  (its minimum over the suite, capped by failures measured out of the suite), because a
  compiler cannot know a new circuit's tier. Some configurations reach a higher tier on
  one circuit than their known tier and are smaller there (the starred cases above, e.g.
  O2's recipe is T3 on parity64; cheap-unit hardened variants are T4 on the suite but
  flip bits of a 256-bit adder). The levels do not take them.
- **Depth envelope of flat units at T4 / T5.** `ultra` and `ultra_bc64` are candidates only
  up to 200 layers: on the 465-layer sandbagger flat units failed T4 in about 3% of runs
  (two dense inputs, about 1% of their perturbed trials), while every circuit of up to
  169 layers held. The threshold is measured on these two depths only; a circuit of fewer
  layers with very long chains of flat copies is not covered by a measurement. Stressed
  the same way (the two vulnerable inputs tiled into 5,120 perturbed trials at bf16+gpu
  b256), the recipes the levels keep on the sandbagger give: `ultra_s` 0 at the T4 and the
  T5 point, `hardened_xf_c` (T4 level; flat xor units in the Keccak rounds, steps
  elsewhere) 1 at the T4 point, `ultra` 42 at the T4 point and 0 at the T5 point. The T4
  level keeps `hardened_xf_c` there because it passes the T4 protocol (0 of 36 runs here,
  64 of 64 in the pareto verifier's runs, three earlier input sets); its rate on these
  inputs is about 40x below ultra's.
- **robust on rare inputs.** On 262,144 dense extractor inputs the robust level
  (`robust_x1_c`, E-exact xor units) misread a 0 as 0.71 on 2 inputs at bf16+gpu b256
  (deterministic, pareto verifier). A 512-bit add flips 3 bits at bf16+gpu b256 (use
  hardened or ultra there; both were exact on add256 / add512, below).
- **O3** is float32 only; its one-layer symmetric sums other than xor (e.g. counters)
  are not bounded in width.
- **float16 range.** The first layer reads BOS = 1 among n_in inputs; `fp16_bound(mlp)` bounds
  every layer's pre-activations and hidden products. Recipes stay in range up to about
  1000 (wave 6's hardened, q 256) to 2000 inputs (ultra, q 128); `ultra_s` (fit16) at any
  width, and the T5 level prefers it when ultra would overflow. A host that scales the circuit *up* (s > 1) can still overflow
  fit16's float16 layers (outside the threat model).
- **T6 and T7 with the standard input** are blocked in the first layer (above); the
  extended-input presets reach them by changing the input format.
- **Margins just beyond T5** are thin: sha256_2r fails n 0.015-0.02; `ultra_bc64` fails n
  0.02 more often than `ultra`.
- **Selection cost**: a level compiles every candidate's gate graph (up to 11 for O3) and
  builds MLPs while their lower bound can still win: up to about a minute on the
  sandbagger, 5-10x a single recipe.
- **Generic gates**: the fan-in bounds cover xor, and_, or_, parity and add. A `gate()` whose
  sum can exceed its threshold by more than about 2 rounds in 16-bit floats at every level
  (a 65-input majority misses boolify under robust and wave 6's hardened).

## Code

The library keeps what the five levels and main's default compile use: the wave-8 trim
removed every knob, preset and pass option that no level reaches. The measurements above
were made before it; every level, every recipe and the default compile give the same
weights after it (state_dict hashes on the 10 suite circuits, float32 / bfloat16 /
float16 builds), with one intended change: the second BOS of wide rows now fires in
float16 builds only beyond float16's 11 significant bits (bfloat16 and float32 builds:
8), so `Compiler(mlp_dtype=t.float16)` equals main's again on the suite (parity64 and
the extractor had a second BOS). `hardened` is no longer a candidate (it never won), and
the float16 range warning is for level builds only (`Compiler()` never warns, as in main).

The levels are the optional package `reifier.opt` on main; the core's `Compiler` stays
main's. The measurements above were made with the levels' first version inside the core
(waves 6-8, not published), with which `reifier.opt` gives identical weights:
- `reifier/opt/compiler.py`: `Compiler(level, mlp_dtype, select=True, knobs={})`,
  `Compiler.chosen`, the selection and the float16 warning.
- `reifier/opt/recipes.py`: `RECIPES`, `TIERS`, `MAX_DEPTH`, `candidates`, `recipe`. The
  knobs are `fanin_*` (`fanin.Fanin`), `passes` (`graph.GraphOptions`) and the SwiGLU
  construction's (`build.MLPOptions`).
- `reifier/opt/fanin.py`, `graph.py`, `build.py`: the fan-in bounds and the prefix adder,
  the graph compiler and its passes, and the levels' own SwiGLU construction with
  `fp16_bound` and `dense_bound`.
- Tests: `tests/opt/`.
- Drivers in `experiments/levels/`: `ladder_eval.py` (the final run), `robust_eval.py` (the
  threat model; `compiler()` maps a driver's configuration to a compiler), `suite.py`,
  `wide_circuits.py`, `summarize.py`.

Configurations of `levels_eval.py`, `noise_eval.py`, `tiers_eval.py` and `ext_eval.py`
that use removed knobs need wave 7's library (the O1 and R presets, `fold_not`,
`schedule`, `min_gain`, `flat_edges`, `zero_rails`, `norm_scale`, `rep`, `quad_prologue`,
`ballast`, `input_zeros` / `input_rep` / `input_bos`, a fixed count of `bos_copies`, a
fixed `out_scale`). `robust_tree` and `hardened_tree` (`passes=False`) need the first
version.

## Reproduce

```bash
R=<repo>; export PYTHONPATH=$R/src:$R/experiments/levels OMP_NUM_THREADS=2
python -m pytest -q $R/tests
/usr/bin/python3 $R/experiments/levels/ladder_eval.py --circuits adder32,parity64 \
    --refs /tmp/refs --out /tmp/ladder.jsonl           # every level, fresh set 3100
```

## Appendix: wave 6

### Fixes after verification (wave 6)

Three verifiers re-checked this ladder: fresh input sets (seeds 600 / 700 with one-hot,
one-cold, prefix and carry-chain inputs, and the backdoor / sandbagger triggers), batches
b1-b16384, out-of-suite circuits (wide AND / OR / xor gates, equality checks, adders of
64-1024 bits, mul8, cmp_sel16, more SHA-256 and Keccak rounds) and 300 random circuits of
library ops. The robust and hardened suite claims held; these did not:

| finding | action | after the fix (measured) |
|---|---|---|
| hardened: AND-8 trees of 256-2048 inputs read 0s as 1 (bf16 split reductions, b32-b4096); T4 point failed on 256 / 512-bit equality checks (5 / 2 of 24 runs) | **fixed**: AND/OR trees of 2 | and256-2048, or512, multi256, eqconst256-2048, eqpair128/256: 0 wrong in bf16 at b1-b16384 (fp16 fails from 1024 inputs, with the warning); T4 point 0 of 336 runs (6 circuits x 8 seeds x bf16/fp16 GPU bases and CPU) |
| robust: AND-8 trees of 1024 inputs lost 1s at >= 3000 rows; x == 2048-bit constant missed its match at b256 / b1024 | **fixed**: AND/OR trees of 2 | and256-2048, or512/1024, eqconst256-2048, eqpair, multi256: 0 wrong in bf16 at b1-b16384 (fp16 of and2048 fails with the warning) |
| robust: a 512-bit add gets single long carry chains wrong (3 bits at bf16+gpu b256) | **downgraded**: robust's T3 covers adders of up to 256 bits | add256 exact at b32-b4096; add512 still 3 wrong bits at b256 (passes={"cheap": False} fixes it, but changes every suite circuit) |
| robust: e 0.01 envelope missed boolify on xof_w4 in 5 of 42 seeds (by <= 0.0013) | **downgraded**: robust's e envelope is 0.01 on 8 circuits, 0.005 on xof_w4 | xof_w4 e0.005: 0 of 42 seeds fail (margin <= 0.0173) |
| O1 below main on the sandbagger at b24 / b32, and on 11 of 300 random circuits with parity() (also at gpu_exact) | **fixed**: main's tree layout unless the graph compile is >= 10x smaller | sandbagger b16/b24/b32/b256 exact; 300 random circuits: no O1 flag; O1 T3 on 7 of 9 suite circuits on both sets (T1 on the extractor and parity64, like main) |
| O1 / main T3 on sha3_w2 / backdoor_w2 only at b1/b256/b1024 | **documented** (main's own limit) | 8 bits of 1 of 8192 inputs at b24-b64 for main's bf16 build and O1 alike |
| O2 not T2 on 60 of 300 random circuits and on small add compositions | **fixed**: O2 uses main's adder | 300 random circuits: no O2 flag; the 3 small add compositions exact; adder32 / add4x16 T3, sha256_2r T2 (was T3), 2.2-7.3x dense instead of 4.5-9.0x |
| O2 described as "bf16 exact" | **wording**: T2 = correct by nearest and boolify; steps that exceed their thresholds by 2+ round (up to 0.0075 on 1-bits) | |
| float16 guard bounded only the first layer's pre-activations | **fixed**: `fp16_bound()` covers every layer and the hidden products silu(g) v | warns on every case the verifiers found (or/xor of 2100 inputs, AND-1100 behind the copy layer, hardened threshold-70-of-80, 3160 hidden features, robust or2048 / add1024); conservative for units whose value depends on the inputs (O2 warns at 2000 inputs, where float16 was still exact) |
| the float16 warning fired on bf16 builds | **fixed**: float16 builds, and float32 builds of the robust levels (wave 8: of the levels only) | |
| `Compiler().c` / `.q` were None (broke `xof_shrink/bf16x/e2e_compiler.py`) | **fixed**: the fields hold the resolved knobs | `Compiler().q == 8`; e2e_compiler runs |
| `settings` returned the level's pass dict by reference; nested `fanin()` reset outer bounds; passes=None could not be asked for under a level | **fixed**: copies; `fanin` arguments left at None keep the enclosing setting; `passes=False` is the tree compiler | tests |
| the wide-row fix fired for c or q not powers of 2 (an extra layer, no gain) | **fixed**: only for powers of 2 | q 12 / c 3 / c 5 q 6: main's 6 layers |
| the wide-row fix splits BOS weights only; wide xors still round on general inputs; a single wide first-layer row costs a layer of input copies | **documented** (Limits) | |


### Wave 6: where the hand-built XOF points sit (xof_w4)

| tier | hand-built XOF builder (`experiments/xof_levels` (not on this branch), `experiments/xof_shrink`) | depth | x dense | general level | depth | x dense |
|---|---|---|---|---|---|---|
| T1 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` (wave 5) | 8 | 75.0 | `O3` | 6 | 32.8 |
| T2 | `bf16x.bx:pk1q`, E-rule units (re-run here: bf16 and gpu_exact exact, bf16+gpu b256 17,762 wrong) | 9 | 46.1 | `O2` | 9 | 24.3 |
| T3 | `rb:sep_xp_rqff`, exact steps + flat edge units + 2 flat re-quantization layers, input BOS 4 (re-run here on 725 fresh messages: T3 at b1/b256/b1024) | 11 | 15.8 | `robust` | 21 | 4.0 |
| T4 min | `rb:sep_xp_rqff` + q 128, zero-sum rows: passes s/m/e on CPU bases (wave 5), but the T4 point on the bf16+gpu b256 base flips 5-32 bits (re-run here, seeds 0-1) | 11 | 15.7 | `hardened` (T4 on the GPU bases) | 22 | 2.8 |

The hand-built points are 1.9-4x smaller than the general levels at T1-T3: they use Keccak's
structure (theta's column parities shared by 11 readers, chi as a 3-bit cone, packed
digests) that no general pass recovers. At T4 the comparison is not like for like: the
hand-built point was only T4 on CPU bases. The builders stay in `experiments/`.

# Wave 2, avenue "packing": narrower interfaces between layers

**Setup**
- Configuration: log_w=6, 3 XOF steps, default steepness (c=4, q=8).
- Architecture: plain untied `MLP_SwiGLU`: no residuals, no weight tying, no embeddings.
- The compiler (`repo/src`) is unchanged. All the work is in two new builders:
  - `repo/experiments/xof_shrink/xp3.py`: xs3 plus options, for depths 6-8;
  - `repo/experiments/xof_shrink/xp4.py`: xs4 plus options, for depth 5.

**Verification**
- Every number below comes from `xofbench.py` plus `audit/adv_check.py` on both reference
  sets:
  - `ref777_w6.pt`, 69 messages;
  - `combined-1/adv/ref_w6.pt`, 63 messages.
- Every listed point has `ok: true`, `wrong_bits: 0` and no eager mismatch on both sets.
  The raw lines are in `results.jsonl`, rebuilt from `runs/` by `collect.py`.
- The rebuilt baselines (`xp3:*_base`, `xp4:d5_m3k2_mp_base`) reproduce the old frontier
  exactly.
- For 15 variants, including the dense-best point of each depth, `validate_bench.py`
  gives harness weights bit-equal to `Compiler().get_mlp_from_tree` at log_w 0-2, with
  verify ok (`runs/val_*.txt`).

## Result: the audited Pareto front, depths 5-8

"Old" is the brief's frontier. "×base" is the gain against the baseline
(3,305,445,348 / 2,034,788).

| depth | variant | dense | sparse | vs old (dense / sparse) | ×base (dense / sparse) | audit margins (69 / 63) |
|---|---|---|---|---|---|---|
| 5 | `xp4:d5_m4k2_mp_pc` | **81,486,372** | 234,811 | −8.7% / +2.1% | 40.6 / 8.7 | 1.5e-3 / 1.7e-3 |
| 5 | `xp4:d5_m3k2_mp_pc` | 81,555,805 | 232,333 | −8.6% / +1.1% | 40.5 / 8.8 | 8.5e-4 / 1.7e-3 |
| 5 | `xp4:d5_m3k2_mp_pa` | 82,837,245 | 231,431 | −7.2% / +0.7% | 39.9 / 8.8 | 5.2e-4 / 1.1e-3 |
| 6 | `xp3:lazy4c_middle_m4_mp_pkc54` | **59,852,598** | 189,035 | −13.7% / +6.4% | 55.2 / 10.8 | 1.5e-3 / 1.3e-3 |
| 6 | `xp3:lazy4c_middle_m3_mp_pkc5` | 60,200,529 | 188,873 | −13.2% / +6.3% | 54.9 / 10.8 | 1.2e-3 / 1.3e-3 |
| 6 | `xp3:lazy4c_middle_m4_mp_pc54` | 60,639,235 | 184,393 | −12.6% / +3.8% | 54.5 / 11.0 | 1.5e-3 / 1.3e-3 |
| 6 | `xp3:lazy4c_middle_m4_mp_pc4` | 61,354,467 | 180,249 | −11.6% / +1.5% | 53.9 / 11.3 | 1.6e-3 / 1.4e-3 |
| 6 | `xp3:lazy4c_middle_m3_mp_pc` | 61,679,613 | 180,087 | −11.1% / +1.4% | 53.6 / 11.3 | 8.1e-4 / 9.4e-4 |
| 6 | `xp3:lazy4c_middle_m3_mp_pa` | 62,961,053 | 179,185 | −9.2% / +0.9% | 52.5 / 11.4 | 3.5e-4 / 1.1e-3 |
| 7 | `xp3:split_first_lazy4c_m4_mp_pkg54` | **51,869,373** | 137,338 | −18.5% / +7.7% | 63.7 / 14.8 | 3.4e-3 / 3.8e-3 |
| 7 | `xp3:split_first_lazy4c_m3_mp_pkg5` | 52,217,304 | 137,176 | −18.0% / +7.6% | 63.3 / 14.8 | 3.4e-3 / 3.8e-3 |
| 7 | `xp3:split_first_lazy4c_m4_mp_pg54` | 52,656,010 | 132,696 | −17.3% / +4.1% | 62.8 / 15.3 | 3.4e-3 / 3.8e-3 |
| 7 | `xp3:split_first_lazy4c_m4_mp_pg4` | 53,371,242 | 128,552 | −16.2% / +0.8% | 61.9 / 15.8 | 2.8e-3 / 2.9e-3 |
| 7 | `xp3:split_first_lazy4c_m3_mp_pg` | 53,696,388 | 128,390 | −15.6% / +0.7% | 61.6 / 15.8 | 1.1e-3 / 1.2e-3 |
| 7 | `xp3:split_first_lazy4c_m3_mp_pag` | 54,977,828 | 127,488 | −13.6% / +0.0% | 60.1 / 16.0 | 1.0e-3 / 2.4e-3 |
| 7 | `xp3:split_first_lazy4c_m3_mp_dg` | 61,385,028 | **125,926** | −3.6% / −1.2% | 53.8 / 16.2 | 3.7e-4 / 6.5e-4 |
| 8 | `xp3:split_first_middle_m4m2_mp_pkg4` | **49,708,089** | 112,487 | −18.6% / +3.1% | 66.5 / 18.1 | 2.9e-3 / 2.9e-3 |
| 8 | `xp3:split_first_middle_m4m2_mp_pg4` | 50,069,254 | 111,365 | −18.0% / +2.1% | 66.0 / 18.3 | 2.9e-3 / 2.9e-3 |
| 8 | `xp3:split_first_middle_m3m2_mp_pg` | 50,582,294 | 111,279 | −17.2% / +2.0% | 65.3 / 18.3 | 1.4e-3 / 1.2e-3 |
| 8 | `xp3:split_first_middle_mp_pcg` | 51,129,894 | 110,021 | −16.3% / +0.8% | 64.6 / 18.5 | 1.4e-3 / 9.9e-4 |
| 8 | `xp3:split_first_middle_mp_pag` | 52,411,014 | 109,045 | −14.2% / −0.1% | 63.1 / 18.7 | 1.6e-3 / 1.0e-3 |
| 8 | `xp3:split_first_middle_mp_dg` | 58,816,614 | **107,557** | −3.7% / −1.4% | 56.2 / 18.9 | 4.9e-4 / 4.1e-4 |

**Which points dominate the old frontier outright:**
- `_dg` (depths 7 and 8) is better on both dense and sparse.
- `split_first_middle_mp_pag` is −14% dense at −0.1% sparse.
- `split_first_lazy4c_m3_mp_pag` is −13.6% dense at +18 nonzeros.

**Depth 8** is now below wave 1's estimated depth-8 floor of about 50M.

**Depth 4** (`xc:d4a_m4k3_mp`, 160,358,065 / 425,212) is unchanged: it has no interface
this avenue can narrow (see below).

**Worst audit margin** of the listed points: 3.8e-3. The tolerance is 0.02.

**Excluded, although it passes.** `xp3:lazy4c_middle_m6_mp_pkc5` (59,805,258 / 191,685)
passes both audits, but with margins of 0.0093 and 0.010. The same construction (`s6`)
fails at depths 7 and 8 with margins of 0.026 and 0.030. I do not count it: see "What
failed".

## Where it comes from (dense per layer, [in, hidden, out])

| layer | depth 8 old (`split_first_middle_mp`) | depth 8 new (`m4m2_mp_pkg4`) | depth 6 old (`lazy4c_middle_m3_mp`) | depth 6 new (`m4_mp_pkc54`) |
|---|---|---|---|---|
| 1 | [1145,2163,1465] 8.12M | [1145,2163,1145] 7.43M | [1145,5571,1465] 20.92M | same |
| 2 | [1465,1466,1601] 6.64M | [1145,1466,1465] 5.51M | [1465,1602,1921] 7.77M | [1465,1602,961] 6.23M |
| 3 | [1601,1602,1921] 8.21M | [1465,1602,961] 6.23M | [1921,3203,1676] 17.68M | [961,3203,1657] 11.46M |
| 4 | [1921,3202,1713] 17.79M | [961,3202,1657] 11.46M | [1676,2861,620] 11.37M | [1657,2730,489] 10.38M |
| 5 | [1713,1714,1713] 8.81M | [1657,1658,1657] 8.24M | [620,5645,508] 9.87M | [489,6859,449] 9.79M |
| 6 | [1713,1714,545] 6.81M | [1657,1658,489] 6.31M | [508,1045,673] 1.77M | [449,675,673] 1.06M |
| 7 | [545,2146,545] 3.51M | [489,2427,449] 3.46M | | |
| 8 | [545,675,673] 1.19M | [449,675,673] 1.06M | | |

**Depth-7 new (`pkg54`):** [1145,2163,1145] [1145,1466,1465] [1465,1602,961]
[961,3203,1657] [1657,2730,489] [489,6859,449] [449,675,673].

**Depth-5 new (`d5_m4k2_mp_pc`):** [1145,5571,1465] [1465,1602,961] [961,3203,1657]
[1657,5338,489] [489,13667,673].

**Ablation at depth 8**, from `split_first_middle_mp` (61,082,590 / 109,101; dense /
sparse change):

| options | dense | sparse |
|---|---|---|
| `pa` | −6.41M | +1,488 |
| `dy` | −0.64M | −272 |
| `dy` + `pd1` | −1.45M | −399 |
| `dy` + `g1` | −2.27M | −1,544 |
| `pa` + `dy` + `g1` | −8.67M | −56 |
| + `pc` | −9.95M | +920 |
| + digest 1 at m3 | −10.50M | +2,178 |
| digest 1 at m4 + `s4` instead | −11.01M | +2,264 |
| + `lf` | −11.37M | +3,386 |

**At depth 6**, from `lazy4c_middle_m3_mp`: `pa` −6.41M, `pc` −1.28M, `lf` −0.74M,
`d2p5` −0.74M, `m4` + `s4` −0.35M.

## Method, and why each step is exact

**Why packing only works at copies.** Every interface of the frontier designs is full
rank for the units that read it (`packing/ranks.py`: the rank of `[wg; wv]` over a
layer's input equals the input width, at every layer of depths 6 and 8). So no feature
can simply be dropped. The width only shrinks in two cases:
- a reading unit can take two bits from one feature at no extra units;
- a feature the next layer needs anyway can carry more.

**The one full-width copy layer.** The X layer (E = a + D) copies all 1600 state bits of
the chi layer before it. Its interface costs about 2 × hidden(X) × width, with
hidden(X) = 3202, so that is where the big saving is.

1. **`pa`: pairs of copied bits.**
   - The chi layer emits f = a0 + 2 a1 (as f/4). Its wo is linear, so this is free.
   - The X layer spends the 2 units that 2 copies would cost: `relu(f) · 1 = f` and
     `relu(f − 1)(4 − f)/2 = a1` (0, 0, 1, 1). Then `a0 = f − 2 a1`.
   - Both knots are lattice points (silu(0) = 0), the slopes at lattice points are at
     most 1, and the result is exact on the 4 points.
   - Pairs are the limit for pure copies. A unit reading one feature is 0 on one side of
     its knot and a quadratic on the other. With 3 or more bits in a feature, every
     non-copy additive function is constant on 3 or more active points, and a quadratic
     equals a constant at only 2 points. By the same argument, 5 bits cannot travel in
     2 features with 5 units; `packing/col2.py` confirms this exhaustively for small
     weights.
2. **`pc`: the column sum as a carrier.**
   - The X layer's D units need the column-pair count P = C(x−1, z) + C(x+1, z+1), where
     C is a column sum. Before, the 320 counts P were separate features.
   - Now the chi layer emits, per column, C (as C/8) and the pairs (y0, y1) and (y2, y3).
   - The 5th bit is linear in X's units: a4 = C − (f1 − a1) − (f2 − a3), with one copy
     unit of C in place of the copy of a4.
   - The D units read C(x−1, z) + C(x+1, z+1) as a 2-feature gate, with the same range 10
     and the same 5 units.
   - So a column costs 3 features instead of 3.5.
3. **`g1`: G = a + 2D in the first X layer** (depths 7 and 8).
   - G ∈ {0, 1, 2, 3} gives two things with one unit each:
     - theta = a ⊕ D = `relu(G)(3 − G)/2`;
     - D = `relu(G − 1)(4 − G)/2`.
   - The capacity lanes' theta1 bits are D. They are read from any message position's G
     in the same column (every column has message bits in rows 0-2), so the 320 D-only
     features vanish.
   - The first layer's output width equals its input width: 1145.
4. **`dy`: de-duplicated Y outputs** (depths 7 and 8). Only 1465 of the first Y layer's
   1600 theta bits are distinct.
5. **`pd1`: paired D-only values.** This is the `pa` decoder on the capacity-lane Ds, and
   `g1` supersedes it.
6. **`lf`: L-form last theta.**
   - A chi unit reads the single form L = 2a − b + c: `chi = relu(L)(3 − L)/2`, with iota
     folded into a.
   - The last theta layer emits the 224 values L/4 instead of the 320 theta bits. Its
     parity units are shared by CSE.
   - It trades +1.1K to +4.6K nonzeros for −0.4M to −0.7M dense.
7. **`d2p5`: lazy digest values two per feature** (depths 6 and 7).
   - The lazy chi layer's digest values o ∈ [0, 4] travel as F = o0 + 5 o1 (as F/32),
     and the two share one pass unit.
   - The next, 6.5K-unit layer maps F to the exact pair b0 + 2 b1 with the DP-minimal
     integer-knot decoder of that 25-point function (12 units, `decode.check`).
8. **`m2` and `s4`: digest routing.**
   - Digest 1 crosses 3 layers at about 26-38K parameters per feature. It now travels
     4 bits per feature (m1=4).
   - The last theta layer splits each feature p into two pairs:
     - hi = floor(p/4), from a 6-unit DP staircase;
     - lo = p − 4 hi, which uses the copy unit.
   - The last layer then decodes pairs at 2 units each (about 8 per 3 bits for m3).
   - Digest 2, which crosses one layer, stays in pairs (`m2`).
   - The feature scale is 16, and the margins stay below 3e-3.

**Why the whole construction is exact.** Every packed value is an integer combination of
exact bits (or lazy values). Every decoder is a sum of gated units that is exact on the
lattice, with knots on lattice points. The eager function of every variant equals the
reference xof (the eager check of the audit).

## What failed, and why

- **`s6`: 6 bits per feature, split 3+3 in the last theta layer.**
  - Dense −0.3M to −0.4M, but the feature scale is 64. Float32 errors of the carried
    feature, times 64 and the staircase slopes, reach margins of 0.030 (depth 8) and
    0.026 (depth 7): audit fails. At depth 6 the margin is 0.010, which passes but is
    excluded.
  - The precision limit for packed digests at this size is therefore about 4 bits per
    feature (scale 16).
- **Chi reading packed theta.**
  - No row of 5 chi bits comes from 2 pairs plus a single with 5 units. This holds for
    every pairing and for weight ratios 2, 3 and −2, with gates |w| ≤ 3
    (`packing/chi_exact5.py` finds only the constant direction).
  - About 2 units per chi bit would be needed, and the break-even is 1.35.
- **Y ([E==1]) reading packed E pairs.**
  - For any linear packing G = αE0 + βE1, [E1 == 1] equals 1 at 3 points, which one unit
    cannot produce. So it needs 2 or more units per E, and Y's input stays at 1600.
  - X passing a-pairs plus D to Y fails the same way: 4 units per pair in Y, net +0.3M.
- **Digest 1 inside G = a + 2D of the second X layer:** 2 units per bit in Y2 (2.2M)
  against 0.87M for the packed carry.
- **Digest m4 decoded directly in the last layer** (no split): the decoders grow from
  about 8 to 20 units per group (m4m2 52.44M against m3m2 52.28M, before `pc`). `s4` fixes
  this by splitting one layer earlier.
- **Lazy digests as o' ∈ {0,1,2} in base-3 pairs:** the decoder is only 4 units, but o'
  costs 2 more units per bit in the lazy layer (+0.15M net). Base-4 pairs of [Ea==1] + Q'
  (7-unit decoder) break even with `d2p5`.
- **Depth 4:**
  - It has no copy layer. Layer 3 reads 1600 counts as singles and row pairs (rank 1600).
  - Its digests enter the last layer directly (no split layer), and m4k3 is already the
    optimum.
  - Counts cannot carry extra bits: an offset of 2^k either breaks the glu_xor units or
    doubles them.
- **Depth 5:** the digests also enter the last (Walsh) layer directly, so `s4` and
  `d2p5` do not apply.

## Headroom and lower-bound reasoning

**Dense.** For these layouts and unit types, every interface is now at its bound:
- the rank bound: `ranks.py`, full rank everywhere;
- the pair bound: 2 copied bits per feature, 5 per column with C;
- the first layer's output equals its input.

Per-layer floors at depth 8: hidden ≥ units per bit × 1600 and width ≥ rank. That gives
- X1 7.4M;
- Y1 5.5M;
- chi1 6.2M;
- X2 11.5M (1600 pair-decode units plus 1600 D units);
- Y2 8.2M;
- chi2 6.3M;
- the last round about 4.5M.

That totals about 49.6M, which is where `pkg4` is. So the packing avenue is exhausted;
what is left here is about ±0.3M.

Estimated floors for these layouts: depth 8 about 49.5M, depth 7 about 51.5M, depth 6
about 59.5M, depth 5 about 81M.

Going lower needs fewer units per bit, which is not a packing question:
- the D units are half of X2;
- Y and chi are 1 unit per bit;
- the 432 pass units of the lazy layer cost 1.7M at depths 6 and 7.

A Y layer that reads 1600 E's costs at least 1600 × (2 × 1600 + 1600) = 7.7M, and a chi
layer that reads 1600 bits at least 5.1M plus its outputs.

**Sparse.**
- Packing adds nonzeros: 2 per decoded pair, and more for the DP decoders.
- The cuts that add none (`dy`, `g1`) give the sparse-leaning points: 125,926 at depth 7
  and 107,557 at depth 8. Both dominate the old frontier.
- The floor here is set by about 1 unit per bit per layer, perhaps about 100K at depth 8.

**Depth.** Packing does not change depth, and depth 4 is untouched.

## Composes with

- **Unit-count reductions in X (the D units), Y, chi or the lazy layer.** `pa`, `pc` and
  `g1` only change which features carry the bits.
- **Min-parity** (on) and `decode.py`.
- **Any layout with a layer that copies the state.** `pa`/`pc` apply to any layer that
  reads bits only through copies. `g1` applies to any split theta whose Y layer needs
  both a ⊕ D and D.
- **Digest routing (`s4`, `d2p5`).** It applies wherever a digest crosses a wide layer
  before a narrow last layer.

## Files

- **Builders.**
  - `repo/experiments/xof_shrink/xp3.py`, depths 6-8. Options: `pa`, `pc`, `dy`, `pd1`,
    `g1`, `lf`, `m2`, `d2p5`, `s4`, `s6`.
  - `repo/experiments/xof_shrink/xp4.py`, depth 5. Options: `pa`, `pc`.
  - The variants are defined at the end of each file.
- **`repo/experiments/xof_shrink/packing/`:**
  - `chi_exact5.py`: chi from packed pairs;
  - `col2.py`: 5 bits in 2 features;
  - `ranks.py`: interface ranks.
- **`patch.diff`:** `git -C repo diff`. The new files are included through `git add -N`.
  The compiler is unchanged.
- **`results.jsonl`:** every harness line, plus both audit lines for the audited variants
  (with `ref` set to the reference set used). Rebuilt by `collect.py`.
- **`runs/`:**
  - harness output `*.json`;
  - audits `*.adv1` (ref777) and `*.adv2` (combined-1);
  - `val_*.txt` (validate_bench) and `ranks_*.txt`.
- **Reproduce:** `run.sh <mod:fn>` for the harness and `audit.sh <mod:fn>` for both
  audits.

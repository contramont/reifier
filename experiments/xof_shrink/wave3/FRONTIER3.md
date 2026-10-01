# Wave 3 frontier: 1-round Keccak XOF as a plain SwiGLU MLP (combined-3)

> **Superseded in part (final check, 2026-09-30).** Larger float32 stress sets (1792 mixed-density and
> 1248 dense messages, `audit/ref_stress.py`) showed several dense-best rows below to be float32-tight or
> out of tolerance: d3 `kt1_xca`, d5 `_tr`, d6 `_tr_bt`, d7 `lz5` and d8 `_pf`. The robust frontier
> (worst error <= 0.01 everywhere) is in `../README.md`. The paths below refer to the session scratchpad
> where the waves ran.

**Target.** Unchanged from wave 2.
- Circuit: `xof(msg, depth=3, k)` with `k = Keccak(log_w=6, n=1, c=448, pad_char="_")`.
- Network: reifier's `MLP_SwiGLU`. No residual stream, attention = identity, untied layers, no
  embedding or readout.
- Input: BOS + 1144 message bits. Output: BOS + 672 digest bits.
- Steepness: c = 4, q = 8.

**What counts as verified.** A row counts only if all three checks pass, with `ok: true`, 0 wrong
bits and an empty eager mismatch:
- the harness (`xofbench.py`, 16 random messages);
- `audit/adv_check.py` on `ref777_w6.pt` (69 messages);
- `adv_check.py` on `combined-1/adv/ref_w6.pt` (63 messages).

Every row in section 1 was rebuilt and verified in this directory's `repo/`. The raw JSON is in
`runs3/`, and `results3.jsonl` has one line per variant (`collect3.py`).

Reference points:
- **Baseline** (main, threshold xor): depth 20, dense 3,305,445,348, sparse 2,034,788.
- **Wave 1** (dense / sparse):
  - d4: 160,358,065 / 425,212
  - d5: 89,244,445 / 229,869
  - d6: 69,368,253 / 177,623
  - d7: 63,651,004 / 127,470
  - d8: 61,082,590 / 109,101
- **Wave 2** (combined-2), dense-best points (dense / sparse):
  - d3: 253,823,869 / 1,292,483
  - d4: 134,661,305 / 547,787
  - d5: 79,361,129 / 240,057
  - d6: 57,250,043 / 192,041
  - d7: 49,375,781 / 136,618
  - d8: 45,071,004 / 115,841
- **Wave 2** sparse-best points (dense / sparse):
  - d8: 64,162,035 / 95,955
  - d9: 68,233,410 / 91,778

## 1. Best circuit per depth

The worst audit margin is the larger of the two audit margins; the tolerance is 0.02. The
harness margins are in the Pareto table and in `runs3/`.

| depth | variant | dense | sparse | margins harness / ref777 / c1 | x baseline (dense / sparse) | vs wave 1 (dense) | vs wave 2 best (dense / sparse) | from |
|---|---|---|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt1_xca` | **231,671,853** | 1,222,721 | 3.7e-4 / 1.2e-2 / 8.1e-3 | 14.3 / 1.7 | (new depth) | -8.7% / -5.4% (-22.15M) | and-of-parities, re-verified |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | **124,087,564** | 526,518 | 9.3e-4 / 5.6e-3 / 5.6e-3 | 26.6 / 3.9 | -22.6% | -7.9% / -3.9% (-10.57M) | and-of-parities, re-verified |
| 4 (sparse end) | `xc:d4a_m4k3_mpc_p1dxt_tp` | 143,675,838 | **396,090** | 1.6e-3 / 2.2e-3 / 2.2e-3 | 23.0 / 5.1 | -10.4% | sparse -6.8% vs the w2 sparse end (425,212) | **combined-3** |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_tr` | **73,443,039** | 227,576 | 2.7e-3 / 8.0e-3 / 6.6e-3 | 45.0 / 8.9 | -17.7% | -7.5% / -5.2% (-5.92M) | **combined-3** |
| 5 (sparse end) | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 74,441,726 | **219,208** | 2.2e-3 / 4.0e-3 / 5.3e-3 | 44.4 / 9.3 | -16.6% | sparse -3.2% vs the w2 sparse end (226,445) | **combined-3** |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tr_bt` | **53,284,758** | 179,373 | 2.9e-3 / 8.5e-3 / 8.5e-3 | 62.0 / 11.3 | -23.2% | -6.9% / -6.6% (-3.97M) | **combined-3** |
| 6 (robust) | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 53,915,598 | 178,277 | 4.7e-4 / 1.8e-3 / 1.4e-3 | 61.3 / 11.4 | -22.3% | -5.8% / -7.2% | **combined-3** (theta1 direct instead of the pool: +0.63M, worst margin 1.8e-3 against 8.5e-3) |
| 6 (sparse end) | `c2:d6_lazy_cp_p1dct_tp_bt` | 62,253,531 | **162,221** | 1.5e-3 / 2.4e-3 / 3.0e-3 | 53.1 / 12.5 | -10.3% | sparse -3.5% vs the w2 sparse end (168,175) | **combined-3** |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt` | **48,066,365** | 132,948 | 1.8e-3 / 5.9e-3 / 4.6e-3 | 68.8 / 15.3 | -24.5% | -2.7% / -2.7% (-1.31M) | **combined-3** (dense = and-of-parities' point, 1.7K fewer nonzeros) |
| 7 (sparse end) | `c2:sp_lazy2_l3_p1d` | 67,786,082 | **113,772** | 2.0e-4 / 2.7e-3 / 3.0e-3 | 48.8 / 17.9 | +6.5% | sparse -0.6% vs `sp:lazy2_l3` (114,454) | and-of-parities, re-verified |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra` | **43,761,588** | 114,731 | 3.2e-3 / 1.5e-2 / 1.3e-2 | **75.5** / 17.7 | **-28.4%** | -2.9% / -1.0% (-1.31M) | and-of-parities, re-verified |
| 8 (robust) | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` | 44,218,228 | 114,557 | 1.5e-3 / 4.5e-3 / 3.4e-3 | 74.8 / 17.8 | -27.6% | -1.9% / -1.1% | and-of-parities, re-verified |
| 8 (sparse end) | `c2:sp_base_p1d` | 62,287,910 | **95,637** | 1.1e-4 / 1.0e-3 / 6.3e-4 | 53.1 / 21.3 | +2.0% | sparse -0.3% vs `sp:base` (95,955) | and-of-parities, re-verified |
| 9 (sparse only) | `c2:sp_col1b_p1d` | 66,359,285 | **91,460** | 8.2e-5 / 1.1e-4 / 1.1e-4 | 49.8 / **22.2** | | sparse -0.3% vs `sp:col1b` (91,778) | and-of-parities, re-verified |
| 10 | none | | | | | | | |

For depth 10, nothing changed: no layout at depth 10 beats depth <= 9 on either metric (FRONTIER.md
section 5.3).

**Depth 8 is 43.76M dense, 75.5x below the baseline, and 28.4% below wave 1.** The whole table moves
down again:
- d3 -8.7%, d4 -7.9%, d5 -7.5%, d6 -6.9%, d7 -2.7%, d8 -2.9% dense against wave 2;
- every dense-best row also has fewer nonzeros than the wave-2 row it replaces.

The d5 and d6 bests are new stacks built here and not in any avenue:
- d5: 73.44M against the best avenue point, 74.07M;
- d6: 53.28M against 53.92M.

Extra checks, all passing (`extra_checks3.jsonl`; raw files `runs3/*.val.txt`,
`*.lw4/lw5.json`, `*.stress256.json`). They cover the new d5, d6 and d7 bests, plus the z11 twin
of the d6 best:
- `validate_bench` gives bit-equal weights at log_w 0-2, and depth, dense and sparse match;
- log_w 4 and 5 with 64 messages;
- 256 messages at log_w 6, with worst margins 5.6e-3 (d5), 6.5e-3 (d6) and 3.7e-3 (d7).

## 2. What wave 3 changed

### 2.1 The avenues

**and-of-parities: every even-range parity costs one unit less.**
- Parity of an integer count s in [0, n] needs floor((n-1)/2) gated units for every n >= 6. Before,
  it was ceil(n/2) for even n.
- The core is an n = 6 form with 2 units, with irrational knots at 3 +- (2 sqrt 2 - 2). glu_xor
  ramps extend it to any even n: `p1d_forms.ext`, `ext_c` (centred) and `ext_t` (core at the top of
  the range).
- It is applied wherever an even-range parity sits:
  - the fused-round raw parities (d3, d4: -1816 L1 units at d4);
  - X2's D (5 -> 4 units per column pair, -1.15M at each of d5-d8);
  - the even count parities of d3, d4x and d5 (the lazy column parities, 10; the Walsh pairs, 26);
  - the round-1 raw parities (X1's D at d7-d8, and theta1 direct at d5-d6).
- The AND itself did not get cheaper (section 3).

**d-parity-2d: the same construction, found independently, with an optimality proof.**
- D = parity(C_L + C_R) in 4 units from "parabola bumps" P_c(s) = 8 - (s - c)^2, with knots at
  odd +- 2 sqrt 2. Those are the same knots as and-of-parities' core.
- 3 units is impossible even with real affine 2-D gates: every row and column of the 6x6 grid needs
  2 interior knots, so there are 8 crossings, and 3 lines give at most 6. So 4 is optimal.
- Every d-parity-2d point is dominated by an and-of-parities point of the same depth: bumps have
  slope +-2 at even s, and and-of-parities' top-core placement is flatter where counts are common.

**theta1-t8: round-1 theta reaches its structural floor.**
- The zero lane's units form a column pool that the live bits read for one wo entry each.
- F1 is exact for T <= 8, so it covers the T = 8 columns. It has fewer nonzeros than TH1S.
- F3 is exact for T <= 7, with irrational knots and the layer's shared constant.
- F13 (F3 where T <= 7, F1 where T = 8) takes the round-1 layer from 5202 units (TH1S) to 4451.
  That is the structure's floor of 4448, plus 3.

**lazy-double-and: no new points** (section 3).

### 2.2 Stacks built here (combined-3)

1. **Pool theta + the new count forms (d5, d6).** and-of-parities' d5/d6 bests compute theta1
   directly on the new raw forms: 4619 units. theta1-t8's F13 pool needs 4451, and it stacks with
   P1DX (d5) and P1DC top-core D (d6) without conflict.
   - The gain is 168 units x 3755 = -0.63M at both depths: 74.07M -> 73.44M (d5) and 53.92M ->
     53.28M (d6).
   - The worst audit margins rise from 2.1e-3 / 3.8e-3 to 8.0e-3 / 8.5e-3, still under half the
     tolerance.
   - The `_ra` points (theta1 direct) stay on the front as the margin-robust options at +0.63M.
     At d6 that is `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt`: 53.92M / 178,277, worst audit margin
     1.8e-3.
   - `P1DR` (`_ra`) is a no-op on top of the pool: every round-1 column goes through the pool.
     `_tr_ra` builds the same network as `_tr`.
2. **P1DT: the last theta's even count parity on the top-core form** (`c2.with_p1dt`).
   - C5 (one reduction unit per column) gives counts in [0, 34]. That used to cost 17 glu_xor
     units against Z11 + MPC's 16 on [0, 33], and it now costs 16.
   - C5's column unit reads 5 features where Z11's zigzag reads 11. So in the dense-best d6 and d7
     layouts, `c5 + _bt` gives identical dense and **1.7K fewer nonzeros** than `z11`: d6 181,087 ->
     179,373 and d7 134,662 -> 132,948.
   - On the sparse-leaning c5 points it is -593K dense and -318 sparse:
     - d6 `cp_c5_p1dct_ra`: 56.94M -> 56.34M;
     - d7 `cp_u1_c5`: 51.09M -> 50.50M;
     - d7 `cp_sp_c5`: 51.48M -> 50.88M.
   - d-parity-2d's `_bc` bump version of this (61.81M / 175,253 at d6) was rebuilt in the overlay
     with identical numbers.
3. **The F1 pool (`tp`) + the new forms** gives new sparse ends:
   - d4 `xc:d4a_m4k3_mpc_p1dxt_tp`: 396,090, against 400,884 (and-of-parities) and 413,978
     (theta1-t8);
   - d5 `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp`: 219,208, against 221,099;
   - d6 `c2:d6_lazy_cp_p1dct_tp_bt`: 162,221, against 162,903.

### 2.3 Where the dense goes now (layers as [in, hidden, out], dense in M)

| layer | d8 `..._p1dct_pf_ra` 43.76M | d7 `..._c5_lz5_sl_p1dct_ra_bt` 48.07M | d6 `..._c5_lz5_sl_p1dct_tr_bt` 53.28M | d5 `..._p1dxt_tr` 73.44M |
|---|---|---|---|---|
| 1 | X1 (u1) [1145,1603,641] 4.70 | X1 (u1) 4.70 | theta1 (F13 pool) [1145,4451,1465] 16.71 | theta1 (F13 pool) 16.71 |
| 2 | Y1 [641,2474,1465] 6.80 | Y1 6.80 | chi1 [1465,1602,961] 6.23 | chi1 6.23 |
| 3 | chi1 [1465,1602,961] 6.23 | chi1 6.23 | X2 (4-unit D) [961,2883,1657] 10.32 | X2 10.32 |
| 4 | X2 (4-unit D) [961,2883,1657] 10.32 | X2 10.32 | lazy4c (C5) [1657,2410,489] 9.17 | lazy chi, column parities [1657,5019,489] 19.09 |
| 5 | lazy4 [1657,2202,601] 8.62 | lazy4c (C5) [1657,2410,489] 9.17 | theta3 (range 34, 16 units) [489,6859,449] 9.79 | Walsh [489,12771,673] 21.09 |
| 6 | fold [601,1786,489] 3.02 | theta3 [489,6859,449] 9.79 | chi3 [449,675,673] 1.06 | |
| 7 | parity (flat range 11) [489,2107,449] 3.01 | chi3 [449,675,673] 1.06 | | |
| 8 | chi3 [449,675,673] 1.06 | | | |

Shallower bests:
- d4 `d4x`: [1145,15967,1601] 62.13 | [1601,4483,1657] 21.78 | [1657,5019,489] 19.09 |
  [489,12771,673] 21.09.
- d3: [1145,21977,1676] 87.16 | [1676,28878,471] 110.40 | [471,21119,673] 34.11.

## 3. What wave 3 ruled out

- **A cheaper exact AND of two parities than "free singles + one pair parity"** (and-of-parities).
  None was found in:
  - exhaustive individual-bit integer gates (weights in [-2, 2], offsets in Z/2), 3+3 bits,
    2 units: 0 of 267,850 gate pairs;
  - real-gate variable projection in the 2-D count model, from 4+4 up to 8+9 bits;
  - the same with every per-set function free (4+4, 8+8);
  - real individual-bit gates at 4+4.

  Enumerating every small-range (window-3) lift of chi1 shows that only lifts built on
  p(B xor C) need a single extra parity. So in the fused layout the AND is exactly the pair
  parity. The d3/d4 L1 layers are at the floor((n-1)/2) parity floor, and only a different fused
  layout could move them.
- **D in 3 units**: impossible for any real affine 2-D gates (d-parity-2d's proof).
- **D in 4 units with rational knots**: no such form exists with integer directions
  |a|, |b| <= 4 and rational offsets over the searched (A, Q) grid. The knots must be irrational,
  which is why wave 2's MILP and the "exhaustive" n = 8 and 10 searches missed it.
- **The two-unit mod-2 double AND for the lazy chi** (NEXT item 3; lazy-double-and): none with
  window <= 3. Checked:
  - |w| <= 2 with integer or half-integer knots: 9.27M gate pairs;
  - |w| <= 3 with integer knots: 67.5M pairs and 18.2M exact checks.

  The solver was cross-checked by planted solutions and an independent HiGHS MILP. So the
  planned -1.9M at d6/d7 is not available from pair units.
- **Round-1 theta below the pool structure** (theta1-t8):
  - a 3-unit pool is impossible for T = 8;
  - 3-unit + constant parity pools on [0, 7] were enumerated on 1/10, 1/12 and 1/30 knot grids;
  - at least 3 units per live bit are needed.
- **A silu-flat 5-unit parity of [0, 11]** (NEXT item 5): impossible (exact proof in theta1-t8's
  `syn/flat11.py`). The flat range-11 VP form (`P1DP`, slope 2.0 against MIN_PARITY's 2.85) is
  what the d8 best uses instead.
- **Float32 limits of count parities with knots between integers.** These placements failed an
  audit and are kept with `ok: false` in the avenue runs:
  - bottom-core forms at d3 and d4 (0.021-0.036);
  - VP forms for D at d8 (0.025-0.037);
  - min-parity on d3's odd counts (0.064);
  - bumps on top of TH1S plus a second error source (0.0205-0.029).

  Rule used here: put the non-flat points at the top of the count range (`_t` forms). Where a
  margin matters, use `_g` / `_ra` (d8 robust 44.22M at 4.5e-3; d5/d6 `_ra` at 2.1e-3 / 3.8e-3).

## 4. Verified Pareto table (2-D per depth; 3-D marked)

Generated by `pareto3.py` from four sources:
- wave-1 and wave-2 points (`avenue_points.txt` and combined-2's `runs/`);
- wave-3 avenue points (`avenue_points3.txt`), each verified in its own avenue repo;
- combined-3's runs (`runs3/`, marked "run here").

Rows that build the same network under two names are shown once, with `=`. The margin column is
the worst audit margin for rows run here or in combined-2, and "-" for avenue-only rows.

| depth | variant | dense | sparse | baseline / this (dense / sparse) | this / wave-1 frontier (dense) | this / wave-2 best (dense) | 3-D Pareto | source | worst audit margin |
|---|---|---|---|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt1_xca` | 231,671,853 | 1,222,721 | 14.3x / 1.7x | - | 0.913 | yes | w3:and-of-parities, re-verified (run here) | 1.2e-02 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | 124,087,564 | 526,518 | 26.6x / 3.9x | 0.774 | 0.921 | yes | w3:and-of-parities, re-verified (run here) | 5.6e-03 |
| 4 | `xc:d4a_m4k3_mp_mpc_p1dxt_tr` | 142,688,273 | 401,980 | 23.2x / 5.1x | 0.890 | 1.060 | yes | combined-3 (run here) | 5.3e-03 |
| 4 | `xc:d4a_m4k3_mp_mpc_p1dxt_ra` | 143,319,113 | 400,884 | 23.1x / 5.1x | 0.894 | 1.064 | yes | w3:and-of-parities, re-verified (run here) | 1.3e-03 |
| 4 | `xc:d4a_m4k3_mpc_p1dxt_tp` | 143,675,838 | 396,090 | 23.0x / 5.1x | 0.896 | 1.067 | yes | combined-3 (run here) (= `xc:d4a_m4k3_mp_mpc_p1dxt_tp`) | 2.2e-03 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_tr` | 73,443,039 | 227,576 | 45.0x / 8.9x | 0.823 | 0.925 | yes | combined-3 (run here) (= `xc:d5_m4k2_mp_cp_mpc_p1dxt_tr_ra`) | 8.0e-03 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tr` | 73,454,161 | 225,098 | 45.0x / 9.0x | 0.823 | 0.926 | yes | combined-3 (run here) | 8.0e-03 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_tp` | 74,430,604 | 221,686 | 44.4x / 9.2x | 0.834 | 0.938 | yes | combined-3 (run here) | 5.4e-03 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 74,441,726 | 219,208 | 44.4x / 9.3x | 0.834 | 0.938 | yes | combined-3 (run here) | 5.3e-03 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tr_bt` | 53,284,758 | 179,373 | 62.0x / 11.3x | 0.768 | 0.931 | yes | combined-3 (run here) | 8.5e-03 |
| 6 | `c2:d6_m3_mp_cp_c5_lz5_sl_p1dct_tr_bt` | 53,608,369 | 179,211 | 61.7x / 11.4x | 0.773 | 0.936 | yes | combined-3 (run here) | 8.6e-03 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 53,915,598 | 178,277 | 61.3x / 11.4x | 0.777 | 0.942 | yes | combined-3 (run here) | 1.8e-03 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tp_bt` | 54,272,323 | 173,483 | 60.9x / 11.7x | 0.782 | 0.948 | yes | combined-3 (run here) | 6.4e-03 |
| 6 | `c2:d6_m3_mp_cp_c5_lz5_sl_p1dct_tp_bt` | 54,595,934 | 173,321 | 60.5x / 11.7x | 0.787 | 0.954 | yes | combined-3 (run here) | 6.4e-03 |
| 6 | `c2:d6_m3_mp_cp_c5_p1dct_tr_bt` | 55,053,361 | 170,541 | 60.0x / 11.9x | 0.794 | 0.962 | yes | combined-3 (run here) | 5.6e-03 |
| 6 | `c2:d6_m3_mp_cp_c5_p1dct_tp_bt` | 56,040,926 | 164,651 | 59.0x / 12.4x | 0.808 | 0.979 | yes | combined-3 (run here) | 2.5e-03 |
| 6 | `c2:d6_cp_c5_p1dct_tp_bt` | 56,701,339 | 163,245 | 58.3x / 12.5x | 0.817 | 0.990 | yes | combined-3 (run here) | 2.4e-03 |
| 6 | `c2:d6_lazy_cp_p1dct_tp_bt` | 62,253,531 | 162,221 | 53.1x / 12.5x | 0.897 | 1.087 | yes | combined-3 (run here) | 3.0e-03 |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt` | 48,066,365 | 132,948 | 68.8x / 15.3x | 0.755 | 0.973 | yes | combined-3 (run here) | 5.9e-03 |
| 7 | `c2:d7_m3_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt` | 48,389,976 | 132,786 | 68.3x / 15.3x | 0.760 | 0.980 | yes | combined-3 (run here) | 5.9e-03 |
| 7 | `c2:d7_cp_u1_c5_lz5_sl_p1dct_ra_bt` | 49,192,469 | 131,380 | 67.2x / 15.5x | 0.773 | 0.996 | yes | combined-3 (run here) | 5.8e-03 |
| 7 | `c2:d7_cp_u1_c5_sl_p1dct_ra_bt` | 49,820,117 | 127,236 | 66.3x / 16.0x | 0.783 | 1.009 | yes | combined-3 (run here) | 4.6e-03 |
| 7 | `c2:d7_cp_u1_c5_p1dct_ra_bt` | 50,495,381 | 122,710 | 65.5x / 16.6x | 0.793 | 1.023 | yes | combined-3 (run here) | 4.3e-03 |
| 7 | `c2:d7_cp_sp_c5_p1dct_ra_bt` | 50,884,973 | 121,894 | 65.0x / 16.7x | 0.799 | 1.031 | yes | combined-3 (run here) | 2.5e-03 |
| 7 | `rs:split_first_lazy4c_xpg_x1s` | 58,913,522 | 120,488 | 56.1x / 16.9x | 0.926 | 1.193 | yes | w2:round-structure | - |
| 7 | `rs:split_first_lazy_xpg_x1s` | 63,754,034 | 115,944 | 51.8x / 17.5x | 1.002 | 1.291 | yes | w2:round-structure | - |
| 7 | `c2:sp_lazy2_l3_p1d` | 67,786,082 | 113,772 | 48.8x / 17.9x | 1.065 | 1.373 | yes | w3:and-of-parities, re-verified (run here) | 3.0e-03 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra` | 43,761,588 | 114,731 | 75.5x / 17.7x | 0.716 | 0.971 | yes | w3:and-of-parities, re-verified (run here) | 1.5e-02 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` | 44,218,228 | 114,557 | 74.8x / 17.8x | 0.724 | 0.981 | yes | w3:and-of-parities, re-verified (run here) | 4.5e-03 |
| 8 | `c2:d8_rp_cp_sp_p1dct_pf_ra` | 45,425,684 | 111,917 | 72.8x / 18.2x | 0.744 | 1.008 | yes | w3:and-of-parities, re-verified (run here) | 8.8e-03 |
| 8 | `c2:d8_rp_cp_sp` | 47,598,310 | 111,629 | 69.4x / 18.2x | 0.779 | 1.056 | yes | combined-2 | 6.6e-03 |
| 8 | `xo:split_first_middle_mp_pkud` | 49,708,462 | 109,285 | 66.5x / 18.6x | 0.814 | 1.103 | yes | w2:optimizer | - |
| 8 | `xo:split_first_middle_pkud_mc` | 49,932,667 | 107,683 | 66.2x / 18.9x | 0.817 | 1.108 | yes | w2:optimizer(flagged-margin) | - |
| 8 | `rs:split_first_middle_xpyd_x1p_x2pc` | 51,355,587 | 107,259 | 64.4x / 19.0x | 0.841 | 1.139 | yes | w2:round-structure | - |
| 8 | `rs:split_first_middle_mp_xpydg3_x1s_gm` | 53,712,700 | 105,657 | 61.5x / 19.3x | 0.879 | 1.192 | yes | w2:round-structure | - |
| 8 | `rs:split_first_middle_mp_xpg_x1s` | 54,686,660 | 101,609 | 60.4x / 20.0x | 0.895 | 1.213 | yes | w2:round-structure | - |
| 8 | `rs:split_first_middle_xpg_x1s` | 55,644,185 | 100,327 | 59.4x / 20.3x | 0.911 | 1.235 | yes | w2:round-structure | - |
| 8 | `rs:split_first_middle_mp_xpg_x12sc` | 56,066,324 | 98,617 | 59.0x / 20.6x | 0.918 | 1.244 | yes | w2:round-structure | - |
| 8 | `rs:split_first_middle_xpg_x12sc` | 57,023,849 | 97,335 | 58.0x / 20.9x | 0.934 | 1.265 | yes | w2:round-structure | - |
| 8 | `c2:sp_base_p1d` | 62,287,910 | 95,637 | 53.1x / 21.3x | 1.020 | 1.382 | yes | w3:and-of-parities, re-verified (run here) | 1.0e-03 |
| 9 | `c2:sp_col1b_x2e_p1d` | 64,339,445 | 94,340 | 51.4x / 21.6x | - | - | yes | w3:and-of-parities, re-verified (run here) | 1.2e-04 |
| 9 | `c2:sp_col1b_p1d` | 66,359,285 | 91,460 | 49.8x / 22.2x | - | - | yes | w3:and-of-parities, re-verified (run here) | 1.1e-04 |

## 5. Headroom (dense), updated

| depth | now | estimate for these layouts | what is left |
|---|---|---|---|
| 8 | 43.76M | ~42.2M (and-of-parities) | Every full-width layer is at its unit floor for the known forms. The slack is the digest carries (~2M), the u1 decode, and the fold + parity split. |
| 7 | 48.07M | ~47.3M | theta3 costs 16 units per count on range 33 or 34. NEXT 1 would save -0.46M. Wave 2's ~47.5M estimate counted the double AND, which is now ruled out. |
| 6 | 53.28M | ~52.5M | Round 1 is at its pool floor (4451 units), and chi1 and X2 are at their unit floors. What is left is NEXT 1 (-0.46M) and the digest carries. |
| 5 | 73.44M | ~73M | Every parity is at floor((n-1)/2), round 1 is at its pool floor, and X2 is at its floor. What is left is the digest carries. |
| 4 | 124.09M | ~110M (and-of-parities, d4x family) | L1 (62.1M) is 1464 singles plus 1600 pair parities at floor((n-1)/2). Only a different fused layout would move it (section 3). |
| 3 | 231.67M | ~210M | The fused round 2's range-26 pair parities, again pair-parity bound. |

- 100x the baseline is 33.05M. At depth 8 that needs a further -24%.
- The known unit forms leave about 1.5M at depth 8.
- My estimate for this architecture is still about 42M dense at depth 8 (about 79x), and about 88K
  sparse (the sparse-focus family floor).

## 6. NEXT (ranked)

1. **lazy-double-and's chained pair, now that even ranges are cheap (d6, d7; about -0.46M
   each; not built).**
   - lazy-double-and found a chained pair: the own bit (x, x, z) and the column bit (x-1, x, z),
     which share E(x+1, x, z). With one valid single-AND product unit each plus the free pass
     term, the pair reaches window 3 instead of 4.
   - With Z11 that takes the count range from 33 to 32. They rated it worthless because
     MPC(33) = glu_xor(32) = 16 units.
   - With the floor((n-1)/2) forms, range 32 costs 15 units (`FORMS_EXTT[32]`, as `_bt` does for
     34). That is -320 theta3 units, about -0.46M at each of d6 and d7, at the same lazy width.
   - The forms are in `lazy-double-and/search/chain4.py`: 24 window-3 solutions at |w| <= 2, each
     with a half-integer knot.
   - The product units are shared by the counts that read a column, so each replaced unit must
     stay a valid single AND in those counts. The search imposed that.
   - Chaining C5 instead saves nothing: 34 -> 33 is still 16 units.
   - Check the margin: the chain has knots between integers on E inputs, and range 32 puts its
     top core at 29-32.
2. **d8's fold + parity.** Fold windows and parity costs now trade differently:
   - a window of 10 gives a 4-unit parity but needs a 5th fold unit;
   - measured by d-parity-2d as `_w10` (43.99M), it loses to the flat range-11 parity `_pf`
     (43.76M).
   - A different split of the lazy4 count, for example 2 folds of mixed windows, is unexplored.
3. **A cheaper fused round at d3/d4.** The AND is the pair parity in the current fused layout.
   A different layout needs a different chi1 lift, and every window-3 lift was enumerated. A wider
   window costs L2 units (about +15.6M for window 4 at d4).

## 7. Merge notes, reproduce, files

**How the repo was merged.** `repo/` = combined-2's repo, then:
- **and-of-parities' patch**, applied as-is: `xs3.py`, `xs4.py`, `xc.py`, `c2.py`,
  `depth_low/lib/python/dl3.py`, plus the new `depth_low/lib/python/p1d_forms.py`.
- **theta1-t8's patch (TH1P)**, 3-way merged on top of it. `xs3.py` merged cleanly. `c2.py` and
  `xc.py` had only additive conflicts, where both appended variants, and both sides were kept.
- **combined-3's variants**, at the end of `c2.py` (`with_p1dt` and the `_tr`, `_tp` and `_bt`
  stacks) and of `xc.py`.

**d-parity-2d's builder changes** conflict with and-of-parities' in 21 hunks: both rewrote the same
D / count-parity code paths in `xs3`, `xs4`, `xc`, `c2` and `dl3`.
- Its construction is the same parity form, and every one of its points is dominated.
- So its files are kept as an overlay, `repo/experiments/xof_shrink/overlay_d4/`. It works like
  combined-2's `overlay_opt/`.
- Regression through the overlay, with identical numbers and margins to d-parity-2d's own runs:
  - `run.sh c2:d8_rp_m4s4_mp_cp_u1_sl_g_d4_w10_r overlay_d4`: 43,986,756 / 116,923;
  - `c2:d6_cp_c5_d4_bc`: 61,808,139 / 175,253.

**Regression checks in the merged repo.** These rebuild with identical dense, sparse and margins:
- combined-2's `c2:d8_rp_m4s4_mp_cp_u1_sl` (45,071,004 / 115,841);
- theta1-t8's `c2:d6_m4s4_mp_cp_z11_lz5_tr_sl` (54,430,038 / 181,451);
- every and-of-parities headline row: d3, d4, d4a, d5, d6, d7, d8, d8 robust, sp_base,
  sp_lazy2_l3 and sp_col1b.

**Switches** (module globals, one variant per process):
- and-of-parities: `P1D`, `P1DK` (dl3), `P1DC`, `P1DP`, `P1DR` (xs3), `P1DX` (xs4);
- theta1-t8: `TH1P` (xs3, `with_th1p` / `xc._th1p`, forms F1, F3, F13 and F12);
- combined-3: `with_p1dt` (c2), which adds `FORMS_EXTT` for every even n to `P1DP`.

Suffixes:

| suffix | meaning |
|---|---|
| `_tr` | F13 pool |
| `_tp` | F1 pool |
| `_p1dct` | top-core D |
| `_p1dxt` | xs4's count parities on the top core |
| `_ra` | round-1 raw parities on the new forms |
| `_pf` | flat range-11 parity |
| `_bt` | P1DT |

**Reproduce.**

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
C=$S/xof3/combined-3
$C/run.sh c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tr_bt   # harness + both audits -> runs3/
$C/extra_checks.sh xc:d5_m4k2_mp_cp_mpc_p1dxt_tr   # validate_bench, log_w 4/5, 256-message stress
python3 $C/collect3.py                              # results3.jsonl + summary
python3 $C/pareto3.py > $C/pareto3_table.md         # section 4
```

**Files (wave 3).**
- `FRONTIER3.md`: this file.
- `results3.jsonl`, `runs3/`, `extra_checks3.jsonl`: every combined-3 run, including the extra
  checks and the overlay runs (`overlay_d4__*`).
- `pareto3.py`, `pareto3_table.md`, `avenue_points3.txt`.
- `patch.diff`: `git -C repo diff --cached` against the wave-2 base commit (everything, including
  new files).
- `patch_vs_combined2.diff`: combined-3's `experiments/xof_shrink` against combined-2's.
  `patch -p2` inside a copy of combined-2's `repo/` reproduces `repo/experiments` exactly
  (checked). `src/` is unchanged from combined-2.
- `run.sh`, `extra_checks.sh`, `collect3.py`, `batch*.txt`.
- `merge3/`, `merge3at/`: the 3-way merge work files.

The avenue notes and searches stay in their own directories under `$S/xof3/`:
- `and-of-parities/`: `NOTES.md` and `search/`;
- `d-parity-2d/`: `NOTES.md` and `search/`;
- `theta1-t8/`: `NOTES.md` and `syn/`;
- `lazy-double-and/`: `NOTES.md` and `search/`.

The wave-2 files (FRONTIER.md, results.jsonl, runs/, NEXT.md, ...) are combined-2's copies,
unchanged.

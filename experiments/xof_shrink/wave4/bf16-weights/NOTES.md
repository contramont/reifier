# Wave 4, avenue bf16-weights: which size reductions survive bfloat16 weights?

Mode **w16** of `audit/bf16_check.py`: weights rounded to bfloat16 (8 significant bits), float32
activations. Configuration: 3 XOF steps, 1 round, log_w 6 (and 4). Every size comes from the
harness (`xofbench.build_layers` + `metrics`, printed by `bf16_check.py` and `xofbench.py`); every
run is a line of `results.jsonl`.

Criterion (brief): **correct** = 0 wrong bits (nearest) and 0 boolify errors; **robust** = worst
|out/BOS - bit| <= 0.01 on the ref_gen audit (69 messages: 9 edge cases + 60 random) and on the
stress sets (log_w 6: `ref_stress mixed 8 16` = 1792 + `dense 8 12` = 1248 messages; log_w 4:
`mixed 4 8` = 448 + `dense 4 8` = 416). Reference sets: `refs/` of this avenue, made in fresh
processes from the unpatched keccak.

## Headline

- **Every point of the float32 robust frontier breaks with bf16 weights.** At log_w 6 on the audit:
  d8 28,713 wrong bits (errors up to 1e17), d7 29,181, d6 17,814, d5 17,311, d4 17,014, d3 20,815,
  and even the sparse end `c2:sp_col1b_p1d` 6,199 (table F0).
- **The SwiGLU construction is never the cause.** c*q = 32 (gates), v = 4/q = 1/2 (values),
  1/(c*q*v) = 1/16 (outs), the step offsets 1/2 -+ 1/(2c) = 0.625 / 0.375, the norm weights (1)
  and all step units (integer weights times 32) are representable. So the threshold baseline,
  glu_xor, chi, chi+iota and every integer / half-integer layout compute **bit-identical** numbers
  in w16 and float32 (table L). This needs 4/q dyadic: with q = 12 (v = 1/3) even
  `variants:glu_chi_iota` gets 1,809 wrong bits in w16 at log_w 4; q = 8 and 16 are exact.
- **What breaks are unit coefficients with more than 8 significant bits**, and most of them can be
  removed without changing the function (`bf16_units.py`, below). After that every weight is
  representable (certified with `audit/bf16_weights.py`: 0 non-representable weights for every
  frontier point at log_w 6), so **w16 = float32 bit for bit**, and float32 margins decide.
- **What cannot be fixed**: non-dyadic unit products and irrational knots **on the raw message**
  (layer 1 has no second constant feature): min-parity on raw bits (`mp`, `mpx`, `mp17`), p1d on
  raw bits (`_r`, `_ra`, dl3's `p1d`/`xca`), TH1S (`th`), the F13 pool (`tr`), and the flat forms
  (`pf`, `p1dcf`, irrational products on counts). MIN_PARITY on counts (`mpc`) is replaced at
  equal unit count by the p1d form of range n + 1 (`pm`).
- **w16-correct robust frontier at log_w 6** (float32 = w16 error <= 0.01 on audit + 3040 stress
  messages; table F1):

  | depth | variant (XOF_BF16) | dense | sparse | vs float32 frontier |
  |---|---|---|---|---|
  | 3 | `bfv:d3_kt` (ub) | 252,252,979 | 1,308,567 | +5.8% |
  | 4 | `bfv:d4x_m4k2_kt1` (ub) | 136,155,752 | 546,288 | +9.7% |
  | 5 | `bfv:d5_k2_tp` (u1) | 79,469,937 | 230,257 | +7.3% |
  | 6 | `bfv:d6_m4s4_cp_c5_sl_tp_bt` (ub1) | 56,283,191 | 181,151 | +4.4% |
  | 7 | `bfv:d7_c5_p1dct_bt` (ub) | 49,787,174 | 134,246 | +2.0% |
  | 8 | `bfv:d8_g_p1dct` (ub) | 45,144,507 | 117,433 | +2.1% |
  | 9 | `sp:col1b` (u), sparse end | 68,233,410 | **91,778** | sparse +0.3% |

  The cost is in layer 1 (round-1 parities on raw bits lose `mp` and `_ra`), so layouts whose
  first layer is a full theta pay 4.4-9.7% (d3-d6) and the split-round layouts (d7, d8) about 2%.
  All points are also correct and robust at log_w 4 (table F4).
- **b16 (information only; the bf16-full avenue owns it)**: every point fails with thousands of
  wrong bits (tables F0, F1, F4, last column). `variants:glu_xor_clean_everywhere` is the only circuit
  seen correct in b16 (log_w 4, margin 2.0e-2).

## Method: why the constructions are exact

1. **Representability decides everything.** bf16 keeps 8 significant bits. If every weight of a
   circuit is representable, mode w16 computes exactly the float32 numbers, so the w16 error equals
   the float32 error and the float32 robustness criterion carries over unchanged.
   `audit/bf16_weights.py LOG_W STEPS mod:fn` lists the non-representable weights per layer and
   matrix, in unit terms (gate = wg/32, value = wv*2, out = wo*16).
2. **Unit rescaling** (`bf16_units.rescale_units`, `XOF_BF16=u`). A gated unit
   `outs * max(0, g.x) * (v.x)` equals `(outs/f) * max(0, alpha g.x) * (v.x f/alpha)` for
   alpha > 0; powers of 2 are free, so alpha, f are taken in [1, 2) and gates only get sharper.
   Per unit (cached by pattern), the pair that makes all entries representable is searched among
   candidates from the entries' own mantissas and odd ratios p/q <= 15. This removes every
   rational non-dyadic factor that sits on one side of a unit:
   - unit-sharing ratios: u1's 6/7 and 4/7 (X1 shares D between u = a1 + 2a2 - 3D and
     u3 = 2(a1 + 2a2) - 7D), the 1/3 of sp's Y1 decode and of the Walsh layer (dl3);
   - constant units shared by outputs whose constants are 7/2, 4, 3 (the p1d forms' c = -7 times
     coefficients): base 1/(7/2) makes the outs -1, 3/4, 7/8;
   - lazy4c's column reduction `max(0, 9 - 3c)(-2/3 - c/3)` = `max(0, 12 - 4c)(-1/2 - c/4)`;
   - the p1d slopes 3 + 2 sqrt 2 (gate) and 3 - 2 sqrt 2 (value): their product is exactly 1, so
     alpha = 8/(3 + 2 sqrt 2) makes them 8 and 1/8 (the knots stay, in the biases);
   - the F1 pool theta's 7/3 and 70/3 values and 4/7 outs, except 80 units per 1600 bits whose
     gate x value x out product is 40/3 (below).
3. **Split out columns** (in `rescale_units`). Out columns that stay non-representable get extra
   hidden units with the same gate and the value / 256^j, carrying the lower bits of the outs:
   o = p0 + p1/256 (+ p2/65536). Used for constant units whose outs are sums of many forms'
   constants (d3c1's layer 1: constants such as 1891/32, outs such as 1891/70) and for the pool's
   40/3 units. One level (16 bits) is not enough for d3c1: its layer 1 moves by 2.6e-4 and its
   output by 2e-2 (float64); two levels (24 bits) match to 1e-5. For the pool one level (`ub1`,
   +1 unit per split unit) moves the d6 point's output by up to 1.7e-3 in float64 (two levels:
   4.2e-4, the pool amplifies layer-1 noise ~100x), yet its float32 = w16 errors stay within
   7.0e-3 on all stress messages, and it is 1.2M smaller than with two levels.
4. **Split biases** (`bf16_units.split_bias`, `XOF_BF16=ub`). The p1d knots 3 -+ (2 sqrt 2 - 2)
   are irrational, so their biases need more than 8 bits. Every layer after the first whose biases
   need it gets the constant features BOS/256 and BOS/65536 (the layer before emits them from its
   BOS unit: one more wo row each), and every gate or value bias is written
   p0 BOS + p1 BOS/256 + p2 BOS/65536 with bf16 parts: 24 bits. RMSNorm scales all features
   alike, so this is exact. Only layers with non-representable biases get the features (at d8
   only X2: 2 features, 14.7K dense at log_w 6; adding them to all 7 layers cost 86.8K). One level
   (16 bits) is not enough: the top-core forms' biases reach ~150 gate units at the top of wide
   count ranges, and 2^-17 of that moved depth-low's d4x output by 2.7e-2 (float64). Two levels
   match the unsplit circuit to 2e-7 (d4x) and 1e-5 (d8) in float64 (`dbg_cmp.py`).
5. **Layer 1** reads the message (BOS + bits), which has no second constant feature, so irrational
   knots on raw bits keep ~2^-9 relative precision: the p1d core's value root 7.83 rounds by 0.015,
   a lattice error of ~0.06 per form. `raw_p1d_err.py` (this folder) measures it on the lattice for
   every position and orientation of the core (slopes 8 and 1/8, all four biases rounded to bf16):
   the best is 5.8e-2 at n = 6, 8.9e-2 at n = 8, 0.12 at n = 10, 0.15-0.22 for n = 14-22, far
   above the budget. The gate and value slopes must stay powers of 2, because gate slope x value
   slope x out = 1 exactly, so the biases cannot borrow precision from the slopes. Min-parity
   forms have non-dyadic unit products (MINPAR7: 25/6, 47/3, 5/3; MIN_PARITY and TH1S: irrational
   or 1/5, 1/13), which no rescaling can fix, because gate slope x value slope x out is invariant. `wave4_bf16w/mp7_family.py` solves
   MINPAR7's active pattern symbolically (a 1-parameter family); for every knot b0 = p/q with
   q <= 32, the only rational member is MINPAR7 itself, which is not dyadic. So layer 1 uses
   glu_xor (or dl3's plain/centred integer pieces), and that is where the size goes.
6. **`pm`**: MPC's MIN_PARITY on odd count ranges n >= 5 is replaced by the even p1d top-core
   form of range n + 1 (exact on [0, n], (n - 1)/2 units as MIN_PARITY), made exact by 2 and 4.

## Tables

### F0. The float32 robust frontier with bf16 weights (log_w 6, audit, no fix)

| depth | variant | dense | f32 | w16 (wrong nearest / boolify) | b16 |
|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt_xca` | 238,512,173 | 2.2e-3 | 5.2e4 (20,815 / 19,709) | 23,805 / 20,790 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | 124,087,564 | 2.0e-3 | 1.1e5 (17,014 / 18,835) | 19,011 / 20,974 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` | 74,073,879 | 1.2e-3 | 1.8e1 (17,311 / 19,467) | 18,642 / 21,091 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 53,915,598 | 1.4e-3 | 8.1 (17,814 / 20,080) | 19,853 / 21,501 |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` | 48,831,997 | 4.3e-3 | 4.9e24 (29,181 / 21,814) | 30,003 / 22,182 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` | 44,218,228 | 3.0e-3 | 1.1e17 (28,713 / 22,008) | 28,607 / 22,191 |
| 9 | `c2:sp_col1b_p1d` | 66,359,285 | 1.1e-4 | 1.1 (6,199 / 12,881) | 11,542 / 18,612 |

(69 messages x 672 bits = 46,368 bits per mode.)

### F1. w16-correct robust frontier, log_w 6 (float32 = w16, bit-identical)

Worst |out/BOS - bit| on audit / mixed stress / dense stress; ratios to the threshold baseline
(3,305,445,348 dense, 2,034,788 sparse).

| depth | variant | XOF_BF16 | dense | sparse | x baseline (dense / sparse) | vs f32 frontier (dense) | error audit / mixed / dense | b16 on the audit (wrong nearest / boolify) |
|---|---|---|---|---|---|---|---|---|
| 3 | `bfv:d3_kt` | ub | 252,252,979 | 1,308,567 | 13.1 / 1.6 | +5.8% | 1.5e-03 / 5.2e-03 / 4.3e-03 | 22,691 / 20,175 |
| 4 | `bfv:d4x_m4k2_kt1` | ub | 136,155,752 | 546,288 | 24.3 / 3.7 | +9.7% | 1.2e-03 / 2.2e-03 / 1.9e-03 | 19,671 / 20,651 |
| 4 | `bfv:d4x_m4k2_kt` | ub | 136,897,192 | 543,394 | 24.1 / 3.7 | +10.3% | 1.2e-03 / 2.2e-03 / 1.9e-03 | 17,977 / 20,665 |
| 5 | `bfv:d5_k2_tp` | u1 | 79,469,937 | 230,257 | 41.6 / 8.8 | +7.3% | 4.4e-03 / 7.0e-03 / 6.8e-03 | 18,876 / 20,951 |
| 5 | `bfv:d5_k2_p1dxt_pm` | ub | 79,639,110 | 239,774 | 41.5 / 8.5 | +7.5% | 1.6e-03 / 4.1e-03 / 3.5e-03 | 17,266 / 20,406 |
| 6 | `bfv:d6_m4s4_cp_c5_sl_tp_bt` | ub1 | 56,283,191 | 181,151 | 58.7 / 11.2 | +4.4% | 4.5e-03 / 7.0e-03 / 7.0e-03 | 23,128 / 21,188 |
| 6 | `bfv:d6_m3_cp_c5_sl_tp_bt` | ub1 | 56,542,180 | 180,989 | 58.5 / 11.2 | +4.9% | 2.7e-03 / 6.9e-03 / 4.9e-03 | 23,151 / 20,856 |
| 6 | `c2:d6_cp_c5_p1dct_tp_bt` | ub1 | 57,945,567 | 175,057 | 57.0 / 11.6 | +7.5% | 2.8e-03 / 6.5e-03 / 4.9e-03 | 22,459 / 20,671 |
| 6 | `bfv:d6_c5_lz5_p1dct_bt` (no pool) | ub | 59,426,119 | 184,863 | 55.6 / 11.0 | +10.2% | 1.5e-03 / 2.9e-03 / 3.3e-03 | 17,450 / 21,113 |
| 7 | `bfv:d7_c5_p1dct_bt` | ub | 49,787,174 | 134,246 | 66.4 / 15.2 | +2.0% | 2.4e-03 / 9.3e-03 / 7.5e-03 | 17,858 / 21,077 |
| 8 | `bfv:d8_g_p1dct` | ub | 45,144,507 | 117,433 | 73.2 / 17.3 | +2.1% | 2.4e-03 / 5.1e-03 / 4.4e-03 | 17,498 / 21,015 |
| 9 | `c2:sp_col1b_p1d` | ub | 66,374,023 | 94,026 | 49.8 / 21.6 | +0.0% | 1.1e-04 / 1.7e-04 / 1.4e-04 | 11,568 / 18,714 |
| 9 | `sp:col1b` (sparse end) | u | 68,233,410 | 91,778 | 48.4 / 22.2 | +2.8% | 3.2e-05 / 1.1e-04 / 5.4e-05 | 12,028 / 18,523 |

What each point is (`bfv.py`):
- d8 `bfv:d8_g_p1dct` = the float32 d8 point `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` without `mp`
  and `_ra`: X1 (u1) | Y1 | chi1 | X2 (cp, D on the p1d top form) | lazy4 | fold | parity (glu_xor)
  | chi3, digest 1 m4s4, `sl`.
- d7 `bfv:d7_c5_p1dct_bt` = `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` without `mp`, `mpc` (inactive
  there) and `_ra`.
- d6 `bfv:d6_m4s4_cp_c5_sl_tp_bt` = the float32 d6 point `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt`
  with the round-1 theta pool F1 (`tp`) instead of `mp` + `_ra`, and without `lz5`: the pool
  becomes exact with rescaling + one-level out split (`ub1`), but with `lz5` it is float32-tight
  (1.1-1.2e-2, table F3), as the pool already was in float32.
- d5 `bfv:d5_k2_tp` = xs4's d5 layout (digest 1 at 4 bits per feature, k2, cp) with the F1 pool
  in round 1 and glu_xor parities elsewhere: no p1d, so no split features (`u1`). With p1d in the
  Walsh layer (`p1dxt`, `pm`) the pool gets float32-tight (1.1-1.5e-2, table F3); without the pool,
  `bfv:d5_k2_p1dxt_pm` = `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` without `mp`, `_ra`, with `pm` for `mpc`.
- d4 `bfv:d4x_m4k2_kt1` = `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` without the raw forms (`mpx`, `p1d`,
  `xca`) and `wmpc`, with P1DK `kt1` (count parities on the top-core forms, odd ranges on n + 1).
- d3 `bfv:d3_kt` = dl3's d3 (glu_xor raw parities) with P1DK `kt`. The centred raw pieces (`c1`)
  are tight in this setting (`bfv:d3c1` 9.7e-3, `bfv:d3c1_kt` 1.5e-2 on the audit).
- d9 `c2:sp_col1b_p1d` and `sp:col1b`: the sparse-first layouts are integer except for the p1d
  count forms; without p1d (`u`) the sparse count is lowest.

### F2. Sparse ends per depth, log_w 6 (all robust unless marked)

| depth | variant | XOF_BF16 | dense | sparse | float32 sparse end (brief) | error audit / mixed / dense |
|---|---|---|---|---|---|---|
| 4 | `bfv:d4a_m4k3` | u | 162,246,830 | 420,226 | 396,090 (+6.1%) | 2.3e-04 / 6.4e-04 / 4.7e-04 |
| 4 | `bfv:d4a_m4k3_p1dxt_pm` | ub | 148,997,212 | 434,874 | 396,090 (+9.8%) | 1.4e-03 / 2.2e-03 / 2.8e-03 |
| 4 | `bfv:d4a_m4k3_tp_p1dxt_pm` | ub1 | 145,092,012 | 435,306 | 396,090 (+9.9%) | 4.2e-03 / 7.8e-03 / 1.0e-02 **not robust** (max 0.01002) |
| 5 | `bfv:d5_k2` | u | 83,375,137 | 229,825 | 219,208 (+4.8%) | 7.1e-04 / 2.1e-03 / 2.2e-03 |
| 5 | `bfv:d5_k2_tp` | u1 | 79,469,937 | 230,257 | 219,208 (+5.0%) | 4.4e-03 / 7.0e-03 / 6.8e-03 |
| 6 | `xs3:lazy_middle_cp` | u | 69,116,552 | 169,151 | 162,221 (+4.3%) | 2.9e-04 / 9.9e-04 / 1.1e-03 |
| 6 | `bfv:d6_lazy_cp_p1dct_bt` | ub | 67,399,055 | 173,601 | 162,221 (+7.0%) | 1.0e-03 / 1.7e-03 / 2.0e-03 |
| 6 | `c2:d6_lazy_cp_p1dct_tp_bt` | ub1 | 63,493,855 | 174,033 | 162,221 (+7.3%) | 5.9e-03 / 9.6e-03 / 1.0e-02 **not robust** (max 0.01002) |
| 7 | `sp:lazy2_l3` | u | 70,156,703 | 114,454 | 113,772 (+0.6%) | 1.4e-04 / 1.6e-03 / 1.7e-03 |
| 7 | `c2:sp_lazy2_l3_p1d` | ub | 67,824,806 | 118,904 | 113,772 (+4.5%) | 1.7e-03 / 3.8e-03 / 3.4e-03 |
| 8 | `sp:base` | u | 64,162,035 | 95,955 | 95,637 (+0.3%) | 7.8e-05 / 3.2e-04 / 4.0e-04 |
| 8 | `c2:sp_base_p1d` | ub | 62,302,648 | 98,203 | 95,637 (+2.7%) | 9.5e-04 / 1.2e-03 / 1.2e-03 |
| 9 | `sp:col1b` | u | 68,233,410 | 91,778 | 91,460 (+0.3%) | 3.2e-05 / 1.1e-04 / 5.4e-05 |

The split features cost nonzeros (2 per split bias), so at the sparse end the p1d forms stop
paying: `sp:base` / `sp:col1b` without p1d are sparser than their `_p1d` versions in bf16.

### F3. Not robust: error > 0.01 on some check (bits correct unless marked), log_w 6

| depth | variant | XOF_BF16 | dense | error audit / mixed / dense | what makes it tight |
|---|---|---|---|---|---|
| 5 | `bfv:d5_k2_tp_p1dxt_pm` | ub1 | 75,733,910 | 7.3e-03 / 1.2e-02 / 1.1e-02 | pool theta + p1d (pm) in the Walsh layer |
| 5 | `bfv:d5_k2_tp_p1dxt_pm` | ub | 76,935,510 | 5.9e-03 / 1.5e-02 / 1.4e-02 | the same with the two-level out split |
| 5 | `bfv:d5_k2_tp_p1dxt` | ub1 | 76,475,350 | 7.3e-03 / 1.2e-02 / 1.1e-02 | pool theta + p1d on even Walsh ranges |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tp_bt` | ub1 | 55,520,919 | 4.6e-03 / 1.1e-02 / 1.2e-02 | pool + lz5 (float32: 6.4e-3 on its wave-3 audit) |
| 6 | `bfv:d6_m4s4_cp_c5_lz5_tp_bt` | ub1 | 56,309,079 | 4.6e-03 / 1.1e-02 / 1.2e-02 | pool + lz5 |
| 6 | `c2:d6_cp_c5_lz5_sl_p1dct_tp_bt` | ub1 | 56,646,015 | 4.6e-03 / 1.1e-02 / 1.2e-02 | pool + lz5 |
| 7 | `bfv:d7_c5_lz5_p1dct_bt` | ub | 49,024,902 | 4.5e-03 / 9.9e-03 / 1.1e-02 | LZ5 decoders after u1 (as in float32) |
| 8 | `bfv:d8_pm_p1dct` | ub | 44,699,869 | 4.3e-03 / 1.1e-02 / 1.3e-02 | the fold's parity on the range-12 p1d form (5 units instead of 6) |
| 3 | `bfv:d3c1_kt` | ub | 252,260,915 | 1.5e-02 / 5.3e-02 (0 / 22 wrong) / 2.1e-02 | centred raw pieces + P1DK |
| 3 | `bfv:d3c1` | u | 258,781,369 | 9.7e-03 / - / - | centred raw pieces |

All of these compute the same numbers in w16 and float32: they are tight for float32 reasons (the
same tricks are tight in the float32 frontier), and bf16 weights add nothing once every weight is
representable. `bfv:d3c1_kt` even gets 22 boolify errors on the mixed stress set.

### F3b. No split features (`u` only): every weight an exact dyadic by construction, log_w 6

| depth | variant | dense | sparse | vs float32 frontier | error audit / mixed / dense |
|---|---|---|---|---|---|
| 3 | `bfv:d3` | 258,773,437 | 1,297,295 | +8.5% | 1.4e-03 / - / - |
| 4 | `bfv:d4x_m4k2` | 140,256,921 | 536,339 | +13.0% | 7.4e-04 / 1.5e-03 / 1.5e-03 |
| 5 | `bfv:d5_k2` | 83,375,137 | 229,825 | +12.6% | 7.1e-04 / 2.1e-03 / 2.2e-03 |
| 6 | `bfv:d6_c5_lz5` | 60,981,043 | 180,881 | +13.1% | 7.6e-04 / 2.0e-03 / 2.3e-03 |
| 7 | `bfv:d7_c5` | 51,417,138 | 130,264 | +5.3% | 2.4e-03 / 5.1e-03 / 4.4e-03 |
| 8 | `bfv:d8_g` | 46,275,049 | 115,231 | +4.7% | 2.4e-03 / 5.0e-03 / 4.4e-03 |

The p1d count forms with the split bias save 1.13M (d8), 1.63M (d7), 1.55M (d6, same layout),
3.74M (d5, with `pm`) and 4.10M (d4) against these.

### F4. log_w 4 check of the frontier (float32 = w16)

The same variants at log_w 4 (threshold baseline 206,803,140 / 508,772). Stress sets: mixed 448 and
dense 416 messages. All correct and within 4.6e-3; every float32-frontier variant, as built, has
thousands of wrong bits in w16 there too.

| depth | variant | XOF_BF16 | dense | sparse | x baseline (dense / sparse) | float32 frontier variant at log_w 4 (its w16 wrong bits on the audit) | vs it | error audit / mixed / dense | b16 audit wrong |
|---|---|---|---|---|---|---|---|---|---|
| 3 | `bfv:d3_kt` | ub | 15,502,971 | 308,057 | 13.3 / 1.7 | `dl3:d3c1_mp17_p1d_kt_xca` (4,591 / 5,031) | +6.2% | 2.2e-03 / 2.2e-03 / 4.6e-03 | 5,415 / 5,186 |
| 4 | `bfv:d4x_m4k2_kt1` | ub | 8,357,823 | 132,026 | 24.7 / 3.9 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` (4,146 / 4,756) | +10.1% | 6.2e-04 / 1.3e-03 / 1.9e-03 | 4,676 / 5,172 |
| 5 | `bfv:d5_k2_tp` | u1 | 4,947,297 | 57,126 | 41.8 / 8.9 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` (4,327 / 4,929) | +7.8% | 1.4e-03 / 4.0e-03 / 4.2e-03 | 4,707 / 5,294 |
| 6 | `bfv:d6_m4s4_cp_c5_sl_tp_bt` | ub1 | 3,501,653 | 44,851 | 59.1 / 11.3 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` (4,437 / 5,060) | +5.3% | 1.8e-03 / 3.4e-03 / 4.2e-03 | 5,361 / 5,312 |
| 7 | `bfv:d7_c5_p1dct_bt` | ub | 3,109,316 | 33,466 | 66.5 / 15.2 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` (6,982 / 5,516) | +2.0% | 1.2e-03 / 3.0e-03 / 2.8e-03 | 4,394 / 5,216 |
| 8 | `bfv:d8_g_p1dct` | ub | 2,813,157 | 29,269 | 73.5 / 17.4 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` (6,795 / 5,521) | +2.0% | 1.3e-03 / 2.2e-03 / 2.8e-03 | 4,389 / 5,233 |
| 9 | `c2:sp_col1b_p1d` | ub | 4,139,563 | 23,578 | 50.0 / 21.6 | `c2:sp_col1b_p1d` (1,589 / 3,319) | +0.1% | 1.1e-04 / 1.1e-04 / 1.2e-04 | 2,781 / 4,661 |
| 9 | `sp:col1b` | u | 4,252,074 | 23,010 | 48.6 / 22.1 | `c2:sp_col1b_p1d` (1,589 / 3,319) | +2.8% | 3.6e-05 / 3.1e-05 / 3.0e-05 | 2,942 / 4,715 |

## Per-strategy verdicts under bf16 weights (w16)

Evidence at log_w 4 (69-message audit, table L) unless marked; "exact" = no non-representable
weight, so w16 = float32 bit for bit. Verdicts: **robust** (exact as built, or made exact at no
or negligible cost), **fixed** (exact after `bf16_units`, cost given), **breaks** (no exact
bf16 form found), **stops paying** (exact, but no longer the best choice in bf16).

| strategy | weights bf16 cannot hold | w16 as built | fix | verdict |
|---|---|---|---|---|
| SwiGLU scales c*q, 4/q, 1/(c*q*v), step offsets, norm | none at c = 4, q = 8 (all powers of 2) | exact | - | **robust**; needs 4/q dyadic: q = 12 gives 1,809 wrong bits even on `glu_chi_iota` |
| threshold baseline (step units) | none (integer weights x 32, offsets 20 / 12) | exact (log_w 4) | - | **robust** |
| S1 glu_xor, clean glu_xor, chi, chi + iota | none (integers, halves) | exact | - | **robust** |
| S2 constants folded into biases | shared constant units: sums of many constants (d3c1 layer 1: outs such as 27.014286 = 1891/70) | d3c1: 2.8 (4,603 wrong) | out split, 2 levels (+2 hidden units per such unit) | **fixed** (d3c1: 4.0e-3 = float32) |
| S2 unit sharing | ratios 6/7, 4/7 (u1), 1/3 (sp, Walsh), -8/7 and 6/7 (shared constants) | u1: 411 / 1,733 wrong; sp: 8 / 562; dl3 d4x 7.8e-3 | rescale (f), 0 cost | **fixed** |
| S2 no re-threshold after pure units | no weights | - | - | **robust** |
| S3 shared column parities, count features, split and direct layouts | none | exact (xs3 direct / split / lazy / lazy4) | - | **robust** |
| S3 lazy chi: lazy4c's 2-unit reduction | value -2/3 - c/3 | 2.2e-2 (bits still right at log_w 4) | rescale alpha = 4/3: max(0, 12 - 4c)(-1/2 - c/4) | **fixed**, 0 cost |
| S3 lazy chi: C5, Z11 | none | exact | - | **robust** |
| S3 fold layout (rp) | none | exact | - | **robust** (d8 point) |
| S3 fused rounds (dl3 d3, d4x) | Walsh sharing 1/3; raw-bit forms (see mp, p1d raw) | d4x 7.8e-3; d3 188 wrong | rescale; raw forms replaced by glu_xor | **fixed** for the layout; d3 +5.8%, d4 +9.7% at log_w 6 |
| S3 Walsh last round (xs4) | none | exact | - | **robust** |
| S3 sparse-first layouts (sp) | none | exact | - | **robust** (d7-d9 sparse ends within 0.3-0.6% of float32) |
| S4 cp | none | exact | - | **robust** |
| S4 u1 | sharing 6/7, 4/7 | 411 / 1,733 wrong | rescale, 0 cost | **fixed** (d7, d8 points) |
| S4 sp pairs (round 1) | sharing 1/3 | 5 / 245 wrong (d7_cp_sp_c5) | rescale, 0 cost | **fixed** |
| S4 m3, m4s4, lz5, k2, k3, dd | none (multipliers 2^k, 5/32) | exact | - | **robust** (lz5 stays float32-tight at d7, as in float32) |
| S4 sl | none | exact | - | **robust** |
| S5 mp / mpx / mp17 (MINPAR7, MIN_PARITY[9] on raw bits) | products 25/6, 47/3, 5/3; irrational lam-scaled forms | 2,842 / 4,174 wrong (d6_m3_mp_cp_c5) | none: gate x value x out is invariant; MINPAR7's family has no dyadic member (q <= 32) | **breaks**; dropping it costs ~1.0M at d7/d8 (with `_ra`), more at d3-d6 |
| S5 mpc (MIN_PARITY on counts) | irrational (-24.71, 205.70, ...) | 1,077 / 1,910 wrong (split_first_middle_mpc) | `pm`: the p1d form of range n + 1, same unit count, + split bias | **fixed** (d5 point) |
| S5 p1d on counts (p1dc/p1dct, bt, p1dx/p1dxt, P1DK kt/kt1, sp_p1d) | slopes 3 -+ 2 sqrt 2, irrational biases | 1e15-1e24 errors | rescale slopes to 8, 1/8 + 24-bit bias split (2 features in each layer that needs them) | **fixed**: still saves 1.13M (d8), 1.63M (d7), 1.55M (d6), 3.74M (d5), 4.10M (d4) against glu_xor (table F3b); at the sparse end it **stops paying** (the split costs nonzeros: `sp:col1b` 91,778 vs `c2:sp_col1b_p1d` 94,026) |
| S5 p1d on raw bits (`_r`, `_ra`, dl3 P1D, `xca`) | irrational biases in layer 1 | 1e15 errors | none (no second constant feature in the input) | **breaks** |
| S5 flat forms (`pf`, `p1dcf`, FORMS_FLAT) | irrational products | 7,014 wrong (d8 `_pf_ra`) | none | **breaks** (was float32-tight anyway) |
| S5 reduction units c5, z11 | none | exact | - | **robust** |
| S6 th (TH1S) | products with 1/5, 1/13 (17/4, -13/10, 26/5, 75/2, -416/5) | 8.9e5 errors | none | **breaks** |
| S6 tp (F1 pool) | 7/3, 70/3, 4/7; 80 units per 1600 bits with product 40/3 | 2,019 / 2,266 wrong | rescale + out split (1 level: +320 units at log_w 6, ~1.2M) | **fixed**: robust in the d5 (79.47M, `u1`) and d6 (56.28M, `ub1`) points; tight with lz5 (as in float32) and with p1d in d5's Walsh layer |
| S6 tr (F13 pool) | irrational (2.389, 3.1256, 7.0167) | 5,140 wrong | none | **breaks** |
| S6 tq (F12 pool) | 992 gate and 2,128 value entries stay non-representable after rescaling | (not run) | none | **breaks** |

## Table L: the strategy ladder at log_w 4 (69-message audit, every run of this avenue)

`fix` is the XOF_BF16 mode (`-`: built as is). Margin = worst |out/BOS - bit|; "(n / m wrong)" =
wrong bits nearest / boolify out of 69 x 168 = 11,592. Rows appear in run order; for a repeated
(variant, fix) the last run is shown. Earlier runs are in `raw_superseded.jsonl`,
`raw_v1_before_constfix.jsonl` (before constant units were rescaled) and `raw_v2_split16.jsonl`
(one-level bias split). `ub` rows of variants not rerun after the split became per-layer carry the
2 features in every layer after the first (a slightly larger size, same numbers otherwise).

| variant | fix | depth | dense | sparse | f32 | w16 | b16 |
|---|---|---|---|---|---|---|---|
| `variants:glu_xor_everywhere` | - | 14 | 29,647,370 | 170,906 | 3.6e-07 | 3.6e-07 | 1.0e+00 (114/189 wrong) |
| `variants:glu_xor_clean_everywhere` | - | 14 | 41,182,570 | 397,706 | 3.6e-07 | 3.6e-07 | 2.0e-02 |
| `variants:glu_chi_unit` | - | 11 | 19,267,085 | 147,773 | 3.6e-07 | 3.6e-07 | 1.0e+00 (1741/1867 wrong) |
| `baseline` | - | 20 | 206,803,140 | 508,772 | 3.6e-07 | 3.6e-07 | 1.0e+00 (3/3 wrong) |
| `variants:glu_chi_iota` | - | 8 | 15,468,800 | 135,446 | 4.8e-07 | 4.8e-07 | 1.0e+00 (2972/3041 wrong) |
| `xs3:direct` | - | 6 | 5,669,725 | 44,101 | 6.7e-05 | 6.7e-05 | 3.3e+00 (3188/4921 wrong) |
| `xs3:split_middle` | - | 7 | 4,232,692 | 37,788 | 9.8e-05 | 9.8e-05 | 1.9e+00 (3048/4815 wrong) |
| `xs3:split_first_middle` | - | 8 | 3,866,923 | 26,907 | 9.8e-05 | 9.8e-05 | 1.9e+00 (3203/4750 wrong) |
| `xs3:lazy_middle` | - | 6 | 4,740,716 | 40,869 | 8.2e-05 | 8.2e-05 | 1.9e+00 (3127/4610 wrong) |
| `xs3:lazy4_middle` | - | 6 | 4,498,556 | 42,005 | 5.6e-04 | 5.6e-04 | 1.5e+02 (3704/4664 wrong) |
| `xs3:lazy4c_middle` | - | 6 | 4,438,076 | 42,005 | 3.1e-04 | 2.2e-02 | 5.5e+01 (3669/4719 wrong) |
| `xs3:split_first_lazy4c` | - | 7 | 4,072,307 | 31,124 | 2.6e-04 | 2.2e-02 | 5.0e+01 (3719/4865 wrong) |
| `xs3:split_first_middle_mpc` | - | 8 | 3,834,043 | 26,827 | 4.8e-04 | 2.3e+00 (1077/1910 wrong) | 1.3e+01 (3423/4730 wrong) |
| `xs3:direct_walsh` | - | 5 | 6,540,022 | 54,858 | 7.3e-05 | 7.3e-05 | 5.9e+00 (3760/4785 wrong) |
| `xs4:d5` | - | 5 | 5,874,881 | 54,490 | 7.5e-05 | 7.5e-05 | 2.0e+00 (2925/4133 wrong) |
| `dl3:d4x` | - | 4 | 8,595,309 | 128,908 | 5.2e-04 | 7.8e-03 | 2.3e+02 (3505/4819 wrong) |
| `xs4:d4` | - | 4 | 11,237,794 | 103,678 | 1.1e-04 | 1.1e-04 | 6.8e+00 (3889/4299 wrong) |
| `sp:base` | - | 8 | 3,999,963 | 23,971 | 9.1e-05 | 9.1e-05 | 2.0e+00 (3258/4878 wrong) |
| `sp:col1b` | - | 9 | 4,252,074 | 23,010 | 3.6e-05 | 3.6e-05 | 1.9e+00 (2942/4715 wrong) |
| `sp:lazy2_l3` | - | 7 | 4,374,947 | 28,572 | 1.0e-04 | 1.0e-04 | 2.5e+00 (3329/4706 wrong) |
| `c2:d8_rp_cp_sp_g` | - | 8 | 2,999,242 | 27,915 | 3.3e-04 | 7.1e-01 (8/562 wrong) | 5.6e+00 (3811/5145 wrong) |
| `c2:d8_rp_cp_sp` | - | 8 | 2,966,362 | 27,835 | 2.5e-03 | 2.3e+00 (656/1882 wrong) | 1.2e+01 (4075/5136 wrong) |
| `c2:d8_rp_cp_u1_g` | - | 8 | 2,966,082 | 28,107 | 4.2e-04 | 1.1e+00 (411/1733 wrong) | 1.4e+01 (4261/5287 wrong) |
| `c2:d8_rp_mp_cp_sp` | - | 8 | 2,915,773 | 28,157 | 6.5e-03 | 1.2e+01 (3731/4932 wrong) | 1.2e+01 (4617/5284 wrong) |
| `c2:d8_rp_m3_mp_cp_sp` | - | 8 | 2,887,180 | 28,463 | 6.5e-03 | 1.2e+01 (3893/4739 wrong) | 3.9e+01 (4801/5216 wrong) |
| `dl3:d3` | - | 3 | 15,852,825 | 305,258 | 2.2e-03 | 7.6e-02 (0/188 wrong) | 5.0e+04 (5099/5126 wrong) |
| `c2:d8_rp_m3_mp_cp_sp_sl` | - | 8 | 2,864,212 | 28,657 | 6.5e-03 | 1.2e+01 (3893/4751 wrong) | 4.0e+01 (4816/5248 wrong) |
| `c2:d8_rp_m4s4_mp_cp_u1_sl` | - | 8 | 2,806,950 | 28,879 | 4.2e-03 | 3.1e+01 (5133/5238 wrong) | 1.1e+02 (5472/5510 wrong) |
| `c2:d8_rp_m4s4_mp_cp_u1_sl_g` | - | 8 | 2,835,670 | 29,047 | 2.2e-03 | 3.1e+01 (4966/5213 wrong) | 1.1e+02 (5327/5548 wrong) |
| `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dc` | - | 8 | 2,763,910 | 29,265 | 2.1e-03 | 3.1e+01 (5018/5266 wrong) | 3.5e+12 (5596/5527 wrong) |
| `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_r` | - | 8 | 2,758,126 | 28,937 | 2.3e-03 | 2.2e+14 (6357/5452 wrong) | 2.7e+15 (6431/5524 wrong) |
| `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` | - | 8 | 2,758,126 | 28,553 | 1.2e-03 | 4.2e+15 (6795/5521 wrong) | 2.5e+19 (6828/5607 wrong) |
| `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra` | - | 8 | 2,729,406 | 28,591 | 5.8e-03 | 4.0e+16 (7014/5537 wrong) | 1.4e+20 (6994/5613 wrong) |
| `c2:d8_rp_cp_u1_g_gf` | - | 8 | 3,022,362 | 28,627 | 4.2e-04 | 1.1e+00 (452/1775 wrong) | 1.3e+01 (4257/5289 wrong) |
| `c2:d7_cp_sp_c5` | - | 7 | 3,345,507 | 30,644 | 4.5e-04 | 7.2e-01 (5/245 wrong) | 7.5e+01 (4482/5135 wrong) |
| `c2:d7_cp_u1_c5` | - | 7 | 3,312,347 | 30,836 | 4.4e-04 | 1.1e+00 (313/1512 wrong) | 2.3e+02 (4863/5258 wrong) |
| `c2:d7_m3_mp_cp_sp_c5` | - | 7 | 3,251,916 | 31,308 | 1.9e-03 | 6.7e+00 (3724/4704 wrong) | 1.3e+02 (5232/5213 wrong) |
| `c2:d7_m3_mp_cp_sp_z11` | - | 7 | 3,216,716 | 31,628 | 1.9e-03 | 6.7e+00 (3715/4749 wrong) | 4.4e+06 (5004/5198 wrong) |
| `c2:d7_m3_mp_cp_sp_z11_lz5` | - | 7 | 3,178,636 | 32,664 | 2.7e-03 | 6.7e+00 (3836/4793 wrong) | 4.5e+06 (5292/5291 wrong) |
| `c2:d7_cp_u1_c5_sl_p1dct_ra_bt` | - | 7 | 3,108,857 | 31,712 | 1.0e-03 | 1.0e+24 (6749/5567 wrong) | 9.6e+25 (6626/5559 wrong) |
| `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` | - | 7 | 3,047,047 | 32,104 | 1.2e-03 | 1.0e+24 (6982/5516 wrong) | 8.6e+25 (6942/5591 wrong) |
| `c2:d6_cp_c5` | - | 6 | 3,911,436 | 41,741 | 4.8e-04 | 4.8e-04 | 7.8e+01 (4019/4993 wrong) |
| `c2:d6_m3_mp_cp_c5` | - | 6 | 3,758,597 | 43,037 | 5.4e-04 | 4.2e+00 (2842/4174 wrong) | 1.2e+02 (4961/5152 wrong) |
| `c2:d6_m3_mp_cp_z11` | - | 6 | 3,723,397 | 43,357 | 5.4e-04 | 4.2e+00 (2829/4189 wrong) | 5.1e+01 (4650/5130 wrong) |
| `c2:d6_m3_mp_cp_z11_lz5` | - | 6 | 3,685,317 | 44,393 | 9.7e-04 | 4.2e+00 (3058/4407 wrong) | 5.7e+01 (5040/5253 wrong) |
| `c2:d6_m3_mp_cp_z11_th` | - | 6 | 3,648,634 | 44,763 | 1.5e-03 | 8.9e+05 (4687/4975 wrong) | 7.7e+10 (6283/5470 wrong) |
| `c2:d6_cp_c5_tp` | - | 6 | 3,630,844 | 40,581 | 6.3e-04 | 3.7e+00 (2019/2266 wrong) | 8.0e+01 (5090/5260 wrong) |
| `c2:d6_cp_c5_tr` | - | 6 | 3,565,311 | 42,151 | 1.6e-03 | 1.1e+01 (5140/5538 wrong) | 7.2e+01 (5792/5544 wrong) |
| `c2:d6_cp_c5_p1dct_ra` | - | 6 | 3,514,583 | 41,289 | 5.3e-04 | 1.6e+00 (3932/5274 wrong) | 1.2e+02 (4784/5305 wrong) |
| `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | - | 6 | 3,325,896 | 43,765 | 2.8e-03 | 8.3e+00 (4437/5060 wrong) | 1.7e+01 (4931/5409 wrong) |
| `c2:d6_lazy_cp_p1dct_tp_bt` | - | 6 | 3,868,191 | 40,145 | 2.8e-03 | 3.7e+00 (2905/3804 wrong) | 4.0e+21 (5571/5272 wrong) |
| `xc:d5_m3k2_mp_cp` | - | 5 | 5,050,357 | 56,970 | 5.4e-04 | 2.1e+01 (2838/4323 wrong) | 1.6e+02 (4233/5145 wrong) |
| `xc:d5_m3k2_mp_cp_mpc` | - | 5 | 5,002,757 | 56,970 | 7.1e-04 | 2.1e+01 (2865/4315 wrong) | 1.6e+02 (4227/5140 wrong) |
| `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` | - | 5 | 4,590,069 | 55,816 | 2.0e-03 | 1.4e+01 (4327/4929 wrong) | 3.0e+01 (4695/5289 wrong) |
| `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | - | 5 | 4,635,770 | 54,376 | 2.5e-03 | 1.9e+01 (3191/4072 wrong) | 1.3e+11 (5973/5265 wrong) |
| `dl3:d4x_m4k2_mpx` | - | 4 | 8,307,894 | 131,997 | 1.1e-03 | 1.3e+00 (3523/4325 wrong) | 2.2e+03 (4415/5305 wrong) |
| `dl3:d4x_m4k2_mpx_wmpc` | - | 4 | 8,261,414 | 131,997 | 1.1e-03 | 2.2e+00 (3593/4352 wrong) | 2.1e+04 (4562/5308 wrong) |
| `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | - | 4 | 7,589,677 | 127,032 | 1.4e-03 | 9.2e+01 (4146/4756 wrong) | 1.3e+03 (4715/5236 wrong) |
| `xc:d4a_m4k3_mpc_p1dxt_tp` | - | 4 | 8,968,864 | 98,550 | 1.7e-03 | 5.0e+01 (3148/3909 wrong) | 2.0e+07 (5916/5315 wrong) |
| `dl3:d3c1_mp17` | - | 3 | 15,568,045 | 303,803 | 2.9e-03 | 9.2e+00 (4603/5022 wrong) | 1.9e+03 (6100/5424 wrong) |
| `dl3:d3c1` | - | 3 | 15,852,825 | 306,182 | 4.0e-03 | 2.8e+00 (4603/5190 wrong) | 3.6e+01 (6004/5385 wrong) |
| `c2:sp_col1b_p1d` | - | 9 | 4,135,865 | 22,932 | 5.5e-05 | 1.1e+00 (1589/3319 wrong) | 2.0e+00 (2823/4689 wrong) |
| `c2:sp_base_p1d` | - | 8 | 3,883,754 | 23,893 | 5.2e-04 | 1.1e+00 (1588/3319 wrong) | 2.0e+00 (3251/4880 wrong) |
| `dl3:d3c1_mp17_p1d_kt_xca` | - | 3 | 14,598,761 | 289,549 | 1.6e-03 | 1.2e+03 (4591/5031 wrong) | 1.5e+03 (5820/5319 wrong) |
| `c2:sp_lazy2_l3_p1d` | - | 7 | 4,226,774 | 28,392 | 1.2e-03 | 1.1e+00 (1741/2891 wrong) | 1.2e+01 (3368/4720 wrong) |
| `bfv:d6_c5_lz5` | u | 6 | 3,750,349 | 44,419 | 1.1e-03 | 1.1e-03 | 1.1e+02 (5109/5329 wrong) |
| `bfv:d5_k2` | u | 5 | 5,154,049 | 56,646 | 8.0e-04 | 8.0e-04 | 2.8e+01 (3758/5102 wrong) |
| `bfv:d8_g` | u | 8 | 2,881,219 | 28,725 | 1.3e-03 | 1.3e-03 | 1.0e+02 (4369/5280 wrong) |
| `bfv:d7_c5` | u | 7 | 3,203,340 | 32,478 | 1.3e-03 | 1.3e-03 | 7.2e+02 (5006/5319 wrong) |
| `bfv:d4x_m4k2` | u | 4 | 8,587,164 | 129,538 | 5.3e-04 | 5.3e-04 | 2.1e+02 (3673/5058 wrong) |
| `c2:d7_cp_sp_c5` | u | 7 | 3,345,507 | 30,644 | 4.3e-04 | 4.3e-04 | 1.4e+02 (4487/5142 wrong) |
| `c2:d8_rp_cp_u1_g` | u | 8 | 2,966,082 | 28,107 | 4.4e-04 | 4.4e-04 | 1.2e+01 (3885/5182 wrong) |
| `xs3:lazy4c_middle` | u | 6 | 4,438,076 | 42,005 | 3.1e-04 | 3.1e-04 | 5.5e+01 (3662/4705 wrong) |
| `bfv:d4a_m4k3` | u | 4 | 10,093,584 | 104,207 | 3.3e-04 | 3.3e-04 | 6.4e+00 (4872/5058 wrong) |
| `dl3:d4x` | u | 4 | 8,595,309 | 128,908 | 5.2e-04 | 5.2e-04 | 2.3e+02 (3505/4819 wrong) |
| `bfv:d3c1` | u | 3 | 15,854,789 | 306,922 | 4.0e-03 | 4.0e-03 | 6.4e+00 (5815/5309 wrong) |
| `bfv:d4x_m4k2_kt` | - | 4 | 8,368,037 | 129,058 | 8.1e-04 | 7.6e+02 (3577/3852 wrong) | - |
| `bfv:d4x_m4k2_kt` | u | 4 | 8,368,037 | 129,058 | 8.9e-04 | 2.3e+03 (4297/3892 wrong) | - |
| `bfv:d8_g_p1dct` | ub | 8 | 2,831,121 | 29,305 | 1.3e-03 | 1.3e-03 | 1.9e+04 (4480/5265 wrong) |
| `bfv:d7_c5_p1dct_bt` | ub | 7 | 3,123,136 | 33,490 | 1.3e-03 | 1.3e-03 | 7.0e+16 (4595/5306 wrong) |
| `bfv:d6_c5_lz5_p1dct_bt` | ub | 6 | 3,674,171 | 45,425 | 2.0e-03 | 2.0e-03 | 3.2e+01 (4367/5335 wrong) |
| `bfv:d5_k2_p1dxt_pm` | ub | 5 | 4,944,054 | 59,140 | 1.3e-03 | 1.3e-03 | 2.3e+02 (4411/5098 wrong) |
| `c2:sp_col1b_p1d` | ub | 9 | 4,156,931 | 23,620 | 9.6e-05 | 9.6e-05 | 1.9e+00 (2805/4768 wrong) |
| `bfv:d4x_m4k2_kt` | ub | 4 | 8,404,751 | 131,308 | 6.2e-04 | 6.2e-04 | 2.1e+03 (4283/5152 wrong) |
| `bfv:d8_pm_p1dct` | ub | 8 | 2,801,921 | 29,743 | 2.4e-03 | 2.4e-03 | 2.9e+03 (4511/5273 wrong) |
| `bfv:d7_c5_lz5_p1dct_bt` | ub | 7 | 3,076,712 | 34,526 | 4.3e-03 | 4.3e-03 | 6.1e+16 (4954/5421 wrong) |
| `c2:sp_base_p1d` | ub | 8 | 3,902,950 | 24,575 | 5.3e-04 | 5.3e-04 | 2.1e+00 (3110/4781 wrong) |
| `c2:sp_lazy2_l3_p1d` | ub | 7 | 4,250,282 | 29,708 | 1.0e-03 | 1.0e-03 | 4.5e+00 (3331/4661 wrong) |
| `bfv:d6_lazy_cp_p1dct_bt` | ub | 6 | 4,171,861 | 42,615 | 1.3e-03 | 1.3e-03 | 5.3e+04 (3431/4916 wrong) |
| `bfv:d4a_m4k3_p1dxt_pm` | ub | 4 | 9,309,248 | 107,852 | 7.2e-04 | 7.2e-04 | 8.4e+03 (4926/5020 wrong) |
| `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tp_bt` | ub | 6 | 3,540,971 | 47,545 | 5.1e-03 | 5.1e-03 | 6.9e+13 (5616/5445 wrong) |
| `bfv:d5_k2_tp_p1dxt_pm` | ub | 5 | 4,810,854 | 61,260 | 4.7e-03 | 4.7e-03 | 1.4e+10 (6207/5360 wrong) |
| `c2:d6_lazy_cp_p1dct_tp_bt` | ub | 6 | 4,038,661 | 44,735 | 2.7e-03 | 2.7e-03 | 1.1e+20 (5337/5286 wrong) |
| `bfv:d4a_m4k3_tp_p1dxt_pm` | ub | 4 | 9,176,048 | 109,972 | 1.7e-03 | 1.7e-03 | 2.8e+06 (5980/5256 wrong) |
| `bfv:d3c1_kt` | ub | 3 | 15,504,939 | 309,721 | 3.9e-03 | 3.9e-03 | 6.8e+03 (5976/5330 wrong) |
| `bfv:d4x_m4k2_kt1` | ub | 4 | 8,357,823 | 132,026 | 6.2e-04 | 6.2e-04 | 5.2e+03 (4676/5172 wrong) |
| `bfv:d3_kt` | ub | 3 | 15,502,971 | 308,057 | 2.2e-03 | 2.2e-03 | 5.1e+04 (5415/5186 wrong) |
| `bfv:d3_kt1` | ub | 3 | 15,070,939 | 309,007 | 2.6e-03 | 2.6e-03 | 4.7e+05 (5745/5137 wrong) |
| `c2:d6_cp_c5_p1dct_tp_bt` | ub1 | 6 | 3,617,253 | 43,351 | 1.5e-03 | 1.5e-03 | 1.1e+14 (5142/5266 wrong) |
| `sp:col1b` | u | 9 | 4,252,074 | 23,010 | 3.6e-05 | 3.6e-05 | 1.9e+00 (2942/4715 wrong) |
| `bfv:d6_m4s4_cp_c5_sl_tp_bt` | ub1 | 6 | 3,513,395 | 44,869 | 1.9e-03 | 1.9e-03 | 1.1e+14 (5390/5330 wrong) |


Non-representable weights, as found by `audit/bf16_weights.py` at log_w 4 (unit terms):
- `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra`: X1 gates 5.828427 / -12.656855 / 22.313709 (p1d), values
  0.150126 = 0.171573 x 7/8, outs 6/7, -4/7 (u1 sharing); X2 gates 46.627419, values 0.686292,
  outs -8/7 (p1d D): 3,630 entries.
- `c2:d6_m3_mp_cp_c5` (mp): gates 6.992794, 4.057944, -2.945386 (MIN_PARITY[9] with lam), values
  -19/72, 4/3, -1/3 (MINPAR7): 4,584 entries.
- `xs3:split_first_middle_mpc` (mpc): MIN_PARITY[11] gates -24.711967, 205.69574, values 0.36846 ...
- `c2:d6_cp_c5_tp`: values -7/3, 70/3, outs 4/7. `c2:d6_m3_mp_cp_z11_th`: values -13/10, -6/5.
- `c2:d7_cp_sp_c5`, `c2:d8_rp_cp_sp_g`: 80 outs of 1/3 (sp's Y1). `dl3:d4x`, `dl3:d3`: outs -+1/3
  in the Walsh layer. `dl3:d3c1`: layer-1 constant outs such as 7.957142, 34.957146.
- `xs3:lazy4c_middle`: 480 values -1/24 (= -2/3 / 16).

## Remaining ideas

- **Layer 1** is where bf16 costs size. A second constant input feature (e.g. BOS/256 next to BOS in
  the message input) would make the p1d raw forms (`_ra`, dl3's P1D) exact with the same bias
  split, recovering most of the d3-d6 gap; that changes the input format, so it was not done.
- A dyadic min-parity for raw bits: only MINPAR7's active pattern was solved (no dyadic member for
  b0 = p/q, q <= 32). Other active patterns for n = 7, 9 might have dyadic members; the quarter-grid
  search (`wave4_bf16w/search_dyadic_minpar.py`) found no consistent knot set at all, so a search
  should solve for the last knot (the consistency condition is quadratic in it), not grid it.
- The F1 pool has 80 units per 1600 bits with product 40/3; a re-derived pool with a dyadic product
  there would save its out split (+320 units, about 1.2M at d6).
- Splitting value rows (units with the same gate carrying the low bits of a non-dyadic value)
  makes any form with exact gates bf16-exact at +1 unit per such unit; it only pays where a form
  saves more units than it has non-dyadic units (not for MINPAR7: 3 of 3 units).
- b16: nothing here addresses activation rounding; the split features and exact weights do not
  help there (every point still fails b16).

## Files

- `repo/experiments/xof_shrink/bf16_units.py`: `rescale_units`, `split_bias` (the method).
- `repo/experiments/xof_shrink/xofbench.py`: `XOF_BF16` = `u`, `u1`, `ub`, `ub1` in `build_layers`.
- `repo/experiments/xof_shrink/bfv.py`: the bf16 variants (layouts without the breaking forms,
  `pm`).
- `repo/experiments/xof_shrink/audit/bf16_weights.py`: non-representable weights per layer.
- `repo/experiments/xof_shrink/wave4_bf16w/`: `mp7_family.py` (MINPAR7 family scan),
  `search_dyadic_minpar.py` (grid search).
- `q_check.py` (this folder): w16 with steepness q (q = 12 breaks).
- `raw_p1d_err.py` (this folder): lattice error of bf16-rounded p1d cores on raw bits.
- `refs/`: reference sets (log_w 4 and 6; `w3_audit.pt` for debugging).
- `raw.jsonl` -> `results.jsonl`: every bf16_check run (with `ref`, `bf` = XOF_BF16), plus harness
  runs (`harness.jsonl`, `tool: xofbench`), the representability certificates (`wan/`), and
  `q_check.jsonl`.
- Scripts used to run: `bfrun.sh` (one bf16_check run -> `raw.jsonl`), `sched.sh` / `sched_h.sh` /
  `sched_w.sh` (queues capped at 5 processes), `hrun.sh` (harness), `wan.sh` (certificate),
  `make_tables.py` + `make_notes.py` (this file from `NOTES_template.md`), `export.py`
  (`results.jsonl`), `summ.py`, `dbg_cmp.py` (plain vs rescaled builds per layer in float64 and
  float32), `dbg_split.py`, `dbg_units.py`.

## Reproduce

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
A=$S/xof4/av/bf16-weights; R=$A/repo; E=$R/experiments/xof_shrink
export OMP_NUM_THREADS=2 PYTHONPATH=$R/src:$E:$E/depth_low/lib/python
# the float32 d8 point with bf16 weights, and its bf16 version (log_w 4, seconds)
$S/venv/bin/python $E/audit/bf16_check.py $A/refs/w4_audit.pt c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra --modes f32,w16
XOF_BF16=ub $S/venv/bin/python $E/audit/bf16_check.py $A/refs/w4_audit.pt bfv:d8_g_p1dct --modes f32,w16
XOF_BF16=ub $S/venv/bin/python $E/audit/bf16_weights.py 4 3 bfv:d8_g_p1dct | head -1   # nonrep 0
# log_w 6 (under 3 minutes each)
XOF_BF16=ub $S/venv/bin/python $E/audit/bf16_check.py $A/refs/w6_dense.pt bfv:d8_g_p1dct --modes f32,w16
XOF_BF16=ub $S/venv/bin/python $E/xofbench.py --log-w 6 --depth 3 --variant bfv:d8_g_p1dct
```

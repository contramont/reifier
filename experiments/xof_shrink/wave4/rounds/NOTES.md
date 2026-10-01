# Wave 4, avenue `rounds`: which size reductions hold with R Keccak rounds per XOF step

Question: every wave-1..3 builder assumed one Keccak round per XOF step (`k.n = 1`). Which
strategies still shrink the circuit, exactly and float32-robustly, when each step runs R rounds
(`k.n = R`, R = 2, 3, 4, and 24 as a stretch)?

All sizes below come from the harness code (`xofbench.build_layers` / `metrics`, the same code
path `xofbench.py` runs; the frontier points were also re-run through `xofbench.py` itself, see
`results.jsonl`, kind `harness`). "Robust" means the brief's criterion: 0 wrong bits (nearest)
and 0 boolify errors, and worst |out/BOS - bit| <= 0.01 in float32 on the `ref_gen` audit (9 edge
cases + 60 random) AND on stress sets of >= 500 messages with dense ones (mixed `4 12` = 672
messages at densities 0.005-0.995 + lane/z patterns; dense `4 10` = 520 messages at 0.85-0.99).
Each run also checks the variant's eager function against the reference on the 9 edge cases.
Anything not measured that way is labelled an estimate.

## Summary

- `xs3.build` (and so every xs3 / c2 variant) now handles R rounds per XOF step: per-round round
  constants, digests only after every R-th round, "middle" rounds that keep the full state. Exact
  against the reference `xof` in every configuration run; R = 1 sizes are bit-for-bit unchanged.
- **Per-round cost.** A middle round costs 30.8M dense at log_w 6 with split + NOP layers (3
  layers), against 993.7M (6 layers) for a threshold round: 1/32. Direct rounds cost 1/18.5 but
  take 2 layers; lazy-chi rounds cost 1/12 and stop paying. So the ratio to the baseline falls from
  115x (log_w 6, 1 step x 2 rounds) through 63x (1 x 4) and 46x (3 x 2) to 38x (3 x 4), toward ~32x.
- **What breaks with more rounds** (float32, criterion <= 0.01):
  - cp in every X layer (float32 rounding, T >= 6), P1DC's D form in every X layer (silu
    approximation, T >= 4), the 1-round frontier points as they are (T >= 4), u1 together with cp
    in the last X (T >= 6), u1 alone and P1DC in the last X at log_w 6 (T = 12), m3 digest packing (T = 12);
  - the plain split layout itself after ~20 rounds: its Y unit has slope 2 at E = 2, and
    24 rounds give an error of 1.12 (98 wrong bits) in float32, 1.4e-8 in float64: rounding
    amplified round after round.
- **New: a flat Y** (one extra glu_xor unit per bit, every 2nd-4th split round) keeps split
  layouts robust for 24 and 72 rounds (log_w 3: 11.42M = 32.8x for 1 x 24, 39.23M = 30.9x for
  3 x 24). The direct layout is flat by itself (2e-5 after 24 rounds) and gives the depth-minimal points.
- **What holds:** NOP, the fold, s_last, cp in the last X only, mp / ra / sp on round 1, m4 + s4
  digest packing, the lazy4c + z11 end. u1 and P1DC in the last X hold to T = 6 at log_w 6 but
  fail at T = 12 (4.6e-2, 3.7e-2).
- **bf16 weights (w16).** Only flat forms survive. The glu_xor-only family (`W_*`: NOP, fold,
  s_last, m4 + s4, cp in the last X) is w16-correct at +1.7-2.3% of the float32 frontier's dense for
  3 steps (log_w 6, 3 x 2: 143.7M vs 141.4M; 3 x 4: 345.2M vs 338.7M), and +14-24% for 1 step
  (1 x 4: 73.6M vs 64.5M; 1 x 2: 22.6M vs 18.2M), where the round-1 tricks weigh more. Each
  of mp, ra, u1 and sp breaks w16 (errors 7-8). Flat circuits stay w16-correct for 24 and 72 rounds
  (log_w 3: `W_rp_nop_sl_m4_yf2` 12.16M and 40.70M, w16 5.4e-5 and 9.7e-4). Full bf16
  (b16) fails for everything, the baseline included.

## 1. Method: xs3 for R rounds per step

`experiments/xof_shrink/xs3.py` (`build`) now runs the T = steps * R rounds in a row:
- `rcs = k.get_round_constants()` (R constants); round r uses `rcs[r % R]`, which is exactly what
  `xof` does (every step calls `hash_state`, which applies the same R rounds);
- a closure `cur` holds the round of the chi whose iota flags / digest the current layer handles:
  the chi layer of round r, the X layer that copies its bits (round r), the lazy chi of round r+1,
  and the last theta / last chi (round T-1, for `s_last` and `chi_last`);
- only rounds with (r + 1) % R == 0 emit a digest (`dpos()`); the other ("middle") rounds keep
  the full 5x5xw state and only carry the earlier digests;
- `make(kinds_fn)` asks `kinds_fn` for one theta layout per ROUND (T entries), so every existing
  xs3 / c2 variant is now defined for any R (a variant name like `d8_...` keeps its 1-round
  meaning; its depth grows with T);
- R = 1 is unchanged (T = steps, every round emits): the 1-round variants give the same sizes.

Exactness: the construction per round is the 1-round one (same units, same lattice arguments),
only the round constant and the set of emitted digests change. Verified against the unmodified
reference `xof` with R rounds (`ref_gen.py ... R`, `ref_stress.py --rounds R`, fresh processes):
the eager function of every variant matched on all edge cases in every configuration run
(`eager_bad = 0` in all 300+ runs), and the networks matched wherever the float32 error allowed.

New switches (all default off, so nothing changes for R = 1 unless used):
- `xs3.CPK` / `with_cpk(v, K)`: column packing (cp) only in the X layers of the last K rounds
  (the last X layer is round T-2, so K = 2 means "cp only in the last X", the 1-round placement);
- `xs3.P1K` / `with_p1k(v, K)`: P1DC's irrational-knot D form only in the X layers of the last K rounds;
- `xs3.YFLAT` / `with_yflat(v, every)`: a flat split Y (section 3) in the split rounds r >= 1
  with r % every == 0 (round 0's Y, on raw message bits, never needs it);
- `rv.py`: the layouts per round (`sfm`, `sfl4`, `sfl4c`, `dsm`, `alld`, `dj(j)`), the
  combinations `F_*` (`_combo`), T = 2 variants `T2_*`.

Tools: `audit/rounds_eval.py` compiles a variant once and checks it against several reference
sets (float32, optionally float64 / bf16 modes, per-digest-step margins); `ev.sh` wraps it and
appends to `results_raw.jsonl`; `summ.py` prints the per-configuration frontiers.

## 2. Per-round marginal cost of a middle round (measured)

1 XOF step of T rounds, T = 3..7 at log_w 4 and T = 3..5 at log_w 6 (harness, `marginal.jsonl`;
no digest carries in a 1-step circuit, so the difference is exactly one middle round). The
threshold baseline adds 6 layers per round: +62.17M dense per round at log_w 4
(131.24M / 193.41M / 255.59M at R = 2 / 3 / 4) and +993.65M at log_w 6 (2097.7M -> 4085.0M for R = 2 -> 4).

| layout of a middle round | layers | dense, log_w 4 | sparse, log_w 4 | dense, log_w 6 | sparse, log_w 6 | per-round ratio to baseline (dense) | float32 over many rounds |
|---|---|---|---|---|---|---|---|
| baseline (threshold gates) | 6 | 62.17M | 164K | 993.65M | 656K | 1 | flat (2e-7) |
| direct (`xs3:direct`, `rv:F_alld_ra`): chi emits 11-term counts, theta = parity (6 units/bit) | 2 | 3.374M | 17.7K | 53.82M | 70.6K | 18.4x / 18.5x | flat: 2.4e-5 after 24 rounds |
| split (`xs3:split_first_middle`): chi, X (E = a + D, copy + 5 D units per column pair), Y ([E == 1]) | 3 | 2.094M | 11.4K | 33.34M | 45.3K | 29.7x / 29.8x | grows: fails after ~20 rounds (section 3) |
| split + NOP (X gates on the 10 chi bits, no P feature) | 3 | 1.933M | 14.8K | 30.77M | 59.0K | 32.2x / 32.3x | as split |
| split + cp (column packing in this X) | 3 | 1.612M | 11.9K | | | 38.6x | breaks: 0.2 at T = 6 (w4) |
| split + flat Y (`YFLAT`, +1 unit per bit) | 3 | 2.575M | 13.0K | 41.02M | 51.7K | 24.1x / 24.2x | flat |
| split + NOP, flat Y every 2nd round (`_yf2`) | 3 | 2.174M (avg of 1.933 / 2.414) | 15.6K | | | 28.6x | flat enough for 24 rounds |
| lazy4c pair (X, lazy chi, then a direct theta on range-32 counts) | 2 per round | 5.235M per round (10.47M per 2) | 27.3K | | | 11.9x | ok (1e-3) |
| lazy4 pair | 2 per round | 6.583M per round | 30.3K | | | 9.4x | ok |
| fold (`rp`) | only before the LAST theta (one extra narrow layer); not a middle-round layout | | | | | | ok |

Round-1 tricks on raw message bits (u1, sp, mp, p1dr/`ra`, theta pool) and last-round tricks
(`sl`, `s4`, `rp`, `mpc`, `c5`/`z11`/`lz5`, `bt`) do not change the per-round cost: they save a
fixed amount per circuit (section 5). What applies to every middle round: the X/Y/chi layout, NOP,
cp, the D form (P1DC), the flat Y.

So a middle round costs 1/32 of a baseline round (split + NOP), and the ratio of an R-round
circuit to its baseline drifts from the 1-round ratio (75x at log_w 6, 3 steps) toward ~32x as R grows.

## 3. Float32 error growth with the number of rounds, and the flat Y

**What grows.** With one round per step the circuit has 3 rounds; with R rounds it has 3R (or R
for one step), and every sloped unit form is followed by more rounds that can amplify its error.
The float64 runs separate the two error sources: silu approximation (structural, same in float64)
and float32 rounding.

Worst error (max over all sets) against the number of rounds T = steps x R:

| layout | T = 2-4 | T = 6 | T = 9 | T = 12 | T = 18 | T = 24 |
|---|---|---|---|---|---|---|
| direct, glu_xor only (`xs3:direct`) | 4e-5..2e-4 | 1.0e-4 | 6.4e-5 | 4.0e-5 | 1.8e-5 (w3, 39 msgs) | 2.4e-5 (w3) |
| split, glu_xor only (`xs3:split_first_middle`) | 1.1e-4..2.5e-4 | 4.8e-4 | 7.9e-4 | 1.05e-3 | 3.7e-4 (w3, 39 msgs) | **1.12, 98 wrong bits** (w3; float64 1.4e-8) |
| split + flat Y everywhere (`rv:sfm_yf`) | | | | | 1.1e-5 (w3, 39 msgs) | |
| frontier split layout, flat Y every 2nd split round (`rv:F_rp_nop_ra_yf2`) | | | | | | 4.0e-5 (w3) |
| cp in every X (`xs3:split_first_middle_cp`) | 4.2e-4 (T=3), 7.8e-4 (T=4) | **0.20** (float64: 1.0e-6) | | | | |
| P1DC D form + cp in every X (`c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra`) | 6.2e-4 (T=3), **1.6e-2** (T=4) | **1.1, float64 0.65** | | | | |

(log_w 4 unless marked; T = 6, 9, 12 are 3 steps x R = 2, 3, 4.)

**Mechanism.**
- cp breaks by float32 rounding (float64 error 1e-6 at T = 6, float32 0.2): the X decoders read
  packed features c/8 and p/4, so a feature's rounding error enters the decoded bits multiplied by
  8 (or 4) and by the decoder slopes (up to 2) in every round that uses cp.
- P1DC's irrational-knot D form breaks structurally (float64 0.65 at T = 6): its knots sit
  0.17 from the integers, where silu(32 g)/32 is still 7e-4 off relu, and the form's slope at the
  lattice points amplifies it round after round.
- The plain split layout itself is not flat: its Y unit [E == 1] = max(0, E)(2 - E) (E = a + D in
  {0, 1, 2}) has slope -2 at E = 2 (and +1 at E = 0, from silu'(0) = 1/2). The error of a chi bit
  goes through the X copy (slope 1) and Y (slope up to 2) and the next chi (slopes 1-2) each
  round. Harmless for 3-12 rounds, it runs away after ~20: 1.12 at T = 24 in float32 (float64:
  1.4e-8 on the same 1261 messages), i.e. float32 rounding amplified by the slopes.
- The direct layout is flat: its theta is glu_xor on the 11-term count, whose derivative is 0 at
  every lattice point s >= 1 (each knot's half slope cancels the parabola's), so chi's errors are
  squashed every round. Its error stays at 2e-5 for 24 rounds.

**The flat Y (new, `xs3.YFLAT`).** Y = max(0, E)(2 - E) + 4 max(0, E - 2): the added unit is 0
on E in {0, 1, 2}, and at E = 2 its knot's half slope (4 x 1/2 = 2) cancels the -2, so Y is flat
at E = 1 and E = 2 (glu_xor's own form for a count range 3). It costs one unit per bit
(+0.48M per round at log_w 4, +7.69M at log_w 6: split 2.094M / 33.34M -> flat-Y split 2.575M / 41.02M), and one flat Y every 2nd to 4th split round is
enough for 24 rounds (log_w 3, T = 24, `F_rp_nop_ra_u1_cp2_yfN`: N = 2 / 3 / 4 give 4.1e-5 / 1.2e-4 / 1.3e-3, 12.03M / 11.67M / 11.42M;
N = 6 fails with 1.12 and 24 wrong bits). It does not rescue cp in every X
(`rv:F_rp_ra_u1_cpall_yf1`: 1.4e-2 at T = 6): cp's error enters through the decoders, not Y.


## 4. Robust frontiers

Per configuration: the robust Pareto points over (depth, dense) in float32, with ratios to the
threshold baseline of the same configuration (grid baselines from `xof4/baselines.jsonl`; log_w 6
x 3 steps x 2 / 4 rounds and log_w 3 x 1 step x 24 rounds built here, `baselines_own.jsonl`).
Worst errors are per set: ref_gen audit (69 messages) / mixed stress (672) / dense stress (520).
The "bf16 weights (w16)" tables list the points that are also correct with every weight rounded
to bfloat16 (float32 activations; section 6). Every point: 0 wrong bits, 0 boolify errors, eager
function equal to the reference.

How the layouts map to depth for T = steps x R rounds:
- `F_rp_*` (depth 3T - 1): round 0 split (X1, Y1 on raw bits: mp, ra, optionally u1), split middle
  rounds (NOP), a lazy4 X + lazy chi for round T - 2, the fold (rp), the last theta (s_last) and chi;
- `F_d0_*` / `F_l4c_*` (3T - 2): round 0 direct (one layer), or the lazy4c + z11 end;
- `F_dj_*` (3T - 2 - j): j + 1 direct rounds first (2 layers each), then split;
- `F_alld_ra` (2T): every round direct (flat, the depth-minimal layout);
- `T2_*` (T = 2): round 0 split (5 layers) or direct (4 layers), then the direct last round;
- suffixes: `_cp2` cp in the last X only, `_p1` P1DC's D form in the last X only, `_u1` round-1 u
  pairs, `_yfN` a flat Y every N-th split round, `_m2` digests in pairs instead of m4 + s4.


**log_w 4, 1 step x 2 rounds (T = 2)** - threshold baseline: depth 14, dense 131,240,010, sparse 334,762

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 4 | `rv:T2_direct_ra_sl_mpc` | 1,450,906 | 25,064 | 3.5 / 90.5 / 13.4 | 7.3e-04 / 1.3e-03 / 1.2e-03 | not run |
| 5 | `rv:T2_split_u1_ra_sl_mpc` | 1,124,457 | 14,439 | 2.8 / 116.7 / 23.2 | 9.7e-04 / 1.7e-03 / 2.4e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 4 | `rv:W_alld_sl_m4` | 1,792,399 | 25,582 | 73.2 | 9.3e-05 | 9.4e-05 |
| 5 | `rv:W_T2_split_sl` | 1,379,950 | 14,621 | 95.1 | 2.4e-04 | 2.5e-04 |

**log_w 6, 1 step x 2 rounds (T = 2)** - threshold baseline: depth 14, dense 2,097,727,098, sparse 1,338,874

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 4 | `rv:T2_direct_ra_sl_mpc` | 24,097,954 | 103,496 | 3.5 / 87.1 / 12.9 | 9.3e-04 / 9.3e-04 / 1.2e-03 | not run |
| 5 | `rv:T2_split_u1_ra_sl_mpc` | 18,248,721 | 58,167 | 2.8 / 115.0 / 23.0 | 2.3e-03 / 3.1e-03 / 4.1e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 4 | `rv:W_alld_sl_m4` | 29,838,919 | 105,622 | 70.3 | 3.3e-04 | 2.1e-04 |
| 5 | `rv:W_T2_split_sl` | 22,555,174 | 58,901 | 93.0 | 4.0e-04 | 2.7e-04 |

**log_w 4, 1 step x 3 rounds (T = 3)** - threshold baseline: depth 20, dense 193,413,652, sparse 498,804

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 2,843,243 | 40,651 | 3.3 / 68.0 / 12.3 | 1.0e-03 / 8.9e-04 / 1.6e-03 | not run |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` | 2,516,794 | 30,026 | 2.9 / 76.8 / 16.6 | 1.6e-03 / 3.3e-03 / 5.1e-03 | not run |
| 8 | `rv:F_rp_nop_ra_u1_cp2_p1` | 2,349,281 | 26,310 | 2.5 / 82.3 / 19.0 | 1.3e-04 / 6.2e-04 / 5.2e-04 | not run |

**log_w 4, 1 step x 4 rounds (T = 4)** - threshold baseline: depth 26, dense 255,587,294, sparse 662,846

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 8 | `rv:F_alld_ra` | 8,216,454 | 60,557 | 3.2 / 31.1 / 10.9 | 2.6e-05 / 5.5e-05 / 2.2e-05 | not run |
| 9 | `rv:F_d1_rp_nop_ra_cp2` | 6,120,384 | 54,704 | 2.9 / 41.8 / 12.1 | 2.6e-05 / 4.4e-05 / 4.1e-05 | not run |
| 10 | `rv:F_l4c_z11_nop_ra_u1_cp2` | 4,520,655 | 45,359 | 2.6 / 56.5 / 14.6 | 2.6e-04 / 6.4e-04 / 7.5e-04 | not run |
| 11 | `rv:F_rp_ra_u1_cpall` | 4,032,022 | 38,326 | 2.4 / 63.4 / 17.3 | 2.8e-04 / 1.1e-03 / 2.6e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 8 | `rv:W_alld_sl_m4` | 8,540,427 | 60,907 | 29.9 | 2.7e-05 | 4.2e-05 |
| 9 | `rv:W_d1_rp_nop_sl_m4` | 6,765,797 | 57,934 | 37.8 | 4.6e-05 | 6.7e-05 |
| 11 | `rv:W_rp_nop_sl_m4` | 4,912,555 | 44,100 | 52.0 | 4.5e-05 | 4.8e-05 |

**log_w 6, 1 step x 4 rounds (T = 4)** - threshold baseline: depth 26, dense 4,085,035,982, sparse 2,650,958

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 8 | `rv:F_alld_ra` | 132,007,422 | 245,247 | 3.2 / 30.9 / 10.8 | 2.1e-05 / 3.2e-05 / 3.4e-05 | not run |
| 9 | `rv:F_d1_rp_nop_ra_cp2` | 98,509,512 | 221,834 | 2.9 / 41.5 / 12.0 | 3.0e-05 / 5.7e-05 / 5.1e-05 | not run |
| 10 | `rv:F_l4c_z11_nop_ra_u1_cp2` | 72,283,959 | 181,625 | 2.6 / 56.5 / 14.6 | 3.7e-04 / 1.2e-03 / 9.0e-04 | not run |
| 11 | `rv:F_rp_ra_u1_cpall` | 64,492,606 | 153,472 | 2.4 / 63.3 / 17.3 | 1.5e-03 / 8.0e-03 / 5.0e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 8 | `rv:W_alld_sl_m4` | 137,470,947 | 246,701 | 29.7 | 3.2e-05 | 4.5e-05 |
| 11 | `rv:W_rp_nop_sl_m4_cp2` | 73,646,099 | 165,054 | 55.5 | 8.1e-05 | 2.8e-04 |

**log_w 4, 3 steps x 2 rounds (T = 6)** - threshold baseline: depth 38, dense 408,416,850, sparse 1,009,970

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 12 | `rv:F_alld_ra` | 15,756,633 | 98,246 | 3.2 / 25.9 / 10.3 | 1.9e-04 / 3.0e-04 / 5.3e-04 | fails (103425 wrong) |
| 15 | `rv:F_d1_rp_nop_ra` | 10,851,049 | 89,619 | 2.5 / 37.6 / 11.3 | 2.9e-04 / 5.2e-04 / 3.8e-04 | not run |
| 16 | `rv:F_d0_rp_nop_ra_cp2` | 9,082,096 | 83,922 | 2.4 / 45.0 / 12.0 | 2.4e-03 / 4.8e-03 / 5.3e-03 | not run |
| 17 | `rv:F_rp_nop_ra_cp2_p1` | 8,852,967 | 73,139 | 2.2 / 46.1 / 13.8 | 1.9e-03 / 3.7e-03 / 7.0e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 12 | `rv:W_alld_sl_m4` | 16,080,606 | 98,596 | 25.4 | 5.4e-04 | 5.4e-04 |
| 16 | `rv:W_d0_rp_nop_sl_m4` | 9,734,229 | 87,096 | 42.0 | 7.7e-04 | 8.2e-04 |
| 17 | `rv:W_rp_nop_sl_m4_cp2` | 8,993,620 | 73,311 | 45.4 | 6.8e-03 | 7.0e-03 |

**log_w 6, 3 steps x 2 rounds (T = 6)** - threshold baseline: depth 38, dense 6,527,735,970, sparse 4,039,202

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 12 | `rv:F_alld_ra` | 252,273,501 | 395,190 | 3.2 / 25.9 / 10.2 | 2.5e-04 / 7.3e-04 / 5.0e-04 | not run |
| 14 | `rv:F_d2_rp_nop_ra` | 198,118,638 | 372,206 | 2.7 / 32.9 / 10.9 | 2.6e-04 / 7.3e-04 / 5.0e-04 | not run |
| 15 | `rv:F_d1_rp_nop_ra` | 173,821,045 | 360,693 | 2.5 / 37.6 / 11.2 | 3.1e-04 / 6.4e-04 / 1.7e-03 | not run |
| 16 | `rv:F_d0_rp_nop_ra_cp2` | 145,544,572 | 337,884 | 2.4 / 44.9 / 12.0 | 1.6e-03 / 4.1e-03 / 4.9e-03 | not run |
| 17 | `rv:F_rp_nop_ra_cp2_p1` | 141,357,507 | 291,943 | 2.2 / 46.2 / 13.8 | 1.7e-03 / 4.9e-03 / 4.8e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 16 | `rv:W_d0_rp_nop_sl_m4` | 156,241,377 | 350,634 | 41.8 | 9.8e-04 | 3.2e-03 |
| 17 | `rv:W_rp_nop_sl_m4_cp2` | 143,724,352 | 292,617 | 45.4 | 6.6e-03 | 6.9e-03 |

**log_w 4, 3 steps x 3 rounds (T = 9)** - threshold baseline: depth 56, dense 610,030,560, sparse 1,511,168

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 18 | `rv:F_alld_ra` | 26,338,995 | 151,523 | 3.1 / 23.2 / 10.0 | 1.5e-04 / 2.5e-04 / 3.4e-04 | not run |
| 23 | `rv:F_d2_rp_nop_ra` | 18,454,413 | 137,338 | 2.4 / 33.1 / 11.0 | 3.6e-04 / 6.8e-04 / 6.1e-04 | not run |
| 24 | `rv:F_d1_rp_nop_ra` | 17,013,620 | 134,465 | 2.3 / 35.9 / 11.2 | 6.4e-04 / 1.2e-03 / 1.1e-03 | not run |
| 25 | `rv:F_l4c_z11_nop_ra_u1_cp2` | 15,139,099 | 122,128 | 2.2 / 40.3 / 12.4 | 4.9e-03 / 4.7e-03 / 4.7e-03 | not run |
| 26 | `rv:F_rp_nop_ra_cp2` | 15,083,938 | 118,031 | 2.2 / 40.4 / 12.8 | 9.1e-04 / 1.4e-03 / 2.6e-03 | not run |

**log_w 4, 3 steps x 4 rounds (T = 12)** - threshold baseline: depth 74, dense 811,644,270, sparse 2,012,366

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 24 | `rv:F_alld_ra` | 36,921,357 | 204,893 | 3.1 / 22.0 / 9.8 | 2.3e-04 / 3.1e-04 / 2.0e-04 | not run |
| 30 | `rv:F_d4_rp_nop_ra` | 27,576,942 | 188,003 | 2.5 / 29.4 / 10.7 | 4.9e-04 / 6.1e-04 / 6.3e-04 | not run |
| 32 | `rv:F_d2_rp_nop_ra` | 24,616,956 | 182,257 | 2.3 / 33.0 / 11.0 | 1.0e-03 / 1.0e-03 / 1.1e-03 | not run |
| 33 | `rv:F_d1_rp_nop_ra` | 23,176,163 | 179,384 | 2.2 / 35.0 / 11.2 | 8.6e-04 / 2.5e-03 / 1.9e-03 | not run |
| 34 | `rv:F_d0_rp_nop_ra_cp2` | 21,400,490 | 173,631 | 2.2 / 37.9 / 11.6 | 1.3e-03 / 2.2e-03 / 2.9e-03 | not run |
| 35 | `rv:F_rp_nop_ra_cp2_p1` | 21,169,121 | 162,859 | 2.1 / 38.3 / 12.4 | 1.5e-03 / 6.1e-03 / 6.9e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 24 | `rv:W_alld_sl_m4` | 37,245,330 | 205,243 | 21.8 | 2.9e-04 | 4.3e-04 |
| 34 | `rv:W_d0_rp_nop_sl_m4` | 22,059,343 | 176,861 | 36.8 | 4.5e-03 | 2.2e-03 |
| 35 | `rv:W_rp_nop_sl_m4` | 21,646,894 | 165,900 | 37.5 | 4.9e-03 | 5.2e-03 |

**log_w 6, 3 steps x 4 rounds (T = 12)** - threshold baseline: depth 74, dense 12,972,317,214, sparse 8,048,030

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 24 | `rv:F_alld_ra` | 589,879,665 | 821,115 | 3.1 / 22.0 / 9.8 | 2.6e-04 / 6.6e-04 / 2.8e-04 | not run |
| 32 | `rv:F_d2_rp_nop_ra` | 393,083,196 | 730,423 | 2.3 / 33.0 / 11.0 | 1.0e-03 / 1.8e-03 / 2.7e-03 | not run |
| 33 | `rv:F_d1_rp_nop_ra` | 370,040,003 | 718,910 | 2.2 / 35.1 / 11.2 | 8.8e-04 / 3.4e-03 / 2.2e-03 | not run |
| 34 | `rv:F_d0_rp_nop_ra_cp2` | 341,656,010 | 695,877 | 2.2 / 38.0 / 11.6 | 1.6e-03 / 5.4e-03 / 7.7e-03 | not run |
| 35 | `rv:F_rp_nop_ra_cp2` | 338,667,985 | 650,300 | 2.1 / 38.3 / 12.4 | 1.4e-03 / 3.9e-03 / 6.2e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 35 | `rv:W_rp_nop_sl_m4` | 345,176,590 | 662,130 | 37.6 | 3.2e-03 | 5.6e-03 |

**log_w 3, 1 step x 24 rounds (T = 24)** - threshold baseline: depth 144, dense 374,839,360, sparse 1,968,720

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 48 | `rv:F_alld_ra` | 18,983,546 | 207,184 | 3.0 / 19.7 / 9.5 | 2.2e-05 / 2.3e-05 / 2.2e-05 | not run |
| 70 | `rv:F_l4c_z11_nop_ra_u1_cp2_yf2` | 12,069,591 | 179,379 | 2.1 / 31.1 / 11.0 | 2.3e-04 / 4.6e-04 / 3.5e-04 | not run |
| 71 | `rv:F_rp_nop_ra_u1_cp2_yf4` | 11,424,438 | 173,306 | 2.0 / 32.8 / 11.4 | 1.5e-04 / 1.3e-03 / 9.2e-04 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 48 | `rv:W_alld_sl_m4` | 19,058,863 | 207,350 | 19.7 | 2.8e-05 | 2.6e-05 |
| 71 | `rv:W_rp_nop_sl_m4_yf2` | 12,163,187 | 178,752 | 30.8 | 5.4e-05 | 4.6e-05 |

**log_w 3, 3 steps x 24 rounds (T = 72)** - threshold baseline: depth 428, dense 1,210,834,652, sparse 6,007,308

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 144 | `rv:F_alld_ra` | 62,375,478 | 637,638 | 3.0 / 19.4 / 9.4 | 1.8e-04 / 2.5e-04 / 3.3e-04 | not run |
| 215 | `rv:F_rp_nop_ra_cp2_yf3` | 39,230,940 | 551,256 | 2.0 / 30.9 / 10.9 | 1.4e-03 / 1.0e-03 / 2.7e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 215 | `rv:W_rp_nop_sl_m4_yf2` | 40,697,601 | 561,526 | 29.8 | 9.7e-04 | 6.1e-04 |

**Margin notes.** Points with a worst error above 5e-3 pass the criterion but are closer to it:
- log_w 6, 1 x 4, depth 11: `F_rp_ra_u1_cpall` (8.0e-3, cp in both X layers). Alternatives with
  more margin: `F_rp_ra_cpall` 67,353,814 (5.1e-4: the same without u1, so u1 carries the error);
  `c2d8_k2_nop` 68,489,726 (3.5e-3); `F_rp_nop_ra_u1_cp2` 69,617,086 (6.9e-4); `c2:d8_rp_cp_u1_g`
  65,631,280 (6.8e-3).
- log_w 6, 3 x 4, depths 34-35 (6.2e-3 / 7.7e-3, from cp in the last X on the dense set): without
  cp2 the depth-32 point `F_d2_rp_nop_ra` has 2.7e-3.
- log_w 4 points are listed for R = 3 (exploration) and as the smaller copies of the log_w 6 ones;
  the ratios to the baseline match within ~1% between log_w 4 and 6 for the same layout.

**Tight or failing near-misses** (float32 criterion): `F_rp_nop_ra_cp2_p1` at log_w 6, 3 x 4
(P1DC in the last X): 3.7e-2; `F_rp_nop_ra_u1_cp2` at log_w 6, 3 x 2: 1.9e-2;
`F_l4c_z11_nop_ra_u1_cp2` at log_w 6, 3 x 2: 1.4e-2; `c2:d7...` (1-round d7 point) at log_w 6, 1 x 4: 4.2e-2;
`W_rp_nop_sl_m4_cp2` in w16 at log_w 6, 3 x 4: 1.1e-2 (float32 6.6e-3); `F_rp_nop_ra_u1` at log_w 6, 3 x 4: 4.6e-2.


## 5. Per-strategy verdicts

Savings are measured between two robust variants that differ only in the strategy (same layout,
log_w 4 unless marked); "robust up to T" is the largest round count where the variant passed.

| strategy | where it acts | verdict | evidence |
|---|---|---|---|
| S1 gated units (glu xor, one-unit chi + iota) | every round | **robust** | the base of every point; glu_xor parities are flat at the lattice points; chi is exact for all R |
| S2 compiler passes (bias folding, unit sharing, dedupe of equal E) | every layer | **robust** | exact; sizes of every point |
| split layout (X, Y) for middle rounds | every middle round | **robust up to T = 12 on the full sets (T = 18 on a 39-message probe), breaks at T = 24** | 29.8x per round; error 4.8e-4 / 7.9e-4 / 1.05e-3 at T = 6 / 9 / 12 (w4); **1.12 (98 wrong bits) at T = 24** (w3) |
| direct layout for middle rounds | every middle round | **robust (flat)** | 18.5x per round, 2 layers instead of 3; error 2e-5 at T = 24 and 3e-4 at T = 72 (w3); the depth-minimal points |
| flat Y (new, `YFLAT`) | split rounds | **robust, fixes the split layout for long runs** | +1 unit per bit (+0.48M per round at w4); needed every 2nd-4th split round: T = 24 at 4.1e-5 / 1.2e-4 / 1.3e-3 for every 2nd / 3rd / 4th, fails every 6th (1.12); T = 72 at 2.7e-3 (`F_rp_nop_ra_cp2_yf3`, w3) |
| lazy chi (lazy4 / lazy4c) for middle rounds | pairs of middle rounds | **stops paying** | exact and robust (7e-4 at T = 6..9) but 5.2-6.6M per round vs 1.93M for split + NOP (w4): 2.7-3.4x the split cost |
| lazy4c + c5 / z11 / lz5 for the last two rounds | last 2 rounds | **robust** | the depth-(3T-2) points; z11 vs plain lazy4c -77K at T = 6; lz5 no change for R > 1 steps (lazy digest only at step ends) |
| fold (`rp`) before the last theta | last round | **robust** | the dense-best last round: -0.36M at T = 6 (sfl4_rp 9.91M vs sfm 10.27M); error unchanged |
| s_last (`sl`) | last theta | **robust** | -42K at T = 6 (w4); error unchanged |
| NOP (X gates on 10 chi bits) | every split X | **robust** | -0.161M per round at w4 (-7.7%), -2.56M per round at w6; +3.4K sparse per round; error unchanged |
| column packing cp in every X | every X | **breaks for T >= 6** (float32) | -0.48M per round (w4); robust at T = 3-4 (4.2e-4, 7.8e-4 at w4; 8.0e-3 at w6 T = 4 with u1), **0.20 at T = 6** (float64 1e-6): decoders multiply rounding by 8 each round; a flat Y does not help (1.4e-2) |
| cp only in the last X (`cp2`, CPK = 2) | last X | **robust** | -0.33M at w4 (T = 6..12), -5.2M at w6 T = 6 (147.8M -> 142.6M); error 4.8e-3 (w6 T = 6), 6.1e-3 (w4 T = 12); 4e-5..1.3e-3 at T = 24 and 2.7e-3 at T = 72 with flat Y; also correct with bf16 weights (w16: 6.6e-3 at w6 T = 6) |
| P1DC D form (irrational knots) in every X | every X | **breaks for T >= 4** (structural) | float64 error 0.65 at T = 6; 1.6e-2 at T = 4 (w4) |
| P1DC only in the last X (P1K = 2) | last X | **mixed: robust up to T = 6 at log_w 6, breaks at T = 12** | -75K at w4 T = 6 / 12 (`F_rp_nop_ra_cp2_p1` 8.85M / 21.17M, 7.0e-3 / 7.0e-3); -1.2M at w6 T = 6 (141.36M, 4.9e-3); **3.7e-2 at w6 T = 12**; -1.13M at w6 T = 4 (`c2d8_k2_nop` 68.49M, 3.5e-3); with cp in both X at w6 T = 4: 2.1e-2 |
| min-parity on raw round-1 bits (`mp`) | round 1 | **robust in float32; breaks with bf16 weights** | -58K at T = 6 (w4); every F point uses it, up to T = 72; w16: 8.1 |
| raw even/odd parities with irrational knots (`ra`, P1DR) | round 1 | **robust in float32; breaks with bf16 weights** | -7K (w4) and it LOWERS the error in these layouts (1.1e-3 vs 3.3e-3 at T = 6); w16: 8.0 |
| round-1 u pairs (`u1`) | round 1 | **mixed: robust to T = 12 at log_w 4 (tight), breaks at log_w 6 T = 12 and with cp2 at T >= 6; breaks with bf16 weights** | -0.17M at w4, -2.86M at w6 T = 6 (147.79M -> 144.93M, 7.5e-3 on ref + mixed); 9.2e-3 at w4 T = 12; **4.6e-2 at w6 T = 12**; u1 + cp2: 2.4e-2 (w4 T = 6), 1.4e-2 / 1.9e-2 (w6 T = 6); u1 + cp everywhere robust only at T <= 4; w16: 7.9 |
| packed round-1 split (`sp`) | round 1 | **robust in float32 (T = 6); breaks with bf16 weights** | -0.14M at w4 T = 6, 4.4e-3; w16: 7.2 |
| theta pool (`tp`) | round 1 | robust at T = 2 | T = 2 only (direct first round): 2.1e-4 |
| digest packing m4 + s4 | carried digests | **robust (T <= 72 with a flat Y), the main error source for long runs** | -1.1M vs pairs at w4 T = 12 (21.25M vs 22.36M) but 6.1e-3 vs 2.1e-3; T = 72: 2.7e-3 (`F_rp_nop_ra_cp2_yf3`), but with u1 + cp2 + flat Y every 3rd the step-2 digest reaches **1.08e-2 (fails)** where pairs give 4.3e-4 (`F_rp_nop_ra_u1_m2_yf3`); w16-correct |
| digest packing m3 | carried digests | **breaks at T = 12** | 0.39 (10 boolify errors) on the dense set at w4 T = 12 |
| min-parity on the last theta's counts (`mpc`) | last theta | **stops paying** | -26K; error 4.6e-3 (T = 3) -> 9.1e-3 (T = 9..12) at w4 |
| 1-round frontier points as they are (c2 d6 / d7 / d8) | all | **robust only at T <= 3** | T = 3: d8 2.35M (82.3x); T = 4: 1.6e-2..2.4e-2; T = 6: 0.25..1.1 |
| wave-3 theta pool `_tr`, top-core `_bt` forms | round 1 / last theta | not needed | tp tested at T = 2 only; bt at T = 2 changed nothing |


## 6. Remaining ideas

- **Cheaper flatness.** The flat Y costs one unit per bit; one every 2nd-4th split round was
  enough for 24 rounds (every 6th failed), every 2nd-3rd for 72 rounds (log_w 3; every 4th was not
  tried at 72). Not tried: flattening the X copy instead (the copy has slope 1/2 at a = 0), or a
  per-T schedule (denser flat rounds late in the run, where errors are largest). A direct round is also flat (it re-thresholds through glu_xor) and costs +0.80M vs a
  flat-Y split round at log_w 4 (3.374M vs 2.575M), but saves a layer: mixing direct rounds in periodically is the
  depth-trading version of the same fix.
- **cp with flat decoders.** cp's decoders (hi = relu(p - 1)(4 - p)/2 on p/4, the fifth bit as
  c - p1 - p2) amplify rounding; a decoder with a flat form (e.g. glu_xor-style knots) might make cp
  robust in every X (-0.48M per round at log_w 4, about -7.7M at log_w 6 by the w^2 scaling (estimate), the biggest remaining per-round saving).
- **Digest carries.** With R rounds the step-1 digest crosses ~3R layers per step; m4 packing is
  the dominant float32 error for long runs (1.08e-2 at T = 72). A flat-decoded packing, or carrying
  digests in the lazy form, would cut ~1M at log_w 4, T = 12.
- **bf16 weights (w16).** Only flat forms survive rounding the weights to bfloat16: glu_xor
  parities, copies, chi, split X/Y, NOP, the fold, s_last, m4/s4 digest packing and cp in the
  last X passed; each round-1 trick added alone to the flat `W_rp_nop_sl_m4` breaks it (log_w 4, 3 x 2:
  mp 8.1, ra 8.0, u1 7.9, sp 7.2 worst error, tens of thousands of wrong bits), and so does the lazy4c + z11 end (55 wrong bits: `with_z11` also switches on min-parity
  for the range-33 count parities of the last theta). The `W_*` variants in `rv.py` are the bf16-weight frontier (section 4). Full bf16 (b16,
  activations too) fails for every circuit tried, the baseline included (5 wrong bits at log_w 4,
  1 x 2 rounds; 430 at 1 x 4; the gated circuits lose tens of thousands of bits).
- R = 24 at log_w 6 (the real Keccak-f[1600] XOF) was not built: by the per-round costs it would be
  about 72 x 31-35M = 2.2-2.5G dense for 3 steps with split + NOP rounds and a flat Y every 2nd-3rd
  round (estimate), against a threshold baseline of about 72 x 994M = 71.5G (estimate): ~30x.

## 7. Files and how to reproduce

In `$S/xof4/av/rounds/` (`S` = the session scratchpad):
- `repo/`: the base copy with this avenue's changes (`experiments/xof_shrink/xs3.py`, new `rv.py`,
  new `audit/rounds_eval.py`);
- `patch_vs_base.diff`: only this avenue's changes, relative to `$S/xof4/base` (`git apply -p1` there);
  `patch.diff`: `git -C repo add -N . && git -C repo diff`, which also contains the lead's wave-4
  tool changes already present in the base's working tree (ref_gen/ref_stress rounds, bf16_check);
- `results.jsonl`: every verified run (`kind` = `eval`: rounds_eval on ref + stress sets, 383+
  runs; `harness`: xofbench.py on all 66 frontier points, sizes identical; `size`: the marginal-cost
  builds; `baseline`: the three threshold baselines built here); `results_raw.jsonl` is the eval part;
- `frontier.json` (all Pareto points, f32 and w16), `frontier_tables.md`, `marginal.jsonl`;
- `refs/`: reference sets from the unmodified keccak (`ref_*` = ref_gen with 60 random, `sm_*` =
  ref_stress mixed 4 12, `sd_*` = ref_stress dense 4 10; `small_*` = 20-30 random probes);
- scripts: `genrefs.sh W STEPS R`, `ev.sh VARIANT W STEPS R [modes]`, `hb.sh VARIANT W STEPS R`,
  `marg.sh`, `summ.py`, `frontier_md.py`, `frontier_json.py`, `mkresults.py`, `logs/`.

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
R=$S/xof4/av/rounds/repo; E=$R/experiments/xof_shrink; F=$S/xof4/av/rounds/refs
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/src:$E:$E/depth_low/lib/python
# harness: sizes + 16 random messages vs the variant's eager function
$S/venv/bin/python $E/xofbench.py --log-w 6 --depth 3 --rounds 2 --variant rv:F_rp_nop_ra_cp2_p1
# float32 audit + stress (and --modes f32,f64,w16,b16) against the reference xof
$S/venv/bin/python $E/audit/rounds_eval.py rv:F_rp_nop_ra_cp2_p1 $F/ref_w6_s3_r2.pt $F/sm_w6_s3_r2.pt $F/sd_w6_s3_r2.pt
# new reference sets (fresh process, unmodified keccak)
PYTHONPATH=$S/xof4/base/src $S/venv/bin/python $S/xof4/base/experiments/xof_shrink/audit/ref_gen.py 6 3 60 /tmp/ref.pt 2
```

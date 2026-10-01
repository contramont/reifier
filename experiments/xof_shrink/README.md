# Shrinking the compiled SwiGLU circuit of 1-round Keccak XOF

**Circuit.** `xof(msg, depth=3, k)` with `k = Keccak(log_w=6, n=1, c=448, pad_char="_")`:
- one Keccak round per XOF step, 3 steps;
- a 1600-bit state, 1144 message bits, 3 x 224 output bits.

**Architecture (fixed).** reifier's `MLP_SwiGLU`: a stack of `y = wo(silu(wg n) * (wv n))` with
`n = RMSNorm(x)`. That is a Transformer whose attention is the identity and which has:
- no skip connections;
- untied layers;
- no embedding or readout layer.

The input is BOS + the message bits. The output is BOS + the digest bits, each read relative to
BOS within 0.02. Steepness is `c = 4, q = 8` (the new default, see "Bugs and steepness").

**Metrics** are those of the final PyTorch module:
- **depth**: number of layers;
- **dense**: numel of all parameters (norm, wg, wv, wo);
- **sparse**: number of nonzero parameters.

## Results (log_w=6, 3 XOF steps)

### Robust frontier

A circuit counts as robust only if all of the following hold:
- **Correct:** 0 wrong bits on every check.
- **Within half the tolerance:** worst error at most 0.01 everywhere. Float32 errors grow with
  message density and with the number of messages tried. So a point that only just passes a
  small audit is not enough (see "Float32 margins").

The checks are:
- the harness (16 random messages);
- two audits against the reference `xof`: 69 and 63 messages, with edge cases (all zeros,
  all ones, alternating, one-hot, ...) and densities 0.05 / 0.5 / 0.95;
- two stress sets:
  - mixed: 1792 messages at densities 0.005-0.995, plus lane- and z-structured patterns;
  - dense: 1248 messages at densities 0.85-0.99.

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst error (audits / mixed / dense) |
|---|---|---|---|---|---|
| baseline | main, threshold xor | 3,305,445,348 | 2,034,788 | 1 / 1 / 1 | |
| 3 | `dl3:d3c1_mp17_p1d_kt_xca` | 238,512,173 | 1,235,283 | 6.7 / 13.9 / 1.6 | 4.5e-3 / 9.3e-3 / 9.7e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | 124,087,564 | 526,518 | 5.0 / 26.6 / 3.9 | 5.6e-3 / 6.5e-3 / 3.3e-3 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` | 74,073,879 | 226,480 | 4.0 / 44.6 / 9.0 | 2.1e-3 / 4.4e-3 / 3.5e-3 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 53,915,598 | 178,277 | 3.3 / 61.3 / 11.4 | 1.8e-3 / 2.7e-3 / 3.6e-3 |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` | 48,831,997 | 128,804 | 2.9 / 67.7 / 15.8 | 4.7e-3 / 7.9e-3 / 7.5e-3 |
| **8** | **`c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra`** | **44,218,228** | 114,557 | 2.5 / **74.8** / 17.8 | 4.5e-3 / 5.2e-3 / 4.7e-3 |
| 9 | `c2:sp_col1b_p1d` (sparse end) | 66,359,285 | **91,460** | 2.2 / 49.8 / **22.2** | 1.1e-4 / 1.9e-4 / 1.9e-4 |

Robust points at the sparse end of each depth:

| depth | variant | dense | sparse | worst error (audits / mixed / dense) |
|---|---|---|---|---|
| 4 | `xc:d4a_m4k3_mpc_p1dxt_tp` | 143,675,838 | 396,090 | 2.2e-3 / 4.8e-3 / 5.6e-3 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 74,441,726 | 219,208 | 5.3e-3 / 8.0e-3 / 9.8e-3 |
| 6 | `c2:d6_lazy_cp_p1dct_tp_bt` | 62,253,531 | 162,221 | 3.0e-3 / 4.5e-3 / 5.0e-3 |
| 7 | `c2:sp_lazy2_l3_p1d` | 67,786,082 | 113,772 | 3.0e-3 / 2.6e-3 / 3.1e-3 |
| 8 | `c2:sp_base_p1d` | 62,287,910 | 95,637 | 1.0e-3 / 1.7e-3 / 1.4e-3 |

The robust headline points passed three more checks (all numbers in `results/final_frontier.jsonl`):
- `validate_bench.py`: the harness weights are bit-equal to the repo's own
  `Compiler().get_mlp_from_tree` at log_w 0-2, and depth, dense and sparse match;
- the harness at log_w 4 and 5 with 64 random messages;
- the harness at log_w 6 with 256 random messages.

**Wave 4 caveat: random sets do not bound the worst case.**
- An evolutionary search for bad messages (`audit/adv_search.py`) pushes every point above
  from depth 3 to depth 8 over 0.01 in the harness's sparse float32 kernel:

  | depth | worst error found |
  |---|---|
  | 3 | 2.3e-2 |
  | 4 | 2.6e-2, with one boolify misread |
  | 5 | 1.2e-2 |
  | 6 | 1.1e-2 |
  | 7 | 2.8e-2 |
  | 8 | 2.2e-2 |

- Every bit stays correct by nearest rounding.
- The worst case also depends on the summation order. Through reifier's own dense
  `nn.Linear` layers, only depth 3 (1.7e-2) and depth 7 (2.8e-2) stay above 0.01.
- Staying at or below 0.01 on every message found costs 3-10% dense (`ladder.py`):

  | depth | variant | dense |
  |---|---|---|
  | 4 | `b4_3_p1d` | 127,595,249 |
  | 5 | `b5_2_mp` | 81,486,372 |
  | 6 | `b6_7_sl` | 59,092,278 |
  | 7 | `b7_5_mp` | 52,415,738 |
  | 8 | `b8_4m_m3` | 46,379,980 |

- Exhaustive checks at log_w 0 (128 messages) and log_w 1 (2^22 messages) find no wrong bit.
  At log_w 1, the worst error over all messages is 1.3-7.9x the worst on random messages
  (`wave4/word-size/`).

### Smaller but float32-tight

These points pass the two audits, but their worst float32 error comes within 2x of the
tolerance, or crosses it, on the stress sets. In float64, with the same parameters, all of them
pass. Use them only with float64, or where a larger error is acceptable.

| depth | variant | dense | sparse | worst error, float32 (audits / mixed / dense) | float64 (dense) |
|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt1_xca` | 231,671,853 | 1,222,721 | 1.2e-2 / **3.1e-2** / **2.8e-2** | 1.4e-3 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_tr` | 73,443,039 | 227,576 | 8.0e-3 / 1.8e-2 / 1.7e-2 | 6.7e-3 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_tr_bt` | 53,284,758 | 179,373 | 8.5e-3 / 1.8e-2 / 1.3e-2 | 6.3e-3 |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_lz5_sl_p1dct_ra_bt` | 48,066,365 | 132,948 | 5.9e-3 / 8.8e-3 / 1.1e-2 | 3.3e-3 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra` | 43,761,588 | 114,731 | 1.5e-2 / 1.9e-2 / **2.4e-2** | 5.4e-3 |

### Progress (dense at depth 8)

| stage | dense-best | robust |
|---|---|---|
| baseline (depth 20) | 3,305,445,348 | |
| wave 1 (first report, `xs3:split_first_middle_mp`) | 61,082,590 | |
| wave 2 (`c2:d8_rp_m4s4_mp_cp_u1_sl`) | 45,071,004 (tight: 1.6e-2 on mixed) | 45,527,644 |
| wave 3 (this branch) | 43,761,588 (fails dense stress) | **44,218,228** |

Robust dense against wave 1, per depth:

| depth | wave 1 | now | change |
|---|---|---|---|
| 4 | 160.36M | 124.09M | -23% |
| 5 | 89.24M | 74.07M | -17% |
| 6 | 69.37M | 53.92M | -22% |
| 7 | 63.65M | 48.83M | -23% |
| 8 | 61.08M | 44.22M | -28% |

## Robustness across configurations and in bfloat16 (wave 4)

**Setup.**
- Five avenues re-ran the strategies on other Keccak word sizes, other XOF step counts and
  more rounds per step, and in two bfloat16 modes.
- A separate verifier re-ran each avenue's claims on fresh reference sets (new seeds,
  unmodified keccak). The numbers below are the verified ones; several avenue headlines were
  corrected by 5-10% (`wave4/VERIFICATION.md`).
- **Threshold baselines** for every configuration are in `results/wave4_baselines.jsonl`.
- **Tools:** `--rounds` in the harness, `ref_gen.py`, `ref_stress.py` and `adv_check.py`,
  plus `audit/bf16_check.py`.
- **The merged code** rebuilds 24 headline points of all five avenues with identical sizes
  (`results/wave4_regression.log`).
- **Robust** below means 0 wrong bits and a float32 worst error of at most 0.01 on the audit
  plus stress sets of that configuration.

### Word size (3 steps, 1 round per step)

Every strategy stays exact at every word size. Gains are flat from log_w 2 up, so each
depth keeps its ratio to the threshold baseline.

| log_w | baseline dense | depth-8 best | x | sparse end | x sparse |
|---|---|---|---|---|---|
| 0 | 826,914 | 10,880 | 76.0 | 1,334 | 23.5 |
| 1 | 3,341,810 | 42,980 | 77.8 | 2,821 | 22.7 |
| 2 | 12,978,468 | 171,632 | 75.6 | 5,759 | 22.1 |
| 3 | 51,771,764 | 687,727 | 75.3 | 11,467 | 22.2 |
| 4 | 206,803,140 | 2,758,126 | 75.0 | 22,932 | 22.2 |
| 5 | 826,645,028 | 11,047,168 | 74.8 | 45,744 | 22.2 |
| 6 | 3,305,445,348 | 44,218,228 | 74.8 | 91,460 | 22.2 |

- Depth 3 gives 13.9-16x at every word size and depth 6 gives 62-68x (`wave4/word-size/`).
- **At log_w 0-1** only 1-3 lanes per column are live:
  - no round-1 parity set reaches 7 bits, so `mp` and `ra` do nothing at log_w 0;
  - `u1` saves 1% instead of 6%;
  - `sl` costs a little, and `with_slast_auto` turns it off there.

### XOF steps (log_w 6, 1 round per step)

| steps | baseline depth / dense | robust dense-best | depth | dense | x | sparse end (x sparse) |
|---|---|---|---|---|---|---|
| 1 | 8 / 1,104,072,656 | `steps:x_d6_mp_ra` | 2 | 2,858,017 | 386 | same point (35.0x) |
| 2 | 14 / 2,163,132,858 | `steps:d7g` | 5 | 19,479,268 | 111 | `sp:col1b` (27.1x) |
| 3 | 20 / 3,305,445,348 | the robust depth-8 point above | 8 | 44,218,228 | 74.8 | `c2:sp_col1b_p1d` (22.2x) |
| 4 | 26 / 4,534,622,798 | `steps:x_d7_m4_s4_mp_cp_c5_sl_mpc_p1dcl_ra_bt_lp2` | 10 | 84,378,834 | 53.7 | `c2:sp_col1b_p1d` (20.3x) |
| 6 | 38 / 7,268,023,266 | `steps:x_d8_m4_s4_mp_cpl_sl_p1dc_ra` | 17 | 169,736,779 | 42.8 | `sp:col1b` (18.4x) |
| 10 | 62 / 13,936,161,290 (from layer shapes) | `steps:x_d8_m4_s4_mp_sl_ra` | 29 | 378,759,263 | 36.8 | `sp:col1b`, 422,605 |

**Why the ratio falls.** Each extra step adds 3 layers and 33.3M dense of computation, about
32x less than a threshold step. But without skip connections every earlier digest must be
carried to the end:
- carries grow as about 0.06M x S^2 at log_w 4;
- they are 15% of dense at 3-4 steps, 17-18% at 6 and 25-26% at 10;
- 4-bit digest packing (`m4s4`) cuts them 3x.

**The S = 3 stacks as built fail at more steps.** Their worst errors are 3.8e-2 at 4 steps,
0.52 at 6 and 4.4e4 at 10 (log_w 4). Middle steps must avoid forms that do not damp errors:
- column packing, which must be budgeted: every X layer up to 4 steps, the last 1-2 beyond;
- the irrational D forms (last X only);
- min-parity on counts;
- `u1`.

The late digests then travel as pairs (`lp2`). The avenue's own 58.0x / 44.6x / 37.7x points
fail fresh dense sets narrowly (1.2-4.8e-2, bits still correct); the table has the verified
replacements.

### Rounds per XOF step (`--rounds R`, k.n = R)

`xs3.build` now runs R rounds per step: round r uses round constant r % R, and a digest
comes only after every R-th round. A middle round with split layers (X, Y and chi) costs
3 layers and 33.3M dense at log_w 6, or 30.8M with `NOP`. That is 1/30 to 1/32 of a
threshold round (993.7M, 6 layers).

| steps x rounds | log_w | baseline depth / dense | robust dense-best | depth | dense | x |
|---|---|---|---|---|---|---|
| 1 x 2 | 6 | 14 / 2,097,727,098 | `rv:T2_split_u1_ra_sl_mpc` | 5 | 18,248,721 | 115 |
| 1 x 4 | 6 | 26 / 4,085,035,982 | `rv:F_rp_ra_cpall` | 11 | 67,353,814 | 60.6 |
| 3 x 2 | 6 | 38 / 6,527,735,970 | `rv:F_rp_nop_ra_cp2_p1` | 17 | 141,357,507 | 46.2 |
| 3 x 4 | 6 | 74 / 12,972,317,214 | `rv:F_rp_nop_ra` | 35 | 344,008,785 | 37.7 |
| 1 x 24 | 3 | 144 / 374,839,360 | `rv:F_rp_nop_ra_u1_cp2_yf4` | 71 | 11,424,438 | 32.8 |
| 3 x 24 | 3 | 428 / 1,210,834,652 | `rv:F_rp_nop_ra_cp2_yf3` | 215 | 39,230,940 | 30.9 |

- **Past about 20 rounds** even plain split rounds diverge in float32 (1.12 against 1.4e-8 in
  float64). The Y unit max(0, E)(2 - E) has slope -2 at E = 2. A flat Y (+1 glu_xor unit per
  bit) every 2nd-4th split round fixes it (`YFLAT`, `_yfN`).
- **Lazy-chi middle rounds** stop paying: 2.7-3.4x the cost of split rounds.
- **Budgets:**
  - column packing and the irrational D forms only in the last X;
  - `u1` and `m3` break with many rounds.

### bfloat16 weights (w16: weights rounded to bfloat16, float32 activations)

**What breaks.** Every float32 frontier point breaks as built, with thousands of wrong bits.
- **The SwiGLU construction is never the cause.** c*q = 32, the 4/q = 1/2 value scale, 1/16
  and the step offsets are all exact.
  - This needs 4/q to be dyadic: at q = 12, even the chi+iota circuit breaks. So q must stay
    a power of 2.
- **The cause is unit coefficients that need more than bfloat16's 8 significant bits:**
  - unit-sharing ratios such as 6/7 and 1/3;
  - lazy4c's 1/3;
  - irrational knots and slopes;
  - min-parity products;
  - long out weights from folded constants.

**The fix.** Three function-preserving rewrites make every such weight representable
(`bf16_units.py`, `XOF_BF16=u/ub`):
- per-unit rescaling;
- extra hidden units that carry the low bits of long out weights;
- biases split to 24 bits over the constant features BOS/256 and BOS/65536.

After them, w16 equals float32 bit for bit (`audit/bf16_weights.py` certifies
0 non-representable weights).

**What stays out.** Forms that are non-dyadic or irrational on the raw message in layer 1,
which has no second constant feature: `mp`, `mpx`, `mp17` and the raw p1d forms (`_ra`).
The same holds for `th`, `tr`, `tq` and the flat `pf` form.

| depth | variant (`XOF_BF16`) | dense | sparse | x dense | vs float32 frontier |
|---|---|---|---|---|---|
| 4 | `bfv:d4x_m4k2_kt1` (ub) | 136,155,752 | 546,288 | 24.3 | +9.7% |
| 5 | `bfv:d5_k2_tp` (u1) | 79,469,937 | 230,257 | 41.6 | +7.3% |
| 6 | `bfv:d6_m4s4_cp_c5_sl_tp_bt` (ub1) | 56,283,191 | 181,151 | 58.7 | +4.4% (marginal: 1.01e-2 on one fresh set) |
| 7 | `bfv:d7_c5_p1dct_bt` (ub) | 49,787,174 | 134,246 | 66.4 | +2.0% |
| 8 | `bfv:d8_g_p1dct` (ub) | 45,144,507 | 117,433 | 73.2 | +2.1% |
| 9 | `sp:col1b` (u) | 68,233,410 | 91,778 | 48.4 (22.2x sparse) | +0.3% sparse |

- No robust depth-3 point was verified: `bfv:d3_kt` fails in the sparse kernel.
- More steps or rounds, from integer and half-integer forms only:
  - `steps:x_d8_m4_s4_sl` is w16-correct at 88x / 47x / 40x for 2 / 4 / 6 steps, and at
    36.6x for 10 steps with `lp2`;
  - the `rv:W_*` family is w16-correct at 3 x 2 (143.7M, 45.4x) and 3 x 4 (345.2M, 37.6x).

### Full bfloat16 (b16: weights and activations, as `Compiler(mlp_dtype=t.bfloat16)` runs)

**Everything fails as built,** including main's threshold baseline.
- The baseline fails on edge-case messages: 77 / 332 wrong bits at log_w 4. Wide rows put
  step ReLUs at pre-activations in the thousands, where bf16's spacing exceeds the step width.
- Every gated-unit strategy above fails with thousands of wrong bits.

**The E-rule makes bf16 exact** (`bf16x/eunits.py`).
- For every unit max(0, G) * V and every reachable input, one of these holds:
  - G <= -1/2;
  - G = 0;
  - V = 0;
  - G and |V| are both powers of 2.
- With an exact BOS pair relu(2cq) - relu(cq), every feature of every layer is then exactly
  k * BOS, with k in {-1, 1, 2, 4}, or about 0. bf16 rounding snaps it back each layer, so no
  error ever builds up.
- glu_xor breaks the rule for n >= 3, so `bx.py` uses E-exact xor forms. Theta takes 2 layers,
  because no E-exact one-layer symmetric parity exists from 9 bits up.

**The E-exact builder `bf16x.bx`.** Each round has 3 layers:
- C: column parities as copies, plus one feature q = a + C[x-1] - C[x+1]' in {-1, 0, 1, 2};
- T: theta = parity(q), 2 units per bit;
- X: chi with iota, 1 unit per bit.

| configuration | variant | depth | dense | sparse | x dense | also float32-robust |
|---|---|---|---|---|---|---|
| log_w 6, 3 x 1 | `bf16x.bx:pk1q` | 9 | 72,009,024 | 132,565 | 45.9 (15.3x sparse) | yes (4.0e-4) |
| log_w 4, 3 x 1 | `bf16x.bx:pk1q` | 9 | 4,481,880 | 33,167 | 46.1 | yes |
| the whole grid | `bf16x.bx:pk1q` | 3 per round | | | 28-71 (383 at 1 step) | up to about 15 layers; borderline at 18 (1.1e-2) |
| log_w 3, 3 x 24 | `bf16x.bx:q_rt3` (step copy every 3 rounds) | 240 | 54,834,630 | 626,141 | 22.1 | yes |

- Without re-thresholding, `pk1q` stays bf16-exact at 216 layers, but in float32 its error
  doubles per layer.
- **A step-copy re-threshold layer every 3 rounds** keeps float32 correct. Every 4 rounds
  fails at 24 rounds.
- **Compiler fix, on main since fdc78d6** (the avenue's patch is
  `bf16x/swiglu_exact_steps.diff`; main's minimal version has no environment switch):
  - the exact BOS pair;
  - steps that rise over [1/2, 1/2 + 1/c], so both ReLUs sit at exact multiples;
  - rows whose sum can overshoot built as BOS - step(1 - sum).

  These make the default threshold compile bf16-exact: margin about 1e-9 at log_w 3, 4 and
  6, with 2 and with 24 rounds (428 layers). The cost is +0 dense and +2.8% sparse.
  `MLP_SwiGLU.from_matrices` turns it on for 16-bit dtypes. In this harness it is
  `XOF_EXACT=1`; `audit/bf16_check.py` always uses it.
- **Caveat: exactness needs float32 accumulation.** On CUDA, PyTorch's default
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True` breaks it at
  batch >= 32, for the fixed baseline too. Set the flag to False; the results then match the
  CPU bit for bit.

### Which strategies are robust

| strategy | word size 0-6 | more steps | more rounds | bf16 weights | full bf16 |
|---|---|---|---|---|---|
| xor2 and chi (+iota) units | robust | robust | robust | exact | exact (E-rule) |
| glu_xor, n >= 3 | robust | robust | robust | exact | breaks: E-exact xor forms instead |
| constant folding, unit sharing | robust | robust | robust | fixed by rescaling | needs the exact BOS |
| split rounds (X, Y) | robust | robust, best middle step | robust to about 20 rounds, then a flat Y | exact | breaks (counts > 2) |
| direct rounds | robust | 1.4x a split step | robust (flat), fewest layers | exact | breaks |
| lazy chi (lazy4/lazy4c) | robust | stops paying in the middle | stops paying in the middle | fixed | breaks |
| last-round tricks: fold, sl, c5/z11, bt | robust (sl: auto off at log_w 0-1) | robust; share falls as 1/S | robust | exact or fixed | breaks |
| fused rounds (d3/d4), Walsh | exact; float32 error grows with w | stop paying | not built | d4 +9.7%; d3 not robust | breaks |
| sparse-first `sp` | robust, 22-24x sparse | robust, sparse end at every S | not built | exact | breaks |
| column packing `cp` | robust | compounds: budget per S | compounds: last X only | exact | breaks |
| `u1` / `sp` round-1 pairs | gain grows with w (1% -> 6%) | uses the error budget | fails with cp from 6 rounds | fixed | breaks |
| digest packing `m4s4` (+ `lp2`) | robust (weak adversarially at d7/d8) | pays more as S grows | main error at >= 12 rounds | exact | breaks: E-exact pairs |
| raw-bit parities `mp`, `ra` | robust; no-op at log_w 0-1 | robust | robust | breaks (layer 1) | breaks |
| min-parity on counts `mpc` | breaks: error grows with w | breaks in the middle | tight | fixed (p1d top-core form) | breaks |
| irrational D forms `p1dc*` | robust at 3 x 1 (tight adversarially) | compound: last X only | compound: last X only | fixed | breaks |
| round-1 theta pool `tr` / `t8` | `tr` breaks; `t8` random-robust, adversarially tight | not tested | not tested | `tp` fixable | breaks |

## What made the difference

1. **Gated units** (in main since ba689f1). SwiGLU's value path computes products, so one
   hidden unit is `max(0, g) * v`:
   - xor of 2 bits takes one unit, and n-bit xor takes one layer with ceil(n/2) units;
   - chi `a ^ (~b & c)` is one unit, and so is iota folded into it.
2. **Generic compiler passes** (wave 1):
   - constants fold into biases;
   - dead code is removed;
   - a permutation-only output layer is dropped;
   - copies take one unit;
   - equal units share a hidden unit (`Matrices.layer_to_units`).
3. **Keccak layouts** (wave 1, builders `xs.py`, `xs3.py`, `xs4.py`, `xc.py`):
   - theta's column parities are shared;
   - count features (`glu(numeric=True)`) let `wo` hand integer sums to the next layer;
   - a lazy chi computes theta's parity inside its own units;
   - digests are packed several bits per feature for the copies.
4. **Stacking exact tricks** (wave 2, `c2.py`; `notes/FRONTIER.md` section 3 lists every flag):
   - column packing into X2: 961 features for 1600 bits (`cp`);
   - round-1 u pairs: X1 emits 641 features (`u1`);
   - a fold layout for round 3 (`rp`);
   - one reduction unit per column (`c5`) or one zigzag (`z11`);
   - packed digest pairs (`lz5`, `m4s4`);
   - the last theta emitting chi's gate (`sl`).
5. **Parity with fewer units** (wave 3, `wave3/`):
   - The parity of a count in [0, n] needs only floor((n-1)/2) gated units for every n >= 6.
     That is one unit fewer for every even n. Odd n were already at (n-1)/2 with wave 1's
     min-parity.
   - The core is a 2-unit form for n = 6 with irrational knots at 3 +- (2 sqrt 2 - 2):
     `1 - parity(s) = 8 + max(0, d+r)(d-a) + max(0, r-d)(-d-a)`, where d = s - 3,
     r = 2 sqrt 2 - 2 and a = 2 + 2 sqrt 2. glu_xor ramps extend it to any n
     (`depth_low/lib/python/p1d_forms.py`).
   - No form with integer or small-rational knots exists in the searched spaces. That is why
     the earlier exhaustive searches, which only tried such knots, missed it.
   - It applies to every even-range parity:
     - the D of theta (4 units per column pair instead of 5; 3 is impossible);
     - the round-1 raw parities;
     - the last theta on range 34 (16 units instead of 17, `_bt`);
     - the fused rounds of depths 3 and 4.
   - The robust points put the non-flat integers of each form at the top of the count range,
     where counts are rare (`_t` forms).
6. **Round-1 theta pool** (wave 3, `theta1-t8`): 4451 units, 3 above its structural floor.
   Its float32 margin is tight, so the robust depth-5/6 points use the direct form (`_ra`),
   which costs +0.63M.

## Float32 margins

The float32 error of a readout comes from rounding, not from the silu approximation:
- **q does not help.** q = 12 or 16 leaves the error roughly unchanged. At depth 6 the
  mixed-stress error is 1.78e-2 at q=8, 1.66e-2 at q=12 and 2.09e-2 at q=16.
- **float64 removes most of it.** The same depth-3 network drops from 3.1e-2 to 1.6e-3.

**How it builds up at depth 8.** Measured per layer as |float32 - float64| relative to BOS,
on the dense stress set:
- the packed-feature decoders (Y1, chi1, X2) raise it from 7e-6 after layer 1 to about 9e-4
  after layer 6;
- a parity form whose knots sit between integers amplifies it: the flat range-11 form
  reaches 3.6e-3, against 1.2e-3 for glu_xor;
- chi3's gate `2a - b + c` then multiplies it about 5x: 1.9e-2 against 3.8e-3.

**What costs margin, and what each robust point pays instead:**

| costs margin | robust replacement | cost |
|---|---|---|
| 5-unit flat range-11 parity (`_pf`) at depth 8 | glu_xor (`_g`) | +0.46M |
| round-1 theta pool (`_tr`) at depths 5 and 6 | direct forms (`_ra`) | +0.63M |
| LZ5 digest decoders after the u1 round 1 at depth 7 | unpacked digest-2 values | +0.77M |
| odd count ranges on the n+1 top-core form (`_kt1`) at depth 3 | earlier forms for odd ranges (`_kt`) | +6.8M |
| bump forms (overlay_d4, 43.99M at depth 8) | none | fail at 2.9e-2 |

**The 2x rule.** Worst errors grow with the number of messages tried:

| point | audits | mixed stress | dense stress |
|---|---|---|---|
| depth-8 tight point | 1.5e-2 | 1.9e-2 | 2.4e-2 |
| wave-2 depth-8 headline | 8.6e-3 | 1.6e-2 | |

That is why a robust point must stay within half the tolerance.

## Floors and headroom

Proven or exhaustively checked (details in `notes/FRONTIER.md` section 5 and `wave3/*/NOTES.md`):
- The last layer needs at least 673 hidden units. The output functions have real rank 673.
- **Readers of packed features.** A single unit cannot read one bit out of a packed pair
  `p = t + 2s`. So readers that spend one unit per output bit need unpacked inputs, and that
  pins the widths:
  - chi1 at 1464 inputs or more;
  - the lazy chi at 1600 or more.
- **Parity.** With integer or small-rational knots, the parity of a count in [0, n] needs
  floor(n/2) units (exhaustive for n <= 11). Irrational knots reach floor((n-1)/2), one less
  for even n.
  - D = parity(C_L + C_R) in 3 units is impossible for any real affine gates: 8 knot
    crossings are needed, and 3 lines give at most 6.
- **Depth.**
  - No layer can hold a theta after a chi, so depth 2 is out.
  - A fused round needs a pair parity per chi bit.
  - No cheaper exact AND of two parities exists in any of the spaces searched.
- **No 2-unit double AND** for the lazy chi exists at window 3 or less. The search covered
  |w| <= 2 with integer or half-integer knots (9.3M gate pairs) and |w| <= 3 with integer knots
  (67.5M).

**Estimate for this architecture:**
- about 42-43M dense at depth 8, which is 77-79x the baseline;
- about 85-88K sparse, the floor of the sparse-focus family.

The robust depth-8 point is at 44.22M. What is left:
- about 2M in digest carries: without skip connections, the step-1 digest crosses 4 layers;
- the u1 decode;
- float32 headroom: the tight depth-8 parity would save 0.46M.

100x the baseline would be 33.05M, a further -25% at depth 8. No known unit form gets there
inside this architecture. With a residual stream and weights tied across the 3 identical XOF
steps (outside the fixed architecture, `architecture/`, `notes/architecture.md`), depth 9
reaches 30.3M / 56.7K.

Ideas that did not pay off:
- depths 9 and 10 for dense;
- a 3-unit D;
- bump forms;
- 6-bit digest packing, which fails float32 (0.026-0.030);
- min-parity on count features far from the inputs;
- CSE before layout;
- larger q.

## Bugs and steepness (on main, ac9455d)

- **Steepness q = 8** (was 4) for `Compiler`, `SwiGLU.from_matrix` and `MLP_SwiGLU`.
  - Step gates are now within about 1e-5 of 0/1, against 3.6e-3 at q = 4. Gated units are
    k = c*q = 32 sharp.
  - `wv` takes a 4/q share of the scale-down, so hidden activations keep their q = 4 size.
  - Circuits that failed at q = 4 pass at q = 8:
    - SHA3-224 with glu xor and chi (50 layers): 461 wrong bits before;
    - the naive glu XOF: 330 wrong bits before.
  - Full SHA3-224 on threshold gates (146 layers, 23.9B parameters) is within 3.6e-7.
  - Fan-in up to 8192 is within 1e-5.
  - No instability was found at q = 12 either. q changes no size.
- **Tracer.**
  - Functions that exit by raising unbalanced the call stack (PY_UNWIND).
  - Helpers named `gate` or `glu` were taken for the creators; creators are now matched by
    code object.
  - Bits made during tracing by untraced code, such as a generator passed to `gate`, now
    raise instead of compiling wrongly.
  - Constant and passthrough outputs compile.
- **Smaller fixes.**
  - `gate` accepts iterators.
  - `glu` normalizes units given with list weights, and tolerates real-valued rounding (1e-6).
  - `MLP_SwiGLU.from_matrices` honours `dtype` and zips strictly.

## Library changes (on main since fdc78d6) and contents

- `src/` needs nothing beyond main.
  - fdc78d6: exact bfloat16 steps and BOS.
  - ac9455d: the bug fixes and q = 8.
  - 21683d0: the XOF compiler support:
    - `glu(..., numeric=True)` for count features;
    - hidden-unit sharing and dropping of always-zero units in `Matrices.layer_to_units`;
    - the outputs layer after gated units kept only when step gates come before them.
- `min_parity.py` here holds wave 1's `MIN_PARITY` / `glu_xor_min`. They stay out of the
  library because they are exact only on raw 0/1 inputs.
- `experiments/xof_shrink/`:
  - `xofbench.py`: harness;
  - `validate_bench.py`;
  - builders: `xs*.py`, `xc.py`, `c2.py`, `depth_low/`, `sp.py`, `rs*.py`, `fl*.py`, `xp*.py`;
  - wave-4 builders:
    - `ladder.py`: word-size ladders and adversarial-standard points;
    - `steps.py`: any number of XOF steps;
    - `rv.py`: R rounds per step;
    - `bfv.py` + `bf16_units.py`: bf16 weights;
    - `bf16x/`: E-exact full bf16, and the avenue's compiler patch (on main as fdc78d6);
  - overlays: `overlay_opt/`, `overlay_d4/`;
  - `audit/`:
    - `adv_check.py`, `ref_gen.py`, `ref_stress.py`, all with `--rounds`;
    - `bf16_check.py`, `bf16_weights.py`;
    - `adv_search.py`, `multi_check.py`, `ref_exhaustive.py`, `ref_msgs.py`, `rounds_eval.py`;
  - `results/`;
  - `notes/`: wave 1 and wave 2, with `FRONTIER.md` = wave 2;
  - `wave3/`: FRONTIER3.md, and each avenue's NOTES.md and search scripts;
  - `wave4/`: BRIEF4.md, VERIFICATION.md, and each avenue's NOTES.md, frontier and results;
  - `architecture/`.

## Reproduce

```bash
cd <reifier checkout on this branch>
E=experiments/xof_shrink; export PYTHONPATH=src:$E:$E/depth_low/lib/python
V=c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra
python $E/xofbench.py --log-w 6 --depth 3 --variant $V --widths   # sizes + 16 random messages
python $E/audit/ref_gen.py 6 3 60 /tmp/ref_w6.pt                  # edge cases + 60 random
python $E/audit/ref_stress.py mixed 8 16 /tmp/stress_mixed.pt     # 1792 messages, ~3 min
python $E/audit/ref_stress.py dense 8 12 /tmp/stress_dense.pt     # 1248 messages
for R in /tmp/ref_w6.pt /tmp/stress_mixed.pt /tmp/stress_dense.pt; do
  python $E/audit/adv_check.py --eager 0 $R $V; done               # worst error, wrong bits
python $E/validate_bench.py $V                                    # bit-equal to the repo pipeline
# other configurations: word size, XOF steps (--depth), Keccak rounds per step (--rounds)
python $E/audit/ref_gen.py 6 3 60 /tmp/ref_w6_s3_r2.pt 2           # 3 steps x 2 rounds
python $E/xofbench.py --log-w 6 --depth 3 --rounds 2 --variant rv:F_rp_nop_ra_cp2_p1
python $E/audit/adv_check.py --eager 0 /tmp/ref_w6_s3_r2.pt rv:F_rp_nop_ra_cp2_p1
# bfloat16: w16 = bf16 weights, b16 = bf16 weights and activations
XOF_BF16=ub python $E/audit/bf16_check.py /tmp/ref_w6.pt bfv:d8_g_p1dct   # w16-correct
python $E/audit/bf16_check.py /tmp/ref_w6.pt bf16x.bx:pk1q                # b16-correct
```

Generate the reference sets in a fresh process, before any builder has patched keccak.

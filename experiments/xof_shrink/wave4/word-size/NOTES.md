# Wave 4, avenue word-size: which XOF size reductions hold across Keccak word sizes?

Configurations: log_w 0-6, 3 XOF steps, 1 round per step, capacity `xofbench.CAPACITY`
(`k = Keccak(log_w, n=1, c=CAPACITY[log_w], pad_char="_")`). float32 unless said otherwise.

## Summary

- **Exactness holds at every word size.** 90+ variants (the ladders below, the README frontier, the
  sparse ends) x log_w 0-6: no crash and 0 wrong bits (nearest) on 1053 reference messages per log_w.
  Also 0 wrong bits on:
  - all 128 messages at log_w 0 (91 variants);
  - all 4,194,304 messages at log_w 1 (32 variants);
  - ~400 adversarial messages at log_w 6 (every variant).
- **The gains are the same at every log_w >= 2** (within 0.5 points per strategy; section 3). By the
  brief's criterion the robust frontier keeps its ratio to the threshold baseline from log_w 0 to 6:
  - depth 8: 75-79x;
  - depth 7: 67-70x;
  - depth 6: 62-68x;
  - the sparse end: 22-24x sparse.
- **What stops paying at small w, and why:**
  - `mp`, `ra`, `mpx` and `c1_mp17` do nothing at log_w 0-1: with 10 of 25 lanes capacity, no
    round-1 parity set has 7 or more bits;
  - `ra` in the split / u1 layouts only acts on T = 8 column pairs, which exist from log_w 4 on;
  - `sl` does not pay at log_w 0-1, where the digest is the whole row (d = 5w). The new
    `with_slast_auto` switches it off there;
  - the S6 round-1 pools cost 1-4% at log_w 0-1 and gain only on T = 8 columns.
- **Float32 errors grow with w**, so strategies that are tight at log_w 6 are fine at small w. Each
  crosses the threshold at its own w:
  - `mpc` in the d8 parity layer from log_w 1 (over all messages);
  - the `tr` pool from log_w 2-3;
  - `lz5` at depth 7 from log_w 5;
  - the plain fused d3 at log_w 6.
- **Fix: the S6 pool only where it pays (`F18` / `t8`).** The whole `tr` - `ra` gain is 3 units per
  T = 8 column, so `t8` uses F1 only there. It equals `ra` at log_w <= 3 and has `tr`'s dense from
  log_w 4. At log_w 6 it passes the brief's criterion and the README's big sets:
  - d6: **53,284,758** / 174,973 (worst 7.6e-3), against 53,915,598 / 178,277;
  - d5: **73,443,039** / 223,176 (worst 7.2e-3), against 74,073,879 / 226,480.
- **The brief's criterion does not bound the worst case.**
  - At log_w 1, the worst error over all messages is 1.3-7.9x the worst over the 1053 random
    messages. The d8 `mpc` points go from 2.8e-3 to 1.3e-2.
  - An adversarial message search (new `audit/adv_search.py`, section 7) pushes every robust point of
    the README at log_w 6, depths 3 to 8, above 0.01. All values are verified against the reference xof:
    - d4 misreads a bit (2.6e-2);
    - d3 2.3e-2, d5 1.16e-2, d6 1.07e-2, d7 2.8e-2, d8 2.2e-2;
    - the d4 and d5 sparse ends (`_tp`) reach 2.4e-2 (a misread) and 3.6e-2;
    - `lz5` at depth 7 and the `tr` pool misread 8 bits each.
  - The searches have not converged (2-4x spread between seeds), so these are lower bounds.
  - Staying under 0.01 on everything found costs +3% dense at d4, +5% at d8, +7% at d7, +10% at d6
    and +10% at d5 (table at the end of section 9). It means stopping the ladders earlier:
    - before `kt_xc` (d4), `p1dxt` (d5), `p1dct` (d6) or `m4s4` (d7);
    - at d8, `m3` in place of `m4s4` and nothing after it.
  - Below half the tolerance under search:
    - the layouts, `cp`, `c5` / `z11` (<= 1.1e-3);
    - `u1` and `mp` at depth 8 (<= 3.3e-3); the depth-7 `mp` point reaches 6.5e-3;
    - the depth-6 chain up to `sl` (4.6e-3);
    - the sparse ends `c2:sp_col1b_p1d` (3.5e-4) and `c2:sp_base_p1d` (3.5e-3).
  - The weak points are:
    - `m4s4` at depths 7-8, and `bt` and `lz5` at depth 7;
    - `kt_xc` / `xca` and the fused d3 / d4 rounds;
    - `p1dct` / `p1dxt` / `ra` / `bt` at depths 5-6;
    - `mpc` in the d8 parity layer;
    - every S6 pool (`tr`, `tp`, and the new `t8`: 1.1-1.4e-2).

## 1. Method

- **Reference sets** per log_w, made in fresh processes from the unmodified reference keccak:
  - `refs/audit_w{W}.pt`: `ref_gen.py W 3 60` (9 edge cases + 60 random at densities 0.05/0.5/0.95);
  - `refs/mixed_w{W}.pt`: `ref_stress.py mixed 4 12 --log-w W` (672 messages, densities 0.005-0.995
    plus lane / z / lane-xor-z patterns);
  - `refs/dense_w{W}.pt`: `ref_stress.py dense 2 12 --log-w W` (312 messages, densities 0.85-0.99).
  - So each log_w has 69 audit + 984 stress messages. Larger sets for the frontier at log_w 5 and 6:
    `big{mixed,dense}_w{5,6}.pt` (`mixed 8 16` = 1792 and `dense 8 12` = 1248 messages, the README's sets).
  - **Exhaustive sets** (new `audit/ref_exhaustive.py`): all 128 messages at log_w 0 (bit-level
    reference) and all 2^22 = 4,194,304 messages at log_w 1 (vectorized numpy keccak, checked bit for
    bit against the reference xof on 3002 messages and against the 1053 messages of the three sets).
- **Checker** (new `audit/multi_check.py`): builds a variant once with the harness's own code
  (`xofbench.build_layers`, `metrics`, and `verify` = the harness check on 16 random messages against
  the variant's own eager function), then measures on every reference set the worst
  |out/BOS - bit| (margin), wrong bits (err > 0.5) and boolify errors (1 iff within 0.02 of BOS), and
  runs the variant's eager function against the reference on the first 12 audit messages.
  `--modes f32,w16` adds bf16-rounded weights with float32 activations.
  - The sizes are the harness's: depth / dense / sparse come from `xofbench.metrics` on the same
    layers; `xofbench.py --log-w W --variant V` prints the same numbers (spot-checked in
    `results.jsonl`, key `harness_cli`).
- **Robust** (the brief's criterion): harness ok, 0 wrong bits and 0 boolify errors on all sets,
  eager function equal to the reference, worst error <= 0.01 on harness + audit + mixed + dense.
  "Tight" = exact but 0.01 < worst error < 0.02; "fail" = worst error >= 0.02.
- **Ladder** (new `ladder.py`): every rung is a composition of the existing builders' flag wrappers,
  one flag added per rung, so the gain of a rung is the gain of that strategy on top of the previous
  ones. Families (the last rung of each is the robust frontier point of the README):
  - `b8` (depth 8): `split_first_lazy4` + fold `rp` -> `cp` -> `u1` -> `mp` -> `m4s4` -> `sl` ->
    `p1dct` -> `ra` (= `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra`), then `mpc`;
  - `b7`: `split_first_lazy4c` -> `cp` -> `c5` -> `mpc` -> `u1` -> `mp` -> `m4s4` -> `sl` -> `p1dct` ->
    `ra` -> `bt` (= `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt`), then `lz5`;
  - `b6`: `lazy4c_middle` -> `cp` -> `c5` -> `mpc` -> `lz5` -> `mp` -> `m4s4` -> `sl` -> `p1dct` ->
    `ra` -> `bt` (= `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt`), and the pools `tr` / `tp`;
  - `b5`: xs4 d5 (m4k2) -> `cp` -> `mp` -> `mpc` -> `p1dxt` -> `ra` (= `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra`), `tr`;
  - `b4`: dl3 d4x m4k2 -> `mpx` -> `wmpc` -> `p1d` -> `kt_xc` -> `xca` (= `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca`);
  - `b3`: dl3 d3 -> `c1_mp17` -> `p1d` -> `kt_xc` -> `xca` (= `dl3:d3c1_mp17_p1d_kt_xca`);
  - plus S1 (`variants:glu_xor_everywhere`, `variants:glu_chi_iota`), the xs3 layouts, the sparse
    ends (`sp:col1b`, `c2:sp_*_p1d`, the `_tp` points) and alternatives (`b*a_*`: m3 vs m4s4, sl on/off,
    mpc, z11).
- **Adversarial search** (new `audit/adv_search.py`, section 7) at log_w 2-6 for the frontier points,
  and at log_w 6 for every ladder rung.
  - All messages found at log_w 6 form `refs/advall_w6.pt` (349 messages) and `refs/advall2_w6.pt`
    (429, after more searches). The reference outputs come from a fresh process (new
    `audit/ref_msgs.py`).
  - Every log_w-6 variant was checked on the first set, and the 25 main ones on the second.
  - The adversarial results are reported next to the brief's criterion, not folded into it.
- 90 variants x 7 word sizes; every run is one line of `results.jsonl`.


## 2. What changes with the word size

The count ranges that the parity forms work on are fixed by the 5 x 5 structure (5-bit columns,
10-bit column pairs, the lazy chi's [0, 34] counts), so they do not depend on w. Three things do:

1. **How many round-1 bits are live message bits.** At log_w 0-1 the capacity is 10 of the 25 lanes
   (c = 10, 20) and the 8 suffix bits fill 8 or 4 whole lanes. From log_w 2 on the capacity is 7 lanes
   and lane (x=2, y=3) holds the suffix: fully at log_w 3, partly from log_w 4 on (8, 24, 56 live bits
   before it). That sets the sizes of the round-1 raw parity sets.
2. **The digest length.** d = 5w at log_w 0-1 (the whole row y = 0), d = 3.5w from log_w 2 on.
3. **The number of bits per message** (7 to 1144) and output bits (15 to 672). Worst float32 errors
   are maxima over these (section 5).

| log_w | w | state | msg bits | digest d | d / w | capacity lanes | live lanes per column x=0..4 | column-pair live counts T (number of columns) | round-1 raw set sizes T+own of live bits (number of bits) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 25 | 7 | 5 | 5 | 10 | 2 2 1 1 1 | 2: 1, 3: 4 | 3: 1, 4: 6 |
| 1 | 2 | 50 | 22 | 10 | 5 | 10 | 3 2 2 2 2 | 4: 6, 5: 4 | 5: 14, 6: 8 |
| 2 | 4 | 100 | 64 | 14 | 3.5 | 7 | 4 3 3 3 3 | 6: 12, 7: 8 | 7: 40, 8: 24 |
| 3 | 8 | 200 | 136 | 28 | 3.5 | 7 | 4 4 3 3 3 | 6: 8, 7: 32 | 7: 24, 8: 112 |
| 4 | 16 | 400 | 280 | 56 | 3.5 | 7 | 4 4 3+8/16 3 3 | 6: 8, 7: 64, 8: 8 | 7: 24, 8: 224, 9: 32 |
| 5 | 32 | 800 | 568 | 112 | 3.5 | 7 | 4 4 3+24/32 3 3 | 6: 8, 7: 128, 8: 24 | 7: 24, 8: 448, 9: 96 |
| 6 | 64 | 1600 | 1144 | 224 | 3.5 | 7 | 4 4 3+56/64 3 3 | 6: 8, 7: 256, 8: 56 | 7: 24, 8: 896, 9: 224 |

(`structure.py`.) Rotation offsets mod w only permute bits and change no count range. Every builder
handled them: nothing crashed or computed a wrong bit at any log_w.

Consequences, all measured below:
- round-1 min-parity (`mp`, 7- and 9-bit sets), the even raw forms (`ra`, 8-bit sets), `mpx` and
  `c1_mp17` do nothing at log_w 0-1, where no raw set has more than 6 bits;
- in the split / u1 round 1 (depths 7, 8) the raw sets are the column pairs alone (T bits), so `ra`
  only acts on T = 8, which exists from log_w 4 on;
- the round-1 pool (S6) beats the direct forms only on T = 8 columns, by 3 units each (section 9);
- `sl` saves features only when d < 5w.

## 3. Gains per strategy and word size

Percent change of dense against the previous rung of the ladder (a rung adds one flag). A cell is
marked `*` when the circuit after the flag is exact but float32-tight on the random sets
(0.01 < worst < 0.02), and `!` when its worst error is >= 0.02. Unmarked cells are robust by the
brief's criterion (for the adversarial picture see section 7).

| strategy | family | w0 | w1 | w2 | w3 | w4 | w5 | w6 |
|---|---|---|---|---|---|---|---|---|
| S1 glu xor | gated | -86.6% | -85.0% | -85.6% | -85.6% | -85.7% | -85.7% | -85.7% |
| S1 chi+iota unit | gated | -41.2% | -49.1% | -47.9% | -47.9% | -47.8% | -47.8% | -47.8% |
| S2+S3 direct layout (passes, shared parities, counts) | layout | -69.4% | -68.7% | -64.8% | -63.9% | -63.3% | -63.0% | -62.9% |
| S3 split rounds (d8) | layout | -23.9% | -26.4% | -30.5% | -31.1% | -31.8% | -32.2% | -32.3% |
| S3 lazy chi (d6) | layout | -24.0% | -22.6% | -22.4% | -22.0% | -21.7% | -21.6% | -21.5% |
| S3 fold layout rp (d8) | b8 | -2.9% | -2.5% | -4.9% | -4.8% | -4.8% | -4.8% | -4.8% |
| S4 cp | b8 | -19.9% | -17.9% | -15.5% | -14.7% | -14.4% | -14.2% | -14.1% |
| S4 u1 | b8 | -1.0% | -3.7% | -5.0% | -5.6% | -5.9% | -6.1% | -6.1% |
| S4 sp (alt. to u1) | b8 | -0.7% | -2.3% | -3.6% | -4.5% | -4.9% | -5.1% | -5.2% |
| S5 mp | b8 | +0.0% | +0.0% | -0.7% | -1.5% | -1.5% | -1.6% | -1.6% |
| S4 m4s4 | b8 | -2.5% | -2.6% | -1.9% | -2.2% | -2.1% | -2.1% | -2.1% |
| S4 m3 (alt. to m4s4) | b8 | -3.2% | -0.0% | -0.9% | -0.8% | -1.1% | -1.0% | -1.1% |
| S4 sl | b8 | +0.0% | +0.1% | -0.8% | -0.8% | -0.8% | -0.8% | -0.8% |
| S5 p1dct | b8 | -2.1% | -2.6% | -2.6% | -2.6% | -2.5% | -2.5% | -2.5% |
| S5 ra | b8 | +0.0% | +0.0% | +0.0% | +0.0% | -0.2% | -0.3% | -0.4% |
| S5 mpc (parity layer, not robust at w6) | b8 | -1.5% | -1.4% | -1.1%* | -1.1%* | -1.0%* | -1.0%! | -1.0%! |
| S4 cp | b7 | -17.8% | -16.0% | -14.0% | -13.3% | -13.0% | -12.8% | -12.7% |
| S5 c5 | b7 | -1.2% | -1.2% | -1.3% | -1.3% | -1.3% | -1.3% | -1.2% |
| S5 z11 (alt. to c5) | b7 | -2.7% | -2.6% | -2.4% | -2.3% | -2.3% | -2.3% | -2.3% |
| S5 mpc | b7 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| S4 u1 | b7 | -0.9% | -3.3% | -4.5% | -5.1% | -5.3% | -5.5% | -5.5% |
| S4 sp (alt. to u1) | b7 | -0.6% | -2.0% | -3.3% | -4.0% | -4.4% | -4.6% | -4.7% |
| S5 mp | b7 | +0.0% | +0.0% | -0.6% | -1.3% | -1.4% | -1.4% | -1.4% |
| S4 m4s4 | b7 | -2.0% | -2.1% | -1.7% | -2.0% | -1.9% | -1.9% | -1.9% |
| S4 sl | b7 | +0.0% | +0.1% | -1.4% | -1.4% | -1.4% | -1.4% | -1.4% |
| S5 p1dct | b7 | -1.8% | -2.3% | -2.3% | -2.3% | -2.3% | -2.3% | -2.3% |
| S5 ra | b7 | +0.0% | +0.0% | +0.0% | +0.0% | -0.2% | -0.3% | -0.3% |
| S5 bt | b7 | -1.2% | -1.4% | -1.1% | -1.1% | -1.1% | -1.1% | -1.1% |
| S4 lz5 (tight at w6) | b7 | -0.7% | -1.4% | -1.5% | -1.6% | -1.6% | -1.6%* | -1.6%* |
| S4 cp | b6 | -13.0% | -12.6% | -11.5% | -11.1% | -10.9% | -10.7% | -10.7% |
| S5 c5 | b6 | -1.2% | -1.2% | -1.2% | -1.1% | -1.1% | -1.1% | -1.1% |
| S5 z11 (alt. to c5) | b6 | -2.7% | -2.6% | -2.2% | -2.1% | -2.1% | -2.0% | -2.0% |
| S5 mpc | b6 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| S4 lz5 | b6 | -0.4% | -1.0% | -1.0% | -1.0% | -1.0% | -1.0% | -1.0% |
| S5 mp | b6 | +0.0% | +0.0% | -4.4% | -2.6% | -2.8% | -2.9% | -3.0% |
| S4 m4s4 | b6 | -2.4% | -2.5% | -1.8% | -2.0% | -1.9% | -1.9% | -1.9% |
| S4 sl | b6 | +0.0% | +0.1% | -1.5% | -1.4% | -1.4% | -1.4% | -1.4% |
| S5 p1dct | b6 | -1.9% | -2.2% | -2.1% | -2.0% | -2.0% | -1.9% | -1.9% |
| S5 ra | b6 | +0.0% | +0.0% | -2.5% | -5.8% | -6.0% | -6.1% | -6.2% |
| S5 bt | b6 | -1.1% | -1.2% | -0.9% | -0.9% | -0.9% | -0.8% | -0.8% |
| S6 tr (alt. to ra) | b6 | +3.3% | +1.1% | -2.5% | -5.8%* | -6.6%* | -7.0%* | -7.3%* |
| S6 tr vs ra (with bt) | b6 | +3.3% | +1.1% | +0.0% | +0.0%* | -0.7%* | -1.0%* | -1.2%* |
| S6 tp vs ra (with bt) | b6 | +4.3% | +2.5% | +2.0% | +2.2% | +1.3% | +0.9% | +0.7% |
| S6 t8: pool only on T = 8 columns (new) vs ra, with bt | b6 | +0.0% | +0.0% | +0.0% | +0.0% | -0.7% | -1.0% | -1.2% |
| S4 cp | b5 | -8.1% | -8.1% | -8.9% | -8.7% | -8.6% | -8.5% | -8.4% |
| S5 mp | b5 | +0.0% | +0.0% | -3.2% | -2.0% | -2.1% | -2.2% | -2.3% |
| S5 mpc (Walsh) | b5 | -1.7% | -1.6% | -1.0% | -0.9% | -0.9% | -0.9% | -0.9% |
| S5 p1dxt | b5 | -3.9% | -4.2% | -4.1% | -3.9% | -3.9% | -3.8% | -3.8% |
| S5 ra | b5 | +0.0% | +0.0% | -1.8% | -4.3% | -4.5% | -4.6% | -4.6% |
| S6 tr (alt. to ra) | b5 | +1.9% | +0.7% | -1.8%* | -4.3%* | -4.9%* | -5.2%* | -5.4%* |
| S6 t8 (new) vs ra | b5 | +0.0% | +0.0% | +0.0% | +0.0% | -0.5% | -0.7% | -0.9% |
| sl guard (with_slast_auto) vs sl | b8 | +0.0% | -0.1% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| S5 mpx | b4 | +0.0% | -2.0% | -4.8% | -3.5% | -3.3% | -3.4% | -3.5% |
| S5 wmpc | b4 | -1.4% | -1.2% | -0.6% | -0.6% | -0.6% | -0.5% | -0.5% |
| S5 p1d | b4 | -0.5% | -2.6% | -3.5% | -5.3% | -5.5% | -5.3% | -5.2% |
| S5 kt_xc | b4 | -3.6% | -3.7% | -3.4% | -2.9% | -2.8% | -2.8% | -2.7% |
| S5 xca | b4 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| S5 c1_mp17 | b3 | +0.0% | -1.3% | -2.6% | -1.9% | -1.8% | -1.9% | -1.9%* |
| S5 p1d | b3 | -0.2% | -1.4% | -1.9% | -2.8% | -3.0% | -2.9% | -2.8%* |
| S5 kt_xc | b3 | -3.6% | -3.5% | -3.4% | -3.1% | -3.1% | -3.0% | -3.0%* |
| S5 xca | b3 | +0.0% | +0.0% | -0.4% | -0.3% | -0.3% | -0.3% | -0.3% |
| S5 p1d on sp (sparse end) | sp | -2.4% | -2.7% | -2.7% | -2.7% | -2.7% | -2.7% | -2.7% |

Reading the table:
- **Flat across w (log_w 2-6 within 0.5 points)**: every S3 layout, `cp`, `c5` / `z11`, `lz5`,
  `m4s4`, `p1dct`, `p1dxt`, `bt`, `wmpc`, `kt_xc`, `p1d` on sp. These act on the count ranges of the
  5 x 5 structure and on per-column packing, which do not change with w.
- **Stop paying at small w**:
  - `mp`, `mpx`, `c1_mp17`: 0 at log_w 0-1 (`mpx` 0 at log_w 0 only), because no raw set reaches 7 bits;
  - `ra`: 0 at log_w 0-1. In the split / u1 families (b8, b7) it is 0 up to log_w 3 and -0.2..-0.4% from
    log_w 4 on (T = 8 columns only);
  - `sl`: +0.0% dense at log_w 0 and +0.1% dense / +2.5% sparse at log_w 1. There d = 5w, so the last
    theta emits 5w gate features in place of 5w theta bits, and the gates read more nonzeros;
  - `u1` / `sp`: -1.0% / -0.7% at log_w 0, growing to -6.1% / -5.2% at log_w 6 (fewer same-column
    live-bit pairs at small w);
  - `m4s4`: -2.5% at log_w 0, where `m3` is better (-3.2%): a 5-bit digest packs 3 + 2 better than 4 + 1;
  - S6 pools (`tr`, `tp`): cost dense at log_w 0-1 (+1.1..+4.3%). `tr` gains nothing at log_w 2-3 and
    only -0.7..-1.2% from log_w 4 on;
  - `xca` never changes dense (it removes nonzeros).
- **Pay more at small w**: `cp` (-19.9% at log_w 0 vs -14.1% at log_w 6 on b8) and the direct layout
  (-69% vs -63%), since more of the small states are constants.

## 4. Robust frontier per word size (the brief's criterion)

Best dense per depth, and the sparsest robust point per depth when it differs, from 90 variants per
log_w. Worst float32 errors are for the audit / mixed / dense sets of that log_w; "exh" is the worst over
all messages (log_w 0-1), "big" the worst over the 1792 + 1248-message sets (log_w 5-6). A point
counts as robust only if these extra checks pass too. The adversarial column of section 7 is **not**
part of this table's criterion.

### log_w 0 (baseline depth 17, dense 826,914, sparse 31,304)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_3_kt_xc` | 51,814 | 7,661 | 16.0 / 4.1 | 2.0e-04/2.0e-04/2.0e-04; exh 2.0e-04 |
| 4 | dense+sparse | `ladder:b4_4_kt_xc` | 25,113 | 3,991 | 32.9 / 7.8 | 4.6e-04/9.4e-04/5.2e-04; exh 9.4e-04 |
| 5 | dense | `ladder:b5_4_p1dxt` | 21,117 | 3,026 | 39.2 / 10.3 | 1.4e-04/1.5e-04/1.4e-04; exh 1.5e-04 |
| 5 | sparse | `ladder:b5_0` | 24,312 | 3,015 | 34.0 / 10.4 | 2.9e-05/4.0e-05/2.6e-05; exh 4.0e-05 |
| 6 | dense | `ladder:b6a_m3` | 12,097 | 1,798 | 68.4 / 17.4 | 1.8e-04/1.8e-04/1.8e-04; exh 1.8e-04 |
| 6 | sparse | `ladder:b6_2_c5` | 13,001 | 1,716 | 63.6 / 18.2 | 2.2e-04/2.4e-04/2.0e-04; exh 3.8e-04 |
| 7 | dense | `ladder:b7a_z11_m3` | 12,378 | 1,880 | 66.8 / 16.7 | 1.9e-04/2.3e-04/1.9e-04 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 16,872 | 1,615 | 49.0 / 19.4 | 7.6e-05/1.0e-04/7.8e-05; exh 1.0e-04 |
| 8 | dense | `ladder:b8a_m3_mpc` | 10,880 | 1,557 | 76.0 / 20.1 | 4.9e-04/7.7e-04/5.2e-04; exh 1.1e-03 |
| 8 | sparse | `c2:sp_base_p1d` | 15,322 | 1,334 | 54.0 / 23.5 | 3.7e-05/3.4e-05/3.7e-05; exh 3.7e-05 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 15,678 | 1,350 | 52.7 / 23.2 | 2.8e-05/4.7e-05/3.7e-05; exh 4.7e-05 |

### log_w 1 (baseline depth 20, dense 3,341,810, sparse 64,170)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_4_xca` | 219,543 | 21,430 | 15.2 / 3.0 | 2.0e-04/3.0e-04/1.9e-04; exh 6.4e-04 |
| 4 | dense+sparse | `ladder:b4_5_xca` | 106,615 | 10,453 | 31.3 / 6.1 | 8.3e-04/1.2e-03/1.1e-03; exh 3.6e-03 |
| 5 | dense+sparse | `ladder:b5_5_ra` | 83,073 | 6,577 | 40.2 / 9.8 | 1.2e-04/2.7e-04/3.6e-04; exh 8.3e-04 |
| 6 | dense | `ladder:auto_d6` | 49,892 | 4,172 | 67.0 / 15.4 | 1.4e-04/2.5e-04/3.2e-04; exh 7.2e-04 |
| 6 | sparse | `ladder:b6_2_c5` | 53,452 | 3,971 | 62.5 / 16.2 | 4.0e-04/5.5e-04/4.0e-04 |
| 7 | dense | `ladder:b7_11_lz5` | 48,065 | 4,103 | 69.5 / 15.6 | 1.6e-03/7.0e-04/1.2e-03; exh 3.8e-03 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 65,760 | 3,356 | 50.8 / 19.1 | 1.0e-04/1.7e-04/1.6e-04; exh 6.5e-04 |
| 8 | dense | `ladder:auto_d8` | 42,980 | 3,323 | 77.8 / 19.3 | 2.2e-04/3.0e-04/4.3e-04; exh 7.8e-04 |
| 8 | sparse | `c2:sp_base_p1d` | 59,304 | 2,821 | 56.4 / 22.7 | 4.0e-05/6.0e-05/7.1e-05; exh 3.6e-04 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 61,755 | 2,826 | 54.1 / 22.7 | 3.7e-05/4.6e-05/5.8e-05; exh 4.6e-04 |

### log_w 2 (baseline depth 20, dense 12,978,468, sparse 127,268)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_4_xca` | 863,595 | 60,768 | 15.0 / 2.1 | 1.6e-04/6.5e-04/4.9e-04 |
| 4 | dense | `ladder:b4_5_xca` | 445,330 | 28,242 | 29.1 / 4.5 | 1.5e-03/1.4e-03/1.1e-03 |
| 4 | sparse | `xc:d4a_m4k3_mpc_p1dxt_tp` | 560,780 | 24,108 | 23.1 / 5.3 | 1.5e-03/4.3e-03/4.3e-03 |
| 5 | dense | `ladder:b5_5_ra` | 281,841 | 13,230 | 46.0 / 9.6 | 6.2e-04/1.0e-03/2.5e-03 |
| 5 | sparse | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 285,841 | 13,124 | 45.4 / 9.7 | 2.3e-03/3.8e-03/4.5e-03 |
| 6 | dense | `ladder:b6_10_bt` | 201,773 | 10,240 | 64.3 / 12.4 | 5.5e-04/9.4e-04/1.5e-03 |
| 6 | sparse | `c2:d6_lazy_cp_p1dct_tp_bt` | 236,421 | 9,582 | 54.9 / 13.3 | 1.1e-03/3.9e-03/2.3e-03 |
| 7 | dense | `ladder:b7_11_lz5` | 186,980 | 8,095 | 69.4 / 15.7 | 4.0e-03/5.6e-03/4.0e-03 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 261,132 | 6,967 | 49.7 / 18.3 | 2.6e-04/1.2e-03/1.3e-03 |
| 8 | dense | `ladder:b8_7_ra` | 171,632 | 6,960 | 75.6 / 18.3 | 3.5e-03/3.7e-03/3.7e-03 |
| 8 | sparse | `c2:sp_base_p1d` | 239,850 | 5,880 | 54.1 / 21.6 | 3.1e-04/3.6e-04/3.6e-04 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 255,845 | 5,759 | 50.7 / 22.1 | 9.5e-05/1.1e-04/9.1e-05 |

### log_w 3 (baseline depth 20, dense 51,771,764, sparse 254,436)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_4_xca` | 3,587,446 | 136,550 | 14.4 / 1.9 | 4.5e-04/2.4e-03/7.7e-04 |
| 4 | dense | `ladder:b4_5_xca` | 1,847,194 | 60,569 | 28.0 / 4.2 | 1.6e-03/2.1e-03/1.6e-03 |
| 4 | sparse | `xc:d4a_m4k3_mpc_p1dxt_tp` | 2,241,706 | 48,961 | 23.1 / 5.2 | 1.7e-03/3.0e-03/2.5e-03 |
| 5 | dense | `ladder:b5_5_ra` | 1,134,817 | 27,352 | 45.6 / 9.3 | 2.5e-03/3.1e-03/2.0e-03 |
| 5 | sparse | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 1,155,821 | 26,892 | 44.8 / 9.5 | 2.2e-03/6.4e-03/5.3e-03 |
| 6 | dense | `ladder:b6_10_bt` | 817,350 | 21,330 | 63.3 / 11.9 | 1.6e-03/2.1e-03/2.0e-03 |
| 6 | sparse | `c2:d6_lazy_cp_p1dct_tp_bt` | 959,697 | 19,779 | 53.9 / 12.9 | 1.8e-03/3.2e-03/3.6e-03 |
| 7 | dense | `ladder:b7_11_lz5` | 748,325 | 16,489 | 69.2 / 15.4 | 2.2e-03/2.9e-03/4.3e-03 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 1,053,584 | 14,142 | 49.1 / 18.0 | 9.4e-04/2.3e-03/1.8e-03 |
| 8 | dense | `ladder:b8_7_ra` | 687,727 | 14,199 | 75.3 / 17.9 | 1.3e-03/2.7e-03/2.7e-03 |
| 8 | sparse | `c2:sp_base_p1d` | 968,036 | 11,892 | 53.5 / 21.4 | 4.1e-04/8.2e-04/6.7e-04 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 1,030,259 | 11,467 | 50.3 / 22.2 | 6.9e-05/7.6e-05/1.0e-04 |

### log_w 4 (baseline depth 20, dense 206,803,140, sparse 508,772)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_4_xca` | 14,598,761 | 289,549 | 14.2 / 1.8 | 1.6e-03/2.7e-03/3.1e-03 |
| 4 | dense | `ladder:b4_5_xca` | 7,589,677 | 127,032 | 27.2 / 4.0 | 1.6e-03/3.5e-03/1.8e-03 |
| 4 | sparse | `xc:d4a_m4k3_mpc_p1dxt_tp` | 8,968,864 | 98,550 | 23.1 / 5.2 | 1.7e-03/6.2e-03/3.8e-03 |
| 5 | dense | `ladder:b5_5e_t8` | 4,567,917 | 55,344 | 45.3 / 9.2 | 2.8e-03/3.4e-03/5.8e-03 |
| 5 | sparse | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 4,635,770 | 54,376 | 44.6 / 9.4 | 2.6e-03/4.7e-03/4.4e-03 |
| 6 | dense | `ladder:b6_10e_t8_bt` | 3,303,744 | 43,293 | 62.6 / 11.8 | 2.8e-03/4.7e-03/6.0e-03 |
| 6 | sparse | `c2:d6_lazy_cp_p1dct_tp_bt` | 3,868,191 | 40,145 | 53.5 / 12.7 | 3.0e-03/4.4e-03/2.8e-03 |
| 7 | dense | `ladder:b7_11_lz5` | 2,999,447 | 33,140 | 68.9 / 15.4 | 4.8e-03/5.0e-03/5.0e-03 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 4,226,774 | 28,392 | 48.9 / 17.9 | 1.2e-03/2.0e-03/2.1e-03 |
| 8 | dense | `ladder:b8_7_ra` | 2,758,126 | 28,553 | 75.0 / 17.8 | 1.2e-03/4.2e-03/5.1e-03 |
| 8 | sparse | `c2:sp_base_p1d` | 3,883,754 | 23,893 | 53.2 / 21.3 | 5.2e-04/1.0e-03/8.0e-04 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 4,135,865 | 22,932 | 50.0 / 22.2 | 4.9e-05/1.3e-04/1.1e-04 |

### log_w 5 (baseline depth 20, dense 826,645,028, sparse 1,017,444)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_4_xca` | 59,316,760 | 606,201 | 13.9 / 1.7 | 2.0e-03/3.9e-03/3.4e-03; big 4.8e-03 |
| 4 | dense | `ladder:b4_5_xca` | 30,839,576 | 260,912 | 26.8 / 3.9 | 2.3e-03/5.2e-03/2.2e-03; big 5.4e-03 |
| 4 | sparse | `xc:d4a_m4k3_mpc_p1dxt_tp` | 35,915,961 | 197,732 | 23.0 / 5.1 | 3.7e-03/4.1e-03/4.1e-03 |
| 5 | dense | `ladder:b5_5e_t8` | 18,330,763 | 111,282 | 45.1 / 9.1 | 2.3e-03/6.8e-03/5.1e-03; big 9.0e-03 |
| 5 | sparse | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 18,596,897 | 109,322 | 44.5 / 9.3 | 6.3e-03/7.5e-03/5.4e-03 |
| 6 | dense | `ladder:b6_10e_t8_bt` | 13,285,482 | 87,185 | 62.2 / 11.7 | 2.7e-03/6.6e-03/6.0e-03; big 8.8e-03 |
| 6 | sparse | `c2:d6_lazy_cp_p1dct_tp_bt` | 15,532,947 | 80,831 | 53.2 / 12.6 | 2.3e-03/5.4e-03/3.3e-03 |
| 7 | dense | `ladder:b7_10_bt` | 12,201,361 | 64,336 | 67.8 / 15.8 | 3.3e-03/9.6e-03/6.0e-03; big 9.6e-03 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 16,933,178 | 56,846 | 48.8 / 17.9 | 1.1e-03/2.0e-03/2.6e-03 |
| 8 | dense | `ladder:b8_7_ra` | 11,047,168 | 57,215 | 74.8 / 17.8 | 1.9e-03/4.5e-03/3.8e-03; big 4.8e-03 |
| 8 | sparse | `c2:sp_base_p1d` | 15,559,454 | 47,777 | 53.1 / 21.3 | 4.4e-04/8.0e-04/1.0e-03 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 16,574,189 | 45,744 | 49.9 / 22.2 | 7.4e-05/1.4e-04/1.4e-04; big 1.5e-04 |

### log_w 6 (baseline depth 20, dense 3,305,445,348, sparse 2,034,788)

| depth | kind | variant | dense | sparse | x base dense / sparse | worst f32 (audit/mixed/dense) |
|---|---|---|---|---|---|---|
| 3 | dense+sparse | `ladder:b3_4_xca` | 238,512,173 | 1,235,283 | 13.9 / 1.6 | 4.3e-03/7.7e-03/9.7e-03; big 9.7e-03 |
| 4 | dense | `ladder:b4_5_xca` | 124,087,564 | 526,518 | 26.6 / 3.9 | 5.6e-03/6.5e-03/2.6e-03; big 6.5e-03 |
| 4 | sparse | `xc:d4a_m4k3_mpc_p1dxt_tp` | 143,675,838 | 396,090 | 23.0 / 5.1 | 2.2e-03/4.6e-03/4.0e-03 |
| 5 | dense | `ladder:b5_5e_t8` | 73,443,039 | 223,176 | 45.0 / 9.1 | 3.7e-03/6.2e-03/7.0e-03; big 7.2e-03 |
| 5 | sparse | `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 74,441,726 | 219,208 | 44.4 / 9.3 | 5.4e-03/8.0e-03/6.3e-03 |
| 6 | dense | `ladder:b6_10e_t8_bt` | 53,284,758 | 174,973 | 62.0 / 11.6 | 5.2e-03/4.9e-03/3.9e-03; big 7.6e-03 |
| 6 | sparse | `c2:d6_lazy_cp_p1dct_tp_bt` | 62,253,531 | 162,221 | 53.1 / 12.5 | 3.0e-03/3.5e-03/3.9e-03 |
| 7 | dense | `ladder:b7_10_bt` | 48,831,997 | 128,804 | 67.7 / 15.8 | 4.5e-03/5.6e-03/7.5e-03; big 7.9e-03 |
| 7 | sparse | `c2:sp_lazy2_l3_p1d` | 67,786,082 | 113,772 | 48.8 / 17.9 | 3.0e-03/2.6e-03/2.8e-03 |
| 8 | dense | `ladder:b8_7_ra` | 44,218,228 | 114,557 | 74.8 / 17.8 | 3.4e-03/4.9e-03/4.1e-03; big 5.2e-03 |
| 8 | sparse | `c2:sp_base_p1d` | 62,287,910 | 95,637 | 53.1 / 21.3 | 1.0e-03/1.3e-03/1.4e-03 |
| 9 | dense+sparse | `c2:sp_col1b_p1d` | 66,359,285 | 91,460 | 49.8 / 22.2 | 1.1e-04/1.3e-04/1.5e-04; big 1.9e-04 |

Dense ratio to the threshold baseline of the same log_w (best robust point per depth), and the
sparse ratio of the sparse end:

| depth | log_w 0 | log_w 1 | log_w 2 | log_w 3 | log_w 4 | log_w 5 | log_w 6 |
|---|---|---|---|---|---|---|---|
| 3 | 16.0x | 15.2x | 15.0x | 14.4x | 14.2x | 13.9x | 13.9x |
| 4 | 32.9x | 31.3x | 29.1x | 28.0x | 27.2x | 26.8x | 26.6x |
| 5 | 39.2x | 40.2x | 46.0x | 45.6x | 45.3x | 45.1x | 45.0x |
| 6 | 68.4x | 67.0x | 64.3x | 63.3x | 62.6x | 62.2x | 62.0x |
| 7 | 66.8x | 69.5x | 69.4x | 69.2x | 68.9x | 67.8x | 67.7x |
| 8 | 76.0x | 77.8x | 75.6x | 75.3x | 75.0x | 74.8x | 74.8x |
| 9 | 52.7x | 54.1x | 50.7x | 50.3x | 50.0x | 49.9x | 49.8x |
| sparse end (sparse ratio) | 23.5x (d8) | 22.7x (d8) | 22.1x (d9) | 22.2x (d9) | 22.2x (d9) | 22.2x (d9) | 22.2x (d9) |

- The depth-8 point stays at 75-79x from log_w 0 to 6, depth 7 at 67-70x, depth 6 at 62-68x, and the
  sparse end at 22-24x sparse. log_w 0-1 get 1-4% more from choices that do not hold at larger w
  (`m3`, no `sl`, `mpc` in the d8 parity layer at log_w 0, `lz5` up to log_w 4).
- At log_w 6 this reproduces the README's robust frontier (same variants, same sizes; the big-set
  errors match the README to two digits) except at depths 5 and 6, where the new `t8` pool form
  (section 9) passes the criterion with
  - d5 `ladder:b5_5e_t8`: 73,443,039 / 223,176 (README robust: 74,073,879 / 226,480; big sets 7.2e-3);
  - d6 `ladder:b6_10e_t8_bt`: 53,284,758 / 174,973 (README robust: 53,915,598 / 178,277; big sets 7.6e-3).
  Both fail the adversarial check (section 7), so I do not recommend them over the README points.

## 5. Float32 worst error against the word size

Worst |out/BOS - bit| over harness + audit + mixed + dense per log_w (984 stress messages each);
in brackets the worst over all messages (log_w 0-1) or over the big sets (log_w 5-6):

| variant | log_w 0 | log_w 1 | log_w 2 | log_w 3 | log_w 4 | log_w 5 | log_w 6 |
|---|---|---|---|---|---|---|---|
| `ladder:b3_4_xca` | 2.0e-04 [all msgs: 2.0e-04] | 3.0e-04 [all msgs: 6.4e-04] | 6.5e-04 | 2.4e-03 | 3.1e-03 | 3.9e-03 [big: 4.8e-03] | 9.7e-03 [big: 9.7e-03] |
| `ladder:b4_5_xca` | 9.4e-04 [all msgs: 9.4e-04] | 1.2e-03 [all msgs: 3.6e-03] | 1.5e-03 | 2.1e-03 | 3.5e-03 | 5.2e-03 [big: 5.4e-03] | 6.5e-03 [big: 6.5e-03] |
| `ladder:b5_5_ra` | 1.5e-04 [all msgs: 1.5e-04] | 3.6e-04 [all msgs: 8.3e-04] | 2.5e-03 | 3.1e-03 | 3.5e-03 | 4.0e-03 [big: 4.3e-03] | 4.4e-03 [big: 4.4e-03] |
| `ladder:b5_5e_t8` | 1.5e-04 [all msgs: 1.5e-04] | 3.6e-04 [all msgs: 8.3e-04] | 2.5e-03 | 3.1e-03 | 5.8e-03 | 6.8e-03 [big: 9.0e-03] | 7.0e-03 [big: 7.2e-03] |
| `ladder:b5_5t_tr` | 3.4e-04 [all msgs: 3.4e-04] | 3.9e-03 [all msgs: 5.0e-03] | 1.1e-02 tight | 1.4e-02 tight | 1.5e-02 tight | 1.2e-02 tight | 1.6e-02 tight |
| `ladder:b6_10_bt` | 1.8e-04 [all msgs: 1.8e-04] | 3.2e-04 [all msgs: 7.2e-04] | 1.5e-03 | 2.1e-03 | 3.4e-03 | 3.6e-03 [big: 3.6e-03] | 2.5e-03 [big: 3.6e-03] |
| `ladder:b6_10e_t8_bt` | 1.8e-04 [all msgs: 1.8e-04] | 3.2e-04 [all msgs: 7.2e-04] | 1.5e-03 | 2.1e-03 | 6.0e-03 | 6.6e-03 [big: 8.8e-03] | 5.2e-03 [big: 7.6e-03] |
| `ladder:b6_10t_tr_bt` | 3.6e-04 [all msgs: 3.6e-04] | 2.3e-03 [all msgs: 5.5e-03] | 6.7e-03 | 1.0e-02 tight | 1.4e-02 tight | 1.0e-02 tight [big: 1.5e-02 tight] | 1.8e-02 tight [big: 1.8e-02 tight] |
| `ladder:b7_10_bt` | 1.2e-04 [all msgs: 1.2e-04] | 6.9e-04 [all msgs: 3.8e-03] | 5.6e-03 | 2.8e-03 | 5.1e-03 | 9.6e-03 [big: 9.6e-03] | 7.5e-03 [big: 7.9e-03] |
| `ladder:b7a_z11` | 2.4e-04 [all msgs: 2.4e-04] | 4.4e-04 [all msgs: 3.3e-03] | 3.7e-03 | 2.7e-03 | 5.0e-03 | 4.5e-03 [big: 4.7e-03] | 6.3e-03 [big: 6.3e-03] |
| `ladder:b7_11_lz5` | 1.2e-04 [all msgs: 1.2e-04] | 1.6e-03 [all msgs: 3.8e-03] | 5.6e-03 | 4.3e-03 | 5.0e-03 | 1.1e-02 tight | 1.1e-02 tight |
| `ladder:b8_7_ra` | 1.5e-04 [all msgs: 1.5e-04] | 4.3e-04 [all msgs: 7.8e-04] | 3.7e-03 | 2.7e-03 | 5.1e-03 | 4.5e-03 [big: 4.8e-03] | 4.9e-03 [big: 5.2e-03] |
| `ladder:b8_8_mpc` | 9.2e-04 [all msgs: 9.7e-04] | 2.8e-03 [all msgs: 1.3e-02 tight] | 1.3e-02 tight | 1.4e-02 tight | 1.6e-02 tight | 2.4e-02 FAIL | 3.3e-02 FAIL |
| `c2:sp_col1b_p1d` | 4.7e-05 [all msgs: 4.7e-05] | 5.8e-05 [all msgs: 4.6e-04] | 1.1e-04 | 1.0e-04 | 1.3e-04 | 1.4e-04 [big: 1.5e-04] | 1.5e-04 [big: 1.9e-04] |
| `ladder:b3_1_c1mp17` | 8.8e-05 [all msgs: 8.8e-05] | 5.1e-04 | 7.5e-04 | 4.8e-03 | 8.7e-03 | 8.6e-03 | 1.6e-02 tight |
| `ladder:b3_0` | 1.2e-04 [all msgs: 1.2e-04] | 2.6e-04 | 1.4e-03 | 5.4e-03 | 3.5e-03 | 9.8e-03 | 3.8e-02 FAIL, 5 misread |

- Errors grow with w for every layout, although no count range changes. A message at log_w 6 has 672
  outputs and 1144 inputs, so the rare worst local count patterns are hit far more often than with
  15 outputs and 7 inputs.
- A strategy that is float32-tight at log_w 6 is usually robust at small w; each crosses the threshold
  at its own w:
  - `mpc` in the d8 parity layer (MIN_PARITY[11]): 9.2e-4 at log_w 0, but 1.3e-2 over all messages at
    log_w 1, tight on the random sets from log_w 2, fails (>= 0.02) from log_w 5;
  - the S6 pool `tr` (F13): tight from log_w 2 (d5) / 3 (d6);
  - `lz5` at depth 7: robust on the random sets up to log_w 4, tight at 5-6;
  - the depth-3 fused rounds: the frontier `d3c1_mp17_p1d_kt_xca` passes at every w on the random sets,
    only just at log_w 6 (9.7e-3). Its lower rungs are tight at log_w 6, and the plain `dl3:d3` fails
    there with 5 boolify misreads (3.8e-2).
- `bt` at depth 7 doubles the log_w 5 error (4.5e-3 -> 9.6e-3 mixed). `z11` gives the same dense with
  4.5e-3 (`ladder:b7a_z11`, +1.3% nonzeros). Under adversarial search both exceed 0.01 at log_w 6
  (section 7).

## 6. Exhaustive checks at log_w 0 and 1: random sets underestimate the worst case

- log_w 0: all 91 variants on all 128 messages: 0 wrong bits, 0 misreads, worst error 1.05e-3
  (`ladder:b8a_m3_mpc`).
- log_w 1: 32 variants on all 4,194,304 messages. All are exact (0 wrong bits, 0 misreads), but the
  worst error over all messages is 1.3-7.9x the worst over the 1053 audit + stress messages:

| variant | depth | dense | worst, 1053 reference messages | worst, all 2^22 messages | ratio |
|---|---|---|---|---|---|
| `ladder:b5_5_ra` | 5 | 83,073 | 3.6e-04 | 8.3e-04, 0 wrong, 0 misread | 2.3x |
| `ladder:b4_5_xca` | 4 | 106,615 | 1.2e-03 | 3.6e-03, 0 wrong, 0 misread | 2.9x |
| `ladder:b6_10_bt` | 6 | 49,892 | 3.2e-04 | 7.2e-04, 0 wrong, 0 misread | 2.2x |
| `ladder:b7_10_bt` | 7 | 48,750 | 6.9e-04 | 3.8e-03, 0 wrong, 0 misread | 5.5x |
| `ladder:b7_11_lz5` | 7 | 48,065 | 1.6e-03 | 3.8e-03, 0 wrong, 0 misread | 2.3x |
| `ladder:b8_7_ra` | 8 | 43,039 | 4.3e-04 | 7.8e-04, 0 wrong, 0 misread | 1.8x |
| `ladder:b3_4_xca` | 3 | 219,543 | 3.0e-04 | 6.4e-04, 0 wrong, 0 misread | 2.1x |
| `ladder:b8_8_mpc` | 8 | 42,449 | 2.8e-03 | 1.3e-02 (tight), 0 wrong, 0 misread | 4.7x |
| `c2:sp_col1b_p1d` | 9 | 61,755 | 5.8e-05 | 4.6e-04, 0 wrong, 0 misread | 7.9x |
| `c2:sp_base_p1d` | 8 | 59,304 | 7.1e-05 | 3.6e-04, 0 wrong, 0 misread | 5.1x |
| `c2:sp_lazy2_l3_p1d` | 7 | 65,760 | 1.7e-04 | 6.5e-04, 0 wrong, 0 misread | 3.8x |
| `c2:d6_lazy_cp_p1dct_tp_bt` | 6 | 57,429 | 1.7e-03 | 4.2e-03, 0 wrong, 0 misread | 2.4x |
| `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | 5 | 84,977 | 5.3e-03 | 8.7e-03, 0 wrong, 0 misread | 1.6x |
| `ladder:auto_d8` | 8 | 42,980 | 4.3e-04 | 7.8e-04, 0 wrong, 0 misread | 1.8x |
| `ladder:auto_d7` | 7 | 48,750 | 6.8e-04 | 3.8e-03, 0 wrong, 0 misread | 5.6x |
| `xc:d4a_m4k3_mpc_p1dxt_tp` | 4 | 164,928 | 2.0e-03 | 4.9e-03, 0 wrong, 0 misread | 2.4x |
| `ladder:auto_d6` | 6 | 49,892 | 3.2e-04 | 7.2e-04, 0 wrong, 0 misread | 2.2x |
| `ladder:b6_10t_tr_bt` | 6 | 50,445 | 2.3e-03 | 5.5e-03, 0 wrong, 0 misread | 2.4x |
| `ladder:b5_5t_tr` | 5 | 83,626 | 3.9e-03 | 5.0e-03, 0 wrong, 0 misread | 1.3x |
| `ladder:b8a_m3_mpc` | 8 | 43,521 | 2.6e-03 | 1.3e-02 (tight), 0 wrong, 0 misread | 4.9x |
| `ladder:b8a_m3_sl_mpc` | 8 | 43,581 | 2.6e-03 | 1.3e-02 (tight), 0 wrong, 0 misread | 4.9x |
| `ladder:b8a_nosl` | 8 | 42,980 | 4.3e-04 | 7.8e-04, 0 wrong, 0 misread | 1.8x |
| `ladder:b8a_nosl_mpc` | 8 | 42,390 | 2.8e-03 | 1.3e-02 (tight), 0 wrong, 0 misread | 4.7x |
| `ladder:b8a_m3` | 8 | 44,121 | 2.9e-04 | 7.1e-04, 0 wrong, 0 misread | 2.4x |
| `ladder:b7a_z11` | 7 | 48,750 | 4.4e-04 | 3.3e-03, 0 wrong, 0 misread | 7.5x |
| `ladder:b7a_m3` | 7 | 49,661 | 6.6e-04 | 3.9e-03, 0 wrong, 0 misread | 5.8x |
| `ladder:b6a_nosl` | 6 | 49,892 | 3.2e-04 | 7.2e-04, 0 wrong, 0 misread | 2.2x |
| `ladder:b8_2_u1` | 8 | 45,312 | 1.4e-04 | 3.4e-04, 0 wrong, 0 misread | 2.3x |
| `ladder:b6_10e_t8_bt` | 6 | 49,892 | 3.2e-04 | 7.2e-04, 0 wrong, 0 misread | 2.2x |
| `ladder:b6a_m3` | 6 | 50,933 | 2.9e-04 | 6.9e-04, 0 wrong, 0 misread | 2.4x |
| `ladder:b5_5e_t8` | 5 | 83,073 | 3.6e-04 | 8.3e-04, 0 wrong, 0 misread | 2.3x |
| `ladder:b8_4_m4s4` | 8 | 44,140 | 4.4e-04 | 7.5e-04, 0 wrong, 0 misread | 1.7x |

- So a random-set margin of 2.8e-3 (`mpc` in the d8 parity layer) can hide a 1.3e-2 worst case:
  the d8 `mpc` points look robust on the random sets at log_w 1 and are not. The half-tolerance rule
  of the README covers the median ratio (2.3x) but not the tail (4.7-7.9x).
- The worst messages are not dense (11-16 ones of 22); they are specific combinations, which is why
  random densities rarely hit them.

## 7. Adversarial search (log_w 2-6): every README robust point of depth 3-8 fails at log_w 6

New `audit/adv_search.py`: an evolutionary search over messages that maximizes the network's worst
readout error |r - round(r)| (no reference needed while every readout is on the right side of 0.5),
seeded with the three reference sets. 32 parents x 8 mutants per generation (1-4 bit flips or a
segment copied from another parent), 150 generations (39K messages) or 1000 (257K). The 4 worst
messages of every run are then checked against the reference xof. All messages found at log_w 6
(349 in `refs/advall_w6.pt`, then 429 in `refs/advall2_w6.pt` after more searches; reference outputs
from a fresh process by the new `audit/ref_msgs.py`) were then run through every log_w-6 variant (the
first set) or the 25 main ones (the second set) with `multi_check.py`. The last column is the worst
over both sets.

| variant | log_w 2 (stress / search) | log_w 3 (stress / search) | log_w 4 (stress / search) | log_w 5 (stress / search) | log_w 6 (stress / search) | log_w 6, all adversarial msgs (349 + later finds) |
|---|---|---|---|---|---|---|
| `ladder:b8_7_ra` | 3.7e-03 / 4.2e-03 | 2.7e-03 / 4.2e-03 | 5.1e-03 / 7.2e-03 | 4.5e-03 / 8.7e-03 | 4.9e-03 / 1.0e-02 tight | 2.2e-02 FAIL |
| `ladder:b6_10e_t8_bt` | 1.5e-03 / 5.6e-03 | 2.1e-03 / 5.3e-03 | 6.0e-03 / 1.0e-02 tight | 6.6e-03 / 1.1e-02 tight | 5.2e-03 / 1.4e-02 tight | 1.4e-02 tight |
| `ladder:b5_5e_t8` | 2.5e-03 / 5.1e-03 | 3.1e-03 / 5.3e-03 | 5.8e-03 / 8.3e-03 | 6.8e-03 / 1.1e-02 tight | 7.0e-03 / 1.1e-02 tight | 1.1e-02 tight |
| `ladder:b7_10_bt` | 5.6e-03 / 6.9e-03 | 2.8e-03 / 7.6e-03 | 5.1e-03 / 6.9e-03 | 9.6e-03 / 9.6e-03 | 7.5e-03 / 2.8e-02 FAIL | 2.8e-02 FAIL |
| `ladder:b7a_z11` | 3.7e-03 / 3.7e-03 | 2.7e-03 / 4.1e-03 | 5.0e-03 / 8.2e-03 | 4.5e-03 / 7.3e-03 | 6.3e-03 / 1.6e-02 tight | 1.6e-02 tight |
| `ladder:b6_10_bt` | 1.5e-03 / 5.6e-03 | 2.1e-03 / 5.3e-03 | 3.4e-03 / 6.4e-03 | 3.6e-03 / 6.1e-03 | 2.5e-03 / 1.1e-02 tight | 1.1e-02 tight |
| `c2:sp_col1b_p1d` | 1.1e-04 / 3.0e-04 | 1.0e-04 / 4.9e-04 | 1.3e-04 / 2.3e-04 | 1.4e-04 / 3.6e-04 | 1.5e-04 / 3.5e-04 | 3.5e-04 |
| `ladder:b5_5_ra` | 2.5e-03 / 5.1e-03 | 3.1e-03 / 5.3e-03 | 3.5e-03 / 6.4e-03 | 4.0e-03 / 8.3e-03 | 4.4e-03 / 1.2e-02 tight | 9.1e-03 |
| `ladder:b4_5_xca` | 1.5e-03 / 2.2e-03 | 2.1e-03 / 7.2e-03 | 3.5e-03 / 9.1e-03 | 5.2e-03 / 8.1e-03 | 6.5e-03 / 2.6e-02 FAIL, 2 misread | 2.6e-02 FAIL, 1 misread |
| `ladder:b3_4_xca` | 6.5e-04 / 2.5e-03 | 2.4e-03 / 6.7e-03 | 3.1e-03 / 9.1e-03 | 3.9e-03 / 1.1e-02 tight | 9.7e-03 / 2.3e-02 FAIL | 2.3e-02 FAIL |
| `ladder:b8a_m3` | - | - | - | - | 2.3e-03 / 5.0e-03 | 2.0e-02 tight |
| `ladder:b8a_m3_sl` | - | - | - | - | 2.4e-03 / 2.2e-02 FAIL | 2.2e-02 FAIL |
| `ladder:b7a_z11_m3` | - | - | - | - | 6.2e-03 / 1.1e-02 tight | 1.6e-02 tight |
| `ladder:b4_4_kt_xc` | - | - | - | - | 6.7e-03 / 1.6e-02 tight | 1.6e-02 tight |
| `ladder:b8_4m_m3` | - | - | - | - | 3.0e-03 / 7.2e-03 | 7.7e-03 |
| `ladder:b7_5_mp` | - | - | - | - | 2.2e-03 / 6.5e-03 | 3.3e-03 |
| `ladder:b6_7_sl` | - | - | - | - | 2.2e-03 / 4.6e-03 | 4.1e-03 |
| `ladder:b4_3_p1d` | - | - | - | - | 3.7e-03 / 8.4e-03 | 3.7e-03 |
| `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` | - | - | - | - | 8.0e-03 / 3.6e-02 FAIL | 8.3e-03 |
| `ladder:b5_2_mp` | - | - | - | - | 2.2e-03 / 4.9e-03 | 5.0e-03 |
| `c2:sp_base_p1d` | - | - | - | - | 1.4e-03 / 3.5e-03 | 1.6e-03 |
| `c2:sp_lazy2_l3_p1d` | - | - | - | - | 3.0e-03 / 8.0e-03 | 2.7e-03 |
| `c2:d6_lazy_cp_p1dct_tp_bt` | - | - | - | - | 3.9e-03 / 1.2e-02 tight | 3.8e-03 |
| `xc:d4a_m4k3_mpc_p1dxt_tp` | - | - | - | - | 4.6e-03 / 2.4e-02 FAIL, 1 misread | 4.9e-03 |
| `ladder:b8_1_cp` | - | - | - | - | 5.0e-04 / 7.3e-04 | 1.0e-03 |
| `ladder:b8_0_rp` | - | - | - | - | 5.2e-04 / 5.9e-04 | 8.4e-04 |
| `xs3:split_first_middle` | - | - | - | - | 3.2e-04 / 7.2e-04 | 8.6e-04 |
| `ladder:b8_2_u1` | - | - | - | - | 2.1e-03 / 2.7e-03 | 2.7e-03 |
| `ladder:b8_3_mp` | - | - | - | - | 2.2e-03 / 2.4e-03 | 3.3e-03 |
| `ladder:b8_4_m4s4` | - | - | - | - | 5.9e-03 / 1.4e-02 tight | 1.6e-02 tight |
| `ladder:b8_5_sl` | - | - | - | - | 5.9e-03 / 1.2e-02 tight | 1.6e-02 tight |
| `ladder:b8_6_p1dct` | - | - | - | - | 5.9e-03 / 1.6e-02 tight | 1.6e-02 tight |
| `ladder:b8_8_mpc` | - | - | - | - | 3.3e-02 / 5.4e-02 FAIL | 5.4e-02 FAIL |
| `xs3:split_first_lazy4c` | - | - | - | - | 5.2e-04 / 8.2e-04 | 8.4e-04 |
| `ladder:b7_1_cp` | - | - | - | - | 5.6e-04 / 6.9e-04 | 1.0e-03 |
| `ladder:b7_2_c5` | - | - | - | - | 5.9e-04 / 9.9e-04 | 1.0e-03 |
| `ladder:b7_2z_z11` | - | - | - | - | 7.0e-04 / 1.1e-03 | 1.1e-03 |
| `ladder:b7_4_u1` | - | - | - | - | 2.1e-03 / 2.6e-03 | 2.7e-03 |
| `ladder:b7_6_m4s4` | - | - | - | - | 5.8e-03 / 1.6e-02 tight | 1.6e-02 tight |
| `ladder:b7_7_sl` | - | - | - | - | 5.8e-03 / 1.4e-02 tight | 1.6e-02 tight |
| `ladder:b7_8_p1dct` | - | - | - | - | 5.8e-03 / 1.6e-02 tight | 1.6e-02 tight |
| `ladder:b7_9_ra` | - | - | - | - | 4.8e-03 / 9.2e-03 | 1.0e-02 tight |
| `ladder:b7a_m3` | - | - | - | - | 7.5e-03 / 1.9e-02 tight | 2.5e-02 FAIL |
| `ladder:b7_11_lz5` | - | - | - | - | 1.1e-02 / 2.7e-02 FAIL, 1 misread | 2.8e-02 FAIL, 8 misread |
| `xs3:lazy4c_middle` | - | - | - | - | 3.7e-04 / 5.8e-04 | 5.8e-04 |
| `ladder:b6_1_cp` | - | - | - | - | 4.3e-04 / 5.3e-04 | 5.3e-04 |
| `ladder:b6_2_c5` | - | - | - | - | 5.9e-04 / 7.2e-04 | 7.2e-04 |
| `ladder:b6_4_lz5` | - | - | - | - | 1.6e-03 / 3.6e-03 | 3.6e-03 |
| `ladder:b6_5_mp` | - | - | - | - | 2.1e-03 / 3.2e-03 | 4.1e-03 |
| `ladder:b6_6_m4s4` | - | - | - | - | 2.2e-03 / 4.1e-03 | 4.1e-03 |
| `ladder:b6_8_p1dct` | - | - | - | - | 6.3e-03 / 7.8e-03 | 7.8e-03 |
| `ladder:b6_9_ra` | - | - | - | - | 2.5e-03 / 5.8e-03 | 7.9e-03 |
| `ladder:b6a_m3` | - | - | - | - | 2.5e-03 / 5.6e-03 | 7.9e-03 |
| `ladder:b6_10t_tr_bt` | - | - | - | - | 1.8e-02 / 2.8e-02 FAIL, 1 misread | 2.8e-02 FAIL, 8 misread |
| `ladder:b6_10p_tp_bt` | - | - | - | - | 8.2e-03 / 1.6e-02 tight | 1.6e-02 tight |
| `ladder:b5_0` | - | - | - | - | 9.6e-04 / 3.1e-03 | 3.1e-03 |
| `ladder:b5_1_cp` | - | - | - | - | 2.1e-03 / 2.8e-03 | 2.9e-03 |
| `ladder:b5_3_mpc` | - | - | - | - | 2.2e-03 / 4.9e-03 | 5.0e-03 |
| `ladder:b5_4_p1dxt` | - | - | - | - | 4.9e-03 / 9.9e-03 | 9.9e-03 |

(Search columns: worst on the random sets / worst found by the search, max over runs; "misread" =
bits that reifier's boolify reads wrong. The search is stochastic: two runs on the same circuit can
differ by 2x, so a search value is a lower bound on the worst case.)

What this shows at log_w 6 (the last column checks every circuit on the same messages, so its values
are comparable along a ladder; messages found for one circuit often hurt its neighbours):
- **The README's robust frontier fails under search at every depth from 3 to 8**:
  - d3 `dl3:d3c1_mp17_p1d_kt_xca`: 2.3e-2; every d3 rung is above 1.2e-2;
  - d4 `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca`: **2.6e-2 with a boolify misread** (verified against the
    reference). Without `xca` (same dense) it reaches 1.6e-2;
  - d5 `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra`: 1.16e-2 (9.1e-3 on the union set, 1.16e-2 in a 1000-generation
    search);
  - d6 `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt`: 1.07e-2;
  - d7 `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt`: 2.8e-2. The `z11` twin (same dense) and `z11` with `m3`
    reach 1.6e-2;
  - d8 `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra`: 2.2e-2, on a message found for its `m3` variant, which
    fails too (2.2e-2).
- **Misreads**: `lz5` at depth 7 and the `tr` pool (8 misread bits each on the union set), the d4 point,
  and the plain `dl3:d3` (1).
- **Where the error enters** (union set along the ladders):
  - depth 8: `m4s4` takes 3.3e-3 to 1.6e-2; `m3` in its place gives 7.7e-3; the rungs after (`sl`,
    `p1dct`, `ra`) and `m3` + `p1dct` + `ra` end near 2e-2;
  - depth 7: `m4s4` takes 3.3e-3 to 1.6e-2, and `bt` 1.0e-2 to 2.8e-2;
  - depth 6: `p1dct` / `ra` / `bt` take 4.1e-3 to 7.9e-3 .. 1.1e-2;
  - depth 4: `xca` takes 6.6e-3 (union) to 2.6e-2;
  - the S6 pools: `tr` 2.8e-2, `tp` 1.6e-2, `t8` 1.1-1.4e-2;
  - `mpc` in the d8 parity layer: 5.4e-2.
- **Below half the tolerance under search** (<= 5e-3, 2-4 searches each):
  - every layout before digest packing (split / rp / cp / lazy4c / c5 / z11: <= 1.1e-3);
  - `u1` and `mp` at depth 8 (<= 3.3e-3) and depth 5 (`b5_2_mp` 5.0e-3). The depth-7 `mp` point
    (`b7_5_mp`) reaches 6.5e-3 in later searches;
  - the depth-6 chain up to `sl` (4.6e-3);
  - the sparse ends `c2:sp_col1b_p1d` (3.5e-4) and `c2:sp_base_p1d` (3.5e-3).
- `c2:sp_lazy2_l3_p1d` reaches 8.0e-3. The `_tp` sparse ends fail: d4 `xc:d4a_m4k3_mpc_p1dxt_tp`
  2.4e-2 with a misread, d5 `xc:d5_m3k2_mp_cp_mpc_p1dxt_tp` 3.6e-2, d6 `c2:d6_lazy_cp_p1dct_tp_bt`
  1.2e-2. On the union sets they had looked fine (<= 8.3e-3): a dedicated search matters.
- **Smaller w**: the searches at log_w 2-5 find 1.1-2x the random-set worst and cross 0.01 only for
  `t8` (log_w 4-5) and the depth-3 point (log_w 5). They were shorter (150 generations) than at log_w 6.

## 8. Per-strategy verdicts (float32, 3 steps, 1 round, log_w 0-6)

No strategy crashes or computes a wrong bit (nearest) at any word size: 90 variants x 7 word sizes on
1053 reference messages each, all messages at log_w 0-1, 429 adversarial messages at log_w 6. The
verdicts are about paying and float32 margin. "Random" = the brief's criterion; "search" = section 7.

| strategy | verdict | numbers (dense change vs previous rung; worst error) |
|---|---|---|
| S1 glu xor, chi + iota unit | robust | -85..-87% and -41..-49% at every w; errors <= 1.5e-6 |
| S2 + S3 direct layout | robust | -63..-69% vs S1 (more at small w); <= 4e-4 |
| S3 split rounds / lazy chi / fold `rp` | robust (also under search) | -24..-32% / -22..-24% / -2.5..-4.9% |
| S3 fused rounds (`dl3` d3, d4) | robust on random sets, **fail under search at log_w 6** | d3 frontier 2e-4 (w0) .. 9.7e-3 (w6), search 2.3e-2; d4 search 2.6e-2 with misreads |
| S3 Walsh last round (`xs4` d5) | robust on random sets; under search up to `mp` only | frontier <= 4.4e-3 random, 1.16e-2 search; `b5_2_mp` 4.9e-3 |
| S3 sparse-first `sp` | robust (also under search) | 22-24x sparse at every w; random <= 1.5e-4, search 3.5e-4 (d9) / 3.3e-3 (d8) / 8.0e-3 (d7) |
| S4 `cp` | robust (also under search) | -8..-20%, largest at small w |
| S4 `u1` (and `sp`) | robust, pays less at small w | -1.0% (w0) .. -6.1% (w6); `sp` 0.3-1% worse; search 2.7e-3 |
| S4 `m4s4` | robust on random sets; **fails under search at depths 7-8** (log_w 6) | -1.7..-2.6%; search 1.4-1.6e-2 at d7/d8, 4.1e-3 at d6 |
| S4 `m3` | robust (also under search, alone) | -0.0..-3.2%; better than `m4s4` at log_w 0; d8 `mp` + `m3` 7.7e-3 under search, but with `p1dct` + `ra` (+ `sl`) on top 2.0-2.2e-2 |
| S4 `lz5` | robust at depth 6; at depth 7 tight from log_w 5, misreads under search at log_w 6 | -0.4..-1.6% |
| S4 `sl` | robust, stops paying at log_w <= 1 | -0.8..-1.5% from log_w 2; +0.0/+0.1% dense, +2.5% sparse at log_w 0-1; `with_slast_auto` disables it there |
| S5 `mp` | robust, no-op at log_w <= 1 | -0.6..-4.4% from log_w 2; search 2.4e-3 |
| S5 `mpc` on odd count ranges | no-op on `c5` layouts; robust in the Walsh layer | Walsh -0.9..-1.7% |
| S5 `mpc` in the d8 parity layer | **breaks** (float32) from log_w 1 | -1.0..-1.5%; 1.3e-2 over all log_w-1 messages, 2.4e-2 (w5), 5.4e-2 search (w6) |
| S5 `c5` / `z11` | robust (also under search) | -1.1..-1.3% / -2.0..-2.7% (`c5` + `bt` = `z11` + `mpc` in dense) |
| S5 `p1dct`, `p1dxt`, `p1d`, `kt_xc` | robust on random sets, flat in w; under search they carry the d5 / d6 / d4 rise | -1.8..-5.5%; search at log_w 6: `p1dxt` 9.9e-3 (d5), `p1dct` 7.8e-3 (d6), `kt_xc` 1.6e-2 (d4) |
| S5 `bt` | robust at depth 6; at depth 7 the worst point under search (2.8e-2) | -0.8..-1.4% |
| S5 `ra` | robust, no-op at small w | 0 at log_w 0-1 (and <= 3 in b8 / b7); -1.8..-6.2% in b6 / b5 from log_w 2 |
| S5 `xca` (dl3) | dense-neutral; robust on random sets, fails under search at d4 | 0% dense, fewer nonzeros from log_w 1; d4 at log_w 6: 1.6e-2 -> 2.6e-2 with a misread |
| S6 `tr` pool (F13) | **breaks**: tight from log_w 2-3, misreads under search; costs dense at log_w <= 1 | +3.3% (w0) .. -1.2% (w6) vs `ra` |
| S6 `tp` pool (F1) | robust on random sets, sparse-only; **fails under search** | +0.7..+4.3% dense vs `ra`; `_tp` sparse ends 1.2e-2 (d6), 3.6e-2 (d5), 2.4e-2 with a misread (d4) |
| S6 `t8` pool (new: F1 on T = 8 columns only) | clean per-w switch; robust on random and big sets, **tight under search** | = `ra` at log_w <= 3; `tr`'s dense from log_w 4 (-0.5..-1.2%); random 6.0e-3..7.6e-3, search 1.0-1.4e-2 |

## 9. Fixes

**S6 restricted to the columns where it pays (`TH1P` form `F18`, in `xs3._theta1_pool`).**
- The pool form costs 3 units per live bit plus a pool of 3-4 units per column. A direct raw parity
  of n bits costs floor((n - 1) / 2) units (p1d forms, n >= 6) or ceil(n / 2) (glu xor, n <= 5).
  - T = 7 column, 4 live own bits: direct 4 x 3 + 3 = 15 units, pool (F3) 4 x 3 + 3 = 15. No gain,
    but F3's irrational knots cost float32 margin.
  - T = 8 column: direct 4 x 4 + 3 = 19, pool (F1) 4 x 3 + 4 = 16: -3 units.
  - T <= 5 (log_w 0-1): the pool costs more than the direct forms.
- So the whole dense gain of `tr` over `ra` is 3 units per T = 8 column. At log_w 6 that is 56 columns,
  168 units x 3755 = 630,840 dense: exactly the README's `_tr` - `_ra` difference. All of `tr`'s extra
  float32 error on the random sets comes from F3 on the T = 7 columns.
- `F18` uses F1 (exact on {0,1} x [0, 8], wave 3's sympy proof) only where T = 8, and returns None
  elsewhere, so those columns fall back to `par_units` (the `ra` forms).
  - It equals `ra` at log_w <= 3, where there is no T = 8 column (a clean per-w switch).
  - From log_w 4 on it has `tr`'s dense, and its random-set errors are 6.0-7.6e-3 against `tr`'s
    1.0-1.8e-2.
  - Under adversarial search F1 itself is the weak part (1.0-1.4e-2 at log_w 4-6), so `t8` only helps
    where the random-set criterion is accepted.
- Variants: `ladder:b6_10e_t8_bt` (d6), `ladder:b5_5e_t8` (d5), `ladder:auto_d6_t8` (with the `sl` guard).

**`with_slast_auto` (`xs3.py`).** `sl` only when d < 5w; it is off at log_w 0-1.
- Variants: `auto_d8`, `auto_d7`, `auto_d6`.
- At log_w 1, d8: -59 dense and -82 sparse. At log_w 0: -40 sparse.
- Identical to `with_slast` from log_w 2 on.

The existing variant names are unchanged (both fixes are new names), so every README number still
reproduces.

**No counterexample found under search at log_w 6** (the best ladder point per depth whose worst over
all searches and both adversarial sets stays <= 0.01; the cost of that standard):

| depth | README robust point (dense) | worst under search | best point with worst <= 0.01 | dense / sparse | worst under search | dense cost |
|---|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt_xca` 238,512,173 | 2.3e-2 | none in the ladder | | | |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` 124,087,564 | 2.6e-2, misread | `dl3:d4x_m4k2_mpx_wmpc_p1d` (`ladder:b4_3_p1d`) | 127,595,249 / 615,009 | 8.4e-03 (search 8.4e-03, all adversarial msgs 3.7e-03) | +2.8% |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` 74,073,879 | 1.2e-02 (search 1.2e-02, all adversarial msgs 9.1e-03) | `ladder:b5_2_mp` (xs4 d5 m4k2 + cp + mp) | 81,486,372 / 234,811 | 5.0e-03 (search 4.9e-03, all adversarial msgs 5.0e-03) | +10.0% |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` 53,915,598 | 1.1e-02 (search 1.1e-02, all adversarial msgs 1.1e-02) | `ladder:b6_7_sl` | 59,092,278 / 185,867 | 4.6e-03 (search 4.6e-03, all adversarial msgs 4.1e-03) | +9.6% |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` 48,831,997 | 2.8e-2 | `ladder:b7_5_mp` | 52,415,738 / 124,984 | 6.5e-03 (search 6.5e-03, all adversarial msgs 3.3e-03) | +7.3% |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` 44,218,228 | 2.2e-02 (search 1.0e-02, all adversarial msgs 2.2e-02) | `ladder:b8_4m_m3` (m3 in place of m4s4) | 46,379,980 / 115,305 | 7.7e-03 (search 7.2e-03, all adversarial msgs 7.7e-03) | +4.9% |
| 9 | `c2:sp_col1b_p1d` 66,359,285 / 91,460 | 3.5e-4 | the same | | | 0 |
| 8 (sparse) | `c2:sp_base_p1d` 62,287,910 / 95,637 | 3.5e-3 | the same | | | 0 |

Each replacement had three or four dedicated searches (150-1000 generations, different seeds) plus both
union sets. "No counterexample found" is weaker evidence than a counterexample.

## 10. Side data: bf16 weights across word sizes (`--modes f32,w16`, log_w 0-5)

Worst error with bf16-rounded weights and float32 activations on the audit + stress sets (in brackets:
wrong bits / boolify misreads). The wrong bits appear with the flag named in the row and persist
in every later rung:

| variant (flag added) | log_w 0 | log_w 1 | log_w 2 | log_w 3 | log_w 4 | log_w 5 |
|---|---|---|---|---|---|---|
| `baseline` | 1e-06 | 6e-07 | 8e-07 | 5e-07 | 5e-07 | 5e-07 |
| `variants:glu_xor_everywhere` | 2e-06 | 6e-07 | 8e-07 | 5e-07 | 5e-07 | 5e-07 |
| `variants:glu_chi_iota` | 1e-06 | 7e-07 | 8e-07 | 6e-07 | 5e-07 | 5e-07 |
| `xs3:direct` | 2e-05 | 3e-05 | 8e-05 | 9e-05 | 4e-04 | 2e-04 |
| `xs3:split_first_middle` | 3e-05 | 4e-05 | 9e-05 | 1e-04 | 4e-04 | 3e-04 |
| `xs3:lazy4c_middle` | 3e-02 | 5e-02 | 4e-02 | 3e-02 | 4e-02 | 6e-02 |
| `ladder:b8_1_cp` | 3e-05 | 4e-05 | 1e-04 | 4e-04 | 3e-04 | 3e-04 |
| `ladder:b8_2_u1` | 2e-05 | 8e-01 (4 / 293) | 1e+00 (681 / 4424) | 1e+00 (2036 / 10739) | 1e+00 (5389 / 23452) | 1e+00 (10912 / 47311) |
| `ladder:b8_2s_sp` | 6e-01 (764 / 1013) | 1e+00 (121 / 1627) | 9e-01 (99 / 2429) | 9e-01 (127 / 4572) | 1e+00 (278 / 9892) | 9e-01 (352 / 19352) |
| `ladder:b8_3_mp` | 2e-05 | 8e-01 (4 / 293) | 9e+00 (15429 / 17817) | 9e+00 (35154 / 39920) | 1e+01 (73116 / 81707) | 9e+00 (148555 / 163939) |
| `ladder:b8_4_m4s4` | 1e-04 | 8e-01 (4 / 361) | 5e+01 (16567 / 17928) | 7e+01 (37900 / 39273) | 1e+02 (77598 / 79487) | 1e+02 (156948 / 159795) |
| `ladder:b8_6_p1dct` | 1e+00 (2971 / 3444) | 1e+00 (3024 / 8325) | 5e+01 (16621 / 18175) | 7e+01 (38149 / 39594) | 1e+02 (77580 / 79610) | 1e+02 (156663 / 159890) |
| `ladder:b7_2_c5` | 4e-04 | 4e-04 | 5e-04 | 5e-04 | 5e-04 | 6e-04 |
| `ladder:b7_2z_z11` | 5e-01 (0 / 2202) | 7e-01 (51 / 4031) | 7e-01 (35 / 5485) | 7e-01 (37 / 11557) | 7e-01 (84 / 23301) | 8e-01 (74 / 46538) |
| `ladder:b6_4_lz5` | 2e-04 | 4e-04 | 7e-04 | 1e-03 | 1e-03 | 1e-03 |
| `ladder:b5_1_cp` | 6e-03 | 8e-03 | 8e-03 | 8e-03 | 8e-03 | 8e-03 |
| `ladder:b5_3_mpc` | 3e-01 (0 / 2622) | 3e-01 (0 / 4873) | 7e+00 (11759 / 17379) | 8e+00 (21139 / 30444) | 5e+01 (48329 / 66381) | 5e+01 (98918 / 132046) |
| `ladder:b4_0` | 6e-03 | 8e-03 | 8e-03 | 8e-03 | 8e-03 | 8e-03 |
| `ladder:b4_1_mpx` | 6e-03 | 1e+00 (2099 / 7249) | 1e+00 (14218 / 17094) | 1e+00 (25415 / 30318) | 8e+00 (58092 / 67001) | 3e+00 (121135 / 136642) |
| `ladder:b3_0` | 4e-01 (0 / 2550) | 7e-02 (0 / 329) | 3e-01 (0 / 7670) | 8e-02 (0 / 1577) | 8e-02 (0 / 3140) | 8e-02 (0 / 6595) |
| `sp:col1b` | 2e-05 | 5e-05 | 4e-05 | 5e-05 | 5e-05 | 4e-05 |
| `c2:sp_col1b_p1d` | 1e+00 (2460 / 4465) | 1e+00 (4531 / 8961) | 1e+00 (6451 / 12079) | 1e+00 (12302 / 25081) | 1e+00 (24865 / 49985) | 1e+00 (48607 / 99047) |

- Exact in w16 at every word size: gated units, the direct / split layouts, `rp`, `cp`, `c5`, `lz5`,
  and the Walsh d5 / d4x bases (7.8e-3, the xs4 digest decoders). `sp:col1b` is exact too.
- Break in w16: `u1` (from log_w 1; at log_w 0 there are no u pairs to break), `sp` pairs, `mp` /
  `mpx` (from where they act), the lazy4c `z11`, the Walsh `mpc`, every irrational-knot form (`p1dct`,
  p1d on sp).

## 11. Remaining ideas

- **Adversarial robustness as the criterion.** Random stress sets underestimate the worst case by up
  to 8x at log_w 1 (exhaustive) and 2-4x at log_w 6 (search). A robust point should pass an
  adversarial search, or a bound: the error depends on local count patterns, which could be enumerated
  once per layer and composed. The search itself can be sharpened (more generations, restarts, seeding
  from the other circuits' worst messages as in `advall_w6.pt`).
- Repair the adversarial failures while keeping dense:
  - d7: `z11` (done, same dense);
  - d8: `m3` (+1.9%), or an `s4` staircase with more margin;
  - d3 / d4: find which fused-round form carries the error (per-layer float64 comparison on the
    adversarial messages).
- log_w 0-1 have their own best stacks (`m3`, no `sl`); a chooser like `with_slast_auto` for `m3` vs
  `m4s4` would need only d.
- `t8`-like restrictions for other knot-between-integer forms: use them only where they save a unit.

## 12. Files

- `repo/` (patch in `patch.diff`, against the base):
  - `experiments/xof_shrink/ladder.py` (new): the ladders, the alternatives, the fixed variants;
  - `experiments/xof_shrink/xs3.py`: the `F18` pool form and `with_slast_auto`;
  - `experiments/xof_shrink/audit/multi_check.py` (new): harness + float32 (+ w16) checks on several
    reference sets from one build;
  - `experiments/xof_shrink/audit/ref_exhaustive.py` (new): every message of log_w 0 / 1;
  - `experiments/xof_shrink/audit/adv_search.py` (new): adversarial message search;
  - `experiments/xof_shrink/audit/ref_msgs.py` (new): reference outputs for given messages.
- `refs/`: reference sets (`audit`, `mixed`, `dense` per log_w; `big*_w5/6`; `all_w0/1`;
  `adv_w2-6`, `advall_w6`).
- `runs/`: raw JSON per run:
  - `lad.all.jsonl` (log_w 0-5), `lad6.all.jsonl` (log_w 6);
  - `exh`, `big`, `adv`, `advlong`, `advrung`, `advx`, `cli` (`*.all.jsonl`).
- `results.jsonl`: all of them with baseline ratios (`collect.py`).
- Scripts:
  - `mc.sh` / `mcx.sh` / `mcb.sh` / `mca.sh` (one check on the standard / exhaustive / big /
    adversarial sets), `adv.sh` (one search), `cli.sh` (harness CLI);
  - `summ.py`, `gains.py`, `frontier.py`, `tables.py`, `structure.py`, `collect.py`, `make_notes.py`.

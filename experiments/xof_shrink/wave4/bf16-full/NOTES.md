# bf16-full: Keccak XOF circuits that are correct in bfloat16 (weights and activations)

Avenue of wave 4. Question: can the compiled XOF run in mode `b16` of `audit/bf16_check.py`
(weights and activations in bfloat16, which is what `Compiler(mlp_dtype=t.bfloat16)` runs)
with every output bit correct, and what is the smallest such circuit?

**Answer: yes, bit-exactly.**

1. **A rule for gated units that bfloat16 computes exactly** (the E-rule, section 2), and a
   builder made only of such units (`bf16x/bx.py`). At log_w 6, 3 steps, 1 round:
   **depth 9, dense 72,009,024, sparse 132,565** (`bf16x.bx:pk1q`). Against the threshold
   baseline that is 2.2x / 45.9x / 15.3x smaller in depth / dense / sparse. The worst b16
   error is 5.4e-7 on 560 stress messages, and float32 is robust too (2.7e-4). At log_w 4 it
   is depth 9, 4,481,880 dense (46.1x) and 33,167 sparse.
   - The builder works for any number of rounds and any word size.
   - Across the lead's configuration grid it is 28-71x smaller in dense than the baseline, and
     383x for 1 step and 1 round.
   - It is b16-exact wherever checked, up to 24 rounds per step (216 layers).
   - Float32 limit: in float32 its errors roughly double per layer, so beyond depth 18 it needs
     a re-threshold layer every 3 rounds (`q_rt3`, +26% dense at 4 rounds).
2. **A compiler-level change that makes the default threshold compile b16-correct:** three
   small edits in `SwiGLU.from_matrix` (section 4).
   - Margin about 1e-9 at log_w 4 and 6, at log_w 3 and 4 with 2 rounds, and at log_w 3 with
     24 rounds (428 layers).
   - Dense +0, sparse +2.8%.
   - It is on by default when `MLP_SwiGLU.from_matrices` or `Compiler` get a 16-bit dtype.
3. **Earlier strategies:** every earlier gated-unit strategy that uses counts, packings or
   knots off the E-rule fails in b16. Verdicts per strategy are in section 5.

**Verification sets.** Everything below was measured with the harness (`xofbench.py`) and
`bf16_check.py` on two sets, at log_w 4 and at log_w 6 (3 steps, 1 round):
- **audit:** `ref_gen`, 9 edge cases plus 60 random messages at densities 0.05 / 0.5 / 0.95
  (69 messages);
- **stress:** `ref_stress.py mixed 4 10`, 560 messages at densities 0.005-0.995 (dense ones
  included), plus lane, z and lane-xor-z patterns.

Every run is a JSON line in `results.jsonl`: `kind` harness, bf16_check, robust or
e2e_compiler, with `steps_env` giving the exact-steps flag.

## 1. Diagnosis: why bfloat16 breaks the compiled circuits

**What bfloat16 does in each layer** (CPU torch 2.8, checked in `arith_test.py`):
- **RMSNorm** maps every feature equal to BOS to one and the same bf16 number N. N is BOS/rms,
  about 0.4-30. The bf16 eps = 2^-7 only rescales N.
- **The matmuls** `n @ wg.T` and `n @ wv.T` accumulate in float32, then round the result to
  bf16: 8 significant bits, relative error at most 2^-9. `silu`, the product and `wo` each
  round to bf16 as well.
- **So a pre-activation is exact iff it is N times a number with at most 8 significant bits.**
  32N, 64N and 16N are exact; 96N, 20N, 12N and 24N are not. For example, 32·3·N came out as
  134.0 instead of 134.25.
- `silu(z)` rounds to z for z >= 6. `silu(-16) = -1.8e-6`.

**Per-layer measurement of the threshold baseline** (`bf16x/cleanness.py`: the largest distance
of a feature/BOS from the nearest integer, with b16 activations propagated; log_w 4, audit):

| layer | in -> hidden -> out | max \|g\|/N | b16 distance |
|---|---|---|---|
| L1 (narrow gates) | 281 -> 1378 -> 689 | 64 | 1.9e-2 |
| L2 (theta counters, fan-in 11) | 689 -> 8802 -> 4401 | 341 | 2.4e-1 |
| L3 (alternating sum) | 4401 -> 802 -> 401 | 64 | 5.9e-2 |
| L8 (round-2 counters) | 401 -> 8914 -> 4457 | 341 | 3.0e-1 |
| L9 | 4457 -> 914 -> 457 | 64 | 5.0e-1 (bits flip) |

Unmodified, the baseline gets 3 wrong bits on the audit and 77 (332 by boolify) on the stress
set.

Gated-unit circuits, the same measurement at log_w 4 with exact BOS:
- **`glu_xor_everywhere`:**
  - L1 (steps) is off by 1.9e-2;
  - L2, the 11-bit glu_xor with max |g|/N = 321, is off by 3.0e-1;
  - round 2's theta (L6) then reaches 0.46 and flips bits.
- **The robust d8 frontier point** (`c2:d8_..._ra`): |b16 - f32| per layer is 0.22, 2.6, 12,
  1.1e3, 6.5e5, 1e9, 1.7e9, 1.6e19. Its packed and irrational features amplify every rounding
  error, and w16 blows up the same way.

**Error sources, largest first:**
1. **Wide threshold gates.** A step is relu(a) - relu(b) with a - b = qN.
   - For a counter [s >= i] over 11 inputs, s exceeds the threshold by up to 10. Then a and b
     are about 32N·10 while a - b = 8N: a cancellation factor of about 2c(s - theta) = 80.
   - Rounding a and b to bf16 (2^-9 each) leaves errors of 16-30%.
   - The alternating-sum layer adds 11 such counters and bits flip.
2. **The step offsets.** Even a copy (s = 1) has a = 20N and b = 12N at c = 4. Neither is
   representable, so every step layer creates 1-2% error afresh (L1 above).
3. **BOS** is relu(32N) - relu(24N). 24N is inexact, so BOS itself is off by up to about 1.4%,
   and every readout is relative to it.
4. **Gated units do not re-threshold.**
   - An input that is off by d comes out off by d times the unit's slope, so errors grow about
     2x per unit layer.
   - Wide glu_xor forms also cancel huge terms: glu_xor of 11 bits at s = 11 is -99 + 100.
   - `glu_xor_everywhere` gets 109 wrong bits of 5376 at log_w 4.
5. **Not a source:**
   - RMSNorm's scaling, which is common to all features of a message;
   - silu itself.

## 2. The E-rule: gated units that bfloat16 computes exactly

**Setting.** Every input feature is exactly 0 or exactly BOS, so RMSNorm maps it to 0 or N.
A gated unit max(0, G)·V has G and V integer or half-integer affine in the bits, in BOS
units. SwiGLU computes it as p = silu(32NG)·(NV/2) and reads it out through wo = 1/16.

**What makes each step exact:**
- **G > 0 a power of 2:** 32NG is exact, and silu(32NG) = 32NG exactly in bf16.
- **V zero or ± a power of 2:** NV/2 is exact, and p = GV·round(16N²). That is a power-of-2
  multiple of one rounded number, because power-of-2 scaling commutes with rounding.
- **V = 0:** p = 0 whatever G is.
- **G = 0:** silu(0) = 0.
- **G <= -1/2:** the unit is off. silu(-16N) is about -2e-6·N, which is 1e-7 of an output;
  G <= -1 gives 1e-13.
- **The sum:** wo adds these terms exactly in the float32 accumulator.

**The rule.** At every reachable input point, each unit must satisfy one of:
- G <= -1/2;
- G = 0;
- V = 0;
- G > 0 is a power of 2 and |V| is a power of 2.

**Why that suffices.** A layer whose units all satisfy the rule outputs exactly round(N²) for
a one-bit, which is BOS's own output, and about 0 for a zero bit. Its outputs are clean again,
so by induction the whole network is exact.
- This needs BOS itself to be exact: BOS := relu(64N) - relu(32N) (compiler change 1,
  section 4).
- Integer features with values in {0, ±1, ±2, ±4, ...} x BOS are clean too.
- Features with the values 3, 5, 6 or 7 are not clean.

**E-exact forms** (`bf16x/eunits.py`, checked on every input point with rationals):

| function | units | form (s = number of ones) |
|---|---|---|
| copy | 1 | max(0, x)·1 |
| xor2 | 1 | max(0, s)(2 - s) (= glu_xor, n = 2) |
| xor3 | 2 | max(0, 2 - s)s + max(0, s - 1)(s - 2)/2 |
| xor4 | 2 | max(0, 2 - s)s + max(0, s - 2)(4 - s) |
| xor5 | 3 | max(0, 2 - s)s + max(0, s - 2)(5 - s)/2 + max(0, s - 3)(3s/2 - 7) |
| xor_lin, k <= 5 | copies + 0/1/1/2/2 | s + max(0, 2s - 2)(s/2 - 2) + max(0, 2s - 6)(-2); the copies are shared with a layer that copies the bits anyway |
| chi (+ iota) | 1 | the repo's CHI unit: G in {-1, 0, 1, 2, 3}, and V = 0 where G = 3 |
| parity of q in {-1, 0, 1, 2} | 2 | max(0, -q) + max(0, q)(2 - q) |
| pack 2 digest bits | 3 | p = t + 2u + tu in {0, 1, 2, 4}; carried by max(0, p), 1 unit |
| unpack | 2 + 1 | t = max(0, 2 - p)p + max(0, p - 2)(p - 2)/4; u = max(0, 3p - 4)(7/8 - 3p/16) |

**Forms that are not E-exact:**
- glu_xor for n >= 3: G = s = 3 with V = -1;
- glu_xor(clean=True): odd G values;
- min-parity: knots between integers give G = 3/2;
- wave 3's irrational knots;
- count features and packings whose values include 3, 5, ...

**Searches** (`bf16x/search_*.py`, exhaustive over slopes {±1/2, ±1, ±2}, quarter-integer
knots and fitted V):
- **Symmetric parity:**
  - xor5 has no 2-unit form;
  - xor6 and xor7 have no 3-unit form;
  - the 4-unit searches timed out.
- **Why one layer cannot do theta.** A unit can be on at no more than 4 consecutive lattice
  points: its G values form an arithmetic progression, which holds at most 3 powers of 2 plus
  one point where V = 0. So a one-layer parity from symmetric units is impossible for n >= 9,
  and theta over 11 bits (8-9 live bits in round 1) needs 2 layers.
- **Sharing across a column.** No single per-bit unit h(a, C1, C2) or h(a, e), plus units
  shared by the 5 bits of a column, computes theta. So theta costs 2 units per bit.
- **Linear corrections.** A 2-unit correction exists for parity(s) - s, which gives xor_lin.
- **A packed pair (2t - u) plus e:** `search_pair.py` was stopped without a result; by hand it
  needs at least 5-6 units per pair, so it does not pay.

## 3. The E-exact XOF builder (`bf16x/bx.py`)

**Structure.**
- One traced call per layer, so the compiler adds no copies.
- Constants and negations are folded as in `xs.py`.
- It handles any number of rounds per step and any log_w.
- The last round computes only the digest and what the digest needs.

**Three layers per round:**
- **C:** the column parities, plus a one-unit copy of each state bit that theta reads.
- **T:** theta = a ^ C[x-1][z] ^ C[x+1][z-1], with 2 units per bit (1 unit when it has 2
  inputs).
- **X:** chi with iota folded in, 1 unit per bit.

**Variants.** Each builds on the previous one.

| variant | adds |
|---|---|
| `variant` | parities as xor_e (3 units for 5 bits) |
| `pk1` | digests that travel at least one more layer, packed two bits per feature |
| `pk1l` | parities as xor_lin: 2 units for 5 bits, the copies shared |
| `pk1q` | one feature q = a + C[x-1] - C[x+1]' in {-1, 0, 1, 2} per theta position, whose parity is theta, instead of copies plus C features |

For `pk1q`:
- For constant a, q = C[x-1] + C[x+1]' and theta takes 1 unit.
- The units of q are a's copy and the two parities' units, all shared across the layer.
- Digest pairs are packed already in the C layer, because their copies are the q's.

| config | variant | depth | dense | sparse | vs baseline (d / dense / sparse) | b16 audit / stress | f32 audit / stress |
|---|---|---|---|---|---|---|---|
| w6 s3 r1 | baseline (threshold) | 20 | 3,305,445,348 | 2,034,788 | 1 / 1 / 1 | fails (section 4) | 2.4e-7 |
| w6 s3 r1 | `variant` | 9 | 80,338,175 | 139,624 | 2.2 / 41.1 / 14.6 | 1.8e-15 / 3.9e-14 | 1.1e-4 / 2.3e-4 |
| w6 s3 r1 | `pk1` | 9 | 77,768,559 | 139,620 | 2.2 / 42.5 / 14.6 | 3.1e-11 / 1.6e-8 | 1.2e-4 / 2.3e-4 |
| w6 s3 r1 | `pk1l` | 9 | 75,618,519 | 135,100 | 2.2 / 43.7 / 15.1 | 3.1e-11 / 1.6e-8 | 1.4e-4 / 3.2e-4 |
| w6 s3 r1 | **`pk1q`** | 9 | **72,009,024** | 132,565 | 2.2 / **45.9** / 15.3 | 3.4e-8 / 5.4e-7 | 1.8e-4 / 2.7e-4 |
| w4 s3 r1 | baseline | 20 | 206,803,140 | 508,772 | 1 / 1 / 1 | 3 / 77 wrong | 4.8e-7 |
| w4 s3 r1 | `variant` | 9 | 5,004,599 | 34,904 | 2.2 / 41.3 / 14.6 | 6.8e-16 / 4.0e-14 | 5.3e-5 / 1.5e-4 |
| w4 s3 r1 | `pk1` | 9 | 4,843,347 | 34,902 | 2.2 / 42.7 / 14.6 | 1.6e-10 / 1.2e-9 | 5.4e-5 / 1.5e-4 |
| w4 s3 r1 | `pk1l` | 9 | 4,703,787 | 33,742 | 2.2 / 44.0 / 15.1 | 1.6e-10 / 1.2e-9 | 1.3e-4 / 1.6e-4 |
| w4 s3 r1 | **`pk1q`** | 9 | **4,481,880** | 33,167 | 2.2 / **46.1** / 15.3 | 3.4e-8 / 1.1e-7 | 1.6e-4 / 2.0e-4 |

**The table's checks:**
- A second, dense stress set (`ref_stress.py dense 4 10`: 520 messages at densities
  0.85-0.99) gives, for `pk1q`, b16 margins of 1.8e-7 (log_w 4) and 3.7e-7 (log_w 6), with
  float32 at 2.3e-4 and 2.9e-4 and 0 wrong bits.
- `validate_bench.py bf16x.bx:pk1q`: the harness weights are bit-equal to the repo's
  `Compiler().get_mlp_from_tree` at log_w 0-2, and depth, dense and sparse match.
- Every b16 and w16 check had 0 wrong bits, both nearest and boolify.
- The end-to-end repo pipeline (`bf16x/e2e_compiler.py`: `Compiler(mlp_dtype=t.bfloat16)`,
  then the `MLP_SwiGLU` module's own forward) gives the same margins: `pk1q` at log_w 4 gets
  3.4e-8 on the audit and 1.1e-7 on the stress set; `pk1` and the baseline are checked at
  log_w 2 and 4.

**Why pk1q's b16 margins are about 1e-7 rather than 1e-14.** Features of value 2 or 4 raise
the RMS, so N drops to about 0.5 in some layers. The off-units' silu(-32N) tails then reach
about 1e-7 of an output.

**Per layer at log_w 6** (`pk1`; format [in, hidden, out], dense in M):

| layer | [in, hidden, out] | dense (M) |
|---|---|---|
| C1 | [1145, 1786, 1465] | 6.71 |
| T1 | [1465, 2610, 1465] | 11.47 |
| X1 | [1465, 1602, 1601] | 7.26 |
| C2 | [1601, 2562, 1921] | 13.13 |
| T2 | [1921, 3538, 1713] | 19.66 |
| X2 | [1713, 1714, 1713] | 8.81 |
| C3 | [1713, 1554, 913] | 6.74 |
| T3 | [913, 978, 657] | 2.43 |
| X3 | [657, 786, 673] | 1.56 |

`pk1q` removes the 320 C features from C2's output and from T2's input, and the same in
round 3.

**Configuration grid** (`bf16x.bx:pk1q`, sizes from the harness, baselines from
`xof4/baselines.jsonl`):

| log_w | steps | rounds | depth (baseline) | dense | x baseline | sparse | x baseline | f32 harness |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 1 | 9 (17) | 16,902 | 48.9 | 1,889 | 16.6 | 4.6e-6 |
| 1 | 3 | 1 | 9 (20) | 68,396 | 48.9 | 3,973 | 16.2 | 2.7e-5 |
| 2 | 3 | 1 | 9 (20) | 276,132 | 47.0 | 8,329 | 15.3 | 1.9e-5 |
| 3 | 3 | 1 | 9 (20) | 1,113,750 | 46.5 | 16,588 | 15.3 | 2.9e-5 |
| 4 | 3 | 1 | 9 (20) | 4,481,880 | 46.1 | 33,167 | 15.3 | 5.2e-5 |
| 5 | 3 | 1 | 9 (20) | 17,974,092 | 46.0 | 66,287 | 15.3 | 2.6e-5 |
| 6 | 3 | 1 | 9 (20) | 72,009,024 | 45.9 | 132,565 | 15.3 | 5.6e-5 |
| 4 | 1 | 1 | 3 (8) | 180,408 | 382.8 | 3,207 | 53.2 | 1.2e-6 |
| 6 | 1 | 1 | 3 (8) | 2,872,896 | 384.3 | 12,857 | 53.1 | 1.6e-6 |
| 4 | 2 | 1 | 6 (14) | 1,987,478 | 68.1 | 16,534 | 20.5 | 1.3e-5 |
| 6 | 2 | 1 | 6 (14) | 32,321,100 | 66.9 | 66,154 | 20.4 | 2.7e-5 |
| 4 | 4 | 1 | 12 (26) | 7,273,838 | 39.0 | 50,136 | 13.6 | 8.5e-5 |
| 6 | 4 | 1 | 12 (26) | 116,445,412 | 38.9 | 200,320 | 13.6 | 8.9e-5 |
| 4 | 6 | 1 | 18 (38) | 13,806,870 | 32.9 | 85,082 | 12.2 | 5.7e-4 |
| 6 | 6 | 1 | 18 (38) | 220,466,748 | 33.0 | 339,862 | 12.2 | 3.5e-3 |
| 4 | 1 | 2 | 6 (14) | 1,857,544 | 70.7 | 15,725 | 21.3 | 1.3e-5 |
| 4 | 3 | 2 | 18 (38) | 11,593,964 | 35.2 | 80,815 | 12.5 | 9.8e-4 |
| 4 | 1 | 3 | 9 (20) | 3,984,448 | 48.5 | 31,279 | 15.9 | 2.4e-5 |
| 4 | 3 | 3 | 27 (56) | 18,758,648 | 32.5 | 128,485 | 11.8 | **2.4e-2 fails** |
| 4 | 1 | 4 | 12 (26) | 6,111,352 | 41.8 | 46,844 | 14.2 | 5.9e-5 |
| 4 | 3 | 4 | 36 (74) | 25,923,332 | 31.3 | 176,188 | 11.4 | **1.1 fails** |
| 6 | 1 | 2 | 6 (14) | 30,269,632 | 69.3 | 62,908 | 21.3 | 1.8e-5 |
| 6 | 1 | 4 | 12 (26) | 97,972,720 | 41.7 | 187,202 | 14.2 | 1.6e-4 |
| 3 | 3 | 24 | 216 (428) | 42,567,494 | 28.4 | 566,739 | 10.6 | **inf fails** |

**`bf16_check` of pk1q on other configurations** (the ref_gen audit with 60 random
messages, 0 wrong bits in b16; `kind: robust` in results.jsonl):

| configuration | b16 margin | f32 margin |
|---|---|---|
| log_w 2 | 6.9e-7 | 1.4e-4 |
| log_w 5 | 1.8e-8 | 1.3e-4 |
| log_w 4, 2 rounds | 2.7e-8 | 2.0e-3 |
| log_w 6, 1 step, 4 rounds | 9.1e-17 | 5.9e-4 |
| log_w 3, 24 rounds (216 layers) | 5.7e-8 | inf, 4489 wrong |

**Float32 and E-exact circuits.** In float32, N has 24 significant bits, so partial sums such
as 3N round inside the matmul.
- The error is 5e-7 after layer 1 and grows about 1.5-2x per layer through units that do not
  re-threshold.
- It reaches 2e-4 at depth 9, 2e-3 at depth 18 and fails at depth 27 and beyond.
- So these circuits are float32-robust up to about depth 18, and exact in bf16 at any depth.

**Fix for float32 at depth: `bf16x.bx:q_rt3`.**
- **What it adds:** a layer of threshold-gate copies after every third round, built with the
  exact steps of section 4. The steps are flat, so they re-threshold float32, and in bf16 they
  stay exact.
- **Digests are left unpacked**, so that every feature is a bit.
- **Result at log_w 4, 3 steps, 4 rounds:** depth 40, dense 32,690,132, sparse 192,948:
  - 24.8x less dense than the baseline (811,644,270);
  - float32 at most 6e-3 inside and 3.8e-5 at the output (the audit's margin);
  - b16 margin 2.8e-9;
  - plain pk1q is 25.9M here and b16-exact (8.3e-8), but fails float32 (978 wrong bits).
  - The full check is in `logs/r4.log`.
- **The same at 24 rounds** (log_w 3, 3 steps): `q_rt4` has depth 234, dense 52,926,564,
  float32 4.3e-5 and b16 2.5e-9 (`logs/r24_rt4.log`).
- **The interval:**
  - every 4 rounds (`q_rt4`, unpacked digests) also works: depth 39, 31,110,605 dense,
    float32 4.9e-5, b16 2.8e-9;
  - every 6 rounds (`q_rt6`) fails float32 (76 wrong bits): 18 layers between re-thresholds
    is too many;
  - with packed digests, every 4 rounds (`pk1q_rt4`) also fails, because the packed carries
    are not re-thresholded.
- **Result at log_w 3, 3 steps, 24 rounds:** depth 240, dense 54,834,630 (22.1x the
  baseline's 1,210,834,652), sparse 626,141 (9.6x). Float32 3.8e-5 and b16 2.5e-9 on the
  audit (`logs/r24_rt3.log`).

**Why depth 9, and what a smaller circuit would need.**
- Theta needs 2 E-exact layers: no one-layer E-exact parity of 9 or more bits exists, and the
  round-1 theta has 8-9 live bits per position. Chi needs 1. That is 3 layers per round.
- More depth only adds layers: a C | D | a^D | chi round costs 9900 w² against 9325 w².
- The size per round is set by:
  - C: xor_lin, 2 units per column plus 1 copy per bit;
  - T: 2 units per bit;
  - X: 1 unit per bit.
- The searches above find none of these can drop within the E-rule.

## 4. Compiler-level change: the default threshold compile, b16-correct

**Where it lives.** In `SwiGLU.from_matrix` (`src/reifier/tensors/swiglu.py`, keyword
`exact=True`). It is on by default for 16-bit dtypes in `MLP_SwiGLU.from_matrices`, so
`Compiler(mlp_dtype=t.bfloat16)` gets it, and the env var `REIFIER_EXACT_STEPS=1` sets the
default for the harness.

**The three edits:**
1. **Exact BOS (always on).** BOS becomes relu(2cq) - relu(cq) instead of
   relu(cq) - relu(cq - q), with its `wo` scaled by 1/c. The hidden count is unchanged.
2. **Shifted steps.** A step rises over [1/2, 1/2 + 1/c] instead of [1/2 - 1/2c, 1/2 + 1/2c].
   Where the sum is 1, its two ReLUs then sit at 16N and 8N (c = 4, q = 8), both exact.
3. **Complementary rows.** A row whose sum can exceed 1 by more than it can fall below 0
   (judged from its weights, assuming 0/1 inputs) is built as BOS - step(1 - sum). The large
   sums of wide counters then land where both ReLUs are off, and the inexact side is at most
   half the fan-in.

**Why it works.**
- Rows whose sums lie in {<= 0, 1} are E-exact.
- The remaining inexact rows are the middle counters of theta, with error up to about 9%.
  They feed the alternating-sum layer, whose flat step absorbs the error.
- The next flat layers snap the value back to exact, once the input error is below bf16's
  rounding of the pre-activation.
- Every later layer is then exact again, the outputs included. Per layer: L2 1.1e-1,
  L3 1.4e-2, L4 7.8e-3, L5 6e-3, L6 1.5e-11, ..., L20 8e-10.

**Ablation** (log_w 4, stress set, 560 messages, b16):

| compile | wrong nearest | wrong boolify | margin |
|---|---|---|---|
| unmodified base src | 77 | 332 | 1.0 |
| exact BOS only | 105 | 999 | 1.0 |
| exact BOS + complementary rows | 0 | 673 | 2.3e-2 |
| exact BOS + shifted steps | 2848 | 2848 | 1.0 |
| **all three** | **0** | **0** | **1.0e-9** |

**Results with all three:**

| configuration | depth | b16 margin |
|---|---|---|
| log_w 4, 3 steps | 20 | audit 8.1e-10, stress 1.0e-9 |
| log_w 6, 3 steps | 20 | audit 7.2e-10, stress 1.3e-9 |
| log_w 3, 2 steps, 2 rounds | 26 | 8.1e-10 (39 msgs) |
| log_w 4, 3 steps, 2 rounds | 38 | audit 8.1e-10 |
| log_w 3, 3 steps, 24 rounds | 428 | audit 1.5e-9 |

Float32 margins stay at about 2e-5. The shifted step's 8N ReLU is 3.4e-4 off in float32 at
N about 1; q = 16 would move it to 16N.

**Repo tests.** With the change, the repo's tests pass: 55 passed (`pytest tests`, without
`hash_long`, `legacy_tests` and `data_tests`).

**Size cost:**
- dense +0 (log_w 6: 3,305,445,348);
- sparse +2.8%: 2,034,788 -> 2,092,406 at log_w 6, 508,772 -> 523,190 at log_w 4. The extra
  nonzeros are the BOS entries of the complementary rows.

## 4b. The candidate constructions of the brief, and what each costs

| construction | effect in b16 | cost |
|---|---|---|
| split wide gates | theta as C (parities of <= 5 bits) plus T (xor of 3): the core of the E-exact layouts | none: 46x smaller than the baseline |
| smaller c·q per layer | no help: bf16 precision is relative, and a step's error is about c(s - theta)·2^-8 whatever the scale; c = 1 is exact up to s' = 2 but loses flatness | - |
| scale the input features | no help: RMSNorm removes the scale, and the precision is relative | - |
| exact BOS | needed by every construction; without it even `bx` has 1850 wrong bits | 0 |
| exact (shifted) steps plus complementary rows | the threshold baseline becomes exact after each flat layer (section 4) | dense +0, sparse +2.8% |
| re-threshold layers (step copies every 3 rounds) | resets float32 errors of E-exact circuits; exact in b16 | log_w 4, 4 rounds: +4 layers, 25.9M -> 32.7M dense |
| units flat at lattice points (glu_xor clean) | flat but not E-exact (odd G): 357 wrong bits | - |
| small products, power-of-2 values (E-rule) | exact | xor5 3 units (2 with copies), theta 2 per bit |
| bf16-representable weights | every E-exact weight is a small dyadic rational, so w16 equals f32 | 0 |

## 5. Per-strategy verdicts in b16 (S1-S6)

Measured with `bf16_check` at log_w 4 on the audit set, with exact BOS. Without exact BOS,
every gated-unit circuit fails anyway; `bx` itself gets 1850 wrong bits.

| circuit | strategies | depth / dense | f32 | w16 | b16 wrong (nearest / boolify) |
|---|---|---|---|---|---|
| `variants:glu_xor_everywhere` | S1 glu_xor (11 bits) | 14 / 29.6M | 4.8e-7 | ok | 109 / 362 |
| `variants:glu_chi_iota` | S1 glu_xor + chi unit | 8 / 15.5M | 3.6e-7 | ok | 3047 / 3017 |
| `variants:glu_clean_chi_iota` | S1 glu_xor(clean) | 8 / 24.4M | 3.6e-7 | ok | 357 / 491 |
| `xs:variant` | S2, S3 shared parities, S4 digest | 6 / 6.3M | 2.3e-5 | ok | 2379 / 3987 |
| `xs3:split_first_middle` | S3 split rounds, counts | 8 / 3.9M | 8.9e-5 | ok | 2578 / 3985 |
| `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` | S3-S5 | 5 / 4.6M | 2.0e-3 | 4327 wrong | 4624 / 5202 |
| `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | S3-S6 | 6 / 3.3M | 2.8e-3 | 4437 wrong | 4926 / 5418 |
| `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` | S3-S5 (fold) | 8 / 2.8M | 1.0e-3 | 6736 wrong | 6847 / 5574 |
| `c2:sp_col1b_p1d` | S3 sparse-first, S5 | 9 / 4.1M | 3.9e-5 | 1588 wrong | 1984 / 3694 |
| `dl3:d3c1_mp17_p1d_kt_xca` | S3 fused rounds, S5 | 3 / 14.6M | 1.7e-3 | 4591 wrong | 5685 / 5292 |
| **`bf16x.bx:pk1q`** | S1 E-exact, S2, S3 shared parities, S4 E-exact digest pairs | 9 / 4.48M | 1.6e-4 | ok | **0 / 0** |

- **S1 gated units: mixed.**
  - The chi unit (with iota) and xor2 are E-exact, and the b16 frontier uses them.
  - glu_xor for n >= 3 and glu_xor(clean) are not E-exact (G = 3, 5, ... with V != 0), and
    they fail.
  - E-exact xor3/4/5 forms cost 2, 2 and 3 units, or 1, 2 and 2 beyond copies that already
    exist.
  - No E-exact form exists for n >= 9, so theta takes 2 layers.
- **S2 compiler passes: robust.** Folding constants into biases, sharing units and dropping
  the re-threshold layer after pure units are all exact: the biases are small integers or
  halves. The BOS pair has to become exact (section 4).
- **S3 layouts: mixed.**
  - Shared column parities are robust: they are the C layer.
  - Count features break unless their values lie in {0, ±1, ±2, ±4}. The q feature of `pk1q`
    is such a count feature.
  - These all break: the lazy chi; the fold (`rp`, counts up to 33); `z11`; split rounds (X
    computes D, the parity of 10 bits, in one layer, which has no E-exact form); fused rounds;
    the Walsh last round; the sparse-first layouts.
- **S4 packing: breaks as built.** Examples:
  - cp packs c = the sum of 5 bits and p = a0 + 2a1 in {0..3};
  - u1 packs u = a1 + 2a2 - 3D;
  - m4s4, lz5 and k2 hold values up to 15 or 24;
  - sl emits the chi gate 2a - b + c in {-1..3}.

  Exact packings exist: t + 2u + tu in {0, 1, 2, 4}, or 2t - u in {-1, 0, 1, 2}. As digest
  carries they save 3.2% (`pk1`). As theta inputs they need more decoding units than they
  save.
- **S5 parity forms: break.**
  - Min-parity knots between integers give G = 3/2 or 5/2.
  - Wave 3's irrational knots break even w16.
  - The reduction units work on counts above 2.
- **S6 round-1 theta pool: breaks.** It uses min-parity and count forms on raw sums.

## 6. b16 frontier (verified with the criterion)

| config | depth | variant | dense | sparse | vs baseline: depth / dense / sparse | b16 margin (audit / stress) | f32 margin (audit / stress) |
|---|---|---|---|---|---|---|---|
| log_w 6, 3 steps, 1 round | 9 | `bf16x.bx:pk1q` | 72,009,024 | 132,565 | 2.2 / 45.9 / 15.3 | 3.4e-8 / 5.4e-7 | 1.8e-4 / 2.7e-4 |
| log_w 6, 3 steps, 1 round | 20 | baseline, exact steps | 3,305,445,348 | 2,092,406 | 1 / 1 / 0.97 | 7.2e-10 / 1.3e-9 | - |
| log_w 4, 3 steps, 1 round | 9 | `bf16x.bx:pk1q` | 4,481,880 | 33,167 | 2.2 / 46.1 / 15.3 | 3.4e-8 / 1.1e-7 | 1.6e-4 / 2.0e-4 |
| log_w 4, 3 steps, 1 round | 20 | baseline, exact steps | 206,803,140 | 523,190 | 1 / 1 / 0.97 | 8.1e-10 / 1.0e-9 | 2.0e-5 / 2.3e-5 |
| log_w 4, 3 steps, 4 rounds | 39 | `bf16x.bx:q_rt4` (exact steps) | 31,110,605 | 188,333 | 1.90 / 26.1 / 10.7 | 2.8e-9 (audit only) | 4.9e-5 |
| log_w 4, 3 steps, 4 rounds | 40 | `bf16x.bx:q_rt3` (exact steps) | 32,690,132 | 192,948 | 1.85 / 24.8 / 10.4 | 2.8e-9 (audit only) | 3.8e-5 |
| log_w 3, 3 steps, 24 rounds | 234 | `bf16x.bx:q_rt4` (exact steps) | 52,926,564 | 613,787 | 1.83 / 22.9 / 9.8 | 2.5e-9 (audit only) | 4.3e-5 |
| log_w 3, 3 steps, 24 rounds | 240 | `bf16x.bx:q_rt3` (exact steps) | 54,834,630 | 626,141 | 1.78 / 22.1 / 9.6 | 2.5e-9 (audit only) | 3.8e-5 |

Plain `pk1q` is smaller and b16-exact at these depths (4 rounds: 25,923,332, b16 8.3e-8;
24 rounds: 42,567,494, b16 5.7e-8), but it fails float32. It is the b16 frontier there if
float32 is not required.

No b16-correct point below depth 9 was found; see "Why depth 9". Deeper E-exact layouts are
larger.

## 7. Remaining ideas

- **Pack digest 2** in the last C layer. Done in pk1q; a 3-bit packing would save more.
- **An 8-layer circuit** needs one round in 2 layers. The best candidate is the last round:
  compute chi(parity(qa), parity(qb), parity(qc)) directly from the C layer's q features,
  only for the digest bits (224 at log_w 6).
  - `search_chiq.py` finds 4741 E-exact unit directions on {-1,0,1,2}^3. Their span has full
    rank, so some exact form exists.
  - Greedy OMP with up to 16 units never reached it (residual still 0.61), so the form is
    large.
  - A basis-pursuit or MILP search (no scipy in the venv) could bound the minimum. At
    16 units the depth-8 circuit would cost about 4M more than T3 + X3.
  - Depth 6 needs the same fusion in every round.
- **A one-layer round-1 theta** would need non-symmetric E-exact parity forms; only
  symmetric ones were searched.
- **`q = 16` with exact steps** would put the smaller ReLU of every step at 16N: float32 error
  1e-7 instead of 3.4e-4, sizes unchanged.
- **Scale the packed values to {0, 1/2, 1, 2}.** That keeps N near 1 and the silu tails at
  1e-13.
- **Float32 robustness at depth.** Test `q_rt3`-style re-thresholding on 24 rounds, and measure
  its cost on the grid.

## 8. Files and reproduce

**Code:**
- `repo/src/reifier/tensors/swiglu.py`: exact BOS, the `exact` steps, and the
  `MLP_SwiGLU.from_matrices` default.
- `repo/experiments/xof_shrink/bf16x/`:
  - `eunits.py`: the E-rule, the forms, and a lattice check that `python eunits.py` prints;
  - `bx.py`: the builder;
  - `cleanness.py`: per-layer bf16 and f32 diagnostics;
  - `e2e_compiler.py`: the Compiler path;
  - `search_sym.py`, `search_lin.py`, `search_theta*.py`, `search_pair.py`: the searches.

**Scripts:** `check.sh` (harness + bf16_check -> results.jsonl), `base_w6.sh`, `strat.sh`,
`robust.sh`, `grid.sh`, `grid_q.sh`.

**Data:** `refs/`, `logs/`, `arith_test.py`.

**patch.diff** covers only this avenue's changes: `swiglu.py` and `bf16x/`. The base's own
uncommitted edits to the audit scripts and the harness are not in it.

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
A=$S/xof4/av/bf16-full; R=$A/repo; E=$R/experiments/xof_shrink
export PYTHONPATH=$R/src:$E:$E/depth_low/lib/python OMP_NUM_THREADS=2
$S/venv/bin/python $E/xofbench.py --log-w 6 --depth 3 --variant bf16x.bx:pk1q
$S/venv/bin/python $E/audit/bf16_check.py $A/refs/m_w6.pt bf16x.bx:pk1q
REIFIER_EXACT_STEPS=1 $S/venv/bin/python $E/audit/bf16_check.py $A/refs/m_w4.pt baseline
PYTHONPATH=$PYTHONPATH:$E/bf16x $S/venv/bin/python $E/bf16x/eunits.py
```

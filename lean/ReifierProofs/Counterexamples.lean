import ReifierProofs.Main
import ReifierProofs.Examples

/-!
# Counterexamples: each PATCH of `Spec.lean` and `Interfaces.lean` is needed

For each well-formedness condition marked `PATCH (CE-n)`, a program on tokens that satisfies every
other hypothesis of the main theorem, and on which the compiled transformer does not compute it.
`RealConcl`, `RealDeepConcl` and `BF16Concl` are the conclusions of `MainReal`, `MainRealDeep` and
`MainBF16` at one program, coding, token sequence and position (`mainReal_iff`, `mainRealDeep_iff`,
`mainBF16_iff` check this, by `Iff.rfl`), and `F_*` are the fields of `IsProgram`, `RealOK`,
`RealCore` and `BF16OK` one by one (`isProgram_iff`, `realOK_iff`, `realCore_iff`, `bf16OK_iff`).

Evidence, condition by condition:

* CE-1, step margin. **Lean, end to end** (`ce1`): `t1` is `[-1/2 >= 0]` (no inputs), which is 0;
  its sum `-1/2` is strictly between `-3/4` and 0, which the margin excludes. The reglu form of the
  gate is `1/2` there (`ce1_local`), and in real arithmetic the transformer's output feature is
  `-1 + (gelu 16 - gelu (-16)) / 16 ≥ -1/16`, not `-1` (`t1_final`).
* CE-2, the (weak) gate margin of `IsProgram`: a gate 0 or at least 1/2 from 0. **Lean, end to
  end, in real arithmetic and in bf16 for every rounding** (`ce2`, `ce2_bf16`): `t2` is the GLU
  `relu(1/32) * 32`, which is 1; its gate `1/32` is neither. Its compiled gate is `32 * 1/32 = 1`,
  where `gelu` is not `relu`. In real arithmetic `gelu 1 < 0.85` (`gelu_one_lt`, `ce2_local`) and
  the output feature is `-1 + 2 gelu 1 < 0.7`, not 1 (`t2_final`); `t2` then fails `RealOK`'s strict
  margin too, and satisfies its other fields. In bf16 `fl(gelu 1) ≤ 221/256` (`fl_gelu_one`) and the
  output feature is at most `47/64`, not 1 (`t2_bfinal`), while the weights, gates, values and
  products of `t2` are 1, 1, 1, 2, bf16 numbers (`t2_bf16OK`): every other hypothesis of `MainBF16`.
* CE-2b, the strict gate margin of `RealOK` and `RealCore` (`Trace.GateMargin`): beyond the weak
  one, in real arithmetic. **Lean, the hypotheses and the unit; numeric simulation, the failure**
  (`ce2b`, `ce2b_local`; `sim.py`, `ce2b`): `zg C a`, `y = 2 relu(2x-1) - relu(4x-3) + C relu(a(x-1))`,
  is `x` on bits, flat (where a gate is positive the slopes cancel), of weights at most `2^16`, no
  GLU; its third gate is 0 at `x = 1`, which the weak margin allows. `t2b` (one `zg (2^15) 1`) and
  `t2b2` (two `zg (2^15) (2^12)` in a chain) satisfy every hypothesis of `MainReal`, and of
  `MainRealDeep` with `σ = 0`, but the strict margin. At `p = 1` on the tokens 0, 1 the head leaks
  `ε ≈ 1.87e-13` below the bit 1 into `x`; at the zero gate `relu` passes nothing, but `gelu` passes
  between `0.45` and `0.5` of it (`gelu'(0) = 1/2`, `gelu_leak`, `ce2b_local`), times `C a`: `t2b`'s
  output feature is `1 - 3.07e-9`, off by more than `1e-9` (and `1e-10`), and `t2b2`'s is `-345.1`,
  the wrong bit; a chain of `zg (2^15) 1` gives the wrong bit at depth 4. `t2b_units` checks the
  simulated layer against the Lean model.
* CE-3, distinct outputs. **Lean, end to end** (`ce3`): `t3` returns `[0 >= 0]` twice; the token
  grid moves the first output onto the second's feature, and feature 1 keeps `code[0]` (`t3_grid`;
  on `xor3` returning `odd` twice, `xor3_twice_grid`): the output feature is the token's own bit
  `-1`, not 1 (`t3_final`).
* CE-4, outputs made by MLPs. **Lean, end to end** (`ce4`): `t4` returns its input; row 0 holds the
  input at feature 1 (`t4_grid`, `xor3_input_grid`), where the embedding writes `code[0]` and the
  head adds the input: the output feature is 2, not 1 (`t4_final`).
* CE-5, window ≤ CONTEXT. **Lean, through the attention, not end to end** (`ce5`): `t5` is `[x0 >= 1]`
  on a window of 513 tokens, `x0` the bit of the oldest; on tokens `1, 0, 0, ..., 0` at `p = 512`,
  the oldest token is not visible (`not_visible`, `oldest_not_visible`), the head meant for it
  writes the mean of what it sees (`head_const`), all -1, and the stream after the attention holds
  -1 at the feature of `x0`, whose bit is 1 (`t5_afterAttn`): the conclusion of `Stmt_attn_real`
  fails, by 2 (`ce5` proves this failure of the attention, not of `MainReal`'s conclusion).
  **Numeric simulation, end to end** (`sim.py`): the output feature is then -1, not 1.
* CE-6, one token per code. **Lean, end to end, real and bf16** (`ce6`): two tokens of one code have
  one logit (`logit6`), so neither has the largest, for every rounding too.
* CE-7, flat GLUs (`RealOK.flat`). **Numeric simulation** (`sim.py`, mpmath at 60 digits, the exact
  real-arithmetic forward pass: norms, skip, gelu units, softmax attention with RoPE; tokens `[0, 1]`,
  `p = 1`, the neighbour holding the other bit leaks `2 e^-30 ≈ 1.87e-13` into the input): a chain
  of `n` GLUs `relu(y + 1/2) * (2/3) y` (identity on bits, gates 1/2 and 3/2, weights ≤ 256, second-
  order gain 2/3, slope 5/3 at 1) is off by more than `1e-9` from depth `n = 17` and gives the wrong
  bit at `n = 58`; with `relu(y + 1/2) * 10 y + relu(y - 1/2) * (52 y - 80)` (slope 23 at 1), from
  `n = 3` and at `n = 10`. **Lean, local** (`glu7_props`, `glu7_gain`, `glu7_amplifies`,
  `glu7_iterate`): the GLU is a well-formed unit of slope 5/3 that multiplies an error `δ ≤ 1/4` by at
  least 3/2. Such GLUs are what `MainRealDeep` covers, to the depth where `(1 + σ)^D ≤ 10^4`
  (`NonVacuous.deep_beyond_realOK`): for `σ = 5/3`, `D ≤ 9`, where the error is below `1e-9`.
* CE-8, bounded weights (`RealOK.bounded`, now `2^16`). **Numeric simulation** (`sim.py`):
  `[w x - w >= 0]` after one head, as in CE-7, is exact to `1e-49` for `w ≤ 2e12`, off by 0.74 for
  `w = 5e12`, and the wrong bit for `w ≥ 1e13`. **Lean, local** (`ce8_local`): an input error of
  `1/(2w)` puts the gate at `-1/2`.
* CE-9, bf16 weights. **Lean, end to end, every rounding** (`ce9`): `t9` is two step units, both on,
  with weights `1027/2` and `-1025/2` (output 1); its compiled weights 1027 and -1025 round to 1024
  and -1024 under every rounding (`fl_1027`), the two units cancel, and the output feature is at
  most 0, not 1 (`t9_final`). Its gates, values and products are bf16 numbers (`F_sums`, by its
  second disjunct). So the condition is needed, not only sufficient.
* CE-10, bf16 sums. **Lean, end to end, every rounding** (`ce10`): `t10` is `[128 x - 64 >= 0]` on
  the last token's bit, its weights all bf16 numbers; at `x = 1` the gates of `u+` and `u-` are 8272
  and 8240, which every rounding takes to 8256 (`fl_8272`, `fl_8240`), so the units cancel and the
  output feature is at most 0, not 1 (`t10_final`). The gate 8272 is above 0 and not a bf16 number,
  so neither disjunct of `sums` holds (the relaxed `sums` only lets a gate at most 0 be inexact).
* CE-11, width. **Argument, with a Lean lemma** (`ce11`, `ce11_embed`): a `Rounding` is free beyond
  `maxBF16`; if `2 sqrt(width) > maxBF16` one rounding holds `ONE` as 0 in the embedding, the norm
  divides it by 0, and every unit then reads `ONE` as 0. Such a width (`W > maxBF16^2 / 4 ≈ 2.9e76`
  features) cannot be built, so no concrete network is given.
* CE-12, the norm's `ONE` in bf16 (the model's `BF16.one` in `BF16.norm`). Python's
  `rmsnorm(x, row, one)` reads `ONE` as the float `one = 2 sqrt W`, while a bf16 stream holds
  `fl(one)`. **Lean, every rounding** (`fl_two_sqrt21`, `ce12_ratio`, `ce12`): at `W = 21`,
  `fl(2 sqrt 21) = 147/16` (`2 sqrt 21 ≈ 9.16515`; the next bf16 numbers are `146/16`, `147/16`), and
  on the embedding `[147/16, ±1, 0, ...]` of a one-bit program (`count = 1`) the literal norm gives
  the bit as `fl(sqrt(85/21 + EPS) / sqrt(((147/16)^2 + 1)/21 + EPS)) = 255/256`, not 1; with `one`
  the bf16 number the stream holds (our model) it gives 1. **Numeric simulation, end to end in bf16**
  (`sim.py`, `ce12`, round to nearest even): the majority of 19 one-bit tokens (one threshold gate,
  width 21, `IsProgram` and `BF16OK`) under the literal reading: the heads write the inputs as
  `±255/256`, and at exactly 10 ones the output feature is `1.015625`, not 1; at every other count
  it is exact, the sign is always right, and so are the logits. Under our model all are exact. So
  the literal reading breaks `MainBF16`'s exactness, not the bit. The patch, which `BF16.one`
  models: round `one` once, `one = round_to_bf16(2 * sqrt(W))`, and use it in the stream and the norms.
* CE-13, RoPE angles. **Argument, with a numeric simulation of the scores** (`ce13.py`, numpy
  float32; no Lean): the model turns each pair by the exact angle `pos * turn i` and rounds what
  RoPE gives. With float32 angles `fl32(pos * fl32(turn i))`, pair `i` at `p` is off by up to
  `ulp(p turn i)/2 ≈ 2^-24 p turn i` (pair 0, `turn 0 = 1`, exact while `p < 2^24`), and the gap of 30
  between a head's target score and the best other one loses up to `attnScale * Σ_i |Δφ_i|`, linear
  in `p`. Simulated (every head `h < 512`, every visible offset, 12 random `p` per octave): the gap
  loses `< 0.035` below `2^16`, `0.07` in `[2^16, 2^17)`, past the `0.045` that the attention's budget
  in real arithmetic allows (`1022 e^-30 ≤ 1e-10`, `Stmt_attn_real`): so from `p ≈ 2^16` the real
  bound fails with float32 angles. It loses 1 to 10 in `[2^20, 2^24)`, which bf16 survives (a head's
  leak rounds away while the gap is at least `ln(511 * 2^10) ≈ 13.2`). From `p = 2^24 = 16777216`
  float32 cannot hold every position, pair 0's angles collide, and the gap is negative (`-10` to
  `-200`): the margin is exhausted and a head reads the wrong token, in any arithmetic. As the
  theorems hold at every position, an implementation with float32 angles needs `p < 2^16` (real
  bound) and `p < 2^24` (bf16), or angles taken mod `2 pi` in higher precision.

The end-to-end Lean counterexamples avoid the attention (no inputs: CE-1, 2, 3, 6, 9), or read
only position 0, where it sees one token (CE-4, 10). CE-5, proved through the attention only, has
every visible token hold one bit, so that every head writes it exactly. `sim.py` and `ce13.py`
(in `lean/simulations/`) also recompute CE-1, 2, 3, 4, 6 and agree:
`0.0`, `0.6824`, `-1`, `2`, equal logits.
-/

open Reifier Finset

namespace Reifier.CE

noncomputable section

/-! ## bf16 numbers -/

lemma isBF16_of_e (x : ℝ) (e : ℤ) (h1 : ((⌊x / 2 ^ e⌋ : ℤ) : ℝ) = x / 2 ^ e)
    (h2 : |⌊x / 2 ^ e⌋| < 256) (h3 : -133 ≤ e) (h4 : e ≤ 120) : IsBF16 x := by
  refine ⟨⌊x / 2 ^ e⌋, e, ?_, h2, h3, h4⟩
  rw [h1]
  field_simp

/-- Proves `IsBF16 x` for a numeral `x`, trying the exponents that the numerals here need. -/
macro "ce_bf16" : tactic => `(tactic| first
  | (refine isBF16_of_e _ 0 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 4 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 6 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 13 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-1) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-5) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 1 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 2 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 3 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-2) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-3) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-4) ?_ ?_ ?_ ?_ <;> norm_num <;> done))


lemma bf16_abs_le {x : ℝ} (h : IsBF16 x) : |x| ≤ maxBF16 := by
  obtain ⟨m, e, rfl, hm, he1, he2⟩ := h
  have hm' : |(m : ℝ)| ≤ 255 := by
    have : |m| ≤ 255 := by omega
    exact_mod_cast this
  have he : (2 : ℝ) ^ e ≤ 2 ^ (120 : ℤ) := zpow_le_zpow_right₀ (by norm_num) he2
  have hpos : (0 : ℝ) < 2 ^ e := zpow_pos (by norm_num) e
  rw [abs_mul, abs_of_pos hpos]
  unfold maxBF16
  have : (2 : ℝ) ^ (120 : ℤ) = 2 ^ (120 : ℕ) := by norm_num
  rw [this] at he
  nlinarith [abs_nonneg (m : ℝ)]

lemma fl_bf16 (R : Rounding) {x : ℝ} (h : IsBF16 x) : R.fl x = x := by
  have := R.fl_nearest x x (bf16_abs_le h) h
  simp at this
  linarith

lemma fl_near (R : Rounding) {x r : ℝ} (hx : |x| ≤ maxBF16) (hr : IsBF16 r) :
    |R.fl x - x| ≤ |r - x| := R.fl_nearest x r hx hr

lemma bf16_zero : IsBF16 0 := ⟨0, 0, by simp, by norm_num, by norm_num, by norm_num⟩

lemma fl_nonpos (R : Rounding) {x : ℝ} (hx : |x| ≤ maxBF16) (h : x ≤ 0) : R.fl x ≤ 0 := by
  have := fl_near R hx bf16_zero
  rw [zero_sub, abs_neg, abs_of_nonpos h] at this
  have := (abs_le.mp this).2
  linarith

lemma maxBF16_big : (10 : ℝ) ^ 30 ≤ maxBF16 := by unfold maxBF16; norm_num

/-- A bf16 number at least `2^(k+7)` in size is a multiple of `2^k`: it has 8 bits. -/
lemma bf16_grid {r : ℝ} (hr : IsBF16 r) (k : ℤ) (hk : (2 : ℝ) ^ (k + 7) ≤ |r|) :
    ∃ n : ℤ, r = n * 2 ^ k := by
  obtain ⟨m, e, rfl, hm, _, _⟩ := hr
  by_cases hek : k ≤ e
  · refine ⟨m * 2 ^ (e - k).toNat, ?_⟩
    push_cast
    rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ (by norm_num)]
    congr 2
    omega
  · exfalso
    have hm' : |(m : ℝ)| ≤ 255 := by
      have : |m| ≤ 255 := by omega
      exact_mod_cast this
    have hpos : (0 : ℝ) < 2 ^ e := zpow_pos (by norm_num) e
    have he : (2 : ℝ) ^ e ≤ 2 ^ (k - 1) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h7 : (2 : ℝ) ^ (k + 7) = 2 ^ (k - 1) * 256 := by
      rw [show k + 7 = (k - 1) + 8 by ring, zpow_add₀ (by norm_num)]; norm_num
    rw [abs_mul, abs_of_pos hpos, h7] at hk
    have : (0 : ℝ) < 2 ^ (k - 1) := zpow_pos (by norm_num) _
    nlinarith [abs_nonneg (m : ℝ)]

/-- `R.fl x = c` when `c` is the only bf16 number as near to `x` as `c`. -/
lemma fl_eq_of_unique (R : Rounding) {x c : ℝ} (hx : |x| ≤ maxBF16) (hc : IsBF16 c)
    (h : ∀ r, IsBF16 r → |r - x| ≤ |c - x| → r = c) : R.fl x = c :=
  h _ (R.fl_mem x hx) (R.fl_nearest x c hx hc)

/-- The bf16 numbers within 32 of 8256: only 8256 itself, its neighbours being 8192 and 8320. -/
lemma bf16_near_8256 {r : ℝ} (hr : IsBF16 r) (h : |r - 8256| ≤ 32) : r = 8256 := by
  have h1 : 8224 ≤ r := by linarith [(abs_le.mp h).1]
  have h2 : r ≤ 8288 := by linarith [(abs_le.mp h).2]
  obtain ⟨n, rfl⟩ := bf16_grid hr 6 (by rw [abs_of_pos (by linarith)]; norm_num; linarith)
  norm_num at h1 h2 ⊢
  have a : (128 : ℝ) < n := by linarith
  have b : (n : ℝ) < 130 := by linarith
  have a' : (128 : ℤ) < n := by exact_mod_cast a
  have b' : n < (130 : ℤ) := by exact_mod_cast b
  have : n = 129 := by omega
  subst this; norm_num

/-- **CE-10's rounding**: whatever it does with ties, every rounding to bf16 rounds both 8272 and
8240 to 8256. -/
theorem fl_8272 (R : Rounding) : R.fl 8272 = 8256 := by
  refine fl_eq_of_unique R (by rw [abs_of_pos (by norm_num)]; linarith [maxBF16_big])
    ⟨129, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩ (fun r hr h => ?_)
  apply bf16_near_8256 hr
  norm_num at h
  have := abs_le.mp h
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem fl_8240 (R : Rounding) : R.fl 8240 = 8256 := by
  refine fl_eq_of_unique R (by rw [abs_of_pos (by norm_num)]; linarith [maxBF16_big])
    ⟨129, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩ (fun r hr h => ?_)
  apply bf16_near_8256 hr
  norm_num at h
  have := abs_le.mp h
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem not_bf16_8272 : ¬ IsBF16 (8272 : ℝ) := by
  intro h
  obtain ⟨n, hn⟩ := bf16_grid h 6 (by norm_num)
  norm_num at hn
  have a : (129 : ℝ) < n := by linarith
  have b : (n : ℝ) < 130 := by linarith
  have a' : (129 : ℤ) < n := by exact_mod_cast a
  have b' : n < (130 : ℤ) := by exact_mod_cast b
  omega


/-- The argument of `tanh` in `gelu`. -/
def geluArg (z : ℝ) : ℝ := Real.sqrt (2 / Real.pi) * (z + 0.044715 * z ^ 3)

/-- `gelu z = z / (1 + exp (-2 y))`, `y` the argument of its `tanh`. -/
lemma gelu_eq (z : ℝ) : gelu z = z / (1 + Real.exp (-2 * geluArg z)) := by
  unfold gelu
  rw [← geluArg, Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  set y := geluArg z
  have ha : 0 < Real.exp y := Real.exp_pos y
  have hb : Real.exp (-y) = (Real.exp y)⁻¹ := Real.exp_neg y
  have hc : Real.exp (-2 * y) = ((Real.exp y)⁻¹) ^ 2 := by
    rw [show -2 * y = -y + -y by ring, Real.exp_add, hb]; ring
  rw [hb, hc]
  field_simp
  ring

lemma one_add_exp_pos (y : ℝ) : 0 < 1 + Real.exp y := by positivity

lemma gelu_le_self {z : ℝ} (hz : 0 ≤ z) : gelu z ≤ z := by
  rw [gelu_eq]
  have h := one_add_exp_pos (-2 * geluArg z)
  rw [div_le_iff₀ h]
  nlinarith [Real.exp_pos (-2 * geluArg z)]

lemma gelu_nonpos {z : ℝ} (hz : z ≤ 0) : gelu z ≤ 0 := by
  rw [gelu_eq]
  exact div_nonpos_of_nonpos_of_nonneg hz (one_add_exp_pos _).le

lemma gelu_nonneg {z : ℝ} (hz : 0 ≤ z) : 0 ≤ gelu z := by
  rw [gelu_eq]
  exact div_nonneg hz (one_add_exp_pos _).le

lemma self_le_gelu {z : ℝ} (hz : z ≤ 0) : z ≤ gelu z := by
  rw [gelu_eq]
  have h := one_add_exp_pos (-2 * geluArg z)
  rw [le_div_iff₀ h]
  nlinarith [Real.exp_pos (-2 * geluArg z)]

lemma sqrt_two_div_pi_gt : (0.79 : ℝ) < Real.sqrt (2 / Real.pi) := by
  rw [Real.lt_sqrt (by norm_num)]
  rw [lt_div_iff₀ Real.pi_pos]
  nlinarith [Real.pi_lt_d2]

lemma sqrt_two_div_pi_lt : Real.sqrt (2 / Real.pi) < 0.8 := by
  rw [Real.sqrt_lt' (by norm_num)]
  rw [div_lt_iff₀ Real.pi_pos]
  nlinarith [Real.pi_gt_d2]

/-- For `z ≥ 0`, `gelu z ≥ z - z / (1 + 2 y)`. -/
lemma gelu_ge {z : ℝ} (hz : 0 ≤ z) : z - z / (1 + 2 * geluArg z) ≤ gelu z := by
  have hy : 0 ≤ geluArg z := by
    unfold geluArg; have := sqrt_two_div_pi_gt; positivity
  rw [gelu_eq]
  set y := geluArg z
  set e := Real.exp (-2 * y)
  have he : 0 < e := Real.exp_pos _
  -- e * (1 + 2 y) ≤ 1
  have h1 : 1 + 2 * y ≤ Real.exp (2 * y) := by
    have := Real.add_one_le_exp (2 * y); linarith
  have h2 : e * Real.exp (2 * y) = 1 := by
    rw [← Real.exp_add]; simp
  have h3 : e * (1 + 2 * y) ≤ 1 := by nlinarith
  have hz' : z / (1 + 2 * y) ≥ z * e := by
    rw [ge_iff_le, le_div_iff₀ (by positivity)]; nlinarith
  have : z - z * e ≤ z / (1 + e) := by
    rw [le_div_iff₀ (by positivity)]; nlinarith [sq_nonneg e]
  linarith

lemma gelu_16 : 15 ≤ gelu 16 := by
  have h := gelu_ge (z := 16) (by norm_num)
  have hy : 100 ≤ geluArg 16 := by
    unfold geluArg; nlinarith [sqrt_two_div_pi_gt]
  have : (16 : ℝ) / (1 + 2 * geluArg 16) ≤ 1 := by
    rw [div_le_one (by linarith)]; linarith
  linarith

lemma exp_0836 : Real.exp 0.836 < 2.31 := by
  have h := Real.exp_bound (x := 0.836) (by norm_num [abs_of_pos]) (n := 5) (by norm_num)
  have h' := (abs_le.mp h).2
  norm_num [Finset.sum_range_succ, Nat.factorial, abs_of_pos] at h'
  linarith

/-- `gelu 1 < 0.85`: `gelu` is far from `relu` at a gate of 1. -/
lemma gelu_one_lt : gelu 1 < 0.85 := by
  rw [gelu_eq]
  have hy : geluArg 1 ≤ 0.836 := by
    unfold geluArg; nlinarith [sqrt_two_div_pi_lt, Real.sqrt_nonneg (2 / Real.pi)]
  have h2 : Real.exp (2 * geluArg 1) < 5.34 := by
    calc Real.exp (2 * geluArg 1) ≤ Real.exp (0.836 + 0.836) := Real.exp_le_exp.mpr (by linarith)
      _ = Real.exp 0.836 * Real.exp 0.836 := Real.exp_add _ _
      _ < 2.31 * 2.31 := by
          have := Real.exp_pos 0.836
          nlinarith [exp_0836]
      _ < 5.34 := by norm_num
  have h3 : Real.exp (-2 * geluArg 1) * Real.exp (2 * geluArg 1) = 1 := by
    rw [← Real.exp_add]; simp
  have he : 0 < Real.exp (-2 * geluArg 1) := Real.exp_pos _
  have h4 : 0.18 < Real.exp (-2 * geluArg 1) := by nlinarith
  rw [div_lt_iff₀ (by positivity)]
  nlinarith


/-- A unit's data on the features `< W`: its gate row, its value row, its column of `w_out`. -/
def unitData (W : ℕ) (u : Hidden) : List ℚ × List ℚ × List ℚ :=
  ((List.range W).map u.gate, (List.range W).map u.val, (List.range W).map u.out)

/-- `w . z` for a row given as a list -/
def dotL (l : List ℚ) (z : ℕ → ℝ) : ℝ := ∑ g ∈ range l.length, ((l.getD g 0 : ℚ) : ℝ) * z g

/-- A layer given by the data of its units, at feature `f`. -/
def evalData (act : ℝ → ℝ) (D : List (List ℚ × List ℚ × List ℚ)) (z : ℕ → ℝ) (f : ℕ) : ℝ :=
  (D.map fun d => ((d.2.2.getD f 0 : ℚ) : ℝ) * (act (dotL d.1 z) * dotL d.2.1 z)).sum

lemma getD_map_range (W : ℕ) (w : ℕ → ℚ) (g : ℕ) (hg : g < W) :
    ((List.range W).map w).getD g 0 = w g := by
  simp [List.getD_eq_getElem?_getD, hg]

lemma dot_eq_dotL (W : ℕ) (w : ℕ → ℚ) (z : ℕ → ℝ) : dot W w z = dotL ((List.range W).map w) z := by
  unfold dot dotL
  rw [List.length_map, List.length_range]
  refine sum_congr rfl (fun g hg => ?_)
  rw [getD_map_range W w g (mem_range.mp hg)]

lemma layerApply_eq (act : ℝ → ℝ) (W : ℕ) (us : List Hidden) (z : ℕ → ℝ) (f : ℕ) (hf : f < W) :
    layerApply act W us z f = evalData act (us.map (unitData W)) z f := by
  unfold layerApply evalData
  rw [List.map_map]
  congr 1
  refine List.map_congr_left (fun u _ => ?_)
  simp only [Function.comp, unitData, getD_map_range W u.out f hf, dot_eq_dotL]

lemma rows_two {l : List Row} (h : l.length = 2) :
    l = [l.getD 0 (fun _ => none), l.getD 1 (fun _ => none)] := by
  match l, h with
  | [a, b], _ => rfl

lemma rows_one {l : List Row} (h : l.length = 1) : l = [l.getD 0 (fun _ => none)] := by
  match l, h with
  | [a], _ => rfl

/-- The row `d` of the token grid. -/
abbrev GR (t : Trace) (d : ℕ) : Row := (tokenGrid t).rows.getD d (fun _ => none)

/-- The width of the token grid. -/
abbrev Wd (t : Trace) : ℕ := (tokenGrid t).width

/-- With two rows, the transformer's final stream is one layer after the attention. -/
lemma final_two (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ)
    (h : (tokenGrid t).rows.length = 2) :
    (transformer t codes).final toks p =
      netStep gelu (Wd t) (2 * Real.sqrt (Wd t)) (GR t 0) (layerUnits (Wd t) t.mlps (GR t 0) (GR t 1))
        ((transformer t codes).afterAttn toks p) := by
  unfold Transformer.final Network.final
  simp only [transformer, emit]
  rw [rows_two h]
  simp

/-- With one row, there is no layer. -/
lemma final_one (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ)
    (h : (tokenGrid t).rows.length = 1) :
    (transformer t codes).final toks p = (transformer t codes).afterAttn toks p := by
  unfold Transformer.final Network.final
  simp only [transformer, emit]
  rw [rows_one h]
  simp

/-- A program that reads no token has no heads: the stream after the attention is the
embedding. -/
lemma afterAttn_noheads (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ)
    (h : t.inputs.length / t.outputs.length = 0) :
    (transformer t codes).afterAttn toks p = (transformer t codes).embed (toks p) := by
  funext f
  simp [Transformer.afterAttn, transformer, h]

/-- The norm of a stream whose square sum is what the row says is the identity, but for `ONE`. -/
lemma rmsnorm_of_sum (W : ℕ) (row : Row) (one : ℝ) (x : ℕ → ℝ)
    (h : ∑ g ∈ range W, x g ^ 2 = one ^ 2 + count W row) :
    rmsnorm W row one x = fun f => if f = 0 then x f / one else x f := by
  funext f
  unfold rmsnorm rmsScale
  rw [h]
  have hpos : 0 < Real.sqrt ((one ^ 2 + count W row) / W + EPS) := by
    apply Real.sqrt_pos.mpr
    have : 0 ≤ (one ^ 2 + (count W row : ℝ)) / W := by positivity
    unfold EPS; linarith
  simp only
  rw [mul_div_assoc, mul_div_cancel₀ _ hpos.ne']


/-- An MLP of no inputs: one step gate `[b >= 0]`, with weight 1 into its output `y`. -/
def stepConst (y : Sig) (b : ℚ) : MLP where
  ins := []
  outs := [y]
  act := .step
  hidden := 1
  wi := fun _ _ => 0
  b := fun _ => b
  wo := fun _ _ => 1

/-- CE-1: the program `[-1/2 >= 0]`, which is 0. -/
def t1 : Trace where
  inputs := []
  mlps := [stepConst 0 (-1 / 2)]
  outputs := [0]

def codes2 : List (ℕ × List Bool) := [(0, [false]), (1, [true])]


lemma t1_rows : (tokenGrid t1).rows.length = 2 := by decide
lemma t1_width : Wd t1 = 3 := by decide
lemma t1_count : count 3 (GR t1 0) = 1 := by decide
lemma t1_units : (layerUnits 3 t1.mlps (GR t1 0) (GR t1 1)).map (unitData 3) =
    [([32, 0, 0], [1 / 32, 0, 0], [0, -1, 0]), ([16, 0, 0], [1 / 32, 0, 0], [0, 2, 0]),
     ([-16, 0, 0], [1 / 32, 0, 0], [0, -2, 0]), ([32, 0, 0], [0, 1 / 32, 0], [0, -1, 0])] := by
  decide +kernel


/-- The embedding of a token of `codes2` for a one-bit program of width 3. -/
lemma embed_codes2 (t : Trace) (hW : Wd t = 3) (hb : t.outputs.length = 1) (τ : ℕ) (hτ : τ < 2)
    (f : ℕ) : (transformer t codes2).embed τ f =
      if f = 0 then 2 * Real.sqrt 3 else if f = 1 then (if τ = 1 then 1 else -1) else 0 := by
  simp only [Wd] at hW
  have l0 : codes2.lookup 0 = some [false] := rfl
  have l1 : codes2.lookup 1 = some [true] := rfl
  simp only [transformer, emit, hW, hb]
  rcases Nat.lt_or_ge f 2 with h | h
  · interval_cases τ <;> interval_cases f <;> simp [l0, l1]
  · have h0 : f ≠ 0 := by omega
    have h1' : f ≠ 1 := by omega
    have h1 : ¬ f ≤ 1 := by omega
    interval_cases τ <;> simp [h0, h1', h1, l0, l1]

lemma sum_embed_codes2 (t : Trace) (hW : Wd t = 3) (hb : t.outputs.length = 1) (τ : ℕ)
    (hτ : τ < 2) : ∑ g ∈ range 3, (transformer t codes2).embed τ g ^ 2 = (2 * Real.sqrt 3) ^ 2 + 1 := by
  interval_cases τ <;> simp [sum_range_succ, embed_codes2 t hW hb _ hτ]

/-- For a one-bit program of no inputs and one layer, of width 3: feature 1 at the end is the
token's code bit plus what the layer's units write there, on the normed stream `[1, ±1, 0]`. -/
lemma final_noinput (t : Trace) (hrows : (tokenGrid t).rows.length = 2) (hW : Wd t = 3)
    (hb : t.outputs.length = 1) (hin : t.inputs = []) (hc : count 3 (GR t 0) = 1)
    (D : List (List ℚ × List ℚ × List ℚ))
    (hD : (layerUnits 3 t.mlps (GR t 0) (GR t 1)).map (unitData 3) = D) (τ : ℕ) (hτ : τ < 2) (p : ℕ) :
    (transformer t codes2).final (fun _ => τ) p 1 = (if τ = 1 then 1 else -1) +
      evalData gelu D (fun f => if f = 0 then 1 else if f = 1 then (if τ = 1 then 1 else -1) else 0) 1 := by
  rw [final_two t codes2 _ p hrows]
  unfold netStep
  rw [hW, afterAttn_noheads _ _ _ _ (by simp [hin])]
  simp only [Nat.cast_ofNat]
  have hsum : ∑ g ∈ range 3, (transformer t codes2).embed τ g ^ 2 =
      (2 * Real.sqrt 3) ^ 2 + count 3 (GR t 0) := by
    rw [hc, sum_embed_codes2 t hW hb τ hτ]; simp
  rw [rmsnorm_of_sum _ _ _ _ hsum, layerApply_eq _ _ _ _ _ (by norm_num), hD]
  have h3 : (2 * Real.sqrt 3) ≠ 0 := by positivity
  congr 1
  · simp [embed_codes2 t hW hb τ hτ]
  · congr 1
    funext f
    rw [embed_codes2 t hW hb τ hτ]
    split_ifs <;> simp_all

theorem t1_final (p : ℕ) : (transformer t1 codes2).final (fun _ => 0) p 1 =
    -1 + (gelu 16 - gelu (-16)) / 16 := by
  rw [final_noinput t1 t1_rows t1_width rfl rfl t1_count _ t1_units 0 (by norm_num) p]
  simp [evalData, dotL, sum_range_succ]
  ring_nf

/-! ## The fields of the hypotheses, one by one -/

def F_nodup (t : Trace) : Prop := (t.inputs ++ t.mlps.flatMap MLP.outs).Nodup
def F_outs_ne (t : Trace) : Prop := ∀ m ∈ t.mlps, m.outs ≠ []
def F_topo (t : Trace) : Prop := ∀ (i : ℕ) (hi : i < t.mlps.length), ∀ x ∈ t.mlps[i].ins,
    x ∈ t.inputs ∨ ∃ (j : ℕ) (hj : j < i), x ∈ (t.mlps[j]'(by omega)).outs
def F_bits (t : Trace) : Prop :=
  ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, ∀ y ∈ m.outs, IsBit (t.values (t.env x) y)
def F_step_margin (t : Trace) : Prop := ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act = .step →
    ∀ j < m.hidden, 0 ≤ m.pre (t.inVals x m) j ∨ m.pre (t.inVals x m) j ≤ -3 / 4
def F_gate_margin (t : Trace) : Prop := ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step →
    ∀ j < m.hidden, m.pre (t.inVals x m) j = 0 ∨ 1 / 2 ≤ |m.pre (t.inVals x m) j|
def F_outputs_ne (t : Trace) : Prop := t.outputs ≠ []
def F_outputs_nodup (t : Trace) : Prop := t.outputs.Nodup
def F_outputs_made (t : Trace) : Prop := ∀ y ∈ t.outputs, ∃ m ∈ t.mlps, y ∈ m.outs
def F_window_dvd (t : Trace) : Prop := t.outputs.length ∣ t.inputs.length
def F_window_le (t : Trace) : Prop := t.window ≤ CONTEXT

/-- The fields above are exactly those of `IsProgram`. -/
theorem isProgram_iff (t : Trace) : t.IsProgram ↔
    F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_step_margin t ∧ F_gate_margin t ∧
      F_outputs_ne t ∧ F_outputs_nodup t ∧ F_outputs_made t ∧ F_window_dvd t ∧ F_window_le t :=
  ⟨fun h => ⟨h.nodup, h.outs_ne, h.topo, h.bits, h.step_margin, h.gate_margin, h.outputs_ne,
      h.outputs_nodup, h.outputs_made, h.window_dvd, h.window_le⟩,
    fun ⟨a, b, c, d, e, f, g, h, i, j, k⟩ => ⟨⟨a, b, c, d, e, f⟩, g, h, i, j, k⟩⟩

/-- The fields of `RealOK` but its gate margin (`Trace.GateMargin`). -/
def F_flat (t : Trace) : Prop := ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step →
    ∀ i < m.outs.length, ∀ k < m.ins.length, m.slope (t.inVals x m) i k = 0
def F_bounded (t : Trace) : Prop := ∀ m ∈ t.mlps, m.Bounded (2 ^ 16)
def F_glu_gain (t : Trace) : Prop :=
  ∀ m ∈ t.mlps, m.act = .reglu → ∀ i < m.outs.length, m.gluGain i ≤ 10 ^ 8

/-- The fields above, and the strict gate margin, are exactly those of `RealOK`. -/
theorem realOK_iff (t : Trace) :
    t.RealOK ↔ t.GateMargin ∧ F_flat t ∧ F_bounded t ∧ F_glu_gain t :=
  ⟨fun h => ⟨h.gate_margin, h.flat, h.bounded, h.glu_gain⟩, fun ⟨a, b, c, d⟩ => ⟨a, b, c, d⟩⟩

/-- The fields of `RealCore G σ` but its gate margin. -/
def F_slope (t : Trace) (σ : ℝ) : Prop := ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step →
    ∀ i < m.outs.length, m.slopeSum (t.inVals x m) i ≤ σ
def F_glu_gainG (t : Trace) (G : ℝ) : Prop :=
  ∀ m ∈ t.mlps, m.act = .reglu → ∀ i < m.outs.length, (m.gluGain i : ℝ) ≤ G

theorem realCore_iff (t : Trace) (G σ : ℝ) :
    t.RealCore G σ ↔ t.GateMargin ∧ F_slope t σ ∧ F_bounded t ∧ F_glu_gainG t G :=
  ⟨fun h => ⟨h.gate_margin, h.slope, h.bounded, h.glu_gain⟩, fun ⟨a, b, c, d⟩ => ⟨a, b, c, d⟩⟩

/-- A trace of threshold gates only is `RealOK` if its weights are bounded: every other field is
about `relu` or GLU units. -/
theorem realOK_of_steps (t : Trace) (hs : ∀ m ∈ t.mlps, m.act = .step)
    (hb : ∀ m ∈ t.mlps, m.Bounded (2 ^ 16)) : t.RealOK where
  gate_margin := fun _ m hm hact => absurd (hs m hm) hact
  flat := fun _ m hm hact => absurd (hs m hm) hact
  bounded := hb
  glu_gain := fun m hm hact => by rw [hs m hm] at hact; exact absurd hact (by decide)

/-- The conclusion of `MainReal` at one program, coding, token sequence and position. -/
def RealConcl (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) : Prop :=
  let M := transformer t codes
  let x := windowBits t codes toks p
  (∀ i < t.outputs.length, |M.final toks p (1 + i) - (2 * t.outVal x i - 1)| ≤ 1 / 10 ^ 9) ∧
  ∀ τ, CodeIs t codes τ (valAt t codes toks p) →
    ∀ τ' < M.vocab, τ' ≠ τ → M.logit toks p τ' < M.logit toks p τ

theorem mainReal_iff : MainReal ↔ ∀ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ)
    (p : ℕ), t.IsProgram → t.RealOK → CodesOK t codes → PosOK t codes toks p →
      RealConcl t codes toks p := Iff.rfl

/-- The conclusion of `MainRealDeep` at one gain, program, coding, token sequence and position. -/
def RealDeepConcl (σ : ℝ) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) :
    Prop :=
  let M := transformer t codes
  let x := windowBits t codes toks p
  (∀ i < t.outputs.length, |M.final toks p (1 + i) - (2 * t.outVal x i - 1)| ≤
    (1 + σ) ^ ((tokenGrid t).rows.length - 1) / 10 ^ 10) ∧
  ∀ τ, CodeIs t codes τ (valAt t codes toks p) →
    ∀ τ' < M.vocab, τ' ≠ τ → M.logit toks p τ' < M.logit toks p τ

theorem mainRealDeep_iff : MainRealDeep ↔ ∀ (σ : ℝ) (t : Trace) (codes : List (ℕ × List Bool))
    (toks : ℕ → ℕ) (p : ℕ), t.IsProgram → t.RealCore (10 ^ 5) σ → 0 ≤ σ →
      (1 + σ) ^ ((tokenGrid t).rows.length - 1) ≤ 10 ^ 4 → CodesOK t codes →
      PosOK t codes toks p → RealDeepConcl σ t codes toks p := Iff.rfl

/-- The conclusion of `MainBF16` at one rounding, program, coding, token sequence and position. -/
def BF16Concl (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) :
    Prop :=
  let M := transformer t codes
  let x := windowBits t codes toks p
  (∀ i < t.outputs.length, BF16.finalAt R M toks p (1 + i) = 2 * t.outVal x i - 1) ∧
  (2 * Real.sqrt (tokenGrid t).width + t.outputs.length ≤ 250 →
    ∀ τ, CodeIs t codes τ (valAt t codes toks p) →
      ∀ τ' < M.vocab, τ' ≠ τ → BF16.logit R M toks p τ' < BF16.logit R M toks p τ)

theorem mainBF16_iff : MainBF16 ↔ ∀ (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool))
    (toks : ℕ → ℕ) (p : ℕ), t.IsProgram → t.BF16OK → CodesOK t codes → PosOK t codes toks p →
      WidthOK t → BF16Concl R t codes toks p := Iff.rfl

def F_weights (t : Trace) : Prop := ∀ m ∈ t.mlps, ∀ j < m.compiled.n,
    (∀ s, IsBF16 (m.compiled.gOn j s)) ∧ (∀ s, IsBF16 (m.compiled.vOn j s)) ∧
      IsBF16 (m.compiled.gb j) ∧ IsBF16 (m.compiled.vb j) ∧
        ∀ i < m.outs.length, IsBF16 (m.compiled.wo i j)
def F_sums (t : Trace) : Prop := ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, ∀ j < m.compiled.n,
    (-maxBF16 ≤ m.compiled.gateAt (t.inVals x m) j ∧ m.compiled.gateAt (t.inVals x m) j ≤ 0) ∨
    (IsBF16 (m.compiled.gateAt (t.inVals x m) j) ∧ IsBF16 (m.compiled.valAt (t.inVals x m) j) ∧
      IsBF16 (relu (m.compiled.gateAt (t.inVals x m) j) * m.compiled.valAt (t.inVals x m) j))

/-- The fields above are exactly those of `BF16OK`. -/
theorem bf16OK_iff (t : Trace) : t.BF16OK ↔ F_weights t ∧ F_sums t :=
  ⟨fun h => ⟨h.weights, h.sums⟩, fun ⟨a, b⟩ => ⟨a, b⟩⟩

/-! ## CE-1 -/

/-- The local failure: at `z = -1/2` the reglu form of `[z >= 0]` is `1/2`, not `0`. -/
theorem ce1_local : relu (4 * (-1 / 2) + 2 + 1 / 2) - relu (4 * (-1 / 2) + 2 - 1 / 2) = 1 / 2 ∧
    heaviside (-1 / 2) = 0 := by
  norm_num [relu, heaviside]

lemma t1_outVal (x : ℕ → Bool) : t1.outVal x 0 = 0 := by
  simp [Trace.outVal, Trace.values, t1, stepConst, MLP.apply, MLP.out, MLP.unit, MLP.pre,
    heaviside]
  norm_num

lemma stepConst_bits (y : Sig) (b : ℚ) (t : Trace) (ht : t.mlps = [stepConst y b])
    (_htin : t.inputs = []) : F_bits t := by
  intro x m hm z hz
  rw [ht] at hm
  simp at hm; subst hm
  simp [stepConst] at hz; subst hz
  simp only [Trace.values, ht, List.foldl, MLP.apply, stepConst]
  simp [MLP.out, MLP.unit, MLP.pre, heaviside]
  split_ifs <;> simp [IsBit]

theorem ce1 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    (F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_gate_margin t ∧
      F_outputs_ne t ∧ F_outputs_nodup t ∧ F_outputs_made t ∧ F_window_dvd t ∧ F_window_le t) ∧
    ¬ F_step_margin t ∧ t.RealOK ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
    ¬ RealConcl t codes toks p := by
  refine ⟨t1, codes2, fun _ => 0, 0, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · unfold F_nodup; decide
  · simp [F_outs_ne, t1, stepConst]
  · intro i hi x hx; simp [t1] at hi; subst hi; simp [t1, stepConst] at hx
  · exact stepConst_bits 0 _ t1 rfl rfl
  · intro x m hm hact; simp [t1] at hm; subst hm; simp [stepConst] at hact
  · simp [F_outputs_ne, t1]
  · unfold F_outputs_nodup; decide
  · intro y hy; simp [t1] at hy; subst hy
    exact ⟨stepConst 0 (-1 / 2), by simp [t1], by simp [stepConst]⟩
  · unfold F_window_dvd; decide
  · unfold F_window_le; decide
  · intro h
    have := h (fun _ => false) (stepConst 0 (-1 / 2)) (by simp [t1]) rfl 0 (by simp [stepConst])
    simp [MLP.pre, stepConst] at this
    norm_num at this
  · refine realOK_of_steps _ (fun m hm => by simp [t1] at hm; subst hm; rfl) (fun m hm => ?_)
    simp [t1] at hm; subst hm
    refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, stepConst] at hj ⊢ <;> norm_num
  · refine ⟨by decide, by decide, by decide⟩
  · refine ⟨by simp [Trace.window, t1], fun h _ => by simp [codes2]⟩
  · intro h
    have := h.1 0 (by simp [t1])
    rw [t1_outVal, show 1 + 0 = 1 from rfl, t1_final] at this
    have h16 := gelu_16
    have hn := gelu_nonpos (z := -16) (by norm_num)
    rw [abs_le] at this
    linarith [this.2]

/-! ## CE-2 -/

/-- An MLP of no inputs: one GLU unit `relu(1/32) * 32`, which is 1. -/
def gluConst : MLP where
  ins := []
  outs := [0]
  act := .reglu
  hidden := 1
  wi := fun _ _ => 0
  b := fun j => if j = 0 then 1 / 32 else 32
  wo := fun _ _ => 1

/-- CE-2: the program `relu(1/32) * 32`, which is 1. -/
def t2 : Trace where
  inputs := []
  mlps := [gluConst]
  outputs := [0]

lemma t2_rows : (tokenGrid t2).rows.length = 2 := by decide
lemma t2_width : Wd t2 = 3 := by decide
lemma t2_count : count 3 (GR t2 0) = 1 := by decide
lemma t2_units : (layerUnits 3 t2.mlps (GR t2 0) (GR t2 1)).map (unitData 3) =
    [([32, 0, 0], [1 / 32, 0, 0], [0, -1, 0]), ([1, 0, 0], [1, 0, 0], [0, 2, 0]),
     ([32, 0, 0], [0, 1 / 32, 0], [0, -1, 0])] := by
  decide +kernel

theorem t2_final (p : ℕ) : (transformer t2 codes2).final (fun _ => 0) p 1 = -1 + 2 * gelu 1 := by
  rw [final_noinput t2 t2_rows t2_width rfl rfl t2_count _ t2_units 0 (by norm_num) p]
  simp [evalData, dotL, sum_range_succ]

lemma t2_values (x : ℕ → Bool) : t2.values (t2.env x) 0 = 1 := by
  simp [Trace.values, t2, gluConst, MLP.apply, MLP.out, MLP.unit, MLP.pre, relu]

/-- The local failure: the compiled unit's gate is `32 * 1/32 = 1`, where `gelu` is not `relu`. -/
theorem ce2_local : gelu (32 * (1 / 32)) < 0.85 ∧ relu (32 * (1 / 32)) = 1 := by
  refine ⟨by rw [show (32 : ℝ) * (1 / 32) = 1 by norm_num]; exact gelu_one_lt, by norm_num [relu]⟩

/-- `t2` is a program on tokens but for its gate, which is `1/32`: neither 0 nor 1/2 from 0. -/
theorem t2_fields : F_nodup t2 ∧ F_outs_ne t2 ∧ F_topo t2 ∧ F_bits t2 ∧ F_step_margin t2 ∧
    F_outputs_ne t2 ∧ F_outputs_nodup t2 ∧ F_outputs_made t2 ∧ F_window_dvd t2 ∧
    F_window_le t2 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · unfold F_nodup; decide
  · simp [F_outs_ne, t2, gluConst]
  · intro i hi x hx; simp [t2] at hi; subst hi; simp [t2, gluConst] at hx
  · intro x m hm y hy
    simp [t2] at hm; subst hm; simp [gluConst] at hy; subst hy
    rw [t2_values]; right; rfl
  · intro x m hm hact; simp [t2] at hm; subst hm; simp [gluConst] at hact
  · simp [F_outputs_ne, t2]
  · unfold F_outputs_nodup; decide
  · intro y hy; simp [t2] at hy; subst hy
    exact ⟨gluConst, by simp [t2], by simp [gluConst]⟩
  · unfold F_window_dvd; decide
  · unfold F_window_le; decide

theorem t2_not_gate_margin : ¬ F_gate_margin t2 := by
  intro h
  have := h (fun _ => false) gluConst (by simp [t2]) (by simp [gluConst]) 0 (by simp [gluConst])
  simp [MLP.pre, gluConst] at this
  norm_num at this

/-- CE-2 in real arithmetic: `t2` satisfies every hypothesis of `MainReal` but the gate margins,
the weak one of `IsProgram` and so the strict one of `RealOK` too, and the transformer's output
is `-1 + 2 gelu 1 < 0.7`, not 1. -/
theorem ce2 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    (F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_step_margin t ∧
      F_outputs_ne t ∧ F_outputs_nodup t ∧ F_outputs_made t ∧ F_window_dvd t ∧ F_window_le t) ∧
    ¬ F_gate_margin t ∧ ¬ t.GateMargin ∧ (F_flat t ∧ F_bounded t ∧ F_glu_gain t) ∧
    CodesOK t codes ∧ PosOK t codes toks p ∧ ¬ RealConcl t codes toks p := by
  refine ⟨t2, codes2, fun _ => 0, 0, t2_fields, t2_not_gate_margin, ?_, ⟨?_, ?_, ?_⟩, ?_, ?_, ?_⟩
  · intro h
    have := h (fun _ => false) gluConst (by simp [t2]) (by simp [gluConst]) 0 (by simp [gluConst])
    simp [MLP.pre, gluConst] at this
    norm_num [abs_of_pos] at this
  · intro x m hm _ i _ k hk; simp [t2] at hm; subst hm; simp [gluConst] at hk
  · intro m hm; simp [t2] at hm; subst hm
    refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, gluConst] at hj ⊢
    all_goals (try interval_cases j)
    all_goals norm_num
  · intro m hm _ i _; simp [t2] at hm; subst hm
    simp [MLP.gluGain, gluConst]
  · refine ⟨by decide, by decide, by decide⟩
  · refine ⟨by simp [Trace.window, t2], fun h _ => by simp [codes2]⟩
  · intro h
    have := h.1 0 (by simp [t2])
    have hv : t2.outVal (windowBits t2 codes2 (fun _ => 0) 0) 0 = 1 := t2_values _
    rw [hv, show 1 + 0 = 1 from rfl, t2_final] at this
    have := gelu_one_lt
    rw [abs_le] at *
    linarith

/-! ## CE-3 -/

/-- CE-3: the program `[0 >= 0]`, which is 1, returned twice. -/
def t3 : Trace where
  inputs := []
  mlps := [stepConst 0 0]
  outputs := [0, 0]

def codes3 : List (ℕ × List Bool) := [(0, [false, false]), (1, [true, true])]

def cells (g : Grid) : List (List (Option Name)) := g.rows.map fun r => (List.range g.width).map r

/-- The token grid of `t3`: the second output takes the first one's feature, and `held` gives
feature 1 back to `code[0]`. -/
theorem t3_grid : cells (tokenGrid t3) =
    [[some .one, some (.code 0), some (.code 1), none],
     [some .one, some (.code 0), some (.sig 0), none]] := by decide

/-- As asked, on the README's `xor3` returning `odd` twice: the last row holds `code[0]` at
feature 1, where the logits read the first output. -/
theorem xor3_twice_grid :
    ((tokenGrid { Examples.xor3 with outputs := [6, 6] }).rows.getLast?.getD fun _ => none) 1 =
      some (.code 0) := by decide

lemma t3_rows : (tokenGrid t3).rows.length = 2 := by decide
lemma t3_width : Wd t3 = 4 := by decide
lemma t3_units : (layerUnits 4 t3.mlps (GR t3 0) (GR t3 1)).map (unitData 4) =
    [([32, 0, 0, 0], [1 / 32, 0, 0, 0], [0, 0, -1, 0]), ([80, 0, 0, 0], [1 / 32, 0, 0, 0], [0, 0, 2, 0]),
     ([48, 0, 0, 0], [1 / 32, 0, 0, 0], [0, 0, -2, 0]),
     ([32, 0, 0, 0], [0, 0, 1 / 32, 0], [0, 0, -1, 0])] := by
  decide +kernel

/-- Nothing writes into feature 1: it keeps the token's own `code[0]`. -/
theorem t3_final (p : ℕ) : (transformer t3 codes3).final (fun _ => 0) p 1 = -1 := by
  rw [final_two t3 codes3 _ p t3_rows]
  unfold netStep
  rw [t3_width, layerApply_eq _ _ _ _ _ (by norm_num), t3_units,
    afterAttn_noheads _ _ _ _ (by decide)]
  have l0 : codes3.lookup 0 = some [false, false] := rfl
  simp [evalData, transformer, l0, t3]

lemma t3_values (x : ℕ → Bool) : t3.values (t3.env x) 0 = 1 := by
  simp [Trace.values, t3, stepConst, MLP.apply, MLP.out, MLP.unit, MLP.pre, heaviside]

theorem ce3 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    (F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_step_margin t ∧ F_gate_margin t ∧
      F_outputs_ne t ∧ F_outputs_made t ∧ F_window_dvd t ∧ F_window_le t) ∧
    ¬ F_outputs_nodup t ∧ t.RealOK ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
    ¬ RealConcl t codes toks p := by
  refine ⟨t3, codes3, fun _ => 0, 0, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · unfold F_nodup; decide
  · simp [F_outs_ne, t3, stepConst]
  · intro i hi x hx; simp [t3] at hi; subst hi; simp [t3, stepConst] at hx
  · exact stepConst_bits 0 _ t3 rfl rfl
  · intro x m hm _ j hj; simp [t3] at hm; subst hm; simp [stepConst] at hj; subst hj
    left; simp [MLP.pre, stepConst]
  · intro x m hm hact; simp [t3] at hm; subst hm; simp [stepConst] at hact
  · simp [F_outputs_ne, t3]
  · intro y hy; simp [t3] at hy; subst hy
    exact ⟨stepConst 0 0, by simp [t3], by simp [stepConst]⟩
  · unfold F_window_dvd; decide
  · unfold F_window_le; decide
  · unfold F_outputs_nodup; decide
  · refine realOK_of_steps _ (fun m hm => by simp [t3] at hm; subst hm; rfl) (fun m hm => ?_)
    simp [t3] at hm; subst hm
    refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, stepConst] at hj ⊢
    norm_num
  · refine ⟨by decide, by decide, by decide⟩
  · refine ⟨by simp [Trace.window, t3], fun h _ => by simp [codes3]⟩
  · intro h
    have := h.1 0 (by simp [t3])
    have hv : t3.outVal (windowBits t3 codes3 (fun _ => 0) 0) 0 = 1 := t3_values _
    rw [hv, show 1 + 0 = 1 from rfl, t3_final] at this
    rw [abs_le] at this
    linarith [this.1]

/-! ## CE-4 -/

/-- CE-4: the program that returns its input, the bit of the last token. -/
def t4 : Trace where
  inputs := [0]
  mlps := []
  outputs := [0]

/-- Row 0 holds the input at feature 1, where the embedding writes `code[0]`; there is no
other row. -/
theorem t4_grid : cells (tokenGrid t4) = [[some .one, some (.sig 0), none]] := by decide

/-- As asked, on `xor3` with output `xs0`: row 0 holds the input `xs0` at feature 1. -/
theorem xor3_input_grid :
    ((tokenGrid { Examples.xor3 with outputs := [0] }).rows.headD fun _ => none) 1 =
      some (.sig 0) := by decide

lemma visible_zero : visible 0 = {0} := by decide

lemma t4_rows : (tokenGrid t4).rows.length = 1 := by decide

lemma t4_count : count 3 (transformer t4 codes2).embedded = 1 := by decide +kernel

lemma t4_o : (transformer t4 codes2).o 0 1 0 = 1 := by
  simp only [transformer]
  split_ifs with h
  · rfl
  · exact absurd (by decide) h

lemma t4_embed (f : ℕ) : (transformer t4 codes2).embed 1 f =
    if f = 0 then 2 * Real.sqrt 3 else if f = 1 then 1 else 0 := by
  have hW : (tokenGrid t4).width = 3 := by decide
  have l1 : codes2.lookup 1 = some [true] := rfl
  simp only [transformer, emit, hW]
  rcases Nat.lt_or_ge f 2 with h | h
  · interval_cases f <;> simp [l1, t4]
  · have h0 : f ≠ 0 := by omega
    have h1' : f ≠ 1 := by omega
    have h1 : ¬ f ≤ 1 := by omega
    simp [h0, h1', h1, l1, t4]

/-- The attention adds the input to the code bit already there: feature 1 holds `2`, not `1`. -/
theorem t4_final : (transformer t4 codes2).final (fun _ => 1) 0 1 = 2 := by
  rw [final_one t4 codes2 _ 0 t4_rows]
  have hheads : (transformer t4 codes2).heads = 1 := rfl
  have hbits : (transformer t4 codes2).bits = 1 := rfl
  have hW : (transformer t4 codes2).net.W = 3 := by
    show (tokenGrid t4).width = 3; decide
  have hone : (transformer t4 codes2).net.one = 2 * Real.sqrt 3 := by
    show 2 * Real.sqrt ((tokenGrid t4).width : ℝ) = _
    rw [show (tokenGrid t4).width = 3 by decide]; norm_num
  have hattn : (transformer t4 codes2).attn (fun _ => 1) 0 0 0 = 1 := by
    unfold Transformer.attn
    rw [visible_zero, sum_singleton, div_self (Real.exp_pos _).ne']
  have hsum : ∑ g ∈ range 3, (transformer t4 codes2).embed 1 g ^ 2 =
      ((transformer t4 codes2).net.one) ^ 2 + count 3 (transformer t4 codes2).embedded := by
    rw [t4_count, hone]; simp [sum_range_succ, t4_embed]
  have hval : (transformer t4 codes2).value (fun _ => 1) 0 0 = 1 := by
    unfold Transformer.value Transformer.matvec Transformer.zvec
    rw [hW, rmsnorm_of_sum _ _ _ _ hsum]
    have l1 : codes2.lookup 1 = some [true] := rfl
    simp [sum_range_succ, transformer, l1, t4]
  unfold Transformer.afterAttn
  rw [hheads, hbits, visible_zero]
  simp only [sum_range_one, sum_singleton, hattn, hval, t4_o, t4_embed]
  norm_num

theorem ce4 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    (F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_step_margin t ∧ F_gate_margin t ∧
      F_outputs_ne t ∧ F_outputs_nodup t ∧ F_window_dvd t ∧ F_window_le t) ∧
    ¬ F_outputs_made t ∧ t.RealOK ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
    ¬ RealConcl t codes toks p := by
  refine ⟨t4, codes2, fun _ => 1, 0, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · unfold F_nodup; decide
  · simp [F_outs_ne, t4]
  · intro i hi; simp [t4] at hi
  · intro x m hm; simp [t4] at hm
  · intro x m hm; simp [t4] at hm
  · intro x m hm; simp [t4] at hm
  · simp [F_outputs_ne, t4]
  · unfold F_outputs_nodup; decide
  · unfold F_window_dvd; decide
  · unfold F_window_le; decide
  · intro h; obtain ⟨m, hm, _⟩ := h 0 (by simp [t4]); simp [t4] at hm
  · exact realOK_of_steps _ (fun m hm => by simp [t4] at hm) (fun m hm => by simp [t4] at hm)
  · refine ⟨by decide, by decide, by decide⟩
  · refine ⟨by simp [Trace.window, t4], fun h _ => by simp [codes2]⟩
  · intro h
    have := h.1 0 (by simp [t4])
    have hv : t4.outVal (windowBits t4 codes2 (fun _ => 1) 0) 0 = 1 := by
      have l1 : codes2.lookup 1 = some [true] := rfl
      simp [Trace.outVal, Trace.values, Trace.env, windowBits, t4, l1]
    rw [hv, show 1 + 0 = 1 from rfl, t4_final] at this
    norm_num at this

/-! ## CE-6 -/

/-- CE-6: the program `[0 >= 0]`, which is 1. -/
def t6 : Trace where
  inputs := []
  mlps := [stepConst 0 0]
  outputs := [0]

/-- Two tokens of one code. -/
def codes6 : List (ℕ × List Bool) := [(0, [true]), (1, [true])]

lemma t6_values (x : ℕ → Bool) : t6.values (t6.env x) 0 = 1 := by
  simp [Trace.values, t6, stepConst, MLP.apply, MLP.out, MLP.unit, MLP.pre, heaviside]

theorem t6_isProgram : t6.IsProgram where
  nodup := by decide
  outs_ne := by simp [t6, stepConst]
  topo := by intro i hi x hx; simp [t6] at hi; subst hi; simp [t6, stepConst] at hx
  bits := stepConst_bits 0 _ t6 rfl rfl
  step_margin := by
    intro x m hm _ j hj; simp [t6] at hm; subst hm; left; simp [MLP.pre, stepConst]
  gate_margin := by intro x m hm hact; simp [t6] at hm; subst hm; simp [stepConst] at hact
  outputs_ne := by simp [t6]
  outputs_nodup := by decide
  outputs_made := by
    intro y hy; simp [t6] at hy; subst hy; exact ⟨stepConst 0 0, by simp [t6], by simp [stepConst]⟩
  window_dvd := by decide
  window_le := by decide

theorem t6_realOK : t6.RealOK := by
  refine realOK_of_steps _ (fun m hm => by simp [t6] at hm; subst hm; rfl) (fun m hm => ?_)
  simp [t6] at hm; subst hm
  refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, stepConst] at hj ⊢
  norm_num

/-- The compiled form of a step gate of no inputs. -/
lemma compiled_stepConst (y : Sig) (b : ℚ) :
    let c := (stepConst y b).compiled
    c.n = 2 ∧ c.ins = [] ∧ c.gb 0 = 32 * (4 * b + 2 + 1 / 2) ∧ c.gb 1 = 32 * (4 * b + 2 - 1 / 2) ∧
      (∀ j, c.vb j = 1 / 32) ∧ (∀ i, c.wo i 0 = 2) ∧ (∀ i, c.wo i 1 = -2) := by
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, stepConst, Gated.sharpness]

theorem t6_bf16OK : t6.BF16OK where
  weights := by
    intro m hm j hj
    simp [t6] at hm; subst hm
    obtain ⟨hn, hins, hgb0, hgb1, hvb, hwo0, hwo1⟩ := compiled_stepConst 0 0
    rw [hn] at hj
    refine ⟨fun s => ?_, fun s => ?_, ?_, ?_, ?_⟩
    · simp [Gated.gOn, hins]; ce_bf16
    · simp [Gated.vOn, hins]; ce_bf16
    · interval_cases j
      · rw [hgb0]; ce_bf16
      · rw [hgb1]; ce_bf16
    · rw [hvb]; ce_bf16
    · intro i _
      interval_cases j
      · rw [hwo0]; ce_bf16
      · rw [hwo1]; ce_bf16
  sums := by
    intro x m hm j hj
    simp [t6] at hm; subst hm
    obtain ⟨hn, hins, hgb0, hgb1, hvb, _⟩ := compiled_stepConst 0 0
    rw [hn] at hj
    simp only [Gated.gateAt, Gated.valAt, hins, List.length_nil, sum_range_zero, zero_add, hvb]
    right
    interval_cases j
    · rw [hgb0]; refine ⟨by ce_bf16, by ce_bf16, ?_⟩; norm_num [relu]; ce_bf16
    · rw [hgb1]; refine ⟨by ce_bf16, by ce_bf16, ?_⟩; norm_num [relu]; ce_bf16

lemma sqrt3_le : Real.sqrt 3 ≤ 2 := by
  rw [Real.sqrt_le_left (by norm_num)]; norm_num

theorem widthOK_of_3 (t : Trace) (h : (tokenGrid t).width = 3) : WidthOK t := by
  unfold WidthOK maxBF16; rw [h]; push_cast; nlinarith [sqrt3_le]

lemma embed6 : (transformer t6 codes6).embed 1 = (transformer t6 codes6).embed 0 := by
  have l0 : codes6.lookup 0 = some [true] := rfl
  have l1 : codes6.lookup 1 = some [true] := rfl
  funext f
  simp only [transformer, l0, l1]

/-- Two tokens of one code have one logit, in real arithmetic and in bf16. -/
theorem logit6 (toks : ℕ → ℕ) (p : ℕ) :
    (transformer t6 codes6).logit toks p 1 = (transformer t6 codes6).logit toks p 0 ∧
    ∀ R : Rounding, BF16.logit R (transformer t6 codes6) toks p 1 =
      BF16.logit R (transformer t6 codes6) toks p 0 := by
  refine ⟨?_, fun R => ?_⟩
  · unfold Transformer.logit; rw [embed6]
  · unfold BF16.logit; rw [embed6]

lemma codeIs6 (toks : ℕ → ℕ) (p : ℕ) : CodeIs t6 codes6 0 (valAt t6 codes6 toks p) := by
  refine ⟨[true], rfl, fun i hi => ?_⟩
  simp [t6] at hi; subst hi
  show (if [true].getD 0 false then (1 : ℝ) else 0) = valAt t6 codes6 toks p (.sig 0)
  simp only [valAt, t6_values]; simp

theorem ce6 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram ∧ t.RealOK ∧ t.BF16OK ∧ WidthOK t ∧
    (codes.map Prod.fst).Nodup ∧ (∀ c ∈ codes, c.2.length = t.outputs.length) ∧
    ¬ (∀ a ∈ codes, ∀ b ∈ codes, a.2 = b.2 → a.1 = b.1) ∧ PosOK t codes toks p ∧
    ¬ RealConcl t codes toks p ∧ ∀ R : Rounding, ¬ BF16Concl R t codes toks p := by
  have hW : (tokenGrid t6).width = 3 := by decide
  refine ⟨t6, codes6, fun _ => 0, 0, t6_isProgram, t6_realOK, t6_bf16OK, widthOK_of_3 t6 hW,
    by decide, by decide, by decide, ⟨by simp [Trace.window, t6], fun h _ => by simp [codes6]⟩,
    ?_, fun R => ?_⟩
  · intro h
    have := h.2 0 (codeIs6 _ _) 1 (by decide) (by norm_num)
    rw [(logit6 _ _).1] at this
    exact lt_irrefl _ this
  · intro h
    have hsmall : 2 * Real.sqrt (tokenGrid t6).width + t6.outputs.length ≤ 250 := by
      rw [hW]; push_cast; simp [t6]; nlinarith [sqrt3_le]
    have := h.2 hsmall 0 (codeIs6 _ _) 1 (by decide) (by norm_num)
    rw [(logit6 _ _).2 R] at this
    exact lt_irrefl _ this

/-! ## The forward pass in bf16, on concrete networks -/

lemma gelu_32 : 31.98 ≤ gelu 32 ∧ gelu 32 ≤ 32 := by
  refine ⟨?_, gelu_le_self (by norm_num)⟩
  have h := gelu_ge (z := 32) (by norm_num)
  have hy : 1000 ≤ geluArg 32 := by
    unfold geluArg; nlinarith [sqrt_two_div_pi_gt]
  have : (32 : ℝ) / (1 + 2 * geluArg 32) ≤ 0.02 := by
    rw [div_le_iff₀ (by linarith)]; linarith
  linarith

lemma small_le_max {x : ℝ} (h : |x| ≤ 10 ^ 6) : |x| ≤ maxBF16 := by
  linarith [maxBF16_big]

lemma bf16_32 : IsBF16 (32 : ℝ) := by ce_bf16
lemma bf16_one : IsBF16 (1 : ℝ) := by ce_bf16
lemma bf16_two : IsBF16 (2 : ℝ) := by ce_bf16
lemma bf16_neg_two : IsBF16 (-2 : ℝ) := by ce_bf16

/-- What a layer writes into feature 1 when its only units there are `ONE` and an erasure, each
`-fl(fl(gelu 32) / 32)`, over a stream holding 1: at most 0, for every rounding. -/
theorem fl_bad (R : Rounding) :
    R.fl (1 + R.fl (-2 * R.fl (R.fl (gelu 32) * (1 / 32)))) ≤ 0 := by
  obtain ⟨g1, g2⟩ := gelu_32
  set g := gelu 32
  have hA := fl_near R (x := g) (small_le_max (by rw [abs_le]; constructor <;> linarith)) bf16_32
  set A := R.fl g
  have hA' := abs_le.mp (hA.trans (by rw [abs_le]; constructor <;> linarith : |(32 : ℝ) - g| ≤ 0.02))
  have hB := fl_near R (x := A * (1 / 32)) (small_le_max (by rw [abs_le]; constructor <;> linarith))
    bf16_one
  set B := R.fl (A * (1 / 32))
  have hB' := abs_le.mp (hB.trans
    (by rw [abs_le]; constructor <;> linarith : |(1 : ℝ) - A * (1 / 32)| ≤ 0.002))
  have hC := fl_near R (x := -2 * B) (small_le_max (by rw [abs_le]; constructor <;> linarith))
    bf16_neg_two
  set C := R.fl (-2 * B)
  have hC' := abs_le.mp (hC.trans
    (by rw [abs_le]; constructor <;> linarith : |(-2 : ℝ) - -2 * B| ≤ 0.01))
  exact fl_nonpos R (small_le_max (by rw [abs_le]; constructor <;> linarith)) (by linarith)

/-- `w . z` in bf16, for a row given as a list: every weight rounded. -/
def bdotL (R : Rounding) (l : List ℚ) (z : ℕ → ℝ) : ℝ :=
  ∑ g ∈ range l.length, R.fl ((l.getD g 0 : ℚ) : ℝ) * z g

/-- A layer in bf16, given by the data of its units, at feature `f`. -/
def bevalData (R : Rounding) (D : List (List ℚ × List ℚ × List ℚ)) (z : ℕ → ℝ) (f : ℕ) : ℝ :=
  R.fl ((D.map fun d => R.fl ((d.2.2.getD f 0 : ℚ) : ℝ) *
    R.fl (R.fl (gelu (R.fl (bdotL R d.1 z))) * R.fl (bdotL R d.2.1 z))).sum)

lemma bdot_eq (R : Rounding) (W : ℕ) (w : ℕ → ℚ) (z : ℕ → ℝ) :
    ∑ g ∈ range W, R.fl (w g : ℝ) * z g = bdotL R ((List.range W).map w) z := by
  unfold bdotL
  rw [List.length_map, List.length_range]
  refine sum_congr rfl (fun g hg => ?_)
  rw [getD_map_range W w g (mem_range.mp hg)]

lemma blayerApply_eq (R : Rounding) (W : ℕ) (us : List Hidden) (z : ℕ → ℝ) (f : ℕ) (hf : f < W) :
    BF16.layerApply R W us z f = bevalData R (us.map (unitData W)) z f := by
  unfold BF16.layerApply bevalData
  rw [List.map_map]
  congr 2
  refine List.map_congr_left (fun u _ => ?_)
  simp only [Function.comp, unitData, getD_map_range W u.out f hf, bdot_eq]

lemma bfinal_two (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ)
    (h : (tokenGrid t).rows.length = 2) :
    BF16.finalAt R (transformer t codes) toks p =
      BF16.netStep R (Wd t) (R.fl (2 * Real.sqrt (Wd t))) (GR t 0)
        (layerUnits (Wd t) t.mlps (GR t 0) (GR t 1)) (BF16.afterAttn R (transformer t codes) toks p) := by
  unfold BF16.finalAt BF16.final BF16.one
  simp only [transformer, emit]
  rw [rows_two h]
  simp

lemma bafterAttn_noheads (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ)
    (p : ℕ) (h : t.inputs.length / t.outputs.length = 0) :
    BF16.afterAttn R (transformer t codes) toks p = BF16.embed R (transformer t codes) (toks p) := by
  unfold BF16.afterAttn
  have : (transformer t codes).heads = 0 := h
  rw [this]
  rfl

lemma one_b_pos (R : Rounding) : 0 < R.fl (2 * Real.sqrt 3) := by
  have h3 : Real.sqrt 3 ≤ 2 := by rw [Real.sqrt_le_left (by norm_num)]; norm_num
  have h3' : 1.5 ≤ Real.sqrt 3 := by rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num
  have := fl_near R (x := 2 * Real.sqrt 3)
    (small_le_max (by rw [abs_le]; constructor <;> linarith)) bf16_two
  have := (abs_le.mp (this.trans (by rw [abs_le]; constructor <;> linarith :
    |(2 : ℝ) - 2 * Real.sqrt 3| ≤ 2))).1
  linarith [abs_le.mp (show |R.fl (2 * Real.sqrt 3) - 2 * Real.sqrt 3| ≤ 2 from
    (fl_near R (small_le_max (by rw [abs_le]; constructor <;> linarith)) bf16_two).trans
      (by rw [abs_le]; constructor <;> linarith))]

/-! ## CE-9 -/

lemma bf16_near_1024 {r : ℝ} (hr : IsBF16 r) (h1 : 1024 ≤ r) (h2 : r ≤ 1030) : r = 1024 := by
  obtain ⟨n, rfl⟩ := bf16_grid hr 3 (by rw [abs_of_pos (by linarith)]; norm_num; linarith)
  norm_num at h1 h2 ⊢
  have a : (127 : ℝ) < n := by linarith
  have b : (n : ℝ) < 129 := by linarith
  have a' : (127 : ℤ) < n := by exact_mod_cast a
  have b' : n < (129 : ℤ) := by exact_mod_cast b
  have : n = 128 := by omega
  subst this; norm_num

lemma bf16_near_m1024 {r : ℝ} (hr : IsBF16 r) (h1 : -1030 ≤ r) (h2 : r ≤ -1024) : r = -1024 := by
  obtain ⟨n, rfl⟩ := bf16_grid hr 3 (by rw [abs_of_neg (by linarith)]; norm_num; linarith)
  norm_num at h1 h2 ⊢
  have a : (-129 : ℝ) < n := by linarith
  have b : (n : ℝ) < -127 := by linarith
  have a' : (-129 : ℤ) < n := by exact_mod_cast a
  have b' : n < (-127 : ℤ) := by exact_mod_cast b
  have : n = -128 := by omega
  subst this; norm_num

lemma bf16_1024 : IsBF16 (1024 : ℝ) := ⟨128, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩
lemma bf16_m1024 : IsBF16 (-1024 : ℝ) := ⟨-128, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- **CE-9's rounding**: every rounding takes the weights 1027 and -1025 to 1024 and -1024. -/
theorem fl_1027 (R : Rounding) : R.fl 1027 = 1024 ∧ R.fl (-1025) = -1024 ∧
    R.fl (-1027) = -1024 ∧ R.fl 1025 = 1024 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · refine fl_eq_of_unique R (small_le_max (by norm_num)) bf16_1024 (fun r hr h => ?_)
    norm_num at h; have := abs_le.mp h; exact bf16_near_1024 hr (by linarith) (by linarith)
  · refine fl_eq_of_unique R (small_le_max (by norm_num)) bf16_m1024 (fun r hr h => ?_)
    norm_num at h; have := abs_le.mp h; exact bf16_near_m1024 hr (by linarith) (by linarith)
  · refine fl_eq_of_unique R (small_le_max (by norm_num)) bf16_m1024 (fun r hr h => ?_)
    norm_num at h; have := abs_le.mp h; exact bf16_near_m1024 hr (by linarith) (by linarith)
  · refine fl_eq_of_unique R (small_le_max (by norm_num)) bf16_1024 (fun r hr h => ?_)
    norm_num at h; have := abs_le.mp h; exact bf16_near_1024 hr (by linarith) (by linarith)

theorem not_bf16_1027 : ¬ IsBF16 (1027 : ℝ) := by
  intro h
  obtain ⟨n, hn⟩ := bf16_grid h 3 (by norm_num)
  norm_num at hn
  have a : (128 : ℝ) < n := by linarith
  have b : (n : ℝ) < 129 := by linarith
  have a' : (128 : ℤ) < n := by exact_mod_cast a
  have b' : n < (129 : ℤ) := by exact_mod_cast b
  omega

/-- CE-9: two step units, both on, with weights `1027/2` and `-1025/2`: the output is 1. -/
def m9 : MLP where
  ins := []
  outs := [0]
  act := .step
  hidden := 2
  wi := fun _ _ => 0
  b := fun _ => 0
  wo := fun _ j => if j = 0 then 1027 / 2 else -1025 / 2

def t9 : Trace where
  inputs := []
  mlps := [m9]
  outputs := [0]

lemma t9_rows : (tokenGrid t9).rows.length = 2 := by decide
lemma t9_width : Wd t9 = 3 := by decide
lemma t9_count : count 3 (GR t9 0) = 1 := by decide
lemma t9_units : (layerUnits 3 t9.mlps (GR t9 0) (GR t9 1)).map (unitData 3) =
    [([32, 0, 0], [1 / 32, 0, 0], [0, -1, 0]), ([80, 0, 0], [1 / 32, 0, 0], [0, 1027, 0]),
     ([80, 0, 0], [1 / 32, 0, 0], [0, -1025, 0]), ([48, 0, 0], [1 / 32, 0, 0], [0, -1027, 0]),
     ([48, 0, 0], [1 / 32, 0, 0], [0, 1025, 0]), ([32, 0, 0], [0, 1 / 32, 0], [0, -1, 0])] := by
  decide +kernel

/-- The bf16 stream of a one-bit program of no inputs, of width 3, normed: `[1, 1, 0]` for the
token 1 of `codes2`. -/
lemma bnorm_noinput {p : ℕ} (R : Rounding) (t : Trace) (hW : Wd t = 3) (hb : t.outputs.length = 1)
    (hin : t.inputs = []) (hc : count 3 (GR t 0) = 1) (g : ℕ) :
    BF16.norm R 3 (R.fl (2 * Real.sqrt 3)) (GR t 0)
      (BF16.afterAttn R (transformer t codes2) (fun _ => 1) p) g =
      if g = 0 then 1 else if g = 1 then 1 else 0 := by
  rw [bafterAttn_noheads _ _ _ _ _ (by simp [hin])]
  have hx : ∀ f, BF16.embed R (transformer t codes2) 1 f =
      if f = 0 then R.fl (2 * Real.sqrt 3) else if f = 1 then 1 else 0 := by
    intro f
    simp only [BF16.embed, embed_codes2 t hW hb 1 (by norm_num) f]
    split_ifs <;> simp [fl_bf16 R bf16_one, fl_bf16 R bf16_zero]
  have hsum : ∑ g ∈ range 3, BF16.embed R (transformer t codes2) 1 g ^ 2 =
      (R.fl (2 * Real.sqrt 3)) ^ 2 + count 3 (GR t 0) := by
    rw [hc]; simp [sum_range_succ, hx]
  unfold BF16.norm
  rw [rmsnorm_of_sum _ _ _ _ hsum]
  have hob := (one_b_pos R).ne'
  rcases Nat.lt_or_ge g 2 with h | h
  · interval_cases g <;> simp [hx, hob, fl_bf16 R bf16_one]
  · have h0 : g ≠ 0 := by omega
    have h1 : g ≠ 1 := by omega
    simp [hx, h0, h1, fl_bf16 R bf16_zero]

/-- The numerals the networks here hold are bf16 numbers, which every rounding keeps. -/
lemma fl_nums (R : Rounding) : R.fl 0 = 0 ∧ R.fl 1 = 1 ∧ R.fl (-1) = -1 ∧ R.fl 32 = 32 ∧
    R.fl 32⁻¹ = 32⁻¹ ∧ R.fl (1 / 32) = 1 / 32 ∧ R.fl 2 = 2 ∧ R.fl (-2) = -2 ∧ R.fl 80 = 80 ∧
    R.fl 48 = 48 ∧ R.fl 8192 = 8192 := by
  refine ⟨fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_,
    fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_, fl_bf16 R ?_⟩ <;> ce_bf16

/-- In bf16, the units of 1027 and -1025 cancel: feature 1 ends at most 0. -/
theorem t9_final (R : Rounding) (p : ℕ) :
    BF16.finalAt R (transformer t9 codes2) (fun _ => 1) p 1 ≤ 0 := by
  rw [bfinal_two R t9 codes2 _ p t9_rows]
  unfold BF16.netStep
  rw [t9_width]
  simp only [Nat.cast_ofNat]
  rw [blayerApply_eq _ _ _ _ _ (by norm_num), t9_units]
  have hz := bnorm_noinput (p := p) R t9 t9_width rfl rfl t9_count
  have hx1 : BF16.afterAttn R (transformer t9 codes2) (fun _ => 1) p 1 = 1 := by
    rw [bafterAttn_noheads _ _ _ _ _ (by decide)]
    simp only [BF16.embed, embed_codes2 t9 t9_width rfl 1 (by norm_num) 1]
    simp [fl_bf16 R bf16_one]
  obtain ⟨f1, f2, f3, f4⟩ := fl_1027 R
  simp only [bevalData, bdotL, List.map, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil, sum_range_succ, sum_range_zero, hz, hx1]
  obtain ⟨n0, n1, nm1, n32, n32i, n32d, n2, nm2, n80, n48, n8192⟩ := fl_nums R
  norm_num [n0, n1, nm1, n32, n32i, n32d, n80, n48, f1, f2, f3, f4]
  have := fl_bad R
  rwa [show ∀ B : ℝ, -B + -B = -2 * B from fun B => by ring]

lemma compiled_m9 :
    let c := m9.compiled
    c.n = 4 ∧ c.ins = [] ∧ (∀ j, c.gb j = if j < 2 then 80 else 48) ∧ (∀ j, c.vb j = 1 / 32) ∧
      c.wo 0 0 = 1027 := by
  refine ⟨rfl, rfl, fun j => ?_, fun j => ?_, ?_⟩ <;>
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, m9, Gated.sharpness]
  · split_ifs <;> norm_num
  · norm_num

lemma t9_values (x : ℕ → Bool) : t9.values (t9.env x) 0 = 1 := by
  simp [Trace.values, t9, m9, MLP.apply, MLP.out, MLP.unit, MLP.pre, heaviside, sum_range_succ]
  norm_num

theorem t9_isProgram : t9.IsProgram where
  nodup := by decide
  outs_ne := by simp [t9, m9]
  topo := by intro i hi x hx; simp [t9] at hi; subst hi; simp [t9, m9] at hx
  bits := by
    intro x m hm y hy
    simp [t9] at hm; subst hm; simp [m9] at hy; subst hy
    rw [t9_values]; right; rfl
  step_margin := by
    intro x m hm _ j hj; simp [t9] at hm; subst hm; left; simp [MLP.pre, m9]
  gate_margin := by intro x m hm hact; simp [t9] at hm; subst hm; simp [m9] at hact
  outputs_ne := by simp [t9]
  outputs_nodup := by decide
  outputs_made := by
    intro y hy; simp [t9] at hy; subst hy; exact ⟨m9, by simp [t9], by simp [m9]⟩
  window_dvd := by decide
  window_le := by decide

theorem ce9 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram ∧ ¬ F_weights t ∧ F_sums t ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
      WidthOK t ∧ ∀ R : Rounding, ¬ BF16Concl R t codes toks p := by
  obtain ⟨hn, hins, hgb, hvb, hwo⟩ := compiled_m9
  refine ⟨t9, codes2, fun _ => 1, 0, t9_isProgram, ?_, ?_, ⟨by decide, by decide, by decide⟩,
    ⟨by simp [Trace.window, t9], fun h _ => by simp [codes2]⟩, widthOK_of_3 t9 (by decide),
    fun R h => ?_⟩
  · intro h
    have := (h m9 (by simp [t9]) 0 (by rw [hn]; norm_num)).2.2.2.2 0 (by simp [m9])
    rw [hwo] at this
    exact not_bf16_1027 (by exact_mod_cast this)
  · intro x m hm j hj
    simp [t9] at hm; subst hm
    rw [hn] at hj
    simp only [Gated.gateAt, Gated.valAt, hins, List.length_nil, sum_range_zero, zero_add, hvb,
      hgb]
    right
    interval_cases j <;> norm_num [relu] <;> refine ⟨by ce_bf16, by ce_bf16, by ce_bf16⟩
  · have := h.1 0 (by simp [t9])
    have hv : t9.outVal (windowBits t9 codes2 (fun _ => 1) 0) 0 = 1 := t9_values _
    rw [hv, show 1 + 0 = 1 from rfl] at this
    have := t9_final R 0
    linarith

/-! ## CE-2 in bf16 -/

/-- A rounding takes `x` to at most any bf16 number above `x`. -/
lemma fl_le_of_le (R : Rounding) {x r : ℝ} (hx : |x| ≤ maxBF16) (hr : IsBF16 r) (h : x ≤ r) :
    R.fl x ≤ r := by
  have := fl_near R hx hr
  rw [abs_of_nonneg (by linarith : 0 ≤ r - x)] at this
  linarith [(abs_le.mp this).2]

/-- A rounding takes `x` to at least any bf16 number below `x`. -/
lemma le_fl_of_le (R : Rounding) {x r : ℝ} (hx : |x| ≤ maxBF16) (hr : IsBF16 r) (h : r ≤ x) :
    r ≤ R.fl x := by
  have := fl_near R hx hr
  rw [abs_of_nonpos (by linarith : r - x ≤ 0)] at this
  linarith [(abs_le.mp this).1]

/-- **CE-2's unit, in bf16, for every rounding**: `fl(gelu 1) ≤ 221/256`, not `relu 1 = 1`. -/
theorem fl_gelu_one (R : Rounding) : R.fl (gelu 1) ≤ 221 / 256 ∧ 0 ≤ R.fl (gelu 1) := by
  have h0 := gelu_nonneg (z := 1) (by norm_num)
  have h1 := gelu_one_lt
  have hx : |gelu 1| ≤ maxBF16 := small_le_max (by rw [abs_le]; constructor <;> linarith)
  exact ⟨fl_le_of_le R hx ⟨221, -8, by norm_num, by norm_num, by norm_num, by norm_num⟩
    (by linarith), le_fl_of_le R hx bf16_zero h0⟩

lemma compiled_gluConst :
    let c := gluConst.compiled
    c.n = 1 ∧ c.ins = [] ∧ c.gb 0 = 1 ∧ c.vb 0 = 1 ∧ (∀ i, c.wo i 0 = 2) := by
  refine ⟨rfl, rfl, ?_, ?_, fun i => ?_⟩ <;>
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gluConst, Gated.sharpness]

theorem t2_bf16OK : t2.BF16OK where
  weights := by
    intro m hm j hj
    simp [t2] at hm; subst hm
    obtain ⟨hn, hins, hgb, hvb, hwo⟩ := compiled_gluConst
    rw [hn] at hj
    have hj0 : j = 0 := by omega
    subst hj0
    refine ⟨fun s => ?_, fun s => ?_, ?_, ?_, fun i _ => ?_⟩
    · simp [Gated.gOn, hins]; exact bf16_zero
    · simp [Gated.vOn, hins]; exact bf16_zero
    · rw [hgb]; exact_mod_cast bf16_one
    · rw [hvb]; exact_mod_cast bf16_one
    · rw [hwo]; exact_mod_cast bf16_two
  sums := by
    intro x m hm j hj
    simp [t2] at hm; subst hm
    obtain ⟨hn, hins, hgb, hvb, _⟩ := compiled_gluConst
    rw [hn] at hj
    have hj0 : j = 0 := by omega
    subst hj0
    simp only [Gated.gateAt, Gated.valAt, hins, List.length_nil, sum_range_zero, zero_add, hgb,
      hvb]
    right
    norm_num [relu]
    exact bf16_one

/-- In bf16, for every rounding, the GLU's unit gives `2 fl(gelu 1) ≤ 2 * 221/256` where it should
give 2, and the output feature is at most `47/64`, not 1. -/
theorem t2_bfinal (R : Rounding) (p : ℕ) :
    BF16.finalAt R (transformer t2 codes2) (fun _ => 1) p 1 ≤ 47 / 64 := by
  rw [bfinal_two R t2 codes2 _ p t2_rows]
  unfold BF16.netStep
  rw [t2_width]
  simp only [Nat.cast_ofNat]
  rw [blayerApply_eq _ _ _ _ _ (by norm_num), t2_units]
  have hz := bnorm_noinput (p := p) R t2 t2_width rfl rfl t2_count
  have hx1 : BF16.afterAttn R (transformer t2 codes2) (fun _ => 1) p 1 = 1 := by
    rw [bafterAttn_noheads _ _ _ _ _ (by decide)]
    simp only [BF16.embed, embed_codes2 t2 t2_width rfl 1 (by norm_num) 1]
    simp [fl_bf16 R bf16_one]
  simp only [bevalData, bdotL, List.map, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil, sum_range_succ, sum_range_zero, hz, hx1]
  obtain ⟨n0, n1, nm1, n32, n32i, n32d, n2, nm2, n80, n48, n8192⟩ := fl_nums R
  norm_num [n0, n1, nm1, n32, n32i, n32d, n2]
  have sm : ∀ x : ℝ, -10 ≤ x → x ≤ 10 → |x| ≤ maxBF16 :=
    fun x h1 h2 => small_le_max (by rw [abs_le]; constructor <;> linarith)
  obtain ⟨g1, g2⟩ := gelu_32
  have hA1 : (255 / 8 : ℝ) ≤ R.fl (gelu 32) :=
    le_fl_of_le R (small_le_max (by rw [abs_le]; constructor <;> linarith))
      ⟨255, -3, by norm_num, by norm_num, by norm_num, by norm_num⟩ (by linarith)
  have hA2 : R.fl (gelu 32) ≤ 32 :=
    fl_le_of_le R (small_le_max (by rw [abs_le]; constructor <;> linarith)) bf16_32 g2
  set A := R.fl (gelu 32)
  have hB1 : (255 / 256 : ℝ) ≤ R.fl (A * (1 / 32)) :=
    le_fl_of_le R (sm _ (by linarith) (by linarith))
      ⟨255, -8, by norm_num, by norm_num, by norm_num, by norm_num⟩ (by linarith)
  have hB2 : R.fl (A * (1 / 32)) ≤ 1 :=
    fl_le_of_le R (sm _ (by linarith) (by linarith)) bf16_one (by linarith)
  set B := R.fl (A * (1 / 32))
  obtain ⟨hg1, hg2⟩ := fl_gelu_one R
  have hg := gelu_one_lt
  have hg0 := gelu_nonneg (z := 1) (by norm_num)
  rw [fl_bf16 R (R.fl_mem _ (sm _ (by linarith) (by linarith)))]
  set g := R.fl (gelu 1)
  have hF2 : R.fl (-B + (2 * g + -B)) ≤ -17 / 64 :=
    fl_le_of_le R (sm _ (by linarith) (by linarith))
      ⟨-17, -6, by norm_num, by norm_num, by norm_num, by norm_num⟩ (by linarith)
  have hF1 : -2 ≤ R.fl (-B + (2 * g + -B)) :=
    le_fl_of_le R (sm _ (by linarith) (by linarith)) bf16_neg_two (by linarith)
  exact fl_le_of_le R (sm _ (by linarith) (by linarith))
    ⟨47, -6, by norm_num, by norm_num, by norm_num, by norm_num⟩ (by linarith)

/-- **CE-2, in bf16, end to end, every rounding**: `t2`, the GLU `relu(1/32) * 32`, violates only
the (weak) gate margin of `IsProgram`, satisfies every other hypothesis of `MainBF16` (its weights,
gates, values and products are 1, 1, 1, 2: bf16 numbers), and its output feature is at most
`47/64`, not 1, as the compiled gate is 1, where `fl(gelu 1) ≤ 221/256`. -/
theorem ce2_bf16 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    (F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_step_margin t ∧
      F_outputs_ne t ∧ F_outputs_nodup t ∧ F_outputs_made t ∧ F_window_dvd t ∧ F_window_le t) ∧
    ¬ F_gate_margin t ∧ t.BF16OK ∧ CodesOK t codes ∧ PosOK t codes toks p ∧ WidthOK t ∧
    ∀ R : Rounding, ¬ BF16Concl R t codes toks p := by
  refine ⟨t2, codes2, fun _ => 1, 0, t2_fields, t2_not_gate_margin, t2_bf16OK,
    ⟨by decide, by decide, by decide⟩, ⟨by simp [Trace.window, t2], fun h _ => by simp [codes2]⟩,
    widthOK_of_3 t2 (by decide), fun R h => ?_⟩
  have := h.1 0 (by simp [t2])
  have hv : t2.outVal (windowBits t2 codes2 (fun _ => 1) 0) 0 = 1 := t2_values _
  rw [hv, show 1 + 0 = 1 from rfl] at this
  have := t2_bfinal R 0
  linarith

/-! ## CE-10 -/

/-- CE-10: `[128 x - 64 >= 0]` on the bit of the last token: its bf16 weights are bf16 numbers,
but its gate at `x = 1`, `128 * 64 + 80 = 8272`, is not. -/
def m10 : MLP where
  ins := [0]
  outs := [1]
  act := .step
  hidden := 1
  wi := fun _ _ => 128
  b := fun _ => -64
  wo := fun _ _ => 1

def t10 : Trace where
  inputs := [0]
  mlps := [m10]
  outputs := [1]

lemma t10_rows : (tokenGrid t10).rows.length = 2 := by decide
lemma t10_width : Wd t10 = 3 := by decide
lemma t10_count : count 3 (GR t10 0) = 2 := by decide
lemma t10_ecount : count 3 (transformer t10 codes2).embedded = 1 := by decide +kernel
lemma t10_units : (layerUnits 3 t10.mlps (GR t10 0) (GR t10 1)).map (unitData 3) =
    [([32, 0, 0], [1 / 32, 0, 0], [0, -1, 0]), ([80, 0, 8192], [1 / 32, 0, 0], [0, 2, 0]),
     ([48, 0, 8192], [1 / 32, 0, 0], [0, -2, 0]), ([32, 0, 0], [0, 1 / 32, 0], [0, -1, 0])] := by
  decide +kernel

lemma t10_o (f : ℕ) : (transformer t10 codes2).o 0 f 0 = if f = 2 then 1 else 0 := by
  have h : featIn (tokenGrid t10).width ((tokenGrid t10).rows.headD fun _ => none)
      (.sig (t10.inputs.getD ((t10.inputs.length / t10.outputs.length - 1 - 0) *
        t10.outputs.length + 0) 0)) = 2 := by decide
  simp only [transformer]
  rw [h]
  have : 0 < t10.inputs.length / t10.outputs.length ∧ 0 < t10.outputs.length := by decide
  simp [this]

lemma fl_fl (R : Rounding) (x : ℝ) (hx : |x| ≤ maxBF16) : R.fl (R.fl x) = R.fl x :=
  fl_bf16 R (R.fl_mem x hx)

lemma sqrt3_abs : |2 * Real.sqrt 3| ≤ maxBF16 :=
  small_le_max (by rw [abs_of_pos (by positivity)]; nlinarith [sqrt3_le, Real.sqrt_nonneg 3])

/-- In bf16, at position 0, the stream after the attention: `ONE`, the code bit 1 of the token,
and the input, the same bit, copied by the one head. -/
lemma t10_afterAttn (R : Rounding) (f : ℕ) :
    BF16.afterAttn R (transformer t10 codes2) (fun _ => 1) 0 f =
      if f = 0 then R.fl (2 * Real.sqrt 3) else if f = 1 then 1 else if f = 2 then 1 else 0 := by
  obtain ⟨n0, n1, nm1, n32, n32i, n32d, n2, nm2, n80, n48, n8192⟩ := fl_nums R
  have hheads : (transformer t10 codes2).heads = 1 := rfl
  have hbits : (transformer t10 codes2).bits = 1 := rfl
  have hW : (transformer t10 codes2).net.W = 3 := by
    show (tokenGrid t10).width = 3; decide
  have hone : BF16.one R (transformer t10 codes2).net = R.fl (2 * Real.sqrt 3) := by
    show R.fl (2 * Real.sqrt ((tokenGrid t10).width : ℝ)) = _
    rw [show (tokenGrid t10).width = 3 by decide]; norm_num
  have hemb : ∀ g, BF16.embed R (transformer t10 codes2) 1 g =
      if g = 0 then R.fl (2 * Real.sqrt 3) else if g = 1 then 1 else 0 := by
    intro g
    simp only [BF16.embed, embed_codes2 t10 t10_width rfl 1 (by norm_num) g]
    split_ifs <;> simp [n0, n1]
  have hattn : BF16.attn R (transformer t10 codes2) (fun _ => 1) 0 0 0 = 1 := by
    unfold BF16.attn
    rw [visible_zero, sum_singleton, div_self (Real.exp_pos _).ne', n1]
  have hsum : ∑ g ∈ range 3, BF16.embed R (transformer t10 codes2) 1 g ^ 2 =
      (BF16.one R (transformer t10 codes2).net) ^ 2 + count 3 (transformer t10 codes2).embedded := by
    rw [t10_ecount, hone]; simp [sum_range_succ, hemb]
  have hob := (one_b_pos R).ne'
  have hv : ∀ f, (transformer t10 codes2).v 0 f = if f = 1 then 1 else 0 := by
    intro f; simp [transformer, t10]
  have hval : BF16.value R (transformer t10 codes2) (fun _ => 1) 0 0 = 1 := by
    unfold BF16.value BF16.matvec BF16.zvec
    rw [hW, rmsnorm_of_sum _ _ _ _ hsum, hone]
    simp [sum_range_succ, hv, hemb, hob, n0, n1]
  have hhead : BF16.head R (transformer t10 codes2) (fun _ => 1) 0 0 0 = 1 := by
    unfold BF16.head
    rw [visible_zero, sum_singleton, hattn, hval, one_mul, n1]
  unfold BF16.afterAttn
  rw [hheads, hbits]
  simp only [List.range_one, List.foldl_cons, List.foldl_nil, sum_range_one, hhead, t10_o, hemb]
  rcases Nat.lt_or_ge f 3 with h | h
  · interval_cases f <;> simp [n0, n1, fl_fl R _ sqrt3_abs]
  · have h0 : f ≠ 0 := by omega
    have h1 : f ≠ 1 := by omega
    have h2 : f ≠ 2 := by omega
    simp [h0, h1, h2, n0]

/-- In bf16, the gates `u+` and `u-` both round to 8256 and cancel: feature 1 ends at most 0. -/
theorem t10_final (R : Rounding) :
    BF16.finalAt R (transformer t10 codes2) (fun _ => 1) 0 1 ≤ 0 := by
  obtain ⟨n0, n1, nm1, n32, n32i, n32d, n2, nm2, n80, n48, n8192⟩ := fl_nums R
  rw [bfinal_two R t10 codes2 _ 0 t10_rows]
  unfold BF16.netStep
  rw [t10_width]
  simp only [Nat.cast_ofNat]
  rw [blayerApply_eq _ _ _ _ _ (by norm_num), t10_units]
  have hsum : ∑ g ∈ range 3, BF16.afterAttn R (transformer t10 codes2) (fun _ => 1) 0 g ^ 2 =
      (R.fl (2 * Real.sqrt 3)) ^ 2 + count 3 (GR t10 0) := by
    rw [t10_count]; simp [sum_range_succ, t10_afterAttn]; ring
  have hob := (one_b_pos R).ne'
  have hz : ∀ g, BF16.norm R 3 (R.fl (2 * Real.sqrt 3)) (GR t10 0)
      (BF16.afterAttn R (transformer t10 codes2) (fun _ => 1) 0) g =
      if g = 0 then 1 else if g = 1 then 1 else if g = 2 then 1 else 0 := by
    intro g
    unfold BF16.norm
    rw [rmsnorm_of_sum _ _ _ _ hsum]
    rcases Nat.lt_or_ge g 3 with h | h
    · interval_cases g <;> simp [t10_afterAttn, hob, n1]
    · have h0 : g ≠ 0 := by omega
      have h1 : g ≠ 1 := by omega
      have h2 : g ≠ 2 := by omega
      simp [t10_afterAttn, h0, h1, h2, n0]
  simp only [bevalData, bdotL, List.map, List.sum_cons, List.sum_nil, List.length_cons,
    List.length_nil, sum_range_succ, sum_range_zero, hz, t10_afterAttn]
  norm_num [n0, n1, nm1, n32, n32i, n32d, n2, nm2, n80, n48, n8192, fl_8272, fl_8240]
  have := fl_bad R
  rwa [show ∀ B : ℝ, -B + -B = -2 * B from fun B => by ring]

lemma compiled_m10 :
    let c := m10.compiled
    c.n = 2 ∧ c.ins = [0] ∧ (∀ j k, j < 2 → c.g j k = 8192) ∧ (∀ j k, c.v j k = 0) ∧
      c.gb 0 = 80 ∧ c.gb 1 = 48 ∧ (∀ j, c.vb j = 1 / 32) ∧ (∀ i, c.wo i 0 = 2) ∧
      (∀ i, c.wo i 1 = -2) := by
  refine ⟨rfl, rfl, fun j k hj => ?_, fun j k => ?_, ?_, ?_, fun j => ?_, fun i => ?_, fun i => ?_⟩ <;>
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, m10, Gated.sharpness] <;>
    norm_num

lemma t10_values (x : ℕ → Bool) :
    t10.values (t10.env x) 1 = heaviside (128 * (if x 0 then 1 else 0) - 64) ∧
    t10.values (t10.env x) 0 = (if x 0 then 1 else 0) := by
  constructor
  · simp [Trace.values, t10, m10, MLP.apply, MLP.out, MLP.unit, MLP.pre, MLP.read, Trace.env]
    ring_nf
  · simp [Trace.values, t10, m10, MLP.apply, Trace.env]

theorem t10_isProgram : t10.IsProgram where
  nodup := by decide
  outs_ne := by simp [t10, m10]
  topo := by intro i hi x hx; simp [t10] at hi; subst hi; left; simpa [t10, m10] using hx
  bits := by
    intro x m hm y hy
    simp [t10] at hm; subst hm; simp [m10] at hy; subst hy
    rw [(t10_values x).1]; unfold heaviside IsBit; split_ifs <;> simp
  step_margin := by
    intro x m hm _ j hj; simp [t10] at hm; subst hm
    have : t10.inVals x m10 0 = if x 0 then 1 else 0 := (t10_values x).2
    simp only [MLP.pre]
    rw [show m10.ins.length = 1 from rfl, sum_range_one, this]
    simp only [m10]
    split_ifs <;> norm_num
  gate_margin := by intro x m hm hact; simp [t10] at hm; subst hm; simp [m10] at hact
  outputs_ne := by simp [t10]
  outputs_nodup := by decide
  outputs_made := by
    intro y hy; simp [t10] at hy; subst hy; exact ⟨m10, by simp [t10], by simp [m10]⟩
  window_dvd := by decide
  window_le := by decide

theorem ce10 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    t.IsProgram ∧ F_weights t ∧ ¬ F_sums t ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
      WidthOK t ∧ ∀ R : Rounding, ¬ BF16Concl R t codes toks p := by
  obtain ⟨hn, hins, hg, hv, hgb0, hgb1, hvb, hwo0, hwo1⟩ := compiled_m10
  refine ⟨t10, codes2, fun _ => 1, 0, t10_isProgram, ?_, ?_, ⟨by decide, by decide, by decide⟩,
    ⟨by simp [Trace.window, t10], fun h _ => by simp [codes2]⟩, widthOK_of_3 t10 (by decide),
    fun R h => ?_⟩
  · intro m hm j hj
    simp [t10] at hm; subst hm
    rw [hn] at hj
    refine ⟨fun s => ?_, fun s => ?_, ?_, ?_, ?_⟩
    · simp only [Gated.gOn, hins, hg j _ hj, List.length_cons, List.length_nil, zero_add,
        sum_range_one, List.getD_cons_zero]
      split_ifs <;> ce_bf16
    · simp [Gated.vOn, hv]; ce_bf16
    · interval_cases j
      · rw [hgb0]; ce_bf16
      · rw [hgb1]; ce_bf16
    · rw [hvb]; ce_bf16
    · intro i _
      interval_cases j
      · rw [hwo0]; ce_bf16
      · rw [hwo1]; ce_bf16
  · intro h
    have hu : t10.inVals (fun _ => true) m10 0 = 1 := by
      rw [show t10.inVals (fun _ => true) m10 0 = t10.values (t10.env fun _ => true) 0 from rfl,
        (t10_values _).2]; simp
    have hG : m10.compiled.gateAt (t10.inVals (fun _ => true) m10) 0 = 8272 := by
      simp only [Gated.gateAt, hins, hg 0 _ (by norm_num), List.length_cons, List.length_nil,
        zero_add, sum_range_one, hu, hgb0]
      norm_num
    rcases h (fun _ => true) m10 (by simp [t10]) 0 (by rw [hn]; norm_num) with h1 | h1
    · rw [hG] at h1; norm_num at h1
    · rw [hG] at h1; exact not_bf16_8272 h1.1
  · have := h.1 0 (by simp [t10])
    have hv : t10.outVal (windowBits t10 codes2 (fun _ => 1) 0) 0 = 1 := by
      have l1 : codes2.lookup 1 = some [true] := rfl
      show t10.values (t10.env _) 1 = 1
      rw [(t10_values _).1]
      simp [windowBits, l1, t10, heaviside]
      norm_num
    rw [hv, show 1 + 0 = 1 from rfl] at this
    have := t10_final R
    linarith

/-! ## CE-5 -/

/-- A head at `p` cannot see the token `h ≥ CONTEXT` back. -/
theorem not_visible (p h : ℕ) (hh : CONTEXT ≤ h) (hp : h ≤ p) : p - h ∉ visible p := by
  simp only [visible, mem_filter, mem_range, not_and, not_lt]
  intro _; omega

/-- With a window of `CONTEXT + 1` tokens, the input that the program reads first (input 0, bit 0
of the oldest token) is a bit of the token at `p - CONTEXT`, which no head at `p` sees. -/
theorem oldest_not_visible (t : Trace) (hw : t.window = CONTEXT + 1)
    (p : ℕ) (hp : CONTEXT ≤ p) : p + 1 + 0 / t.outputs.length - t.window ∉ visible p := by
  have : p + 1 + 0 / t.outputs.length - t.window = p - CONTEXT := by
    rw [Nat.zero_div, hw]; omega
  rw [this]
  exact not_visible p CONTEXT le_rfl hp

/-- What a head writes is a mean of the values at the positions it sees: if they all hold `c`,
it writes `c`, whatever the token it was meant to read. -/
theorem head_const (M : Transformer) (toks : ℕ → ℕ) (h p j : ℕ) (c : ℝ)
    (hc : ∀ p' ∈ visible p, M.value toks p' j = c) :
    ∑ p' ∈ visible p, M.attn toks h p p' * M.value toks p' j = c := by
  have hpos : 0 < ∑ p'' ∈ visible p, Real.exp (M.score toks h p p'') := by
    apply sum_pos (fun _ _ => Real.exp_pos _)
    exact ⟨p, mem_filter.mpr ⟨mem_range.mpr (Nat.lt_succ_self p), by simp [CONTEXT]⟩⟩
  rw [sum_congr rfl (fun p' hp' => by rw [hc p' hp'])]
  unfold Transformer.attn
  rw [← sum_mul, ← sum_div, div_self hpos.ne', one_mul]

/-- CE-5's program: `[x0 >= 1]`, the bit of the oldest of 513 tokens. -/
def m5 : MLP where
  ins := [0]
  outs := [513]
  act := .step
  hidden := 1
  wi := fun _ _ => 1
  b := fun _ => -1
  wo := fun _ _ => 1

def t5 : Trace where
  inputs := List.range 513
  mlps := [m5]
  outputs := [513]

lemma t5_inputs : t5.inputs = List.range 513 := by simp only [t5]
lemma t5_mlps : t5.mlps = [m5] := by simp only [t5]
lemma t5_outputs : t5.outputs = [513] := by simp only [t5]
lemma t5_bits : t5.outputs.length = 1 := by rw [t5_outputs]; rfl
lemma t5_window : t5.inputs.length / t5.outputs.length = 513 := by
  rw [t5_inputs, List.length_range, t5_bits]

/-- Its tokens: 1 at position 0, the oldest the program reads at 512, then 0. -/
def toks5 : ℕ → ℕ := fun p => if p = 0 then 1 else 0

lemma t5_width : Wd t5 = 515 := by decide +kernel

/-- Row 0 of the token grid. -/
abbrev row0 (t : Trace) : Row := (tokenGrid t).rows.headD fun _ => none

lemma t5_feat : featIn (Wd t5) (row0 t5) (.sig 0) = 2 := by decide +kernel

lemma t5_row2 : row0 t5 2 = some (.sig 0) := by decide +kernel

lemma t5_ecount : count (Wd t5) (transformer t5 codes2).embedded = 1 := by decide +kernel

lemma featIn_spec (W : ℕ) (row : Row) (s : Name) (h : featIn W row s < W) :
    row (featIn W row s) = some s := by
  unfold featIn at *
  cases hf : (List.range W).find? (fun f => row f == some s) with
  | none => simp [hf] at h
  | some f =>
    have := List.find?_some hf
    simp only [Option.getD_some]
    simpa using this

lemma range_getD (n k : ℕ) (hk : k < n) : (List.range n).getD k 0 = k := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_range hk]; rfl

/-- Only head 512 writes into feature 2, the feature of `x0`. -/
lemma t5_o (h : ℕ) (hh : h < 513) : (transformer t5 codes2).o h 2 0 = if h = 512 then 1 else 0 := by
  have hk : t5.inputs.getD ((t5.inputs.length / t5.outputs.length - 1 - h) *
      t5.outputs.length + 0) 0 = 512 - h := by
    rw [t5_window, t5_bits, t5_inputs, show (513 - 1 - h) * 1 + 0 = 512 - h by omega]
    exact range_getD _ _ (by omega)
  have hf := t5_feat
  have h2 := t5_row2
  have hW := t5_width
  simp only [Wd] at hf hW
  unfold transformer
  dsimp only
  simp only [row0] at hf h2
  split_ifs with hc h512 h512
  · rfl
  · exfalso
    obtain ⟨_, _, hc⟩ := hc
    rw [hk] at hc
    have := featIn_spec _ _ _ (hc ▸ (by omega : 2 < (tokenGrid t5).width))
    rw [← hc, h2] at this
    simp only [Option.some.injEq, Name.sig.injEq] at this
    have h' : (0 : ℕ) = 512 - h := this
    omega
  · exfalso
    apply hc
    refine ⟨by rw [t5_window]; omega, by rw [t5_bits]; omega, ?_⟩
    rw [hk, h512]
    exact hf.symm
  · rfl

/-- The embedding of a token of `codes2` for a one-bit program. -/
lemma embed_codes2' (t : Trace) (hb : t.outputs.length = 1) (τ : ℕ) (hτ : τ < 2) (f : ℕ) :
    (transformer t codes2).embed τ f =
      if f = 0 then 2 * Real.sqrt (Wd t) else if f = 1 then (if τ = 1 then 1 else -1) else 0 := by
  have l0 : codes2.lookup 0 = some [false] := rfl
  have l1 : codes2.lookup 1 = some [true] := rfl
  simp only [transformer, emit, hb]
  rcases Nat.lt_or_ge f 2 with h | h
  · interval_cases τ <;> interval_cases f <;> simp [l0, l1]
  · have h0 : f ≠ 0 := by omega
    have h1' : f ≠ 1 := by omega
    have h1 : ¬ f ≤ 1 := by omega
    interval_cases τ <;> simp [h0, h1', h1, l0, l1]

lemma sum_sq_two (W : ℕ) (hW : 2 ≤ W) (e : ℕ → ℝ) (he : ∀ g, 2 ≤ g → e g = 0) :
    ∑ g ∈ range W, e g ^ 2 = e 0 ^ 2 + e 1 ^ 2 := by
  rw [← sum_range_add_sum_Ico _ hW, sum_eq_zero (s := Ico 2 W) (fun g hg => by
    rw [he g (mem_Ico.mp hg).1]; ring)]
  simp [sum_range_succ]

lemma t5_embed0 : (transformer t5 codes2).embed 0 0 = 2 * Real.sqrt (Wd t5) ∧
    (transformer t5 codes2).embed 0 1 = -1 ∧
    ∀ g, 2 ≤ g → (transformer t5 codes2).embed 0 g = 0 := by
  have he := embed_codes2' t5 t5_bits 0 (by norm_num)
  refine ⟨by rw [he, ite_eq_left rfl], by rw [he, ite_eq_right one_ne_zero, ite_eq_left rfl, ite_eq_right zero_ne_one],
    fun g hg => by rw [he, ite_eq_right (by omega), ite_eq_right (by omega)]⟩

/-- A token of code `[false]` has the value -1 (its code bit, held as -1). -/
lemma t5_value (p' : ℕ) (hp : toks5 p' = 0) : (transformer t5 codes2).value toks5 p' 0 = -1 := by
  have hW : (transformer t5 codes2).net.W = 515 := t5_width
  have hone : (transformer t5 codes2).net.one = 2 * Real.sqrt (Wd t5) := rfl
  obtain ⟨he0, he1, he2⟩ := t5_embed0
  have hsum : ∑ g ∈ range 515, (transformer t5 codes2).embed 0 g ^ 2 =
      ((transformer t5 codes2).net.one) ^ 2 + count 515 (transformer t5 codes2).embedded := by
    rw [← t5_width, t5_ecount, sum_sq_two _ (by rw [t5_width]; norm_num) _ he2, he0, he1, hone,
      Nat.cast_one]
    ring
  have hv : ∀ f, (transformer t5 codes2).v 0 f = if f = 1 then 1 else 0 := by
    intro f
    unfold transformer
    dsimp only
    rw [t5_bits]
    by_cases hf : f = 1
    · rw [ite_eq_left ⟨by norm_num, by omega⟩, ite_eq_left hf]
    · rw [ite_eq_right (fun h => hf (by omega)), ite_eq_right hf]
  unfold Transformer.value Transformer.matvec Transformer.zvec
  rw [hp, hW, rmsnorm_of_sum _ _ _ _ hsum]
  rw [sum_eq_single 1 (fun g _ hg => by rw [hv, ite_eq_right hg, zero_mul]) (by simp)]
  rw [hv, ite_eq_left rfl, one_mul]
  beta_reduce
  rw [ite_eq_right one_ne_zero, he1]

/-- **CE-5, the attention, Lean**: with a window of 513 tokens, the head meant for the oldest token
writes the mean of the 512 it sees, all 0 (held -1), into the feature of `x0`, whose bit is 1. -/
theorem t5_afterAttn : (transformer t5 codes2).afterAttn toks5 512 2 = -1 := by
  have hheads : (transformer t5 codes2).heads = 513 := t5_window
  have hbits : (transformer t5 codes2).bits = 1 := t5_bits
  have hmean : ∀ h, ∑ p' ∈ visible 512, (transformer t5 codes2).attn toks5 h 512 p' *
      (transformer t5 codes2).value toks5 p' 0 = -1 := by
    intro h
    apply head_const
    intro p' hp'
    apply t5_value
    rw [visible, Finset.mem_filter, Finset.mem_range] at hp'
    simp only [toks5, CONTEXT] at hp' ⊢
    rw [ite_eq_right (by omega)]
  unfold Transformer.afterAttn
  rw [hheads, hbits]
  simp only [sum_range_one, hmean]
  rw [sum_congr rfl (fun h hh => by rw [t5_o h (mem_range.mp hh)])]
  rw [show toks5 512 = 0 from rfl, t5_embed0.2.2 2 le_rfl]
  simp

lemma t5_val0 : valAt t5 codes2 toks5 512 (.sig 0) = 1 := by
  have l1 : codes2.lookup 1 = some [true] := rfl
  have hi : List.idxOf 0 t5.inputs = 0 := by
    rw [t5_inputs, List.range_succ_eq_map, List.idxOf_cons_self]
  have hw : t5.window = 513 := t5_window
  have hx : windowBits t5 codes2 toks5 512 0 = true := by
    simp only [windowBits, hw, t5_bits]
    simp [toks5, l1]
  simp only [valAt, Trace.values, t5_mlps, List.foldl_cons, List.foldl_nil, MLP.apply, m5,
    List.mem_singleton, Trace.env]
  rw [ite_eq_right (by norm_num), hi, hx]
  simp

lemma t5_inVals (x : ℕ → Bool) : t5.inVals x m5 0 = if x 0 then 1 else 0 := by
  have hi : List.idxOf 0 t5.inputs = 0 := by
    rw [t5_inputs, List.range_succ_eq_map, List.idxOf_cons_self]
  simp only [Trace.inVals, MLP.read, m5, List.getD_cons_zero, Trace.values, t5_mlps,
    List.foldl_cons, List.foldl_nil, MLP.apply, List.mem_singleton, Trace.env]
  rw [ite_eq_right (by norm_num), hi]

theorem ce5 : ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
    (F_nodup t ∧ F_outs_ne t ∧ F_topo t ∧ F_bits t ∧ F_step_margin t ∧ F_gate_margin t ∧
      F_outputs_ne t ∧ F_outputs_nodup t ∧ F_outputs_made t ∧ F_window_dvd t) ∧
    ¬ F_window_le t ∧ t.RealOK ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
    ¬ (let M := transformer t codes
       Close M.net.W (1 / 10 ^ 10) M.net.one (valAt t codes toks p)
        ((tokenGrid t).rows.headD fun _ => none) (M.afterAttn toks p)) := by
  refine ⟨t5, codes2, toks5, 512, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · unfold F_nodup
    rw [t5_inputs, t5_mlps]
    simp only [m5, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [List.nodup_append]
    refine ⟨List.nodup_range, List.nodup_singleton _, ?_⟩
    intro a ha b hb
    rw [List.mem_range] at ha
    rw [List.mem_singleton] at hb
    rintro rfl
    rw [hb] at ha
    exact lt_irrefl _ ha
  · intro m hm; rw [t5_mlps, List.mem_singleton] at hm; subst hm; simp [m5]
  · intro i hi x hx
    have hl : t5.mlps.length = 1 := by rw [t5_mlps]; rfl
    have hi0 : i = 0 := by omega
    subst hi0
    left
    have : t5.mlps[0] = m5 := by simp only [t5_mlps, List.getElem_cons_zero]
    rw [this] at hx
    simp only [m5, List.mem_singleton] at hx
    rw [hx, t5_inputs, List.mem_range]; norm_num
  · intro x m hm y hy
    rw [t5_mlps, List.mem_singleton] at hm; subst hm
    simp only [m5, List.mem_singleton] at hy; subst hy
    simp only [Trace.values, t5_mlps, List.foldl_cons, List.foldl_nil, MLP.apply, m5,
      List.mem_singleton, ite_true]
    simp only [MLP.out, MLP.unit, sum_range_one]
    unfold heaviside IsBit; split_ifs <;> simp
  · intro x m hm _ j hj
    rw [t5_mlps, List.mem_singleton] at hm; subst hm
    have h0 := t5_inVals x
    simp only [MLP.pre]
    rw [show m5.ins.length = 1 from rfl, sum_range_one, h0]
    simp only [m5]
    split_ifs <;> norm_num
  · intro x m hm hact; rw [t5_mlps, List.mem_singleton] at hm; subst hm; simp [m5] at hact
  · rw [F_outputs_ne, t5_outputs]; simp
  · rw [F_outputs_nodup, t5_outputs]; exact List.nodup_singleton _
  · intro y hy; rw [t5_outputs, List.mem_singleton] at hy; subst hy
    exact ⟨m5, by rw [t5_mlps]; simp, by simp [m5]⟩
  · rw [F_window_dvd, t5_bits]; exact one_dvd _
  · rw [F_window_le, Trace.window, t5_window, CONTEXT]; norm_num
  · refine realOK_of_steps _ (fun m hm => by rw [t5_mlps, List.mem_singleton] at hm; subst hm; rfl)
      (fun m hm => ?_)
    rw [t5_mlps, List.mem_singleton] at hm; subst hm
    refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, m5] at hj ⊢ <;> norm_num
  · refine ⟨by decide, fun c hc => ?_, by decide⟩
    rw [t5_bits]; revert c; decide
  · refine ⟨by rw [Trace.window, t5_window], fun h _ => ?_⟩
    simp only [toks5]; split_ifs <;> simp [codes2]
  · intro h
    have hW : (transformer t5 codes2).net.W = 515 := t5_width
    have := h.near 2 (by norm_num) (by rw [hW]; norm_num)
    rw [t5_afterAttn] at this
    have h2 := t5_row2
    simp only [row0] at h2
    simp only [ideal, h2, t5_val0] at this
    norm_num [abs_le] at this

/-! ## CE-12: the norm reads `ONE` as the float `2 sqrt W`, the stream holds it in bf16 -/

lemma sqrt21_bounds : 9.1651 ≤ 2 * Real.sqrt 21 ∧ 2 * Real.sqrt 21 ≤ 9.1652 := by
  have h1 : 4.58255 ≤ Real.sqrt 21 := by rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num
  have h2 : Real.sqrt 21 ≤ 4.5826 := by rw [Real.sqrt_le_left (by norm_num)]; norm_num
  constructor <;> linarith

/-- `2 sqrt 21 ≈ 9.16515` rounds to `147/16 = 9.1875` under every rounding: its bf16 neighbours are
`146/16` and `147/16`, at about `0.040` and `0.022`. -/
theorem fl_two_sqrt21 (R : Rounding) : R.fl (2 * Real.sqrt 21) = 147 / 16 := by
  obtain ⟨h1, h2⟩ := sqrt21_bounds
  refine fl_eq_of_unique R (small_le_max (by rw [abs_le]; constructor <;> linarith))
    ⟨147, -4, by norm_num, by norm_num, by norm_num, by norm_num⟩ (fun r hr h => ?_)
  rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ 147 / 16 - 2 * Real.sqrt 21)] at h
  have hr2 : 2 * Real.sqrt 21 - (147 / 16 - 2 * Real.sqrt 21) ≤ r := by
    have := neg_abs_le (r - 2 * Real.sqrt 21); linarith
  have hr3 : r ≤ 147 / 16 := by
    have := le_abs_self (r - 2 * Real.sqrt 21); linarith
  obtain ⟨n, rfl⟩ := bf16_grid hr (-4) (by rw [abs_of_pos (by linarith)]; norm_num; linarith)
  have hn1 : (146 : ℝ) < n := by
    have : (146 : ℝ) / 16 < n * 2 ^ (-4 : ℤ) := by linarith
    norm_num at this; linarith
  have hn2 : (n : ℝ) ≤ 147 := by
    have : (n : ℝ) * 2 ^ (-4 : ℤ) ≤ 147 / 16 := hr3
    norm_num at this; linarith
  have a' : (146 : ℤ) < n := by exact_mod_cast hn1
  have b' : n ≤ (147 : ℤ) := by exact_mod_cast hn2
  have : n = 147 := by omega
  subst this; norm_num

/-- The ratio the norm puts on a bit, on the embedding of a token of a one-bit program of width
21, if the norm's weight reads `ONE` as `2 sqrt 21` and the stream holds it as `147/16`. -/
def q12 : ℝ :=
  Real.sqrt ((84 + 1) / 21 + EPS) / Real.sqrt (((147 / 16 : ℝ) ^ 2 + 1) / 21 + EPS)

lemma q12_bounds : 0.9973 ≤ q12 ∧ q12 ≤ 0.9978 := by
  unfold q12 EPS
  rw [← Real.sqrt_div (by norm_num)]
  constructor
  · rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num
  · rw [Real.sqrt_le_left (by norm_num)]; norm_num

/-- **CE-12, Lean, every rounding**: the bf16 bit that the norm gives is `255/256`, not 1. -/
theorem ce12_ratio (R : Rounding) :
    R.fl (Real.sqrt ((84 + 1) / 21 + EPS) /
      Real.sqrt ((R.fl (2 * Real.sqrt 21) ^ 2 + 1) / 21 + EPS)) = 255 / 256 := by
  rw [fl_two_sqrt21]
  change R.fl q12 = 255 / 256
  obtain ⟨h1, h2⟩ := q12_bounds
  refine fl_eq_of_unique R (small_le_max (by rw [abs_le]; constructor <;> linarith))
    ⟨255, -8, by norm_num, by norm_num, by norm_num, by norm_num⟩ (fun r hr h => ?_)
  rw [abs_of_nonpos (by linarith : (255 / 256 : ℝ) - q12 ≤ 0)] at h
  have hr2 : r ≤ q12 + (q12 - 255 / 256) := by
    have := le_abs_self (r - q12); linarith
  have hr3 : 255 / 256 ≤ r := by
    have := neg_abs_le (r - q12); linarith
  have hr4 : r < 1 := by linarith
  obtain ⟨n, rfl⟩ := bf16_grid hr (-8) (by rw [abs_of_pos (by linarith)]; norm_num; linarith)
  have hn1 : (255 : ℝ) ≤ n := by
    have : (255 : ℝ) / 256 ≤ n * 2 ^ (-8 : ℤ) := hr3
    norm_num at this; linarith
  have hn2 : (n : ℝ) < 256 := by
    norm_num at hr4; linarith
  have a' : (255 : ℤ) ≤ n := by exact_mod_cast hn1
  have b' : n < (256 : ℤ) := by exact_mod_cast hn2
  have : n = 255 := by omega
  subst this; norm_num

/-- **CE-12, in the norm itself**: on the bf16 embedding `[fl(2 sqrt 21), 1, 0, ...]` of a token
of a one-bit program of width 21 (a row of one signal but `ONE`, the code bit), Python's
`rmsnorm(x, row, one)` with the float `one = 2 sqrt 21` gives the bit as `255/256` in bf16, for
every rounding; with `one` the bf16 number the stream holds, `fl(2 sqrt 21)` (our model,
`BF16.norm`, and the patch `one = round_to_bf16(2 * sqrt(W))`), it gives 1. -/
theorem ce12 (R : Rounding) (row : Row) (hc : count 21 row = 1) (x : ℕ → ℝ)
    (hx0 : x 0 = R.fl (2 * Real.sqrt 21)) (hx1 : x 1 = 1) (hx : ∀ g, 2 ≤ g → x g = 0) :
    R.fl (rmsnorm 21 row (2 * Real.sqrt 21) x 1) = 255 / 256 ∧
      R.fl (rmsnorm 21 row (R.fl (2 * Real.sqrt 21)) x 1) = 1 := by
  have hsum : ∑ g ∈ range 21, x g ^ 2 = R.fl (2 * Real.sqrt 21) ^ 2 + 1 := by
    rw [sum_sq_two 21 (by norm_num) x hx, hx0, hx1]; ring
  have h84 : (2 * Real.sqrt 21) ^ 2 = 84 := by
    rw [mul_pow, Real.sq_sqrt (by norm_num)]; norm_num
  constructor
  · unfold rmsnorm rmsScale
    simp only [one_ne_zero, ite_false, hc, hsum, hx1, h84, mul_one]
    push_cast
    exact ce12_ratio R
  · rw [rmsnorm_of_sum _ _ _ _ (by rw [hsum, hc]; simp)]
    simp only [one_ne_zero, ite_false, hx1]
    exact fl_bf16 R bf16_one

/-! ## CE-2b: a gate exactly 0, in real arithmetic -/

/-- CE-2b's MLP: `y = 2 relu(2x - 1) - relu(4x - 3) + C relu(a x - a)`, which is `x` on bits. Its
gates on bits are -1 or 1, -3 or 1, and `-a` or 0: the weak margin holds, the strict one does not.
Where a gate is positive (the first two, at `x = 1`) their slopes cancel: it is flat. -/
def zg (C a : ℚ) (x y : Sig) : MLP where
  ins := [x]
  outs := [y]
  act := .relu
  hidden := 3
  wi := fun j _ => if j = 0 then 2 else if j = 1 then 4 else a
  b := fun j => if j = 0 then -1 else if j = 1 then -3 else -a
  wo := fun _ j => if j = 0 then 2 else if j = 1 then -1 else C

lemma zg_pre (C a : ℚ) (x y : Sig) (u : ℕ → ℝ) :
    (zg C a x y).pre u 0 = 2 * u 0 - 1 ∧ (zg C a x y).pre u 1 = 4 * u 0 - 3 ∧
      (zg C a x y).pre u 2 = a * u 0 - a := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [MLP.pre, zg] <;> ring

lemma zg_out (C a : ℚ) (x y : Sig) (u : ℕ → ℝ) (i : ℕ) :
    (zg C a x y).out u i =
      2 * relu (2 * u 0 - 1) - relu (4 * u 0 - 3) + C * relu (a * u 0 - a) := by
  obtain ⟨h0, h1, h2⟩ := zg_pre C a x y u
  simp only [MLP.out, MLP.unit, show (zg C a x y).hidden = 3 from rfl,
    show (zg C a x y).act = .relu from rfl, sum_range_succ, sum_range_zero, h0, h1, h2]
  simp [zg]
  ring

/-- It is `x` on bits. -/
lemma zg_bit (C a : ℚ) (ha : 0 < a) (x y : Sig) (u : ℕ → ℝ) (hu : u 0 = 0 ∨ u 0 = 1) (i : ℕ) :
    (zg C a x y).out u i = u 0 := by
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha
  rw [zg_out]
  rcases hu with h | h <;> rw [h]
  · have e3 : relu ((a : ℝ) * 0 - a) = 0 := by unfold relu; rw [max_eq_right (by linarith)]
    rw [e3]; norm_num [relu]
  · have e3 : relu ((a : ℝ) * 1 - a) = 0 := by unfold relu; rw [max_eq_right (by linarith)]
    rw [e3]; norm_num [relu]

/-- Its gates on bits are 0 or at least 1/2 from 0. -/
lemma zg_weak (C a : ℚ) (ha : 1 / 2 ≤ a) (x y : Sig) (u : ℕ → ℝ) (hu : u 0 = 0 ∨ u 0 = 1)
    (j : ℕ) (hj : j < 3) :
    (zg C a x y).pre u j = 0 ∨ 1 / 2 ≤ |(zg C a x y).pre u j| := by
  have ha' : (1 / 2 : ℝ) ≤ a := by
    have := (Rat.cast_le (K := ℝ)).mpr ha; push_cast at this; exact this
  obtain ⟨h0, h1, h2⟩ := zg_pre C a x y u
  interval_cases j
  · right; rw [h0]; rcases hu with h | h <;> rw [h] <;> norm_num
  · right; rw [h1]; rcases hu with h | h <;> rw [h] <;> norm_num
  · rw [h2]; rcases hu with h | h <;> rw [h]
    · right; rw [abs_of_nonpos (by linarith)]; linarith
    · left; ring

/-- At `x = 1` its third gate is 0: the strict margin fails. -/
lemma zg_zero (C a : ℚ) (x y : Sig) (u : ℕ → ℝ) (hu : u 0 = 1) : (zg C a x y).pre u 2 = 0 := by
  rw [(zg_pre C a x y u).2.2, hu]; ring

/-- It is flat on bits: no slope along its input. -/
lemma zg_slope (C a : ℚ) (ha : 0 < a) (x y : Sig) (u : ℕ → ℝ) (hu : u 0 = 0 ∨ u 0 = 1) (i : ℕ) :
    (zg C a x y).slope u i 0 = 0 := by
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha
  obtain ⟨h0, h1, h2⟩ := zg_pre C a x y u
  simp only [MLP.slope, show (zg C a x y).hidden = 3 from rfl, sum_range_succ, sum_range_zero,
    h0, h1, h2, show (zg C a x y).act = .relu from rfl]
  rcases hu with h | h <;> rw [h] <;> norm_num [zg]
  all_goals intro h'; exact absurd h' (not_lt.mpr ha.le)

lemma zg_bounded (C a : ℚ) (hC : 0 ≤ C) (hC' : C ≤ 2 ^ 16 - 3) (ha : 0 < a) (ha' : a ≤ 2 ^ 16)
    (x y : Sig) : (zg C a x y).Bounded (2 ^ 16) := by
  refine ⟨fun j hj => ?_, fun j hj => ?_, fun i _ => ?_⟩
  · simp [MLP.rows, zg] at hj
    simp only [zg, List.length_cons, List.length_nil, zero_add, sum_range_one]
    interval_cases j
    all_goals norm_num [abs_of_pos ha]
    all_goals linarith
  · simp [MLP.rows, zg] at hj
    simp only [zg]
    interval_cases j
    all_goals norm_num [abs_of_pos ha]
    all_goals linarith
  · simp only [zg, sum_range_succ, sum_range_zero]
    norm_num [abs_of_nonneg hC]
    linarith

/-- `gelu` near 0 is half the identity: `z / 2 ≤ gelu z ≤ 0.45 z` on `[-1/10, 0]`, where `relu` is 0. -/
lemma gelu_leak {z : ℝ} (h1 : -1 / 10 ≤ z) (h2 : z ≤ 0) : z / 2 ≤ gelu z ∧ gelu z ≤ 45 / 100 * z := by
  rw [gelu_eq]
  have hs1 := sqrt_two_div_pi_gt
  have hs2 := sqrt_two_div_pi_lt
  have hy1 : geluArg z ≤ 0 := by
    unfold geluArg
    have : z + 0.044715 * z ^ 3 ≤ 0 := by nlinarith [sq_nonneg z]
    nlinarith
  have hy2 : -0.081 ≤ geluArg z := by
    unfold geluArg
    have h3 : -0.10005 ≤ z + 0.044715 * z ^ 3 := by nlinarith [sq_nonneg z, sq_nonneg (z + 1 / 10)]
    have : z + 0.044715 * z ^ 3 ≤ 0 := by nlinarith [sq_nonneg z]
    nlinarith
  set e := Real.exp (-2 * geluArg z)
  have he1 : 1 ≤ e := Real.one_le_exp (by linarith)
  have he2 : e ≤ 1.2 := by
    have h := Real.exp_bound (x := 0.162) (by norm_num [abs_of_pos]) (n := 3) (by norm_num)
    have h' := (abs_le.mp h).2
    norm_num [Finset.sum_range_succ, Nat.factorial, abs_of_pos] at h'
    calc e ≤ Real.exp 0.162 := Real.exp_le_exp.mpr (by linarith)
      _ ≤ 1.2 := by linarith
  have hpos : 0 < 1 + e := by linarith
  constructor
  · rw [le_div_iff₀ hpos]; nlinarith
  · rw [div_le_iff₀ hpos]; nlinarith

/-- **CE-2b, local**: the compiled third unit of `zg C a` (gate `16 a` on `x`, `-16 a` on `ONE`,
value `1/32` on `ONE`, weight `2 C` into `y`), on a normed stream that holds `ONE` as `c` and `x`
as `c (1 - ε)`, an error `ε` below the bit 1: its gate is `-16 a c ε`, where `relu` gives 0, but
`gelu` writes between `-C a c^2 ε / 2` and `-0.45 C a c^2 ε` into `y`. So a chain multiplies `ε`
by about `C a / 2` at each depth (`sim.py`). -/
theorem ce2b_local (C a : ℚ) (x y : Sig) (c ε : ℝ) (hC : 0 ≤ C) (ha : 0 < a) (hc : 0 < c)
    (hε : 0 ≤ ε) (hsmall : 16 * a * c * ε ≤ 1 / 10) :
    let m := (zg C a x y).compiled
    let G := (m.g 2 0 : ℝ) * (c * (1 - ε)) + m.gb 2 * c
    m.vb 2 = 1 / 32 ∧ m.wo 0 2 = 2 * C ∧ G = -16 * a * c * ε ∧ relu G = 0 ∧
      -(C * a * c ^ 2 * ε) / 2 ≤ (m.wo 0 2 : ℝ) * (gelu G * (m.vb 2 * c)) ∧
      (m.wo 0 2 : ℝ) * (gelu G * (m.vb 2 * c)) ≤ -(45 / 100) * (C * a * c ^ 2 * ε) := by
  have hg : (zg C a x y).compiled.g 2 0 = 16 * a := by
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, zg, Gated.sharpness]; ring
  have hgb : (zg C a x y).compiled.gb 2 = -16 * a := by
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, zg, Gated.sharpness]; ring
  have hvb : (zg C a x y).compiled.vb 2 = 1 / 32 := by
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, zg, Gated.sharpness]
  have hwo : (zg C a x y).compiled.wo 0 2 = 2 * C := by
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, zg, Gated.sharpness]
  have hC' : (0 : ℝ) ≤ C := by exact_mod_cast hC
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha
  have hsm : (16 * a * c * ε : ℝ) ≤ 1 / 10 := by exact_mod_cast hsmall
  intro m G
  have hG : G = -16 * a * c * ε := by
    simp only [G, m, hg, hgb]; push_cast; ring
  have hG0 : G ≤ 0 := by rw [hG]; have := mul_nonneg (mul_nonneg ha'.le hc.le) hε; nlinarith
  obtain ⟨l1, l2⟩ := gelu_leak (z := G) (by rw [hG]; linarith) hG0
  refine ⟨hvb, hwo, hG, by simp [relu, hG0], ?_, ?_⟩
  · simp only [m, hvb, hwo]; push_cast
    have := mul_le_mul_of_nonneg_left l1 (by positivity : (0 : ℝ) ≤ C * c / 16)
    rw [hG] at this ⊢
    nlinarith
  · simp only [m, hvb, hwo]; push_cast
    have := mul_le_mul_of_nonneg_left l2 (by positivity : (0 : ℝ) ≤ C * c / 16)
    rw [hG] at this ⊢
    nlinarith

/-- The fields of a trace of `zg C a` MLPs whose inputs are bits: all of `IsProgram`'s margins
(the weak one), flatness, bounds, and a first-order gain of 0 (`RealCore _ 0`). -/
lemma zg_fields (t : Trace) (C a : ℚ) (hC : 0 ≤ C) (hC' : C ≤ 2 ^ 16 - 3) (ha : 1 / 2 ≤ a)
    (ha' : a ≤ 2 ^ 16) (hm : ∀ m ∈ t.mlps, ∃ x y, m = zg C a x y)
    (hbit : ∀ x, ∀ m ∈ t.mlps, t.inVals x m 0 = 0 ∨ t.inVals x m 0 = 1) :
    F_step_margin t ∧ F_gate_margin t ∧ F_flat t ∧ F_bounded t ∧ F_glu_gain t ∧ F_slope t 0 ∧
      F_glu_gainG t (10 ^ 5) := by
  have ha0 : 0 < a := by linarith
  refine ⟨fun x m hm' hact => ?_, fun x m hm' _ j hj => ?_, fun x m hm' _ i _ k hk => ?_,
    fun m hm' => ?_, fun m hm' hact => ?_, fun x m hm' _ i _ => ?_, fun m hm' hact => ?_⟩
  all_goals obtain ⟨x', y', rfl⟩ := hm m hm'
  · simp [zg] at hact
  · exact zg_weak C a ha x' y' _ (hbit x _ hm') j hj
  · simp [zg] at hk; subst hk; exact zg_slope C a ha0 x' y' _ (hbit x _ hm') i
  · exact zg_bounded C a hC hC' ha0 ha' x' y'
  · simp [zg] at hact
  · simp only [MLP.slopeSum, show (zg C a x' y').ins.length = 1 from rfl, sum_range_one,
      zg_slope C a ha0 x' y' _ (hbit x _ hm') i, abs_zero, le_refl]
  · simp [zg] at hact

def bv (b : Bool) : ℝ := if b then 1 else 0

lemma bv_bit (b : Bool) : bv b = 0 ∨ bv b = 1 := by cases b <;> simp [bv]

/-- CE-2b at depth 1: `zg (2^15) 1` on the bit of the last token. -/
def t2b : Trace where
  inputs := [0]
  mlps := [zg (2 ^ 15) 1 0 1]
  outputs := [1]

/-- CE-2b at depth 2: two `zg (2^15) (2^12)` in a chain. -/
def t2b2 : Trace where
  inputs := [0]
  mlps := [zg (2 ^ 15) (2 ^ 12) 0 1, zg (2 ^ 15) (2 ^ 12) 1 2]
  outputs := [2]

/-- The tokens of `sim.py`: 0, then 1; at `p = 1` the head leaks the 0's -1 into the input. -/
def toks2b : ℕ → ℕ := fun q => if q = 0 then 0 else 1

lemma t2b_values (x : ℕ → Bool) :
    t2b.values (t2b.env x) 0 = bv (x 0) ∧ t2b.values (t2b.env x) 1 = bv (x 0) := by
  have h0 : t2b.env x 0 = bv (x 0) := by simp [Trace.env, t2b, bv]
  set m1 := zg (2 ^ 15) 1 0 1
  have hv : t2b.values (t2b.env x) = m1.apply (t2b.env x) := rfl
  have hr : m1.read (t2b.env x) 0 = bv (x 0) := by
    simp only [MLP.read, m1, zg, List.getD_cons_zero]; exact h0
  rw [hv]
  constructor
  · simp only [MLP.apply, m1, show (zg (2 ^ 15) 1 0 1).outs = [1] from rfl, List.mem_singleton]
    simp [h0]
  · simp only [MLP.apply, m1, show (zg (2 ^ 15) 1 0 1).outs = [1] from rfl, List.mem_singleton,
      ite_true]
    rw [zg_bit _ _ (by norm_num) _ _ _ (by rw [hr]; exact bv_bit _), hr]

lemma t2b_inVals (x : ℕ → Bool) (m : MLP) (hm : m ∈ t2b.mlps) (k : ℕ) :
    t2b.inVals x m k = bv (x 0) := by
  simp [t2b] at hm; subst hm
  show t2b.values (t2b.env x) ([0].getD k 0) = _
  rcases k with _ | k <;> simp [(t2b_values x).1]

lemma t2b2_values (x : ℕ → Bool) :
    t2b2.values (t2b2.env x) 0 = bv (x 0) ∧ t2b2.values (t2b2.env x) 1 = bv (x 0) ∧
      t2b2.values (t2b2.env x) 2 = bv (x 0) := by
  have h0 : t2b2.env x 0 = bv (x 0) := by simp [Trace.env, t2b2, bv]
  set m1 := zg (2 ^ 15) (2 ^ 12) 0 1
  set m2 := zg (2 ^ 15) (2 ^ 12) 1 2
  have hv : t2b2.values (t2b2.env x) = m2.apply (m1.apply (t2b2.env x)) := rfl
  have e1 : m1.apply (t2b2.env x) 1 = bv (x 0) := by
    simp only [MLP.apply, m1, show (zg (2 ^ 15) (2 ^ 12) 0 1).outs = [1] from rfl,
      List.mem_singleton, ite_true]
    have hr : (zg (2 ^ 15) (2 ^ 12) 0 1).read (t2b2.env x) 0 = bv (x 0) := by
      simp only [MLP.read, zg, List.getD_cons_zero]; exact h0
    rw [zg_bit _ _ (by norm_num) _ _ _ (by rw [hr]; exact bv_bit _), hr]
  have e0 : m1.apply (t2b2.env x) 0 = bv (x 0) := by
    simp only [MLP.apply, m1, show (zg (2 ^ 15) (2 ^ 12) 0 1).outs = [1] from rfl,
      List.mem_singleton]
    simp [h0]
  refine ⟨?_, ?_, ?_⟩ <;> rw [hv]
  · simp only [MLP.apply, m2, show (zg (2 ^ 15) (2 ^ 12) 1 2).outs = [2] from rfl,
      List.mem_singleton]
    simp only [show ¬ (0 : ℕ) = 2 by decide, ite_false]; exact e0
  · simp only [MLP.apply, m2, show (zg (2 ^ 15) (2 ^ 12) 1 2).outs = [2] from rfl,
      List.mem_singleton]
    simp only [show ¬ (1 : ℕ) = 2 by decide, ite_false]; exact e1
  · simp only [MLP.apply, m2, show (zg (2 ^ 15) (2 ^ 12) 1 2).outs = [2] from rfl,
      List.mem_singleton, ite_true]
    have hr : (zg (2 ^ 15) (2 ^ 12) 1 2).read (m1.apply (t2b2.env x)) 0 = bv (x 0) := by
      simp only [MLP.read, zg, List.getD_cons_zero]; exact e1
    rw [zg_bit _ _ (by norm_num) _ _ _ (by rw [hr]; exact bv_bit _), hr]

lemma t2b2_inVals (x : ℕ → Bool) (m : MLP) (hm : m ∈ t2b2.mlps) (k : ℕ) :
    t2b2.inVals x m k = bv (x 0) := by
  obtain ⟨h0, h1, _⟩ := t2b2_values x
  simp [t2b2] at hm
  rcases hm with rfl | rfl
  · show t2b2.values (t2b2.env x) ([0].getD k 0) = _
    rcases k with _ | k <;> simp [h0]
  · show t2b2.values (t2b2.env x) ([1].getD k 0) = _
    rcases k with _ | k <;> simp [h0, h1]

/-- The layer of `t2b`, as `sim.py` computes it too (the cross-check that the simulation runs this
network): `ONE`, the units of `2 relu(2x-1)`, `-relu(4x-3)`, `2^15 relu(x-1)` with gates on `ONE`
(feature 0) and `x` (feature 2), and the erasure of `code[0]` (feature 1, where `y` goes). -/
lemma t2b_units : (layerUnits 3 t2b.mlps (GR t2b 0) (GR t2b 1)).map (unitData 3) =
    [([32, 0, 0], [1 / 32, 0, 0], [0, -1, 0]), ([0, 0, 32], [1 / 32, 0, 0], [0, 4, 0]),
     ([-32, 0, 64], [1 / 32, 0, 0], [0, -2, 0]), ([-16, 0, 16], [1 / 32, 0, 0], [0, 65536, 0]),
     ([32, 0, 0], [0, 1 / 32, 0], [0, -1, 0])] := by
  decide +kernel

lemma t2b_width : Wd t2b = 3 := by decide

/-- **CE-2b, the hypotheses, Lean**: `t2b` and `t2b2` are programs on tokens (with the weak gate
margin), flat, of weights at most `2^16` and no GLU (so `F_slope _ 0`: they satisfy `RealCore
(10^5) 0` but for its margin, and `(1 + 0)^depth = 1`); at `p = 1` on the tokens 0, 1 the position
is fine; but at `x = 1` the gate of their third unit is 0, so the strict margin fails. That is all
of `MainReal`'s (and `MainRealDeep`'s) hypotheses but `GateMargin`. **Numeric simulation**
(`sim.py`, `ce2b`): the head leaks `ε ≈ 1.87e-13` below the bit 1, the zero gate passes it on
through `gelu'(0) = 1/2` (`ce2b_local`), times `C a / 2`: `t2b`'s output feature is
`0.99999999693`, off by `3.07e-9 > 1e-9` (and `> 1e-10`); `t2b2`'s is `-345.1`, the wrong bit. -/
theorem ce2b :
    (t2b.IsProgram ∧ ¬ t2b.GateMargin ∧ (F_flat t2b ∧ F_bounded t2b ∧ F_glu_gain t2b) ∧
      (F_slope t2b 0 ∧ F_glu_gainG t2b (10 ^ 5)) ∧ CodesOK t2b codes2 ∧ PosOK t2b codes2 toks2b 1 ∧
      (tokenGrid t2b).rows.length = 2) ∧
    (t2b2.IsProgram ∧ ¬ t2b2.GateMargin ∧ (F_flat t2b2 ∧ F_bounded t2b2 ∧ F_glu_gain t2b2) ∧
      (F_slope t2b2 0 ∧ F_glu_gainG t2b2 (10 ^ 5)) ∧ CodesOK t2b2 codes2 ∧
      PosOK t2b2 codes2 toks2b 1 ∧ (tokenGrid t2b2).rows.length = 3) := by
  constructor
  · obtain ⟨hs, hg, hf, hb, hgg, hsl, hgG⟩ := zg_fields t2b (2 ^ 15) 1 (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (fun m hm => ⟨0, 1, by simpa [t2b] using hm⟩)
      (fun x m hm => by rw [t2b_inVals x m hm]; exact bv_bit _)
    refine ⟨⟨⟨by decide, ?_, ?_, ?_, hs, hg⟩, by simp [t2b], by decide, ?_, by decide, by decide⟩,
      ?_, ⟨hf, hb, hgg⟩, ⟨hsl, hgG⟩, ⟨by decide, by decide, by decide⟩,
      ⟨by simp [Trace.window, t2b], fun h _ => by simp only [toks2b]; split_ifs <;> simp [codes2]⟩,
      by decide⟩
    · intro m hm; simp [t2b] at hm; subst hm; simp [zg]
    · intro i hi x hx; simp [t2b] at hi; subst hi; left; simpa [t2b, zg] using hx
    · intro x m hm y hy; simp [t2b] at hm; subst hm; simp [zg] at hy; subst hy
      rw [(t2b_values x).2]; exact bv_bit _
    · intro y hy; simp [t2b] at hy; subst hy
      exact ⟨zg (2 ^ 15) 1 0 1, by simp [t2b], by simp [zg]⟩
    · intro h
      have := h (fun _ => true) _ (by simp [t2b] : zg (2 ^ 15) 1 0 1 ∈ t2b.mlps) (by simp [zg]) 2
        (by simp [zg])
      rw [zg_zero _ _ _ _ _ (by rw [t2b_inVals _ _ (by simp [t2b])]; simp [bv])] at this
      norm_num at this
  · obtain ⟨hs, hg, hf, hb, hgg, hsl, hgG⟩ := zg_fields t2b2 (2 ^ 15) (2 ^ 12) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num)
      (fun m hm => by
        simp [t2b2] at hm
        rcases hm with rfl | rfl
        · exact ⟨0, 1, rfl⟩
        · exact ⟨1, 2, rfl⟩)
      (fun x m hm => by rw [t2b2_inVals x m hm]; exact bv_bit _)
    refine ⟨⟨⟨by decide, ?_, ?_, ?_, hs, hg⟩, by simp [t2b2], by decide, ?_, by decide, by decide⟩,
      ?_, ⟨hf, hb, hgg⟩, ⟨hsl, hgG⟩, ⟨by decide, by decide, by decide⟩,
      ⟨by simp [Trace.window, t2b2], fun h _ => by simp only [toks2b]; split_ifs <;> simp [codes2]⟩,
      by decide⟩
    · intro m hm; simp [t2b2] at hm; rcases hm with rfl | rfl <;> simp [zg]
    · intro i hi x hx
      simp only [t2b2, List.length_cons, List.length_nil] at hi
      interval_cases i
      · left; simpa [t2b2, zg] using hx
      · right; refine ⟨0, by omega, ?_⟩
        simp [t2b2, zg] at hx ⊢; exact hx
    · intro x m hm y hy
      obtain ⟨_, h1, h2⟩ := t2b2_values x
      simp [t2b2] at hm
      rcases hm with rfl | rfl <;> simp [zg] at hy <;> subst hy
      · rw [h1]; exact bv_bit _
      · rw [h2]; exact bv_bit _
    · intro y hy; simp [t2b2] at hy; subst hy
      exact ⟨zg (2 ^ 15) (2 ^ 12) 1 2, by simp [t2b2], by simp [zg]⟩
    · intro h
      have := h (fun _ => true) _ (by simp [t2b2] : zg (2 ^ 15) (2 ^ 12) 0 1 ∈ t2b2.mlps)
        (by simp [zg]) 2 (by simp [zg])
      rw [zg_zero _ _ _ _ _ (by rw [t2b2_inVals _ _ (by simp [t2b2])]; simp [bv])] at this
      norm_num at this

/-! ## CE-7 -/

/-- CE-7's GLU: `relu(y + 1/2) * (2/3) y`, the identity on bits, of slope 5/3 at 1. -/
def glu7 (x y : Sig) : MLP where
  ins := [x]
  outs := [y]
  act := .reglu
  hidden := 1
  wi := fun j _ => if j = 0 then 1 else 2 / 3
  b := fun j => if j = 0 then 1 / 2 else 0
  wo := fun _ _ => 1

lemma glu7_out (x y : Sig) (u : ℕ → ℝ) (hu : -1 / 2 ≤ u 0) :
    (glu7 x y).out u 0 = (u 0 + 1 / 2) * (2 / 3 * u 0) := by
  have h1 : (glu7 x y).pre u 0 = u 0 + 1 / 2 := by simp [MLP.pre, glu7]
  have h2 : (glu7 x y).pre u 1 = 2 / 3 * u 0 := by simp [MLP.pre, glu7]
  have : (glu7 x y).out u 0 = relu ((glu7 x y).pre u 0) * (glu7 x y).pre u 1 := by
    simp [MLP.out, MLP.unit, glu7]
  rw [this, h1, h2, relu, max_eq_left (by linarith)]

/-- It computes the identity on bits, its gate is at least 1/2 from 0 there, its weights are
small, and it is not flat: its slope at 1 is 5/3. -/
theorem glu7_props (x y : Sig) :
    (∀ u : ℕ → ℝ, (u 0 = 0 ∨ u 0 = 1) → (glu7 x y).out u 0 = u 0 ∧
      1 / 2 ≤ |(glu7 x y).pre u 0|) ∧
    (glu7 x y).Bounded 256 ∧
    (glu7 x y).slope (fun _ => 1) 0 0 = 5 / 3 := by
  refine ⟨fun u hu => ?_, ?_, ?_⟩
  · rcases hu with h | h
    · rw [glu7_out x y u (by rw [h]; norm_num), h]
      simp [MLP.pre, glu7, h]
    · rw [glu7_out x y u (by rw [h]; norm_num), h]
      simp [MLP.pre, glu7, h]
      norm_num
  · refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, glu7] at hj ⊢ <;>
      interval_cases j <;> norm_num
  · simp [MLP.slope, MLP.pre, glu7]
    norm_num

/-- Its second-order gain is `2/3`, within `RealOK`'s `10^8` and `RealCore`'s `10^5`. -/
theorem glu7_gain (x y : Sig) : (glu7 x y).gluGain 0 = 2 / 3 := by
  simp [MLP.gluGain, glu7]
  norm_num

/-- It multiplies an error below 1 by at least 3/2: on `1 - δ` it gives `1 - 5/3 δ + 2/3 δ^2`. -/
theorem glu7_amplifies (x y : Sig) (u : ℕ → ℝ) (δ : ℝ) (h0 : 0 ≤ δ) (h1 : δ ≤ 1 / 4)
    (hu : u 0 = 1 - δ) : (glu7 x y).out u 0 ≤ 1 - 3 / 2 * δ := by
  rw [glu7_out x y u (by rw [hu]; linarith), hu]
  nlinarith

/-- After `n` such GLUs an error `δ` is at least `(3/2)^n δ`, until it passes 1/4. -/
theorem glu7_iterate (f : ℕ → ℝ) (δ : ℝ) (h0 : 0 ≤ δ)
    (hf : ∀ n, 0 ≤ 1 - f n → 1 - f n ≤ 1 / 4 → 1 - f (n + 1) ≥ 3 / 2 * (1 - f n))
    (hf0 : 1 - f 0 = δ) (n : ℕ) (hsmall : ∀ m < n, 1 - f m ≤ 1 / 4) :
    (3 / 2) ^ n * δ ≤ 1 - f n := by
  induction n with
  | zero => simp [hf0]
  | succ n ih =>
    have ih' := ih (fun m hm => hsmall m (by omega))
    have hn := hsmall n (by omega)
    have hpos : 0 ≤ 1 - f n := le_trans (by positivity) ih'
    have := hf n hpos hn
    rw [pow_succ]
    nlinarith

/-! ## CE-8 -/

/-- A step gate `[w (x - 1) >= 0]` with a weight `w`: on `x = 1 - 1/(2w)`, an error of `1/(2w)`
in its input, the reglu form gives `1/2`, neither bit. With `w = 10^13`, an error of
`5 * 10^-14`, below what the attention leaks, does it. -/
theorem ce8_local (w : ℝ) (hw : 0 < w) :
    let z := w * ((1 - 1 / (2 * w)) - 1)
    relu (4 * z + 2 + 1 / 2) - relu (4 * z + 2 - 1 / 2) = 1 / 2 := by
  intro z
  have : z = -1 / 2 := by simp only [z]; field_simp; ring
  rw [this]; norm_num [relu]

/-! ## CE-11 -/

/-- A rounding is free beyond the finite range: it may send every larger number to 0. -/
def zeroBeyond (R : Rounding) : Rounding where
  fl := fun x => if |x| ≤ maxBF16 then R.fl x else 0
  fl_mem := fun x hx => by simp only [hx, ite_true]; exact R.fl_mem x hx
  fl_nearest := fun x r hx hr => by simp only [hx, ite_true]; exact R.fl_nearest x r hx hr

/-- CE-11: if `2 sqrt(width)`, the stream's `ONE`, is past `maxBF16`, some rounding holds `ONE` as
0, in the embedding and in every norm; every unit then reads `ONE` as 0. -/
theorem ce11 (t : Trace) (codes : List (ℕ × List Bool)) (h : ¬ WidthOK t) (R : Rounding) :
    BF16.one (zeroBeyond R) (transformer t codes).net = 0 := by
  unfold WidthOK at h
  have h' : maxBF16 < |(transformer t codes).net.one| := by
    have : (transformer t codes).net.one = 2 * Real.sqrt (tokenGrid t).width := rfl
    rw [this, abs_of_pos (by linarith [maxBF16_big])]; linarith
  simp [BF16.one, zeroBeyond, not_le.mpr h']

/-- And the embedding then holds `ONE` as 0 too: the norm divides it by 0, so every unit's gate,
`32 ONE` and the biases on `ONE`, reads 0. -/
theorem ce11_embed (t : Trace) (codes : List (ℕ × List Bool)) (h : ¬ WidthOK t) (R : Rounding)
    (τ : ℕ) : BF16.embed (zeroBeyond R) (transformer t codes) τ 0 = 0 := by
  unfold WidthOK at h
  have h' : maxBF16 < |(transformer t codes).net.one| := by
    have : (transformer t codes).net.one = 2 * Real.sqrt (tokenGrid t).width := rfl
    rw [this, abs_of_pos (by linarith [maxBF16_big])]; linarith
  unfold BF16.embed
  rcases hl : codes.lookup τ with _ | c
  · have : (transformer t codes).embed τ 0 = 0 := by simp [transformer, hl]
    rw [this]; simp [zeroBeyond, abs_zero, maxBF16_big.trans' (by norm_num : (0:ℝ) ≤ 10 ^ 30),
      fl_bf16 R bf16_zero]
  · have : (transformer t codes).embed τ 0 = (transformer t codes).net.one := by
      simp [transformer, hl]
    rw [this]; simp [zeroBeyond, not_le.mpr h']

end

end Reifier.CE

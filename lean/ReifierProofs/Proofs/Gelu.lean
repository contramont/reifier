import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace GeluAux

lemma gelu_neg (z : ℝ) : gelu (-z) = gelu z - z := by
  unfold gelu
  have : Real.sqrt (2 / Real.pi) * (-z + 0.044715 * (-z) ^ 3) =
      -(Real.sqrt (2 / Real.pi) * (z + 0.044715 * z ^ 3)) := by ring
  rw [this, Real.tanh_neg]; ring

lemma relu_neg (z : ℝ) : relu (-z) = relu z - z := by
  unfold relu
  rcases le_total z 0 with h | h
  · rw [max_eq_left (by linarith), max_eq_right h]; ring
  · rw [max_eq_right (by linarith), max_eq_left h]; ring

lemma one_sub_tanh (w : ℝ) :
    0 ≤ 1 - Real.tanh w ∧ 1 - Real.tanh w ≤ 2 * Real.exp (-2 * w) := by
  have hc := Real.cosh_pos w
  have h1 : 1 - Real.tanh w = Real.exp (-w) / Real.cosh w := by
    rw [Real.tanh_eq_sinh_div_cosh, ← Real.cosh_sub_sinh]; field_simp
  have hcosh : Real.exp w / 2 ≤ Real.cosh w := by
    rw [Real.cosh_eq]; have := Real.exp_pos (-w); linarith
  constructor
  · rw [h1]; positivity
  · rw [h1, div_le_iff₀ hc]
    have : Real.exp (-2 * w) * Real.exp w = Real.exp (-w) := by
      rw [← Real.exp_add]; ring_nf
    nlinarith [Real.exp_pos (-2 * w)]

lemma sqrt_two_div_pi_ge : (0.7978 : ℝ) ≤ Real.sqrt (2 / Real.pi) := by
  rw [Real.le_sqrt (by norm_num) (by positivity)]
  have hp := Real.pi_lt_d6
  have hp0 := Real.pi_pos
  have : 2 / (3.141593 : ℝ) ≤ 2 / Real.pi := by
    apply div_le_div_of_nonneg_left (by norm_num) hp0 hp.le
  have h2 : (0.7978 : ℝ) ^ 2 ≤ 2 / 3.141593 := by norm_num
  linarith

lemma exp_49 : (8 : ℝ) * 10 ^ 20 ≤ Real.exp 49 := by
  have h := Real.exp_one_gt_d9
  have h1 : (2.718 : ℝ) ^ 49 ≤ Real.exp 1 ^ 49 := by
    apply pow_le_pow_left₀ (by norm_num); linarith
  rw [Real.exp_one_pow] at h1
  have h2 : (8 : ℝ) * 10 ^ 20 ≤ 2.718 ^ 49 := by norm_num
  have : ((49 : ℕ) : ℝ) = 49 := by norm_num
  rw [this] at h1
  linarith

lemma gelu_pos (z : ℝ) (hz : 8 ≤ z) : |gelu z - relu z| ≤ 1 / 10 ^ 20 := by
  set a := Real.sqrt (2 / Real.pi) with ha_def
  set w := a * (z + 0.044715 * z ^ 3) with hw_def
  have ha := sqrt_two_div_pi_ge
  have hrelu : relu z = z := max_eq_left (by linarith)
  have hg : gelu z - relu z = -(z / 2 * (1 - Real.tanh w)) := by
    rw [hrelu]; unfold gelu; ring
  obtain ⟨ht0, ht1⟩ := one_sub_tanh w
  have hz0 : 0 ≤ z := by linarith
  rw [hg, abs_neg, abs_of_nonneg (by positivity)]
  -- z/2 (1 - tanh w) ≤ z e^{-2w}
  have s1 : z / 2 * (1 - Real.tanh w) ≤ z * Real.exp (-2 * w) := by
    have := mul_le_mul_of_nonneg_left ht1 (by positivity : (0:ℝ) ≤ z / 2)
    linarith
  -- z ≤ 8 e^{z/8 - 1}
  have s2 : z ≤ 8 * Real.exp (z / 8 - 1) := by
    have := Real.add_one_le_exp (z / 8 - 1); linarith
  have hz3 : (512 : ℝ) ≤ z ^ 3 := by
    have := pow_le_pow_left₀ (by norm_num : (0:ℝ) ≤ 8) hz 3
    norm_num at this; linarith
  have s3 : z / 8 - 1 + -2 * w ≤ -49 := by
    have e1 : (2 * a - 1 / 8) * (z - 8) ≥ 0 := by
      apply mul_nonneg <;> linarith
    have e2 : a * (z ^ 3 - 512) ≥ 0 := by
      apply mul_nonneg <;> linarith
    rw [hw_def]; nlinarith
  have s4 : z * Real.exp (-2 * w) ≤ 8 * Real.exp (-49) := by
    calc z * Real.exp (-2 * w) ≤ 8 * Real.exp (z / 8 - 1) * Real.exp (-2 * w) := by
          apply mul_le_mul_of_nonneg_right s2 (Real.exp_pos _).le
      _ = 8 * Real.exp (z / 8 - 1 + -2 * w) := by rw [Real.exp_add]; ring
      _ ≤ 8 * Real.exp (-49) := by
          apply mul_le_mul_of_nonneg_left _ (by norm_num)
          exact Real.exp_le_exp.mpr s3
  have s5 : 8 * Real.exp (-49) ≤ 1 / 10 ^ 20 := by
    rw [Real.exp_neg, ← div_eq_mul_inv]
    have h := exp_49
    have hp := Real.exp_pos 49
    rw [div_le_div_iff₀ hp (by positivity)]
    linarith
  linarith

end GeluAux

open GeluAux in
theorem stmt_gelu : Stmt_gelu := by
  intro z hz
  rcases le_total 0 z with h | h
  · rw [abs_of_nonneg h] at hz
    exact gelu_pos z hz
  · rw [abs_of_nonpos h] at hz
    have := gelu_pos (-z) hz
    rwa [gelu_neg, relu_neg, sub_sub_sub_cancel_right] at this

end

end Reifier

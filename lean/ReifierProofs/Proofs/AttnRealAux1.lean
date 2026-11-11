import ReifierProofs.Main

namespace Reifier
namespace AttnRealPf

open Finset

noncomputable section

lemma turn_zero : turn 0 = 1 := by simp [turn]

lemma turn_pos (i : ℕ) : 0 < turn i := by unfold turn; positivity

lemma turn_le_one (i : ℕ) : turn i ≤ 1 := by
  unfold turn
  apply Real.rpow_le_one_of_one_le_of_nonpos
  · norm_num
  · have : (0:ℝ) ≤ (i:ℝ) / PAIRS := by positivity
    linarith

lemma cos_nat_lt_one (e : ℕ) (he : 1 ≤ e) : Real.cos e < 1 := by
  rcases lt_or_eq_of_le (Real.cos_le_one e) with h | h
  · exact h
  · exfalso
    rw [Real.cos_eq_one_iff] at h
    obtain ⟨n, hn⟩ := h
    have he' : (1:ℝ) ≤ e := by exact_mod_cast he
    have hn0 : (n:ℝ) ≠ 0 := by
      intro h0; rw [h0] at hn; simp at hn; linarith
    apply irrational_pi
    refine ⟨(e : ℚ) / (2 * n), ?_⟩
    push_cast
    field_simp
    linarith

lemma meanCos_lt_one (e : ℕ) (he : 1 ≤ e) : meanCos e < 1 := by
  unfold meanCos
  rw [div_lt_one (by norm_num [PAIRS])]
  simp only [PAIRS]
  rw [Finset.sum_range_succ']
  have h1 : ∑ i ∈ range 7, Real.cos (e * turn (i + 1)) ≤ ∑ i ∈ range 7, (1:ℝ) :=
    Finset.sum_le_sum (fun i _ => Real.cos_le_one _)
  have h2 := cos_nat_lt_one e he
  simp only [turn_zero, mul_one]
  simp at h1
  push_cast
  linarith

lemma runnerUp_lt_one : runnerUp < 1 := by
  unfold runnerUp
  refine (Finset.sup'_lt_iff (H := ⟨1, by simp [CONTEXT]⟩)).2 ?_
  intro e he
  simp at he
  exact meanCos_lt_one e he.1

lemma meanCos_one_pos : 0 < meanCos 1 := by
  unfold meanCos
  apply div_pos _ (by norm_num [PAIRS])
  apply Finset.sum_pos
  · intro i _
    apply Real.cos_pos_of_mem_Ioo
    have h1 := turn_pos i
    have h2 := turn_le_one i
    have h3 := Real.pi_gt_three
    constructor <;> push_cast <;> nlinarith
  · simp [PAIRS]

lemma runnerUp_pos : 0 < runnerUp := by
  have : meanCos 1 ≤ runnerUp := by
    unfold runnerUp
    exact Finset.le_sup' meanCos (by simp [CONTEXT])
  linarith [meanCos_one_pos]

lemma attnScale_pos : 0 < attnScale := by
  unfold attnScale MARGIN
  have := runnerUp_lt_one
  apply div_pos (by norm_num)
  simp [PAIRS]; linarith

lemma attnScale_margin : 8 * attnScale * (1 - runnerUp) = 30 := by
  unfold attnScale MARGIN
  have := runnerUp_lt_one
  have h : (1 - runnerUp) ≠ 0 := by linarith
  simp only [PAIRS]
  field_simp
  norm_num

lemma meanCos_le_runnerUp (e : ℕ) (h1 : 1 ≤ e) (h2 : e ≤ 511) : meanCos e ≤ runnerUp := by
  unfold runnerUp
  exact Finset.le_sup' meanCos (by simp [CONTEXT]; omega)

lemma sum_cos_le (x : ℝ) (n : ℕ) (hx : x = n ∨ x = -n) (h1 : 1 ≤ n) (h2 : n ≤ 511) :
    ∑ i ∈ range 8, Real.cos (x * turn i) ≤ 8 * runnerUp := by
  have h := meanCos_le_runnerUp n h1 h2
  unfold meanCos at h
  simp only [PAIRS] at h
  have e : ∑ i ∈ range 8, Real.cos (x * turn i) = ∑ i ∈ range 8, Real.cos (n * turn i) := by
    rcases hx with hx | hx
    · rw [hx]
    · rw [hx]; congr 1; ext i; rw [neg_mul, Real.cos_neg]
  rw [e]
  push_cast at h
  linarith

lemma exp_bound : 1024 * Real.exp (-30) < 1 / 10 ^ 10 := by
  have h1 := Real.exp_one_gt_d9
  have h2 : Real.exp 30 = Real.exp 1 ^ 30 := by
    rw [← Real.exp_nat_mul]; norm_num
  have h3 : (2.7182818283 : ℝ) ^ 30 < Real.exp 1 ^ 30 := by
    apply pow_lt_pow_left₀ h1 (by norm_num) (by norm_num)
  have h4 : (1024 * 10 ^ 10 : ℝ) < (2.7182818283 : ℝ) ^ 30 := by norm_num
  rw [Real.exp_neg]
  rw [← h2] at h3
  have h5 : 0 < Real.exp 30 := Real.exp_pos _
  rw [mul_inv_lt_iff₀ h5]
  nlinarith

end

end AttnRealPf
end Reifier

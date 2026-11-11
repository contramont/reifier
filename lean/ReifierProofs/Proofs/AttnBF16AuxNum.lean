import ReifierProofs.Transformer

namespace Reifier
namespace AttnBF16Num
open Finset Real

lemma turn_pos (i : ℕ) : 0 < turn i := by
  unfold turn; positivity

lemma turn_sq (i : ℕ) : turn i ^ 2 = 1 / 10 ^ i := by
  unfold turn PAIRS
  have h1 : (10000 : ℝ) = 10 ^ (4 : ℝ) := by norm_num
  rw [h1, ← Real.rpow_mul (by norm_num), ← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
  have : (4 * -((i : ℝ) / ((8 : ℕ) : ℝ)) * ((2 : ℕ) : ℝ)) = -(i : ℝ) := by push_cast; ring
  rw [this, Real.rpow_neg (by norm_num), Real.rpow_natCast]
  simp

lemma turn_le_one (i : ℕ) : turn i ≤ 1 := by
  have h := turn_sq i
  have hp := turn_pos i
  have : (1:ℝ) / 10 ^ i ≤ 1 := by
    rw [div_le_one (by positivity)]; exact one_le_pow₀ (by norm_num)
  nlinarith

lemma turn0 : turn 0 = 1 := by
  unfold turn; simp

lemma turn2 : turn 2 = 1 / 10 := by
  have h := turn_sq 2; have hp := turn_pos 2
  norm_num at h
  nlinarith

lemma turn4 : turn 4 = 1 / 100 := by
  have h := turn_sq 4; have hp := turn_pos 4
  norm_num at h
  nlinarith

lemma turn1_bounds : 0.3162 ≤ turn 1 ∧ turn 1 ≤ 0.3163 := by
  have h := turn_sq 1; have hp := turn_pos 1
  norm_num at h
  constructor <;> nlinarith

lemma turn3_bounds : 0.031622 ≤ turn 3 ∧ turn 3 ≤ 0.031623 := by
  have h := turn_sq 3; have hp := turn_pos 3
  norm_num at h
  constructor <;> nlinarith


lemma cos_up {x : ℝ} (hx : |x| ≤ 1) : Real.cos x ≤ 1 - x ^ 2 / 2 + x ^ 4 * (5 / 96) := by
  have h := Real.cos_bound hx
  have h4 : |x| ^ 4 = x ^ 4 := by
    rw [show (4:ℕ) = 2 * 2 from rfl, pow_mul, sq_abs, ← pow_mul]
  rw [h4] at h
  have := (abs_le.mp h).2
  linarith

lemma cos_low {x : ℝ} (hx : |x| ≤ 1) : 1 - x ^ 2 / 2 - x ^ 4 * (5 / 96) ≤ Real.cos x := by
  have h := Real.cos_bound hx
  have h4 : |x| ^ 4 = x ^ 4 := by
    rw [show (4:ℕ) = 2 * 2 from rfl, pow_mul, sq_abs, ← pow_mul]
  rw [h4] at h
  have := (abs_le.mp h).1
  linarith

lemma cos_103 : Real.cos 1.03 ≤ 0.52 := by
  have h2 : (1.03 : ℝ) = 2 * 0.515 := by norm_num
  rw [h2, Real.cos_two_mul]
  have habs : |(0.515 : ℝ)| ≤ 1 := by rw [abs_of_pos (by norm_num)]; norm_num
  have hu := cos_up habs
  have hl := cos_low habs
  have hc0 : 0 ≤ Real.cos 0.515 := by nlinarith
  have hcu : Real.cos 0.515 ≤ 0.8710514 := by nlinarith
  nlinarith

lemma cos_le_052 {x : ℝ} (h1 : 1.03 ≤ x) (h2 : x ≤ 5.25) : Real.cos x ≤ 0.52 := by
  have hpi := Real.pi_gt_d2
  rcases le_or_gt x Real.pi with hx | hx
  · calc Real.cos x ≤ Real.cos 1.03 :=
          Real.cos_le_cos_of_nonneg_of_le_pi (by norm_num) hx h1
      _ ≤ 0.52 := cos_103
  · rw [← Real.cos_two_pi_sub]
    calc Real.cos (2 * Real.pi - x) ≤ Real.cos 1.03 :=
          Real.cos_le_cos_of_nonneg_of_le_pi (by norm_num) (by linarith) (by linarith)
      _ ≤ 0.52 := cos_103

lemma sum8_le (f : ℕ → ℝ) (hf : ∀ i, f i ≤ 1) (i0 : ℕ) (hi0 : i0 < 8) (h0 : f i0 ≤ 0.52) :
    ∑ i ∈ range 8, f i ≤ 7.52 := by
  have : ∑ i ∈ range 8, f i ≤ ∑ i ∈ range 8, (if i = i0 then (0.52:ℝ) else 1) := by
    apply Finset.sum_le_sum
    intro i _
    split_ifs with h
    · subst h; exact h0
    · exact hf i
  refine this.trans (le_of_eq ?_)
  interval_cases i0 <;> simp [Finset.sum_range_succ] <;> norm_num

lemma meanCos_one : meanCos 1 ≤ 0.94 := by
  unfold meanCos PAIRS
  simp only [Nat.cast_one, one_mul]
  have t1 := turn1_bounds
  have t3 := turn3_bounds
  have s1 := turn_sq 1
  have s3 := turn_sq 3
  have p1 := turn_pos 1
  have p3 := turn_pos 3
  have c0 : Real.cos (turn 0) ≤ 1 - 1 / 2 + 5 / 96 := by
    rw [turn0]; have := cos_up (x := 1) (by simp); norm_num at this ⊢; linarith
  have c1 : Real.cos (turn 1) ≤ 1 - 1 / 20 + (1 / 100) * (5 / 96) := by
    have := cos_up (x := turn 1) (by rw [abs_of_pos p1]; linarith)
    have h4 : turn 1 ^ 4 = (turn 1 ^ 2) ^ 2 := by ring
    rw [h4, s1] at this; norm_num at this ⊢; linarith
  have c2 : Real.cos (turn 2) ≤ 1 - 1 / 200 + (1 / 10000) * (5 / 96) := by
    rw [turn2]; have := cos_up (x := 1 / 10) (by rw [abs_of_pos (by norm_num)]; norm_num)
    norm_num at this ⊢; linarith
  have c3 : Real.cos (turn 3) ≤ 1 - 1 / 2000 + (1 / 1000000) * (5 / 96) := by
    have := cos_up (x := turn 3) (by rw [abs_of_pos p3]; linarith)
    have h4 : turn 3 ^ 4 = (turn 3 ^ 2) ^ 2 := by ring
    rw [h4, s3] at this; norm_num at this ⊢; linarith
  have c4 := Real.cos_le_one (turn 4)
  have c5 := Real.cos_le_one (turn 5)
  have c6 := Real.cos_le_one (turn 6)
  have c7 := Real.cos_le_one (turn 7)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero]
  push_cast
  rw [div_le_iff₀ (by norm_num)]
  linarith

lemma meanCos_ge2 (e : ℕ) (h2 : 2 ≤ e) (h5 : e ≤ 511) : meanCos e ≤ 0.94 := by
  unfold meanCos PAIRS
  push_cast
  rw [div_le_iff₀ (by norm_num)]
  have t1 := turn1_bounds
  have t3 := turn3_bounds
  have he2 : (2:ℝ) ≤ e := by exact_mod_cast h2
  have he5 : (e:ℝ) ≤ 511 := by exact_mod_cast h5
  have key : ∃ i0 < 8, Real.cos (e * turn i0) ≤ 0.52 := by
    by_cases a : e ≤ 5
    · refine ⟨0, by norm_num, cos_le_052 ?_ ?_⟩ <;> rw [turn0]
      · linarith
      · have : (e:ℝ) ≤ 5 := by exact_mod_cast a
        linarith
    by_cases b : e ≤ 16
    · have ha : (6:ℝ) ≤ e := by exact_mod_cast (by omega : 6 ≤ e)
      have hb : (e:ℝ) ≤ 16 := by exact_mod_cast b
      refine ⟨1, by norm_num, cos_le_052 ?_ ?_⟩ <;> nlinarith
    by_cases c : e ≤ 52
    · have ha : (17:ℝ) ≤ e := by exact_mod_cast (by omega : 17 ≤ e)
      have hb : (e:ℝ) ≤ 52 := by exact_mod_cast c
      refine ⟨2, by norm_num, cos_le_052 ?_ ?_⟩ <;> rw [turn2] <;> linarith
    by_cases d : e ≤ 102
    · have ha : (53:ℝ) ≤ e := by exact_mod_cast (by omega : 53 ≤ e)
      have hb : (e:ℝ) ≤ 102 := by exact_mod_cast d
      refine ⟨3, by norm_num, cos_le_052 ?_ ?_⟩ <;> nlinarith
    · have ha : (103:ℝ) ≤ e := by exact_mod_cast (by omega : 103 ≤ e)
      refine ⟨4, by norm_num, cos_le_052 ?_ ?_⟩ <;> rw [turn4] <;> linarith
  obtain ⟨i0, hi0, hc⟩ := key
  have := sum8_le (fun i => Real.cos (e * turn i)) (fun i => Real.cos_le_one _) i0 hi0 hc
  linarith

lemma meanCos_le (e : ℕ) (h1 : 1 ≤ e) (h5 : e ≤ 511) : meanCos e ≤ 0.94 := by
  rcases Nat.eq_or_lt_of_le h1 with h | h
  · subst h; exact_mod_cast meanCos_one
  · exact meanCos_ge2 e h h5

lemma meanCos_one_nonneg : 0 ≤ meanCos 1 := by
  unfold meanCos
  apply div_nonneg _ (by positivity)
  apply Finset.sum_nonneg
  intro i _
  apply le_of_lt
  apply Real.cos_pos_of_mem_Ioo
  have := turn_pos i; have := turn_le_one i; have := Real.pi_gt_d2
  constructor <;> simp <;> linarith

lemma runnerUp_le : runnerUp ≤ 0.94 := by
  unfold runnerUp
  apply Finset.sup'_le
  intro e he
  simp only [Finset.mem_Icc, CONTEXT] at he
  exact meanCos_le e he.1 (by omega)

lemma runnerUp_nonneg : 0 ≤ runnerUp := by
  have : meanCos 1 ≤ runnerUp := by
    unfold runnerUp
    exact Finset.le_sup' meanCos (by simp [CONTEXT])
  linarith [meanCos_one_nonneg]

lemma meanCos_le_runnerUp (e : ℕ) (h1 : 1 ≤ e) (h5 : e ≤ 511) : meanCos e ≤ runnerUp := by
  unfold runnerUp
  exact Finset.le_sup' meanCos (by simp [CONTEXT]; omega)

lemma attnScale_le : attnScale ≤ 62.5 := by
  unfold attnScale MARGIN PAIRS
  have := runnerUp_le
  rw [div_le_iff₀ (by push_cast; nlinarith)]
  push_cast; nlinarith

lemma attnScale_ge : 3.75 ≤ attnScale := by
  unfold attnScale MARGIN PAIRS
  have := runnerUp_le
  have := runnerUp_nonneg
  rw [le_div_iff₀ (by push_cast; nlinarith)]
  push_cast; nlinarith

lemma attnScale_gap : 8 * attnScale * (1 - runnerUp) = 30 := by
  unfold attnScale MARGIN PAIRS
  have := runnerUp_le
  have h : (1 - runnerUp) ≠ 0 := by intro h; linarith
  push_cast
  field_simp

end AttnBF16Num
end Reifier

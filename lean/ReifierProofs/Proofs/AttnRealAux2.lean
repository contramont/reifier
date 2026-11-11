import ReifierProofs.Proofs.AttnRealAux1

namespace Reifier
namespace AttnRealPf

open Finset

noncomputable section

lemma softmax_close (S : Finset ℕ) (s v : ℕ → ℝ) (t0 : ℕ) (ht0 : t0 ∈ S)
    (hs : ∀ q ∈ S, q ≠ t0 → s q ≤ s t0 - 30) (hv : ∀ q ∈ S, |v q - v t0| ≤ 2)
    (hcard : S.card ≤ 512) :
    |∑ q ∈ S, Real.exp (s q) / (∑ r ∈ S, Real.exp (s r)) * v q - v t0|
      ≤ 1024 * Real.exp (-30) := by
  set Z := ∑ r ∈ S, Real.exp (s r) with hZ
  have hZpos : 0 < Z := Finset.sum_pos (fun r _ => Real.exp_pos _) ⟨t0, ht0⟩
  have ht0Z : Real.exp (s t0) ≤ Z :=
    Finset.single_le_sum (f := fun r => Real.exp (s r)) (fun r _ => (Real.exp_pos _).le) ht0
  have hsum1 : ∑ q ∈ S, Real.exp (s q) / Z = 1 := by
    rw [← Finset.sum_div]; exact div_self hZpos.ne'
  have e1 : ∑ q ∈ S, Real.exp (s q) / Z * v q - v t0
      = ∑ q ∈ S, Real.exp (s q) / Z * (v q - v t0) := by
    have : v t0 = ∑ q ∈ S, Real.exp (s q) / Z * v t0 := by
      rw [← Finset.sum_mul, hsum1, one_mul]
    conv_lhs => rw [this]
    rw [← Finset.sum_sub_distrib]
    congr 1; ext q; ring
  rw [e1]
  have hterm : ∀ q ∈ S, |Real.exp (s q) / Z * (v q - v t0)| ≤ 2 * Real.exp (-30) := by
    intro q hq
    rw [abs_mul, abs_of_pos (div_pos (Real.exp_pos _) hZpos)]
    by_cases hqt : q = t0
    · subst hqt; simp; positivity
    · have h1 : Real.exp (s q) / Z ≤ Real.exp (-30) := by
        rw [div_le_iff₀ hZpos]
        calc Real.exp (s q) ≤ Real.exp (s t0 - 30) := Real.exp_le_exp.2 (hs q hq hqt)
          _ = Real.exp (-30) * Real.exp (s t0) := by rw [← Real.exp_add]; ring_nf
          _ ≤ Real.exp (-30) * Z := by gcongr
      have h2 := hv q hq
      have h3 : 0 ≤ Real.exp (s q) / Z := (div_pos (Real.exp_pos _) hZpos).le
      calc Real.exp (s q) / Z * |v q - v t0| ≤ Real.exp (-30) * 2 := by
            apply mul_le_mul h1 h2 (abs_nonneg _) (Real.exp_pos _).le
        _ = 2 * Real.exp (-30) := by ring
  calc |∑ q ∈ S, Real.exp (s q) / Z * (v q - v t0)|
      ≤ ∑ q ∈ S, |Real.exp (s q) / Z * (v q - v t0)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ q ∈ S, 2 * Real.exp (-30) := Finset.sum_le_sum hterm
    _ = S.card * (2 * Real.exp (-30)) := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ 512 * (2 * Real.exp (-30)) := by
        gcongr
        exact_mod_cast hcard
    _ = 1024 * Real.exp (-30) := by ring

lemma mem_visible {p p' : ℕ} : p' ∈ visible p ↔ p' ≤ p ∧ p - p' < 512 := by
  unfold visible
  rw [Finset.mem_filter, Finset.mem_range]
  simp only [CONTEXT]
  omega

lemma visible_card (p : ℕ) : (visible p).card ≤ 512 := by
  have : (visible p).card ≤ (range 512).card := by
    apply Finset.card_le_card_of_injOn (fun p' => p - p')
    · intro a ha
      rw [Finset.mem_coe, mem_visible] at ha
      rw [Finset.mem_coe, Finset.mem_range]; show p - a < 512; omega
    · intro a ha b hb hab
      rw [Finset.mem_coe, mem_visible] at ha hb
      have hab' : p - a = p - b := hab
      omega
  simpa using this

end

end AttnRealPf
end Reifier

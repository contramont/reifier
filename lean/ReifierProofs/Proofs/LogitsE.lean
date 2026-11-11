import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace DeepAux

lemma logE_mem_of_lookup {l : List (ℕ × List Bool)} {k : ℕ} {c : List Bool}
    (h : l.lookup k = some c) : (k, c) ∈ l := by
  obtain ⟨l₁, l₂, rfl, -⟩ := List.lookup_eq_some_iff.mp h
  simp

/-- A bit held as `-1` or `1`. -/
def logE_sgn (β : Bool) : ℝ := if β then 1 else -1

lemma logE_sum_split (W b : ℕ) (hW : b + 1 ≤ W) (F : ℕ → ℝ)
    (hF : ∀ f, b < f → f < W → F f = 0) :
    ∑ f ∈ range W, F f = F 0 + ∑ i ∈ range b, F (1 + i) := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hW
  rw [Finset.sum_range_add, Finset.sum_range_succ']
  have h0 : ∑ x ∈ range k, F (b + 1 + x) = 0 :=
    Finset.sum_eq_zero (fun x hx => hF _ (by omega) (by simp at hx; omega))
  rw [h0, add_zero, add_comm]
  congr 1
  exact Finset.sum_congr rfl (fun i _ => by rw [add_comm])

lemma logE_exists_diff {t : Trace} {codes : List (ℕ × List Bool)} (hcodes : CodesOK t codes)
    {τ τ' : ℕ} {c c' : List Bool} (h1 : codes.lookup τ = some c) (h2 : codes.lookup τ' = some c')
    (hne : τ' ≠ τ) : ∃ i < t.outputs.length, c.getD i false ≠ c'.getD i false := by
  by_contra hcon
  push Not at hcon
  have m1 := logE_mem_of_lookup h1
  have m2 := logE_mem_of_lookup h2
  have l1 := hcodes.length _ m1
  have l2 := hcodes.length _ m2
  simp only at l1 l2
  have hcc : c = c' := List.ext_getElem (by omega) (fun i h₁ h₂ => by
    have := hcon i (by omega)
    rwa [List.getD_eq_getElem _ _ h₁, List.getD_eq_getElem _ _ h₂] at this)
  exact hne (hcodes.inj _ m2 _ m1 (by simp [hcc]))

lemma logE_getLast?_getD_eq {α : Type*} (l : List α) (hl : l ≠ []) (d : α) :
    l.getLast?.getD d = l.getLast hl := by
  rw [List.getLast?_eq_some_getLast hl]; rfl

lemma logE_embed_some {t : Trace} {codes : List (ℕ × List Bool)} {τ : ℕ} {c : List Bool}
    (h : codes.lookup τ = some c) (f : ℕ) :
    (transformer t codes).embed τ f =
      if f = 0 then 2 * Real.sqrt (tokenGrid t).width
      else if f ≤ t.outputs.length then logE_sgn (c.getD (f - 1) false) else 0 := by
  simp only [transformer, h, logE_sgn]
  rfl

lemma logE_embed_none {t : Trace} {codes : List (ℕ × List Bool)} {τ : ℕ}
    (h : codes.lookup τ = none) (f : ℕ) : (transformer t codes).embed τ f = 0 := by
  simp only [transformer, h]

lemma logE_mul_sgn_ge (E x : ℝ) (β : Bool) (h : |x - logE_sgn β| ≤ E) :
    1 - E ≤ x * logE_sgn β := by
  rw [abs_le] at h
  cases β <;> simp only [logE_sgn, Bool.false_eq_true, ite_false, ite_true] at h ⊢ <;>
    linarith [h.1, h.2]

lemma logE_diff_nonneg (E x : ℝ) (hE : E ≤ 1) (β β' : Bool) (h : |x - logE_sgn β| ≤ E) :
    0 ≤ x * logE_sgn β - x * logE_sgn β' := by
  rw [abs_le] at h
  cases β <;> cases β' <;> simp only [logE_sgn, Bool.false_eq_true, ite_false, ite_true] at h ⊢ <;>
    linarith [h.1, h.2]

lemma logE_diff_pos (E x : ℝ) (hE : E < 1) (β β' : Bool) (h : |x - logE_sgn β| ≤ E)
    (hne : β ≠ β') :
    0 < x * logE_sgn β - x * logE_sgn β' := by
  rw [abs_le] at h
  cases β <;> cases β' <;> simp only [logE_sgn, Bool.false_eq_true, ite_false, ite_true, ne_eq,
    not_true_eq_false] at h hne ⊢ <;> linarith [h.1, h.2]

/-- The real logit of a coded token. -/
lemma logE_logit_coded (t : Trace) (codes : List (ℕ × List Bool)) (x : ℕ → ℝ) (τ : ℕ)
    (c : List Bool) (k : ℝ) (hW : t.outputs.length + 1 ≤ (tokenGrid t).width)
    (hk : ∀ f, rmsnorm (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (2 * Real.sqrt (tokenGrid t).width) x f = if f = 0 then k else k * x f)
    (h : codes.lookup τ = some c) :
    logitOf (transformer t codes) x τ = k * (2 * Real.sqrt (tokenGrid t).width) +
      k * ∑ i ∈ range t.outputs.length, x (1 + i) * logE_sgn (c.getD i false) := by
  show ∑ f ∈ range (tokenGrid t).width,
    rmsnorm (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (2 * Real.sqrt (tokenGrid t).width) x f * (transformer t codes).embed τ f = _
  rw [logE_sum_split _ _ hW]
  · rw [hk, logE_embed_some h, Finset.mul_sum]
    simp only [ite_true]
    congr 1
    refine Finset.sum_congr rfl (fun i hi => ?_)
    simp at hi
    rw [hk, logE_embed_some h]
    simp only [show 1 + i ≠ 0 by omega, ite_false, show 1 + i ≤ t.outputs.length by omega,
      ite_true, show 1 + i - 1 = i by omega]
    ring
  · intro f hf1 hf2
    rw [logE_embed_some h]
    simp only [show f ≠ 0 by omega, ite_false, show ¬ f ≤ t.outputs.length by omega, mul_zero]

lemma logE_logit_uncoded (t : Trace) (codes : List (ℕ × List Bool)) (x : ℕ → ℝ) (τ : ℕ)
    (h : codes.lookup τ = none) : logitOf (transformer t codes) x τ = 0 := by
  show ∑ f ∈ range (tokenGrid t).width,
    rmsnorm (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (2 * Real.sqrt (tokenGrid t).width) x f * (transformer t codes).embed τ f = _
  simp only [logE_embed_none h, mul_zero, Finset.sum_const_zero]

end DeepAux

open DeepAux in
theorem logit_real_E_of (hg : Stmt_grid) (hn : Stmt_norm_close) : Stmt_logit_real_E := by
  intro t codes val x τ E hprog hcodes hval hE0 hE1 M hclose hcode τ' hτ' hne
  have hG := hg t hprog
  have hW : t.outputs.length + 1 ≤ (tokenGrid t).width := hG.row0.width
  have hlast : ((tokenGrid t).rows.getLast?.getD fun _ => none) =
      (tokenGrid t).rows.getLast hG.rows_ne := logE_getLast?_getD_eq _ _ _
  have hrow : RowOK t (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none) := by
    rw [hlast]; exact hG.rows_ok _ (List.getLast_mem _)
  have hlast1 : ∀ i < t.outputs.length, ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
      some (.sig (t.outputs.getD i 0)) := by
    rw [hlast]; exact hG.last
  obtain ⟨k, hk0, -, hk⟩ := hn t (tokenGrid t).width _ val x E hrow hval
    hprog.toWellFormed (by omega) hE0 hE1 hclose.one_eq hclose.near
  obtain ⟨cτ, hcτ, hcode⟩ := hcode
  have hEl : E < 1 := lt_of_le_of_lt hE1 (by norm_num)
  have hx : ∀ i < t.outputs.length, |x (1 + i) - logE_sgn (cτ.getD i false)| ≤ E := by
    intro i hi
    have h := hclose.near (1 + i) (by omega) (by show 1 + i < (tokenGrid t).width; omega)
    have hid : ideal M.net.one val ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
        logE_sgn (cτ.getD i false) := by
      unfold ideal; rw [hlast1 i hi]; simp only; rw [← hcode i hi]; unfold logE_sgn
      split_ifs <;> norm_num
    rwa [hid] at h
  have hpos : 0 < 2 * Real.sqrt (tokenGrid t).width := by
    have : (0 : ℝ) < (tokenGrid t).width := by
      exact_mod_cast (show 0 < (tokenGrid t).width by omega)
    positivity
  rw [logE_logit_coded t codes x τ cτ k hW hk hcτ]
  rcases hτ'c : codes.lookup τ' with _ | c'
  · rw [logE_logit_uncoded t codes x τ' hτ'c]
    have : 0 ≤ ∑ i ∈ range t.outputs.length, x (1 + i) * logE_sgn (cτ.getD i false) :=
      Finset.sum_nonneg (fun i hi => by
        simp at hi; have := logE_mul_sgn_ge _ _ _ (hx i hi); linarith)
    positivity
  · rw [logE_logit_coded t codes x τ' c' k hW hk hτ'c]
    obtain ⟨i, hi, hdi⟩ := logE_exists_diff hcodes hcτ hτ'c hne
    have : 0 < ∑ i ∈ range t.outputs.length,
        (x (1 + i) * logE_sgn (cτ.getD i false) - x (1 + i) * logE_sgn (c'.getD i false)) :=
      Finset.sum_pos' (fun j hj => by
          simp at hj; exact logE_diff_nonneg _ _ hEl.le _ _ (hx j hj))
        ⟨i, by simpa using hi, logE_diff_pos _ _ hEl _ _ (hx i hi) hdi⟩
    rw [Finset.sum_sub_distrib] at this
    have := mul_pos hk0 this
    nlinarith

end

end Reifier

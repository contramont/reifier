import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace DeepAux

/-- Folding a step over each row and the layer between it and the next keeps an invariant,
indexed by depth, that each such step keeps from depth `d` to `d + 1`. -/
theorem asm_foldl_pairs_idx_cons {α β γ : Type*} (G : α → β → γ → γ) (F : α × α → β)
    (P : ℕ → α → γ → Prop) :
    ∀ (rs : List α) (n : ℕ) (r : α) (x : γ), P n r x →
    (∀ (d : ℕ) (hd : d + 1 < (r :: rs).length) (y : γ), P (n + d) ((r :: rs)[d]) y →
        P (n + d + 1) ((r :: rs)[d + 1])
          (G (r :: rs)[d] (F ((r :: rs)[d], (r :: rs)[d + 1])) y)) →
    P (n + rs.length) ((r :: rs).getLast (List.cons_ne_nil _ _))
      (((r :: rs).zip (((r :: rs).zip (r :: rs).tail).map F)).foldl
        (fun y bl => G bl.1 bl.2 y) x) := by
  intro rs
  induction rs with
  | nil => intro n r x h0 _; simpa using h0
  | cons r' rs ih =>
    intro n r x h0 hstep
    have h1 := hstep 0 (by simp) x (by simpa using h0)
    simp only [List.getElem_cons_zero, List.getElem_cons_succ, add_zero] at h1
    have := ih (n + 1) r' (G r (F (r, r')) x) h1 (fun d hd y hy => by
      have := hstep (d + 1) (by simp at hd ⊢; omega) y
        (by rw [show n + (d + 1) = n + 1 + d by omega]; simpa using hy)
      rw [show n + (d + 1) + 1 = n + 1 + d + 1 by omega] at this
      simpa using this)
    rw [show n + (r' :: rs).length = n + 1 + rs.length by simp; omega]
    simpa [List.getLast_cons_cons] using this

theorem asm_foldl_pairs_idx {α β γ : Type*} (G : α → β → γ → γ) (F : α × α → β)
    (P : ℕ → α → γ → Prop) (l : List α) (hl : l ≠ []) (dflt : α) (x : γ)
    (h0 : P 0 (l.headD dflt) x)
    (hstep : ∀ (d : ℕ) (hd : d + 1 < l.length) (y : γ), P d l[d] y →
        P (d + 1) l[d + 1] (G l[d] (F (l[d], l[d + 1])) y)) :
    P (l.length - 1) (l.getLast?.getD dflt)
      ((l.zip ((l.zip l.tail).map F)).foldl (fun y bl => G bl.1 bl.2 y) x) := by
  cases l with
  | nil => exact absurd rfl hl
  | cons r rs =>
    rw [List.getLast?_eq_some_getLast (List.cons_ne_nil _ _)]
    have := asm_foldl_pairs_idx_cons G F P rs 0 r x h0
      (fun d hd y hy => by
        rw [zero_add] at hy ⊢
        exact hstep d hd y hy)
    simpa using this

lemma asm_valOK_valAt (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) :
    ValOK t (valAt t codes toks p) := by
  refine ⟨⟨windowBits t codes toks p, fun s => rfl⟩, fun i => ?_⟩
  show IsBit (if ((codes.lookup (toks p)).getD []).getD i false then 1 else 0)
  split_ifs
  · right; rfl
  · left; rfl

lemma asm_getLast?_getD_eq {α : Type*} (l : List α) (hl : l ≠ []) (d : α) :
    l.getLast?.getD d = l.getLast hl := by
  rw [List.getLast?_eq_some_getLast hl]; rfl

lemma asm_ideal_last (t : Trace) (hG : GridOK t (tokenGrid t)) (one : ℝ) (val : Name → ℝ)
    (i : ℕ) (hi : i < t.outputs.length) :
    ideal one val ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
      2 * val (.sig (t.outputs.getD i 0)) - 1 := by
  unfold ideal
  rw [asm_getLast?_getD_eq _ hG.rows_ne, hG.last i hi]

/-- The error bound at depth `d`. -/
lemma asm_err_mono (σ : ℝ) (hσ : 0 ≤ σ) {d D : ℕ} (hd : d ≤ D) :
    (1 + σ) ^ d / 10 ^ 10 ≤ (1 + σ) ^ D / 10 ^ 10 := by
  have : (1 + σ) ^ d ≤ (1 + σ) ^ D := pow_le_pow_right₀ (by linarith) hd
  linarith [show (0:ℝ) < 10 ^ 10 by positivity, div_le_div_of_nonneg_right this
    (show (0:ℝ) ≤ 10 ^ 10 by positivity)]

lemma asm_err_ge (σ : ℝ) (hσ : 0 ≤ σ) (d : ℕ) : 1 / 10 ^ 10 ≤ (1 + σ) ^ d / 10 ^ 10 := by
  have : (1 : ℝ) ≤ (1 + σ) ^ d := one_le_pow₀ (by linarith)
  exact div_le_div_of_nonneg_right this (by positivity)

lemma asm_err_le (σ : ℝ) (D : ℕ) (hD : (1 + σ) ^ D ≤ 10 ^ 4) :
    (1 + σ) ^ D / 10 ^ 10 ≤ 1 / 10 ^ 6 := by
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith

lemma asm_err_succ (σ : ℝ) (d : ℕ) :
    (1 + σ) * ((1 + σ) ^ d / 10 ^ 10) = (1 + σ) ^ (d + 1) / 10 ^ 10 := by
  rw [pow_succ]; ring

end DeepAux

-- `Stmt_trace` is not needed: `valAt` is defined from `Trace.values` directly.
set_option linter.unusedVariables false in
open DeepAux in
theorem mainRealDeep_of (hg : Stmt_grid) (ht : Stmt_trace) (hl : Stmt_layer_gain)
    (ha : Stmt_attn_real) (hlog : Stmt_logit_real_E) : MainRealDeep := by
  intro σ t codes toks p hprog hcore hσ hdepth hcodes hpos M x
  have hG := hg t hprog
  have hW : t.outputs.length + 1 ≤ (tokenGrid t).width := hG.row0.width
  have hval := asm_valOK_valAt t codes toks p
  set D := (tokenGrid t).rows.length - 1 with hDdef
  have hED : (1 + σ) ^ D / 10 ^ 10 ≤ 1 / 10 ^ 6 := asm_err_le σ D hdepth
  have hfin : Close (tokenGrid t).width ((1 + σ) ^ D / 10 ^ 10)
      (2 * Real.sqrt (tokenGrid t).width)
      (valAt t codes toks p) ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (M.final toks p) := by
    have := asm_foldl_pairs_idx
      (fun before us y =>
        netStep gelu (tokenGrid t).width (2 * Real.sqrt (tokenGrid t).width) before us y)
      (fun br => layerUnits (tokenGrid t).width t.mlps br.1 br.2)
      (fun d row y => Close (tokenGrid t).width ((1 + σ) ^ d / 10 ^ 10)
        (2 * Real.sqrt (tokenGrid t).width) (valAt t codes toks p) row y)
      (tokenGrid t).rows hG.rows_ne (fun _ => none) (M.afterAttn toks p) ?_ ?_
    · exact this
    · have h := ha t codes toks p hprog hcodes hpos
      simp only [pow_zero]
      exact h
    · intro d hd y hy
      have hdD : d + 1 ≤ D := by omega
      have hEd1 : (1 + σ) ^ (d + 1) / 10 ^ 10 ≤ 1 / 10 ^ 6 :=
        (asm_err_mono σ hσ hdD).trans hED
      have hEd : (1 + σ) ^ d / 10 ^ 10 ≤ 1 / 10 ^ 6 :=
        (asm_err_mono σ hσ (by omega : d ≤ D)).trans hED
      have hE0 : 0 ≤ (1 + σ) ^ d / 10 ^ 10 := by positivity
      have hG' : (10 : ℝ) ^ 5 * ((1 + σ) ^ d / 10 ^ 10) ≤ 1 / 10 := by
        nlinarith
      have hσE : σ * ((1 + σ) ^ d / 10 ^ 10) ≤ 1 / 100 := by
        have h1 : σ * ((1 + σ) ^ d / 10 ^ 10) ≤ (1 + σ) * ((1 + σ) ^ d / 10 ^ 10) := by
          nlinarith
        rw [asm_err_succ] at h1
        linarith
      have := hl t _ _ _ _ y _ (10 ^ 5) σ hprog.toWellFormed hcore (hG.pairs d hd) hval
        (by omega) (asm_err_ge σ hσ d) hEd (by norm_num) hG' hσ hσE hy
      rw [asm_err_succ] at this
      exact this
  refine ⟨fun i hi => ?_, fun τ hcode τ' hτ' hne => ?_⟩
  · have h := hfin.near (1 + i) (by omega) (by omega)
    rw [asm_ideal_last t hG _ _ i hi] at h
    exact h
  · exact hlog t codes _ (M.final toks p) τ ((1 + σ) ^ D / 10 ^ 10) hprog hcodes hval
      (by positivity) hED hfin hcode τ' hτ' hne

end

end Reifier

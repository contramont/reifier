import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace AssemblyAux

/-- Folding a step over each row and the layer between it and the next keeps an invariant that
each such step keeps. -/
theorem foldl_pairs_cons {α β γ : Type*} (G : α → β → γ → γ) (F : α × α → β) (P : α → γ → Prop) :
    ∀ (rs : List α) (r : α) (x : γ), P r x →
    (∀ (d : ℕ) (hd : d + 1 < (r :: rs).length) (y : γ), P ((r :: rs)[d]) y →
        P ((r :: rs)[d + 1]) (G (r :: rs)[d] (F ((r :: rs)[d], (r :: rs)[d + 1])) y)) →
    P ((r :: rs).getLast (List.cons_ne_nil _ _))
      (((r :: rs).zip (((r :: rs).zip (r :: rs).tail).map F)).foldl
        (fun y bl => G bl.1 bl.2 y) x) := by
  intro rs
  induction rs with
  | nil => intro r x h0 _; simpa using h0
  | cons r' rs ih =>
    intro r x h0 hstep
    have h1 := hstep 0 (by simp) x h0
    simp only [List.getElem_cons_zero, List.getElem_cons_succ] at h1
    have := ih r' (G r (F (r, r')) x) h1 (fun d hd y hy => by
      have := hstep (d + 1) (by simp at hd ⊢; omega) y (by simpa using hy)
      simpa using this)
    simpa [List.getLast_cons_cons] using this

theorem foldl_pairs {α β γ : Type*} (G : α → β → γ → γ) (F : α × α → β) (P : α → γ → Prop)
    (l : List α) (hl : l ≠ []) (dflt : α) (x : γ) (h0 : P (l.headD dflt) x)
    (hstep : ∀ (d : ℕ) (hd : d + 1 < l.length) (y : γ), P l[d] y →
        P l[d + 1] (G l[d] (F (l[d], l[d + 1])) y)) :
    P (l.getLast?.getD dflt)
      ((l.zip ((l.zip l.tail).map F)).foldl (fun y bl => G bl.1 bl.2 y) x) := by
  cases l with
  | nil => exact absurd rfl hl
  | cons r rs =>
    rw [List.getLast?_eq_some_getLast (List.cons_ne_nil _ _)]
    exact foldl_pairs_cons G F P rs r x h0 hstep

end AssemblyAux

namespace AssemblyAux

lemma valOK_valAt (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ) :
    ValOK t (valAt t codes toks p) := by
  refine ⟨⟨windowBits t codes toks p, fun s => rfl⟩, fun i => ?_⟩
  show IsBit (if ((codes.lookup (toks p)).getD []).getD i false then 1 else 0)
  split_ifs
  · right; rfl
  · left; rfl

lemma getLast?_getD_eq {α : Type*} (l : List α) (hl : l ≠ []) (d : α) :
    l.getLast?.getD d = l.getLast hl := by
  rw [List.getLast?_eq_some_getLast hl]; rfl

lemma ideal_last (t : Trace) (hG : GridOK t (tokenGrid t)) (one : ℝ) (val : Name → ℝ)
    (i : ℕ) (hi : i < t.outputs.length) :
    ideal one val ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
      2 * val (.sig (t.outputs.getD i 0)) - 1 := by
  unfold ideal
  rw [getLast?_getD_eq _ hG.rows_ne, hG.last i hi]

end AssemblyAux

-- `Stmt_trace` is not needed: `valAt` is defined from `Trace.values` directly.
set_option linter.unusedVariables false in
open AssemblyAux in
theorem mainReal_of (hg : Stmt_grid) (ht : Stmt_trace) (hl : Stmt_layer_real)
    (ha : Stmt_attn_real) (hlog : Stmt_logit_real) : MainReal := by
  intro t codes toks p hprog hreal hcodes hpos M x
  have hG := hg t hprog
  have hW : t.outputs.length + 1 ≤ (tokenGrid t).width := hG.row0.width
  have hval := valOK_valAt t codes toks p
  have hfin : Close (tokenGrid t).width (1 / 10 ^ 9) (2 * Real.sqrt (tokenGrid t).width)
      (valAt t codes toks p) ((tokenGrid t).rows.getLast?.getD fun _ => none) (M.final toks p) := by
    have := foldl_pairs
      (fun before us y =>
        netStep gelu (tokenGrid t).width (2 * Real.sqrt (tokenGrid t).width) before us y)
      (fun br => layerUnits (tokenGrid t).width t.mlps br.1 br.2)
      (fun row y => Close (tokenGrid t).width (1 / 10 ^ 9) (2 * Real.sqrt (tokenGrid t).width)
        (valAt t codes toks p) row y)
      (tokenGrid t).rows hG.rows_ne (fun _ => none) (M.afterAttn toks p) ?_ ?_
    · exact this
    · have h := ha t codes toks p hprog hcodes hpos
      exact ⟨h.one_eq, h.none_eq, fun f h1 h2 => (h.near f h1 h2).trans (by norm_num)⟩
    · intro d hd y hy
      exact hl t _ _ _ _ y hprog.toWellFormed hreal (hG.pairs d hd) hval (by omega) hy
  refine ⟨fun i hi => ?_, fun τ hcode τ' hτ' hne => ?_⟩
  · have h := hfin.near (1 + i) (by omega) (by omega)
    rw [ideal_last t hG _ _ i hi] at h
    exact h
  · exact hlog t codes _ (M.final toks p) τ hprog hcodes hval hfin hcode τ' hτ' hne

set_option linter.unusedVariables false in
open AssemblyAux in
theorem mainBF16_of (hg : Stmt_grid) (ht : Stmt_trace) (hr : Stmt_round) (hl : Stmt_layer_bf16)
    (ha : Stmt_attn_bf16) (hlog : Stmt_logit_bf16) : MainBF16 := by
  intro R t codes toks p hprog hbf hcodes hpos hwidth M x
  obtain ⟨hfix, -, -, -, -, hmono, -, -, -⟩ := hr
  have hG := hg t hprog
  have hW : t.outputs.length + 1 ≤ (tokenGrid t).width := hG.row0.width
  have hval := valOK_valAt t codes toks p
  have hwidth' : |2 * Real.sqrt (tokenGrid t).width| ≤ maxBF16 := by
    rw [abs_of_nonneg (show (0:ℝ) ≤ 2 * Real.sqrt (tokenGrid t).width by positivity)]
    exact hwidth
  have hsq1 : 1 ≤ Real.sqrt (tokenGrid t).width := by
    rw [Real.one_le_sqrt]; exact_mod_cast (show 1 ≤ (tokenGrid t).width by omega)
  have honeb : IsBF16 (BF16.one R M.net) := R.fl_mem _ hwidth'
  have h2 : R.fl 2 = 2 := hfix R 2 ⟨2, 0, by norm_num, by norm_num, by norm_num, by norm_num⟩
  have hone1 : 1 ≤ BF16.one R M.net := by
    have := hmono R 2 (2 * Real.sqrt (tokenGrid t).width) (by norm_num [maxBF16]) hwidth'
      (by linarith)
    rw [h2] at this
    show 1 ≤ R.fl (2 * Real.sqrt (tokenGrid t).width)
    linarith
  have hfin : BF16.finalAt R M toks p = ideal (BF16.one R M.net) (valAt t codes toks p)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) := by
    have := foldl_pairs
      (fun before us y => BF16.netStep R (tokenGrid t).width (BF16.one R M.net) before us y)
      (fun br => layerUnits (tokenGrid t).width t.mlps br.1 br.2)
      (fun row y => y = ideal (BF16.one R M.net) (valAt t codes toks p) row)
      (tokenGrid t).rows hG.rows_ne (fun _ => none) (BF16.afterAttn R M toks p) ?_ ?_
    · exact this
    · funext f
      exact ha R t codes toks p hprog hcodes hpos hwidth f
    · intro d hd y hy
      subst hy
      funext f
      exact hl R t _ _ _ _ _ hprog.toWellFormed hbf (hG.pairs d hd) hval (by omega) honeb hone1 f
  refine ⟨fun i hi => ?_, fun hsize τ hcode τ' hτ' hne => ?_⟩
  · rw [hfin, ideal_last t hG _ _ i hi]
    rfl
  · show blogitOf R M (BF16.finalAt R M toks p) τ' < blogitOf R M (BF16.finalAt R M toks p) τ
    rw [hfin]
    exact hlog R t codes _ τ hprog hcodes hval hsize hcode τ' hτ' hne

end
end Reifier

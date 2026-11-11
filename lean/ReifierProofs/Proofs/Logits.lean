import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace LogitsAux

lemma mem_of_lookup {l : List (ℕ × List Bool)} {k : ℕ} {c : List Bool}
    (h : l.lookup k = some c) : (k, c) ∈ l := by
  obtain ⟨l₁, l₂, rfl, -⟩ := List.lookup_eq_some_iff.mp h
  simp

/-- A bit held as `-1` or `1`. -/
def sgn (β : Bool) : ℝ := if β then 1 else -1

lemma sum_split (W b : ℕ) (hW : b + 1 ≤ W) (F : ℕ → ℝ) (hF : ∀ f, b < f → f < W → F f = 0) :
    ∑ f ∈ range W, F f = F 0 + ∑ i ∈ range b, F (1 + i) := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hW
  rw [Finset.sum_range_add, Finset.sum_range_succ']
  have h0 : ∑ x ∈ range k, F (b + 1 + x) = 0 :=
    Finset.sum_eq_zero (fun x hx => hF _ (by omega) (by simp at hx; omega))
  rw [h0, add_zero, add_comm]
  congr 1
  exact Finset.sum_congr rfl (fun i _ => by rw [add_comm])

lemma exists_diff {t : Trace} {codes : List (ℕ × List Bool)} (hcodes : CodesOK t codes)
    {τ τ' : ℕ} {c c' : List Bool} (h1 : codes.lookup τ = some c) (h2 : codes.lookup τ' = some c')
    (hne : τ' ≠ τ) : ∃ i < t.outputs.length, c.getD i false ≠ c'.getD i false := by
  by_contra hcon
  push Not at hcon
  have m1 := mem_of_lookup h1
  have m2 := mem_of_lookup h2
  have l1 := hcodes.length _ m1
  have l2 := hcodes.length _ m2
  simp only at l1 l2
  have hcc : c = c' := List.ext_getElem (by omega) (fun i h₁ h₂ => by
    have := hcon i (by omega)
    rwa [List.getD_eq_getElem _ _ h₁, List.getD_eq_getElem _ _ h₂] at this)
  exact hne (hcodes.inj _ m2 _ m1 (by simp [hcc]))

lemma getLast?_getD_eq {α : Type*} (l : List α) (hl : l ≠ []) (d : α) :
    l.getLast?.getD d = l.getLast hl := by
  rw [List.getLast?_eq_some_getLast hl]; rfl

lemma embed_some {t : Trace} {codes : List (ℕ × List Bool)} {τ : ℕ} {c : List Bool}
    (h : codes.lookup τ = some c) (f : ℕ) :
    (transformer t codes).embed τ f =
      if f = 0 then 2 * Real.sqrt (tokenGrid t).width
      else if f ≤ t.outputs.length then sgn (c.getD (f - 1) false) else 0 := by
  simp only [transformer, h, sgn]
  rfl

lemma embed_none {t : Trace} {codes : List (ℕ × List Bool)} {τ : ℕ}
    (h : codes.lookup τ = none) (f : ℕ) : (transformer t codes).embed τ f = 0 := by
  simp only [transformer, h]


lemma mul_sgn_ge (x : ℝ) (β : Bool) (h : |x - sgn β| ≤ 1 / 10 ^ 9) :
    1 - 1 / 10 ^ 9 ≤ x * sgn β := by
  rw [abs_le] at h
  cases β <;> simp only [sgn, Bool.false_eq_true, ite_false, ite_true] at h ⊢ <;> linarith [h.1, h.2]

lemma diff_nonneg (x : ℝ) (β β' : Bool) (h : |x - sgn β| ≤ 1 / 10 ^ 9) :
    0 ≤ x * sgn β - x * sgn β' := by
  rw [abs_le] at h
  cases β <;> cases β' <;> simp only [sgn, Bool.false_eq_true, ite_false, ite_true] at h ⊢ <;>
    linarith [h.1, h.2]

lemma diff_pos (x : ℝ) (β β' : Bool) (h : |x - sgn β| ≤ 1 / 10 ^ 9) (hne : β ≠ β') :
    0 < x * sgn β - x * sgn β' := by
  rw [abs_le] at h
  cases β <;> cases β' <;> simp only [sgn, Bool.false_eq_true, ite_false, ite_true, ne_eq,
    not_true_eq_false] at h hne ⊢ <;> linarith [h.1, h.2]

/-- The real logit of a coded token. -/
lemma logit_coded (t : Trace) (codes : List (ℕ × List Bool)) (x : ℕ → ℝ) (τ : ℕ)
    (c : List Bool) (k : ℝ) (hW : t.outputs.length + 1 ≤ (tokenGrid t).width)
    (hk : ∀ f, rmsnorm (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (2 * Real.sqrt (tokenGrid t).width) x f = if f = 0 then k else k * x f)
    (h : codes.lookup τ = some c) :
    logitOf (transformer t codes) x τ = k * (2 * Real.sqrt (tokenGrid t).width) +
      k * ∑ i ∈ range t.outputs.length, x (1 + i) * sgn (c.getD i false) := by
  show ∑ f ∈ range (tokenGrid t).width,
    rmsnorm (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (2 * Real.sqrt (tokenGrid t).width) x f * (transformer t codes).embed τ f = _
  rw [sum_split _ _ hW]
  · rw [hk, embed_some h, Finset.mul_sum]
    simp only [ite_true]
    congr 1
    refine Finset.sum_congr rfl (fun i hi => ?_)
    simp at hi
    rw [hk, embed_some h]
    simp only [show 1 + i ≠ 0 by omega, ite_false, show 1 + i ≤ t.outputs.length by omega,
      ite_true, show 1 + i - 1 = i by omega]
    ring
  · intro f hf1 hf2
    rw [embed_some h]
    simp only [show f ≠ 0 by omega, ite_false, show ¬ f ≤ t.outputs.length by omega, mul_zero]

lemma logit_uncoded (t : Trace) (codes : List (ℕ × List Bool)) (x : ℕ → ℝ) (τ : ℕ)
    (h : codes.lookup τ = none) : logitOf (transformer t codes) x τ = 0 := by
  show ∑ f ∈ range (tokenGrid t).width,
    rmsnorm (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none)
      (2 * Real.sqrt (tokenGrid t).width) x f * (transformer t codes).embed τ f = _
  simp only [embed_none h, mul_zero, Finset.sum_const_zero]

end LogitsAux

open LogitsAux in
theorem logit_real_of (hg : Stmt_grid) (hn : Stmt_norm_close) : Stmt_logit_real := by
  intro t codes val x τ hprog hcodes hval M hclose hcode τ' hτ' hne
  have hG := hg t hprog
  have hW : t.outputs.length + 1 ≤ (tokenGrid t).width := hG.row0.width
  have hlast : ((tokenGrid t).rows.getLast?.getD fun _ => none) =
      (tokenGrid t).rows.getLast hG.rows_ne := getLast?_getD_eq _ _ _
  have hrow : RowOK t (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none) := by
    rw [hlast]; exact hG.rows_ok _ (List.getLast_mem _)
  have hlast1 : ∀ i < t.outputs.length, ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
      some (.sig (t.outputs.getD i 0)) := by
    rw [hlast]; exact hG.last
  obtain ⟨k, hk0, -, hk⟩ := hn t (tokenGrid t).width _ val x (1 / 10 ^ 9) hrow hval
    hprog.toWellFormed (by omega) (by norm_num) (by norm_num) hclose.one_eq hclose.near
  obtain ⟨cτ, hcτ, hcode⟩ := hcode
  have hx : ∀ i < t.outputs.length, |x (1 + i) - sgn (cτ.getD i false)| ≤ 1 / 10 ^ 9 := by
    intro i hi
    have h := hclose.near (1 + i) (by omega) (by show 1 + i < (tokenGrid t).width; omega)
    have hid : ideal M.net.one val ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
        sgn (cτ.getD i false) := by
      unfold ideal; rw [hlast1 i hi]; simp only; rw [← hcode i hi]; unfold sgn
      split_ifs <;> norm_num
    rwa [hid] at h
  have hpos : 0 < 2 * Real.sqrt (tokenGrid t).width := by
    have : (0 : ℝ) < (tokenGrid t).width := by exact_mod_cast (show 0 < (tokenGrid t).width by omega)
    positivity
  rw [logit_coded t codes x τ cτ k hW hk hcτ]
  rcases hτ'c : codes.lookup τ' with _ | c'
  · rw [logit_uncoded t codes x τ' hτ'c]
    have : 0 ≤ ∑ i ∈ range t.outputs.length, x (1 + i) * sgn (cτ.getD i false) :=
      Finset.sum_nonneg (fun i hi => by
        simp at hi; have := mul_sgn_ge _ _ (hx i hi); linarith)
    positivity
  · rw [logit_coded t codes x τ' c' k hW hk hτ'c]
    obtain ⟨i, hi, hdi⟩ := exists_diff hcodes hcτ hτ'c hne
    have : 0 < ∑ i ∈ range t.outputs.length,
        (x (1 + i) * sgn (cτ.getD i false) - x (1 + i) * sgn (c'.getD i false)) :=
      Finset.sum_pos' (fun j hj => by simp at hj; exact diff_nonneg _ _ _ (hx j hj))
        ⟨i, by simpa using hi, diff_pos _ _ _ (hx i hi) hdi⟩
    rw [Finset.sum_sub_distrib] at this
    have := mul_pos hk0 this
    nlinarith


namespace LogitsAux

lemma maxBF16_ge : (251 : ℝ) ≤ maxBF16 := by norm_num [maxBF16]

lemma fl_strict (R : Rounding)
    (hmono : ∀ (x y : ℝ), |x| ≤ maxBF16 → |y| ≤ maxBF16 → x ≤ y → R.fl x ≤ R.fl y)
    (herr : ∀ (x : ℝ), |x| ≤ maxBF16 → |R.fl x - x| ≤ |x| / 2 ^ 8 + 1 / 2 ^ 134)
    (A B S : ℝ) (hA : 1 ≤ A) (hB : 1 ≤ B) (hAB : A + B ≤ 251) (hS1 : -B ≤ S) (hS2 : S ≤ B - 2) :
    R.fl (A + S) < R.fl (A + B) := by
  have hm := maxBF16_ge
  have h1 : R.fl (A + S) ≤ R.fl (A + B - 2) :=
    hmono _ _ (by rw [abs_le]; constructor <;> linarith) (by rw [abs_le]; constructor <;> linarith)
      (by linarith)
  have h2 := herr (A + B - 2) (by rw [abs_le]; constructor <;> linarith)
  have h3 := herr (A + B) (by rw [abs_le]; constructor <;> linarith)
  rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ A + B - 2)] at h2
  rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ A + B)] at h3
  rw [abs_le] at h2 h3
  have : (1:ℝ) / 2 ^ 134 ≤ 1 / 1000 := by norm_num
  linarith [h2.1, h2.2, h3.1, h3.2]

lemma fl_pos (R : Rounding)
    (herr : ∀ (x : ℝ), |x| ≤ maxBF16 → |R.fl x - x| ≤ |x| / 2 ^ 8 + 1 / 2 ^ 134)
    (y : ℝ) (hy : 1 ≤ y) (hy2 : y ≤ 251) : 0 < R.fl y := by
  have hm := maxBF16_ge
  have h := herr y (by rw [abs_le]; constructor <;> linarith)
  rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ y), abs_le] at h
  have : (1:ℝ) / 2 ^ 134 ≤ 1 / 1000 := by norm_num
  linarith [h.1]

/-- The bf16 logit of a coded token. -/
lemma blogit_coded (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (x : ℕ → ℝ)
    (τ : ℕ) (c : List Bool) (s : ℕ → ℝ) (hW : t.outputs.length + 1 ≤ (tokenGrid t).width)
    (hn0 : BF16.norm R (tokenGrid t).width (BF16.one R (transformer t codes).net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x 0 = 1)
    (hni : ∀ i < t.outputs.length, BF16.norm R (tokenGrid t).width
      (BF16.one R (transformer t codes).net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x (1 + i) = s i)
    (h0 : R.fl 0 = 0) (hs : ∀ β, R.fl (sgn β) = sgn β)
    (h : codes.lookup τ = some c) :
    blogitOf R (transformer t codes) x τ = R.fl (BF16.one R (transformer t codes).net +
      ∑ i ∈ range t.outputs.length, s i * sgn (c.getD i false)) := by
  show R.fl (∑ f ∈ range (tokenGrid t).width,
    BF16.norm R (tokenGrid t).width (BF16.one R (transformer t codes).net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x f *
      R.fl ((transformer t codes).embed τ f)) = _
  congr 1
  rw [sum_split _ _ hW]
  · rw [hn0, embed_some h]
    simp only [ite_true, one_mul]
    congr 1
    refine Finset.sum_congr rfl (fun i hi => ?_)
    simp at hi
    rw [hni i hi, embed_some h]
    simp only [show 1 + i ≠ 0 by omega, ite_false, show 1 + i ≤ t.outputs.length by omega,
      ite_true, show 1 + i - 1 = i by omega, hs]
  · intro f hf1 hf2
    rw [embed_some h]
    simp only [show f ≠ 0 by omega, ite_false, show ¬ f ≤ t.outputs.length by omega, h0,
      mul_zero]

lemma blogit_uncoded (R : Rounding) (t : Trace) (codes : List (ℕ × List Bool)) (x : ℕ → ℝ)
    (τ : ℕ) (h0 : R.fl 0 = 0) (h : codes.lookup τ = none) :
    blogitOf R (transformer t codes) x τ = 0 := by
  show R.fl (∑ f ∈ range (tokenGrid t).width,
    BF16.norm R (tokenGrid t).width (BF16.one R (transformer t codes).net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x f *
      R.fl ((transformer t codes).embed τ f)) = _
  simp only [embed_none h, h0, mul_zero, Finset.sum_const_zero]

lemma sum_sgn_self (b : ℕ) (β : ℕ → Bool) :
    ∑ i ∈ range b, sgn (β i) * sgn (β i) = b := by
  have : ∀ i, sgn (β i) * sgn (β i) = 1 := fun i => by
    unfold sgn; split_ifs <;> norm_num
  simp [this]

lemma sum_sgn_bounds (b : ℕ) (β β' : ℕ → Bool) :
    -(b : ℝ) ≤ ∑ i ∈ range b, sgn (β i) * sgn (β' i) := by
  have : ∀ i, -1 ≤ sgn (β i) * sgn (β' i) := fun i => by
    unfold sgn; split_ifs <;> norm_num
  have := Finset.sum_le_sum (s := range b) (fun i _ => this i)
  simpa using this

lemma sum_sgn_diff (b : ℕ) (β β' : ℕ → Bool) (i : ℕ) (hi : i < b) (hne : β i ≠ β' i) :
    ∑ i ∈ range b, sgn (β i) * sgn (β' i) ≤ b - 2 := by
  have hle : ∀ j, sgn (β j) * sgn (β' j) ≤ 1 := fun j => by
    unfold sgn; split_ifs <;> norm_num
  have hi' : sgn (β i) * sgn (β' i) = -1 := by
    revert hne; unfold sgn; cases β i <;> cases β' i <;> simp
  have h1 : ∑ j ∈ range b, (1 - sgn (β j) * sgn (β' j)) ≥ 1 - sgn (β i) * sgn (β' i) :=
    Finset.single_le_sum (f := fun j => 1 - sgn (β j) * sgn (β' j))
      (fun j _ => by linarith [hle j]) (by simpa using hi)
  rw [Finset.sum_sub_distrib, hi'] at h1
  simp at h1
  linarith

end LogitsAux

open LogitsAux in
theorem logit_bf16_of (hg : Stmt_grid) (hn : Stmt_norm_exact) (hr : Stmt_round) :
    Stmt_logit_bf16 := by
  intro R t codes val τ hprog hcodes hval hsize M x hcode τ' hτ' hne
  obtain ⟨-, -, -, -, -, hmono, hpm, hzero, herr⟩ := hr
  specialize hmono R
  specialize hpm R
  specialize hzero R
  specialize herr R
  have hG := hg t hprog
  have hW : t.outputs.length + 1 ≤ (tokenGrid t).width := hG.row0.width
  have hb : 1 ≤ t.outputs.length := List.length_pos_iff.mpr hprog.outputs_ne
  have hlast : ((tokenGrid t).rows.getLast?.getD fun _ => none) =
      (tokenGrid t).rows.getLast hG.rows_ne := getLast?_getD_eq _ _ _
  have hrow : RowOK t (tokenGrid t).width ((tokenGrid t).rows.getLast?.getD fun _ => none) := by
    rw [hlast]; exact hG.rows_ok _ (List.getLast_mem _)
  have hlast1 : ∀ i < t.outputs.length, ((tokenGrid t).rows.getLast?.getD fun _ => none) (1 + i) =
      some (.sig (t.outputs.getD i 0)) := by
    rw [hlast]; exact hG.last
  have hm := maxBF16_ge
  -- bounds on `one`
  have hsq1 : 1 ≤ Real.sqrt (tokenGrid t).width := by
    rw [Real.one_le_sqrt]; exact_mod_cast (show 1 ≤ (tokenGrid t).width by omega)
  have hbR : (1 : ℝ) ≤ t.outputs.length := by exact_mod_cast hb
  have hA := herr (2 * Real.sqrt (tokenGrid t).width)
    (by rw [abs_of_nonneg (by positivity)]; linarith)
  rw [abs_of_nonneg (show (0:ℝ) ≤ 2 * Real.sqrt (tokenGrid t).width by positivity), abs_le] at hA
  have htiny : (1:ℝ) / 2 ^ 134 ≤ 1 / 1000 := by norm_num
  have hA1 : 1 ≤ BF16.one R M.net := by
    show 1 ≤ R.fl (2 * Real.sqrt (tokenGrid t).width); linarith [hA.1]
  have hA2 : BF16.one R M.net + t.outputs.length ≤ 251 := by
    show R.fl (2 * Real.sqrt (tokenGrid t).width) + _ ≤ _; linarith [hA.2]
  -- the rounding of small exact values
  have h0 : R.fl 0 = 0 := hzero 0 (by norm_num)
  have hs : ∀ β, R.fl (sgn β) = sgn β := fun β => by
    cases β
    · exact hpm _ _ (Or.inr rfl) (by simp [sgn])
    · exact hpm _ _ (Or.inl rfl) (by simp [sgn])
  have h1 : R.fl 1 = 1 := hpm _ _ (Or.inl rfl) (by simp)
  -- the norm of the last row's stream
  obtain ⟨cτ, hcτ, hcode⟩ := hcode
  have hnorm : ∀ f, BF16.norm R (tokenGrid t).width (BF16.one R M.net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x f =
      R.fl (ideal 1 val ((tokenGrid t).rows.getLast?.getD fun _ => none) f) := fun f =>
    congrArg R.fl (hn t _ _ val _ hrow hval hprog.toWellFormed (by linarith) (by omega) f)
  have hn0 : BF16.norm R (tokenGrid t).width (BF16.one R M.net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x 0 = 1 := by
    rw [hnorm]; unfold ideal; rw [hrow.one_zero]; exact h1
  have hni : ∀ i < t.outputs.length, BF16.norm R (tokenGrid t).width (BF16.one R M.net)
      ((tokenGrid t).rows.getLast?.getD fun _ => none) x (1 + i) = sgn (cτ.getD i false) := by
    intro i hi
    rw [hnorm]; unfold ideal; rw [hlast1 i hi]; simp only; rw [← hcode i hi, ← hs]
    congr 1; unfold sgn; split_ifs <;> norm_num
  have hτv := blogit_coded R t codes x τ cτ (fun i => sgn (cτ.getD i false)) hW hn0 hni h0 hs hcτ
  rw [sum_sgn_self] at hτv
  rw [hτv]
  rcases hτ'c : codes.lookup τ' with _ | c'
  · rw [blogit_uncoded R t codes x τ' h0 hτ'c]
    exact fl_pos R herr _ (by linarith) hA2
  · rw [blogit_coded R t codes x τ' c' (fun i => sgn (cτ.getD i false)) hW hn0 hni h0 hs hτ'c]
    obtain ⟨i, hi, hdi⟩ := exists_diff hcodes hcτ hτ'c hne
    exact fl_strict R hmono herr _ _ _ hA1 hbR hA2
      (sum_sgn_bounds _ (fun i => cτ.getD i false) (fun i => c'.getD i false))
      (sum_sgn_diff _ (fun i => cτ.getD i false) (fun i => c'.getD i false) i hi hdi)

end
end Reifier

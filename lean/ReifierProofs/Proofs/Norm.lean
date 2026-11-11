import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace NormAux

lemma foldl_apply_not_mem (ms : List MLP) (v : Sig → ℝ) (s : Sig)
    (h : s ∉ ms.flatMap MLP.outs) :
    ms.foldl (fun v m => m.apply v) v s = v s := by
  induction ms generalizing v with
  | nil => rfl
  | cons m ms ih =>
    simp only [List.flatMap_cons, List.mem_append, not_or] at h
    rw [List.foldl_cons, ih _ h.2]
    simp [MLP.apply, h.1]

lemma values_input (t : Trace) (hwf : t.WellFormed) (x : Sig → ℝ) (s : Sig)
    (hs : s ∈ t.inputs) : t.values x s = x s := by
  apply foldl_apply_not_mem
  intro hm
  exact (List.nodup_append.mp hwf.nodup).2.2 s hs s hm rfl

lemma val_bit {t : Trace} {W : ℕ} {row : Row} {val : Name → ℝ} (hrow : RowOK t W row)
    (hval : ValOK t val) (hwf : t.WellFormed) {f : ℕ} {n : Name} (h : row f = some n)
    (hn : n ≠ .one) : IsBit (val n) := by
  rcases hrow.names f n h with h1 | ⟨i, rfl, _⟩ | ⟨s, rfl, hs⟩
  · exact absurd h1 hn
  · exact hval.2 i
  · obtain ⟨x, hx⟩ := hval.1
    rw [hx s]
    rcases hs with hs | ⟨m, hm, hs⟩
    · rw [values_input t hwf _ s hs]; unfold Trace.env; split_ifs <;> simp [IsBit]
    · exact hwf.bits x m hm s hs

lemma ideal_zero {t : Trace} {W : ℕ} {row : Row} (hrow : RowOK t W row) (one : ℝ)
    (val : Name → ℝ) : ideal one val row 0 = one := by
  unfold ideal; rw [hrow.one_zero]

lemma ideal_ne_zero {t : Trace} {W : ℕ} {row : Row} (hrow : RowOK t W row) (a b : ℝ)
    (val : Name → ℝ) {f : ℕ} (hf : f ≠ 0) : ideal a val row f = ideal b val row f := by
  unfold ideal
  rcases h : row f with _ | n
  · rfl
  · cases n with
    | one => exact absurd (hrow.one_only f h) hf
    | sig s => rfl
    | code i => rfl

lemma ideal_cases {t : Trace} {W : ℕ} {row : Row} {val : Name → ℝ} (hrow : RowOK t W row)
    (hval : ValOK t val) (hwf : t.WellFormed) (one : ℝ) {f : ℕ} (hf : f ≠ 0) :
    ((row f).isSome = false ∧ ideal one val row f = 0) ∨
      ((row f).isSome = true ∧ (ideal one val row f = 1 ∨ ideal one val row f = -1)) := by
  rcases h : row f with _ | n
  · left; refine ⟨rfl, ?_⟩; unfold ideal; rw [h]
  · right
    refine ⟨rfl, ?_⟩
    have hn : n ≠ .one := fun e => hf (hrow.one_only f (e ▸ h))
    have hb := val_bit hrow hval hwf h hn
    have : ideal one val row f = 2 * val n - 1 := by
      unfold ideal; rw [h]; cases n <;> simp_all
    rw [this]
    rcases hb with hb | hb <;> rw [hb] <;> norm_num

lemma count_succ (W : ℕ) (row : Row) :
    count (W + 1) row = count W row + if (1 ≤ W ∧ (row W).isSome) then 1 else 0 := by
  unfold count
  rw [List.range_succ, List.filter_append, List.length_append]
  simp [List.filter_cons]
  split_ifs <;> rfl

lemma count_le (W : ℕ) (row : Row) : count W row ≤ W := by
  unfold count
  calc _ ≤ (List.range W).length := List.length_filter_le _ _
    _ = W := List.length_range

/-- The indicator of the cells after `ONE` that hold a name. -/
def ind (row : Row) (g : ℕ) : ℝ := if g = 0 then 0 else if (row g).isSome then 1 else 0

lemma sum_ind (W : ℕ) (row : Row) : ∑ g ∈ range W, ind row g = count W row := by
  induction W with
  | zero => simp [count]
  | succ W ih =>
    rw [Finset.sum_range_succ, ih, count_succ]
    unfold ind
    by_cases hW : W = 0
    · subst hW; simp
    · have h1 : 1 ≤ W := by omega
      by_cases hs : (row W).isSome = true <;> simp [hW, hs, h1]

lemma sum_sq_ideal {t : Trace} {W : ℕ} {row : Row} {val : Name → ℝ} (hrow : RowOK t W row)
    (hval : ValOK t val) (hwf : t.WellFormed) (one : ℝ) (hW : 0 < W) :
    ∑ g ∈ range W, ideal one val row g ^ 2 = one ^ 2 + count W row := by
  have h : ∀ g, ideal one val row g ^ 2 = (if g = 0 then one ^ 2 else 0) + ind row g := by
    intro g
    by_cases hg : g = 0
    · subst hg; simp [ind, ideal_zero hrow]
    · rcases ideal_cases hrow hval hwf one hg with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · simp [ind, hg, h1, h2]
      · rcases h2 with h2 | h2 <;> simp [ind, hg, h1, h2]
  rw [Finset.sum_congr rfl (fun g _ => h g), Finset.sum_add_distrib, sum_ind,
    Finset.sum_ite_eq' (range W) 0]
  simp [hW]

lemma sq_err (x I E : ℝ) (h : |x - I| ≤ E) : |x ^ 2 - I ^ 2| ≤ 2 * |I| * E + E ^ 2 := by
  have e : x ^ 2 - I ^ 2 = (x - I) ^ 2 + 2 * I * (x - I) := by ring
  rw [e]
  have h0 := abs_nonneg (x - I)
  have hI := abs_nonneg I
  calc |(x - I) ^ 2 + 2 * I * (x - I)| ≤ |(x - I) ^ 2| + |2 * I * (x - I)| := abs_add_le _ _
    _ = |x - I| ^ 2 + 2 * |I| * |x - I| := by
        rw [abs_pow, abs_mul, abs_mul]; norm_num
    _ ≤ E ^ 2 + 2 * |I| * E := by gcongr
    _ = _ := by ring

lemma term_bound {t : Trace} {W : ℕ} {row : Row} {val : Name → ℝ} (hrow : RowOK t W row)
    (hval : ValOK t val) (hwf : t.WellFormed) (one : ℝ) (x : ℕ → ℝ) (E : ℝ)
    (hx0 : x 0 = one) (hx : ∀ f, 1 ≤ f → f < W → |x f - ideal one val row f| ≤ E)
    (_hE : 0 ≤ E) (g : ℕ) (hg : g < W) :
    |x g ^ 2 - ((if g = 0 then one ^ 2 else 0) + ind row g)| ≤ 2 * E * ind row g + E ^ 2 := by
  by_cases hg0 : g = 0
  · subst hg0; simp [ind, hx0]; positivity
  · have hb := hx g (by omega) hg
    have hs := sq_err _ _ _ hb
    rcases ideal_cases hrow hval hwf one hg0 with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · rw [h2] at hs
      simp [ind, hg0, h1]
      simpa using hs
    · rcases h2 with h2 | h2 <;> rw [h2] at hs <;> simp [ind, hg0, h1] <;> norm_num at hs <;>
        linarith

end NormAux

open NormAux

theorem stmt_norm_exact : Stmt_norm_exact := by
  intro t W row val one hrow hval hwf hone hW f
  have hWr : (0:ℝ) < W := by exact_mod_cast hW
  have hS : 0 < (one ^ 2 + count W row) / W + EPS := by unfold EPS; positivity
  have hsq : 0 < Real.sqrt ((one ^ 2 + count W row) / W + EPS) := Real.sqrt_pos.mpr hS
  unfold rmsnorm rmsScale
  rw [sum_sq_ideal hrow hval hwf one hW]
  simp only
  rw [mul_div_cancel_left₀ _ hsq.ne']
  split_ifs with hf
  · subst hf; rw [ideal_zero hrow, ideal_zero hrow, div_self hone.ne']
  · exact ideal_ne_zero hrow one 1 val hf

theorem stmt_norm_close : Stmt_norm_close := by
  intro t W row val x E hrow hval hwf hW hE0 hE1 hx0 hx
  have hWr : (0:ℝ) < W := by exact_mod_cast hW
  have hone2 : (2 * Real.sqrt W) ^ 2 = 4 * W := by rw [mul_pow, Real.sq_sqrt hWr.le]; ring
  have hone_pos : 0 < 2 * Real.sqrt W := by positivity
  have hEPS : (0:ℝ) < EPS := by unfold EPS; norm_num
  obtain ⟨n, hn⟩ : ∃ n : ℝ, n = count W row := ⟨_, rfl⟩
  obtain ⟨D, hD⟩ : ∃ D : ℝ, D = ∑ g ∈ range W, x g ^ 2 := ⟨_, rfl⟩
  have hn0 : 0 ≤ n := by rw [hn]; positivity
  have hnW : n ≤ W := by rw [hn]; exact_mod_cast count_le W row
  have hD0 : 0 ≤ D := by rw [hD]; positivity
  have hdelta : |D - (4 * W + n)| ≤ 2 * E * n + W * E ^ 2 := by
    have e1 : D - (4 * W + n) = ∑ g ∈ range W,
        (x g ^ 2 - ((if g = 0 then (2 * Real.sqrt W) ^ 2 else 0) + ind row g)) := by
      rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, sum_ind,
        Finset.sum_ite_eq' (range W) 0, hD, hn, hone2]
      simp [hW]
    rw [e1]
    calc _ ≤ ∑ g ∈ range W,
          |x g ^ 2 - ((if g = 0 then (2 * Real.sqrt W) ^ 2 else 0) + ind row g)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ g ∈ range W, (2 * E * ind row g + E ^ 2) :=
          Finset.sum_le_sum (fun g hg => term_bound hrow hval hwf _ x E hx0 hx hE0 g
            (Finset.mem_range.mp hg))
      _ = 2 * E * n + W * E ^ 2 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_ind, Finset.sum_const,
            Finset.card_range, hn]
          simp
  have hS : 0 < ((2 * Real.sqrt W) ^ 2 + (count W row : ℝ)) / W + EPS := by positivity
  have hT : 0 < D / W + EPS := by positivity
  refine ⟨Real.sqrt (((2 * Real.sqrt W) ^ 2 + (count W row : ℝ)) / W + EPS) /
    Real.sqrt (D / W + EPS), by positivity, ?_, ?_⟩
  · rw [div_pow, Real.sq_sqrt hS.le, Real.sq_sqrt hT.le, div_sub_one hT.ne', abs_div,
      abs_of_pos hT, div_le_iff₀ hT]
    have hST : ((2 * Real.sqrt W) ^ 2 + (count W row : ℝ)) / W + EPS - (D / W + EPS) =
        -(D - (4 * W + n)) / W := by
      rw [hone2, ← hn]; field_simp; ring
    rw [hST, abs_div, abs_neg, abs_of_pos hWr, div_le_iff₀ hWr]
    have hTW : (D / W + EPS) * W = D + EPS * W := by field_simp
    obtain ⟨δ, hδ⟩ : ∃ δ : ℝ, δ = D - (4 * W + n) := ⟨_, rfl⟩
    rw [← hδ] at hdelta ⊢
    have k1 : (2 * n + W * E) * (1 + 41 / 100 * E) ≤ 41 / 100 * (4 * W + n) := by
      have a1 : n * E ≤ W * E := mul_le_mul_of_nonneg_right hnW hE0
      have a2 : W * E ≤ W * (1 / 10 ^ 6) := mul_le_mul_of_nonneg_left hE1 hWr.le
      have a3 : W * E * E ≤ W * E * (1 / 10 ^ 6) :=
        mul_le_mul_of_nonneg_left hE1 (by positivity)
      nlinarith
    have k2 : |δ| * (1 + 41 / 100 * E) ≤ E * (41 / 100 * (4 * W + n)) := by
      calc |δ| * (1 + 41 / 100 * E) ≤ (2 * E * n + W * E ^ 2) * (1 + 41 / 100 * E) :=
            mul_le_mul_of_nonneg_right hdelta (by positivity)
        _ = E * ((2 * n + W * E) * (1 + 41 / 100 * E)) := by ring
        _ ≤ E * (41 / 100 * (4 * W + n)) := mul_le_mul_of_nonneg_left k1 hE0
    have key : |δ| ≤ 41 / 100 * E * (4 * W + n - |δ|) := by linarith
    calc |δ| ≤ 41 / 100 * E * (4 * W + n - |δ|) := key
      _ ≤ 41 / 100 * E * (D + EPS * W) := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          have := neg_abs_le δ
          have := mul_pos hEPS hWr
          linarith
      _ = 41 / 100 * E * (D / W + EPS) * W := by rw [mul_assoc (41 / 100 * E), hTW]
  · intro f
    unfold rmsnorm rmsScale
    rw [← hD]
    simp only
    split_ifs with hf
    · subst hf; rw [hx0]; field_simp
    · ring

end

end Reifier

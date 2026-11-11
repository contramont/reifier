import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace DecompAux

/-! ## Basic facts about compiled MLPs -/

lemma compiled_ins (m : MLP) : m.compiled.ins = m.ins := by
  unfold MLP.compiled Gated.signed Gated.sharpen MLP.toReglu
  split <;> rfl

lemma compiled_outs (m : MLP) : m.compiled.outs = m.outs := by
  unfold MLP.compiled Gated.signed Gated.sharpen MLP.toReglu
  split <;> rfl

/-! ## Lists -/

lemma list_sum_range (n : ℕ) (g : ℕ → ℝ) :
    ((List.range n).map g).sum = ∑ j ∈ range n, g j := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.sum_range_succ, Finset.sum_range_succ, ih]

lemma list_sum_single {α : Type*} {l : List α} (hl : l.Nodup) (g : α → ℝ) (a : α) (ha : a ∈ l)
    (h : ∀ x ∈ l, x ≠ a → g x = 0) : (l.map g).sum = g a := by
  classical
  rw [← List.sum_toFinset _ hl]
  exact Finset.sum_eq_single_of_mem a (by simpa using ha)
    (fun x hx hne => h x (by simpa using hx) hne)

lemma list_sum_zero {α : Type*} {l : List α} (g : α → ℝ) (h : ∀ x ∈ l, g x = 0) :
    (l.map g).sum = 0 := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [h a (by simp), ih (fun x hx => h x (by simp [hx]))]
    simp


lemma getD_map_range {β : Type*} (n : ℕ) (U : ℕ → β) (d : β) (j : ℕ) (hj : j < n) :
    ((List.range n).map U).getD j d = U j := by
  rw [List.getD_eq_getElem _ _ (by simpa using hj)]
  simp

/-! ## Traces -/

lemma flat_nodup {t : Trace} (hwf : t.WellFormed) : (t.mlps.flatMap MLP.outs).Nodup :=
  List.Nodup.of_append_right hwf.nodup

lemma outs_nodup {t : Trace} (hwf : t.WellFormed) {m : MLP} (hm : m ∈ t.mlps) : m.outs.Nodup :=
  (List.nodup_flatMap.1 (flat_nodup hwf)).1 m hm

lemma outs_pairwise {t : Trace} (hwf : t.WellFormed) :
    t.mlps.Pairwise (Function.onFun List.Disjoint MLP.outs) :=
  (List.nodup_flatMap.1 (flat_nodup hwf)).2

lemma mlps_nodup {t : Trace} (hwf : t.WellFormed) : t.mlps.Nodup := by
  have hp := outs_pairwise hwf
  refine List.Pairwise.imp_of_mem ?_ hp
  intro a b ha _ hab heq
  subst heq
  obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ (hwf.outs_ne a ha)
  exact hab hy hy

lemma eq_of_common_out {t : Trace} (hwf : t.WellFormed) {m m' : MLP} (hm : m ∈ t.mlps)
    (hm' : m' ∈ t.mlps) {y : Sig} (hy : y ∈ m.outs) (hy' : y ∈ m'.outs) : m = m' := by
  by_contra hne
  have := List.Pairwise.forall (R := Function.onFun List.Disjoint MLP.outs) (outs_pairwise hwf)
    hm hm' hne
  exact this hy hy'

/-! ## Rows -/

lemma inRow_iff (W : ℕ) (row : Row) (s : Name) :
    inRow W row s = true ↔ ∃ f < W, row f = some s := by
  unfold inRow
  simp [List.any_eq_true]

lemma featIn_spec {W : ℕ} {row : Row} {s : Name} (h : inRow W row s = true) :
    featIn W row s < W ∧ row (featIn W row s) = some s := by
  unfold featIn
  have hs : ((List.range W).find? (fun f => row f == some s)).isSome := by
    rw [List.find?_isSome]
    obtain ⟨f, hf, hf'⟩ := (inRow_iff W row s).1 h
    exact ⟨f, by simpa using hf, by simpa using hf'⟩
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp hs
  rw [ha]
  simp only [Option.getD_some]
  have h1 := List.mem_of_find?_eq_some ha
  have h2 := List.find?_some ha
  simp only [List.mem_range, beq_iff_eq] at h1 h2
  exact ⟨h1, h2⟩

lemma rowOK_lt_of_isSome {t : Trace} {W : ℕ} {r : Row} (hr : RowOK t W r) {f : ℕ}
    (hf : (r f).isSome) : f < W := by
  by_contra h
  push Not at h
  rw [hr.none_beyond f h] at hf
  simp at hf

lemma rowOK_lt_of_eq_some {t : Trace} {W : ℕ} {r : Row} (hr : RowOK t W r) {f : ℕ} {s : Name}
    (hf : r f = some s) : f < W :=
  rowOK_lt_of_isSome hr (by rw [hf]; rfl)

lemma rowOK_inRow_of {t : Trace} {W : ℕ} {r : Row} (hr : RowOK t W r) {f : ℕ} {s : Name}
    (hf : r f = some s) : inRow W r s = true :=
  (inRow_iff W r s).2 ⟨f, rowOK_lt_of_eq_some hr hf, hf⟩

lemma rowOK_featIn_eq {t : Trace} {W : ℕ} {r : Row} (hr : RowOK t W r) {f : ℕ} {s : Name}
    (hf : r f = some s) : featIn W r s = f := by
  obtain ⟨_, h2⟩ := featIn_spec (rowOK_inRow_of hr hf)
  exact hr.nodup _ _ (h2.trans hf.symm) (by rw [h2]; rfl)

lemma rowOK_W_pos {t : Trace} {W : ℕ} {r : Row} (hr : RowOK t W r) : 0 < W :=
  rowOK_lt_of_eq_some hr hr.one_zero


/-! ## Pairs of rows -/

section Pair

variable {t : Trace} {W : ℕ} {before row : Row}

lemma new_out_spec (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {y : Sig} (hy : y ∈ m.outs) :
    featIn W row (.sig y) < W ∧ row (featIn W row (.sig y)) = some (.sig y) :=
  featIn_spec (hp.new_outs m hm hnew y hy).1

lemma new_out_not_before (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {y : Sig} (hy : y ∈ m.outs) (g : ℕ) :
    before g ≠ some (.sig y) := by
  intro h
  have h1 := rowOK_inRow_of hp.before_ok h
  rw [(hp.new_outs m hm hnew y hy).2] at h1
  exact Bool.false_ne_true h1

lemma new_out_feat (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {y : Sig} (hy : y ∈ m.outs) {f : ℕ}
    (hf : featIn W row (.sig y) = f) : row f = some (.sig y) := by
  rw [← hf]; exact (new_out_spec hp hm hnew hy).2

lemma new_in_spec (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {x : Sig} (hx : x ∈ m.ins) :
    featIn W before (.sig x) < W ∧ before (featIn W before (.sig x)) = some (.sig x) :=
  featIn_spec (hp.new_ins m hm hnew x hx)

end Pair

/-! ## Sums over units -/

lemma unitSum_cons (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (u : Hidden) (us : List Hidden) (f : ℕ) :
    unitSum φ ψ (u :: us) f = φ (u.out f) * ψ u + unitSum φ ψ us f := by
  simp [unitSum]

lemma unitSum_append (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (a b : List Hidden) (f : ℕ) :
    unitSum φ ψ (a ++ b) f = unitSum φ ψ a f + unitSum φ ψ b f := by
  simp [unitSum]

lemma unitSum_flatMap {α : Type*} (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (L : List α)
    (g : α → List Hidden) (f : ℕ) :
    unitSum φ ψ (L.flatMap g) f = (L.map fun x => unitSum φ ψ (g x) f).sum := by
  induction L with
  | nil => simp [unitSum]
  | cons a L ih => rw [List.flatMap_cons, unitSum_append, ih]; simp

lemma unitSum_zero (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (hφ : φ 0 = 0) (us : List Hidden) (f : ℕ)
    (h : ∀ u ∈ us, u.out f = 0) : unitSum φ ψ us f = 0 := by
  unfold unitSum
  apply list_sum_zero
  intro u hu
  rw [h u hu]; simp [hφ]

lemma unitSum_map_range (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (n : ℕ) (U : ℕ → Hidden) (f : ℕ) :
    unitSum φ ψ ((List.range n).map U) f =
      ∑ j ∈ range n, φ ((((List.range n).map U).getD j ⟨0, 0, 0⟩).out f) *
        ψ (((List.range n).map U).getD j ⟨0, 0, 0⟩) := by
  unfold unitSum
  rw [List.map_map, list_sum_range]
  apply Finset.sum_congr rfl
  intro j hj
  rw [getD_map_range _ _ _ _ (by simpa using hj)]
  rfl


/-! ## The units -/

lemma mlpUnits_getD (W : ℕ) (before row : Row) (m : MLP) (j : ℕ) (hj : j < m.compiled.n) :
    (mlpUnits W before row m).getD j ⟨0, 0, 0⟩ =
      { gate := fun f =>
          (∑ k ∈ range m.ins.length,
              if featIn W before (.sig (m.ins.getD k 0)) = f then m.compiled.g j k else 0)
            + (if f = 0 then m.compiled.gb j else 0)
        val := fun f =>
          (∑ k ∈ range m.ins.length,
              if featIn W before (.sig (m.ins.getD k 0)) = f then m.compiled.v j k else 0)
            + (if f = 0 then m.compiled.vb j else 0)
        out := fun f =>
          ∑ i ∈ range m.outs.length,
            if featIn W row (.sig (m.outs.getD i 0)) = f then m.compiled.wo i j else 0 } := by
  unfold mlpUnits
  rw [getD_map_range _ _ _ _ hj]
  simp only [compiled_ins, compiled_outs]

lemma mem_mlpUnits {W : ℕ} {before row : Row} {m : MLP} {u : Hidden}
    (hu : u ∈ mlpUnits W before row m) : ∃ j, u.out = fun f =>
      ∑ i ∈ range m.outs.length,
        if featIn W row (.sig (m.outs.getD i 0)) = f then m.compiled.wo i j else 0 := by
  unfold mlpUnits at hu
  simp only [List.mem_map, List.mem_range] at hu
  obtain ⟨j, _, rfl⟩ := hu
  exact ⟨j, by simp only [compiled_outs]⟩

lemma unitSum_mlpUnits (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (W : ℕ) (before row : Row) (m : MLP)
    (f : ℕ) :
    unitSum φ ψ (mlpUnits W before row m) f =
      ∑ j ∈ range m.compiled.n, φ (((mlpUnits W before row m).getD j ⟨0, 0, 0⟩).out f) *
        ψ ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩) := by
  unfold mlpUnits
  exact unitSum_map_range φ ψ _ _ f

lemma mem_eraseUnits {W : ℕ} {before row : Row} {f : ℕ} :
    f ∈ (List.range W).filter
        (fun f => decide (1 ≤ f) && before f != none && before f != row f) ↔
      f < W ∧ 1 ≤ f ∧ before f ≠ none ∧ before f ≠ row f := by
  simp [List.mem_filter, Bool.and_eq_true, bne_iff_ne, and_assoc]

section Pair

variable {t : Trace} {W : ℕ} {before row : Row}

lemma oneUnit_out_zero (hp : PairOK t W before row) {f : ℕ} (hf : row f = before f) :
    (oneUnit W before row).out f = 0 := by
  have : isNewFeat W before row f = false := by
    unfold isNewFeat
    cases h : row f with
    | none => simp
    | some s =>
      have h1 : inRow W before s = true := rowOK_inRow_of hp.before_ok (hf.symm.trans h)
      simp [h1]
  simp [oneUnit, this]

lemma oneUnit_out_new (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {y : Sig} (hy : y ∈ m.outs) {f : ℕ}
    (hf : row f = some (.sig y)) : (oneUnit W before row).out f = -1 := by
  have : isNewFeat W before row f = true := by
    unfold isNewFeat
    have hfW : f < W := rowOK_lt_of_eq_some hp.row_ok hf
    rw [hf]
    simp [hfW, (hp.new_outs m hm hnew y hy).2]
  simp [oneUnit, this]

end Pair


/-! ## Stmt_decomp, part 1 -/

lemma decomp_keep (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (t : Trace) (W : ℕ) (before row : Row)
    (hφ : φ 0 = 0) (hp : PairOK t W before row) (f : ℕ) (hf : row f = before f) :
    unitSum φ ψ (layerUnits W t.mlps before row) f = 0 := by
  apply unitSum_zero φ ψ hφ
  intro u hu
  simp only [layerUnits, List.mem_cons, List.mem_append, List.mem_flatMap, List.mem_filter,
    List.mem_map] at hu
  rcases hu with rfl | ⟨m, ⟨hm, hnew⟩, hu⟩ | ⟨f', hf', rfl⟩
  · exact oneUnit_out_zero hp hf
  · obtain ⟨j, hj⟩ := mem_mlpUnits hu
    rw [hj]
    apply Finset.sum_eq_zero
    intro i hi
    rw [ite_eq_right]
    intro hfe
    have hy : m.outs.getD i 0 ∈ m.outs := by
      rw [List.getD_eq_getElem _ _ (by simpa using hi)]; exact List.getElem_mem _
    have h1 := new_out_feat hp hm hnew hy hfe
    exact new_out_not_before hp hm hnew hy f (hf ▸ h1)
  · simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hf'
    simp only [eraseUnit]
    rw [ite_eq_right]
    rintro rfl
    exact hf'.2.2 hf.symm


/-! ## Stmt_decomp, part 2 -/

section New

variable {t : Trace} {W : ℕ} {before row : Row}

lemma getD_mem_outs {m : MLP} {i : ℕ} (hi : i < m.outs.length) : m.outs.getD i 0 ∈ m.outs := by
  rw [List.getD_eq_getElem _ _ hi]; exact List.getElem_mem _

lemma mlp_out_other (hwf : t.WellFormed) (hp : PairOK t W before row) {m m' : MLP}
    (hm : m ∈ t.mlps) (hm' : m' ∈ t.mlps) (hnew' : mlpIsNew W before row m' = true)
    {y : Sig} (hy : y ∈ m.outs) {f : ℕ} (hf : row f = some (.sig y)) (hne : m' ≠ m)
    (u : Hidden) (hu : u ∈ mlpUnits W before row m') : u.out f = 0 := by
  obtain ⟨j, hj⟩ := mem_mlpUnits hu
  rw [hj]
  apply Finset.sum_eq_zero
  intro i hi
  rw [ite_eq_right]
  intro hfe
  have hy' := getD_mem_outs (by simpa using hi : i < m'.outs.length)
  have h1 := new_out_feat hp hm' hnew' hy' hfe
  rw [hf] at h1
  simp only [Option.some.injEq, Name.sig.injEq] at h1
  rw [← h1] at hy'
  exact hne (eq_of_common_out hwf hm' hm hy' hy)

lemma mlp_out_self (hwf : t.WellFormed) (hp : PairOK t W before row) {m : MLP}
    (hm : m ∈ t.mlps) (hnew : mlpIsNew W before row m = true) {i : ℕ} (hi : i < m.outs.length)
    {f : ℕ} (hf : row f = some (.sig (m.outs.getD i 0))) (j : ℕ) (hj : j < m.compiled.n) :
    ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩).out f = m.compiled.wo i j := by
  rw [mlpUnits_getD _ _ _ _ _ hj]
  dsimp only
  rw [Finset.sum_eq_single_of_mem i (Finset.mem_range.2 hi)]
  · rw [ite_eq_left]
    exact rowOK_featIn_eq hp.row_ok hf
  · intro i' hi' hne
    rw [ite_eq_right]
    intro hfe
    have hi'' : i' < m.outs.length := by simpa using hi'
    have h1 := new_out_feat hp hm hnew (getD_mem_outs hi'') hfe
    rw [hf] at h1
    simp only [Option.some.injEq, Name.sig.injEq] at h1
    rw [List.getD_eq_getElem _ _ hi, List.getD_eq_getElem _ _ hi''] at h1
    exact hne (((outs_nodup hwf hm).getElem_inj_iff).1 h1.symm)

lemma decomp_mlps (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (hφ : φ 0 = 0) (hwf : t.WellFormed)
    (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {i : ℕ} (hi : i < m.outs.length)
    {f : ℕ} (hf : row f = some (.sig (m.outs.getD i 0))) :
    ((t.mlps.filter (mlpIsNew W before row)).map
        fun x => unitSum φ ψ (mlpUnits W before row x) f).sum =
      ∑ j ∈ range m.compiled.n,
        φ (m.compiled.wo i j) * ψ ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩) := by
  rw [list_sum_single ((mlps_nodup hwf).filter _) _ m (List.mem_filter.2 ⟨hm, hnew⟩)]
  · rw [unitSum_mlpUnits]
    apply Finset.sum_congr rfl
    intro j hj
    rw [mlp_out_self hwf hp hm hnew hi hf j (Finset.mem_range.1 hj)]
  · intro x hx hne
    obtain ⟨hx1, hx2⟩ := List.mem_filter.1 hx
    exact unitSum_zero φ ψ hφ _ f
      (mlp_out_other hwf hp hm hx1 hx2 (getD_mem_outs hi) hf hne)

lemma decomp_erase (φ : ℚ → ℝ) (ψ : Hidden → ℝ) (hφ : φ 0 = 0)
    (hp : PairOK t W before row) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true) {y : Sig} (hy : y ∈ m.outs)
    {f : ℕ} (hf : row f = some (.sig y)) :
    unitSum φ ψ (((List.range W).filter
        (fun f => decide (1 ≤ f) && before f != none && before f != row f)).map eraseUnit) f =
      if (before f).isSome then φ (-1) * ψ (eraseUnit f) else 0 := by
  unfold unitSum
  rw [List.map_map]
  have hnd : ((List.range W).filter
      (fun f => decide (1 ≤ f) && before f != none && before f != row f)).Nodup :=
    List.nodup_range.filter _
  by_cases hb : (before f).isSome
  · rw [ite_eq_left hb]
    have hmem : f ∈ (List.range W).filter
        (fun f => decide (1 ≤ f) && before f != none && before f != row f) := by
      rw [mem_eraseUnits]
      refine ⟨rowOK_lt_of_eq_some hp.row_ok hf, ?_, ?_, ?_⟩
      · rcases Nat.eq_zero_or_pos f with h0 | h0
        · subst h0; rw [hp.row_ok.one_zero] at hf; cases hf
        · exact h0
      · intro h; rw [h] at hb; simp at hb
      · rw [hf]; exact new_out_not_before hp hm hnew hy f
    rw [list_sum_single hnd _ f hmem]
    · simp [eraseUnit]
    · intro x _ hne
      simp only [Function.comp, eraseUnit]
      rw [ite_eq_right (fun h => hne h.symm), hφ, zero_mul]
  · rw [ite_eq_right hb]
    apply list_sum_zero
    intro x hx
    simp only [Function.comp, eraseUnit]
    rw [ite_eq_right, hφ, zero_mul]
    rintro rfl
    have := (mem_eraseUnits.1 hx).2.2.1
    apply hb
    exact Option.isSome_iff_ne_none.2 this

end New


/-! ## Stmt_unit_dots -/

lemma dot_single (W a : ℕ) (c : ℚ) (z : ℕ → ℝ) (ha : a < W) :
    dot W (fun f => if f = a then c else 0) z = c * z a := by
  unfold dot
  rw [Finset.sum_eq_single_of_mem a (Finset.mem_range.2 ha)]
  · simp
  · intro b _ hb; simp [hb]

lemma dot_placed (W n : ℕ) (F : ℕ → ℕ) (w : ℕ → ℚ) (b : ℚ) (z : ℕ → ℝ) (hW : 0 < W)
    (hF : ∀ k < n, F k < W) :
    dot W (fun f => (∑ k ∈ range n, if F k = f then w k else 0) + (if f = 0 then b else 0)) z =
      ∑ k ∈ range n, (w k : ℝ) * z (F k) + b * z 0 := by
  unfold dot
  push_cast
  simp only [apply_ite (Rat.cast : ℚ → ℝ), Rat.cast_zero, add_mul, Finset.sum_add_distrib, Finset.sum_mul,
    ite_mul, zero_mul]
  rw [Finset.sum_comm]
  congr 1
  · apply Finset.sum_congr rfl
    intro k hk
    rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_range.2 (hF k (Finset.mem_range.1 hk)))]
  · rw [Finset.sum_ite_eq', ite_eq_left (Finset.mem_range.2 hW)]

lemma placed_zero (n : ℕ) (F : ℕ → ℕ) (w : ℕ → ℚ) (b : ℚ) (hF : ∀ k < n, F k ≠ 0) :
    (∑ k ∈ range n, if F k = 0 then w k else 0) + (if (0 : ℕ) = 0 then b else 0) = b := by
  rw [Finset.sum_eq_zero]
  · simp
  · intro k hk
    exact ite_eq_right (hF k (Finset.mem_range.1 hk))

lemma placed_at (n : ℕ) (F : ℕ → ℕ) (w : ℕ → ℚ) (b : ℚ) (g : ℕ) (hg : g ≠ 0)
    (P : ℕ → Prop) [DecidablePred P] (hP : ∀ k < n, (F k = g ↔ P k)) :
    (∑ k ∈ range n, if F k = g then w k else 0) + (if g = 0 then b else 0) =
      ∑ k ∈ range n, if P k then w k else 0 := by
  rw [ite_eq_right hg, add_zero]
  apply Finset.sum_congr rfl
  intro k hk
  exact if_congr (hP k (Finset.mem_range.1 hk)) rfl rfl

lemma placed_none (n : ℕ) (F : ℕ → ℕ) (w : ℕ → ℚ) (b : ℚ) (g : ℕ) (hg : g ≠ 0)
    (hF : ∀ k < n, F k ≠ g) :
    (∑ k ∈ range n, if F k = g then w k else 0) + (if g = 0 then b else 0) = 0 := by
  rw [ite_eq_right hg, add_zero]
  apply Finset.sum_eq_zero
  intro k hk
  exact ite_eq_right (hF k (Finset.mem_range.1 hk))

end DecompAux

open DecompAux

theorem stmt_decomp : Stmt_decomp := by
  intro φ ψ t W before row hφ hwf hp
  refine ⟨fun f hf => decomp_keep φ ψ t W before row hφ hp f hf, ?_⟩
  intro f m i hm hnew hi hf
  have hy := getD_mem_outs hi
  unfold layerUnits
  rw [unitSum_cons, unitSum_append, unitSum_flatMap, oneUnit_out_new hp hm hnew hy hf,
    decomp_mlps φ ψ hφ hwf hp hm hnew hi hf, decomp_erase φ ψ hφ hp hm hnew hy hf]
  ring

lemma getD_mem_list {l : List ℕ} {i : ℕ} (hi : i < l.length) : l.getD i 0 ∈ l := by
  rw [List.getD_eq_getElem _ _ hi]; exact List.getElem_mem _

lemma unit_dots_mlp (t : Trace) (W : ℕ) (before row : Row) (m : MLP) (j : ℕ)
    (hp : PairOK t W before row) (hm : m ∈ t.mlps) (hnew : mlpIsNew W before row m = true)
    (hj : j < m.compiled.n) :
    let u := (mlpUnits W before row m).getD j ⟨0, 0, 0⟩
    (∀ z, dot W u.gate z = gateDot W before m j z ∧ dot W u.val z = valDot W before m j z) ∧
    (u.gate 0 = m.compiled.gb j ∧ u.val 0 = m.compiled.vb j) ∧
    (∀ g s, before g = some (.sig s) →
      u.gate g = m.compiled.gOn j s ∧ u.val g = m.compiled.vOn j s) ∧
    (∀ g, 1 ≤ g → (∀ s, before g ≠ some (.sig s)) → u.gate g = 0 ∧ u.val g = 0) := by
  dsimp only
  rw [mlpUnits_getD _ _ _ _ _ hj]
  dsimp only
  have hW : 0 < W := rowOK_W_pos hp.before_ok
  have hins : ∀ k < m.ins.length, featIn W before (.sig (m.ins.getD k 0)) < W ∧
      before (featIn W before (.sig (m.ins.getD k 0))) = some (.sig (m.ins.getD k 0)) :=
    fun k hk => new_in_spec hp hm hnew (getD_mem_list hk)
  have hne0 : ∀ k < m.ins.length, featIn W before (.sig (m.ins.getD k 0)) ≠ 0 := by
    intro k hk h
    have h1 := (hins k hk).2
    rw [h, hp.before_ok.one_zero] at h1
    cases h1
  have hkey : ∀ g s, before g = some (.sig s) → ∀ k < m.ins.length,
      (featIn W before (.sig (m.ins.getD k 0)) = g ↔ m.ins.getD k 0 = s) := by
    intro g s hg k hk
    constructor
    · intro h
      have h1 := (hins k hk).2
      rw [h, hg] at h1
      simp only [Option.some.injEq, Name.sig.injEq] at h1
      exact h1.symm
    · intro h
      rw [h]
      exact rowOK_featIn_eq hp.before_ok hg
  have hg0 : ∀ g s, before g = some (.sig s) → g ≠ 0 := by
    rintro g s hg rfl
    rw [hp.before_ok.one_zero] at hg
    cases hg
  refine ⟨fun z => ⟨?_, ?_⟩, ⟨?_, ?_⟩, fun g s hg => ⟨?_, ?_⟩, fun g hg1 hg2 => ⟨?_, ?_⟩⟩
  · exact dot_placed W m.ins.length (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ z hW
      (fun k hk => (hins k hk).1)
  · exact dot_placed W m.ins.length (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ z hW
      (fun k hk => (hins k hk).1)
  · exact placed_zero _ (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ hne0
  · exact placed_zero _ (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ hne0
  · unfold Gated.gOn
    rw [compiled_ins]
    exact placed_at _ (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ g (hg0 g s hg)
      (fun k => m.ins.getD k 0 = s) (hkey g s hg)
  · unfold Gated.vOn
    rw [compiled_ins]
    exact placed_at _ (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ g (hg0 g s hg)
      (fun k => m.ins.getD k 0 = s) (hkey g s hg)
  · refine placed_none _ (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ g (by omega) ?_
    intro k hk h
    exact hg2 _ (h ▸ (hins k hk).2)
  · refine placed_none _ (fun k => featIn W before (.sig (m.ins.getD k 0))) _ _ g (by omega) ?_
    intro k hk h
    exact hg2 _ (h ▸ (hins k hk).2)

theorem stmt_unit_dots : Stmt_unit_dots := by
  refine ⟨?_, ?_, ?_⟩
  · intro W before row z hW
    constructor
    · show dot W (fun f => if f = 0 then (32 : ℚ) else 0) z = 32 * z 0
      rw [dot_single _ _ _ _ hW]; push_cast; ring
    · show dot W (fun f => if f = 0 then (1 / 32 : ℚ) else 0) z = z 0 / 32
      rw [dot_single _ _ _ _ hW]; push_cast; ring
  · intro W f z hf hfW
    constructor
    · show dot W (fun f' => if f' = 0 then (32 : ℚ) else 0) z = 32 * z 0
      rw [dot_single _ _ _ _ (by omega)]; push_cast; ring
    · show dot W (fun f' => if f' = f then (1 / 32 : ℚ) else 0) z = z f / 32
      rw [dot_single _ _ _ _ hfW]; push_cast; ring
  · intro t W before row m j _ hp hm hnew hj
    exact unit_dots_mlp t W before row m j hp hm hnew hj

end

end Reifier

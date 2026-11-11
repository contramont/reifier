import ReifierProofs.Proofs.GridAuxLayout

/-!
# `tokenGrid` in closed form (helper for `Proofs/Grid.lean`)
-/

namespace Reifier
namespace GridAux

open Layout

set_option linter.unusedSectionVars false
set_option linter.deprecated false

/-! ## Generic list facts -/

lemma foldl_map_comm {α β : Type*} (g : β → α → α) (l : List β) : ∀ (rows : List α),
    l.foldl (fun rows yi => rows.map (g yi)) rows =
      rows.map (fun r => l.foldl (fun r yi => g yi r) r) := by
  induction l with
  | nil => intro rows; simp
  | cons y l ih =>
    intro rows
    simp only [List.foldl_cons]
    rw [ih, List.map_map]
    rfl

/-! ## `held` -/

/-- The held rows, as a function of depth. -/
def Hrow (M : ℕ → Row) : ℕ → Row
  | 0 => M 0
  | d + 1 => heldRow (Hrow M d) (M (d + 1))

lemma heldRows_eq (M : ℕ → Row) (n : ℕ) : ∀ k,
    heldRows (Hrow M k) ((List.range n).map (fun i => M (i + k + 1))) =
      (List.range n).map (fun i => Hrow M (i + k + 1)) := by
  induction n with
  | zero => intro k; simp [heldRows]
  | succ n ih =>
    intro k
    rw [List.range_succ_eq_map]
    simp only [List.map_cons, List.map_map, heldRows]
    have h1 : heldRow (Hrow M k) (M (0 + k + 1)) = Hrow M (k + 1) := by
      simp only [Nat.zero_add]; rfl
    rw [h1]
    have h2 := ih (k + 1)
    have e1 : ((fun i => M (i + k + 1)) ∘ Nat.succ) = fun i => M (i + (k + 1) + 1) := by
      funext i; simp only [Function.comp, Nat.succ_eq_add_one]; congr 1; omega
    have e2 : ((fun i => Hrow M (i + k + 1)) ∘ Nat.succ) = fun i => Hrow M (i + (k + 1) + 1) := by
      funext i; simp only [Function.comp, Nat.succ_eq_add_one]; congr 1; omega
    rw [e1, e2, h2]
    simp

lemma held_rows_eq (M : ℕ → Row) (W n : ℕ) :
    (held { width := W, rows := (List.range (n + 1)).map M }).rows =
      (List.range (n + 1)).map (Hrow M) := by
  simp only [held]
  rw [List.range_succ_eq_map]
  simp only [List.map_cons, List.map_map]
  have e1 : (M ∘ Nat.succ) = fun i => M (i + 0 + 1) := by
    funext i; simp
  have e2 : (Hrow M ∘ Nat.succ) = fun i => Hrow M (i + 0 + 1) := by
    funext i; simp
  rw [e1, e2, ← heldRows_eq M n 0]
  rfl

lemma held_width (g : Grid) : (held g).width = g.width := rfl

section HeldFacts

variable (M : ℕ → Row)

lemma Hrow_succ (d : ℕ) (f : ℕ) :
    Hrow M (d + 1) f = match M (d + 1) f with
      | some s => some s
      | none => Hrow M d f := rfl

lemma Hrow_of_M {d f : ℕ} {n : Name} (h : M d f = some n) : Hrow M d f = some n := by
  cases d with
  | zero => exact h
  | succ d => rw [Hrow_succ, h]

lemma M_of_Hrow {d f : ℕ} {n : Name} (h : Hrow M d f = some n) : ∃ d' ≤ d, M d' f = some n := by
  induction d with
  | zero => exact ⟨0, le_rfl, h⟩
  | succ d ih =>
    rw [Hrow_succ] at h
    split at h
    · rename_i s hs
      exact ⟨d + 1, le_rfl, hs.trans h⟩
    · obtain ⟨d', hd', h'⟩ := ih h
      exact ⟨d', by omega, h'⟩

lemma Hrow_succ_cases {d f : ℕ} {n : Name} (h : Hrow M (d + 1) f = some n) :
    M (d + 1) f = some n ∨ Hrow M d f = some n := by
  rw [Hrow_succ] at h
  split at h
  · rename_i s hs
    exact Or.inl (hs.trans h)
  · exact Or.inr h

lemma Hrow_isSome {d f : ℕ} (h : (Hrow M d f).isSome) : (Hrow M (d + 1) f).isSome := by
  rw [Hrow_succ]
  split
  · rfl
  · exact h

lemma Hrow_changed {d f : ℕ} (h : Hrow M (d + 1) f ≠ Hrow M d f) :
    ∃ n, M (d + 1) f = some n ∧ Hrow M (d + 1) f = some n := by
  rw [Hrow_succ] at h ⊢
  split at h
  · rename_i s hs
    exact ⟨s, hs, rfl⟩
  · exact absurd rfl h

lemma Hrow_zero_feat (h0 : ∀ d, M d 0 = some .one) (d : ℕ) : Hrow M d 0 = some .one :=
  Hrow_of_M M (h0 d)

end HeldFacts

/-! ## `moveOut` -/

lemma moveOut_unique {W i : ℕ} {y : Sig} {r : Row} {g : ℕ} (hg : g < W)
    (hrg : r g = some (.sig y)) (huniq : ∀ g', r g' = some (.sig y) → g' = g) :
    moveOut W i y r = Function.update (Function.update r g none) (1 + i) (some (.sig y)) := by
  have hfind : (List.range W).find? (fun f => r f == some (.sig y)) = some g := by
    cases h : (List.range W).find? (fun f => r f == some (.sig y)) with
    | none =>
      rw [List.find?_eq_none] at h
      exact absurd (by simp [hrg]) (h g (List.mem_range.2 hg))
    | some g' =>
      have := List.find?_some h
      simp only [beq_iff_eq] at this
      rw [huniq g' this]
  unfold moveOut
  rw [hfind]

lemma moveOut_absent {W i : ℕ} {y : Sig} {r : Row} (habs : ∀ g, r g ≠ some (.sig y)) :
    moveOut W i y r = r := by
  have hfind : (List.range W).find? (fun f => r f == some (.sig y)) = none := by
    rw [List.find?_eq_none]
    intro g _
    simp [habs g]
  unfold moveOut
  rw [hfind]

/-! ## The rows of `tokenGrid`, in closed form -/

/-- What a cell beyond the code holds after the first `k` outputs are moved out. -/
def highCell (t : Trace) (k : ℕ) : Option Name → Option Name
  | some (.sig x) => if x ∈ t.outputs ∧ t.outputs.idxOf x < k then none else some (.sig x)
  | o => o

open Classical in
/-- Row `d` of the token grid before `held`, after the first `k` outputs are moved out. -/
noncomputable def part (t : Trace) (d k : ℕ) : Row := fun f =>
  if f = 0 then some .one
  else if f ≤ t.outputs.length then
    (if f - 1 < k ∧ alive t (t.outputs.getD (f - 1) 0) d then
      some (.sig (t.outputs.getD (f - 1) 0)) else some (.code (f - 1)))
  else highCell t k ((S t d).row (f - t.outputs.length))

lemma highCell_zero (t : Trace) (o : Option Name) : highCell t 0 o = o := by
  unfold highCell
  split
  · simp
  · rfl

lemma highCell_some {t : Trace} {k : ℕ} {o : Option Name} {n : Name} (h : highCell t k o = some n) :
    o = some n := by
  unfold highCell at h
  split at h
  · split_ifs at h
    exact h
  · exact h

lemma highCell_sig {t : Trace} {k : ℕ} {o : Option Name} {z : Sig}
    (h : highCell t k o = some (.sig z)) : ¬ (z ∈ t.outputs ∧ t.outputs.idxOf z < k) := by
  unfold highCell at h
  split at h
  · split_ifs at h with hc
    simp only [Option.some.injEq, Name.sig.injEq] at h
    subst h
    exact hc
  · rename_i hn
    exact absurd h (by intro he; exact hn z he)

lemma part_zero_k (t : Trace) (d : ℕ) : part t d 0 = tokenRow t.outputs.length (S t d).row := by
  funext f
  unfold part tokenRow
  split_ifs with h1 h2 h3
  · rfl
  · omega
  · rfl
  · rw [highCell_zero]

lemma highCell_sig_eq (t : Trace) (k : ℕ) (x : Sig) :
    highCell t k (some (.sig x)) =
      if x ∈ t.outputs ∧ t.outputs.idxOf x < k then none else some (.sig x) := rfl

lemma part_eq_zero (t : Trace) (d k : ℕ) : part t d k 0 = some .one := by
  unfold part; simp

open Classical in
lemma part_eq_code (t : Trace) (d k f : ℕ) (h1 : 1 ≤ f) (h2 : f ≤ t.outputs.length) :
    part t d k f = (if f - 1 < k ∧ alive t (t.outputs.getD (f - 1) 0) d then
      some (Name.sig (t.outputs.getD (f - 1) 0)) else some (Name.code (f - 1))) := by
  unfold part
  rw [if_neg (by omega), if_pos h2]

lemma part_eq_high (t : Trace) (d k f : ℕ) (h : t.outputs.length < f) :
    part t d k f = highCell t k ((S t d).row (f - t.outputs.length)) := by
  unfold part
  rw [if_neg (by omega), if_neg (by omega)]

lemma trich (b f : ℕ) : f = 0 ∨ (1 ≤ f ∧ f ≤ b) ∨ b < f := by omega

section WF

variable {t : Trace} (hp : t.IsProgram)
include hp

lemma outputs_getD_mem {i : ℕ} (hi : i < t.outputs.length) : t.outputs.getD i 0 ∈ t.outputs := by
  rw [List.getD_eq_getElem _ _ hi]; exact List.getElem_mem _

lemma outputs_idxOf_getD {i : ℕ} (hi : i < t.outputs.length) :
    t.outputs.idxOf (t.outputs.getD i 0) = i := by
  rw [List.getD_eq_getElem _ _ hi]
  exact List.Nodup.idxOf_getElem hp.outputs_nodup i hi

lemma outputs_getD_idxOf {x : Sig} (hx : x ∈ t.outputs) :
    t.outputs.getD (t.outputs.idxOf x) 0 = x := by
  have hlt := List.idxOf_lt_length_iff.2 hx
  rw [List.getD_eq_getElem _ _ hlt, List.getElem_idxOf hlt]

lemma outputs_getD_inj {i j : ℕ} (hi : i < t.outputs.length) (hj : j < t.outputs.length)
    (h : t.outputs.getD i 0 = t.outputs.getD j 0) : i = j := by
  rw [← outputs_idxOf_getD hp hi, ← outputs_idxOf_getD hp hj, h]

lemma highCell_succ {k : ℕ} (_hk : k < t.outputs.length) (o : Option Name)
    (h : ∀ x, o = some (.sig x) → x ≠ t.outputs.getD k 0) :
    highCell t k o = highCell t (k + 1) o := by
  rcases o with _ | ⟨_ | x | i⟩
  · rfl
  · rfl
  · rw [highCell_sig_eq, highCell_sig_eq]
    have hne := h x rfl
    have hidx : x ∈ t.outputs → t.outputs.idxOf x ≠ k := by
      intro hxo he
      apply hne
      rw [← he, outputs_getD_idxOf hp hxo]
    by_cases hxo : x ∈ t.outputs
    · have := hidx hxo
      by_cases hlt : t.outputs.idxOf x < k
      · rw [if_pos ⟨hxo, hlt⟩, if_pos ⟨hxo, by omega⟩]
      · rw [if_neg (fun h => hlt h.2), if_neg (fun h => by have := h.2; omega)]
    · rw [if_neg (fun h => hxo h.1), if_neg (fun h => hxo h.1)]
  · rfl

lemma part_step {d k : ℕ} (hd : d ≤ maxDepth t) (hk : k < t.outputs.length) :
    moveOut ((layout t).width + t.outputs.length) k (t.outputs.getD k 0) (part t d k) =
      part t d (k + 1) := by
  have hw := hp.toWellFormed
  have hI := S_inv hw d
  have hymem : t.outputs.getD k 0 ∈ t.outputs := outputs_getD_mem hp hk
  have hyidx : t.outputs.idxOf (t.outputs.getD k 0) = k := outputs_idxOf_getD hp hk
  by_cases hal : alive t (t.outputs.getD k 0) d
  · obtain ⟨p1, p2, p3⟩ := hI.present _ hal
    have hwid : (S t d).width ≤ (layout t).width := by
      rw [layout_width]; exact S_width_mono hw hd
    have hgb : (S t d).feat (t.outputs.getD k 0) + t.outputs.length - t.outputs.length =
        (S t d).feat (t.outputs.getD k 0) := by omega
    have hpg : part t d k ((S t d).feat (t.outputs.getD k 0) + t.outputs.length) =
        some (.sig (t.outputs.getD k 0)) := by
      rw [part_eq_high _ _ _ _ (by omega), hgb, p3, highCell_sig_eq,
        if_neg (fun h => by have := h.2; omega)]
    have huniq : ∀ g', part t d k g' = some (.sig (t.outputs.getD k 0)) →
        g' = (S t d).feat (t.outputs.getD k 0) + t.outputs.length := by
      intro g' h'
      rcases trich t.outputs.length g' with c | ⟨c1, c2⟩ | c
      · subst c; rw [part_eq_zero] at h'; simp at h'
      · rw [part_eq_code _ _ _ _ c1 c2] at h'
        split_ifs at h' with c3
        · simp only [Option.some.injEq, Name.sig.injEq] at h'
          have := outputs_getD_inj hp (by omega) hk h'
          omega
        · simp at h'
      · rw [part_eq_high _ _ _ _ c] at h'
        have h2 := highCell_some h'
        have := (hI.sound _ _ h2).2
        omega
    rw [moveOut_unique (by omega) hpg huniq]
    funext f
    by_cases hf1 : f = 1 + k
    · subst hf1
      rw [Function.update_self, part_eq_code _ _ _ _ (by omega) (by omega)]
      have e : 1 + k - 1 = k := by omega
      rw [if_pos ⟨by omega, by rw [e]; exact hal⟩, e]
    · rw [Function.update_of_ne hf1]
      by_cases hfg : f = (S t d).feat (t.outputs.getD k 0) + t.outputs.length
      · subst hfg
        rw [Function.update_self, part_eq_high _ _ _ _ (by omega), hgb, p3, highCell_sig_eq,
          if_pos ⟨hymem, by omega⟩]
      · rw [Function.update_of_ne hfg]
        rcases trich t.outputs.length f with c | ⟨c1, c2⟩ | c
        · subst c; rw [part_eq_zero, part_eq_zero]
        · rw [part_eq_code _ _ _ _ c1 c2, part_eq_code _ _ _ _ c1 c2]
          have hfk : f - 1 ≠ k := by omega
          by_cases hA : alive t (t.outputs.getD (f - 1) 0) d
          · by_cases hlt : f - 1 < k
            · rw [if_pos ⟨hlt, hA⟩, if_pos ⟨by omega, hA⟩]
            · rw [if_neg (fun h => hlt h.1), if_neg (fun h => by have := h.1; omega)]
          · rw [if_neg (fun h => hA h.2), if_neg (fun h => hA h.2)]
        · rw [part_eq_high _ _ _ _ c, part_eq_high _ _ _ _ c]
          apply highCell_succ hp hk
          rintro x hx rfl
          have := (hI.sound _ _ hx).2
          omega
  · have habs : ∀ g, part t d k g ≠ some (.sig (t.outputs.getD k 0)) := by
      intro g' h'
      rcases trich t.outputs.length g' with c | ⟨c1, c2⟩ | c
      · subst c; rw [part_eq_zero] at h'; simp at h'
      · rw [part_eq_code _ _ _ _ c1 c2] at h'
        split_ifs at h' with c3
        · simp only [Option.some.injEq, Name.sig.injEq] at h'
          have := outputs_getD_inj hp (by omega) hk h'
          omega
        · simp at h'
      · rw [part_eq_high _ _ _ _ c] at h'
        have h2 := highCell_some h'
        exact hal (hI.sound _ _ h2).1
    rw [moveOut_absent habs]
    funext f
    rcases trich t.outputs.length f with c | ⟨c1, c2⟩ | c
    · subst c; rw [part_eq_zero, part_eq_zero]
    · rw [part_eq_code _ _ _ _ c1 c2, part_eq_code _ _ _ _ c1 c2]
      by_cases hA : alive t (t.outputs.getD (f - 1) 0) d
      · have hfk : f - 1 ≠ k := by
          intro he; rw [he] at hA; exact hal hA
        by_cases hlt : f - 1 < k
        · rw [if_pos ⟨hlt, hA⟩, if_pos ⟨by omega, hA⟩]
        · rw [if_neg (fun h => hlt h.1), if_neg (fun h => by have := h.1; omega)]
      · rw [if_neg (fun h => hA h.2), if_neg (fun h => hA h.2)]
    · rw [part_eq_high _ _ _ _ c, part_eq_high _ _ _ _ c]
      apply highCell_succ hp hk
      rintro x hx rfl
      exact hal (hI.sound _ _ hx).1

lemma mvAll_eq {d : ℕ} (hd : d ≤ maxDepth t) :
    t.outputs.zipIdx.foldl
      (fun r yi => moveOut ((layout t).width + t.outputs.length) yi.2 yi.1 r)
      (tokenRow t.outputs.length (S t d).row) = part t d t.outputs.length := by
  have key : ∀ k, k ≤ t.outputs.length → (t.outputs.take k).zipIdx.foldl
      (fun r yi => moveOut ((layout t).width + t.outputs.length) yi.2 yi.1 r)
      (tokenRow t.outputs.length (S t d).row) = part t d k := by
    intro k
    induction k with
    | zero => intro _; simp [part_zero_k]
    | succ k ih =>
      intro hk
      rw [List.take_succ_eq_append_getElem (by omega), List.zipIdx_append, List.foldl_append,
        ih (by omega), List.zipIdx_singleton, List.length_take, min_eq_left (by omega)]
      simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add]
      rw [← List.getD_eq_getElem _ 0]
      exact part_step hp hd (by omega)
  have := key t.outputs.length le_rfl
  rwa [List.take_length] at this

/-- Row `d` of the token grid before `held`. -/
noncomputable abbrev Mr (t : Trace) (d : ℕ) : Row := part t d t.outputs.length

lemma tokenGrid_rows :
    (tokenGrid t).rows = (List.range (maxDepth t + 1)).map (Hrow (Mr t)) := by
  have hc := foldl_map_comm (fun (yi : Sig × ℕ) (r : Row) =>
    moveOut ((layout t).width + t.outputs.length) yi.2 yi.1 r) t.outputs.zipIdx
    ((layout t).rows.map (tokenRow t.outputs.length))
  have hrows : t.outputs.zipIdx.foldl
      (fun rows yi => rows.map (moveOut ((layout t).width + t.outputs.length) yi.2 yi.1))
      ((layout t).rows.map (tokenRow t.outputs.length)) =
      (List.range (maxDepth t + 1)).map (Mr t) := by
    rw [hc, layout_rows, List.map_map, List.map_map]
    apply List.map_congr_left
    intro d hd
    rw [List.mem_range] at hd
    exact mvAll_eq hp (by omega)
  show (held { width := _, rows := _ }).rows = _
  rw [hrows]
  exact held_rows_eq _ _ _

end WF

end GridAux
end Reifier

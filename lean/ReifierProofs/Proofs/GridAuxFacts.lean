import ReifierProofs.Proofs.GridAuxToken

/-!
# What each row of the token grid holds (helper for `Proofs/Grid.lean`)
-/

namespace Reifier
namespace GridAux

open Layout

set_option linter.unusedSectionVars false
set_option linter.deprecated false

/-- The width of the token grid. -/
abbrev gW (t : Trace) : ℕ := (layout t).width + t.outputs.length

/-- Where a signal sits in the token grid while it is alive. -/
def pos (t : Trace) (x : Sig) : ℕ :=
  if x ∈ t.outputs then 1 + t.outputs.idxOf x else featAt t x + t.outputs.length

/-- Where a name sits in the token grid. -/
def posN (t : Trace) : Name → ℕ
  | .one => 0
  | .code i => 1 + i
  | .sig x => pos t x

/-- The names a row of a grid for `t` may hold. -/
def NameOK (t : Trace) (n : Name) : Prop :=
  n = .one ∨ (∃ i, n = .code i ∧ i < t.outputs.length) ∨
    (∃ s, n = .sig s ∧ (s ∈ t.inputs ∨ ∃ m ∈ t.mlps, s ∈ m.outs))

lemma inRow_of {W : ℕ} {r : Row} {s : Name} {f : ℕ} (hf : f < W) (h : r f = some s) :
    inRow W r s = true := by
  unfold inRow
  rw [List.any_eq_true]
  exact ⟨f, List.mem_range.2 hf, by simp [h]⟩

lemma of_inRow {W : ℕ} {r : Row} {s : Name} (h : inRow W r s = true) :
    ∃ f, f < W ∧ r f = some s := by
  unfold inRow at h
  rw [List.any_eq_true] at h
  obtain ⟨f, hf, h⟩ := h
  exact ⟨f, List.mem_range.1 hf, by simpa using h⟩

lemma Mr_zero (t : Trace) (d : ℕ) : Mr t d 0 = some .one := part_eq_zero _ _ _

section WF

variable {t : Trace} (hp : t.IsProgram)
include hp

lemma layout_width_pos : 1 ≤ (layout t).width := by
  rw [layout_width]
  exact (S_inv hp.toWellFormed _).width_pos

lemma output_order {y : Sig} (hy : y ∈ t.outputs) : y ∈ order t :=
  mem_order.2 (Or.inr (hp.outputs_made y hy))

lemma output_alive_max {y : Sig} (hy : y ∈ t.outputs) : alive t y (maxDepth t) := by
  refine ⟨output_order hp hy, depth_le_maxDepth (output_order hp hy), ?_⟩
  rw [freedAt_output t hy]
  exact geOpt_none _

lemma output_not_alive_zero {y : Sig} (hy : y ∈ t.outputs) : ¬ alive t y 0 := by
  intro h
  rw [alive_zero_iff hp.toWellFormed] at h
  obtain ⟨m, hm, hym⟩ := hp.outputs_made y hy
  exact wf_in_not_out hp.toWellFormed h m hm hym

lemma pos_output {i : ℕ} (hi : i < t.outputs.length) : pos t (t.outputs.getD i 0) = 1 + i := by
  unfold pos
  rw [if_pos (outputs_getD_mem hp hi), outputs_idxOf_getD hp hi]

/-- (M2) Where a name is, that it is below the width, and that it is a name of the trace. -/
lemma Mr_some {d f : ℕ} {n : Name} (hd : d ≤ maxDepth t) (h : Mr t d f = some n) :
    f = posN t n ∧ f < gW t ∧ NameOK t n := by
  unfold gW
  have hw := hp.toWellFormed
  have hI := S_inv hw d
  have hlw := layout_width_pos hp
  rcases trich t.outputs.length f with c | ⟨c1, c2⟩ | c
  · subst c
    rw [Mr_zero] at h
    cases h
    exact ⟨rfl, by omega, Or.inl rfl⟩
  · simp only [Mr] at h
    rw [part_eq_code _ _ _ _ c1 c2] at h
    split_ifs at h with c3
    · cases h
      have hlt : f - 1 < t.outputs.length := by omega
      refine ⟨?_, by omega, ?_⟩
      · show f = pos t _
        rw [pos_output hp hlt]; omega
      · exact Or.inr (Or.inr ⟨_, rfl, Or.inr (hp.outputs_made _ (outputs_getD_mem hp hlt))⟩)
    · cases h
      exact ⟨by show f = 1 + (f - 1); omega, by omega, Or.inr (Or.inl ⟨_, rfl, by omega⟩)⟩
  · simp only [Mr] at h
    rw [part_eq_high _ _ _ _ c] at h
    have h1 := highCell_some h
    rcases hI.names _ _ h1 with ⟨_, h0⟩ | ⟨x, rfl⟩
    · omega
    · have hx := highCell_sig h
      have hxo : x ∉ t.outputs := by
        intro hxo
        exact hx ⟨hxo, List.idxOf_lt_length_iff.2 hxo⟩
      obtain ⟨hal, hfe⟩ := hI.sound _ _ h1
      have hlt := hI.lt_width h1
      have hwid : (S t d).width ≤ (layout t).width := by
        rw [layout_width]; exact S_width_mono hw hd
      refine ⟨?_, by omega, ?_⟩
      · show f = pos t x
        unfold pos
        rw [if_neg hxo, ← feat_eq_featAt hw hal.2.1, hfe]
        omega
      · exact Or.inr (Or.inr ⟨x, rfl, mem_order.1 hal.1⟩)

/-- (M3) A signal in a row is alive there. -/
lemma Mr_alive {d f : ℕ} {x : Sig} (h : Mr t d f = some (.sig x)) : alive t x d := by
  have hI := S_inv hp.toWellFormed d
  rcases trich t.outputs.length f with c | ⟨c1, c2⟩ | c
  · subst c; rw [Mr_zero] at h; cases h
  · simp only [Mr] at h
    rw [part_eq_code _ _ _ _ c1 c2] at h
    split_ifs at h with c3
    · cases h; exact c3.2
    · cases h
  · simp only [Mr] at h
    rw [part_eq_high _ _ _ _ c] at h
    exact (hI.sound _ _ (highCell_some h)).1

/-- (M4) An alive signal is in its place. -/
lemma Mr_pos {d : ℕ} {x : Sig} (h : alive t x d) : Mr t d (pos t x) = some (.sig x) := by
  have hw := hp.toWellFormed
  have hI := S_inv hw d
  obtain ⟨p1, p2, p3⟩ := hI.present x h
  simp only [Mr]
  by_cases hxo : x ∈ t.outputs
  · have hlt := List.idxOf_lt_length_iff.2 hxo
    unfold pos
    rw [if_pos hxo, part_eq_code _ _ _ _ (by omega) (by omega)]
    have e : 1 + t.outputs.idxOf x - 1 = t.outputs.idxOf x := by omega
    rw [e, outputs_getD_idxOf hp hxo, if_pos ⟨hlt, h⟩]
  · unfold pos
    rw [if_neg hxo, ← feat_eq_featAt hw h.2.1, part_eq_high _ _ _ _ (by omega)]
    have e : (S t d).feat x + t.outputs.length - t.outputs.length = (S t d).feat x := by omega
    rw [e, p3, highCell_sig_eq, if_neg (fun h => hxo h.1)]

/-- (M5) The code bit of an output not yet made. -/
lemma Mr_code {d i : ℕ} (hi : i < t.outputs.length) (h : ¬ alive t (t.outputs.getD i 0) d) :
    Mr t d (1 + i) = some (.code i) := by
  simp only [Mr]
  rw [part_eq_code _ _ _ _ (by omega) (by omega)]
  have e : 1 + i - 1 = i := by omega
  rw [e, if_neg (fun h' => h h'.2)]

lemma pos_lt {d : ℕ} (hd : d ≤ maxDepth t) {x : Sig} (h : alive t x d) : pos t x < gW t :=
  (Mr_some hp hd (Mr_pos hp h)).2.1

/-! ## The held rows -/

/-- Row `d` of the token grid. -/
noncomputable abbrev H (t : Trace) : ℕ → Row := Hrow (Mr t)

lemma H_some {d f : ℕ} {n : Name} (hd : d ≤ maxDepth t) (h : H t d f = some n) :
    f = posN t n ∧ f < gW t ∧ NameOK t n := by
  obtain ⟨d', hd', h'⟩ := M_of_Hrow _ h
  exact Mr_some hp (by omega) h'

lemma H_not_in {d : ℕ} {y : Sig} (hy : d < depth t y) : inRow (gW t) (H t d) (.sig y) = false := by
  by_contra hc
  rw [Bool.not_eq_false] at hc
  obtain ⟨g, _, hg⟩ := of_inRow hc
  obtain ⟨d', hd', h'⟩ := M_of_Hrow _ hg
  have := (Mr_alive hp h').2.1
  omega

lemma H_in {d : ℕ} (hd : d ≤ maxDepth t) {x : Sig} (h : alive t x d) :
    inRow (gW t) (H t d) (.sig x) = true :=
  inRow_of (pos_lt hp hd h) (Hrow_of_M _ (Mr_pos hp h))

lemma rowOK {d : ℕ} (hd : d ≤ maxDepth t) : RowOK t (gW t) (H t d) where
  one_zero := Hrow_zero_feat _ (Mr_zero t) d
  one_only := fun f h => (H_some hp hd h).1
  none_beyond := by
    intro f hf
    cases h : H t d f with
    | none => rfl
    | some n => have := (H_some hp hd h).2.1; omega
  nodup := by
    intro f g hfg hs
    obtain ⟨n, hn⟩ := Option.isSome_iff_exists.1 hs
    rw [hn] at hfg
    rw [(H_some hp hd hn).1, (H_some hp hd hfg.symm).1]
  names := fun f n h => (H_some hp hd h).2.2

lemma isNew_dM {d : ℕ} (hd : d + 1 ≤ maxDepth t) {m : MLP} (hm : m ∈ t.mlps)
    (hnew : mlpIsNew (gW t) (H t d) (H t (d + 1)) m = true) : dM t m = d + 1 := by
  have hw := hp.toWellFormed
  unfold mlpIsNew at hnew
  split at hnew
  · rename_i y0 hy0
    obtain ⟨rest, hrest⟩ := List.head?_eq_some_iff.1 hy0
    have hy0m : y0 ∈ m.outs := by rw [hrest]; simp
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hnew
    obtain ⟨h1, h2⟩ := hnew
    obtain ⟨g, hg, hHg⟩ := of_inRow h1
    have hal : alive t y0 (d + 1) := by
      rcases Hrow_succ_cases _ hHg with h | h
      · exact Mr_alive hp h
      · rw [inRow_of hg h] at h2; cases h2
    have hnal : ¬ alive t y0 d := by
      intro ha
      rw [H_in hp (by omega) ha] at h2
      cases h2
    rw [← depth_out hw hm hy0m]
    exact alive_succ_not hw hal hnal
  · cases hnew

lemma dM_isNew {d : ℕ} (hd : d + 1 ≤ maxDepth t) {m : MLP} (hm : m ∈ t.mlps)
    (hdm : dM t m = d + 1) : mlpIsNew (gW t) (H t d) (H t (d + 1)) m = true := by
  have hw := hp.toWellFormed
  obtain ⟨y0, rest, hrest⟩ := List.exists_cons_of_ne_nil (hw.outs_ne m hm)
  have hy0m : y0 ∈ m.outs := by rw [hrest]; simp
  have hdy : depth t y0 = d + 1 := by rw [depth_out hw hm hy0m, hdm]
  have hal : alive t y0 (d + 1) := by
    have := alive_depth hw (mem_order.2 (Or.inr ⟨m, hm, hy0m⟩))
    rwa [hdy] at this
  unfold mlpIsNew
  rw [hrest]
  simp only [List.head?_cons, Bool.and_eq_true, Bool.not_eq_true']
  exact ⟨H_in hp hd hal, H_not_in hp (by omega)⟩

lemma pairOK {d : ℕ} (hd : d + 1 ≤ maxDepth t) :
    PairOK t (gW t) (H t d) (H t (d + 1)) where
  before_ok := rowOK hp (by omega)
  row_ok := rowOK hp hd
  held := fun f h => Hrow_isSome _ h
  changed := by
    have hw := hp.toWellFormed
    intro f hf
    obtain ⟨n, hMn, hHn⟩ := Hrow_changed _ hf
    obtain ⟨hpos, _, hname⟩ := Mr_some hp hd hMn
    rcases n with _ | y | i
    · -- `ONE`
      exfalso; apply hf
      simp only [posN] at hpos
      subst hpos
      exact (Hrow_zero_feat _ (Mr_zero t) (d + 1)).trans (Hrow_zero_feat _ (Mr_zero t) d).symm
    · -- a signal
      have hal := Mr_alive hp hMn
      have hnal : ¬ alive t y d := by
        intro ha
        apply hf
        simp only [posN] at hpos
        show Hrow (Mr t) (d + 1) f = Hrow (Mr t) d f
        rw [hHn, Hrow_of_M _ (hpos ▸ Mr_pos hp ha)]
      have hdy := alive_succ_not hw hal hnal
      have hyo := hal.1
      rcases mem_order.1 hyo with hin | ⟨m, hm, hym⟩
      · rw [depth_input hw hin] at hdy; omega
      · refine ⟨m, hm, dM_isNew hp hd hm ?_, y, hym, hHn⟩
        rw [← depth_out hw hm hym, hdy]
    · -- a code bit
      exfalso
      have hi : i < t.outputs.length := by
        rcases hname with h | ⟨j, hj, hjb⟩ | ⟨s, hs, _⟩
        · cases h
        · cases hj; exact hjb
        · cases hs
      simp only [posN] at hpos
      subst hpos
      have hn1 : ¬ alive t (t.outputs.getD i 0) (d + 1) := by
        intro ha
        have := Mr_pos hp ha
        rw [pos_output hp hi, hMn] at this
        cases this
      have hn0 : ¬ alive t (t.outputs.getD i 0) d := fun ha =>
        hn1 (alive_output_succ hw (outputs_getD_mem hp hi) ha)
      apply hf
      show Hrow (Mr t) (d + 1) (1 + i) = Hrow (Mr t) d (1 + i)
      rw [hHn, Hrow_of_M _ (Mr_code hp hi hn0)]
  new_ins := by
    intro m hm hnew x hx
    have hdm := isNew_dM hp hd hm hnew
    exact H_in hp (by omega) (alive_ins hp.toWellFormed hm hx hdm)
  new_outs := by
    have hw := hp.toWellFormed
    intro m hm hnew y hy
    have hdm := isNew_dM hp hd hm hnew
    have hdy : depth t y = d + 1 := by rw [depth_out hw hm hy, hdm]
    have hal : alive t y (d + 1) := by
      have := alive_depth hw (mem_order.2 (Or.inr ⟨m, hm, hy⟩))
      rwa [hdy] at this
    exact ⟨H_in hp hd hal, H_not_in hp (by omega)⟩

lemma row0OK : Row0OK t (gW t) (H t 0) where
  width := by have := layout_width_pos hp; unfold gW; omega
  code := by
    intro i hi
    exact Mr_code hp hi (output_not_alive_zero hp (outputs_getD_mem hp hi))
  inputs := by
    intro x hx
    exact H_in hp (Nat.zero_le _) ((alive_zero_iff hp.toWellFormed x).2 hx)
  names := by
    intro f n h
    obtain ⟨_, _, hname⟩ := H_some hp (Nat.zero_le _) h
    rcases hname with h1 | h2 | ⟨s, rfl, _⟩
    · exact Or.inl h1
    · exact Or.inr (Or.inl h2)
    · have hal := Mr_alive hp (show Mr t 0 f = some (.sig s) from h)
      exact Or.inr (Or.inr ⟨s, (alive_zero_iff hp.toWellFormed s).1 hal, rfl⟩)

lemma lastOK {i : ℕ} (hi : i < t.outputs.length) :
    H t (maxDepth t) (1 + i) = some (.sig (t.outputs.getD i 0)) := by
  have := Mr_pos hp (output_alive_max hp (outputs_getD_mem hp hi))
  rw [pos_output hp hi] at this
  exact Hrow_of_M _ this

end WF

end GridAux
end Reifier

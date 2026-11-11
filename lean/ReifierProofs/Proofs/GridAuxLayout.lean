import ReifierProofs.Proofs.GridAuxScan

/-!
# The invariants of `layout` (helper for `Proofs/Grid.lean`)
-/

namespace Reifier
namespace GridAux

open Layout

set_option linter.unusedSectionVars false
set_option linter.deprecated false

/-- The invariant of a state of `layout`'s second loop, where `P` is the set of signals it holds. -/
structure Inv (st : State) (P : Sig → Prop) : Prop where
  row0 : st.row 0 = some .one
  beyond : ∀ f, st.width ≤ f → st.row f = none
  present : ∀ x, P x → 1 ≤ st.feat x ∧ st.feat x < st.width ∧ st.row (st.feat x) = some (.sig x)
  sound : ∀ f x, st.row f = some (.sig x) → P x ∧ st.feat x = f
  names : ∀ f n, st.row f = some n → (n = .one ∧ f = 0) ∨ ∃ x, n = .sig x
  free_nodup : st.free.Nodup
  free_iff : ∀ f, f ∈ st.free ↔ 1 ≤ f ∧ f < st.width ∧ st.row f = none

lemma Inv.congr {st : State} {P Q : Sig → Prop} (h : Inv st P) (hPQ : ∀ x, P x ↔ Q x) :
    Inv st Q where
  row0 := h.row0
  beyond := h.beyond
  present := fun x hx => h.present x ((hPQ x).2 hx)
  sound := fun f x hf => ⟨(hPQ x).1 (h.sound f x hf).1, (h.sound f x hf).2⟩
  names := h.names
  free_nodup := h.free_nodup
  free_iff := h.free_iff

lemma Inv.width_pos {st : State} {P : Sig → Prop} (h : Inv st P) : 1 ≤ st.width := by
  by_contra hc
  have := h.beyond 0 (by omega)
  rw [h.row0] at this
  cases this

lemma Inv.lt_width {st : State} {P : Sig → Prop} (h : Inv st P) {f : ℕ} {n : Name}
    (hf : st.row f = some n) : f < st.width := by
  by_contra hc
  rw [h.beyond f (by omega)] at hf
  cases hf

/-! ## `freeOne` -/

lemma freeOne_inv {st : State} {P : Sig → Prop} {x : Sig} (h : Inv st P) (hx : P x) :
    Inv (freeOne st x) (fun z => P z ∧ z ≠ x) := by
  obtain ⟨h1, h2, h3⟩ := h.present x hx
  have hne0 : (0 : ℕ) ≠ st.feat x := by omega
  constructor
  · simp only [freeOne]
    rw [Function.update_of_ne hne0]
    exact h.row0
  · intro f hf
    simp only [freeOne] at hf ⊢
    by_cases hfx : f = st.feat x
    · subst hfx; simp
    · rw [Function.update_of_ne hfx]; exact h.beyond f hf
  · rintro z ⟨hz, hzx⟩
    obtain ⟨g1, g2, g3⟩ := h.present z hz
    have hne : st.feat z ≠ st.feat x := by
      intro he
      rw [he, h3] at g3
      simp only [Option.some.injEq, Name.sig.injEq] at g3
      exact hzx g3.symm
    simp only [freeOne]
    refine ⟨g1, g2, ?_⟩
    rw [Function.update_of_ne hne]; exact g3
  · intro f z hf
    simp only [freeOne] at hf ⊢
    by_cases hfx : f = st.feat x
    · subst hfx; simp at hf
    · rw [Function.update_of_ne hfx] at hf
      obtain ⟨hz, hzf⟩ := h.sound f z hf
      refine ⟨⟨hz, ?_⟩, hzf⟩
      rintro rfl
      exact hfx hzf.symm
  · intro f n hf
    simp only [freeOne] at hf
    by_cases hfx : f = st.feat x
    · subst hfx; simp at hf
    · rw [Function.update_of_ne hfx] at hf
      exact h.names f n hf
  · simp only [freeOne]
    refine List.Nodup.cons ?_ h.free_nodup
    intro hm
    have := ((h.free_iff _).1 hm).2.2
    rw [h3] at this
    cases this
  · intro f
    simp only [freeOne, List.mem_cons]
    by_cases hfx : f = st.feat x
    · subst hfx; simp; omega
    · rw [Function.update_of_ne hfx, h.free_iff f]
      simp [hfx]

lemma freeAll_inv (l : List Sig) : ∀ (st : State) (P : Sig → Prop), l.Nodup → (∀ x ∈ l, P x) →
    Inv st P →
    Inv (l.foldl freeOne st) (fun z => P z ∧ z ∉ l) ∧ (l.foldl freeOne st).feat = st.feat ∧
      (l.foldl freeOne st).width = st.width := by
  induction l with
  | nil =>
    intro st P _ _ h
    exact ⟨h.congr (by simp), rfl, rfl⟩
  | cons x l ih =>
    intro st P hnd hP h
    rw [List.nodup_cons] at hnd
    simp only [List.foldl_cons]
    have h1 := freeOne_inv h (hP x (by simp))
    obtain ⟨i1, i2, i3⟩ := ih (freeOne st x) (fun z => P z ∧ z ≠ x) hnd.2
      (fun z hz => ⟨hP z (by simp [hz]), fun he => hnd.1 (he ▸ hz)⟩) h1
    refine ⟨i1.congr ?_, i2, i3⟩
    intro z
    simp only [List.mem_cons, not_or]
    tauto

/-! ## `place` -/

/-- The first half of `place`: a new feature if none is free. -/
def grow (st : State) : State :=
  if st.free = [] then { st with free := [st.width], width := st.width + 1 } else st

lemma place_eq (st : State) (y : Sig) :
    place st y =
      match (grow st).free with
      | f :: rest =>
          { grow st with free := rest, feat := Function.update (grow st).feat y f,
                         row := Function.update (grow st).row f (some (.sig y)) }
      | [] => grow st := rfl

lemma grow_inv {st : State} {P : Sig → Prop} (h : Inv st P) :
    Inv (grow st) P ∧ (grow st).free ≠ [] ∧ (grow st).feat = st.feat ∧
      st.width ≤ (grow st).width := by
  unfold grow
  split_ifs with hf
  · have hw := h.width_pos
    refine ⟨?_, by simp, rfl, by simp⟩
    constructor
    · exact h.row0
    · intro f hf'; exact h.beyond f (by simp at hf'; omega)
    · intro x hx
      obtain ⟨a, b, c⟩ := h.present x hx
      exact ⟨a, by simp; omega, c⟩
    · exact h.sound
    · exact h.names
    · simp
    · intro f
      simp only [List.mem_singleton]
      constructor
      · rintro rfl; exact ⟨hw, by omega, h.beyond _ le_rfl⟩
      · rintro ⟨a, b, c⟩
        by_contra hne
        have : f ∈ st.free := (h.free_iff f).2 ⟨a, by omega, c⟩
        rw [hf] at this
        simp at this
  · exact ⟨h, hf, rfl, le_rfl⟩

lemma placeHead_inv {st : State} {P : Sig → Prop} {y : Sig} {f : ℕ} {rest : List ℕ}
    (h : Inv st P) (hy : ¬ P y) (hfree : st.free = f :: rest) :
    Inv { st with free := rest, feat := Function.update st.feat y f,
                  row := Function.update st.row f (some (.sig y)) } (fun z => P z ∨ z = y) := by
  have hfm : f ∈ st.free := by rw [hfree]; simp
  obtain ⟨f1, f2, f3⟩ := (h.free_iff f).1 hfm
  have hnd := h.free_nodup
  rw [hfree, List.nodup_cons] at hnd
  constructor
  · simp only
    rw [Function.update_of_ne (by omega)]; exact h.row0
  · intro g hg
    simp only at hg ⊢
    rw [Function.update_of_ne (by omega)]; exact h.beyond g hg
  · rintro z (hz | rfl)
    · obtain ⟨g1, g2, g3⟩ := h.present z hz
      have hzy : z ≠ y := fun he => hy (he ▸ hz)
      have hne : st.feat z ≠ f := by
        intro he; rw [he, f3] at g3; cases g3
      simp only
      rw [Function.update_of_ne hzy, Function.update_of_ne hne]
      exact ⟨g1, g2, g3⟩
    · simp only [Function.update_self]
      exact ⟨f1, f2, trivial⟩
  · intro g z hg
    simp only at hg ⊢
    by_cases hgf : g = f
    · subst hgf
      simp only [Function.update_self, Option.some.injEq, Name.sig.injEq] at hg
      subst hg
      simp
    · rw [Function.update_of_ne hgf] at hg
      obtain ⟨hz, hzg⟩ := h.sound g z hg
      have hzy : z ≠ y := fun he => hy (he ▸ hz)
      rw [Function.update_of_ne hzy]
      exact ⟨Or.inl hz, hzg⟩
  · intro g n hg
    simp only at hg
    by_cases hgf : g = f
    · subst hgf
      simp only [Function.update_self, Option.some.injEq] at hg
      exact Or.inr ⟨y, hg.symm⟩
    · rw [Function.update_of_ne hgf] at hg
      exact h.names g n hg
  · exact hnd.2
  · intro g
    simp only
    by_cases hgf : g = f
    · subst hgf
      simp only [Function.update_self, reduceCtorEq, and_false, iff_false]
      exact hnd.1
    · rw [Function.update_of_ne hgf, ← h.free_iff g, hfree]
      simp [hgf]

lemma place_inv {st : State} {P : Sig → Prop} {y : Sig} (h : Inv st P) (hy : ¬ P y) :
    Inv (place st y) (fun z => P z ∨ z = y) ∧ (∀ z, z ≠ y → (place st y).feat z = st.feat z) ∧
      st.width ≤ (place st y).width := by
  obtain ⟨g1, g2, g3, g4⟩ := grow_inv h
  rw [place_eq]
  obtain ⟨f, rest, hfr⟩ := List.exists_cons_of_ne_nil g2
  rw [hfr]
  simp only
  refine ⟨placeHead_inv g1 hy hfr, ?_, g4⟩
  intro z hz
  rw [Function.update_of_ne hz, g3]

lemma placeAll_inv (l : List Sig) : ∀ (st : State) (P : Sig → Prop), l.Nodup → (∀ y ∈ l, ¬ P y) →
    Inv st P →
    Inv (l.foldl place st) (fun z => P z ∨ z ∈ l) ∧
      (∀ z, z ∉ l → (l.foldl place st).feat z = st.feat z) ∧ st.width ≤ (l.foldl place st).width := by
  induction l with
  | nil =>
    intro st P _ _ h
    exact ⟨h.congr (by simp), fun _ _ => rfl, le_rfl⟩
  | cons y l ih =>
    intro st P hnd hP h
    rw [List.nodup_cons] at hnd
    simp only [List.foldl_cons]
    obtain ⟨p1, p2, p3⟩ := place_inv h (hP y (by simp))
    obtain ⟨i1, i2, i3⟩ := ih (place st y) (fun z => P z ∨ z = y) hnd.2
      (by
        rintro z hz (hz' | rfl)
        · exact hP z (by simp [hz]) hz'
        · exact hnd.1 hz) p1
    refine ⟨i1.congr ?_, ?_, le_trans p3 i3⟩
    · intro z; simp only [List.mem_cons]; tauto
    · intro z hz
      simp only [List.mem_cons, not_or] at hz
      rw [i2 z hz.2, p2 z hz.1]

/-! ## The states, depth by depth -/

/-- The state after depth `d`. -/
def S (t : Trace) : ℕ → State
  | 0 => init t
  | d + 1 => stepDepth t (S t d) (d + 1)

lemma init_inv {t : Trace} (hnd : t.inputs.Nodup) : Inv (init t) (fun z => z ∈ t.inputs) := by
  constructor
  · simp [init]
  · intro f hf
    simp only [init] at hf ⊢
    rw [if_neg (by omega), if_neg (by omega)]
  · intro x hx
    have hlt := List.idxOf_lt_length_iff.2 hx
    simp only [init]
    refine ⟨by omega, by omega, ?_⟩
    rw [if_neg (by omega), if_pos (by omega)]
    simp only [Nat.add_sub_cancel]
    rw [List.getD_eq_getElem _ _ hlt, List.getElem_idxOf hlt]
  · intro f x hf
    simp only [init] at hf ⊢
    split_ifs at hf with h1 h2
    · cases hf
    · simp only [Option.some.injEq, Name.sig.injEq] at hf
      have hlt : f - 1 < t.inputs.length := by omega
      rw [List.getD_eq_getElem _ _ hlt] at hf
      subst hf
      refine ⟨List.getElem_mem _, ?_⟩
      rw [List.Nodup.idxOf_getElem hnd]
      omega
  · intro f n hf
    simp only [init] at hf
    split_ifs at hf with h1 h2
    · left; simp only [Option.some.injEq] at hf; exact ⟨hf.symm, h1⟩
    · right; simp only [Option.some.injEq] at hf; exact ⟨_, hf.symm⟩
  · simp [init]
  · intro f
    simp only [init, List.not_mem_nil, false_iff, not_and]
    intro h1 h2
    rw [if_neg (by omega), if_pos (by omega)]
    simp

section WF

variable {t : Trace} (hw : t.WellFormed)
include hw

lemma inputs_nodup : t.inputs.Nodup := (List.nodup_append.1 hw.nodup).1

lemma step_inv (d : ℕ) (h : Inv (S t d) (fun z => alive t z d)) :
    Inv (S t (d + 1)) (fun z => alive t z (d + 1)) ∧
      (∀ z, z ∉ made t (d + 1) → (S t (d + 1)).feat z = (S t d).feat z) ∧
      (S t d).width ≤ (S t (d + 1)).width := by
  obtain ⟨a1, a2, a3⟩ := freeAll_inv (freed t (d + 1)) (S t d) (fun z => alive t z d)
    (freed_nodup hw _) (fun x hx => freed_alive hw hx) h
  obtain ⟨b1, b2, b3⟩ := placeAll_inv (made t (d + 1)) _ _ (made_nodup hw _)
    (fun y hy hy' => made_not_alive hw hy hy'.1) a1
  refine ⟨?_, ?_, ?_⟩
  · refine b1.congr ?_
    intro z
    rw [alive_step hw]
  · intro z hz
    show ((made t (d + 1)).foldl place ((freed t (d + 1)).foldl freeOne (S t d))).feat z = _
    rw [b2 z hz, a2]
  · show _ ≤ ((made t (d + 1)).foldl place ((freed t (d + 1)).foldl freeOne (S t d))).width
    rw [← a3]; exact b3

lemma S_inv (d : ℕ) : Inv (S t d) (fun z => alive t z d) := by
  induction d with
  | zero =>
    refine (init_inv (inputs_nodup hw)).congr ?_
    intro z
    rw [alive_zero_iff hw]
  | succ d ih => exact (step_inv hw d ih).1

lemma S_width_mono {d d' : ℕ} (hdd : d ≤ d') : (S t d).width ≤ (S t d').width := by
  induction d', hdd using Nat.le_induction with
  | base => exact le_rfl
  | succ d' _ ih => exact le_trans ih (step_inv hw d' (S_inv hw d')).2.2

lemma S_feat_stable {z : Sig} {d d' : ℕ} (hz : depth t z ≤ d) (hdd : d ≤ d') :
    (S t d').feat z = (S t d).feat z := by
  induction d', hdd using Nat.le_induction with
  | base => rfl
  | succ d' hd ih =>
    rw [(step_inv hw d' (S_inv hw d')).2.1 z (by rw [mem_made]; omega), ih]

end WF

/-- Where a signal goes when it is placed. -/
def featAt (t : Trace) (z : Sig) : ℕ := (S t (depth t z)).feat z

lemma feat_eq_featAt {t : Trace} (hw : t.WellFormed) {z : Sig} {d : ℕ} (hz : depth t z ≤ d) :
    (S t d).feat z = featAt t z :=
  S_feat_stable hw le_rfl hz

/-! ## `layout` in terms of `S` -/

lemma foldl_range_S (t : Trace) (i : ℕ) :
    (List.range i).foldl (fun st d => stepDepth t st (d + 1)) (init t) = S t i := by
  induction i with
  | zero => rfl
  | succ i ih => rw [List.range_succ, List.foldl_append, ih]; rfl

lemma states_eq (t : Trace) : states t = (List.range (maxDepth t + 1)).map (S t) := by
  apply List.ext_getElem
  · simp [states]
  · intro i h1 h2
    simp only [states, List.getElem_scanl, List.take_range, List.getElem_map, List.getElem_range]
    rw [foldl_range_S]
    simp only [List.length_map, List.length_range] at h2
    congr 1
    omega

lemma layout_rows (t : Trace) :
    (layout t).rows = (List.range (maxDepth t + 1)).map (fun d => (S t d).row) := by
  simp [layout, states_eq, List.map_map, Function.comp_def]

lemma layout_width (t : Trace) : (layout t).width = (S t (maxDepth t)).width := by
  simp only [layout, states_eq, List.getLast?_map]
  rw [List.getLast?_range]
  simp

end GridAux
end Reifier

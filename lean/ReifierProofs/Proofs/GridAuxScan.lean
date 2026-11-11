import ReifierProofs.Interfaces

/-!
# Facts about `Layout.scan` for a well-formed trace (helper for `Proofs/Grid.lean`)
-/

namespace Reifier
namespace GridAux

open Layout

/-- The step of `scan`. -/
def scanStep (st : (Sig → ℕ) × (Sig → Option ℕ)) (m : MLP) : (Sig → ℕ) × (Sig → Option ℕ) :=
  (fun s => if s ∈ m.outs then 1 + (m.ins.map st.1).foldl max 0 else st.1 s,
   fun s => if s ∈ m.ins then maxLast (st.2 s) (1 + (m.ins.map st.1).foldl max 0) else st.2 s)

/-- The start of `scan`. -/
def scanInit (t : Trace) : (Sig → ℕ) × (Sig → Option ℕ) :=
  (fun _ => 0, fun s => if s ∈ t.outputs then none else some 0)

lemma scan_eq (t : Trace) : scan t = t.mlps.foldl scanStep (scanInit t) := rfl

/-- The state of `scan` after the first `k` MLPs. -/
def stk (t : Trace) (k : ℕ) : (Sig → ℕ) × (Sig → Option ℕ) :=
  (t.mlps.take k).foldl scanStep (scanInit t)

lemma stk_zero (t : Trace) : stk t 0 = scanInit t := by simp [stk]

lemma stk_succ (t : Trace) (k : ℕ) (hk : k < t.mlps.length) :
    stk t (k + 1) = scanStep (stk t k) t.mlps[k] := by
  unfold stk
  rw [List.take_succ_eq_append_getElem hk, List.foldl_append]
  rfl

lemma stk_len (t : Trace) : stk t t.mlps.length = scan t := by
  simp [stk, scan_eq]

lemma depth_eq (t : Trace) : depth t = (stk t t.mlps.length).1 := by
  rw [stk_len]; rfl

lemma last_eq (t : Trace) : last t = (stk t t.mlps.length).2 := by
  rw [stk_len]; rfl

/-! ## `foldl max` -/

lemma foldl_max_ge (l : List ℕ) (a : ℕ) : a ≤ l.foldl max a ∧ ∀ x ∈ l, x ≤ l.foldl max a := by
  induction l generalizing a with
  | nil => simp
  | cons y l ih =>
    simp only [List.foldl_cons, List.mem_cons]
    obtain ⟨h1, h2⟩ := ih (max a y)
    refine ⟨le_trans (le_max_left _ _) h1, ?_⟩
    rintro x (rfl | hx)
    · exact le_trans (le_max_right _ _) h1
    · exact h2 x hx

lemma le_foldl_max {l : List ℕ} {x : ℕ} (a : ℕ) (hx : x ∈ l) : x ≤ l.foldl max a :=
  (foldl_max_ge l a).2 x hx

/-! ## Well-formedness, by index -/

set_option linter.unusedSectionVars false
set_option linter.deprecated false

/-- An MLP's depth: one more than the deepest of its inputs. -/
def dM (t : Trace) (m : MLP) : ℕ := 1 + (m.ins.map (depth t)).foldl max 0

lemma one_le_dM (t : Trace) (m : MLP) : 1 ≤ dM t m := by
  unfold dM; omega

lemma depth_lt_dM {t : Trace} {m : MLP} {x : Sig} (hx : x ∈ m.ins) : depth t x < dM t m := by
  unfold dM
  have := le_foldl_max 0 (List.mem_map_of_mem (f := depth t) hx)
  omega

section WF

variable {t : Trace} (hw : t.WellFormed)
include hw

lemma wf_in_not_out {x : Sig} (hx : x ∈ t.inputs) : ∀ m ∈ t.mlps, x ∉ m.outs := by
  intro m hm hxm
  have h := (List.nodup_append.1 hw.nodup).2.2 x hx x (List.mem_flatMap.2 ⟨m, hm, hxm⟩)
  exact h rfl

lemma wf_out_disj {i j : ℕ} (hi : i < t.mlps.length) (hj : j < t.mlps.length) (hij : i ≠ j)
    {x : Sig} (hx : x ∈ t.mlps[i].outs) : x ∉ t.mlps[j].outs := by
  intro hxj
  have hnd := (List.nodup_append.1 hw.nodup).2.1
  have hp := (List.nodup_flatMap.1 hnd).2
  rw [List.pairwise_iff_getElem] at hp
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · exact hp i j hi hj h hx hxj
  · exact hp j i hj hi h hxj hx

lemma wf_in_not_later_out {i j : ℕ} (hi : i < t.mlps.length) (hj : j < t.mlps.length)
    (hij : i ≤ j) {x : Sig} (hx : x ∈ t.mlps[i].ins) : x ∉ t.mlps[j].outs := by
  rcases hw.topo i hi x hx with h | ⟨j', hj', hx'⟩
  · exact wf_in_not_out hw h _ (List.getElem_mem _)
  · exact wf_out_disj hw (by omega) hj (by omega) hx'

/-- The depth of a signal no later MLP makes is what the scan gives it so far. -/
lemma stk_fst_stable (x : Sig) (k : ℕ) :
    ∀ k', k ≤ k' → k' ≤ t.mlps.length →
      (∀ j (hj : j < t.mlps.length), k ≤ j → j < k' → x ∉ t.mlps[j].outs) →
      (stk t k').1 x = (stk t k).1 x := by
  intro k' hkk'
  induction k', hkk' using Nat.le_induction with
  | base => intros; rfl
  | succ k' hk ih =>
    intro hlen hno
    rw [stk_succ t k' (by omega)]
    simp only [scanStep]
    rw [if_neg (hno k' (by omega) hk (by omega))]
    exact ih (by omega) (fun j hj h1 h2 => hno j hj h1 (by omega))

lemma depth_eq_stk (x : Sig) (k : ℕ) (hk : k ≤ t.mlps.length)
    (hno : ∀ j (hj : j < t.mlps.length), k ≤ j → x ∉ t.mlps[j].outs) :
    depth t x = (stk t k).1 x := by
  rw [depth_eq]
  exact stk_fst_stable hw x k _ hk le_rfl (fun j hj h1 _ => hno j hj h1)

lemma depth_input {x : Sig} (hx : x ∈ t.inputs) : depth t x = 0 := by
  rw [depth_eq_stk hw x 0 (Nat.zero_le _)
    (fun j hj _ => wf_in_not_out hw hx _ (List.getElem_mem _))]
  simp [stk_zero, scanInit]

lemma depth_ins_stk {i : ℕ} (hi : i < t.mlps.length) {x : Sig} (hx : x ∈ t.mlps[i].ins) :
    (stk t i).1 x = depth t x :=
  (depth_eq_stk hw x i hi.le (fun _ hj hij => wf_in_not_later_out hw hi hj hij hx)).symm

lemma dM_eq_stk {i : ℕ} (hi : i < t.mlps.length) :
    1 + (t.mlps[i].ins.map (stk t i).1).foldl max 0 = dM t t.mlps[i] := by
  unfold dM
  congr 2
  exact List.map_congr_left (fun x hx => depth_ins_stk hw hi hx)

lemma depth_out_idx {i : ℕ} (hi : i < t.mlps.length) {y : Sig} (hy : y ∈ t.mlps[i].outs) :
    depth t y = dM t t.mlps[i] := by
  rw [depth_eq_stk hw y (i + 1) hi
    (fun j hj hij => wf_out_disj hw hi hj (by omega) hy), stk_succ t i hi]
  simp only [scanStep, if_pos hy]
  exact dM_eq_stk hw hi

lemma depth_out {m : MLP} (hm : m ∈ t.mlps) {y : Sig} (hy : y ∈ m.outs) :
    depth t y = dM t m := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hm
  exact depth_out_idx hw hi hy

end WF

/-! ## `last` and `freedAt` -/

/-- `o ≥ v`, where `none` is infinity. -/
def geOpt (o : Option ℕ) (v : ℕ) : Prop := ∀ a, o = some a → v ≤ a

lemma geOpt_none (v : ℕ) : geOpt none v := by intro a h; cases h

lemma geOpt_maxLast {o : Option ℕ} {v : ℕ} (h : geOpt o v) (d : ℕ) : geOpt (maxLast o d) v := by
  intro a ha
  cases o with
  | none => simp [maxLast] at ha
  | some b =>
    simp only [maxLast, Option.some.injEq] at ha
    have := h b rfl
    omega

lemma geOpt_maxLast_right (o : Option ℕ) (d : ℕ) : geOpt (maxLast o d) d := by
  intro a ha
  cases o with
  | none => simp [maxLast] at ha
  | some b =>
    simp only [maxLast, Option.some.injEq] at ha
    omega

lemma geOpt_mono {o : Option ℕ} {v w : ℕ} (h : geOpt o v) (hwv : w ≤ v) : geOpt o w :=
  fun a ha => le_trans hwv (h a ha)

lemma stk_snd_mono (t : Trace) (x : Sig) (v k : ℕ) (hv : geOpt ((stk t k).2 x) v) :
    ∀ k', k ≤ k' → k' ≤ t.mlps.length → geOpt ((stk t k').2 x) v := by
  intro k' hkk'
  induction k', hkk' using Nat.le_induction with
  | base => intros; exact hv
  | succ k' hk ih =>
    intro hlen
    rw [stk_succ t k' (by omega)]
    simp only [scanStep]
    split_ifs
    · exact geOpt_maxLast (ih (by omega)) _
    · exact ih (by omega)

section WF2

variable {t : Trace} (hw : t.WellFormed)
include hw

lemma last_ge_dM_idx {i : ℕ} (hi : i < t.mlps.length) {x : Sig} (hx : x ∈ t.mlps[i].ins) :
    geOpt (last t x) (dM t t.mlps[i]) := by
  rw [last_eq]
  apply stk_snd_mono t x _ (i + 1) _ _ hi le_rfl
  rw [stk_succ t i hi]
  simp only [scanStep, if_pos hx]
  rw [dM_eq_stk hw hi]
  exact geOpt_maxLast_right _ _

lemma last_ge_dM {m : MLP} (hm : m ∈ t.mlps) {x : Sig} (hx : x ∈ m.ins) :
    geOpt (last t x) (dM t m) := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hm
  exact last_ge_dM_idx hw hi hx

lemma stk_snd_depth (x : Sig) : ∀ k, k ≤ t.mlps.length →
    (stk t k).2 x ≠ some 0 → geOpt ((stk t k).2 x) (depth t x + 1) := by
  intro k
  induction k with
  | zero =>
    intro _ h
    simp only [stk_zero, scanInit] at h ⊢
    split_ifs at h ⊢
    · exact geOpt_none _
    · exact absurd rfl h
  | succ k ih =>
    intro hk
    rw [stk_succ t k (by omega)]
    simp only [scanStep]
    split_ifs with hx
    · intro _
      have hd : depth t x + 1 ≤ 1 + (t.mlps[k].ins.map (stk t k).1).foldl max 0 := by
        rw [dM_eq_stk hw (by omega)]
        have := depth_lt_dM (t := t) hx
        omega
      intro a ha
      cases h : (stk t k).2 x with
      | none => rw [h] at ha; simp [maxLast] at ha
      | some b =>
        rw [h] at ha
        simp only [maxLast, Option.some.injEq] at ha
        omega
    · exact ih (by omega)

lemma freedAt_ge_depth (x : Sig) : geOpt (freedAt t x) (depth t x + 1) := by
  unfold freedAt
  split
  · intro a ha; simp only [Option.some.injEq] at ha; omega
  · rename_i l hl
    have := stk_snd_depth hw x t.mlps.length le_rfl
    rw [← last_eq] at this
    exact this hl

lemma freedAt_ge_dM {m : MLP} (hm : m ∈ t.mlps) {x : Sig} (hx : x ∈ m.ins) :
    geOpt (freedAt t x) (dM t m) := by
  have h := last_ge_dM hw hm hx
  unfold freedAt
  split
  · rename_i h0
    have := h 0 h0
    have := one_le_dM t m
    omega
  · exact h

end WF2

lemma stk_snd_output (t : Trace) {y : Sig} (hy : y ∈ t.outputs) :
    ∀ k, k ≤ t.mlps.length → (stk t k).2 y = none := by
  intro k
  induction k with
  | zero => intro _; simp [stk_zero, scanInit, hy]
  | succ k ih =>
    intro hk
    rw [stk_succ t k (by omega)]
    simp only [scanStep]
    rw [ih (by omega)]
    split_ifs <;> rfl

lemma freedAt_output (t : Trace) {y : Sig} (hy : y ∈ t.outputs) : freedAt t y = none := by
  have h : last t y = none := by
    rw [last_eq]; exact stk_snd_output t hy _ le_rfl
  unfold freedAt
  rw [h]

/-! ## `order`, `made`, `freed`, `maxDepth` -/

lemma mem_order {t : Trace} {x : Sig} :
    x ∈ order t ↔ x ∈ t.inputs ∨ ∃ m ∈ t.mlps, x ∈ m.outs := by
  simp [order, List.mem_append, List.mem_flatMap]

lemma order_nodup {t : Trace} (hw : t.WellFormed) : (order t).Nodup := hw.nodup

lemma mem_made {t : Trace} {d : ℕ} {x : Sig} : x ∈ made t d ↔ x ∈ order t ∧ depth t x = d := by
  simp [made]

lemma mem_freed {t : Trace} {d : ℕ} {x : Sig} :
    x ∈ freed t d ↔ x ∈ order t ∧ freedAt t x = some d := by
  simp [freed]

lemma made_nodup {t : Trace} (hw : t.WellFormed) (d : ℕ) : (made t d).Nodup :=
  (order_nodup hw).filter _

lemma freed_nodup {t : Trace} (hw : t.WellFormed) (d : ℕ) : (freed t d).Nodup :=
  (order_nodup hw).filter _

lemma depth_le_maxDepth {t : Trace} {x : Sig} (hx : x ∈ order t) : depth t x ≤ maxDepth t :=
  le_foldl_max 0 (List.mem_map_of_mem hx)

lemma order_of_ins {t : Trace} (hw : t.WellFormed) {m : MLP} (hm : m ∈ t.mlps) {x : Sig}
    (hx : x ∈ m.ins) : x ∈ order t := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hm
  rcases hw.topo i hi x hx with h | ⟨j, hj, hxj⟩
  · exact mem_order.2 (Or.inl h)
  · exact mem_order.2 (Or.inr ⟨_, List.getElem_mem _, hxj⟩)

lemma input_of_depth_zero {t : Trace} (hw : t.WellFormed) {x : Sig} (hx : x ∈ order t)
    (h0 : depth t x = 0) : x ∈ t.inputs := by
  rcases mem_order.1 hx with h | ⟨m, hm, hxm⟩
  · exact h
  · have := depth_out hw hm hxm
    have := one_le_dM t m
    omega

/-! ## Alive signals -/

/-- A signal is alive at depth `d`: made by then, and not yet freed. -/
def alive (t : Trace) (x : Sig) (d : ℕ) : Prop :=
  x ∈ order t ∧ depth t x ≤ d ∧ geOpt (freedAt t x) (d + 1)

section WF3

variable {t : Trace} (hw : t.WellFormed)
include hw

lemma alive_depth {x : Sig} (hx : x ∈ order t) : alive t x (depth t x) :=
  ⟨hx, le_rfl, freedAt_ge_depth hw x⟩

lemma alive_succ_not {x : Sig} {d : ℕ} (h1 : alive t x (d + 1)) (h2 : ¬ alive t x d) :
    depth t x = d + 1 := by
  obtain ⟨hx, hd, hf⟩ := h1
  by_contra hne
  exact h2 ⟨hx, by omega, geOpt_mono hf (by omega)⟩

lemma alive_output_succ {y : Sig} (hy : y ∈ t.outputs) {d : ℕ} (h : alive t y d) :
    alive t y (d + 1) := by
  obtain ⟨hx, hd, _⟩ := h
  refine ⟨hx, by omega, ?_⟩
  rw [freedAt_output t hy]
  exact geOpt_none _

lemma alive_output_mono {y : Sig} (hy : y ∈ t.outputs) {d d' : ℕ} (hdd : d ≤ d')
    (h : alive t y d) : alive t y d' := by
  induction d', hdd using Nat.le_induction with
  | base => exact h
  | succ d' _ ih => exact alive_output_succ hw hy ih

lemma alive_step (z : Sig) (d : ℕ) :
    alive t z (d + 1) ↔ (alive t z d ∧ z ∉ freed t (d + 1)) ∨ z ∈ made t (d + 1) := by
  rw [mem_freed, mem_made]
  constructor
  · rintro ⟨hz, hd, hf⟩
    by_cases h : depth t z = d + 1
    · exact Or.inr ⟨hz, h⟩
    · left
      refine ⟨⟨hz, by omega, geOpt_mono hf (by omega)⟩, ?_⟩
      rintro ⟨_, hfd⟩
      have := hf _ hfd
      omega
  · rintro (⟨⟨hz, hd, hf⟩, hnf⟩ | ⟨hz, hd⟩)
    · refine ⟨hz, by omega, ?_⟩
      intro a ha
      have := hf a ha
      have : a ≠ d + 1 := by
        rintro rfl
        exact hnf ⟨hz, ha⟩
      omega
    · refine ⟨hz, by omega, ?_⟩
      have := freedAt_ge_depth hw z
      rw [hd] at this
      exact this

lemma freed_alive {z : Sig} {d : ℕ} (hz : z ∈ freed t (d + 1)) : alive t z d := by
  rw [mem_freed] at hz
  obtain ⟨hz, hf⟩ := hz
  have h1 := freedAt_ge_depth hw z _ hf
  refine ⟨hz, by omega, ?_⟩
  intro a ha
  rw [hf] at ha
  cases ha
  exact le_rfl

lemma made_not_alive {z : Sig} {d : ℕ} (hz : z ∈ made t (d + 1)) : ¬ alive t z d := by
  rw [mem_made] at hz
  rintro ⟨_, hd, _⟩
  omega

lemma alive_zero_iff (z : Sig) : alive t z 0 ↔ z ∈ t.inputs := by
  constructor
  · rintro ⟨hz, hd, _⟩
    exact input_of_depth_zero hw hz (by omega)
  · intro hz
    have h0 := depth_input hw hz
    have := alive_depth hw (mem_order.2 (Or.inl hz))
    rwa [h0] at this

lemma alive_ins {m : MLP} (hm : m ∈ t.mlps) {x : Sig} (hx : x ∈ m.ins) {d : ℕ}
    (hd : dM t m = d + 1) : alive t x d := by
  refine ⟨order_of_ins hw hm hx, ?_, ?_⟩
  · have := depth_lt_dM (t := t) hx
    omega
  · exact geOpt_mono (freedAt_ge_dM hw hm hx) (by omega)

end WF3

end GridAux
end Reifier

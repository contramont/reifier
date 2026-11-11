import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace TraceProof

/-- Folding `apply` over MLPs none of which writes `s` leaves `s` alone. -/
theorem foldl_apply_not_out (ms : List MLP) (v : Sig → ℝ) (s : Sig)
    (hs : s ∉ ms.flatMap MLP.outs) :
    (ms.foldl (fun v m => m.apply v) v) s = v s := by
  induction ms generalizing v with
  | nil => rfl
  | cons m ms ih =>
    simp only [List.flatMap_cons, List.mem_append, not_or] at hs
    rw [List.foldl_cons, ih _ hs.2]
    simp [MLP.apply, hs.1]

/-- `pre` depends only on the values of the inputs that exist. -/
theorem pre_congr (m : MLP) (u u' : ℕ → ℝ) (h : ∀ k < m.ins.length, u k = u' k) (j : ℕ) :
    m.pre u j = m.pre u' j := by
  unfold MLP.pre
  congr 1
  refine Finset.sum_congr rfl ?_
  intro k hk
  rw [h k (Finset.mem_range.mp hk)]

theorem unit_congr (m : MLP) (u u' : ℕ → ℝ) (h : ∀ k < m.ins.length, u k = u' k) (j : ℕ) :
    m.unit u j = m.unit u' j := by
  unfold MLP.unit
  rw [pre_congr m u u' h j, pre_congr m u u' h (m.hidden + j)]

theorem out_congr (m : MLP) (u u' : ℕ → ℝ) (h : ∀ k < m.ins.length, u k = u' k) (i : ℕ) :
    m.out u i = m.out u' i := by
  unfold MLP.out
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [unit_congr m u u' h j]

end TraceProof

open TraceProof in
theorem stmt_trace : Stmt_trace := by
  intro t x hwf
  have hnd := hwf.nodup
  refine ⟨?_, ?_⟩
  · intro s hs
    apply foldl_apply_not_out
    intro hs'
    exact (List.nodup_append.mp hnd).2.2 s hs s hs' rfl
  · intro m hm i hi
    obtain ⟨n, hn, hmn⟩ := List.mem_iff_getElem.mp hm
    set L := t.mlps with hL
    set A := L.take n with hA
    set B := L.drop (n + 1) with hB
    have hsplit : L = A ++ m :: B := by
      rw [hA, hB, ← hmn, ← List.drop_eq_getElem_cons hn, List.take_append_drop]
    have hfm : L.flatMap MLP.outs = A.flatMap MLP.outs ++ (m.outs ++ B.flatMap MLP.outs) := by
      rw [hsplit]; simp [List.flatMap_append]
    rw [hfm] at hnd
    -- disjointness facts
    have hnd2 := List.nodup_append.mp hnd
    have hndA := List.nodup_append.mp hnd2.2.1
    have hndm := List.nodup_append.mp hndA.2.1
    -- the valuations
    set vA := A.foldl (fun v m => m.apply v) x with hvA
    have hval : t.values x = B.foldl (fun v m => m.apply v) (m.apply vA) := by
      unfold Trace.values
      rw [← hL, hsplit, List.foldl_append, List.foldl_cons]
    -- inputs of m have the same value in vA and in the final valuation
    have hins : ∀ k < m.ins.length, m.read vA k = m.read (t.values x) k := by
      intro k hk
      simp only [MLP.read]
      rw [List.getD_eq_getElem _ _ hk]
      set y := m.ins[k] with hy
      have hymem : y ∈ m.ins := List.getElem_mem hk
      have htopo := hwf.topo n hn y (by simpa [hmn] using hymem)
      -- y is an input or an output of an MLP in A
      have hy' : y ∈ t.inputs ∨ y ∈ A.flatMap MLP.outs := by
        rcases htopo with h | ⟨j, hj, hyj⟩
        · exact Or.inl h
        · right
          rw [List.mem_flatMap]
          refine ⟨L[j]'(by omega), ?_, hyj⟩
          have hjA : j < A.length := by simp [hA]; omega
          have : A[j] = L[j]'(by omega) := by simp [hA]
          rw [← this]
          exact List.getElem_mem hjA
      have hnotm : y ∉ m.outs := by
        intro hym
        rcases hy' with h | h
        · exact hnd2.2.2 y h y (List.mem_append_right _ (List.mem_append_left _ hym)) rfl
        · exact hndA.2.2 y h y (List.mem_append_left _ hym) rfl
      have hnotB : y ∉ B.flatMap MLP.outs := by
        intro hyB
        rcases hy' with h | h
        · exact hnd2.2.2 y h y (List.mem_append_right _ (List.mem_append_right _ hyB)) rfl
        · exact hndA.2.2 y h y (List.mem_append_right _ hyB) rfl
      rw [hval, foldl_apply_not_out B _ y hnotB]
      simp [MLP.apply, hnotm]
    -- the output
    have hyi : m.outs.getD i 0 = m.outs[i] := List.getD_eq_getElem _ _ hi
    have hmemi : m.outs[i] ∈ m.outs := List.getElem_mem hi
    have hnotB : m.outs[i] ∉ B.flatMap MLP.outs := by
      intro h
      exact hndm.2.2 _ hmemi _ h rfl
    rw [hyi, hval, foldl_apply_not_out B _ _ hnotB, ← hval]
    simp only [MLP.apply, hmemi, ite_true]
    rw [List.Nodup.idxOf_getElem hndm.1 i hi]
    exact out_congr m _ _ hins i

end

end Reifier

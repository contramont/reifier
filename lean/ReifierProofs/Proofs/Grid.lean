import ReifierProofs.Main
import ReifierProofs.Proofs.GridAuxFacts

namespace Reifier

open Finset

noncomputable section

open GridAux in
theorem stmt_grid : Stmt_grid := by
  intro t hp
  have hrows : (tokenGrid t).rows = (List.range (Layout.maxDepth t + 1)).map (H t) :=
    tokenGrid_rows hp
  have hwidth : (tokenGrid t).width = gW t := rfl
  generalize tokenGrid t = g at hrows hwidth
  obtain ⟨W, rows⟩ := g
  simp only at hrows hwidth
  subst hrows hwidth
  refine
    { rows_ne := by simp
      rows_ok := ?_
      pairs := ?_
      row0 := ?_
      last := ?_ }
  · intro r hr
    obtain ⟨d, hd, rfl⟩ := List.mem_map.1 hr
    exact rowOK hp (by rw [List.mem_range] at hd; omega)
  · intro d hd
    simp only [List.length_map, List.length_range] at hd
    simp only [List.getElem_map, List.getElem_range]
    exact pairOK hp (by omega)
  · simp only [List.head_map, List.head_range]
    exact row0OK hp
  · intro i hi
    simp only [List.getLast_map, List.getLast_range, Nat.add_sub_cancel]
    exact lastOK hp hi

end

end Reifier

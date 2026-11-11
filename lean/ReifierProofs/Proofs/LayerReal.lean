import ReifierProofs.Proofs.LayerGain

/-!
# One layer in real arithmetic (`Stmt_layer_real`)

The case `G = 10^8`, `σ = 0`, `E = 10^-9` of `Stmt_layer_gain`: a flat trace has every
`slopeSum` 0, and `RealOK` gives the rest of `RealCore (10^8) 0`.
-/

namespace Reifier
open Finset
noncomputable section

namespace LayerRealPf

lemma realCore_of_realOK (t : Trace) (h : t.RealOK) : t.RealCore (10 ^ 8) 0 where
  gate_margin := h.gate_margin
  slope := by
    intro x m hm hs i hi
    unfold MLP.slopeSum
    rw [sum_eq_zero (fun k hk => by rw [h.flat x m hm hs i hi k (mem_range.mp hk), abs_zero])]
  bounded := h.bounded
  glu_gain := by
    intro m hm hs i hi
    exact_mod_cast h.glu_gain m hm hs i hi

end LayerRealPf

open LayerRealPf in
theorem layer_real_of (hd : Stmt_decomp) (hu : Stmt_unit_dots) (hc : Stmt_compiled)
    (ht : Stmt_trace) (hn : Stmt_norm_close) (hg : Stmt_gelu) : Stmt_layer_real := by
  intro t W before row val x hWF hRO hP hV hW hCl
  have h := layer_gain_of hd hu hc ht hn hg t W before row val x (1 / 10 ^ 9) (10 ^ 8) 0 hWF
    (realCore_of_realOK t hRO) hP hV hW (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    le_rfl (by norm_num) hCl
  rwa [add_zero, one_mul] at h

end
end Reifier

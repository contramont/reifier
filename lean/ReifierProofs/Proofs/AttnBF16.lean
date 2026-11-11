import ReifierProofs.Main
import ReifierProofs.Proofs.AttnBF16AuxModel

namespace Reifier

open Finset

namespace AttnBF16Fold
open Finset AttnBF16Err AttnBF16Num AttnBF16Model

noncomputable section

variable (t : Trace) (codes : List (ℕ × List Bool))

/-- the feature where head `h` writes bit `j` -/
abbrev F (h j : ℕ) : ℕ :=
  featIn (tokenGrid t).width (row0 t)
    (.sig (t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0))

omit t codes in
lemma divmod (a j b : ℕ) (hj : j < b) : (a * b + j) / b = a ∧ (a * b + j) % b = j := by
  have hb : 0 < b := by omega
  constructor
  · rw [add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hj, zero_add]
  · rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]

lemma len_eq (hprog : t.IsProgram) : t.inputs.length = t.window * t.outputs.length := by
  unfold Trace.window
  exact (Nat.div_mul_cancel hprog.window_dvd).symm

lemma inputs_nodup (hprog : t.IsProgram) : t.inputs.Nodup :=
  (List.nodup_append.mp hprog.nodup).1

lemma k_lt (hprog : t.IsProgram) (h j : ℕ) (hh : h < t.window) (hj : j < t.outputs.length) :
    (t.window - 1 - h) * t.outputs.length + j < t.inputs.length := by
  rw [len_eq t hprog]
  have : (t.window - 1 - h) * t.outputs.length + j < (t.window - 1 - h + 1) * t.outputs.length := by
    rw [add_mul, one_mul]; omega
  calc _ < (t.window - 1 - h + 1) * t.outputs.length := this
    _ ≤ t.window * t.outputs.length := Nat.mul_le_mul_right _ (by omega)

lemma F_spec (hprog : t.IsProgram) (hG : GridOK t (tokenGrid t)) (h j : ℕ) (hh : h < t.window)
    (hj : j < t.outputs.length) :
    F t h j < (tokenGrid t).width ∧
      row0 t (F t h j) = some (.sig (t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0)) := by
  apply featIn_spec
  apply (row0_ok t hG).inputs
  rw [List.getD_eq_getElem _ _ (k_lt t hprog h j hh hj)]
  exact List.getElem_mem _

lemma F_gt (hprog : t.IsProgram) (hG : GridOK t (tokenGrid t)) (h j : ℕ) (hh : h < t.window)
    (hj : j < t.outputs.length) : t.outputs.length < F t h j := by
  obtain ⟨_, h2⟩ := F_spec t hprog hG h j hh hj
  by_contra hcon
  replace hcon := not_lt.mp hcon
  by_cases h0 : F t h j = 0
  · rw [h0, row0_zero t hG] at h2; simp at h2
  · rw [row0_code t hG _ (by omega) hcon] at h2; simp at h2

lemma F_inj (hprog : t.IsProgram) (hG : GridOK t (tokenGrid t)) (h1 j1 h2 j2 : ℕ)
    (hh1 : h1 < t.window) (hj1 : j1 < t.outputs.length) (hh2 : h2 < t.window)
    (hj2 : j2 < t.outputs.length) (heq : F t h1 j1 = F t h2 j2) : h1 = h2 ∧ j1 = j2 := by
  obtain ⟨_, e1⟩ := F_spec t hprog hG h1 j1 hh1 hj1
  obtain ⟨_, e2⟩ := F_spec t hprog hG h2 j2 hh2 hj2
  rw [heq, e2] at e1
  simp only [Option.some.injEq, Name.sig.injEq] at e1
  have k1 := k_lt t hprog h1 j1 hh1 hj1
  have k2 := k_lt t hprog h2 j2 hh2 hj2
  rw [List.getD_eq_getElem _ _ k1, List.getD_eq_getElem _ _ k2] at e1
  have hk := ((inputs_nodup t hprog).getElem_inj_iff).mp e1.symm
  have d1 := divmod (t.window - 1 - h1) j1 _ hj1
  have d2 := divmod (t.window - 1 - h2) j2 _ hj2
  rw [hk] at d1
  omega

/-! ## What each head writes -/

variable (R : Rounding) (toks : ℕ → ℕ) (p : ℕ)

/-- bit `j` of the token `h` back, as -1 or 1 -/
def Hv (h j : ℕ) : ℝ := if ((codes.lookup (toks (p - h))).getD []).getD j false then 1 else -1

lemma Hv_pm (h j : ℕ) : Hv codes toks p h j = 1 ∨ Hv codes toks p h j = -1 := by
  unfold Hv; split_ifs <;> simp

lemma Hv_bf16 (hr : Stmt_round) (h j : ℕ) : IsBF16 (Hv codes toks p h j) := by
  rcases Hv_pm codes toks p h j with e | e <;> rw [e]
  · exact bf16_one hr
  · exact bf16_neg_one hr

lemma head_Hv (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (hpos : PosOK t codes toks p)
    (h : ℕ) (hh : h < t.window) (j : ℕ) (hj : j < t.outputs.length) :
    BF16.head R (transformer t codes) toks h p j = Hv codes toks p h j := by
  have hwl := hprog.window_le
  have hp1 := hpos.1
  unfold CONTEXT at hwl
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp (hpos.2 0 (by omega))
  obtain ⟨ct, hct⟩ := Option.isSome_iff_exists.mp (hpos.2 h (by omega))
  rw [Nat.sub_zero] at hc
  rw [head_eq t codes R hr hn hprog hG hw toks p hc h hh (by omega) (by omega) hct j hj]
  unfold Hv; rw [hct]; rfl

/-- what head `h` adds to feature `f` -/
abbrev C (h f : ℕ) : ℝ :=
  R.fl (∑ j ∈ range (transformer t codes).bits,
    R.fl ((transformer t codes).o h f j) * BF16.head R (transformer t codes) toks h p j)

lemma C_write (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (hpos : PosOK t codes toks p)
    (h : ℕ) (hh : h < t.window) (j0 : ℕ) (hj0 : j0 < t.outputs.length) :
    C t codes R toks p h (F t h j0) = Hv codes toks p h j0 := by
  unfold C
  rw [M_bits, sum_single _ _ j0 hj0]
  · rw [M_o]
    simp only [hh, hj0, true_and, ite_true]
    rw [fl_one hr R, one_mul, head_Hv t codes R toks p hr hn hprog hG hw hpos h hh j0 hj0]
    exact fl_exact hr R (Hv_bf16 codes toks p hr h j0)
  · intro j hj
    rw [M_o]
    split_ifs with hc
    · exfalso
      obtain ⟨_, hjb, he⟩ := hc
      exact hj (F_inj t hprog hG h j h j0 hh hjb hh hj0 he.symm).2
    · rw [fl_zero hr R, zero_mul]

lemma C_zero (hr : Stmt_round) (h f : ℕ) (hf : ∀ j < t.outputs.length, f ≠ F t h j) :
    C t codes R toks p h f = 0 := by
  unfold C
  rw [Finset.sum_eq_zero, fl_zero hr R]
  intro j hj
  rw [M_o]
  split_ifs with hc
  · exact absurd hc.2.2 (hf j hc.2.1)
  · rw [fl_zero hr R, zero_mul]

/-! ## The heads, added in turn -/

/-- the stream after the first `n` heads -/
def X (n : ℕ) : ℕ → ℝ :=
  (List.range n).foldl (fun x h f => R.fl (x f + C t codes R toks p h f))
    (BF16.embed R (transformer t codes) (toks p))

lemma afterAttn_X : BF16.afterAttn R (transformer t codes) toks p = X t codes R toks p t.window := rfl

lemma X_zero : X t codes R toks p 0 = BF16.embed R (transformer t codes) (toks p) := rfl

lemma X_succ (n f : ℕ) : X t codes R toks p (n + 1) f =
    R.fl (X t codes R toks p n f + C t codes R toks p n f) := by
  unfold X
  rw [List.range_succ, List.foldl_append]
  rfl

lemma embed_p (hr : Stmt_round) (hpos : PosOK t codes toks p) :
    ∃ c, codes.lookup (toks p) = some c ∧ ∀ f, BF16.embed R (transformer t codes) (toks p) f =
      if f = 0 then BF16.one R (transformer t codes).net
      else if f ≤ t.outputs.length then (if c.getD (f - 1) false then 1 else -1) else 0 := by
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp (hpos.2 0 (by omega))
  rw [Nat.sub_zero] at hc
  exact ⟨c, hc, embed_coded t codes R hr hc⟩

lemma embed_bf16 (hr : Stmt_round) (hw : WidthOK t) (hpos : PosOK t codes toks p) (f : ℕ) :
    IsBF16 (BF16.embed R (transformer t codes) (toks p) f) := by
  obtain ⟨c, _, he⟩ := embed_p t codes R toks p hr hpos
  rw [he]
  split_ifs
  · exact oneB_bf16 t codes R hw
  · exact bf16_one hr
  · exact bf16_neg_one hr
  · exact bf16_zero hr

lemma embed_F (hr : Stmt_round) (hprog : t.IsProgram) (hG : GridOK t (tokenGrid t))
    (hpos : PosOK t codes toks p) (h j : ℕ) (hh : h < t.window) (hj : j < t.outputs.length) :
    BF16.embed R (transformer t codes) (toks p) (F t h j) = 0 := by
  obtain ⟨c, _, he⟩ := embed_p t codes R toks p hr hpos
  have := F_gt t hprog hG h j hh hj
  rw [he, ite_eq_right_iff.mpr (by omega), ite_eq_right_iff.mpr (by omega)]

lemma X_inv (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (hpos : PosOK t codes toks p) :
    ∀ n ≤ t.window,
      (∀ h < n, ∀ j < t.outputs.length, X t codes R toks p n (F t h j) = Hv codes toks p h j) ∧
      (∀ f, (∀ h < n, ∀ j < t.outputs.length, f ≠ F t h j) →
        X t codes R toks p n f = BF16.embed R (transformer t codes) (toks p) f) := by
  intro n
  induction n with
  | zero =>
    intro _
    exact ⟨fun h hh => absurd hh (Nat.not_lt_zero _), fun f _ => rfl⟩
  | succ n ih =>
    intro hn1
    obtain ⟨I1, I2⟩ := ih (by omega)
    simp only [X_succ]
    refine ⟨?_, ?_⟩
    · intro h hh j hj
      rcases Nat.lt_succ_iff_lt_or_eq.mp hh with hlt | heq
      · rw [I1 h hlt j hj, C_zero t codes R toks p hr n _ ?_, add_zero]
        · exact fl_exact hr R (Hv_bf16 codes toks p hr h j)
        · intro j' hj' he
          have := (F_inj t hprog hG h j n j' (by omega) hj (by omega) hj' he).1
          omega
      · subst heq
        rw [C_write t codes R toks p hr hn hprog hG hw hpos h (by omega) j hj]
        rw [I2 (F t h j) ?_, embed_F t codes R toks p hr hprog hG hpos h j (by omega) hj, zero_add]
        · exact fl_exact hr R (Hv_bf16 codes toks p hr h j)
        · intro h' hh' j' hj' he
          have := (F_inj t hprog hG h j h' j' (by omega) hj (by omega) hj' he).1
          omega
    · intro f hf
      rw [C_zero t codes R toks p hr n f (fun j hj => hf n (by omega) j hj), add_zero,
        I2 f (fun h hh j hj => hf h (by omega) j hj)]
      exact fl_exact hr R (embed_bf16 t codes R toks p hr hw hpos f)

/-! ## Row 0's stream -/

lemma ideal_F (ht : Stmt_trace) (hprog : t.IsProgram) (hG : GridOK t (tokenGrid t))
    (hpos : PosOK t codes toks p) (h j : ℕ) (hh : h < t.window) (hj : j < t.outputs.length) :
    ideal (BF16.one R (transformer t codes).net) (valAt t codes toks p) (row0 t) (F t h j) =
      Hv codes toks p h j := by
  obtain ⟨_, hrow⟩ := F_spec t hprog hG h j hh hj
  have hk := k_lt t hprog h j hh hj
  set k := (t.window - 1 - h) * t.outputs.length + j with hkdef
  have hs : t.inputs.getD k 0 = t.inputs[k] := List.getD_eq_getElem _ _ hk
  unfold ideal
  rw [hrow]
  simp only [valAt]
  rw [(ht t _ hprog.toWellFormed).1 _ (by rw [hs]; exact List.getElem_mem _)]
  unfold Trace.env
  rw [hs, (inputs_nodup t hprog).idxOf_getElem k hk]
  have d := divmod (t.window - 1 - h) j _ hj
  rw [← hkdef] at d
  have hp1 := hpos.1
  have hidx : p + 1 + k / t.outputs.length - t.window = p - h := by omega
  unfold windowBits Hv
  simp only [hidx, d.2]
  split_ifs <;> norm_num

lemma ideal_rest (hr : Stmt_round) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hpos : PosOK t codes toks p) (f : ℕ)
    (hf : ∀ h < t.window, ∀ j < t.outputs.length, f ≠ F t h j) :
    BF16.embed R (transformer t codes) (toks p) f =
      ideal (BF16.one R (transformer t codes).net) (valAt t codes toks p) (row0 t) f := by
  obtain ⟨c, hc, he⟩ := embed_p t codes R toks p hr hpos
  rw [he]
  have hrok := row0_rowok t hG
  unfold ideal
  split_ifs with h1 h2 h3
  · subst h1; rw [row0_zero t hG]
  · rw [row0_code t hG f (by omega) h2]
    simp only [valAt, hc, Option.getD_some, h3]
    norm_num
  · rw [row0_code t hG f (by omega) h2]
    simp only [valAt, hc, Option.getD_some, h3]
    norm_num
  · -- beyond the code: a feature of row 0 holds nothing, or an input, which a head writes
    cases hrow : row0 t f with
    | none => rfl
    | some n =>
      exfalso
      rcases (row0_ok t hG).names f n hrow with hn | ⟨i, hi, hib⟩ | ⟨x, hx, hxn⟩
      · subst hn; have := hrok.one_only f hrow; omega
      · subst hi
        have h1i := row0_code t hG (1 + i) (by omega) (by omega)
        rw [show 1 + i - 1 = i by omega] at h1i
        have := hrok.nodup f (1 + i) (by rw [hrow, h1i]) (by rw [hrow]; rfl)
        omega
      · subst hxn
        have hfW : f < (tokenGrid t).width := by
          by_contra hcon
          rw [hrok.none_beyond f (by omega)] at hrow
          simp at hrow
        obtain ⟨k, hk, hkx⟩ := List.getElem_of_mem hx
        have hb : 0 < t.outputs.length := List.length_pos_of_ne_nil hprog.outputs_ne
        have hlen := len_eq t hprog
        have hkw : k / t.outputs.length < t.window := by
          rw [Nat.div_lt_iff_lt_mul hb]; omega
        have hdm := Nat.div_add_mod' k t.outputs.length
        obtain ⟨q, hq⟩ : ∃ q, q = k / t.outputs.length := ⟨_, rfl⟩
        rw [← hq] at hkw hdm
        apply hf (t.window - 1 - q) (by omega) (k % t.outputs.length) (Nat.mod_lt _ hb)
        have hkk : (t.window - 1 - (t.window - 1 - q)) * t.outputs.length
            + k % t.outputs.length = k := by
          rw [show t.window - 1 - (t.window - 1 - q) = q by omega]
          exact hdm
        unfold F
        rw [hkk, List.getD_eq_getElem _ _ hk, hkx]
        exact (featIn_eq _ _ _ f hfW hrow hrok.nodup).symm

end

end AttnBF16Fold

open AttnBF16Model AttnBF16Fold in
theorem attn_bf16_of (hg : Stmt_grid) (ht : Stmt_trace) (hn : Stmt_norm_exact)
    (hr : Stmt_round) : Stmt_attn_bf16 := by
  intro R t codes toks p hprog _ hpos hw
  have hG := hg t hprog
  show ∀ f, BF16.afterAttn R (transformer t codes) toks p f =
    ideal (BF16.one R (transformer t codes).net) (valAt t codes toks p) (row0 t) f
  intro f
  rw [afterAttn_X]
  obtain ⟨I1, I2⟩ := X_inv t codes R toks p hr hn hprog hG hw hpos t.window le_rfl
  by_cases hex : ∃ h < t.window, ∃ j < t.outputs.length, f = F t h j
  · obtain ⟨h, hh, j, hj, rfl⟩ := hex
    rw [I1 h hh j hj, ideal_F t codes R toks p ht hprog hG hpos h j hh hj]
  · have hex' : ∀ h < t.window, ∀ j < t.outputs.length, f ≠ F t h j :=
      fun h hh j hj he => hex ⟨h, hh, j, hj, he⟩
    rw [I2 f hex', ideal_rest t codes R toks p hr hprog hG hpos f hex']

end Reifier

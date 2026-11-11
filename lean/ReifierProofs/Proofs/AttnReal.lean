import ReifierProofs.Proofs.AttnRealAux2

namespace Reifier
namespace AttnRealPf

open Finset

noncomputable section

local notation "GW" t:max => Grid.width (tokenGrid t)
local notation "R0" t:max => List.headD (Grid.rows (tokenGrid t)) (fun _ => none)
local notation "MM" t:max codes:max => transformer t codes

section fields

variable (t : Trace) (codes : List (ℕ × List Bool))

lemma tr_W : (MM t codes).net.W = GW t := rfl
lemma tr_one : (MM t codes).net.one = 2 * Real.sqrt (GW t) := rfl
lemma tr_bits : (MM t codes).bits = t.outputs.length := rfl
lemma tr_heads : (MM t codes).heads = t.window := rfl
lemma tr_embedded (f : ℕ) :
    (MM t codes).embedded f = if f ≤ t.outputs.length then (R0 t) f else none := rfl
lemma tr_embed (τ f : ℕ) : (MM t codes).embed τ f =
    match codes.lookup τ with
    | some c => if f = 0 then 2 * Real.sqrt (GW t)
        else if f ≤ t.outputs.length then (if c.getD (f - 1) false then 1 else -1) else 0
    | none => 0 := rfl
lemma tr_q (h d f : ℕ) : (MM t codes).q h d f =
    if h < t.window ∧ f = 0 then attnScale * lookback h d else 0 := rfl
lemma tr_k (d f : ℕ) : (MM t codes).k d f = if d < PAIRS ∧ f = 0 then 1 else 0 := rfl
lemma tr_v (j f : ℕ) : (MM t codes).v j f =
    if j < t.outputs.length ∧ f = j + 1 then 1 else 0 := rfl
lemma tr_o (h f j : ℕ) : (MM t codes).o h f j =
    if h < t.window ∧ j < t.outputs.length ∧
        f = featIn (GW t) (R0 t) (.sig (t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0))
    then 1 else 0 := rfl

lemma embed_some {τ : ℕ} {c : List Bool} (hc : codes.lookup τ = some c) (f : ℕ) :
    (MM t codes).embed τ f = if f = 0 then 2 * Real.sqrt (GW t)
        else if f ≤ t.outputs.length then (if c.getD (f - 1) false then 1 else -1) else 0 := by
  rw [tr_embed, hc]

lemma embed_none {τ : ℕ} (hc : codes.lookup τ = none) (f : ℕ) :
    (MM t codes).embed τ f = 0 := by
  rw [tr_embed, hc]

end fields

lemma list_filter_length (W : ℕ) (P : ℕ → Bool) :
    ((List.range W).filter P).length = ((Finset.range W).filter (fun x => P x = true)).card := by
  rw [Finset.card_def, Finset.filter_val, Finset.range_val, Multiset.range, Multiset.filter_coe,
    Multiset.coe_card]
  simp

lemma filter_Icc (W b : ℕ) (h : b < W) :
    (Finset.range W).filter (fun g => 1 ≤ g ∧ g ≤ b) = Finset.Icc 1 b := by
  ext g; simp; omega

section zvec

variable {t : Trace} {codes : List (ℕ × List Bool)}

lemma count_embedded (hR0 : Row0OK t (GW t) (R0 t)) :
    count (GW t) (MM t codes).embedded = t.outputs.length := by
  unfold count
  rw [list_filter_length]
  have hW := hR0.width
  have e : (Finset.range (GW t)).filter
      (fun x => (decide (1 ≤ x) && ((MM t codes).embedded x).isSome) = true)
      = (Finset.range (GW t)).filter (fun g => 1 ≤ g ∧ g ≤ t.outputs.length) := by
    apply Finset.filter_congr
    intro g _
    rw [tr_embedded]
    by_cases h1 : 1 ≤ g
    · by_cases h2 : g ≤ t.outputs.length
      · have := hR0.code (g - 1) (by omega)
        rw [show 1 + (g - 1) = g by omega] at this
        simp [h1, h2, this]
      · simp [h2]
    · simp [h1]
  rw [e, filter_Icc _ _ (by omega)]
  simp

lemma sumsq_embed (hR0 : Row0OK t (GW t) (R0 t)) {τ : ℕ} {c : List Bool}
    (hc : codes.lookup τ = some c) :
    ∑ g ∈ range (GW t), (MM t codes).embed τ g ^ 2
      = (2 * Real.sqrt (GW t)) ^ 2 + t.outputs.length := by
  have hW := hR0.width
  have e : ∀ g ∈ range (GW t), (MM t codes).embed τ g ^ 2
      = (if g = 0 then (2 * Real.sqrt (GW t)) ^ 2 else 0)
        + (if 1 ≤ g ∧ g ≤ t.outputs.length then 1 else 0) := by
    intro g _
    rw [embed_some t codes hc]
    by_cases h0 : g = 0
    · simp [h0]
    · by_cases h2 : g ≤ t.outputs.length
      · have h1 : 1 ≤ g := by omega
        simp only [h0, h2, h1, ite_false, ite_true, and_self, zero_add]
        split_ifs <;> norm_num
      · simp [h0, h2]
  rw [Finset.sum_congr rfl e, Finset.sum_add_distrib, Finset.sum_ite_eq' (range (GW t)) 0,
    Finset.sum_boole, filter_Icc _ _ (by omega)]
  have h0 : 0 ∈ range (GW t) := by simp; omega
  simp [h0]

lemma one_pos' (hR0 : Row0OK t (GW t) (R0 t)) : 0 < 2 * Real.sqrt (GW t) := by
  have hW := hR0.width
  have : (0:ℝ) < GW t := by exact_mod_cast (show 0 < GW t by omega)
  positivity

lemma zvec_some (hR0 : Row0OK t (GW t) (R0 t)) {τ : ℕ} {c : List Bool}
    (hc : codes.lookup τ = some c) (f : ℕ) :
    (MM t codes).zvec τ f = if f = 0 then 1 else (MM t codes).embed τ f := by
  unfold Transformer.zvec rmsnorm rmsScale
  simp only [tr_W, tr_one]
  rw [count_embedded hR0, sumsq_embed hR0 hc]
  have hpos : 0 < ((2 * Real.sqrt (GW t)) ^ 2 + (t.outputs.length : ℝ)) / (GW t : ℝ) + EPS := by
    have hW := hR0.width
    have : (0:ℝ) < GW t := by exact_mod_cast (show 0 < GW t by omega)
    unfold EPS; positivity
  have hs : 0 < Real.sqrt (((2 * Real.sqrt (GW t)) ^ 2 + (t.outputs.length : ℝ)) / (GW t : ℝ) + EPS) :=
    Real.sqrt_pos.2 hpos
  rw [mul_div_cancel_left₀ _ hs.ne']
  split_ifs with h0
  · subst h0
    rw [embed_some t codes hc]
    simp only [ite_true]
    exact div_self (one_pos' hR0).ne'
  · rfl

lemma zvec_none {τ : ℕ} (hc : codes.lookup τ = none) (f : ℕ) :
    (MM t codes).zvec τ f = 0 := by
  unfold Transformer.zvec rmsnorm
  simp [embed_none t codes hc]

end zvec

section scores

variable {t : Trace} {codes : List (ℕ × List Bool)}

/-- what the value of a token holds at bit `j`: its code bit as `±1`, or 0 if it has no code -/
def cv (codes : List (ℕ × List Bool)) (τ j : ℕ) : ℝ :=
  match codes.lookup τ with
  | some c => if c.getD j false then 1 else -1
  | none => 0

lemma cv_abs (τ j : ℕ) : |cv codes τ j| ≤ 1 := by
  unfold cv
  split
  · split_ifs <;> norm_num
  · norm_num

lemma value_eq (hR0 : Row0OK t (GW t) (R0 t)) (toks : ℕ → ℕ) (p' j : ℕ)
    (hj : j < t.outputs.length) :
    (MM t codes).value toks p' j = cv codes (toks p') j := by
  have hW := hR0.width
  have e : (MM t codes).value toks p' j = (MM t codes).zvec (toks p') (j + 1) := by
    simp only [Transformer.value, Transformer.matvec, tr_W, tr_v]
    rw [Finset.sum_eq_single (j + 1)]
    · simp [hj]
    · intro f _ hf; simp [hf]
    · intro h; exfalso; simp at h; omega
  rw [e]
  unfold cv
  cases hc : codes.lookup (toks p') with
  | none => exact zvec_none hc _
  | some c =>
    rw [zvec_some hR0 hc, embed_some t codes hc]
    simp only [Nat.add_one_ne_zero, ite_false, show j + 1 ≤ t.outputs.length by omega, ite_true,
      Nat.add_sub_cancel]

lemma matvec_k (hR0 : Row0OK t (GW t) (R0 t)) (z : ℕ → ℝ) (d : ℕ) :
    (MM t codes).matvec (MM t codes).k z d = if d < 8 then z 0 else 0 := by
  have hW := hR0.width
  simp only [Transformer.matvec, tr_W, tr_k]
  rw [Finset.sum_eq_single 0]
  · simp [PAIRS]
  · intro f _ hf; simp [hf]
  · intro h; exfalso; simp at h; omega

lemma matvec_q (hR0 : Row0OK t (GW t) (R0 t)) (z : ℕ → ℝ) (h d : ℕ) (hh : h < t.window) :
    (MM t codes).matvec ((MM t codes).q h) z d = attnScale * lookback h d * z 0 := by
  have hW := hR0.width
  simp only [Transformer.matvec, tr_W, tr_q]
  rw [Finset.sum_eq_single 0]
  · simp [hh]
  · intro f _ hf; simp [hf]
  · intro h; exfalso; simp at h; omega

lemma score_trig (A c c' : ℝ) (h p p' : ℕ) :
    ∑ d ∈ range (2 * PAIRS), rope (fun d => A * lookback h d * c) p d *
        rope (fun d => if d < 8 then c' else 0) p' d
      = A * c * c' * ∑ i ∈ range 8, Real.cos (((p : ℝ) - h - p') * turn i) := by
  rw [show 2 * PAIRS = 8 + 8 from rfl, Finset.sum_range_add, ← Finset.sum_add_distrib,
    Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  have hi : i < 8 := by simpa using hi
  have h2 : ¬ (8 + i < 8) := by omega
  have h3 : 8 + i < 16 := by omega
  have h4 : 8 + i - 8 = i := by omega
  have h5 : ¬ (i + 8 < 8) := by omega
  have h6 : i + 8 < 16 := by omega
  have h7 : i + 8 - 8 = i := by omega
  simp only [rope, lookback, PAIRS, Nat.reduceMul, hi, h2, h3, h4, h5, h6, h7, ↓reduceIte]
  rw [show ((p : ℝ) - h - p') * turn i = (p * turn i - h * turn i) - p' * turn i by ring,
    Real.cos_sub, Real.cos_sub, Real.sin_sub]
  ring

lemma key_eq (hR0 : Row0OK t (GW t) (R0 t)) (toks : ℕ → ℕ) (p' : ℕ) :
    (MM t codes).key toks p' =
      rope (fun d => if d < 8 then (MM t codes).zvec (toks p') 0 else 0) p' := by
  unfold Transformer.key
  congr 1
  funext d
  exact matvec_k hR0 _ d

lemma query_eq (hR0 : Row0OK t (GW t) (R0 t)) (toks : ℕ → ℕ) (h p : ℕ) (hh : h < t.window) :
    (MM t codes).query toks h p =
      rope (fun d => attnScale * lookback h d * (MM t codes).zvec (toks p) 0) p := by
  unfold Transformer.query
  congr 1
  funext d
  exact matvec_q hR0 _ h d hh

lemma score_eq (hR0 : Row0OK t (GW t) (R0 t)) (toks : ℕ → ℕ) (h p p' : ℕ) (hh : h < t.window) :
    (MM t codes).score toks h p p' = attnScale * (MM t codes).zvec (toks p) 0
      * (MM t codes).zvec (toks p') 0 * ∑ i ∈ range 8, Real.cos (((p : ℝ) - h - p') * turn i) := by
  unfold Transformer.score
  rw [query_eq hR0 toks h p hh, key_eq hR0 toks p']
  exact score_trig _ _ _ h p p'

end scores

section heads

variable {t : Trace} {codes : List (ℕ × List Bool)}

lemma head_close (hP : t.IsProgram) (hR0 : Row0OK t (GW t) (R0 t)) {toks : ℕ → ℕ} {p : ℕ}
    (hpos : PosOK t codes toks p) {h j : ℕ} (hh : h < t.window) (hj : j < t.outputs.length) :
    |∑ p' ∈ visible p, (MM t codes).attn toks h p p' * (MM t codes).value toks p' j
      - cv codes (toks (p - h)) j| ≤ 1024 * Real.exp (-30) := by
  have hwin := hP.window_le
  simp only [CONTEXT] at hwin
  have hhp : h ≤ p := by have := hpos.1; omega
  obtain ⟨c0, hc0⟩ : ∃ c, codes.lookup (toks (p - 0)) = some c :=
    Option.isSome_iff_exists.1 (hpos.2 0 (Nat.zero_le _))
  rw [Nat.sub_zero] at hc0
  obtain ⟨c1, hc1⟩ : ∃ c, codes.lookup (toks (p - h)) = some c :=
    Option.isSome_iff_exists.1 (hpos.2 h (by omega))
  have hzq : (MM t codes).zvec (toks p) 0 = 1 := by rw [zvec_some hR0 hc0]; simp
  have hz1 : (MM t codes).zvec (toks (p - h)) 0 = 1 := by rw [zvec_some hR0 hc1]; simp
  have hsc : ∀ p', (MM t codes).score toks h p p' = attnScale * (MM t codes).zvec (toks p') 0 *
      ∑ i ∈ range 8, Real.cos (((p : ℝ) - h - p') * turn i) := by
    intro p'; rw [score_eq hR0 toks h p p' hh, hzq, mul_one]
  have htarget : (MM t codes).score toks h p (p - h) = 8 * attnScale := by
    rw [hsc, hz1]
    have : ((p : ℝ) - h - ((p - h : ℕ) : ℝ)) = 0 := by rw [Nat.cast_sub hhp]; ring
    simp only [this, zero_mul, Real.cos_zero, Finset.sum_const, Finset.card_range]
    simp; ring
  have hmargin := attnScale_margin
  have hApos := attnScale_pos
  have hr := runnerUp_pos
  have hval : ∀ p', (MM t codes).value toks p' j = cv codes (toks p') j :=
    fun p' => value_eq hR0 toks p' j hj
  have key := softmax_close (visible p) ((MM t codes).score toks h p)
    (fun p' => (MM t codes).value toks p' j) (p - h) ?mem ?sc ?vv (visible_card p)
  · simp only [hval] at key ⊢
    unfold Transformer.attn
    exact key
  case mem => rw [mem_visible]; omega
  case sc =>
    intro q hq hqne
    rw [mem_visible] at hq
    rw [htarget, hsc]
    cases hcq : codes.lookup (toks q) with
    | none =>
      rw [zvec_none hcq]
      nlinarith [mul_pos hApos hr]
    | some cq =>
      rw [zvec_some hR0 hcq]
      simp only [ite_true, mul_one]
      have hS : ∑ i ∈ range 8, Real.cos (((p : ℝ) - h - q) * turn i) ≤ 8 * runnerUp := by
        by_cases hle : q ≤ p - h
        · apply sum_cos_le _ (p - h - q) (Or.inl ?_) (by omega) (by omega)
          have e : p - h - q + q + h = p := by omega
          have e' : ((p - h - q : ℕ) : ℝ) + q + h = p := by exact_mod_cast e
          linarith
        · apply sum_cos_le _ (q - (p - h)) (Or.inr ?_) (by omega) (by omega)
          have e : q - (p - h) + p = q + h := by omega
          have e' : ((q - (p - h) : ℕ) : ℝ) + p = q + h := by exact_mod_cast e
          linarith
      nlinarith [mul_le_mul_of_nonneg_left hS hApos.le]
  case vv =>
    intro q _
    simp only [hval]
    have h1 := cv_abs (codes := codes) (toks q) j
    have h2 := cv_abs (codes := codes) (toks (p - h)) j
    rw [abs_le] at h1 h2 ⊢
    constructor <;> linarith

lemma featIn_spec {W : ℕ} {row : Row} {s : Name} (h : inRow W row s = true) :
    featIn W row s < W ∧ row (featIn W row s) = some s := by
  unfold inRow at h
  rw [List.any_eq_true] at h
  obtain ⟨x, hx, hpx⟩ := h
  have hs : ((List.range W).find? (fun f => row f == some s)).isSome :=
    List.find?_isSome.2 ⟨x, hx, hpx⟩
  obtain ⟨g, hg⟩ := Option.isSome_iff_exists.1 hs
  unfold featIn
  rw [hg, Option.getD_some]
  refine ⟨List.mem_range.1 (List.mem_of_find?_eq_some hg), ?_⟩
  have := List.find?_some hg
  simpa using this

lemma featIn_eq {W : ℕ} {row : Row} {s : Name} {f : ℕ} (hf : row f = some s) (hfW : f < W)
    (hnd : ∀ f g, row f = row g → (row f).isSome → f = g) : featIn W row s = f := by
  have hin : inRow W row s = true := by
    unfold inRow; rw [List.any_eq_true]; exact ⟨f, List.mem_range.2 hfW, by simp [hf]⟩
  obtain ⟨_, h2⟩ := featIn_spec hin
  exact hnd _ _ (h2.trans hf.symm) (by simp [h2])

lemma len_eq (hP : t.IsProgram) : t.inputs.length = t.window * t.outputs.length := by
  unfold Trace.window
  exact (Nat.div_mul_cancel hP.window_dvd).symm

lemma idx_lt (hP : t.IsProgram) {h j : ℕ} (hh : h < t.window) (hj : j < t.outputs.length) :
    (t.window - 1 - h) * t.outputs.length + j < t.inputs.length := by
  rw [len_eq hP]
  have : (t.window - 1 - h) + 1 ≤ t.window := by omega
  calc (t.window - 1 - h) * t.outputs.length + j
      < (t.window - 1 - h) * t.outputs.length + t.outputs.length := by omega
    _ = ((t.window - 1 - h) + 1) * t.outputs.length := by ring
    _ ≤ t.window * t.outputs.length := Nat.mul_le_mul_right _ this

lemma getD_mem (hP : t.IsProgram) {h j : ℕ} (hh : h < t.window) (hj : j < t.outputs.length) :
    t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0 ∈ t.inputs := by
  have hK := idx_lt hP hh hj
  rw [List.getD_eq_getElem _ _ hK]
  exact List.getElem_mem hK

lemma o_zero (hP : t.IsProgram) (hR0 : Row0OK t (GW t) (R0 t)) {f : ℕ}
    (hf : ∀ x, (R0 t) f ≠ some (.sig x)) (h j : ℕ) : (MM t codes).o h f j = 0 := by
  rw [tr_o]
  split_ifs with H
  · exfalso
    obtain ⟨hh, hj, hfe⟩ := H
    have hx := getD_mem hP hh hj
    have := (featIn_spec (hR0.inputs _ hx)).2
    rw [← hfe] at this
    exact hf _ this
  · rfl

lemma o_sum (hP : t.IsProgram) (hR0 : Row0OK t (GW t) (R0 t)) (hRw : RowOK t (GW t) (R0 t))
    {f : ℕ} {s : Sig} (hf : (R0 t) f = some (.sig s)) (hfW : f < GW t) (hs : s ∈ t.inputs)
    (H : ℕ → ℕ → ℝ) :
    ∑ h ∈ range t.window, ∑ j ∈ range t.outputs.length, (MM t codes).o h f j * H h j
      = H (t.window - 1 - t.inputs.idxOf s / t.outputs.length)
          (t.inputs.idxOf s % t.outputs.length) := by
  have hb : 0 < t.outputs.length := List.length_pos_iff.2 hP.outputs_ne
  have hk : t.inputs.idxOf s < t.inputs.length := List.idxOf_lt_length_of_mem hs
  have hlen := len_eq hP
  have hq : t.inputs.idxOf s / t.outputs.length < t.window := by
    rw [Nat.div_lt_iff_lt_mul hb]; omega
  have hRlt : t.inputs.idxOf s % t.outputs.length < t.outputs.length := Nat.mod_lt _ hb
  have hkQR := Nat.div_add_mod' (t.inputs.idxOf s) t.outputs.length
  have hnd : t.inputs.Nodup := (List.nodup_append.1 hP.nodup).1
  have iff1 : ∀ h < t.window, ∀ j < t.outputs.length, (f = featIn (GW t) (R0 t)
        (.sig (t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0)) ↔
        (t.window - 1 - h) * t.outputs.length + j = t.inputs.idxOf s) := by
    intro h hh j hj
    have hK := idx_lt hP hh hj
    constructor
    · intro hfe
      have hx := getD_mem hP hh hj
      have h2 := (featIn_spec (hR0.inputs _ hx)).2
      rw [← hfe, hf] at h2
      have hxs : t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0 = s := by
        injection h2 with h2; injection h2 with h2; exact h2.symm
      rw [List.getD_eq_getElem _ _ hK] at hxs
      rw [← hxs]
      exact (hnd.idxOf_getElem _ hK).symm
    · intro hKk
      rw [hKk]
      have : t.inputs.getD (t.inputs.idxOf s) 0 = s := by
        rw [List.getD_eq_getElem _ _ hk]; exact List.getElem_idxOf hk
      rw [this]
      exact (featIn_eq hf hfW hRw.nodup).symm
  have hdiv : ∀ h j, j < t.outputs.length →
      ((t.window - 1 - h) * t.outputs.length + j) / t.outputs.length = t.window - 1 - h ∧
      ((t.window - 1 - h) * t.outputs.length + j) % t.outputs.length = j := by
    intro h j hj
    constructor
    · rw [add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hj, zero_add]
    · rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  generalize t.inputs.idxOf s = k at hq hRlt hkQR iff1 ⊢
  generalize hQ : k / t.outputs.length = Q at hq hkQR ⊢
  generalize hR : k % t.outputs.length = Rr at hRlt hkQR ⊢
  have ho : ∀ h < t.window, ∀ j < t.outputs.length, (MM t codes).o h f j =
      if h = t.window - 1 - Q ∧ j = Rr then 1 else 0 := by
    intro h hh j hj
    rw [tr_o]
    have iff2 : (t.window - 1 - h) * t.outputs.length + j = k ↔ (h = t.window - 1 - Q ∧ j = Rr) := by
      constructor
      · intro e
        have e1 := (hdiv h j hj).1
        have e2 := (hdiv h j hj).2
        rw [e, hQ] at e1
        rw [e, hR] at e2
        omega
      · rintro ⟨rfl, rfl⟩
        have : t.window - 1 - (t.window - 1 - Q) = Q := by omega
        rw [this]; exact hkQR
    simp only [hh, hj, true_and, iff1 h hh j hj, iff2]
  rw [Finset.sum_eq_single_of_mem (t.window - 1 - Q) (by simp; omega)]
  · rw [Finset.sum_eq_single_of_mem Rr (by simp; omega)]
    · rw [ho _ (by omega) _ hRlt]; simp
    · intro j hj hne
      rw [ho _ (by omega) _ (by simpa using hj)]
      simp [hne]
  · intro h hh hne
    apply Finset.sum_eq_zero
    intro j hj
    rw [ho _ (by simpa using hh) _ (by simpa using hj)]
    simp [hne]

end heads

lemma headD_eq_head (l : List Row) (h : l ≠ []) (d : Row) : l.headD d = l.head h := by
  cases l with
  | nil => exact absurd rfl h
  | cons a l => rfl

lemma two_mul_ite (B : Bool) :
    2 * (if B = true then (1 : ℝ) else 0) - 1 = if B = true then 1 else -1 := by
  split_ifs <;> norm_num

theorem attn_close (ht : Stmt_trace) {t : Trace} {codes : List (ℕ × List Bool)}
    {toks : ℕ → ℕ} {p : ℕ} (hP : t.IsProgram) (hpos : PosOK t codes toks p)
    (hR0 : Row0OK t (GW t) (R0 t)) (hRw : RowOK t (GW t) (R0 t)) :
    Close (GW t) (1 / 10 ^ 10) (2 * Real.sqrt (GW t)) (valAt t codes toks p) (R0 t)
      ((MM t codes).afterAttn toks p) := by
  obtain ⟨c0, hc0⟩ : ∃ c, codes.lookup (toks (p - 0)) = some c :=
    Option.isSome_iff_exists.1 (hpos.2 0 (Nat.zero_le _))
  rw [Nat.sub_zero] at hc0
  have hafter : ∀ f, (MM t codes).afterAttn toks p f = (MM t codes).embed (toks p) f +
      ∑ h ∈ range t.window, ∑ j ∈ range t.outputs.length, (MM t codes).o h f j *
        ∑ p' ∈ visible p, (MM t codes).attn toks h p p' * (MM t codes).value toks p' j :=
    fun f => rfl
  have hnosig : ∀ f, (∀ x, (R0 t) f ≠ some (.sig x)) →
      (MM t codes).afterAttn toks p f = (MM t codes).embed (toks p) f := by
    intro f hf
    rw [hafter]
    simp [o_zero hP hR0 hf]
  have hlow : ∀ f, 1 ≤ f → f ≤ t.outputs.length → (R0 t) f = some (.code (f - 1)) := by
    intro f h1 h2
    have := hR0.code (f - 1) (by omega)
    rwa [show 1 + (f - 1) = f by omega] at this
  refine ⟨?_, ?_, ?_⟩
  · -- `ONE`
    have h0 : ∀ x, (R0 t) 0 ≠ some (.sig x) := by intro x; rw [hRw.one_zero]; simp
    rw [hnosig 0 h0, embed_some t codes hc0]
    simp
  · -- empty cells
    intro f hf1 hfn
    have : ∀ x, (R0 t) f ≠ some (.sig x) := by intro x; rw [hfn]; simp
    rw [hnosig f this, embed_some t codes hc0]
    have hfb : ¬ f ≤ t.outputs.length := by
      intro hle
      have := hlow f hf1 hle
      rw [hfn] at this
      simp at this
    simp [hfb, show f ≠ 0 by omega]
  · intro f hf1 hfW
    cases hr : (R0 t) f with
    | none =>
      have : ∀ x, (R0 t) f ≠ some (.sig x) := by intro x; rw [hr]; simp
      rw [hnosig f this, embed_some t codes hc0]
      have hfb : ¬ f ≤ t.outputs.length := by
        intro hle
        have := hlow f hf1 hle
        rw [hr] at this
        simp at this
      simp [ideal, hr, hfb, show f ≠ 0 by omega]
    | some n =>
      rcases hR0.names f n hr with hn | ⟨i, hn, hi⟩ | ⟨x, hx, hn⟩
      · subst hn
        have := hRw.one_only f hr
        omega
      · -- a bit of the token's own code
        subst hn
        have hci := hR0.code i hi
        have hfi : f = 1 + i := hRw.nodup f (1 + i) (hr.trans hci.symm) (by simp [hr])
        have : ∀ x, (R0 t) f ≠ some (.sig x) := by intro x; rw [hr]; simp
        rw [hnosig f this, embed_some t codes hc0]
        simp only [ideal, hr, valAt, hc0, Option.getD_some]
        rw [two_mul_ite]
        simp [hfi, show 1 + i ≤ t.outputs.length by omega]
      · -- an input: one head and one bit write it
        subst hn
        have hb : 0 < t.outputs.length := List.length_pos_iff.2 hP.outputs_ne
        have hfb : ¬ f ≤ t.outputs.length := by
          intro hle
          have := hlow f hf1 hle
          rw [hr] at this
          simp at this
        rw [hafter, o_sum hP hR0 hRw hr hfW hx, embed_some t codes hc0]
        simp only [show f ≠ 0 by omega, hfb, ite_false, zero_add]
        set k := t.inputs.idxOf x with hk
        have hkl : k < t.inputs.length := List.idxOf_lt_length_of_mem hx
        have hlen := len_eq hP
        have hq : k / t.outputs.length < t.window := by
          rw [Nat.div_lt_iff_lt_mul hb]; omega
        have hj : k % t.outputs.length < t.outputs.length := Nat.mod_lt _ hb
        have hlt : t.window - 1 - k / t.outputs.length < t.window := by
          generalize k / t.outputs.length = Q at hq ⊢
          omega
        have hhc := head_close hP hR0 hpos (codes := codes) (h := t.window - 1 - k / t.outputs.length)
          hlt hj
        obtain ⟨c1, hc1⟩ : ∃ c, codes.lookup (toks (p - (t.window - 1 - k / t.outputs.length)))
            = some c := Option.isSome_iff_exists.1 (hpos.2 _ (by omega))
        -- the ideal value
        have hval : valAt t codes toks p (.sig x) =
            if windowBits t codes toks p k then 1 else 0 := by
          simp only [valAt]
          rw [(ht t _ hP.toWellFormed).1 x hx]
          simp only [Trace.env, hk]
        have hwin : p + 1 + k / t.outputs.length - t.window
            = p - (t.window - 1 - k / t.outputs.length) := by
          have := hpos.1
          generalize k / t.outputs.length = Q at hq ⊢
          omega
        have hwb : windowBits t codes toks p k =
            ((codes.lookup (toks (p - (t.window - 1 - k / t.outputs.length)))).getD []).getD
              (k % t.outputs.length) false := by
          simp only [windowBits]
          rw [hwin]
        have hcv : cv codes (toks (p - (t.window - 1 - k / t.outputs.length)))
            (k % t.outputs.length) = 2 * valAt t codes toks p (.sig x) - 1 := by
          rw [hval, hwb, two_mul_ite]
          unfold cv
          rw [hc1]
          simp
        simp only [ideal, hr]
        rw [← hcv]
        exact hhc.trans (exp_bound.le)

end

end AttnRealPf

open Finset

/-- **The attention, in real arithmetic**: the stream after the attention is `10^-10` close to
row 0's. -/
theorem attn_real_of (hg : Stmt_grid) (ht : Stmt_trace) : Stmt_attn_real := by
  intro t codes toks p hP _ hpos
  have hG := hg t hP
  have hhead := AttnRealPf.headD_eq_head (tokenGrid t).rows hG.rows_ne (fun _ => none)
  have hR0 : Row0OK t (tokenGrid t).width ((tokenGrid t).rows.headD fun _ => none) := by
    rw [hhead]; exact hG.row0
  have hRw : RowOK t (tokenGrid t).width ((tokenGrid t).rows.headD fun _ => none) := by
    rw [hhead]; exact hG.rows_ok _ (List.head_mem _)
  exact AttnRealPf.attn_close ht hP hpos hR0 hRw

end Reifier

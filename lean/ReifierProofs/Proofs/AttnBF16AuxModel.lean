import ReifierProofs.Main
import ReifierProofs.Proofs.AttnBF16AuxNum
import ReifierProofs.Proofs.AttnBF16AuxErr

namespace Reifier
namespace AttnBF16Model
open Finset AttnBF16Err AttnBF16Num

noncomputable section

variable (t : Trace) (codes : List (ℕ × List Bool))

/-- row 0 of the token grid -/
abbrev row0 : Row := (tokenGrid t).rows.headD fun _ => none

lemma M_W : (transformer t codes).net.W = (tokenGrid t).width := rfl
lemma M_one : (transformer t codes).net.one = 2 * Real.sqrt (tokenGrid t).width := rfl
lemma M_grid : (transformer t codes).net.grid = (tokenGrid t).rows := rfl
lemma M_bits : (transformer t codes).bits = t.outputs.length := rfl
lemma M_heads : (transformer t codes).heads = t.window := rfl
lemma M_embed (τ f : ℕ) : (transformer t codes).embed τ f =
    match codes.lookup τ with
      | some c => if f = 0 then 2 * Real.sqrt (tokenGrid t).width
          else if f ≤ t.outputs.length then (if c.getD (f - 1) false then 1 else -1) else 0
      | none => 0 := rfl
lemma M_k (d f : ℕ) : (transformer t codes).k d f = if d < PAIRS ∧ f = 0 then 1 else 0 := rfl
lemma M_q (h d f : ℕ) : (transformer t codes).q h d f =
    if h < t.window ∧ f = 0 then attnScale * lookback h d else 0 := rfl
lemma M_v (j f : ℕ) : (transformer t codes).v j f =
    if j < t.outputs.length ∧ f = j + 1 then 1 else 0 := rfl
lemma M_o (h f j : ℕ) : (transformer t codes).o h f j =
    if h < t.window ∧ j < t.outputs.length ∧
      f = featIn (tokenGrid t).width (row0 t)
        (.sig (t.inputs.getD ((t.window - 1 - h) * t.outputs.length + j) 0)) then 1 else 0 := rfl
lemma M_embedded (f : ℕ) : (transformer t codes).embedded f =
    if f ≤ t.outputs.length then row0 t f else none := rfl

/-! ## Row 0 -/

omit t in
lemma headD_eq_head {α : Type*} (l : List α) (h : l ≠ []) (d : α) : l.headD d = l.head h := by
  cases l with
  | nil => exact absurd rfl h
  | cons a l => rfl

lemma row0_eq (hG : GridOK t (tokenGrid t)) : row0 t = (tokenGrid t).rows.head hG.rows_ne :=
  headD_eq_head _ _ _

lemma row0_ok (hG : GridOK t (tokenGrid t)) : Row0OK t (tokenGrid t).width (row0 t) := by
  rw [row0_eq t hG]; exact hG.row0

lemma row0_rowok (hG : GridOK t (tokenGrid t)) : RowOK t (tokenGrid t).width (row0 t) := by
  rw [row0_eq t hG]; exact hG.rows_ok _ (List.head_mem _)

lemma featIn_spec (W : ℕ) (row : Row) (n : Name) (h : inRow W row n = true) :
    featIn W row n < W ∧ row (featIn W row n) = some n := by
  unfold inRow at h
  unfold featIn
  cases hf : (List.range W).find? (fun f => row f == some n) with
  | none =>
    rw [List.find?_eq_none] at hf
    obtain ⟨f, hf1, hf2⟩ := List.any_eq_true.mp h
    exact absurd hf2 (hf f hf1)
  | some f =>
    have h1 := List.find?_some hf
    have h2 := List.mem_of_find?_eq_some hf
    simp only [Option.getD_some]
    exact ⟨List.mem_range.mp h2, by simpa using h1⟩

lemma featIn_eq (W : ℕ) (row : Row) (n : Name) (f : ℕ) (hf : f < W) (hrow : row f = some n)
    (hnd : ∀ f g, row f = row g → (row f).isSome → f = g) : featIn W row n = f := by
  have hin : inRow W row n = true := by
    unfold inRow; exact List.any_eq_true.mpr ⟨f, List.mem_range.mpr hf, by simp [hrow]⟩
  obtain ⟨h1, h2⟩ := featIn_spec W row n hin
  exact hnd _ _ (by rw [h2, hrow]) (by rw [h2]; rfl)

lemma width_ge (hG : GridOK t (tokenGrid t)) : t.outputs.length + 1 ≤ (tokenGrid t).width :=
  (row0_ok t hG).width

lemma row0_zero (hG : GridOK t (tokenGrid t)) : row0 t 0 = some .one := (row0_rowok t hG).one_zero

lemma row0_code (hG : GridOK t (tokenGrid t)) (f : ℕ) (h1 : 1 ≤ f) (h2 : f ≤ t.outputs.length) :
    row0 t f = some (.code (f - 1)) := by
  have := (row0_ok t hG).code (f - 1) (by omega)
  rwa [show 1 + (f - 1) = f by omega] at this

lemma embedded_val (hG : GridOK t (tokenGrid t)) (f : ℕ) : (transformer t codes).embedded f =
    if f = 0 then some .one else if f ≤ t.outputs.length then some (.code (f - 1)) else none := by
  rw [M_embedded]
  split_ifs with h1 h2 h2
  · subst h2; exact row0_zero t hG
  · exact row0_code t hG f (by omega) h1
  · omega
  · rfl

lemma embedded_rowok (hG : GridOK t (tokenGrid t)) :
    RowOK t (tokenGrid t).width (transformer t codes).embedded := by
  have hW := width_ge t hG
  have hval := embedded_val t codes hG
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [hval]; simp
  · intro f hf; rw [hval] at hf; split_ifs at hf with h1 h2 <;> simp_all
  · intro f hf; rw [hval]; split_ifs with h1 h2 <;> first | rfl | omega
  · intro f g hfg hs
    rw [hval, hval] at hfg; rw [hval] at hs
    split_ifs at hfg hs with h1 h2 h3 h4 h5 h6 <;> simp_all
    omega
  · intro f n hf
    rw [hval] at hf
    split_ifs at hf with h1 h2
    · simp_all
    · right; left; refine ⟨f - 1, ?_, by omega⟩; simp_all

/-! ## The embedding -/

variable (R : Rounding)

lemma oneB_eq : BF16.one R (transformer t codes).net = R.fl (2 * Real.sqrt (tokenGrid t).width) :=
  rfl

lemma oneB_bf16 (hw : WidthOK t) : IsBF16 (BF16.one R (transformer t codes).net) := by
  rw [oneB_eq]
  apply R.fl_mem
  rw [abs_of_nonneg (by positivity)]
  exact hw

lemma oneB_ge (hr : Stmt_round) (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) :
    2 ≤ BF16.one R (transformer t codes).net := by
  rw [oneB_eq]
  have hW := width_ge t hG
  have h1 : (1 : ℝ) ≤ (tokenGrid t).width := by exact_mod_cast (by omega : 1 ≤ (tokenGrid t).width)
  have h2 : (1 : ℝ) ≤ Real.sqrt (tokenGrid t).width := by
    rw [Real.le_sqrt (by norm_num) (by positivity)]; simpa using h1
  have h3 := fl_mono hr R (x := 2) (y := 2 * Real.sqrt (tokenGrid t).width)
    (le_max (by norm_num)) (by rw [abs_of_nonneg (by positivity)]; exact hw) (by linarith)
  rwa [fl_exact hr R (bf16_two hr)] at h3

lemma embed_coded (hr : Stmt_round) {τ : ℕ} {c : List Bool} (hc : codes.lookup τ = some c) (f : ℕ) :
    BF16.embed R (transformer t codes) τ f =
      if f = 0 then BF16.one R (transformer t codes).net
      else if f ≤ t.outputs.length then (if c.getD (f - 1) false then 1 else -1) else 0 := by
  unfold BF16.embed
  rw [M_embed]
  simp only [hc]
  split_ifs
  · rfl
  · exact fl_one hr R
  · exact fl_neg_one hr R
  · exact fl_zero hr R

lemma embed_none (hr : Stmt_round) {τ : ℕ} (hc : codes.lookup τ = none) (f : ℕ) :
    BF16.embed R (transformer t codes) τ f = 0 := by
  unfold BF16.embed
  rw [M_embed]
  simp only [hc]
  exact fl_zero hr R

/-- the values of the names of a token's embedding -/
def valTok (c : List Bool) : Name → ℝ
  | .one => 0
  | .sig s => t.values (t.env fun _ => false) s
  | .code i => if c.getD i false then 1 else 0

lemma valTok_ok (c : List Bool) : ValOK t (valTok t c) := by
  refine ⟨⟨fun _ => false, fun s => rfl⟩, fun i => ?_⟩
  simp only [valTok, IsBit]; split_ifs <;> simp

lemma embed_ideal (hr : Stmt_round) (hG : GridOK t (tokenGrid t)) {τ : ℕ} {c : List Bool}
    (hc : codes.lookup τ = some c) :
    BF16.embed R (transformer t codes) τ =
      ideal (BF16.one R (transformer t codes).net) (valTok t c) (transformer t codes).embedded := by
  funext f
  rw [embed_coded t codes R hr hc f]
  unfold ideal
  rw [embedded_val t codes hG f]
  split_ifs with h1 h2 h3
  · rfl
  · simp only [valTok, h3, ite_true]; norm_num
  · simp only [valTok, h3]; norm_num
  · rfl

lemma zvec_coded (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) {τ : ℕ} {c : List Bool}
    (hc : codes.lookup τ = some c) (f : ℕ) :
    BF16.zvec R (transformer t codes) τ f =
      if f = 0 then 1 else if f ≤ t.outputs.length then (if c.getD (f - 1) false then 1 else -1)
      else 0 := by
  unfold BF16.zvec
  rw [embed_ideal t codes R hr hG hc]
  have hW := width_ge t hG
  have h := hn t (tokenGrid t).width (transformer t codes).embedded (valTok t c)
    (BF16.one R (transformer t codes).net) (embedded_rowok t codes hG) (valTok_ok t c)
    hprog.toWellFormed (by linarith [oneB_ge t codes R hr hG hw]) (by omega) f
  rw [M_W, h]
  unfold ideal
  rw [embedded_val t codes hG f]
  split_ifs with h1 h2 h3
  · exact fl_one hr R
  · simp only [valTok, h3, ite_true]; norm_num; exact fl_one hr R
  · simp only [valTok, h3]; norm_num; exact fl_neg_one hr R
  · exact fl_zero hr R

lemma zvec_none (hr : Stmt_round) {τ : ℕ} (hc : codes.lookup τ = none) (f : ℕ) :
    BF16.zvec R (transformer t codes) τ f = 0 := by
  unfold BF16.zvec
  have : BF16.embed R (transformer t codes) τ = fun _ => 0 := funext (embed_none t codes R hr hc)
  rw [this]
  simp [rmsnorm, fl_zero hr R]

/-! ## The matmuls -/

omit t codes R in
lemma sum_single (W : ℕ) (F : ℕ → ℝ) (f0 : ℕ) (hf0 : f0 < W) (h : ∀ f, f ≠ f0 → F f = 0) :
    ∑ f ∈ range W, F f = F f0 :=
  Finset.sum_eq_single_of_mem f0 (Finset.mem_range.mpr hf0) (fun f _ hf => h f hf)

lemma matvec_k (hr : Stmt_round) (hG : GridOK t (tokenGrid t)) (z : ℕ → ℝ) (c0 : ℝ)
    (hz : z 0 = c0) (hc0 : R.fl c0 = c0) (d : ℕ) :
    BF16.matvec R (transformer t codes) (transformer t codes).k z d = if d < 8 then c0 else 0 := by
  have hW := width_ge t hG
  unfold BF16.matvec
  rw [M_W, sum_single _ _ 0 (by omega)]
  · rw [M_k, hz]
    by_cases hd : d < 8
    · simp only [PAIRS, hd, and_self, ite_true, fl_one hr R, one_mul, hc0]
    · simp only [PAIRS, hd, false_and, ite_false, fl_zero hr R, zero_mul]
  · intro f hf
    rw [M_k]
    simp [hf, fl_zero hr R]

lemma matvec_k_coded (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) {τ : ℕ} {c : List Bool}
    (hc : codes.lookup τ = some c) :
    BF16.matvec R (transformer t codes) (transformer t codes).k (BF16.zvec R (transformer t codes) τ)
      = fun d => if d < 8 then 1 else 0 := by
  funext d
  exact matvec_k t codes R hr hG _ 1 (by rw [zvec_coded t codes R hr hn hprog hG hw hc]; simp)
    (fl_one hr R) d

lemma matvec_k_none (hr : Stmt_round) (hG : GridOK t (tokenGrid t)) {τ : ℕ}
    (hc : codes.lookup τ = none) :
    BF16.matvec R (transformer t codes) (transformer t codes).k (BF16.zvec R (transformer t codes) τ)
      = fun _ => 0 := by
  funext d
  rw [matvec_k t codes R hr hG _ 0 (zvec_none t codes R hr hc 0) (fl_zero hr R) d]
  simp

lemma key_lo (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p' : ℕ) {c : List Bool}
    (hc : codes.lookup (toks p') = some c) (i : ℕ) (hi : i < 8) :
    BF16.key R (transformer t codes) toks p' i = R.fl (Real.cos (p' * turn i)) := by
  unfold BF16.key
  rw [matvec_k_coded t codes R hr hn hprog hG hw hc]
  unfold rope
  simp [PAIRS, hi]

lemma key_hi (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p' : ℕ) {c : List Bool}
    (hc : codes.lookup (toks p') = some c) (i : ℕ) (hi : i < 8) :
    BF16.key R (transformer t codes) toks p' (8 + i) = R.fl (Real.sin (p' * turn i)) := by
  unfold BF16.key
  rw [matvec_k_coded t codes R hr hn hprog hG hw hc]
  unfold rope
  simp [PAIRS, hi, show 8 + i < 16 by omega, show ¬ (8 + i < 8) by omega]

lemma key_none (hr : Stmt_round) (hG : GridOK t (tokenGrid t)) (toks : ℕ → ℕ) (p' : ℕ)
    (hc : codes.lookup (toks p') = none) (d : ℕ) :
    BF16.key R (transformer t codes) toks p' d = 0 := by
  unfold BF16.key
  rw [matvec_k_none t codes R hr hG hc]
  unfold rope
  simp [fl_zero hr R]

omit t codes R in
lemma lookback_le (h d : ℕ) : |lookback h d| ≤ 1 := by
  unfold lookback
  split_ifs
  · exact Real.abs_cos_le_one _
  · rw [abs_neg]; exact Real.abs_sin_le_one _
  · simp

lemma qt_eq (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) {τ : ℕ} {c : List Bool}
    (hc : codes.lookup τ = some c) (h : ℕ) (hh : h < t.window) (d : ℕ) :
    BF16.matvec R (transformer t codes) ((transformer t codes).q h) (BF16.zvec R (transformer t codes) τ) d
      = R.fl (attnScale * lookback h d) := by
  have hW := width_ge t hG
  unfold BF16.matvec
  rw [M_W, sum_single _ _ 0 (by omega)]
  · rw [M_q, zvec_coded t codes R hr hn hprog hG hw hc]
    simp only [hh, and_self, ite_true, mul_one]
    apply fl_fl hr R
    apply AttnBF16Err.le_max
    rw [abs_mul]
    have := lookback_le h d
    have := attnScale_le
    have := attnScale_ge
    rw [abs_of_pos (by linarith)]
    nlinarith [abs_nonneg (lookback h d)]
  · intro f hf
    rw [M_q]
    simp [hf, fl_zero hr R]

omit t codes R in
lemma lookback_lo (h i : ℕ) (hi : i < 8) : lookback h i = Real.cos (h * turn i) := by
  unfold lookback; simp [PAIRS, hi]

omit t codes R in
lemma lookback_hi (h i : ℕ) (hi : i < 8) : lookback h (i + 8) = -Real.sin (h * turn i) := by
  unfold lookback; simp [PAIRS, show i + 8 < 16 by omega]

omit t codes R in
lemma lookback_hi' (h i : ℕ) (hi : i < 8) : lookback h (8 + i) = -Real.sin (h * turn i) := by
  rw [add_comm]; exact lookback_hi h i hi

lemma query_lo (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p : ℕ) {c : List Bool}
    (hc : codes.lookup (toks p) = some c) (h : ℕ) (hh : h < t.window) (i : ℕ) (hi : i < 8) :
    BF16.query R (transformer t codes) toks h p i =
      R.fl (R.fl (attnScale * Real.cos (h * turn i)) * Real.cos (p * turn i)
        - R.fl (attnScale * -Real.sin (h * turn i)) * Real.sin (p * turn i)) := by
  unfold BF16.query
  unfold rope
  simp only [PAIRS, hi, ite_true]
  rw [qt_eq t codes R hr hn hprog hG hw hc h hh, qt_eq t codes R hr hn hprog hG hw hc h hh,
    lookback_lo h i hi, lookback_hi h i hi]

lemma query_hi (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p : ℕ) {c : List Bool}
    (hc : codes.lookup (toks p) = some c) (h : ℕ) (hh : h < t.window) (i : ℕ) (hi : i < 8) :
    BF16.query R (transformer t codes) toks h p (8 + i) =
      R.fl (R.fl (attnScale * -Real.sin (h * turn i)) * Real.cos (p * turn i)
        + R.fl (attnScale * Real.cos (h * turn i)) * Real.sin (p * turn i)) := by
  unfold BF16.query
  unfold rope
  simp only [PAIRS, show ¬ (8 + i < 8) by omega, show 8 + i < 2 * 8 by omega, ite_true, ite_false,
    show 8 + i - 8 = i by omega]
  rw [qt_eq t codes R hr hn hprog hG hw hc h hh, qt_eq t codes R hr hn hprog hG hw hc h hh,
    lookback_lo h i hi, lookback_hi' h i hi]

lemma value_coded (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p' : ℕ) {c : List Bool}
    (hc : codes.lookup (toks p') = some c) (j : ℕ) (hj : j < t.outputs.length) :
    BF16.value R (transformer t codes) toks p' j = if c.getD j false then 1 else -1 := by
  have hW := width_ge t hG
  unfold BF16.value BF16.matvec
  rw [M_W, sum_single _ _ (j + 1) (by omega)]
  · rw [M_v, zvec_coded t codes R hr hn hprog hG hw hc]
    simp only [hj, and_self, ite_true, fl_one hr R, one_mul, show j + 1 ≠ 0 by omega,
      show j + 1 ≤ t.outputs.length by omega, show j + 1 - 1 = j by omega, ite_false]
    split_ifs
    · exact fl_one hr R
    · exact fl_neg_one hr R
  · intro f hf
    rw [M_v]
    simp [hf, fl_zero hr R]

lemma value_none (hr : Stmt_round) (toks : ℕ → ℕ) (p' : ℕ)
    (hc : codes.lookup (toks p') = none) (j : ℕ) :
    BF16.value R (transformer t codes) toks p' j = 0 := by
  unfold BF16.value BF16.matvec
  simp [zvec_none t codes R hr hc, fl_zero hr R]

/-! ## The scores -/

lemma fl_abs_le (hr : Stmt_round) {x : ℝ} (hx : |x| ≤ maxBF16) : |R.fl x| ≤ 2 * |x| + 1 := by
  have := fl_err hr R hx
  have := abs_sub_abs_le_abs_sub (R.fl x) x
  have : |x| / 2 ^ 8 + 1 / 2 ^ 134 ≤ |x| + 1 := by
    have : |x| / 2 ^ 8 ≤ |x| := by
      apply div_le_self (abs_nonneg _) (by norm_num)
    have : (1 : ℝ) / 2 ^ 134 ≤ 1 := by norm_num
    linarith
  linarith

omit t codes R in
lemma trig_pair (A α β γ : ℝ) :
    A * ((Real.cos α * Real.cos β + Real.sin α * Real.sin β) * Real.cos γ +
      (Real.cos α * Real.sin β - Real.sin α * Real.cos β) * Real.sin γ) = A * Real.cos (β - α - γ) := by
  rw [Real.cos_sub, Real.cos_sub, Real.sin_sub]; ring

lemma score_coded (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p p' : ℕ) {c c' : List Bool}
    (hc : codes.lookup (toks p) = some c) (hc' : codes.lookup (toks p') = some c')
    (h : ℕ) (hh : h < t.window) :
    |BF16.score R (transformer t codes) toks h p p' -
      ∑ i ∈ range 8, attnScale * Real.cos (((p : ℝ) - h - p') * turn i)| ≤ 8 := by
  unfold BF16.score
  rw [show 2 * PAIRS = 16 from rfl]
  have hA := attnScale_le
  have hA0 := attnScale_ge
  apply score_bound hr R attnScale hA
  · intro i hi
    rw [query_lo t codes R hr hn hprog hG hw toks p hc h hh i hi,
      query_hi t codes R hr hn hprog hG hw toks p hc h hh i hi,
      key_lo t codes R hr hn hprog hG hw toks p' hc' i hi,
      key_hi t codes R hr hn hprog hG hw toks p' hc' i hi]
    set α := (h : ℝ) * turn i
    set β := (p : ℝ) * turn i
    set γ := (p' : ℝ) * turn i
    have hang : ((p : ℝ) - h - p') * turn i = β - α - γ := by ring
    rw [hang, ← trig_pair]
    have b1 : |attnScale * Real.cos α| ≤ 63 := by
      rw [abs_mul, abs_of_pos (by linarith)]
      nlinarith [Real.abs_cos_le_one α, abs_nonneg (Real.cos α)]
    have b2 : |attnScale * -Real.sin α| ≤ 63 := by
      rw [abs_mul, abs_of_pos (by linarith), abs_neg]
      nlinarith [Real.abs_sin_le_one α, abs_nonneg (Real.sin α)]
    have m1 : |attnScale * Real.cos α| ≤ maxBF16 := AttnBF16Err.le_max (by linarith)
    have m2 : |attnScale * -Real.sin α| ≤ maxBF16 := AttnBF16Err.le_max (by linarith)
    have hq1 := fl_abs_le R hr m1
    have hq2 := fl_abs_le R hr m2
    set q1 := R.fl (attnScale * Real.cos α)
    set q2 := R.fl (attnScale * -Real.sin α)
    have hcb := Real.abs_cos_le_one β
    have hsb := Real.abs_sin_le_one β
    have r1 : |q1 * Real.cos β - q2 * Real.sin β| ≤ maxBF16 := by
      apply AttnBF16Err.le_max
      have := abs_sub (q1 * Real.cos β) (q2 * Real.sin β)
      rw [abs_mul, abs_mul] at this
      have : |q1| * |Real.cos β| ≤ |q1| := mul_le_of_le_one_right (abs_nonneg _) hcb
      have : |q2| * |Real.sin β| ≤ |q2| := mul_le_of_le_one_right (abs_nonneg _) hsb
      linarith
    have r2 : |q2 * Real.cos β + q1 * Real.sin β| ≤ maxBF16 := by
      apply AttnBF16Err.le_max
      have := abs_add_le (q2 * Real.cos β) (q1 * Real.sin β)
      rw [abs_mul, abs_mul] at this
      have : |q2| * |Real.cos β| ≤ |q2| := mul_le_of_le_one_right (abs_nonneg _) hcb
      have : |q1| * |Real.sin β| ≤ |q1| := mul_le_of_le_one_right (abs_nonneg _) hsb
      linarith
    have k1 : |Real.cos γ| ≤ maxBF16 := AttnBF16Err.le_max (by linarith [Real.abs_cos_le_one γ])
    have k2 : |Real.sin γ| ≤ maxBF16 := AttnBF16Err.le_max (by linarith [Real.abs_sin_le_one γ])
    exact pair_core attnScale (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) (Real.cos γ)
      (Real.sin γ) q1 q2 _ _ _ _ (1 / 2 ^ 134) (by linarith) hA (by positivity) (by norm_num)
      (Real.cos_sq_add_sin_sq α) (Real.cos_sq_add_sin_sq β) (Real.cos_sq_add_sin_sq γ)
      (fl_err hr R m1) (fl_err hr R m2) (fl_err hr R r1) (fl_err hr R r2) (fl_err hr R k1)
      (fl_err hr R k2)
  · intro i _
    rw [abs_mul, abs_of_pos (by linarith)]
    exact mul_le_of_le_one_right (by linarith) (Real.abs_cos_le_one _)

lemma score_none (hr : Stmt_round) (hG : GridOK t (tokenGrid t)) (toks : ℕ → ℕ) (h p p' : ℕ)
    (hc' : codes.lookup (toks p') = none) :
    BF16.score R (transformer t codes) toks h p p' = 0 := by
  unfold BF16.score
  simp [key_none t codes R hr hG toks p' hc', fl_zero hr R]

omit t codes R in
lemma ideal_target (p h : ℕ) (hp : h ≤ p) :
    ∑ i ∈ range 8, attnScale * Real.cos (((p : ℝ) - h - ((p - h : ℕ) : ℝ)) * turn i) = 8 * attnScale := by
  rw [Nat.cast_sub hp]
  simp

omit t codes R in
lemma ideal_other (p h p' : ℕ) (hp : h ≤ p) (hh : h < 512) (hp' : p' ∈ visible p) (hne : p' ≠ p - h) :
    ∑ i ∈ range 8, attnScale * Real.cos (((p : ℝ) - h - p') * turn i) ≤ 8 * attnScale * runnerUp := by
  unfold visible at hp'
  rw [Finset.mem_filter, Finset.mem_range] at hp'
  unfold CONTEXT at hp'
  have hA := attnScale_ge
  -- the offset
  obtain ⟨e, he1, he2, hcos⟩ : ∃ e : ℕ, 1 ≤ e ∧ e ≤ 511 ∧
      ∀ i, Real.cos (((p : ℝ) - h - p') * turn i) = Real.cos (e * turn i) := by
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · refine ⟨p - h - p', by omega, by omega, fun i => ?_⟩
      congr 1
      rw [Nat.cast_sub (by omega), Nat.cast_sub hp]
    · refine ⟨p' + h - p, by omega, by omega, fun i => ?_⟩
      rw [← Real.cos_neg]
      congr 1
      rw [Nat.cast_sub (by omega)]
      push_cast; ring
  simp_rw [hcos]
  have hm := meanCos_le_runnerUp e he1 he2
  unfold meanCos PAIRS at hm
  push_cast at hm
  rw [← Finset.mul_sum]
  rw [div_le_iff₀ (by norm_num)] at hm
  nlinarith

/-! ## The heads -/

omit t codes R in
lemma visible_card (p : ℕ) : (visible p).card ≤ 512 := by
  have : visible p ⊆ (range 512).image (fun k => p - k) := by
    intro p' hp'
    unfold visible CONTEXT at hp'
    rw [Finset.mem_filter, Finset.mem_range] at hp'
    rw [Finset.mem_image]
    exact ⟨p - p', Finset.mem_range.mpr hp'.2, by omega⟩
  calc (visible p).card ≤ ((range 512).image (fun k => p - k)).card := Finset.card_le_card this
    _ ≤ (range 512).card := Finset.card_image_le
    _ = 512 := Finset.card_range 512

omit t codes R in
lemma target_visible (p h : ℕ) (hp : h ≤ p) (hh : h < 512) : p - h ∈ visible p := by
  unfold visible CONTEXT
  rw [Finset.mem_filter, Finset.mem_range]
  omega

lemma score_gap (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p : ℕ) {c ct : List Bool}
    (hc : codes.lookup (toks p) = some c) (h : ℕ) (hh : h < t.window) (hp : h ≤ p) (hh5 : h < 512)
    (hct : codes.lookup (toks (p - h)) = some ct) :
    ∀ p' ∈ visible p, p' ≠ p - h →
      BF16.score R (transformer t codes) toks h p p' + 14 ≤
        BF16.score R (transformer t codes) toks h p (p - h) := by
  intro p' hp' hne
  have ht := score_coded t codes R hr hn hprog hG hw toks p (p - h) hc hct h hh
  rw [ideal_target p h hp] at ht
  have ht' := (abs_le.mp ht).1
  have hgap := attnScale_gap
  have hA := attnScale_ge
  have hr0 := runnerUp_nonneg
  cases hlk : codes.lookup (toks p') with
  | none =>
    rw [score_none t codes R hr hG toks h p p' hlk]
    nlinarith
  | some c' =>
    have ho := score_coded t codes R hr hn hprog hG hw toks p p' hc hlk h hh
    have hi := ideal_other p h p' hp hh5 hp' hne
    have ho' := (abs_le.mp ho).2
    nlinarith

lemma head_eq (hr : Stmt_round) (hn : Stmt_norm_exact) (hprog : t.IsProgram)
    (hG : GridOK t (tokenGrid t)) (hw : WidthOK t) (toks : ℕ → ℕ) (p : ℕ) {c ct : List Bool}
    (hc : codes.lookup (toks p) = some c) (h : ℕ) (hh : h < t.window) (hp : h ≤ p) (hh5 : h < 512)
    (hct : codes.lookup (toks (p - h)) = some ct) (j : ℕ) (hj : j < t.outputs.length) :
    BF16.head R (transformer t codes) toks h p j = if ct.getD j false then 1 else -1 := by
  rw [← value_coded t codes R hr hn hprog hG hw toks (p - h) hct j hj]
  unfold BF16.head BF16.attn
  apply head_main hr R (visible p) (BF16.score R (transformer t codes) toks h p)
    (fun p' => BF16.value R (transformer t codes) toks p' j) (p - h)
    (target_visible p h hp hh5) (visible_card p)
    (score_gap t codes R hr hn hprog hG hw toks p hc h hh hp hh5 hct)
  · intro p' _
    cases hlk : codes.lookup (toks p') with
    | none => simp [value_none t codes R hr toks p' hlk]
    | some c' =>
      simp only [value_coded t codes R hr hn hprog hG hw toks p' hlk j hj]
      split_ifs <;> simp
  · simp only [value_coded t codes R hr hn hprog hG hw toks (p - h) hct j hj]
    split_ifs <;> simp

end

end AttnBF16Model
end Reifier

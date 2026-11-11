import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace LayerBF16

/-! ### Some bf16 numbers -/

lemma bf16_of_int (n : ℤ) (hn : |n| < 256) : IsBF16 (n : ℝ) :=
  ⟨n, 0, by simp, hn, by norm_num, by norm_num⟩

lemma bf16_zero : IsBF16 (0 : ℝ) := by simpa using bf16_of_int 0 (by norm_num)
lemma bf16_one : IsBF16 (1 : ℝ) := by simpa using bf16_of_int 1 (by norm_num)
lemma bf16_neg_one : IsBF16 (-1 : ℝ) := by simpa using bf16_of_int (-1) (by norm_num)
lemma bf16_two : IsBF16 (2 : ℝ) := by simpa using bf16_of_int 2 (by norm_num)
lemma bf16_neg_two : IsBF16 (-2 : ℝ) := by simpa using bf16_of_int (-2) (by norm_num)
lemma bf16_32 : IsBF16 (32 : ℝ) := by simpa using bf16_of_int 32 (by norm_num)
lemma bf16_inv32 : IsBF16 (1 / 32 : ℝ) :=
  ⟨1, -5, by norm_num, by norm_num, by norm_num, by norm_num⟩
lemma bf16_neg_inv32 : IsBF16 (-1 / 32 : ℝ) :=
  ⟨-1, -5, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-! ### The stream a row describes -/

lemma val_bit (ht : Stmt_trace) {t : Trace} {W : ℕ} {r : Row} {val : Name → ℝ}
    (hwf : t.WellFormed) (hrow : RowOK t W r) (hval : ValOK t val) {g : ℕ} {n : Name}
    (h : r g = some n) (hn : n ≠ .one) : IsBit (val n) := by
  rcases hrow.names g n h with h1 | ⟨i, rfl, _⟩ | ⟨s, rfl, hs | ⟨m, hm, hs⟩⟩
  · exact absurd h1 hn
  · exact hval.2 i
  · obtain ⟨x, hx⟩ := hval.1
    rw [hx s, (ht t (t.env x) hwf).1 s hs]
    unfold Trace.env; split_ifs <;> simp [IsBit]
  · obtain ⟨x, hx⟩ := hval.1
    rw [hx s]; exact hwf.bits x m hm s hs

lemma ideal_some {c : ℝ} {val : Name → ℝ} {r : Row} {g : ℕ} {n : Name}
    (h : r g = some n) (hn : n ≠ .one) : ideal c val r g = 2 * val n - 1 := by
  unfold ideal; rw [h]
  cases n with
  | one => exact absurd rfl hn
  | sig s => rfl
  | code i => rfl

lemma ideal_none {c : ℝ} {val : Name → ℝ} {r : Row} {g : ℕ}
    (h : r g = none) : ideal c val r g = 0 := by
  unfold ideal; rw [h]

lemma ideal_one {c : ℝ} {val : Name → ℝ} {r : Row} {g : ℕ}
    (h : r g = some .one) : ideal c val r g = c := by
  unfold ideal; rw [h]

lemma ideal_pm (ht : Stmt_trace) {t : Trace} {W : ℕ} {r : Row} {val : Name → ℝ}
    (hwf : t.WellFormed) (hrow : RowOK t W r) (hval : ValOK t val) {g : ℕ} {n : Name}
    (h : r g = some n) (hn : n ≠ .one) (c : ℝ) :
    ideal c val r g = 1 ∨ ideal c val r g = -1 := by
  have hb := val_bit ht hwf hrow hval h hn
  rw [ideal_some h hn]
  rcases hb with hb | hb <;> rw [hb] <;> norm_num

lemma ideal_bf16 (ht : Stmt_trace) {t : Trace} {W : ℕ} {r : Row} {val : Name → ℝ}
    (hwf : t.WellFormed) (hrow : RowOK t W r) (hval : ValOK t val) {c : ℝ} (hc : IsBF16 c)
    (g : ℕ) : IsBF16 (ideal c val r g) := by
  rcases hrg : r g with _ | n
  · rw [ideal_none hrg]; exact bf16_zero
  · by_cases hn : n = .one
    · subst hn; rw [ideal_one hrg]; exact hc
    · rcases ideal_pm ht hwf hrow hval hrg hn c with h | h <;> rw [h]
      exacts [bf16_one, bf16_neg_one]

/-- The norm, rounded, gives the row's stream with `ONE` as 1, exactly. -/
lemma norm_eq (hn : Stmt_norm_exact) (hr : Stmt_round) (ht : Stmt_trace) (R : Rounding)
    {t : Trace} {W : ℕ} {before : Row} {val : Name → ℝ} {one : ℝ}
    (hwf : t.WellFormed) (hrow : RowOK t W before) (hval : ValOK t val) (hW : 0 < W)
    (hone : 0 < one) :
    BF16.norm R W one before (ideal one val before) = ideal 1 val before := by
  funext g
  unfold BF16.norm
  rw [hn t W before val one hrow hval hwf hone hW g]
  exact hr.1 R _ (ideal_bf16 ht hwf hrow hval bf16_one g)

/-! ### A unit, in bf16 -/

/-- What a unit gives on the normed stream `z`, in bf16. -/
def psi (R : Rounding) (W : ℕ) (z : ℕ → ℝ) (u : Hidden) : ℝ :=
  R.fl (R.fl (gelu (R.fl (∑ g ∈ range W, R.fl (u.gate g) * z g))) *
    R.fl (∑ g ∈ range W, R.fl (u.val g) * z g))

lemma layerApply_eq (R : Rounding) (W : ℕ) (us : List Hidden) (z : ℕ → ℝ) (f : ℕ) :
    BF16.layerApply R W us z f = R.fl (unitSum (fun q => R.fl (q : ℝ)) (psi R W z) us f) := rfl

lemma sum_fl_eq_dot (hr : Stmt_round) (R : Rounding) (W : ℕ) (w : ℕ → ℚ)
    (hw : ∀ g, IsBF16 (w g : ℝ)) (z : ℕ → ℝ) :
    ∑ g ∈ range W, R.fl (w g : ℝ) * z g = dot W w z := by
  unfold dot; refine Finset.sum_congr rfl fun g _ => ?_; rw [hr.1 R _ (hw g)]

lemma psi_eq (hr : Stmt_round) (R : Rounding) (W : ℕ) (z : ℕ → ℝ) (u : Hidden)
    (hg : ∀ g, IsBF16 (u.gate g : ℝ)) (hv : ∀ g, IsBF16 (u.val g : ℝ)) {G V : ℝ}
    (hG : dot W u.gate z = G) (hV : dot W u.val z = V) (hGb : IsBF16 G) (hG16 : G = 0 ∨ 16 ≤ |G|)
    (hVb : IsBF16 V) (hp : IsBF16 (relu G * V)) :
    psi R W z u = relu G * V := by
  unfold psi
  rw [sum_fl_eq_dot hr R W _ hg, sum_fl_eq_dot hr R W _ hv, hG, hV, hr.1 R _ hGb]
  rcases hG16 with hG0 | hG16
  · subst hG0
    have hgelu : gelu 0 = 0 := by simp [gelu]
    have hrelu : relu 0 = 0 := by simp [relu]
    rw [hgelu, hr.1 R 0 bf16_zero, zero_mul, hr.1 R 0 bf16_zero, hrelu, zero_mul]
  · rw [hr.2.2.2.2.1 R G hGb hG16, hr.1 R _ hVb, hr.1 R _ hp]

/-- A gate at most 0 (and 0 or at most -16) passes 0, whatever its value. -/
lemma psi_eq_nonpos (hr : Stmt_round) (R : Rounding) (W : ℕ) (z : ℕ → ℝ) (u : Hidden)
    (hg : ∀ g, IsBF16 (u.gate g : ℝ)) {G : ℝ} (hG : dot W u.gate z = G)
    (hGm : -maxBF16 ≤ G) (hG0 : G ≤ 0) (hG16 : G = 0 ∨ 16 ≤ |G|) :
    psi R W z u = 0 := by
  unfold psi
  rw [sum_fl_eq_dot hr R W _ hg, hG]
  have hmax : (0 : ℝ) ≤ maxBF16 := by unfold maxBF16; norm_num
  have hGabs : |G| ≤ maxBF16 := abs_le.mpr ⟨hGm, by linarith⟩
  have hgelu0 : R.fl (gelu (R.fl G)) = 0 := by
    rcases hG16 with hG0' | hG16
    · subst hG0'
      have hgelu : gelu 0 = 0 := by simp [gelu]
      rw [hr.1 R 0 bf16_zero, hgelu, hr.1 R 0 bf16_zero]
    · have hGle : G ≤ -16 := by
        rw [abs_of_nonpos hG0] at hG16; linarith
      have hb16 : IsBF16 (-16 : ℝ) := by simpa using bf16_of_int (-16) (by norm_num)
      have hmono : R.fl G ≤ R.fl (-16) :=
        hr.2.2.2.2.2.1 R G (-16) hGabs (by unfold maxBF16; norm_num) hGle
      rw [hr.1 R _ hb16] at hmono
      have hflb : IsBF16 (R.fl G) := R.fl_mem G hGabs
      have h16 : 16 ≤ |R.fl G| := by rw [abs_of_neg (by linarith)]; linarith
      rw [hr.2.2.2.2.1 R _ hflb h16]
      unfold relu; exact max_eq_right (by linarith)
  rw [hgelu0, zero_mul, hr.1 R 0 bf16_zero]

lemma relu_32 : relu 32 = 32 := by unfold relu; norm_num

lemma oneUnit_gate_bf16 (W : ℕ) (before row : Row) (g : ℕ) :
    IsBF16 ((oneUnit W before row).gate g : ℝ) := by
  simp only [oneUnit]; split_ifs
  · push_cast; exact bf16_32
  · push_cast; exact bf16_zero

lemma oneUnit_val_bf16 (W : ℕ) (before row : Row) (g : ℕ) :
    IsBF16 ((oneUnit W before row).val g : ℝ) := by
  simp only [oneUnit]; split_ifs
  · push_cast; exact bf16_inv32
  · push_cast; exact bf16_zero

lemma eraseUnit_gate_bf16 (f g : ℕ) : IsBF16 ((eraseUnit f).gate g : ℝ) := by
  simp only [eraseUnit]; split_ifs
  · push_cast; exact bf16_32
  · push_cast; exact bf16_zero

lemma eraseUnit_val_bf16 (f g : ℕ) : IsBF16 ((eraseUnit f).val g : ℝ) := by
  simp only [eraseUnit]; split_ifs
  · push_cast; exact bf16_inv32
  · push_cast; exact bf16_zero

/-- The unit `ONE` gives 1. -/
lemma psi_oneUnit (hr : Stmt_round) (hu : Stmt_unit_dots) (R : Rounding) {W : ℕ} (hW : 0 < W)
    (before row : Row) (z : ℕ → ℝ) (hz0 : z 0 = 1) :
    psi R W z (oneUnit W before row) = 1 := by
  have h := hu.1 W before row z hW
  rw [psi_eq hr R W z _ (oneUnit_gate_bf16 W before row) (oneUnit_val_bf16 W before row)
    (G := 32) (V := 1 / 32) (by rw [h.1, hz0]; norm_num) (by rw [h.2, hz0]) bf16_32
    (Or.inr (by norm_num)) bf16_inv32 (by rw [relu_32]; norm_num; exact bf16_one)]
  rw [relu_32]; norm_num

/-- The erasure of feature `f` gives what `f` holds, `±1`. -/
lemma psi_eraseUnit (hr : Stmt_round) (hu : Stmt_unit_dots) (R : Rounding) {W f : ℕ}
    (hf1 : 1 ≤ f) (hfW : f < W) (z : ℕ → ℝ) (hz0 : z 0 = 1) (hzf : z f = 1 ∨ z f = -1) :
    psi R W z (eraseUnit f) = z f := by
  have h := hu.2.1 W f z hf1 hfW
  have hV : IsBF16 (z f / 32) := by
    rcases hzf with h' | h' <;> rw [h']
    · exact bf16_inv32
    · have : (-1 : ℝ) / 32 = -1 / 32 := rfl
      exact bf16_neg_inv32
  have hp : IsBF16 (relu 32 * (z f / 32)) := by
    rw [relu_32]
    rcases hzf with h' | h' <;> rw [h'] <;> norm_num
    · exact bf16_one
    · exact bf16_neg_one
  rw [psi_eq hr R W z _ (eraseUnit_gate_bf16 f) (eraseUnit_val_bf16 f)
    (G := 32) (V := z f / 32) (by rw [h.1, hz0]; norm_num) h.2 bf16_32
    (Or.inr (by norm_num)) hV hp]
  rw [relu_32]; ring

/-! ### The units of a new MLP -/

lemma featIn_spec {W : ℕ} {r : Row} {s : Name} (h : inRow W r s = true) :
    featIn W r s < W ∧ r (featIn W r s) = some s := by
  unfold inRow at h
  unfold featIn
  obtain ⟨a, ha⟩ : ∃ a, (List.range W).find? (fun f => r f == some s) = some a := by
    rw [← Option.isSome_iff_exists, List.find?_isSome]; simpa using h
  rw [ha, Option.getD_some]
  have h1 := List.find?_some ha
  have h2 := List.mem_of_find?_eq_some ha
  exact ⟨List.mem_range.mp h2, by simpa using h1⟩

lemma mlp_gate_bf16 (hu : Stmt_unit_dots) {t : Trace} {W : ℕ} {before row : Row} {m : MLP}
    {j : ℕ} (hwf : t.WellFormed) (hbf : t.BF16OK) (hpair : PairOK t W before row)
    (hm : m ∈ t.mlps) (hnew : mlpIsNew W before row m = true) (hj : j < m.compiled.n) (g : ℕ) :
    IsBF16 (((mlpUnits W before row m).getD j ⟨0, 0, 0⟩).gate g : ℝ) := by
  obtain ⟨-, h0, hs, hz⟩ := hu.2.2 t W before row m j hwf hpair hm hnew hj
  obtain ⟨hg, -, hgb, -, -⟩ := hbf.weights m hm j hj
  rcases Nat.eq_zero_or_pos g with rfl | hg1
  · rw [h0.1]; exact hgb
  · by_cases hb : ∃ s, before g = some (.sig s)
    · obtain ⟨s, hs'⟩ := hb; rw [(hs g s hs').1]; exact hg s
    · simp only [not_exists] at hb; rw [(hz g hg1 hb).1]; push_cast; exact bf16_zero

lemma mlp_val_bf16 (hu : Stmt_unit_dots) {t : Trace} {W : ℕ} {before row : Row} {m : MLP}
    {j : ℕ} (hwf : t.WellFormed) (hbf : t.BF16OK) (hpair : PairOK t W before row)
    (hm : m ∈ t.mlps) (hnew : mlpIsNew W before row m = true) (hj : j < m.compiled.n) (g : ℕ) :
    IsBF16 (((mlpUnits W before row m).getD j ⟨0, 0, 0⟩).val g : ℝ) := by
  obtain ⟨-, h0, hs, hz⟩ := hu.2.2 t W before row m j hwf hpair hm hnew hj
  obtain ⟨-, hv, -, hvb, -⟩ := hbf.weights m hm j hj
  rcases Nat.eq_zero_or_pos g with rfl | hg1
  · rw [h0.2]; exact hvb
  · by_cases hb : ∃ s, before g = some (.sig s)
    · obtain ⟨s, hs'⟩ := hb; rw [(hs g s hs').2]; exact hv s
    · simp only [not_exists] at hb; rw [(hz g hg1 hb).2]; push_cast; exact bf16_zero

/-- The normed stream at the feature of input `k` of a new MLP: its bit, as `±1`. -/
lemma z_featIn {t : Trace} {W : ℕ} {before row : Row} {val : Name → ℝ} {m : MLP}
    (hpair : PairOK t W before row) (hm : m ∈ t.mlps) (hnew : mlpIsNew W before row m = true)
    {xb : ℕ → Bool} (hxb : ∀ s, val (.sig s) = t.values (t.env xb) s) {k : ℕ}
    (hk : k < m.ins.length) :
    ideal 1 val before (featIn W before (.sig (m.ins.getD k 0))) = 2 * t.inVals xb m k - 1 := by
  have hx : m.ins.getD k 0 ∈ m.ins := by
    rw [List.getD_eq_getElem _ _ hk]; exact List.getElem_mem hk
  have hin := hpair.new_ins m hm hnew _ hx
  rw [ideal_some (featIn_spec hin).2 (by simp), hxb]
  rfl

lemma gateDot_eq (hc : Stmt_compiled) {t : Trace} {W : ℕ} {before row : Row} {val : Name → ℝ}
    {m : MLP} (hpair : PairOK t W before row) (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true)
    {xb : ℕ → Bool} (hxb : ∀ s, val (.sig s) = t.values (t.env xb) s) (j : ℕ) :
    gateDot W before m j (ideal 1 val before) = m.compiled.gateAt (t.inVals xb m) j := by
  unfold gateDot Gated.gateAt
  rw [(hc.2.2.1 m).2.1, ideal_one hpair.before_ok.one_zero, mul_one]
  congr 1
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [z_featIn hpair hm hnew hxb (Finset.mem_range.mp hk)]

lemma valDot_eq (hc : Stmt_compiled) {t : Trace} {W : ℕ} {before row : Row} {val : Name → ℝ}
    {m : MLP} (hpair : PairOK t W before row) (hm : m ∈ t.mlps)
    (hnew : mlpIsNew W before row m = true)
    {xb : ℕ → Bool} (hxb : ∀ s, val (.sig s) = t.values (t.env xb) s) (j : ℕ) :
    valDot W before m j (ideal 1 val before) = m.compiled.valAt (t.inVals xb m) j := by
  unfold valDot Gated.valAt
  rw [(hc.2.2.1 m).2.1, ideal_one hpair.before_ok.one_zero, mul_one]
  congr 1
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [z_featIn hpair hm hnew hxb (Finset.mem_range.mp hk)]

/-- The gate of unit `j` of the compiled MLP, on bits: 32 times the reglu form's. -/
lemma gateAt_eq (hc : Stmt_compiled) (m : MLP) (u : ℕ → ℝ) (j : ℕ) :
    m.compiled.gateAt u j =
      32 * (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j) := by
  have h := (hc.1 m 1 u j).1
  unfold Gated.gateAt
  rw [(hc.2.2.1 m).2.1]
  simpa using h

lemma valAt_eq (hc : Stmt_compiled) (m : MLP) (u : ℕ → ℝ) (j : ℕ) :
    m.compiled.valAt u j =
      1 / 32 * (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * u k + m.toReglu.vb j) := by
  have h := (hc.1 m 1 u j).2
  unfold Gated.valAt
  rw [(hc.2.2.1 m).2.1]
  simpa using h

lemma gateAt_ge (hc : Stmt_compiled) {t : Trace} {m : MLP} (hwf : t.WellFormed)
    (hm : m ∈ t.mlps) (xb : ℕ → Bool) {j : ℕ} (hj : j < m.compiled.n) :
    m.compiled.gateAt (t.inVals xb m) j = 0 ∨ 16 ≤ |m.compiled.gateAt (t.inVals xb m) j| := by
  have h := (hc.2.2.2 t xb m hwf hm).2.1 j (by rw [← (hc.2.2.1 m).1]; exact hj)
  rw [gateAt_eq hc, abs_mul]
  rcases h with h | h
  · left; rw [h, mul_zero]
  · right
    norm_num
    linarith

lemma relu_mul32 (P : ℝ) : relu (32 * P) = 32 * relu P := by
  unfold relu; rw [mul_max_of_nonneg _ _ (by norm_num : (0 : ℝ) ≤ 32), mul_zero]

/-- The units of a compiled MLP, weighted into output `i`, give twice its output. -/
lemma mlp_sum (hc : Stmt_compiled) {t : Trace} {m : MLP} (hwf : t.WellFormed)
    (hm : m ∈ t.mlps) (xb : ℕ → Bool) (i : ℕ) :
    ∑ j ∈ range m.compiled.n, (m.compiled.wo i j : ℝ) *
      (relu (m.compiled.gateAt (t.inVals xb m) j) * m.compiled.valAt (t.inVals xb m) j) =
      2 * m.out (t.inVals xb m) i := by
  rw [(hc.2.2.2 t xb m hwf hm).1 i, (hc.2.2.1 m).1, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [gateAt_eq hc, valAt_eq hc, hc.2.1 m i j, relu_mul32]
  push_cast
  ring

/-- Unit `j` of a new MLP gives `relu` of its gate times its value, exactly. -/
lemma psi_mlpUnit (hr : Stmt_round) (hu : Stmt_unit_dots) (hc : Stmt_compiled) (R : Rounding)
    {t : Trace} {W : ℕ} {before row : Row} {val : Name → ℝ} {m : MLP} {j : ℕ}
    (hwf : t.WellFormed) (hbf : t.BF16OK) (hpair : PairOK t W before row)
    (hm : m ∈ t.mlps) (hnew : mlpIsNew W before row m = true) (hj : j < m.compiled.n)
    {xb : ℕ → Bool} (hxb : ∀ s, val (.sig s) = t.values (t.env xb) s) :
    psi R W (ideal 1 val before) ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩) =
      relu (m.compiled.gateAt (t.inVals xb m) j) * m.compiled.valAt (t.inVals xb m) j := by
  obtain ⟨hdots, -⟩ := hu.2.2 t W before row m j hwf hpair hm hnew hj
  rcases hbf.sums xb m hm j hj with ⟨hlo, hhi⟩ | ⟨hGb, hVb, hp⟩
  · rw [psi_eq_nonpos hr R W _ _ (mlp_gate_bf16 hu hwf hbf hpair hm hnew hj)
      (by rw [(hdots _).1, gateDot_eq hc hpair hm hnew hxb]) hlo hhi (gateAt_ge hc hwf hm xb hj)]
    have hrelu : relu (m.compiled.gateAt (t.inVals xb m) j) = 0 := by
      unfold relu; exact max_eq_right hhi
    rw [hrelu, zero_mul]
  exact psi_eq hr R W _ _ (mlp_gate_bf16 hu hwf hbf hpair hm hnew hj)
    (mlp_val_bf16 hu hwf hbf hpair hm hnew hj)
    (by rw [(hdots _).1, gateDot_eq hc hpair hm hnew hxb])
    (by rw [(hdots _).2, valDot_eq hc hpair hm hnew hxb])
    hGb (gateAt_ge hc hwf hm xb hj) hVb hp

lemma bf16_bit (o : ℝ) (h : IsBit o) : IsBF16 (2 * o - 1) := by
  rcases h with h | h <;> rw [h] <;> norm_num
  · exact bf16_neg_one
  · exact bf16_one

end LayerBF16

open LayerBF16 in
theorem layer_bf16_of (hd : Stmt_decomp) (hu : Stmt_unit_dots) (hc : Stmt_compiled)
    (ht : Stmt_trace) (hn : Stmt_norm_exact) (hr : Stmt_round) : Stmt_layer_bf16 := by
  intro R t W before row val one hwf hbf hpair hval hW hone1 hone2 f
  have hone0 : (0 : ℝ) < one := by linarith
  obtain ⟨xb, hxb⟩ := hval.1
  have hnorm := norm_eq hn hr ht R hwf hpair.before_ok hval hW hone0
  unfold BF16.netStep
  rw [hnorm, layerApply_eq]
  set z := ideal 1 val before with hz_def
  set φ : ℚ → ℝ := fun q => R.fl (q : ℝ) with hφ
  have hφ0 : φ 0 = 0 := by simp only [hφ]; push_cast; exact hr.1 R 0 bf16_zero
  have hφm1 : φ (-1) = -1 := by simp only [hφ]; push_cast; exact hr.1 R (-1) bf16_neg_one
  have hdec := hd φ (psi R W z) t W before row hφ0 hwf hpair
  by_cases hf : row f = before f
  · rw [hdec.1 f hf, hr.1 R 0 bf16_zero, add_zero,
      hr.1 R _ (ideal_bf16 ht hwf hpair.before_ok hval hone1 f)]
    unfold ideal; rw [hf]
  · obtain ⟨m, hm, hnew, y, hy, hrowf⟩ := hpair.changed f hf
    have hi : m.outs.idxOf y < m.outs.length := List.idxOf_lt_length_of_mem hy
    have hyi : m.outs.getD (m.outs.idxOf y) 0 = y := by
      rw [List.getD_eq_getElem _ _ hi]; exact List.getElem_idxOf hi
    rw [hdec.2 f m (m.outs.idxOf y) hm hnew hi (by rw [hyi]; exact hrowf)]
    have hz0 : z 0 = 1 := ideal_one hpair.before_ok.one_zero
    rw [hφm1, psi_oneUnit hr hu R hW before row z hz0]
    have hmsum : ∑ j ∈ range m.compiled.n, φ (m.compiled.wo (m.outs.idxOf y) j) *
        psi R W z ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩) =
        2 * t.values (t.env xb) y := by
      have hout := (ht t (t.env xb) hwf).2 m hm _ hi
      rw [hyi] at hout
      rw [hout]
      show _ = 2 * m.out (t.inVals xb m) _
      rw [← mlp_sum hc hwf hm xb]
      refine Finset.sum_congr rfl fun j hj => ?_
      have hj' := Finset.mem_range.mp hj
      rw [psi_mlpUnit hr hu hc R hwf hbf hpair hm hnew hj' hxb]
      simp only [hφ]
      rw [hr.1 R _ ((hbf.weights m hm j hj').2.2.2.2 _ hi)]
    rw [hmsum]
    have hf0 : f ≠ 0 := by
      rintro rfl; rw [hpair.row_ok.one_zero] at hrowf; cases hrowf
    have hfW : f < W := by
      by_contra h; rw [hpair.row_ok.none_beyond f (by omega)] at hrowf; cases hrowf
    have hob : IsBit (t.values (t.env xb) y) := hwf.bits xb m hm y hy
    have hrhs : ideal one val row f = 2 * t.values (t.env xb) y - 1 := by
      rw [ideal_some hrowf (by simp), hxb]
    rw [hrhs]
    rcases hbf' : before f with _ | n
    · simp only [Option.isSome_none, Bool.false_eq_true, ite_false]
      rw [ideal_none hbf', add_zero, show (-1 : ℝ) * 1 + 2 * t.values (t.env xb) y =
        2 * t.values (t.env xb) y - 1 by ring, hr.1 R _ (bf16_bit _ hob), zero_add,
        hr.1 R _ (bf16_bit _ hob)]
    · have hn1 : n ≠ .one := by
        rintro rfl; exact hf0 (hpair.before_ok.one_only f hbf')
      simp only [Option.isSome_some, ite_true]
      have hs : z f = 1 ∨ z f = -1 := ideal_pm ht hwf hpair.before_ok hval hbf' hn1 1
      rw [psi_eraseUnit hr hu R (by omega) hfW z hz0 hs]
      have hx : ideal one val before f = z f := by
        rw [hz_def, ideal_some hbf' hn1, ideal_some hbf' hn1]
      rw [hx]
      have hin : IsBF16 (-1 * 1 + 2 * t.values (t.env xb) y + -1 * z f) := by
        rcases hob with ho | ho <;> rcases hs with hs | hs <;> rw [ho, hs] <;> norm_num
        exacts [bf16_neg_two, bf16_zero, bf16_zero, bf16_two]
      rw [hr.1 R _ hin, show z f + (-1 * 1 + 2 * t.values (t.env xb) y + -1 * z f) =
        2 * t.values (t.env xb) y - 1 by ring, hr.1 R _ (bf16_bit _ hob)]

end

end Reifier

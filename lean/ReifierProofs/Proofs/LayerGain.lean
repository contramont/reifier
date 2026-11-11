import ReifierProofs.Main

/-!
# One layer in real arithmetic, with gain (`Stmt_layer_gain`)

The norm scales the stream by `c` (`Stmt_norm_close`); a feature that keeps its name gets
nothing (`Stmt_decomp`); a feature that gets output `i` of a new MLP `m` gets `-c^2` from `ONE`,
`-c^2 x_f` from its erasure, and from the units of `m` (gates `32 c (P + δ)`, values
`c / 32 (Q + δ')`, `Stmt_compiled`) `2 c^2 (m.out u i + LIN + QUAD) + O(10^-11)`: the terms linear
in the errors `δ` cancel for a step (its two halves) and are `Σ_k (e_k / 2) slope_ik` otherwise,
at most `σ E / 2`; the quadratic ones are at most `gluGain i E^2 / 4 ≤ G E^2 / 4`.
-/

namespace Reifier
open Finset
noncomputable section

namespace LayerGainAux

def RP (m : MLP) (u : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j
def RQ (m : MLP) (u : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * u k + m.toReglu.vb j
def Rd (m : MLP) (d : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * d k
def Rv (m : MLP) (d : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * d k

/-- The terms of the units linear in the errors. -/
def LIN (m : MLP) (u d : ℕ → ℝ) (i : ℕ) : ℝ :=
  ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) *
    (if 0 < RP m u j then RP m u j * Rv m d j + Rd m d j * RQ m u j else 0)

/-- The terms of the units quadratic in the errors. -/
def QUAD (m : MLP) (u d : ℕ → ℝ) (i : ℕ) : ℝ :=
  ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) * (if 0 < RP m u j then Rd m d j * Rv m d j else 0)

lemma RP_add (m : MLP) (u d : ℕ → ℝ) (j : ℕ) :
    RP m (fun k => u k + d k) j = RP m u j + Rd m d j := by
  simp only [RP, Rd, mul_add, sum_add_distrib]; ring

lemma RQ_add (m : MLP) (u d : ℕ → ℝ) (j : ℕ) :
    RQ m (fun k => u k + d k) j = RQ m u j + Rv m d j := by
  simp only [RQ, Rv, mul_add, sum_add_distrib]; ring

/-! step facts -/
section step
variable {m : MLP} (hs : m.act = .step)
include hs

lemma st_n : m.toReglu.n = 2 * m.hidden := by simp [MLP.toReglu, hs]
lemma st_v (j k : ℕ) : m.toReglu.v j k = 0 := by simp [MLP.toReglu, hs]
lemma st_vb (j : ℕ) : m.toReglu.vb j = 1 := by simp [MLP.toReglu, hs]
lemma st_g (j k : ℕ) : m.toReglu.g j k = 4 * m.wi (if j < m.hidden then j else j - m.hidden) k := by
  simp [MLP.toReglu, hs]
lemma st_wo (i j : ℕ) : m.toReglu.wo i j = if j < m.hidden then m.wo i j else -m.wo i (j - m.hidden) := by
  simp [MLP.toReglu, hs]
lemma st_gb (j : ℕ) : m.toReglu.gb j =
    if j < m.hidden then 4 * m.b j + 2 + 1 / 2 else 4 * m.b (j - m.hidden) + 2 - 1 / 2 := by
  simp [MLP.toReglu, hs]

lemma st_Rv (d : ℕ → ℝ) (j : ℕ) : Rv m d j = 0 := by simp [Rv, st_v hs]
lemma st_RQ (u : ℕ → ℝ) (j : ℕ) : RQ m u j = 1 := by simp [RQ, st_v hs, st_vb hs]
lemma st_Rd_hi (d : ℕ → ℝ) (j : ℕ) (hj : j < m.hidden) : Rd m d (m.hidden + j) = Rd m d j := by
  simp only [Rd, st_g hs, hj, ite_true, show ¬ (m.hidden + j < m.hidden) by omega, ite_false,
    Nat.add_sub_cancel_left]
lemma st_RP_hi (u : ℕ → ℝ) (j : ℕ) (hj : j < m.hidden) : RP m u (m.hidden + j) = RP m u j - 1 := by
  simp only [RP, st_g hs, st_gb hs, hj, ite_true, show ¬ (m.hidden + j < m.hidden) by omega, ite_false,
    Nat.add_sub_cancel_left]
  push_cast; ring
lemma st_wo_hi (i j : ℕ) (hj : j < m.hidden) : m.toReglu.wo i (m.hidden + j) = - m.toReglu.wo i j := by
  simp only [st_wo hs, hj, ite_true, show ¬ (m.hidden + j < m.hidden) by omega, ite_false,
    Nat.add_sub_cancel_left]
end step

section relu
variable {m : MLP} (hs : m.act = .relu)
include hs
lemma re_n : m.toReglu.n = m.hidden := by simp [MLP.toReglu, hs]
lemma re_wo : m.toReglu.wo = m.wo := by simp [MLP.toReglu, hs]
lemma re_Rv (d : ℕ → ℝ) (j : ℕ) : Rv m d j = 0 := by simp [Rv, MLP.toReglu, hs]
lemma re_RQ (u : ℕ → ℝ) (j : ℕ) : RQ m u j = 1 := by simp [RQ, MLP.toReglu, hs]
lemma re_RP (u : ℕ → ℝ) (j : ℕ) : RP m u j = m.pre u j := by simp [RP, MLP.pre, MLP.toReglu, hs]
lemma re_Rd (d : ℕ → ℝ) (j : ℕ) : Rd m d j = ∑ k ∈ range m.ins.length, (m.wi j k : ℝ) * d k := by
  simp [Rd, MLP.toReglu, hs]
lemma re_slope (u : ℕ → ℝ) (i k : ℕ) : m.slope u i k =
    ∑ j ∈ range m.hidden, if 0 < m.pre u j then (m.wo i j : ℝ) * m.wi j k else 0 := by
  simp [MLP.slope, hs]
end relu

section reglu
variable {m : MLP} (hs : m.act = .reglu)
include hs
lemma rg_n : m.toReglu.n = m.hidden := by simp [MLP.toReglu, hs]
lemma rg_wo : m.toReglu.wo = m.wo := by simp [MLP.toReglu, hs]
lemma rg_Rv (d : ℕ → ℝ) (j : ℕ) : Rv m d j = ∑ k ∈ range m.ins.length, (m.wi (m.hidden + j) k : ℝ) * d k := by
  simp [Rv, MLP.toReglu, hs]
lemma rg_RQ (u : ℕ → ℝ) (j : ℕ) : RQ m u j = m.pre u (m.hidden + j) := by
  simp [RQ, MLP.pre, MLP.toReglu, hs]
lemma rg_RP (u : ℕ → ℝ) (j : ℕ) : RP m u j = m.pre u j := by simp [RP, MLP.pre, MLP.toReglu, hs]
lemma rg_Rd (d : ℕ → ℝ) (j : ℕ) : Rd m d j = ∑ k ∈ range m.ins.length, (m.wi j k : ℝ) * d k := by
  simp [Rd, MLP.toReglu, hs]
lemma rg_slope (u : ℕ → ℝ) (i k : ℕ) : m.slope u i k =
    ∑ j ∈ range m.hidden, if 0 < m.pre u j then (m.wo i j : ℝ) *
      ((m.wi j k : ℝ) * m.pre u (m.hidden + j) + m.pre u j * m.wi (m.hidden + j) k) else 0 := by
  simp [MLP.slope, hs]
end reglu

lemma n_ne_step (m : MLP) (hs : m.act ≠ .step) : m.toReglu.n = m.hidden := by
  rcases h : m.act with _ | _ | _
  · exact absurd h hs
  · exact re_n h
  · exact rg_n h

/-! ## The linear terms -/

lemma lin_step (m : MLP) (hs : m.act = .step) (u d : ℕ → ℝ) (i : ℕ)
    (hside : ∀ j < m.hidden, (0 < RP m u j ↔ 0 < RP m u (m.hidden + j))) : LIN m u d i = 0 := by
  unfold LIN
  rw [st_n hs, two_mul, sum_range_add, ← sum_add_distrib]
  apply sum_eq_zero
  intro j hj
  rw [mem_range] at hj
  have hiff := hside j hj
  rw [st_RP_hi hs u j hj] at hiff
  rw [st_RP_hi hs u j hj, st_Rd_hi hs d j hj, st_wo_hi hs i j hj]
  simp only [st_Rv hs, st_RQ hs]
  by_cases hp : 0 < RP m u j
  · rw [ite_eq_left hp, ite_eq_left (hiff.mp hp)]; push_cast; ring
  · rw [ite_eq_right hp, ite_eq_right (fun h => hp (hiff.mpr h))]; ring

lemma lin_relu (m : MLP) (hs : m.act = .relu) (u d : ℕ → ℝ) (i : ℕ) :
    LIN m u d i = ∑ k ∈ range m.ins.length, d k * m.slope u i k := by
  unfold LIN
  simp only [re_slope hs, mul_sum]
  rw [sum_comm]
  rw [re_n hs]
  apply sum_congr rfl
  intro j _
  rw [re_wo hs, re_RP hs, re_Rv hs, re_RQ hs, re_Rd hs]
  by_cases hp : 0 < m.pre u j
  · simp only [ite_eq_left hp, mul_zero, zero_add, mul_one, mul_sum]
    apply sum_congr rfl; intro k _; ring
  · simp only [ite_eq_right hp, mul_zero, sum_const_zero]

lemma lin_reglu (m : MLP) (hs : m.act = .reglu) (u d : ℕ → ℝ) (i : ℕ) :
    LIN m u d i = ∑ k ∈ range m.ins.length, d k * m.slope u i k := by
  unfold LIN
  simp only [rg_slope hs, mul_sum]
  rw [sum_comm]
  rw [rg_n hs]
  apply sum_congr rfl
  intro j _
  rw [rg_wo hs, rg_RP hs, rg_Rv hs, rg_RQ hs, rg_Rd hs]
  by_cases hp : 0 < m.pre u j
  · simp only [ite_eq_left hp, mul_sum, sum_mul, ← sum_add_distrib]
    apply sum_congr rfl; intro k _; ring
  · simp only [ite_eq_right hp, mul_zero, sum_const_zero]

lemma abs_sum_mul_le' (s : Finset ℕ) (a b : ℕ → ℝ) (M : ℝ)
    (ha : ∀ k ∈ s, |a k| ≤ M) : |∑ k ∈ s, a k * b k| ≤ M * ∑ k ∈ s, |b k| := by
  calc |∑ k ∈ s, a k * b k| ≤ ∑ k ∈ s, |a k * b k| := abs_sum_le_sum_abs _ _
    _ = ∑ k ∈ s, |a k| * |b k| := by simp_rw [abs_mul]
    _ ≤ ∑ k ∈ s, M * |b k| :=
        sum_le_sum fun k hk => mul_le_mul_of_nonneg_right (ha k hk) (abs_nonneg _)
    _ = M * ∑ k ∈ s, |b k| := by rw [mul_sum]

lemma lin_le (m : MLP) (u d : ℕ → ℝ) (i : ℕ) (E σ : ℝ) (hE : 0 ≤ E) (hσ : 0 ≤ σ)
    (hd : ∀ k < m.ins.length, |d k| ≤ E / 2)
    (hside : m.act = .step → ∀ j < m.hidden, (0 < RP m u j ↔ 0 < RP m u (m.hidden + j)))
    (hsl : m.act ≠ .step → m.slopeSum u i ≤ σ) :
    |LIN m u d i| ≤ σ * E / 2 := by
  have gen : ∑ k ∈ range m.ins.length, d k * m.slope u i k = LIN m u d i →
      m.act ≠ .step → |LIN m u d i| ≤ σ * E / 2 := by
    intro h hs
    rw [← h]
    have h1 := abs_sum_mul_le' (range m.ins.length) d (fun k => m.slope u i k) (E / 2)
      (fun k hk => hd k (mem_range.mp hk))
    have h2 := hsl hs
    unfold MLP.slopeSum at h2
    calc _ ≤ _ := h1
      _ ≤ E / 2 * σ := mul_le_mul_of_nonneg_left h2 (by positivity)
      _ = σ * E / 2 := by ring
  rcases hs : m.act with _ | _ | _
  · rw [lin_step m hs u d i (hside hs), abs_zero]; positivity
  · exact gen (lin_relu m hs u d i).symm (by rw [hs]; simp)
  · exact gen (lin_reglu m hs u d i).symm (by rw [hs]; simp)

/-! ## The quadratic terms -/

lemma quad_le (m : MLP) (u d : ℕ → ℝ) (i : ℕ) (E G : ℝ) (hG : 0 ≤ G)
    (hd : ∀ k < m.ins.length, |d k| ≤ E / 2)
    (hgg : m.act = .reglu → (m.gluGain i : ℝ) ≤ G) :
    |QUAD m u d i| ≤ G * E ^ 2 / 4 := by
  unfold QUAD
  rcases hs : m.act with _ | _ | _
  · simp only [st_Rv hs, mul_zero, ite_self, sum_const_zero, abs_zero]; positivity
  · simp only [re_Rv hs, mul_zero, ite_self, sum_const_zero, abs_zero]; positivity
  · rw [rg_n hs, rg_wo hs]
    have hA : ∀ j, |Rd m d j| ≤ E / 2 * ∑ k ∈ range m.ins.length, |(m.wi j k : ℝ)| := by
      intro j
      rw [rg_Rd hs]
      have := abs_sum_mul_le' (range m.ins.length) d (fun k => (m.wi j k : ℝ)) (E / 2)
        (fun k hk => hd k (mem_range.mp hk))
      simpa [mul_comm] using this
    have hB : ∀ j, |Rv m d j| ≤ E / 2 * ∑ k ∈ range m.ins.length, |(m.wi (m.hidden + j) k : ℝ)| := by
      intro j
      rw [rg_Rv hs]
      have := abs_sum_mul_le' (range m.ins.length) d (fun k => (m.wi (m.hidden + j) k : ℝ)) (E / 2)
        (fun k hk => hd k (mem_range.mp hk))
      simpa [mul_comm] using this
    have hgain : (m.gluGain i : ℝ) = ∑ j ∈ range m.hidden, |(m.wo i j : ℝ)| *
        (∑ k ∈ range m.ins.length, |(m.wi j k : ℝ)|) *
        (∑ k ∈ range m.ins.length, |(m.wi (m.hidden + j) k : ℝ)|) := by
      unfold MLP.gluGain; push_cast; rfl
    calc _ ≤ ∑ j ∈ range m.hidden, |(m.wo i j : ℝ) *
          (if 0 < RP m u j then Rd m d j * Rv m d j else 0)| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ range m.hidden, |(m.wo i j : ℝ)| *
          ((E / 2 * ∑ k ∈ range m.ins.length, |(m.wi j k : ℝ)|) *
            (E / 2 * ∑ k ∈ range m.ins.length, |(m.wi (m.hidden + j) k : ℝ)|)) := by
          apply sum_le_sum
          intro j _
          rw [abs_mul]
          apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
          split_ifs
          · rw [abs_mul]; exact mul_le_mul (hA j) (hB j) (abs_nonneg _) ((abs_nonneg _).trans (hA j))
          · rw [abs_zero]; exact mul_nonneg ((abs_nonneg _).trans (hA j)) ((abs_nonneg _).trans (hB j))
      _ = E ^ 2 / 4 * (m.gluGain i : ℝ) := by
          rw [hgain, mul_sum]; apply sum_congr rfl; intro j _; ring
      _ ≤ E ^ 2 / 4 * G := mul_le_mul_of_nonneg_left (hgg hs) (by positivity)
      _ = G * E ^ 2 / 4 := by ring

/-! ## Sizes -/

lemma cast_sum_abs (s : Finset ℕ) (f : ℕ → ℚ) :
    ∑ k ∈ s, |(f k : ℝ)| = ((∑ k ∈ s, |f k| : ℚ) : ℝ) := by
  push_cast; rfl

lemma size_g (m : MLP) (hB : m.Bounded (2 ^ 16)) (j : ℕ) (hj : j < m.toReglu.n) :
    ∑ k ∈ range m.ins.length, |(m.toReglu.g j k : ℝ)| ≤ 4 * 2 ^ 16 := by
  rw [cast_sum_abs]
  obtain ⟨h1, -, -⟩ := hB
  rcases hs : m.act with _ | _ | _
  · rw [st_n hs] at hj
    simp only [st_g hs]
    have hj' : (if j < m.hidden then j else j - m.hidden) < m.rows := by
      simp only [MLP.rows, hs]; split_ifs <;> omega
    have := h1 _ hj'
    have e : ∑ k ∈ range m.ins.length, |4 * m.wi (if j < m.hidden then j else j - m.hidden) k| =
        4 * ∑ k ∈ range m.ins.length, |m.wi (if j < m.hidden then j else j - m.hidden) k| := by
      rw [mul_sum]; apply sum_congr rfl; intro k _; rw [abs_mul]; norm_num
    exact_mod_cast (show _ ≤ (4 * 2 ^ 16 : ℚ) by rw [e]; linarith)
  · rw [re_n hs] at hj
    have := h1 j (by simp only [MLP.rows, hs]; simpa using hj)
    have e : m.toReglu.g = m.wi := by simp [MLP.toReglu, hs]
    rw [e]; exact_mod_cast (show _ ≤ (4 * 2 ^ 16 : ℚ) by linarith)
  · rw [rg_n hs] at hj
    have := h1 j (by simp only [MLP.rows, hs]; simp; omega)
    have e : m.toReglu.g = m.wi := by simp [MLP.toReglu, hs]
    rw [e]; exact_mod_cast (show _ ≤ (4 * 2 ^ 16 : ℚ) by linarith)

lemma size_v (m : MLP) (hB : m.Bounded (2 ^ 16)) (j : ℕ) (hj : j < m.toReglu.n) :
    ∑ k ∈ range m.ins.length, |(m.toReglu.v j k : ℝ)| ≤ 2 ^ 16 ∧
      |(m.toReglu.vb j : ℝ)| ≤ 2 ^ 16 := by
  obtain ⟨h1, h2, -⟩ := hB
  rcases hs : m.act with _ | _ | _
  · simp [st_v hs, st_vb hs]; norm_num
  · have e1 : m.toReglu.v = fun _ _ => 0 := by simp [MLP.toReglu, hs]
    have e2 : m.toReglu.vb = fun _ => 1 := by simp [MLP.toReglu, hs]
    simp [e1, e2]; norm_num
  · rw [rg_n hs] at hj
    have hr : m.hidden + j < m.rows := by simp only [MLP.rows, hs]; simp; omega
    have e1 : m.toReglu.v = fun j k => m.wi (m.hidden + j) k := by simp [MLP.toReglu, hs]
    have e2 : m.toReglu.vb = fun j => m.b (m.hidden + j) := by simp [MLP.toReglu, hs]
    rw [e1, e2, cast_sum_abs]
    have h1' := h1 _ hr
    constructor
    · exact_mod_cast h1'
    · rw [← Rat.cast_abs]; exact_mod_cast h2 _ hr

lemma size_wo (m : MLP) (hB : m.Bounded (2 ^ 16)) (i : ℕ) (hi : i < m.outs.length) :
    ∑ j ∈ range m.toReglu.n, |(m.toReglu.wo i j : ℝ)| ≤ 2 * 2 ^ 16 := by
  obtain ⟨-, -, h3⟩ := hB
  have := h3 i hi
  rw [cast_sum_abs]
  rcases hs : m.act with _ | _ | _
  · rw [st_n hs, two_mul, sum_range_add]
    have e1 : ∑ j ∈ range m.hidden, |m.toReglu.wo i j| = ∑ j ∈ range m.hidden, |m.wo i j| := by
      apply sum_congr rfl; intro j hj; rw [st_wo hs, ite_eq_left (mem_range.mp hj)]
    have e2 : ∑ j ∈ range m.hidden, |m.toReglu.wo i (m.hidden + j)| =
        ∑ j ∈ range m.hidden, |m.wo i j| := by
      apply sum_congr rfl; intro j hj
      rw [st_wo_hi hs i j (mem_range.mp hj), abs_neg, st_wo hs, ite_eq_left (mem_range.mp hj)]
    rw [e1, e2]; exact_mod_cast (show _ ≤ (2 * 2 ^ 16 : ℚ) by linarith)
  · rw [re_n hs, re_wo hs]; exact_mod_cast (show _ ≤ (2 * 2 ^ 16 : ℚ) by linarith)
  · rw [rg_n hs, rg_wo hs]; exact_mod_cast (show _ ≤ (2 * 2 ^ 16 : ℚ) by linarith)

lemma abs_sum_mul_le (s : Finset ℕ) (a b : ℕ → ℝ) (A M : ℝ) (hM : 0 ≤ M)
    (hb : ∀ j ∈ s, |b j| ≤ M) (hA : ∑ j ∈ s, |a j| ≤ A) : |∑ j ∈ s, a j * b j| ≤ A * M := by
  calc |∑ j ∈ s, a j * b j| ≤ ∑ j ∈ s, |a j * b j| := abs_sum_le_sum_abs _ _
    _ = ∑ j ∈ s, |a j| * |b j| := by simp_rw [abs_mul]
    _ ≤ ∑ j ∈ s, |a j| * M :=
        sum_le_sum fun j hj => mul_le_mul_of_nonneg_left (hb j hj) (abs_nonneg _)
    _ = (∑ j ∈ s, |a j|) * M := by rw [sum_mul]
    _ ≤ A * M := mul_le_mul_of_nonneg_right hA hM

lemma Rd_le (m : MLP) (hB : m.Bounded (2 ^ 16)) (D : ℝ) (hD : 0 ≤ D) (d : ℕ → ℝ)
    (hd : ∀ k < m.ins.length, |d k| ≤ D)
    (j : ℕ) (hj : j < m.toReglu.n) : |Rd m d j| ≤ 4 * 2 ^ 16 * D := by
  unfold Rd
  exact abs_sum_mul_le _ _ _ _ _ hD (fun k hk => hd k (mem_range.mp hk)) (size_g m hB j hj)

lemma Rv_le (m : MLP) (hB : m.Bounded (2 ^ 16)) (D : ℝ) (hD : 0 ≤ D) (d : ℕ → ℝ)
    (hd : ∀ k < m.ins.length, |d k| ≤ D)
    (j : ℕ) (hj : j < m.toReglu.n) : |Rv m d j| ≤ 2 ^ 16 * D := by
  unfold Rv
  exact abs_sum_mul_le _ _ _ _ _ hD (fun k hk => hd k (mem_range.mp hk)) (size_v m hB j hj).1

lemma RQ_le (m : MLP) (hB : m.Bounded (2 ^ 16)) (u : ℕ → ℝ) (hu : ∀ k < m.ins.length, |u k| ≤ 1)
    (j : ℕ) (hj : j < m.toReglu.n) : |RQ m u j| ≤ 2 * 2 ^ 16 := by
  unfold RQ
  have h1 := abs_sum_mul_le (range m.ins.length) (fun k => (m.toReglu.v j k : ℝ)) u _ _ zero_le_one
    (fun k hk => hu k (mem_range.mp hk)) (size_v m hB j hj).1
  have h2 := (size_v m hB j hj).2
  calc _ ≤ _ := abs_add_le _ _
    _ ≤ 2 ^ 16 * 1 + 2 ^ 16 := add_le_add h1 h2
    _ = 2 * 2 ^ 16 := by norm_num

/-! ## One output of a new MLP -/

lemma relu_G (c P δ : ℝ) (hc : 0 < c) (hP : 1/2 ≤ |P|) (hδ : |δ| ≤ 1/4) :
    relu (32 * c * (P + δ)) = 32 * c * (if 0 < P then P + δ else 0) := by
  have hδ' := abs_le.mp hδ
  unfold relu
  by_cases hp : 0 < P
  · rw [abs_of_pos hp] at hP
    rw [ite_eq_left hp]
    apply max_eq_left
    have : 0 < P + δ := by linarith
    positivity
  · rw [abs_of_nonpos (not_lt.mp hp)] at hP
    rw [ite_eq_right hp, mul_zero]
    apply max_eq_right
    have : P + δ < 0 := by linarith
    have : 0 < 32 * c := by positivity
    nlinarith

lemma c_bounds (c E : ℝ) (hc0 : 0 < c) (hE : E ≤ 1 / 10 ^ 6) (hc2 : |c^2 - 1| ≤ 41/100 * E) :
    99/100 ≤ c ∧ c ≤ 10001/10000 ∧ c^2 ≤ 1 + 41/100 * E := by
  have h := abs_le.mp hc2
  refine ⟨?_, ?_, ?_⟩
  · nlinarith
  · nlinarith
  · linarith

lemma core (hg : Stmt_gelu) (m : MLP) (u d : ℕ → ℝ) (c E L Q : ℝ) (i : ℕ)
    (hB : m.Bounded (2 ^ 16)) (hi : i < m.outs.length)
    (hout : m.out u i = ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) * (relu (RP m u j) * RQ m u j))
    (hP12 : ∀ j < m.toReglu.n, 1/2 ≤ |RP m u j|)
    (hlin : |LIN m u d i| ≤ L) (hquad : |QUAD m u d i| ≤ Q)
    (hu : ∀ k < m.ins.length, |u k| ≤ 1)
    (hE0 : 0 ≤ E) (hE6 : E ≤ 1 / 10 ^ 6)
    (hd : ∀ k < m.ins.length, |d k| ≤ E / 2)
    (hc0 : 0 < c) (hc2 : |c^2 - 1| ≤ 41/100 * E) :
    |∑ j ∈ range m.toReglu.n, (2 * (m.toReglu.wo i j : ℝ)) *
        (gelu (32 * c * RP m (fun k => u k + d k) j) * (c / 32 * RQ m (fun k => u k + d k) j))
      - 2 * c^2 * m.out u i| ≤ 2 * c ^ 2 * (L + Q) + 11 / 10 ^ 12 := by
  obtain ⟨hcl, hcu, -⟩ := c_bounds c E hc0 hE6 hc2
  have hD0 : 0 ≤ E / 2 := by positivity
  have hD : E / 2 ≤ 1 / (2 * 10 ^ 6) := by linarith
  set n := m.toReglu.n
  set G : ℕ → ℝ := fun j => 32 * c * (RP m u j + Rd m d j) with hG
  set V : ℕ → ℝ := fun j => c / 32 * (RQ m u j + Rv m d j) with hV
  set H := ∑ j ∈ range n, (2 * (m.toReglu.wo i j : ℝ)) * ((gelu (G j) - relu (G j)) * V j) with hH
  have hsum : ∑ j ∈ range n, (2 * (m.toReglu.wo i j : ℝ)) *
        (gelu (32 * c * RP m (fun k => u k + d k) j) * (c / 32 * RQ m (fun k => u k + d k) j)) =
      2 * c^2 * (m.out u i + LIN m u d i + QUAD m u d i) + H := by
    have e : ∀ j ∈ range n, (2 * (m.toReglu.wo i j : ℝ)) *
        (gelu (32 * c * RP m (fun k => u k + d k) j) * (c / 32 * RQ m (fun k => u k + d k) j)) =
        2 * c^2 * ((m.toReglu.wo i j : ℝ) * (relu (RP m u j) * RQ m u j) +
          (m.toReglu.wo i j : ℝ) * (if 0 < RP m u j then RP m u j * Rv m d j + Rd m d j * RQ m u j else 0) +
          (m.toReglu.wo i j : ℝ) * (if 0 < RP m u j then Rd m d j * Rv m d j else 0)) +
        (2 * (m.toReglu.wo i j : ℝ)) * ((gelu (G j) - relu (G j)) * V j) := by
      intro j hj
      have hj' := mem_range.mp hj
      rw [RP_add, RQ_add]
      have hδ : |Rd m d j| ≤ 1/4 := by
        have := Rd_le m hB (E / 2) hD0 d hd j hj'
        have : 4 * 2 ^ 16 * (E / 2) ≤ (4 * 2 ^ 16 : ℝ) * (1 / (2 * 10 ^ 6)) := by gcongr
        linarith
      have hr := relu_G c (RP m u j) (Rd m d j) hc0 (hP12 j hj') hδ
      simp only [hG, hV]
      have : gelu (32 * c * (RP m u j + Rd m d j)) = relu (32 * c * (RP m u j + Rd m d j)) +
          (gelu (32 * c * (RP m u j + Rd m d j)) - relu (32 * c * (RP m u j + Rd m d j))) := by ring
      rw [this, hr]
      by_cases hp : 0 < RP m u j
      · simp only [ite_eq_left hp, relu, max_eq_left hp.le]; ring
      · simp only [ite_eq_right hp, relu, max_eq_right (not_lt.mp hp)]; ring
    rw [sum_congr rfl e, sum_add_distrib, ← mul_sum, sum_add_distrib, sum_add_distrib, ← hout]
    rfl
  rw [hsum]
  have hHb : |H| ≤ (2 * (2 * 2 ^ 16)) * (1 / 10^20 * 4097) := by
    apply abs_sum_mul_le _ _ _ _ _ (by positivity)
    · intro j hj
      have hj' := mem_range.mp hj
      rw [abs_mul]
      have hP := hP12 j hj'
      have hδ := Rd_le m hB (E / 2) hD0 d hd j hj'
      have hδ' : 4 * 2 ^ 16 * (E / 2) ≤ (4 * 2 ^ 16 : ℝ) * (1 / (2 * 10 ^ 6)) := by gcongr
      have hGb : 8 ≤ |G j| := by
        simp only [hG]
        rw [abs_mul, abs_of_pos (by positivity : (0:ℝ) < 32 * c)]
        have : 1/2 - 4 * 2 ^ 16 * (E / 2) ≤ |RP m u j + Rd m d j| := by
          have := abs_sub_abs_le_abs_sub (RP m u j) (-Rd m d j)
          rw [abs_neg, sub_neg_eq_add] at this
          linarith
        nlinarith
      have hVb : |V j| ≤ 4097 := by
        simp only [hV]
        rw [abs_mul, abs_of_pos (by positivity : (0:ℝ) < c / 32)]
        have h1 := RQ_le m hB u hu j hj'
        have h2 := Rv_le m hB (E / 2) hD0 d hd j hj'
        have h2' : 2 ^ 16 * (E / 2) ≤ (2 ^ 16 : ℝ) * (1 / (2 * 10 ^ 6)) := by gcongr
        have : |RQ m u j + Rv m d j| ≤ 2 * 2 ^ 16 + 2 ^ 16 * (1 / (2 * 10 ^ 6)) :=
          (abs_add_le _ _).trans (by linarith)
        have : 0 ≤ |RQ m u j + Rv m d j| := abs_nonneg _
        nlinarith
      exact mul_le_mul (hg _ hGb) hVb (abs_nonneg _) (by positivity)
    · have := size_wo m hB i hi
      calc ∑ j ∈ range n, |2 * (m.toReglu.wo i j : ℝ)| =
            2 * ∑ j ∈ range n, |(m.toReglu.wo i j : ℝ)| := by
            rw [mul_sum]; apply sum_congr rfl; intro j _; rw [abs_mul]; norm_num
        _ ≤ 2 * (2 * 2 ^ 16) := by linarith
  have e2 : 2 * c ^ 2 * (m.out u i + LIN m u d i + QUAD m u d i) + H - 2 * c ^ 2 * m.out u i =
      2 * c^2 * (LIN m u d i + QUAD m u d i) + H := by ring
  rw [e2]
  calc |2 * c^2 * (LIN m u d i + QUAD m u d i) + H|
      ≤ |2 * c^2 * (LIN m u d i + QUAD m u d i)| + |H| := abs_add_le _ _
    _ = 2 * c^2 * |LIN m u d i + QUAD m u d i| + |H| := by rw [abs_mul, abs_of_pos (by positivity)]
    _ ≤ 2 * c^2 * (L + Q) + (2 * (2 * 2 ^ 16)) * (1 / 10^20 * 4097) := by
        gcongr
        exact (abs_add_le _ _).trans (add_le_add hlin hquad)
    _ ≤ 2 * c ^ 2 * (L + Q) + 11 / 10 ^ 12 := by norm_num

/-- The arithmetic of the final bound. -/
lemma budget (c E σ G : ℝ) (hE0 : 1 / 10 ^ 10 ≤ E) (hE6 : E ≤ 1 / 10 ^ 6)
    (hσ : 0 ≤ σ) (hσE : σ * E ≤ 1 / 100) (hG : 0 ≤ G) (hGE : G * E ≤ 1 / 10)
    (hc2u : c ^ 2 ≤ 1 + 41 / 100 * E) (R : ℝ) (hR : R ≤ 41 / 100 * E * (2 + E)) :
    R + (2 * c ^ 2 * (σ * E / 2 + G * E ^ 2 / 4) + 11 / 10 ^ 12) + 1 / 10 ^ 20 ≤ (1 + σ) * E := by
  have hE : 0 ≤ E := le_trans (by norm_num) hE0
  have hσE0 : 0 ≤ σ * E := mul_nonneg hσ hE
  have hGE0 : 0 ≤ G * E := mul_nonneg hG hE
  have hX0 : 0 ≤ σ * E + G * E * E / 2 := by positivity
  have h1 : 2 * c ^ 2 * (σ * E / 2 + G * E ^ 2 / 4) = c ^ 2 * (σ * E + G * E * E / 2) := by ring
  have h2 : c ^ 2 * (σ * E + G * E * E / 2) ≤ (1 + 41 / 100 * E) * (σ * E + G * E * E / 2) :=
    mul_le_mul_of_nonneg_right hc2u hX0
  have h3 : G * E * E ≤ 1 / 10 * E := mul_le_mul_of_nonneg_right hGE hE
  have h4 : E * (σ * E) ≤ E * (1 / 100) := mul_le_mul_of_nonneg_left hσE hE
  have h5 : E * (G * E * E) ≤ E * (1 / 10 * E) := mul_le_mul_of_nonneg_left h3 hE
  have h6 : E * E ≤ 1 / 10 ^ 6 * E := mul_le_mul_of_nonneg_right hE6 hE
  have h7 : (1 + 41 / 100 * E) * (σ * E + G * E * E / 2) =
      σ * E + G * E * E / 2 + 41 / 100 * (E * (σ * E)) + 41 / 200 * (E * (G * E * E)) := by ring
  have h8 : 41 / 100 * E * (2 + E) = 82 / 100 * E + 41 / 100 * (E * E) := by ring
  nlinarith

lemma fin_some (c g1 xf o s S E σ G : ℝ) (hc0 : 0 < c) (hE0 : 1 / 10 ^ 10 ≤ E)
    (hE6 : E ≤ 1 / 10 ^ 6) (hσ : 0 ≤ σ) (hσE : σ * E ≤ 1 / 100) (hG : 0 ≤ G)
    (hGE : G * E ≤ 1 / 10) (hc2 : |c^2 - 1| ≤ 41/100 * E)
    (hg1 : |g1 - 32 * c| ≤ 1/10^20) (ho : o = 0 ∨ o = 1) (hs : s = 1 ∨ s = -1)
    (hx : |xf - s| ≤ E)
    (hS : |S - 2 * c^2 * o| ≤ 2 * c ^ 2 * (σ * E / 2 + G * E ^ 2 / 4) + 11 / 10 ^ 12) :
    |xf + (-1 * (g1 * (c / 32)) + S + -1 * (g1 * (c * xf / 32))) - (2 * o - 1)| ≤ (1 + σ) * E := by
  obtain ⟨hcl, hcu, hc2u⟩ := c_bounds c E hc0 hE6 hc2
  have hE : 0 ≤ E := le_trans (by norm_num) hE0
  have key : xf + (-1 * (g1 * (c / 32)) + S + -1 * (g1 * (c * xf / 32))) - (2 * o - 1) =
      (1 - c^2) * (xf - (2 * o - 1)) + (S - 2 * c^2 * o) - (g1 - 32 * c) * (c * (1 + xf) / 32) := by
    ring
  rw [key]
  have hx' := abs_le.mp hx
  have h1 : |xf - (2 * o - 1)| ≤ 2 + E := by
    rw [abs_le]; rcases ho with rfl | rfl <;> rcases hs with rfl | rfl <;> constructor <;> linarith
  have h2 : |c * (1 + xf) / 32| ≤ 1 := by
    rw [abs_le]; rcases hs with rfl | rfl <;> constructor <;> nlinarith
  have h3 : |1 - c^2| ≤ 41/100 * E := by rw [abs_sub_comm]; exact hc2
  have hR : |1 - c^2| * |xf - (2 * o - 1)| ≤ 41 / 100 * E * (2 + E) :=
    mul_le_mul h3 h1 (abs_nonneg _) (by positivity)
  have hb := budget c E σ G hE0 hE6 hσ hσE hG hGE hc2u _ hR
  calc _ ≤ |(1 - c^2) * (xf - (2 * o - 1)) + (S - 2 * c^2 * o)| +
        |(g1 - 32 * c) * (c * (1 + xf) / 32)| := abs_sub _ _
    _ ≤ |(1 - c^2) * (xf - (2 * o - 1))| + |S - 2 * c^2 * o| +
        |(g1 - 32 * c) * (c * (1 + xf) / 32)| := by
        gcongr; exact abs_add_le _ _
    _ = |1 - c^2| * |xf - (2 * o - 1)| + |S - 2 * c^2 * o| + |g1 - 32 * c| * |c * (1 + xf) / 32| := by
        rw [abs_mul, abs_mul]
    _ ≤ |1 - c^2| * |xf - (2 * o - 1)| + (2 * c ^ 2 * (σ * E / 2 + G * E ^ 2 / 4) + 11 / 10 ^ 12) +
        1/10^20 * 1 := by
        gcongr
    _ ≤ (1 + σ) * E := by linarith

lemma fin_none (c g1 o S E σ G : ℝ) (hc0 : 0 < c) (hE0 : 1 / 10 ^ 10 ≤ E)
    (hE6 : E ≤ 1 / 10 ^ 6) (hσ : 0 ≤ σ) (hσE : σ * E ≤ 1 / 100) (hG : 0 ≤ G)
    (hGE : G * E ≤ 1 / 10) (hc2 : |c^2 - 1| ≤ 41/100 * E)
    (hg1 : |g1 - 32 * c| ≤ 1/10^20) (ho : o = 0 ∨ o = 1)
    (hS : |S - 2 * c^2 * o| ≤ 2 * c ^ 2 * (σ * E / 2 + G * E ^ 2 / 4) + 11 / 10 ^ 12) :
    |0 + (-1 * (g1 * (c / 32)) + S + 0) - (2 * o - 1)| ≤ (1 + σ) * E := by
  obtain ⟨hcl, hcu, hc2u⟩ := c_bounds c E hc0 hE6 hc2
  have hE : 0 ≤ E := le_trans (by norm_num) hE0
  have key : 0 + (-1 * (g1 * (c / 32)) + S + 0) - (2 * o - 1) =
      (S - 2 * c^2 * o) + (c^2 - 1) * (2 * o - 1) - (g1 - 32 * c) * (c / 32) := by ring
  rw [key]
  have h1 : |2 * o - 1| ≤ 1 := by rw [abs_le]; rcases ho with rfl | rfl <;> norm_num
  have h2 : |c / 32| ≤ 1 := by rw [abs_le]; constructor <;> linarith
  have hR : |c^2 - 1| * |2 * o - 1| ≤ 41 / 100 * E * (2 + E) := by
    have := mul_le_mul hc2 h1 (abs_nonneg _) (by positivity)
    nlinarith
  have hb := budget c E σ G hE0 hE6 hσ hσE hG hGE hc2u _ hR
  calc _ ≤ |(S - 2 * c^2 * o) + (c^2 - 1) * (2 * o - 1)| + |(g1 - 32 * c) * (c / 32)| := abs_sub _ _
    _ ≤ |S - 2 * c^2 * o| + |(c^2 - 1) * (2 * o - 1)| + |(g1 - 32 * c) * (c / 32)| := by
        gcongr; exact abs_add_le _ _
    _ = |S - 2 * c^2 * o| + |c^2 - 1| * |2 * o - 1| + |g1 - 32 * c| * |c / 32| := by
        rw [abs_mul, abs_mul]
    _ ≤ (2 * c ^ 2 * (σ * E / 2 + G * E ^ 2 / 4) + 11 / 10 ^ 12) + |c^2 - 1| * |2 * o - 1| +
        1/10^20 * 1 := by gcongr
    _ ≤ (1 + σ) * E := by linarith

/-! ## Rows -/

lemma featIn_spec (W : ℕ) (r : Row) (n : Name) (h : inRow W r n = true) :
    featIn W r n < W ∧ r (featIn W r n) = some n := by
  unfold inRow at h
  unfold featIn
  obtain ⟨a, ha⟩ : ∃ a, (List.range W).find? (fun f => r f == some n) = some a := by
    apply Option.isSome_iff_exists.mp
    rw [List.find?_isSome]
    exact List.any_eq_true.mp h
  rw [ha, Option.getD_some]
  have h1 := List.find?_some ha
  have h2 := List.mem_of_find?_eq_some ha
  exact ⟨List.mem_range.mp h2, by simpa using h1⟩

lemma exists_idx {l : List ℕ} {y : ℕ} (h : y ∈ l) : ∃ i < l.length, l.getD i 0 = y :=
  ⟨l.idxOf y, List.idxOf_lt_length_of_mem h,
    by simp [List.getD_eq_getElem?_getD, List.getElem?_idxOf, h]⟩

lemma getD_mem' {l : List ℕ} {k : ℕ} (hk : k < l.length) : l.getD k 0 ∈ l := by
  rw [List.getD_eq_getElem _ _ hk]; exact List.getElem_mem hk

lemma sig_bit (ht : Stmt_trace) (t : Trace) (hWF : t.WellFormed) (xb : ℕ → Bool) (s : Sig)
    (h : s ∈ t.inputs ∨ ∃ m ∈ t.mlps, s ∈ m.outs) : IsBit (t.values (t.env xb) s) := by
  rcases h with h | ⟨m, hm, hs⟩
  · rw [(ht t (t.env xb) hWF).1 s h]
    unfold Trace.env IsBit
    split_ifs <;> simp
  · exact hWF.bits xb m hm s hs

lemma name_bit (ht : Stmt_trace) (t : Trace) (W : ℕ) (r : Row) (val : Name → ℝ)
    (hr : RowOK t W r) (hV : ValOK t val) (hWF : t.WellFormed) (f : ℕ) (hf : 1 ≤ f) (n : Name)
    (hn : r f = some n) : n ≠ .one ∧ IsBit (val n) := by
  obtain ⟨xb, hxb⟩ := hV.1
  rcases hr.names f n hn with h | ⟨i, rfl, -⟩ | ⟨s, rfl, hs⟩
  · subst h; have := hr.one_only f hn; omega
  · exact ⟨by simp, hV.2 i⟩
  · refine ⟨by simp, ?_⟩
    rw [hxb]; exact sig_bit ht t hWF xb s hs

lemma ideal_some (one : ℝ) (val : Name → ℝ) (r : Row) (f : ℕ) (n : Name) (hn : r f = some n)
    (h1 : n ≠ .one) : ideal one val r f = 2 * val n - 1 := by
  unfold ideal; rw [hn]
  cases n with
  | one => exact absurd rfl h1
  | sig s => rfl
  | code i => rfl

lemma bit_pm (o : ℝ) (h : IsBit o) : 2 * o - 1 = 1 ∨ 2 * o - 1 = -1 := by
  rcases h with rfl | rfl <;> norm_num

end LayerGainAux

open LayerGainAux in
theorem layer_gain_of (hd : Stmt_decomp) (hu : Stmt_unit_dots) (hc : Stmt_compiled)
    (ht : Stmt_trace) (hn : Stmt_norm_close) (hg : Stmt_gelu) : Stmt_layer_gain := by
  intro t W before row val x E G σ hWF hRC hP hV hW hE0 hE6 hG hGE hσ hσE hCl
  have hE : 0 ≤ E := le_trans (by norm_num) hE0
  have hEσ : E ≤ (1 + σ) * E := by nlinarith
  obtain ⟨c, hc0, hc2, hz⟩ := hn t W before val x E hP.before_ok hV hWF hW hE hE6
    hCl.one_eq hCl.near
  obtain ⟨hcl, hcu, -⟩ := c_bounds c E hc0 hE6 hc2
  set z := rmsnorm W before (2 * Real.sqrt W) x with hzdef
  set us := layerUnits W t.mlps before row with hus
  set ψ : Hidden → ℝ := fun u => gelu (dot W u.gate z) * dot W u.val z with hψ
  have hstep : ∀ f, netStep gelu W (2 * Real.sqrt W) before us x f =
      x f + unitSum (fun q : ℚ => (q : ℝ)) ψ us f := fun f => rfl
  have hdec := hd (fun q : ℚ => (q : ℝ)) ψ t W before row (by simp) hWF hP
  have hz0 : z 0 = c := by rw [hz]; simp
  have hzf : ∀ f, f ≠ 0 → z f = c * x f := fun f hf => by rw [hz, ite_eq_right hf]
  refine ⟨?_, ?_, ?_⟩
  · -- one_eq
    rw [hstep, hdec.1 0 (by rw [hP.row_ok.one_zero, hP.before_ok.one_zero]), add_zero, hCl.one_eq]
  · intro f hf hrf
    have hbf : before f = none := by
      by_contra h
      have := hP.held f (Option.ne_none_iff_isSome.mp h)
      rw [hrf] at this; simp at this
    rw [hstep, hdec.1 f (by rw [hrf, hbf]), add_zero, hCl.none_eq f hf hbf]
  · intro f hf1 hfW
    by_cases heq : row f = before f
    · rw [hstep, hdec.1 f heq, add_zero]
      have : ideal (2 * Real.sqrt W) val row f = ideal (2 * Real.sqrt W) val before f := by
        simp only [ideal, heq]
      rw [this]; exact (hCl.near f hf1 hfW).trans hEσ
    · obtain ⟨m, hm, hnew, y, hy, hrow⟩ := hP.changed f heq
      obtain ⟨i, hi, hiy⟩ := exists_idx hy
      obtain ⟨xb, hxb⟩ := hV.1
      set u := t.inVals xb m with hudef
      have hval_u : ∀ k, val (.sig (m.ins.getD k 0)) = u k := fun k => by rw [hxb]; rfl
      have ho : val (.sig y) = m.out u i := by
        rw [hxb, ← hiy]; exact (ht t (t.env xb) hWF).2 m hm i hi
      have hobit : m.out u i = 0 ∨ m.out u i = 1 := by
        rw [← ho, hxb]; exact hWF.bits xb m hm y hy
      have hideal : ideal (2 * Real.sqrt W) val row f = 2 * m.out u i - 1 := by
        rw [ideal_some _ _ _ _ _ hrow (by simp), ho]
      have hfeat : ∀ k < m.ins.length, featIn W before (.sig (m.ins.getD k 0)) < W ∧
          before (featIn W before (.sig (m.ins.getD k 0))) = some (.sig (m.ins.getD k 0)) :=
        fun k hk => featIn_spec W before _ (hP.new_ins m hm hnew _ (getD_mem' hk))
      have hfeat0 : ∀ k < m.ins.length, featIn W before (.sig (m.ins.getD k 0)) ≠ 0 := by
        intro k hk h0
        have := (hfeat k hk).2
        rw [h0, hP.before_ok.one_zero] at this
        simp at this
      have hubit : ∀ k < m.ins.length, |u k| ≤ 1 := by
        intro k hk
        have hb := (name_bit ht t W before val hP.before_ok hV hWF _
          (Nat.one_le_iff_ne_zero.mpr (hfeat0 k hk)) _ (hfeat k hk).2).2
        rw [hval_u] at hb
        rcases hb with h | h <;> rw [h] <;> norm_num
      set d : ℕ → ℝ := fun k => (x (featIn W before (.sig (m.ins.getD k 0))) - (2 * u k - 1)) / 2
        with hddef
      have hd_le : ∀ k < m.ins.length, |d k| ≤ E / 2 := by
        intro k hk
        have h := hCl.near _ (Nat.one_le_iff_ne_zero.mpr (hfeat0 k hk)) (hfeat k hk).1
        rw [ideal_some _ _ _ _ _ (hfeat k hk).2 (by simp), hval_u] at h
        simp only [hddef]
        rw [abs_div, abs_two]
        linarith
      have hxd : ∀ k, x (featIn W before (.sig (m.ins.getD k 0))) = 2 * (u k + d k) - 1 := by
        intro k; simp only [hddef]; ring
      have hS : ∑ j ∈ range m.compiled.n, ((m.compiled.wo i j : ℚ) : ℝ) *
            ψ ((mlpUnits W before row m).getD j ⟨0, 0, 0⟩) =
          ∑ j ∈ range m.toReglu.n, (2 * (m.toReglu.wo i j : ℝ)) *
            (gelu (32 * c * RP m (fun k => u k + d k) j) * (c / 32 * RQ m (fun k => u k + d k) j)) := by
        rw [(hc.2.2.1 m).1]
        apply sum_congr rfl
        intro j hj
        have hj' : j < m.compiled.n := by rw [(hc.2.2.1 m).1]; exact mem_range.mp hj
        have hU := (hu.2.2 t W before row m j hWF hP hm hnew hj').1 z
        simp only [hψ]
        rw [hU.1, hU.2, hc.2.1 m i j]
        have h1 := (hc.1 m c (fun k => u k + d k) j).1
        have h2 := (hc.1 m c (fun k => u k + d k) j).2
        have hgd : gateDot W before m j z = 32 * c * RP m (fun k => u k + d k) j := by
          unfold gateDot RP
          rw [← h1, hz0]
          congr 1
          apply sum_congr rfl
          intro k hk
          rw [hzf _ (hfeat0 k (mem_range.mp hk)), hxd]
        have hvd : valDot W before m j z = c / 32 * RQ m (fun k => u k + d k) j := by
          unfold valDot RQ
          rw [← h2, hz0]
          congr 1
          apply sum_congr rfl
          intro k hk
          rw [hzf _ (hfeat0 k (mem_range.mp hk)), hxd]
        rw [hgd, hvd]; push_cast; ring
      have hC := hc.2.2.2 t xb m hWF hm
      have hout : m.out u i = ∑ j ∈ range m.toReglu.n,
          (m.toReglu.wo i j : ℝ) * (relu (RP m u j) * RQ m u j) := hC.1 i
      have hPst : m.act = .step → (∀ j < m.toReglu.n, 1/2 ≤ |RP m u j|) ∧
          ∀ j < m.hidden, (0 < RP m u j ↔ 0 < RP m u (m.hidden + j)) := hC.2.2.1
      have hP12 : ∀ j < m.toReglu.n, 1/2 ≤ |RP m u j| := by
        intro j hj
        by_cases hs : m.act = .step
        · exact (hPst hs).1 j hj
        · have hpre : RP m u j = m.pre u j := hC.2.2.2 hs j hj
          rw [hpre]
          exact hRC.gate_margin xb m hm hs j (by rw [← n_ne_step m hs]; exact hj)
      have hlin := lin_le m u d i E σ hE hσ hd_le (fun hs => (hPst hs).2)
        (fun hs => hRC.slope xb m hm hs i hi)
      have hquad := quad_le m u d i E G hG hd_le (fun hs => hRC.glu_gain m hm hs i hi)
      have hcore := core hg m u d c E _ _ i (hRC.bounded m hm) hi hout hP12 hlin hquad hubit
        hE hE6 hd_le hc0 hc2
      have hg1 : |gelu (32 * c) - 32 * c| ≤ 1/10^20 := by
        have h8 : 8 ≤ |32 * c| := by rw [abs_of_pos (by positivity)]; linarith
        have := hg _ h8
        rwa [show relu (32 * c) = 32 * c from max_eq_left (by positivity)] at this
      have hone : ψ (oneUnit W before row) = gelu (32 * c) * (c / 32) := by
        simp only [hψ]; rw [(hu.1 W before row z hW).1, (hu.1 W before row z hW).2, hz0]
      have hm1 : ((-1 : ℚ) : ℝ) = -1 := by norm_num
      rw [hstep, hdec.2 f m i hm hnew hi (by rw [hiy]; exact hrow), hS, hone, hideal, hm1]
      by_cases hbf : (before f).isSome
      · rw [ite_eq_left hbf]
        have herase : ψ (eraseUnit f) = gelu (32 * c) * (c * x f / 32) := by
          simp only [hψ]
          rw [(hu.2.1 W f z hf1 hfW).1, (hu.2.1 W f z hf1 hfW).2, hz0, hzf f (by omega)]
        rw [herase]
        obtain ⟨n, hnf⟩ := Option.isSome_iff_exists.mp hbf
        have hnb := name_bit ht t W before val hP.before_ok hV hWF f hf1 n hnf
        have hnear := hCl.near f hf1 hfW
        rw [ideal_some _ _ _ _ _ hnf hnb.1] at hnear
        exact fin_some c _ (x f) _ _ _ E σ G hc0 hE0 hE6 hσ hσE hG hGE hc2 hg1 hobit
          (bit_pm _ hnb.2) hnear hcore
      · rw [ite_eq_right hbf]
        have hbn : before f = none := by simpa using hbf
        rw [hCl.none_eq f hf1 hbn]
        exact fin_none c _ _ _ E σ G hc0 hE0 hE6 hσ hσE hG hGE hc2 hg1 hobit hcore

end
end Reifier

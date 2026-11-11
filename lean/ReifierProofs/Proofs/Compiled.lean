import ReifierProofs.Main

namespace Reifier

open Finset

noncomputable section

namespace CompiledProof

/-! ## The algebra of `signed ∘ sharpen 32` -/

theorem gate_alg (n : ℕ) (a : ℕ → ℝ) (b c : ℝ) (w : ℕ → ℝ) :
    ∑ k ∈ range n, (32 * a k / 2) * (c * (2 * w k - 1)) + (32 * b + (∑ k ∈ range n, 32 * a k) / 2) * c
      = 32 * c * (∑ k ∈ range n, a k * w k + b) := by
  have h1 : ∑ k ∈ range n, (32 * a k / 2) * (c * (2 * w k - 1))
      = 32 * c * ∑ k ∈ range n, a k * w k - 16 * c * ∑ k ∈ range n, a k := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl ?_
    intro k _
    ring
  have h2 : (∑ k ∈ range n, 32 * a k) = 32 * ∑ k ∈ range n, a k := by
    rw [Finset.mul_sum]
  rw [h1, h2]
  ring

theorem val_alg (n : ℕ) (a : ℕ → ℝ) (b c : ℝ) (w : ℕ → ℝ) :
    ∑ k ∈ range n, (a k / 32 / 2) * (c * (2 * w k - 1)) + (b / 32 + (∑ k ∈ range n, a k / 32) / 2) * c
      = c / 32 * (∑ k ∈ range n, a k * w k + b) := by
  have h1 : ∑ k ∈ range n, (a k / 32 / 2) * (c * (2 * w k - 1))
      = c / 32 * ∑ k ∈ range n, a k * w k - c / 64 * ∑ k ∈ range n, a k := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl ?_
    intro k _
    ring
  have h2 : (∑ k ∈ range n, a k / 32) = (∑ k ∈ range n, a k) / 32 := by
    rw [Finset.sum_div]
  rw [h1, h2]
  ring

theorem toReglu_ins (m : MLP) : m.toReglu.ins = m.ins := by
  unfold MLP.toReglu; cases m.act <;> rfl

theorem toReglu_outs (m : MLP) : m.toReglu.outs = m.outs := by
  unfold MLP.toReglu; cases m.act <;> rfl

theorem compiled_g (m : MLP) (j k : ℕ) : m.compiled.g j k = 32 * m.toReglu.g j k / 2 := rfl

theorem compiled_gb (m : MLP) (j : ℕ) : m.compiled.gb j =
    32 * m.toReglu.gb j + (∑ k ∈ range m.ins.length, 32 * m.toReglu.g j k) / 2 := by
  rw [← toReglu_ins]; rfl

theorem compiled_v (m : MLP) (j k : ℕ) : m.compiled.v j k = m.toReglu.v j k / 32 / 2 := rfl

theorem compiled_vb (m : MLP) (j : ℕ) : m.compiled.vb j =
    m.toReglu.vb j / 32 + (∑ k ∈ range m.ins.length, m.toReglu.v j k / 32) / 2 := by
  rw [← toReglu_ins]; rfl

theorem part1 (m : MLP) (c : ℝ) (w : ℕ → ℝ) (j : ℕ) :
    (∑ k ∈ range m.ins.length, (m.compiled.g j k : ℝ) * (c * (2 * w k - 1)) + m.compiled.gb j * c
      = 32 * c * (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * w k + m.toReglu.gb j)) ∧
    (∑ k ∈ range m.ins.length, (m.compiled.v j k : ℝ) * (c * (2 * w k - 1)) + m.compiled.vb j * c
      = c / 32 * (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * w k + m.toReglu.vb j)) := by
  constructor
  · simp only [compiled_g, compiled_gb]
    push_cast
    exact gate_alg _ _ _ _ _
  · simp only [compiled_v, compiled_vb]
    push_cast
    exact val_alg _ _ _ _ _

/-! ## The reglu form on bits -/

theorem toReglu_step (m : MLP) (h : m.act = .step) : m.toReglu =
    { ins := m.ins, outs := m.outs, n := 2 * m.hidden
      g := fun j k => 4 * m.wi (if j < m.hidden then j else j - m.hidden) k
      gb := fun j => if j < m.hidden then 4 * m.b j + 2 + 1 / 2 else 4 * m.b (j - m.hidden) + 2 - 1 / 2
      v := fun _ _ => 0
      vb := fun _ => 1
      wo := fun i j => if j < m.hidden then m.wo i j else -m.wo i (j - m.hidden) } := by
  unfold MLP.toReglu; rw [h]

theorem toReglu_relu (m : MLP) (h : m.act = .relu) : m.toReglu =
    { ins := m.ins, outs := m.outs, n := m.hidden
      g := m.wi, gb := m.b, v := fun _ _ => 0, vb := fun _ => 1, wo := m.wo } := by
  unfold MLP.toReglu; rw [h]

theorem toReglu_reglu (m : MLP) (h : m.act = .reglu) : m.toReglu =
    { ins := m.ins, outs := m.outs, n := m.hidden
      g := m.wi, gb := m.b
      v := fun j k => m.wi (m.hidden + j) k, vb := fun j => m.b (m.hidden + j)
      wo := m.wo } := by
  unfold MLP.toReglu; rw [h]

theorem sum_four (m : MLP) (u : ℕ → ℝ) (j : ℕ) :
    ∑ k ∈ range m.ins.length, ((4 * m.wi j k : ℚ) : ℝ) * u k
      = 4 * ∑ k ∈ range m.ins.length, (m.wi j k : ℝ) * u k := by
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro k _
  push_cast; ring

theorem step_pair (p w : ℝ) (hp : 0 ≤ p ∨ p ≤ -3 / 4) :
    w * (relu (4 * p + 5 / 2) * 1) + (-w) * (relu (4 * p + 3 / 2) * 1) = w * heaviside p := by
  unfold relu heaviside
  rcases hp with hp | hp
  · have : (0:ℝ) ≤ p := hp
    rw [max_eq_left (by linarith), max_eq_left (by linarith)]; simp only [this, ↓reduceIte]; ring
  · have : ¬ (0:ℝ) ≤ p := by linarith
    rw [max_eq_right (by linarith), max_eq_right (by linarith)]; simp only [this, ↓reduceIte]; ring

theorem sem_step (m : MLP) (h : m.act = .step) (u : ℕ → ℝ)
    (hm : ∀ j < m.hidden, 0 ≤ m.pre u j ∨ m.pre u j ≤ -3 / 4) (i : ℕ) :
    m.out u i = ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) *
      (relu (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j) *
        (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * u k + m.toReglu.vb j)) := by
  rw [toReglu_step m h]
  dsimp only
  rw [two_mul, Finset.sum_range_add, ← Finset.sum_add_distrib]
  unfold MLP.out
  refine Finset.sum_congr rfl ?_
  intro j hj
  have hj' : j < m.hidden := Finset.mem_range.mp hj
  have hnj : ¬ (m.hidden + j < m.hidden) := by omega
  have hsub : m.hidden + j - m.hidden = j := by omega
  simp only [hj', hnj, ↓reduceIte, hsub, sum_four]
  simp only [MLP.unit, h]
  have hpre : m.pre u j = ∑ k ∈ range m.ins.length, (m.wi j k : ℝ) * u k + m.b j := rfl
  push_cast
  simp only [zero_mul, Finset.sum_const_zero, zero_add]
  rw [← step_pair (m.pre u j) (m.wo i j) (hm j hj'), hpre]
  ring_nf


theorem sem_relu (m : MLP) (h : m.act = .relu) (u : ℕ → ℝ) (i : ℕ) :
    m.out u i = ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) *
      (relu (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j) *
        (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * u k + m.toReglu.vb j)) := by
  rw [toReglu_relu m h]
  simp [MLP.out, MLP.unit, MLP.pre, h]

theorem sem_reglu (m : MLP) (h : m.act = .reglu) (u : ℕ → ℝ) (i : ℕ) :
    m.out u i = ∑ j ∈ range m.toReglu.n, (m.toReglu.wo i j : ℝ) *
      (relu (∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j) *
        (∑ k ∈ range m.ins.length, (m.toReglu.v j k : ℝ) * u k + m.toReglu.vb j)) := by
  rw [toReglu_reglu m h]
  simp [MLP.out, MLP.unit, MLP.pre, h]

theorem gate_step (m : MLP) (h : m.act = .step) (u : ℕ → ℝ) (j : ℕ) :
    ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j =
      if j < m.hidden then 4 * m.pre u j + 5 / 2 else 4 * m.pre u (j - m.hidden) + 3 / 2 := by
  rw [toReglu_step m h]
  dsimp only
  have hpre : ∀ j', m.pre u j' = ∑ k ∈ range m.ins.length, (m.wi j' k : ℝ) * u k + m.b j' :=
    fun _ => rfl
  by_cases hj : j < m.hidden
  · simp only [hj, ↓reduceIte, sum_four, hpre]
    push_cast; ring
  · simp only [hj, ↓reduceIte, sum_four, hpre]
    push_cast; ring

theorem step_gate_abs (m : MLP) (h : m.act = .step) (u : ℕ → ℝ)
    (hm : ∀ j < m.hidden, 0 ≤ m.pre u j ∨ m.pre u j ≤ -3 / 4) :
    ∀ j < m.toReglu.n,
      1 / 2 ≤ |∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j| := by
  intro j hj
  have hn : m.toReglu.n = 2 * m.hidden := by rw [toReglu_step m h]
  rw [gate_step m h u j]
  by_cases hjh : j < m.hidden
  · simp only [hjh, ↓reduceIte]
    rcases hm j hjh with hp | hp
    · rw [abs_of_nonneg (by linarith)]; linarith
    · rw [abs_of_neg (by linarith)]; linarith
  · simp only [hjh, ↓reduceIte]
    have hj2 : j - m.hidden < m.hidden := by omega
    rcases hm (j - m.hidden) hj2 with hp | hp
    · rw [abs_of_nonneg (by linarith)]; linarith
    · rw [abs_of_neg (by linarith)]; linarith

theorem step_same_side (m : MLP) (h : m.act = .step) (u : ℕ → ℝ)
    (hm : ∀ j < m.hidden, 0 ≤ m.pre u j ∨ m.pre u j ≤ -3 / 4) :
    ∀ j < m.hidden,
      (0 < ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j ↔
        0 < ∑ k ∈ range m.ins.length, (m.toReglu.g (m.hidden + j) k : ℝ) * u k +
          m.toReglu.gb (m.hidden + j)) := by
  intro j hj
  rw [gate_step m h u j, gate_step m h u (m.hidden + j)]
  have hnj : ¬ (m.hidden + j < m.hidden) := by omega
  have hsub : m.hidden + j - m.hidden = j := by omega
  simp only [hj, hnj, hsub, ↓reduceIte]
  rcases hm j hj with hp | hp
  · constructor <;> intro _ <;> linarith
  · constructor <;> intro h' <;> linarith

theorem gate_relu (m : MLP) (h : m.act ≠ .step) (u : ℕ → ℝ) (j : ℕ) :
    ∑ k ∈ range m.ins.length, (m.toReglu.g j k : ℝ) * u k + m.toReglu.gb j = m.pre u j := by
  cases hact : m.act with
  | step => exact absurd hact h
  | relu => rw [toReglu_relu m hact]; rfl
  | reglu => rw [toReglu_reglu m hact]; rfl

theorem toReglu_n_of_ne (m : MLP) (h : m.act ≠ .step) : m.toReglu.n = m.hidden := by
  cases hact : m.act with
  | step => exact absurd hact h
  | relu => rw [toReglu_relu m hact]
  | reglu => rw [toReglu_reglu m hact]

end CompiledProof

open CompiledProof in
theorem stmt_compiled : Stmt_compiled := by
  refine ⟨fun m c w j => part1 m c w j, fun m i j => rfl,
    fun m => ⟨rfl, toReglu_ins m, toReglu_outs m⟩, ?_⟩
  intro t x m hwf hm
  dsimp only
  refine ⟨fun i => ?_, ?_, ?_, ?_⟩
  · cases hact : m.act with
    | step => exact sem_step m hact _ (hwf.step_margin x m hm hact) i
    | relu => exact sem_relu m hact _ i
    | reglu => exact sem_reglu m hact _ i
  · intro j hj
    by_cases hact : m.act = .step
    · have := step_gate_abs m hact _ (hwf.step_margin x m hm hact) j hj
      right
      linarith
    · rw [gate_relu m hact]
      rw [toReglu_n_of_ne m hact] at hj
      exact hwf.gate_margin x m hm hact j hj
  · intro hact
    exact ⟨step_gate_abs m hact _ (hwf.step_margin x m hm hact),
      step_same_side m hact _ (hwf.step_margin x m hm hact)⟩
  · intro hact j _
    exact gate_relu m hact _ j

end

end Reifier

import ReifierProofs.Main
import ReifierProofs.Examples

/-!
# The hypotheses of the main theorems are satisfiable

`Reifier.Examples.xor3` (the README's `trace(xor, 3)`: `odd = [atleast1 - atleast2 + atleast3 >= 1]`
on three input bits) is a program on tokens of one bit (`IsProgram`), whose weights are of bounded
size and whose GLUs are flat and of small gain, there being none (`RealOK`, and `RealCore (10^5) 0`),
and whose compiled weights, gates, values and products are bf16 numbers on every input (`BF16OK`).
The codes `[(0, [false]), (1, [true])]` satisfy `CodesOK`, every position from 2 on satisfies
`PosOK`, its width (5) satisfies `WidthOK` and `2 sqrt 5 + 1 ≤ 250`, its depth bound for
`MainRealDeep` is `(1 + 0)^2 = 1 ≤ 10^4`, and some token's code is the trace's output. So none of
`MainReal`, `MainBF16`, `MainRealDeep` is vacuous: `NV.hypotheses_satisfiable`,
`NV.deep_hypotheses_satisfiable`.

And `MainRealDeep` says more than `MainReal`: `tg`, the GLU `relu(x + 1/2) * (2/3) x` on the bit of
the last token (the identity on bits, gates 1/2 and 3/2, slope 1/3 at 0 and 5/3 at 1), satisfies
its hypotheses with `σ = 5/3` (`(8/3)^1 ≤ 10^4`) but is not flat, so not `RealOK`:
`NV.deep_beyond_realOK`.

And the relaxed `sums` of `BF16OK` (a gate at most 0 need not be a bf16 number) covers `lib.py`'s
`and_` on the bits of 33 one-bit tokens: `tAnd`, one threshold gate of 33 inputs of weight 1 and
threshold 33 (width 35), satisfies every hypothesis of `MainReal` and `MainBF16`, though its gate
`u+` on the all-zero input is `-4144`, not a bf16 number (on bits its gates are `128 c - 4144` and
`128 c - 4176`, `c` the number of ones: at most `-48` unless `c = 33`, where they are 80 and 48):
`NV.and33_covered`.

Everything is proved; the values of the traces come from `values_xor3`, `tg_out` and
`tAnd_value`, by cases on the bits (for `tAnd`, on the number of ones).
-/

open Reifier Reifier.Examples Finset

noncomputable section

namespace Reifier.NV

lemma isBF16_of_e (x : ℝ) (e : ℤ) (h1 : ((⌊x / 2 ^ e⌋ : ℤ) : ℝ) = x / 2 ^ e)
    (h2 : |⌊x / 2 ^ e⌋| < 256) (h3 : -133 ≤ e) (h4 : e ≤ 120) : IsBF16 x := by
  refine ⟨⌊x / 2 ^ e⌋, e, ?_, h2, h3, h4⟩
  rw [h1]
  field_simp

macro "bf16_num" : tactic => `(tactic| first
  | (refine isBF16_of_e _ 0 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 4 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-1) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-5) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 1 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 2 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ 3 ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-2) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-3) ?_ ?_ ?_ ?_ <;> norm_num <;> done)
  | (refine isBF16_of_e _ (-4) ?_ ?_ ?_ ?_ <;> norm_num <;> done))

lemma mem_mlps {m : MLP} (hm : m ∈ xor3.mlps) :
    m = gate [0, 1, 2] 3 [1, 1, 1] 1 ∨ m = gate [0, 1, 2] 4 [1, 1, 1] 2 ∨
      m = gate [0, 1, 2] 5 [1, 1, 1] 3 ∨ m = gate [3, 4, 5] 6 [1, -1, 1] 1 := by
  simpa [xor3] using hm


def bv (b : Bool) : ℝ := if b then 1 else 0

/-- The values of `xor3`'s signals, on the bits `x`. -/
lemma values_xor3 (x : ℕ → Bool) :
    let v := xor3.values (xor3.env x)
    v 0 = bv (x 0) ∧ v 1 = bv (x 1) ∧ v 2 = bv (x 2) ∧
    v 3 = heaviside (bv (x 0) + bv (x 1) + bv (x 2) - 1) ∧
    v 4 = heaviside (bv (x 0) + bv (x 1) + bv (x 2) - 2) ∧
    v 5 = heaviside (bv (x 0) + bv (x 1) + bv (x 2) - 3) ∧
    v 6 = heaviside (heaviside (bv (x 0) + bv (x 1) + bv (x 2) - 1)
      - heaviside (bv (x 0) + bv (x 1) + bv (x 2) - 2)
      + heaviside (bv (x 0) + bv (x 1) + bv (x 2) - 3) - 1) := by
  simp [Trace.values, xor3, MLP.apply, MLP.out, MLP.unit, MLP.pre, MLP.read, gate,
    Trace.env, Finset.sum_range_succ, bv]
  ring_nf
  simp

lemma inVals_gate (x : ℕ → Bool) (ins : List Sig) (out : Sig) (w : List ℚ) (θ : ℚ) (k : ℕ) :
    xor3.inVals x (gate ins out w θ) k = xor3.values (xor3.env x) (ins.getD k 0) := rfl

lemma heaviside_bit (r : ℝ) : IsBit (heaviside r) := by
  unfold heaviside IsBit; split_ifs <;> simp

lemma xor3_bits : ∀ (x : ℕ → Bool), ∀ m ∈ xor3.mlps, ∀ y ∈ m.outs,
    IsBit (xor3.values (xor3.env x) y) := by
  intro x m hm y hy
  obtain ⟨h0, h1, h2, h3, h4, h5, h6⟩ := values_xor3 x
  rcases mem_mlps hm with rfl | rfl | rfl | rfl <;> simp [gate] at hy <;> subst hy <;>
    simp only [h3, h4, h5, h6, heaviside_bit]

/-- With integer weights, a step's sum is at least 0 or at most -1, so at most `-3/4`. -/
lemma xor3_step_margin : ∀ (x : ℕ → Bool), ∀ m ∈ xor3.mlps, m.act = .step → ∀ j < m.hidden,
    0 ≤ m.pre (xor3.inVals x m) j ∨ m.pre (xor3.inVals x m) j ≤ -1 := by
  intro x m hm _ j _
  obtain ⟨h0, h1, h2, h3, h4, h5, h6⟩ := values_xor3 x
  rcases mem_mlps hm with rfl | rfl | rfl | rfl <;>
  · simp only [MLP.pre, inVals_gate]
    simp only [gate, List.length_cons, List.length_nil, sum_range_succ,
      sum_range_zero, List.getD_cons_zero, List.getD_cons_succ, h0, h1, h2, h3, h4, h5]
    cases x 0 <;> cases x 1 <;> cases x 2 <;> norm_num [bv, heaviside]

theorem xor3_wf : xor3.WellFormed where
  nodup := by decide
  outs_ne := by
    intro m hm
    rcases mem_mlps hm with rfl | rfl | rfl | rfl <;> simp [gate]
  topo := by
    intro i hi x hx
    simp only [xor3, List.length_cons, List.length_nil] at hi
    interval_cases i
    · left; simpa [xor3, gate] using hx
    · left; simpa [xor3, gate] using hx
    · left; simpa [xor3, gate] using hx
    · right
      simp [xor3, gate] at hx
      rcases hx with rfl | rfl | rfl
      · exact ⟨0, by omega, by simp [xor3, gate]⟩
      · exact ⟨1, by omega, by simp [xor3, gate]⟩
      · exact ⟨2, by omega, by simp [xor3, gate]⟩
  bits := xor3_bits
  step_margin := fun x m hm hact j hj =>
    (xor3_step_margin x m hm hact j hj).imp_right fun h => h.trans (by norm_num)
  gate_margin := by
    intro x m hm hact
    rcases mem_mlps hm with rfl | rfl | rfl | rfl <;> simp [gate] at hact

theorem xor3_isProgram : xor3.IsProgram where
  toWellFormed := xor3_wf
  outputs_ne := by simp [xor3]
  outputs_nodup := by decide
  outputs_made := by
    intro y hy
    simp [xor3] at hy
    subst hy
    exact ⟨gate [3, 4, 5] 6 [1, -1, 1] 1, by simp [xor3], by simp [gate]⟩
  window_dvd := by decide
  window_le := by decide

lemma xor3_steps : ∀ m ∈ xor3.mlps, m.act = .step := by
  intro m hm
  rcases mem_mlps hm with rfl | rfl | rfl | rfl <;> rfl

lemma xor3_bounded : ∀ m ∈ xor3.mlps, m.Bounded (2 ^ 16) := by
  intro m hm
  rcases mem_mlps hm with rfl | rfl | rfl | rfl <;>
  · refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;>
      simp [MLP.rows, gate, sum_range_succ] at hj ⊢ <;> norm_num

theorem xor3_realOK : xor3.RealOK where
  gate_margin := fun _ m hm hact => absurd (xor3_steps m hm) hact
  flat := fun _ m hm hact => absurd (xor3_steps m hm) hact
  bounded := xor3_bounded
  glu_gain := fun m hm hact => by rw [xor3_steps m hm] at hact; exact absurd hact (by decide)

/-- `xor3` has no GLU: its first-order gain is 0 and its second-order gain too. -/
theorem xor3_realCore : xor3.RealCore (10 ^ 5) 0 where
  gate_margin := fun _ m hm hact => absurd (xor3_steps m hm) hact
  slope := fun _ m hm hact => absurd (xor3_steps m hm) hact
  bounded := xor3_bounded
  glu_gain := fun m hm hact => by rw [xor3_steps m hm] at hact; exact absurd hact (by decide)

/-- The compiled form of a threshold gate. -/
lemma compiled_gate (ins : List Sig) (out : Sig) (w : List ℚ) (θ : ℚ) :
    let c := (gate ins out w θ).compiled
    c.n = 2 ∧ c.ins = ins ∧ c.outs = [out] ∧
    (∀ j k, j < 2 → c.g j k = 64 * w.getD k 0) ∧
    (∀ j k, c.v j k = 0) ∧
    c.gb 0 = 32 * (4 * -θ + 2 + 1 / 2) + (∑ k ∈ range ins.length, 128 * w.getD k 0) / 2 ∧
    c.gb 1 = 32 * (4 * -θ + 2 - 1 / 2) + (∑ k ∈ range ins.length, 128 * w.getD k 0) / 2 ∧
    (∀ j, c.vb j = 1 / 32) ∧
    (∀ i, c.wo i 0 = 2) ∧ (∀ i, c.wo i 1 = -2) := by
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro j k hj
    simp only [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]
    ring
  · intro j k
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]
  · simp only [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]
    norm_num
    congr 1
    funext k
    ring
  · simp only [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]
    norm_num
    congr 1
    funext k
    ring
  · intro j
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]
  · intro i
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]
  · intro i
    simp [MLP.compiled, Gated.signed, Gated.sharpen, MLP.toReglu, gate, Gated.sharpness]

lemma gOn_gate3 (a b c : Sig) (out : Sig) (w : List ℚ) (θ : ℚ) (j : ℕ) (hj : j < 2) (s : Sig) :
    (gate [a, b, c] out w θ).compiled.gOn j s =
      (if a = s then 64 * w.getD 0 0 else 0) + (if b = s then 64 * w.getD 1 0 else 0) +
        (if c = s then 64 * w.getD 2 0 else 0) := by
  obtain ⟨_, hins, _, hg, _⟩ := compiled_gate [a, b, c] out w θ
  simp only [Gated.gOn, hins, fun k => hg j k hj, List.length_cons, List.length_nil,
    sum_range_succ, sum_range_zero, List.getD_cons_zero, List.getD_cons_succ, zero_add]

lemma vOn_gate (ins : List Sig) (out : Sig) (w : List ℚ) (θ : ℚ) (j : ℕ) (s : Sig) :
    (gate ins out w θ).compiled.vOn j s = 0 := by
  obtain ⟨_, _, _, _, hv, _⟩ := compiled_gate ins out w θ
  simp [Gated.vOn, hv]

lemma weights_gate3 (a b c : Sig) (out : Sig) (w : List ℚ) (θ : ℚ)
    (hw : w = [1, 1, 1] ∨ w = [1, -1, 1]) (hθ : θ = 1 ∨ θ = 2 ∨ θ = 3) :
    let m := gate [a, b, c] out w θ
    ∀ j < m.compiled.n,
    (∀ s, IsBF16 (m.compiled.gOn j s)) ∧ (∀ s, IsBF16 (m.compiled.vOn j s)) ∧
      IsBF16 (m.compiled.gb j) ∧ IsBF16 (m.compiled.vb j) ∧
        ∀ i < m.outs.length, IsBF16 (m.compiled.wo i j) := by
  intro m j hj
  obtain ⟨hn, _, _, _, _, hgb0, hgb1, hvb, hwo0, hwo1⟩ := compiled_gate [a, b, c] out w θ
  rw [hn] at hj
  refine ⟨fun s => ?_, fun s => ?_, ?_, ?_, ?_⟩
  · rw [gOn_gate3 a b c out w θ j hj s]
    rcases hw with rfl | rfl <;> split_ifs <;> bf16_num
  · rw [vOn_gate]; bf16_num
  · interval_cases j
    · rw [hgb0]; rcases hw with rfl | rfl <;> rcases hθ with rfl | rfl | rfl <;>
        simp [sum_range_succ] <;> bf16_num
    · rw [hgb1]; rcases hw with rfl | rfl <;> rcases hθ with rfl | rfl | rfl <;>
        simp [sum_range_succ] <;> bf16_num
  · rw [hvb]; bf16_num
  · intro i _
    interval_cases j
    · rw [hwo0]; bf16_num
    · rw [hwo1]; bf16_num

/-- The gate and value of unit `j` of a compiled threshold gate on three inputs. -/
lemma gateAt_gate3 (a b c : Sig) (out : Sig) (w : List ℚ) (θ : ℚ) (u : ℕ → ℝ) (j : ℕ)
    (hj : j < 2) :
    (gate [a, b, c] out w θ).compiled.gateAt u j =
      64 * (w.getD 0 0 : ℝ) * (2 * u 0 - 1) + 64 * (w.getD 1 0 : ℝ) * (2 * u 1 - 1) +
        64 * (w.getD 2 0 : ℝ) * (2 * u 2 - 1) +
        (32 * (4 * -θ + 2 + (if j = 0 then 1 / 2 else - (1 / 2))) +
          (128 * w.getD 0 0 + 128 * w.getD 1 0 + 128 * w.getD 2 0) / 2 : ℚ) ∧
    (gate [a, b, c] out w θ).compiled.valAt u j = 1 / 32 := by
  obtain ⟨_, hins, _, hg, hv, hgb0, hgb1, hvb, _⟩ := compiled_gate [a, b, c] out w θ
  constructor
  · simp only [Gated.gateAt, hins, fun k => hg j k hj, List.length_cons, List.length_nil,
      sum_range_succ, sum_range_zero, zero_add]
    interval_cases j
    · rw [hgb0]; push_cast [List.length_cons, List.length_nil, sum_range_succ]; ring
    · rw [hgb1]; push_cast [List.length_cons, List.length_nil, sum_range_succ]; ring
  · simp [Gated.valAt, hins, hv, hvb]

lemma bv_cases (b : Bool) : bv b = 0 ∨ bv b = 1 := by cases b <;> simp [bv]

lemma heaviside_cases (r : ℝ) : heaviside r = 0 ∨ heaviside r = 1 := by
  unfold heaviside; split_ifs <;> simp

lemma sums_of (G : ℝ) (hG : IsBF16 G) (hG2 : IsBF16 (relu G * (1 / 32))) :
    IsBF16 G ∧ IsBF16 (1 / 32 : ℝ) ∧ IsBF16 (relu G * (1 / 32)) :=
  ⟨hG, by bf16_num, hG2⟩

/-- Every gate of `xor3`, its value and their product are bf16 numbers: the second disjunct of
`BF16OK.sums`, on every input. -/
theorem xor3_sums : ∀ (x : ℕ → Bool), ∀ m ∈ xor3.mlps, ∀ j < m.compiled.n,
    IsBF16 (m.compiled.gateAt (xor3.inVals x m) j) ∧ IsBF16 (m.compiled.valAt (xor3.inVals x m) j) ∧
      IsBF16 (relu (m.compiled.gateAt (xor3.inVals x m) j) * m.compiled.valAt (xor3.inVals x m) j) := by
  intro x m hm j hj
  obtain ⟨h0, h1, h2, h3, h4, h5, h6⟩ := values_xor3 x
  rcases mem_mlps hm with rfl | rfl | rfl | rfl <;>
  · have hj2 : j < 2 := hj
    obtain ⟨hga, hva⟩ := gateAt_gate3 _ _ _ _ _ _ (xor3.inVals x _) j hj2
    rw [hga, hva]
    simp only [inVals_gate, List.getD_cons_zero, List.getD_cons_succ, h0, h1, h2, h3, h4, h5]
    apply fun h h' => sums_of _ h h'
    all_goals
      rcases bv_cases (x 0) with e0 | e0 <;> rcases bv_cases (x 1) with e1 | e1 <;>
      rcases bv_cases (x 2) with e2 | e2 <;> simp only [e0, e1, e2] <;>
      interval_cases j <;> norm_num [heaviside, relu] <;> bf16_num

theorem xor3_bf16OK : xor3.BF16OK where
  weights := by
    intro m hm
    rcases mem_mlps hm with rfl | rfl | rfl | rfl
    · exact weights_gate3 _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl)
    · exact weights_gate3 _ _ _ _ _ _ (Or.inl rfl) (Or.inr (Or.inl rfl))
    · exact weights_gate3 _ _ _ _ _ _ (Or.inl rfl) (Or.inr (Or.inr rfl))
    · exact weights_gate3 _ _ _ _ _ _ (Or.inr rfl) (Or.inl rfl)
  sums := fun x m hm j hj => Or.inr (xor3_sums x m hm j hj)

/-- The codes of a one-bit program: token 0 is the bit 0, token 1 the bit 1. -/
def codes2 : List (ℕ × List Bool) := [(0, [false]), (1, [true])]

theorem xor3_codesOK : CodesOK xor3 codes2 where
  keys_nodup := by decide
  length := by decide
  inj := by decide

theorem xor3_width : (tokenGrid xor3).width = 5 := by decide

theorem xor3_widthOK : WidthOK xor3 := by
  unfold WidthOK
  rw [xor3_width]
  have h : Real.sqrt 5 ≤ 3 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  unfold maxBF16
  push_cast
  nlinarith

theorem xor3_small : 2 * Real.sqrt (tokenGrid xor3).width + xor3.outputs.length ≤ 250 := by
  rw [xor3_width]
  have h : Real.sqrt 5 ≤ 3 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  simp [xor3]
  linarith

theorem xor3_posOK (toks : ℕ → ℕ) (htoks : ∀ p, toks p < 2) (p : ℕ) (hp : 2 ≤ p) :
    PosOK xor3 codes2 toks p := by
  refine ⟨by simp [Trace.window, xor3]; omega, fun h _ => ?_⟩
  have := htoks (p - h)
  interval_cases toks (p - h) <;> decide

/-- On these codes, some token's code is what the trace gives: the claim on the logits is not
vacuous either. -/
theorem xor3_codeIs (toks : ℕ → ℕ) (p : ℕ) :
    ∃ τ, CodeIs xor3 codes2 τ (valAt xor3 codes2 toks p) := by
  have hb := xor3_bits (windowBits xor3 codes2 toks p) (gate [3, 4, 5] 6 [1, -1, 1] 1)
    (by simp [xor3]) 6 (by simp [gate])
  rcases hb with h | h
  · refine ⟨0, [false], by decide, fun i hi => ?_⟩
    simp [xor3] at hi; subst hi
    show (if [false].getD 0 false then (1 : ℝ) else 0) = valAt xor3 codes2 toks p (.sig 6)
    simp only [valAt, h]; simp
  · refine ⟨1, [true], by decide, fun i hi => ?_⟩
    simp [xor3] at hi; subst hi
    show (if [true].getD 0 false then (1 : ℝ) else 0) = valAt xor3 codes2 toks p (.sig 6)
    simp only [valAt, h]; simp

/-- **The hypotheses of both main theorems hold together**, for `xor3` (the README's
`trace(xor, 3)`) on two tokens of one bit each, at any position from 2 on: `MainReal` and
`MainBF16` are not vacuous. -/
theorem hypotheses_satisfiable :
    ∃ (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
      t.IsProgram ∧ t.RealOK ∧ t.BF16OK ∧ CodesOK t codes ∧ PosOK t codes toks p ∧
        WidthOK t ∧ 2 * Real.sqrt (tokenGrid t).width + t.outputs.length ≤ 250 ∧
        ∃ τ, CodeIs t codes τ (valAt t codes toks p) :=
  ⟨xor3, codes2, fun _ => 0, 2, xor3_isProgram, xor3_realOK, xor3_bf16OK, xor3_codesOK,
    xor3_posOK _ (fun _ => by norm_num) 2 le_rfl, xor3_widthOK, xor3_small, xor3_codeIs _ _⟩

/-- The GLU `relu(x + 1/2) * (2/3) x`: the identity on bits, of slope 1/3 at 0 and 5/3 at 1. -/
def glu1 : MLP where
  ins := [0]
  outs := [1]
  act := .reglu
  hidden := 1
  wi := fun j _ => if j = 0 then 1 else 2 / 3
  b := fun j => if j = 0 then 1 / 2 else 0
  wo := fun _ _ => 1

/-- The program `y = relu(x + 1/2) * (2/3) x` on the bit of the last token. -/
def tg : Trace where
  inputs := [0]
  mlps := [glu1]
  outputs := [1]

lemma tg_mem {m : MLP} (hm : m ∈ tg.mlps) : m = glu1 := by simpa [tg] using hm

lemma tg_in (x : ℕ → Bool) : tg.values (tg.env x) 0 = bv (x 0) := by
  simp [Trace.values, tg, glu1, MLP.apply, Trace.env, bv]

lemma tg_inVals (x : ℕ → Bool) (k : ℕ) : tg.inVals x glu1 k = bv (x 0) := by
  show tg.values (tg.env x) (glu1.ins.getD k 0) = _
  rcases k with _ | k
  · exact tg_in x
  · simpa [glu1] using tg_in x

lemma glu1_pre0 (u : ℕ → ℝ) : glu1.pre u 0 = u 0 + 1 / 2 := by simp [MLP.pre, glu1]
lemma glu1_pre1 (u : ℕ → ℝ) : glu1.pre u 1 = 2 / 3 * u 0 := by simp [MLP.pre, glu1]

lemma glu1_out (u : ℕ → ℝ) : glu1.out u 0 = relu (u 0 + 1 / 2) * (2 / 3 * u 0) := by
  simp [MLP.out, MLP.unit, glu1, MLP.pre]

lemma tg_out (x : ℕ → Bool) : tg.values (tg.env x) 1 = bv (x 0) := by
  have h : tg.values (tg.env x) 1 = glu1.out (glu1.read (tg.env x)) 0 := by
    simp [Trace.values, tg, MLP.apply, show glu1.outs = [1] from rfl]
  rw [h, glu1_out]
  have : glu1.read (tg.env x) 0 = bv (x 0) := by simp [MLP.read, glu1, Trace.env, tg, bv]
  rw [this]
  cases x 0 <;> norm_num [bv, relu]

theorem tg_isProgram : tg.IsProgram where
  nodup := by decide
  outs_ne := by intro m hm; rw [tg_mem hm]; simp [glu1]
  topo := by
    intro i hi x hx
    simp only [tg, List.length_cons, List.length_nil] at hi
    interval_cases i
    left; simpa [tg, glu1] using hx
  bits := by
    intro x m hm y hy
    rw [tg_mem hm] at hy; simp [glu1] at hy; subst hy
    rw [tg_out]; rcases bv_cases (x 0) with h | h <;> rw [h] <;> simp [IsBit]
  step_margin := by intro x m hm hact; rw [tg_mem hm] at hact; simp [glu1] at hact
  gate_margin := by
    intro x m hm _ j hj
    rw [tg_mem hm] at hj ⊢; simp [glu1] at hj; subst hj
    right
    rw [glu1_pre0, tg_inVals]
    rcases bv_cases (x 0) with h | h <;> rw [h] <;> norm_num [abs_of_pos]
  outputs_ne := by simp [tg]
  outputs_nodup := by decide
  outputs_made := by
    intro y hy; simp [tg] at hy; subst hy; exact ⟨glu1, by simp [tg], by simp [glu1]⟩
  window_dvd := by decide
  window_le := by decide

lemma glu1_slope (u : ℕ → ℝ) (hu : 0 ≤ u 0) : glu1.slope u 0 0 = 4 / 3 * u 0 + 1 / 3 := by
  have hp : 0 < glu1.pre u 0 := by rw [glu1_pre0]; linarith
  have h1 : glu1.pre u (glu1.hidden + 0) = 2 / 3 * u 0 := glu1_pre1 u
  simp only [MLP.slope, sum_range_one, show glu1.hidden = 1 from rfl] at h1 ⊢
  rw [ite_eq_left hp, h1, glu1_pre0]
  simp [glu1]
  ring

/-- With first-order gain `σ = 5/3` and second-order gain `2/3`, `tg` satisfies `RealCore`. -/
theorem tg_realCore : tg.RealCore (10 ^ 5) (5 / 3) where
  gate_margin := by
    intro x m hm _ j hj
    rw [tg_mem hm] at hj ⊢; simp [glu1] at hj; subst hj
    rw [glu1_pre0, tg_inVals]
    rcases bv_cases (x 0) with h | h <;> rw [h] <;> norm_num [abs_of_pos]
  slope := by
    intro x m hm _ i hi
    rw [tg_mem hm] at hi ⊢; simp [glu1] at hi; subst hi
    simp only [MLP.slopeSum, show glu1.ins.length = 1 from rfl, sum_range_one]
    have h0 : 0 ≤ tg.inVals x glu1 0 := by
      rw [tg_inVals]; unfold bv; split_ifs <;> norm_num
    rw [glu1_slope _ h0, tg_inVals]
    rcases bv_cases (x 0) with h | h <;> rw [h] <;> norm_num [abs_of_pos]
  bounded := by
    intro m hm; rw [tg_mem hm]
    refine ⟨?_, ?_, ?_⟩ <;> intro j hj <;> simp [MLP.rows, glu1] at hj ⊢
    all_goals (try interval_cases j)
    all_goals norm_num
  glu_gain := by
    intro m hm _ i hi
    rw [tg_mem hm]
    simp [MLP.gluGain, glu1]
    norm_num

/-- It is not flat: its slope at 1 is 5/3. -/
theorem tg_not_realOK : ¬ tg.RealOK := by
  intro h
  have := h.flat (fun _ => true) glu1 (by simp [tg]) (by simp [glu1]) 0 (by simp [glu1]) 0
    (by simp [glu1])
  have h0 : 0 ≤ tg.inVals (fun _ => true) glu1 0 := by rw [tg_inVals]; norm_num [bv]
  rw [glu1_slope _ h0, tg_inVals] at this
  norm_num [bv] at this

theorem tg_rows : (tokenGrid tg).rows.length = 2 := by decide

theorem tg_depth : (1 + (5 / 3 : ℝ)) ^ ((tokenGrid tg).rows.length - 1) ≤ 10 ^ 4 := by
  rw [tg_rows]; norm_num


theorem tg_codesOK : CodesOK tg codes2 where
  keys_nodup := by decide
  length := by decide
  inj := by decide

theorem tg_posOK (toks : ℕ → ℕ) (htoks : ∀ p, toks p < 2) (p : ℕ) : PosOK tg codes2 toks p := by
  refine ⟨by simp [Trace.window, tg], fun h _ => ?_⟩
  have := htoks (p - h)
  interval_cases toks (p - h) <;> decide

theorem tg_codeIs (toks : ℕ → ℕ) (p : ℕ) : ∃ τ, CodeIs tg codes2 τ (valAt tg codes2 toks p) := by
  have hb := tg_isProgram.bits (windowBits tg codes2 toks p) glu1 (by simp [tg]) 1 (by simp [glu1])
  rcases hb with h | h
  · refine ⟨0, [false], by decide, fun i hi => ?_⟩
    simp [tg] at hi; subst hi
    show (if [false].getD 0 false then (1 : ℝ) else 0) = valAt tg codes2 toks p (.sig 1)
    simp only [valAt, h]; simp
  · refine ⟨1, [true], by decide, fun i hi => ?_⟩
    simp [tg] at hi; subst hi
    show (if [true].getD 0 false then (1 : ℝ) else 0) = valAt tg codes2 toks p (.sig 1)
    simp only [valAt, h]; simp

/-- **The hypotheses of `MainRealDeep` hold**, for `xor3` with `σ = 0`: its depth bound is
`(1 + 0)^2 = 1`. -/
theorem deep_hypotheses_satisfiable :
    ∃ (σ : ℝ) (t : Trace) (codes : List (ℕ × List Bool)) (toks : ℕ → ℕ) (p : ℕ),
      t.IsProgram ∧ t.RealCore (10 ^ 5) σ ∧ 0 ≤ σ ∧
      (1 + σ) ^ ((tokenGrid t).rows.length - 1) ≤ 10 ^ 4 ∧
      CodesOK t codes ∧ PosOK t codes toks p ∧ ∃ τ, CodeIs t codes τ (valAt t codes toks p) :=
  ⟨0, xor3, codes2, fun _ => 0, 2, xor3_isProgram, xor3_realCore, le_rfl,
    by rw [add_zero, one_pow]; norm_num, xor3_codesOK,
    xor3_posOK _ (fun _ => by norm_num) 2 le_rfl, xor3_codeIs _ _⟩

/-- **`MainRealDeep` covers programs that `MainReal` does not**: `tg`, a GLU that is not flat (of
slope 5/3 at 1), satisfies the hypotheses of `MainRealDeep` with `σ = 5/3` (`(8/3)^1 ≤ 10^4`),
at every position and on every sequence of the two tokens, but not `RealOK`. -/
theorem deep_beyond_realOK :
    ∃ (σ : ℝ) (t : Trace) (codes : List (ℕ × List Bool)),
      t.IsProgram ∧ t.RealCore (10 ^ 5) σ ∧ 0 ≤ σ ∧
      (1 + σ) ^ ((tokenGrid t).rows.length - 1) ≤ 10 ^ 4 ∧ CodesOK t codes ∧
      (∀ (toks : ℕ → ℕ), (∀ p, toks p < 2) → ∀ p, PosOK t codes toks p ∧
        ∃ τ, CodeIs t codes τ (valAt t codes toks p)) ∧
      ¬ t.RealOK :=
  ⟨5 / 3, tg, codes2, tg_isProgram, tg_realCore, by norm_num, tg_depth, tg_codesOK,
    fun toks htoks p => ⟨tg_posOK toks htoks p, tg_codeIs toks p⟩, tg_not_realOK⟩

/-! ## `lib.py`'s `and_` on 33 tokens: the relaxed `sums` of `BF16OK` -/

/-- `lib.py`'s `and_` on 33 inputs: `threshold_gate(xs, [1] * 33, 33)`. -/
def and33 : MLP := gate (List.range 33) 33 (List.replicate 33 1) 33

/-- The program `and_` of the bits of the last 33 tokens of one bit. -/
def tAnd : Trace where
  inputs := List.range 33
  mlps := [and33]
  outputs := [33]

lemma tAnd_mem {m : MLP} (hm : m ∈ tAnd.mlps) : m = and33 := by simpa [tAnd] using hm

lemma range33_getD {k : ℕ} (hk : k < 33) : (List.range 33).getD k 0 = k := by
  rw [List.getD_eq_getElem _ _ (by simpa using hk)]; simp

lemma rep33_getD {k : ℕ} (hk : k < 33) : (List.replicate 33 (1 : ℚ)).getD k 0 = 1 := by
  exact List.getD_replicate 1 hk

/-- The number of ones among the 33 input bits. -/
def cnt (x : ℕ → Bool) : ℕ := ∑ k ∈ range 33, if x k then 1 else 0

lemma cnt_le (x : ℕ → Bool) : cnt x ≤ 33 := by
  unfold cnt
  calc (∑ k ∈ range 33, if x k then 1 else 0) ≤ ∑ _k ∈ range 33, 1 :=
        sum_le_sum fun k _ => by split_ifs <;> norm_num
    _ = 33 := by simp

/-- Input `k` of `and33` reads the bit of input `k`. -/
lemma tAnd_inVals (x : ℕ → Bool) {k : ℕ} (hk : k < 33) : tAnd.inVals x and33 k = bv (x k) := by
  have hg : and33.ins.getD k 0 = k := range33_getD hk
  have hn : k ∉ and33.outs := by
    show k ∉ [33]; simp; omega
  have hi : (List.range 33).idxOf k = k := by
    have h := (List.nodup_range (n := 33)).idxOf_getElem k (by rw [List.length_range]; exact hk)
    rwa [List.getElem_range] at h
  simp only [Trace.inVals, MLP.read, hg, Trace.values, tAnd, List.foldl, MLP.apply, hn, ite_false,
    Trace.env, hi, bv]

lemma sum_bv (x : ℕ → Bool) : ∑ k ∈ range 33, bv (x k) = (cnt x : ℝ) := by
  unfold cnt bv; push_cast; rfl

lemma and33_ins : and33.ins = List.range 33 := rfl
lemma and33_outs : and33.outs = [33] := rfl
lemma and33_act : and33.act = .step := rfl
lemma and33_hidden : and33.hidden = 1 := rfl
lemma and33_wi (j k : ℕ) : and33.wi j k = (List.replicate 33 (1 : ℚ)).getD k 0 := rfl
lemma and33_b (j : ℕ) : and33.b j = -33 := rfl
lemma and33_wo (i j : ℕ) : and33.wo i j = 1 := rfl

lemma and33_pre (x : ℕ → Bool) : and33.pre (tAnd.inVals x and33) 0 = (cnt x : ℝ) - 33 := by
  have h : ∀ k ∈ range 33, (and33.wi 0 k : ℝ) * tAnd.inVals x and33 k = bv (x k) := by
    intro k hk
    rw [mem_range] at hk
    rw [tAnd_inVals x hk, and33_wi, rep33_getD hk]
    simp
  simp only [MLP.pre, and33_ins, List.length_range]
  rw [sum_congr rfl h, sum_bv, and33_b]
  push_cast; ring

lemma and33_out (u : ℕ → ℝ) (i : ℕ) : and33.out u i = heaviside (and33.pre u 0) := by
  simp only [MLP.out, MLP.unit, and33_hidden, and33_act, and33_wo, sum_range_one]
  simp

lemma tAnd_value (x : ℕ → Bool) :
    tAnd.values (tAnd.env x) 33 = heaviside (and33.pre (tAnd.inVals x and33) 0) := by
  have h33 : (33 : ℕ) ∈ and33.outs := by rw [and33_outs]; simp
  have hi : and33.outs.idxOf 33 = 0 := by rw [and33_outs]; simp
  simp only [Trace.values, tAnd, List.foldl, MLP.apply, h33, ite_true, hi, and33_out]
  rfl

lemma sum_rep33 : ∑ k ∈ range (List.range 33).length, 128 * (List.replicate 33 (1 : ℚ)).getD k 0
    = 4224 := by
  rw [List.length_range, sum_congr rfl fun k hk => by rw [rep33_getD (mem_range.mp hk)]]
  rw [sum_const, card_range]; norm_num

/-- The compiled form of `and33`. -/
lemma compiled_and33 :
    let c := and33.compiled
    c.n = 2 ∧ c.ins = List.range 33 ∧ c.outs = [33] ∧
    (∀ j k, j < 2 → c.g j k = 64 * (List.replicate 33 (1 : ℚ)).getD k 0) ∧
    (∀ j k, c.v j k = 0) ∧ c.gb 0 = -2032 ∧ c.gb 1 = -2064 ∧
    (∀ j, c.vb j = 1 / 32) ∧ (∀ i, c.wo i 0 = 2) ∧ (∀ i, c.wo i 1 = -2) := by
  obtain ⟨hn, hins, houts, hg, hv, hgb0, hgb1, hvb, hwo0, hwo1⟩ :=
    compiled_gate (List.range 33) 33 (List.replicate 33 1) 33
  refine ⟨hn, hins, houts, hg, hv, ?_, ?_, hvb, hwo0, hwo1⟩
  · show (gate (List.range 33) 33 (List.replicate 33 1) 33).compiled.gb 0 = -2032
    rw [hgb0, sum_rep33]; norm_num
  · show (gate (List.range 33) 33 (List.replicate 33 1) 33).compiled.gb 1 = -2064
    rw [hgb1, sum_rep33]; norm_num

lemma and33_gateAt (x : ℕ → Bool) (j : ℕ) (hj : j < 2) :
    and33.compiled.gateAt (tAnd.inVals x and33) j =
      128 * (cnt x : ℝ) - 2112 + (and33.compiled.gb j : ℝ) := by
  obtain ⟨_, hins, _, hg, _⟩ := compiled_and33
  simp only [Gated.gateAt, hins, List.length_range]
  have h : ∀ k ∈ range 33, (and33.compiled.g j k : ℝ) * (2 * tAnd.inVals x and33 k - 1) =
      128 * bv (x k) - 64 := by
    intro k hk
    rw [mem_range] at hk
    rw [hg j k hj, rep33_getD hk, tAnd_inVals x hk]
    push_cast; ring
  rw [sum_congr rfl h, sum_sub_distrib, ← mul_sum, sum_bv, sum_const, card_range, nsmul_eq_mul]
  push_cast; ring

lemma and33_valAt (x : ℕ → Bool) (j : ℕ) :
    and33.compiled.valAt (tAnd.inVals x and33) j = 1 / 32 := by
  obtain ⟨_, hins, _, _, hv, _, _, hvb, _⟩ := compiled_and33
  simp only [Gated.valAt, hins, List.length_range, hv, hvb]
  push_cast; simp

theorem tAnd_isProgram : tAnd.IsProgram where
  nodup := by decide
  outs_ne := by intro m hm; rw [tAnd_mem hm]; simp [and33, gate]
  topo := by
    intro i hi x hx
    simp only [tAnd, List.length_cons, List.length_nil] at hi
    interval_cases i
    left; simpa [tAnd, and33, gate] using hx
  bits := by
    intro x m hm y hy
    rw [tAnd_mem hm] at hy; simp [and33, gate] at hy; subst hy
    rw [tAnd_value]; exact heaviside_bit _
  step_margin := by
    intro x m hm _ j hj
    rw [tAnd_mem hm] at hj ⊢
    have hj0 : j = 0 := by simpa [and33, gate] using hj
    subst hj0
    rw [and33_pre]
    have hc := cnt_le x
    rcases (show cnt x = 33 ∨ cnt x ≤ 32 by omega) with h | h
    · left; rw [h]; norm_num
    · right
      have : (cnt x : ℝ) ≤ 32 := by exact_mod_cast h
      linarith
  gate_margin := by intro x m hm hact; rw [tAnd_mem hm] at hact; simp [and33, gate] at hact
  outputs_ne := by simp [tAnd]
  outputs_nodup := by decide
  outputs_made := by
    intro y hy; simp [tAnd] at hy; subst hy; exact ⟨and33, by simp [tAnd], by simp [and33, gate]⟩
  window_dvd := by decide
  window_le := by decide

lemma gOn_and33 (j : ℕ) (hj : j < 2) (s : Sig) :
    and33.compiled.gOn j s = if s < 33 then 64 else 0 := by
  obtain ⟨_, hins, _, hg, _⟩ := compiled_and33
  have h : ∀ k ∈ range 33, (if (List.range 33).getD k 0 = s then and33.compiled.g j k else 0) =
      if k = s then 64 else 0 := by
    intro k hk
    rw [mem_range] at hk
    rw [range33_getD hk, hg j k hj, rep33_getD hk, mul_one]
  simp only [Gated.gOn, hins, List.length_range]
  rw [sum_congr rfl h, sum_ite_eq']
  simp only [mem_range]

theorem tAnd_weights : ∀ m ∈ tAnd.mlps, ∀ j < m.compiled.n,
    (∀ s, IsBF16 (m.compiled.gOn j s)) ∧ (∀ s, IsBF16 (m.compiled.vOn j s)) ∧
      IsBF16 (m.compiled.gb j) ∧ IsBF16 (m.compiled.vb j) ∧
        ∀ i < m.outs.length, IsBF16 (m.compiled.wo i j) := by
  intro m hm j hj
  rw [tAnd_mem hm] at hj ⊢
  obtain ⟨hn, _, _, _, _, hgb0, hgb1, hvb, hwo0, hwo1⟩ := compiled_and33
  rw [hn] at hj
  refine ⟨fun s => ?_, fun s => ?_, ?_, ?_, ?_⟩
  · rw [gOn_and33 j hj s]; split_ifs <;> bf16_num
  · rw [show and33.compiled.vOn j s = 0 from vOn_gate _ _ _ _ j s]; bf16_num
  · interval_cases j
    · rw [hgb0]; bf16_num
    · rw [hgb1]; bf16_num
  · rw [hvb]; bf16_num
  · intro i _
    interval_cases j
    · rw [hwo0]; bf16_num
    · rw [hwo1]; bf16_num

/-- On bits, a gate of `and33` is `128 c - 4144` (`u+`) or `128 c - 4176` (`u-`), `c` the number
of ones: at most `-48`, so at most 0, unless `c = 33`, where it is 80 or 48, and its product with
the value `1/32` is `5/2` or `3/2`, all bf16 numbers. -/
theorem tAnd_sums : ∀ (x : ℕ → Bool), ∀ m ∈ tAnd.mlps, ∀ j < m.compiled.n,
    (-maxBF16 ≤ m.compiled.gateAt (tAnd.inVals x m) j ∧ m.compiled.gateAt (tAnd.inVals x m) j ≤ 0) ∨
    (IsBF16 (m.compiled.gateAt (tAnd.inVals x m) j) ∧ IsBF16 (m.compiled.valAt (tAnd.inVals x m) j) ∧
      IsBF16 (relu (m.compiled.gateAt (tAnd.inVals x m) j) * m.compiled.valAt (tAnd.inVals x m) j)) := by
  intro x m hm j hj
  rw [tAnd_mem hm] at hj ⊢
  obtain ⟨hn, _, _, _, _, hgb0, hgb1, _⟩ := compiled_and33
  rw [hn] at hj
  rw [and33_gateAt x j hj, and33_valAt x j]
  have hc := cnt_le x
  have hmax : (10 : ℝ) ^ 4 ≤ maxBF16 := by unfold maxBF16; norm_num
  rcases (show cnt x = 33 ∨ cnt x ≤ 32 by omega) with h | h
  · right
    rw [h]
    interval_cases j
    · rw [hgb0]; refine ⟨by norm_num; bf16_num, by bf16_num, ?_⟩; norm_num [relu]; bf16_num
    · rw [hgb1]; refine ⟨by norm_num; bf16_num, by bf16_num, ?_⟩; norm_num [relu]; bf16_num
  · left
    have h' : (cnt x : ℝ) ≤ 32 := by exact_mod_cast h
    have h0 : (0 : ℝ) ≤ cnt x := Nat.cast_nonneg _
    interval_cases j
    · rw [hgb0]; push_cast; constructor <;> linarith
    · rw [hgb1]; push_cast; constructor <;> linarith

theorem tAnd_bf16OK : tAnd.BF16OK where
  weights := tAnd_weights
  sums := tAnd_sums

theorem tAnd_width : (tokenGrid tAnd).width = 35 := by decide +kernel

theorem tAnd_widthOK : WidthOK tAnd := by
  unfold WidthOK
  rw [tAnd_width]
  have h : Real.sqrt 35 ≤ 6 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  unfold maxBF16
  push_cast
  nlinarith

theorem tAnd_small : 2 * Real.sqrt (tokenGrid tAnd).width + tAnd.outputs.length ≤ 250 := by
  rw [tAnd_width]
  have h : Real.sqrt 35 ≤ 6 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  simp [tAnd]
  linarith

theorem tAnd_realOK : tAnd.RealOK := by
  have hs : ∀ m ∈ tAnd.mlps, m.act = .step := fun m hm => by rw [tAnd_mem hm]; rfl
  refine ⟨fun _ m hm hact => absurd (hs m hm) hact, fun _ m hm hact => absurd (hs m hm) hact,
    fun m hm => ?_, fun m hm hact => by rw [hs m hm] at hact; exact absurd hact (by decide)⟩
  rw [tAnd_mem hm]
  refine ⟨fun j hj => ?_, fun j hj => ?_, fun i hi => ?_⟩
  · simp only [and33_ins, List.length_range, and33_wi]
    rw [sum_congr rfl fun k hk => by rw [rep33_getD (mem_range.mp hk)]]
    norm_num
  · rw [and33_b]; norm_num
  · simp only [and33_hidden, and33_wo, sum_range_one]; norm_num

theorem tAnd_codesOK : CodesOK tAnd codes2 where
  keys_nodup := by decide
  length := by decide
  inj := by decide

theorem tAnd_posOK (toks : ℕ → ℕ) (htoks : ∀ p, toks p < 2) (p : ℕ) (hp : 32 ≤ p) :
    PosOK tAnd codes2 toks p := by
  refine ⟨by simp [Trace.window, tAnd]; omega, fun h _ => ?_⟩
  have := htoks (p - h)
  interval_cases toks (p - h) <;> decide

theorem tAnd_codeIs (toks : ℕ → ℕ) (p : ℕ) :
    ∃ τ, CodeIs tAnd codes2 τ (valAt tAnd codes2 toks p) := by
  have hb := tAnd_isProgram.bits (windowBits tAnd codes2 toks p) and33 (by simp [tAnd]) 33
    (by rw [and33_outs]; simp)
  rcases hb with h | h
  · refine ⟨0, [false], by decide, fun i hi => ?_⟩
    simp [tAnd] at hi; subst hi
    show (if [false].getD 0 false then (1 : ℝ) else 0) = valAt tAnd codes2 toks p (.sig 33)
    simp only [valAt, h]; simp
  · refine ⟨1, [true], by decide, fun i hi => ?_⟩
    simp [tAnd] at hi; subst hi
    show (if [true].getD 0 false then (1 : ℝ) else 0) = valAt tAnd codes2 toks p (.sig 33)
    simp only [valAt, h]; simp

theorem not_bf16_4144 : ¬ IsBF16 (-4144) := by
  rintro ⟨m, e, h, hm, he1, he2⟩
  rcases le_or_gt 5 e with h5 | h5
  · obtain ⟨k, rfl⟩ : ∃ k : ℕ, e = 5 + k := ⟨(e - 5).toNat, by omega⟩
    have h' : ((m * 2 ^ k * 32 : ℤ) : ℝ) = -4144 := by
      rw [h, zpow_add₀ two_ne_zero]; push_cast; rw [zpow_natCast]; norm_num; ring
    have hz : (m * 2 ^ k * 32 : ℤ) = -4144 := by exact_mod_cast h'
    generalize m * 2 ^ k = a at hz
    omega
  · obtain ⟨k, rfl⟩ : ∃ k : ℕ, e = 4 - k := ⟨(4 - e).toNat, by omega⟩
    have h' : ((m * 16 : ℤ) : ℝ) = ((-4144 * 2 ^ k : ℤ) : ℝ) := by
      have : (m : ℝ) * 2 ^ ((4 : ℤ) - k) * 2 ^ (k : ℤ) = -4144 * 2 ^ (k : ℤ) := by rw [← h]
      rw [mul_assoc, ← zpow_add₀ two_ne_zero] at this
      push_cast; norm_num at this ⊢; linarith
    have hz : m * 16 = -4144 * 2 ^ k := by exact_mod_cast h'
    have hk : (1 : ℤ) ≤ 2 ^ k := one_le_pow₀ (by norm_num)
    have : |m| < 256 := hm
    rw [abs_lt] at this
    nlinarith

/-- On the all-zero input the gate of `u+` is `-4144`, not a bf16 number: `tAnd` needs the
relaxed `sums` (a gate at most 0 need not be exact), not every gate a bf16 number. -/
theorem tAnd_gate_zero :
    and33.compiled.gateAt (tAnd.inVals (fun _ => false) and33) 0 = -4144 ∧
      ¬ IsBF16 (and33.compiled.gateAt (tAnd.inVals (fun _ => false) and33) 0) := by
  have hc : cnt (fun _ => false) = 0 := by simp [cnt]
  obtain ⟨_, _, _, _, _, hgb0, _⟩ := compiled_and33
  have h : and33.compiled.gateAt (tAnd.inVals (fun _ => false) and33) 0 = -4144 := by
    rw [and33_gateAt _ 0 (by norm_num), hc, hgb0]; norm_num
  exact ⟨h, by rw [h]; exact not_bf16_4144⟩

/-- **The relaxed `sums` covers `lib.py`'s `and_` on 33 one-bit tokens**: `tAnd` (one threshold
gate, 33 inputs of weight 1, threshold 33, window 33) satisfies every hypothesis of `MainReal` and
`MainBF16` (`IsProgram`, `RealOK`, `BF16OK`, `CodesOK`, `PosOK` from position 32 on, `WidthOK`, and
`2 sqrt 35 + 1 ≤ 250`) with some token's code the trace's output, though its gate on the all-zero
input, `-4144`, is not a bf16 number. -/
theorem and33_covered :
    tAnd.IsProgram ∧ tAnd.RealOK ∧ tAnd.BF16OK ∧ CodesOK tAnd codes2 ∧ WidthOK tAnd ∧
      2 * Real.sqrt (tokenGrid tAnd).width + tAnd.outputs.length ≤ 250 ∧
      (∀ (toks : ℕ → ℕ), (∀ p, toks p < 2) → ∀ p, 32 ≤ p → PosOK tAnd codes2 toks p ∧
        ∃ τ, CodeIs tAnd codes2 τ (valAt tAnd codes2 toks p)) ∧
      ¬ ∀ (x : ℕ → Bool), ∀ m ∈ tAnd.mlps, ∀ j < m.compiled.n,
        IsBF16 (m.compiled.gateAt (tAnd.inVals x m) j) :=
  ⟨tAnd_isProgram, tAnd_realOK, tAnd_bf16OK, tAnd_codesOK, tAnd_widthOK, tAnd_small,
    fun toks htoks p hp => ⟨tAnd_posOK toks htoks p hp, tAnd_codeIs toks p⟩,
    fun h => tAnd_gate_zero.2 (h _ and33 (by simp [tAnd]) 0 (by rw [compiled_and33.1]; norm_num))⟩

end Reifier.NV

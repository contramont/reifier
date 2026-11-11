import ReifierProofs.BF16

/-!
# Well-formed traces on bits

`WellFormed` is what a trace must be for its layers to compute it at all; `IsProgram` adds what
`transformer` needs of a program on tokens. Each condition excludes a counterexample, logged in
`COUNTEREXAMPLES.md`. `RealOK` and `BF16OK` are the numeric conditions of the two theorems: the
forward pass in real arithmetic, and in bf16.
-/

namespace Reifier

open Finset

noncomputable section

namespace Trace

/-- The values of the inputs of `m` on the input bits `x`. -/
def inVals (t : Trace) (x : ℕ → Bool) (m : MLP) : ℕ → ℝ := m.read (t.values (t.env x))

/-- A well-formed trace on bits. -/
structure WellFormed (t : Trace) : Prop where
  /-- no two signals share a name (`check_trace` asserts it) -/
  nodup : (t.inputs ++ t.mlps.flatMap MLP.outs).Nodup
  /-- every MLP gives a signal (in Python, `wo` is at least 2-d) -/
  outs_ne : ∀ m ∈ t.mlps, m.outs ≠ []
  /-- each MLP reads only inputs and the outputs of MLPs before it -/
  topo : ∀ (i : ℕ) (hi : i < t.mlps.length), ∀ x ∈ t.mlps[i].ins,
    x ∈ t.inputs ∨ ∃ (j : ℕ) (hj : j < i), x ∈ (t.mlps[j]'(by omega)).outs
  /-- on bits, every signal is a bit -/
  bits : ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, ∀ y ∈ m.outs, IsBit (t.values (t.env x) y)
  /-- PATCH (CE-1): a step unit's sum is never strictly between -3/4 and 0 (with integer weights
  it is at least 0 or at most -1). `relu(4 z + 5/2) - relu(4 z + 3/2)` is `[z >= 0]` but on
  `(-5/8, 0)`, and at most -3/4 keeps both its gates at least 1/2 from 0, where `gelu` is `relu` -/
  step_margin : ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act = .step → ∀ j < m.hidden,
    0 ≤ m.pre (t.inVals x m) j ∨ m.pre (t.inVals x m) j ≤ -3 / 4
  /-- PATCH (CE-2): a `relu` or GLU unit's gate is 0 or at least 1/2 from 0, as `emit.py` says
  of every unit made there and in `lib.py`: `gelu` is `relu` there, and only there -/
  gate_margin : ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step → ∀ j < m.hidden,
    m.pre (t.inVals x m) j = 0 ∨ 1 / 2 ≤ |m.pre (t.inVals x m) j|

/-- A program on tokens: a well-formed trace from the bits of the last `window` tokens to the
bits of the next. -/
structure IsProgram (t : Trace) : Prop extends t.WellFormed where
  /-- a code has at least one bit (`len(t.inputs) // bits`) -/
  outputs_ne : t.outputs ≠ []
  /-- PATCH (CE-3): outputs are distinct, or `token_grid` moves one output over another -/
  outputs_nodup : t.outputs.Nodup
  /-- PATCH (CE-4): each output is made by an MLP, not an input, or `token_grid` moves an input
  into the token's code, where the attention adds to it -/
  outputs_made : ∀ y ∈ t.outputs, ∃ m ∈ t.mlps, y ∈ m.outs
  /-- the inputs are the bits of whole tokens -/
  window_dvd : t.outputs.length ∣ t.inputs.length
  /-- PATCH (CE-5): a head finds only a token within `CONTEXT` -/
  window_le : t.window ≤ CONTEXT

end Trace

/-- The bits of each token: one code per token, a code per token at most once, all of `bits`
bits. -/
structure CodesOK (t : Trace) (codes : List (ℕ × List Bool)) : Prop where
  keys_nodup : (codes.map Prod.fst).Nodup
  length : ∀ c ∈ codes, c.2.length = t.outputs.length
  /-- PATCH (CE-6): two tokens of one code tie for the largest logit -/
  inj : ∀ a ∈ codes, ∀ b ∈ codes, a.2 = b.2 → a.1 = b.1

namespace MLP

/-- The rows of `wi`: twice the units if gated. -/
def rows (m : MLP) : ℕ := if m.act = .reglu then 2 * m.hidden else m.hidden

/-- The slope of output `i` along input `k`, at the input values `u`, of a `relu` or GLU MLP.
Zero is flatness: then a GLU passes on only the squares of the errors it reads. -/
def slope (m : MLP) (u : ℕ → ℝ) (i k : ℕ) : ℝ :=
  ∑ j ∈ range m.hidden,
    if 0 < m.pre u j then
      (m.wo i j : ℝ) *
        (match m.act with
          | .reglu => (m.wi j k : ℝ) * m.pre u (m.hidden + j) + m.pre u j * m.wi (m.hidden + j) k
          | _ => (m.wi j k : ℝ))
    else 0

/-- Every weight sum of `m` is at most `B`: each row of `wi` in absolute value, each bias, each
row of `wo`. -/
def Bounded (m : MLP) (B : ℚ) : Prop :=
  (∀ j < m.rows, ∑ k ∈ range m.ins.length, |m.wi j k| ≤ B) ∧ (∀ j < m.rows, |m.b j| ≤ B) ∧
    (∀ i < m.outs.length, ∑ j ∈ range m.hidden, |m.wo i j| ≤ B)

/-- The second-order gain of output `i` of a GLU: what an error `e` in its inputs can become, over
`e^2 / 4`: `Σ_j |wo i j| (Σ_k |gate j k|) (Σ_k |value j k|)`. -/
def gluGain (m : MLP) (i : ℕ) : ℚ :=
  ∑ j ∈ range m.hidden, |m.wo i j| * (∑ k ∈ range m.ins.length, |m.wi j k|) *
    (∑ k ∈ range m.ins.length, |m.wi (m.hidden + j) k|)

/-- The first-order gain of output `i` at the input values `u`: `Σ_k |slope i k|`. -/
def slopeSum (m : MLP) (u : ℕ → ℝ) (i : ℕ) : ℝ := ∑ k ∈ range m.ins.length, |m.slope u i k|

end MLP

/-- PATCH (CE-2b): in real arithmetic, a `relu` or GLU gate at 0 passes the error it reads on at
the first order, through `gelu'(0) = 1/2`: its gates must be at least 1/2 from 0. -/
def Trace.GateMargin (t : Trace) : Prop :=
  ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step → ∀ j < m.hidden,
    1 / 2 ≤ |m.pre (t.inVals x m) j|

/-- What the theorem in real arithmetic needs: gates away from 0, flat GLUs, and weights of a size
the attention's error cannot grow past. -/
structure Trace.RealOK (t : Trace) : Prop where
  gate_margin : t.GateMargin
  /-- PATCH (CE-7): without it a GLU multiplies the attention's error at every depth -/
  flat : ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step →
    ∀ i < m.outs.length, ∀ k < m.ins.length, m.slope (t.inVals x m) i k = 0
  /-- PATCH (CE-8): a weight of 10^13 turns the attention's error of 10^-13 into a wrong bit -/
  bounded : ∀ m ∈ t.mlps, m.Bounded (2 ^ 16)
  /-- PATCH (CE-8): the squares of the errors a GLU passes on stay below the error -/
  glu_gain : ∀ m ∈ t.mlps, m.act = .reglu → ∀ i < m.outs.length, m.gluGain i ≤ 10 ^ 8

/-- The same, with GLUs of first-order gain at most `σ` in place of flat ones, and second-order gain
at most `G`: what one layer needs to hold a stream within `(1 + σ)` times the error it reads. -/
structure Trace.RealCore (t : Trace) (G σ : ℝ) : Prop where
  gate_margin : t.GateMargin
  slope : ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, m.act ≠ .step →
    ∀ i < m.outs.length, m.slopeSum (t.inVals x m) i ≤ σ
  bounded : ∀ m ∈ t.mlps, m.Bounded (2 ^ 16)
  glu_gain : ∀ m ∈ t.mlps, m.act = .reglu → ∀ i < m.outs.length, (m.gluGain i : ℝ) ≤ G

namespace Gated

/-- The weight of gate row `j` on signal `s`: an input read twice adds up. -/
def gOn (c : Gated) (j : ℕ) (s : Sig) : ℚ :=
  ∑ k ∈ range c.ins.length, if c.ins.getD k 0 = s then c.g j k else 0

/-- The weight of value row `j` on signal `s`. -/
def vOn (c : Gated) (j : ℕ) (s : Sig) : ℚ :=
  ∑ k ∈ range c.ins.length, if c.ins.getD k 0 = s then c.v j k else 0

/-- The gate of unit `j` when its inputs hold the bits `u` as -1 or 1 and `ONE` as 1. -/
def gateAt (c : Gated) (u : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range c.ins.length, (c.g j k : ℝ) * (2 * u k - 1) + c.gb j

/-- The value of unit `j` there. -/
def valAt (c : Gated) (u : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range c.ins.length, (c.v j k : ℝ) * (2 * u k - 1) + c.vb j

end Gated

/-- What the theorem in bf16 needs: every weight of every unit is a bf16 number, and on bits so is
each gate that is above 0, its value and their product. (`gemma.py` asserts instead that the
weights are bf16 numbers and that no gate can sum to 4096 or more, which for gates of integer
weights, whose sums are multiples of 16, makes those sums bf16 numbers.) -/
structure Trace.BF16OK (t : Trace) : Prop where
  /-- PATCH (CE-9): bf16 rounds a weight it cannot hold -/
  weights : ∀ m ∈ t.mlps, ∀ j < m.compiled.n,
    (∀ s, IsBF16 (m.compiled.gOn j s)) ∧ (∀ s, IsBF16 (m.compiled.vOn j s)) ∧
      IsBF16 (m.compiled.gb j) ∧ IsBF16 (m.compiled.vb j) ∧
        ∀ i < m.outs.length, IsBF16 (m.compiled.wo i j)
  /-- PATCH (CE-10): bf16 rounds a sum it cannot hold, and the gates `u+` and `u-` of a step
  differ by less than they should. Only a gate that passes something must be exact: one at most
  0 passes 0, however bf16 rounds it -/
  sums : ∀ (x : ℕ → Bool), ∀ m ∈ t.mlps, ∀ j < m.compiled.n,
    (-maxBF16 ≤ m.compiled.gateAt (t.inVals x m) j ∧ m.compiled.gateAt (t.inVals x m) j ≤ 0) ∨
    (IsBF16 (m.compiled.gateAt (t.inVals x m) j) ∧ IsBF16 (m.compiled.valAt (t.inVals x m) j) ∧
      IsBF16 (relu (m.compiled.gateAt (t.inVals x m) j) * m.compiled.valAt (t.inVals x m) j))

end

end Reifier

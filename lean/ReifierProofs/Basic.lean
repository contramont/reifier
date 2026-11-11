import Mathlib

/-!
# Stage 1: traces (`src/reifier/trace.py`)

A trace is a flat list of MLPs. Signals are named by natural numbers rather than strings; a
well-formed trace never gives two signals the same name (`check_trace` asserts the same of the
Python names). Weights are rationals: every float is one.

Python's `MLP.__call__` is `ACTIVATIONS[act](xs @ wi.T + b) @ wo.T`, with
`step z = (z >= 0) * 1.0` and `reglu z = relu(z[:n]) * z[n:]`. We cover the activations that
traces use: `step` (threshold gates), `relu`, and the gated `reglu` (`glu`, and so `lib.py`).
-/

namespace Reifier

open Finset

/-- A signal: a wire, named by a natural number. -/
abbrev Sig := ℕ

/-- The activations of a trace's MLPs that this model covers. -/
inductive Act
  | step
  | relu
  | reglu
  deriving DecidableEq, Repr

/-- `wo @ act(wi @ x + b)` (Python `MLP`). Of its `hidden` units, unit `j` is row `j` of `wi`
(and, if gated, row `hidden + j`, the value its gate multiplies) and column `j` of `wo`.
`wi j k` is the weight of row `j` on input `k`; `wo i j` that of unit `j` in output `i`. -/
structure MLP where
  ins : List Sig
  outs : List Sig
  act : Act
  hidden : ℕ
  wi : ℕ → ℕ → ℚ
  b : ℕ → ℚ
  wo : ℕ → ℕ → ℚ

noncomputable section

/-- `np.maximum(z, 0)` -/
def relu (x : ℝ) : ℝ := max x 0

/-- Python's `step`: `(z >= 0) * 1.0` -/
def heaviside (x : ℝ) : ℝ := if 0 ≤ x then 1 else 0

namespace MLP

/-- Row `j` of `wi @ u + b`, where `u k` is the value of input `k`. -/
def pre (m : MLP) (u : ℕ → ℝ) (j : ℕ) : ℝ :=
  ∑ k ∈ range m.ins.length, (m.wi j k : ℝ) * u k + m.b j

/-- Hidden unit `j`, after the activation. -/
def unit (m : MLP) (u : ℕ → ℝ) (j : ℕ) : ℝ :=
  match m.act with
  | .step => heaviside (m.pre u j)
  | .relu => relu (m.pre u j)
  | .reglu => relu (m.pre u j) * m.pre u (m.hidden + j)

/-- Output `i`. -/
def out (m : MLP) (u : ℕ → ℝ) (i : ℕ) : ℝ :=
  ∑ j ∈ range m.hidden, (m.wo i j : ℝ) * m.unit u j

/-- The values of its inputs, under a valuation of the signals. -/
def read (m : MLP) (v : Sig → ℝ) : ℕ → ℝ := fun k => v (m.ins.getD k 0)

/-- The valuation `v` with the outputs of `m` set (Python: `v.update(zip(m.outputs, ...))`). -/
def apply (m : MLP) (v : Sig → ℝ) : Sig → ℝ :=
  fun s => if s ∈ m.outs then m.out (m.read v) (m.outs.idxOf s) else v s

end MLP

end

/-- What a function computes: its inputs, its MLPs in the order they were made, its outputs. -/
structure Trace where
  inputs : List Sig
  mlps : List MLP
  outputs : List Sig

noncomputable section

namespace Trace

/-- The value of every signal (Python `Trace.values`), from a valuation `x` of the inputs. -/
def values (t : Trace) (x : Sig → ℝ) : Sig → ℝ :=
  t.mlps.foldl (fun v m => m.apply v) x

/-- The valuation that gives input `k` the bit `x k`. -/
def env (t : Trace) (x : ℕ → Bool) : Sig → ℝ :=
  fun s => if x (t.inputs.idxOf s) then 1 else 0

/-- The value of output `i` on the input bits `x`. -/
def outVal (t : Trace) (x : ℕ → Bool) (i : ℕ) : ℝ :=
  t.values (t.env x) (t.outputs.getD i 0)

/-- How many tokens the program reads: `len(t.inputs) // bits` (Python `transformer`). -/
def window (t : Trace) : ℕ := t.inputs.length / t.outputs.length

end Trace

/-- A real number that is a bit. -/
def IsBit (r : ℝ) : Prop := r = 0 ∨ r = 1

end

end Reifier

module

public import Mathlib

@[expose] public section

/-!
# Stage 1, the trace: `src/reifier/trace.py`

A trace is a flat list of MLPs on named signals. This file holds what a trace *is* and what it
computes: the activations, `MLP` and `MLP.eval` (Python's `MLP.__call__`), `Trace`,
`Trace.values` and `Trace.call` (`Trace.__call__`). The part of `trace.py` that runs a Python
function to make a trace (`mlp`, `circuit`, `trace`, `needed`) has no counterpart here: a trace
is taken as it is.

Numbers are reals. Weights are of any field `K`, so that the same definitions also run on
rationals, where `test/` compares them with what the Python makes; the theorems are about `ℝ`.
A vector is a list, and a matrix the list of its rows, as numpy holds them.
-/

namespace Reifier

/-- The names of `ACTIVATIONS`. -/
inductive Act
  | step | relu | silu | gelu | reglu | swiglu | geglu
  deriving DecidableEq, Repr

/-- `GATED`: the gated activations, and what gates each. -/
def Act.gate : Act → Option Act
  | .reglu => some .relu
  | .swiglu => some .silu
  | .geglu => some .gelu
  | _ => none

/-- `activation in GATED` -/
def Act.gated (a : Act) : Bool := a.gate.isSome

/-- `"step": lambda z: (z >= 0) * 1.0` -/
noncomputable def stepF (z : ℝ) : ℝ := if 0 ≤ z then 1 else 0

/-- `"relu": lambda z: np.maximum(z, 0)` -/
noncomputable def reluF (z : ℝ) : ℝ := max z 0

/-- `"silu": lambda z: z / 2 * (1 + np.tanh(z / 2))`, which is `z sigmoid(z)` -/
noncomputable def siluF (z : ℝ) : ℝ := z / 2 * (1 + Real.tanh (z / 2))

/-- `"gelu": lambda z: z / 2 * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z**3)))` -/
noncomputable def geluF (z : ℝ) : ℝ :=
  z / 2 * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (z + 0.044715 * z ^ 3)))

/-- What an activation does to one number: itself, or for a gated one, what gates it. -/
noncomputable def Act.scalar : Act → ℝ → ℝ
  | .step => stepF
  | .relu | .reglu => reluF
  | .silu | .swiglu => siluF
  | .gelu | .geglu => geluF

/-- `gated(act)`: act of the left half of `z`, times the right half. -/
def gatedF (act : ℝ → ℝ) (z : List ℝ) : List ℝ :=
  List.zipWith (fun g v => act g * v) (z.take (z.length / 2)) (z.drop (z.length / 2))

/-- `ACTIVATIONS[a]`, on a vector. -/
noncomputable def Act.apply (a : Act) (z : List ℝ) : List ℝ :=
  if a.gated then gatedF a.scalar z else z.map a.scalar

variable {K : Type}

/-- The dot product of two vectors. -/
def dot [Mul K] [Add K] [Zero K] (u v : List K) : K := (List.zipWith (· * ·) u v).sum

/-- `class MLP`: `outputs = wo @ act(wi @ inputs + b)`, with a name for itself and for each
hidden unit. Of its `n` hidden units, unit `j` is row `j` of `wi` (and, if `act` is gated, row
`n + j`, the value that its gate multiplies) and column `j` of `wo`. -/
structure MLP (K : Type) where
  name : String
  inputs : List String
  outputs : List String
  hidden : List String
  wi : List (List K)
  b : List K
  activation : Act
  wo : List (List K)
  deriving DecidableEq, Repr

/-- `xs @ wi.T + b`: what the activation of `m` is given, for inputs `xs`. -/
def MLP.pre [Mul K] [Add K] [Zero K] (m : MLP K) (xs : List K) : List K :=
  List.zipWith (· + ·) (m.wi.map (dot · xs)) m.b

/-- `MLP.__call__`: the outputs for inputs `xs`,
`ACTIVATIONS[activation](xs @ wi.T + b) @ wo.T`. -/
noncomputable def MLP.eval (m : MLP ℝ) (xs : List ℝ) : List ℝ :=
  m.wo.map (dot · (m.activation.apply (m.pre xs)))

/-- `class Trace`: what a function computes: its inputs, its MLPs in the order they were made
(each after those it reads), and its outputs, all by name. -/
structure Trace (K : Type) where
  name : String
  inputs : List String
  outputs : List String
  mlps : List (MLP K)
  deriving DecidableEq, Repr

/-- `v.update(zip(names, values))`, for a dict `v` from names to numbers. The dict is a
function here, 0 at a name that is not a key, where Python would raise. -/
def assign (v : String → ℝ) (names : List String) (values : List ℝ) : String → ℝ :=
  (names.zip values).foldl (fun v p => Function.update v p.1 p.2) v

/-- `Trace.values`: the value of every signal, on the inputs `xs`. -/
noncomputable def Trace.values (t : Trace ℝ) (xs : List ℝ) : String → ℝ :=
  t.mlps.foldl (fun v m => assign v m.outputs (m.eval (m.inputs.map v)))
    (assign (fun _ => 0) t.inputs xs)

/-- `Trace.__call__`: the outputs on the inputs `xs`. -/
noncomputable def Trace.call (t : Trace ℝ) (xs : List ℝ) : List ℝ :=
  t.outputs.map (t.values xs)

/-- Every signal of a trace: its inputs, then what its MLPs make, in order. -/
def Trace.signals (t : Trace K) : List String :=
  t.inputs ++ t.mlps.flatMap (·.outputs)

end Reifier

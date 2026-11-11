import ReifierProofs.Grid

/-!
# Stage 3: the weights (`src/reifier/emit.py`)

The transformer's layers are `emit(t, grid, "geglu", skip=True, norm=True)`, and that is the
configuration modelled here: `x = x + layer(rmsnorm(x))`, each layer an MLP on features of
`gelu`-gated units, with no biases (biases are weights on `ONE`).

A layer is a list of hidden units. Unit `u` reads the normed stream `z` through its gate row and
its value row and writes `u.out f` times `gelu(gate . z) * (val . z)` into each feature `f`:
the rows of Python's `w_in` (gates, then values) and the columns of `w_out`, unit by unit, in
the same order (`ONE`, the units of each MLP made at this depth, then the erasures).
-/

namespace Reifier

open Finset

/-- A gated MLP `wo @ (act(g @ x + gb) * (v @ x + vb))`, of `n` units. -/
structure Gated where
  ins : List Sig
  outs : List Sig
  n : ℕ
  g : ℕ → ℕ → ℚ
  gb : ℕ → ℚ
  v : ℕ → ℕ → ℚ
  vb : ℕ → ℚ
  wo : ℕ → ℕ → ℚ

namespace MLP

/-- `convert` as far as `reglu`. A step gate `[z >= 0]` becomes `relu(4 z + 5/2) - relu(4 z + 3/2)`
(units `u+` then `u-`); `relu(z)` is `relu(z) * 1`; a `reglu` MLP's halves are its gates and
values. -/
def toReglu (m : MLP) : Gated :=
  match m.act with
  | .step =>
    { ins := m.ins, outs := m.outs, n := 2 * m.hidden
      g := fun j k => 4 * m.wi (if j < m.hidden then j else j - m.hidden) k
      gb := fun j => if j < m.hidden then 4 * m.b j + 2 + 1 / 2 else 4 * m.b (j - m.hidden) + 2 - 1 / 2
      v := fun _ _ => 0
      vb := fun _ => 1
      wo := fun i j => if j < m.hidden then m.wo i j else -m.wo i (j - m.hidden) }
  | .relu =>
    { ins := m.ins, outs := m.outs, n := m.hidden
      g := m.wi, gb := m.b, v := fun _ _ => 0, vb := fun _ => 1, wo := m.wo }
  | .reglu =>
    { ins := m.ins, outs := m.outs, n := m.hidden
      g := m.wi, gb := m.b
      v := fun j k => m.wi (m.hidden + j) k, vb := fun j => m.b (m.hidden + j)
      wo := m.wo }

end MLP

namespace Gated

/-- `SHARPNESS["geglu"]` -/
def sharpness : ℚ := 32

/-- `relu(g) * v` as `gelu(s g) * v / s`: gates `s` times larger, values `s` times smaller. -/
def sharpen (s : ℚ) (m : Gated) : Gated :=
  { m with g := fun j k => s * m.g j k, gb := fun j => s * m.gb j
           v := fun j k => m.v j k / s, vb := fun j => m.vb j / s }

/-- Python `signed`: `m` on bits held as -1 or 1. It reads `2 v - 1` for each bit `v`, and gives
twice each bit it makes: `wi / 2`, `b + wi.sum(1) / 2`, `2 wo`. -/
def signed (m : Gated) : Gated :=
  { m with
    g := fun j k => m.g j k / 2
    gb := fun j => m.gb j + (∑ k ∈ range m.ins.length, m.g j k) / 2
    v := fun j k => m.v j k / 2
    vb := fun j => m.vb j + (∑ k ∈ range m.ins.length, m.v j k) / 2
    wo := fun i j => 2 * m.wo i j }

end Gated

/-- What `layer` makes of an MLP of the trace: `signed(convert(m, "geglu"))`. -/
def MLP.compiled (m : MLP) : Gated := (m.toReglu.sharpen Gated.sharpness).signed

/-- A hidden unit of a layer: its gate row, its value row, and its column of `w_out`. -/
structure Hidden where
  gate : ℕ → ℚ
  val : ℕ → ℚ
  out : ℕ → ℚ

/-- The first feature `f < W` that holds `s` in `row` (Python's `at`, `new`, `row.index`); `W` if
there is none. -/
def featIn (W : ℕ) (row : Row) (s : Name) : ℕ :=
  ((List.range W).find? (fun f => row f == some s)).getD W

/-- Whether some feature `f < W` of `row` holds `s` (Python `s in row`). -/
def inRow (W : ℕ) (row : Row) (s : Name) : Bool :=
  (List.range W).any (fun f => row f == some s)

/-- Feature `f` holds a signal in `row` that `before` does not hold (it is in `new.values()`). -/
def isNewFeat (W : ℕ) (before row : Row) (f : ℕ) : Bool :=
  decide (f < W) && match row f with
    | some s => !inRow W before s
    | none => false

/-- The unit `ONE`: `copy("geglu", ONE, 0.0)` reading feature 0, so `gelu(32 ONE) * (ONE / 32)`.
Its weight into `ONE` is `float(not skip) = 0`, and `w_out[new, 0] = -1` gives each new bit
its -1 (see `signed`). -/
def oneUnit (W : ℕ) (before row : Row) : Hidden where
  gate := fun f => if f = 0 then 32 else 0
  val := fun f => if f = 0 then 1 / 32 else 0
  out := fun f => if isNewFeat W before row f then -1 else 0

/-- `copy("geglu", f"erase {before[f]}", -1.0)` on feature `f`: `gelu(32 ONE) * (x_f / 32)`,
written into `f` with weight -1. -/
def eraseUnit (f : ℕ) : Hidden where
  gate := fun f' => if f' = 0 then 32 else 0
  val := fun f' => if f' = f then 1 / 32 else 0
  out := fun f' => if f' = f then -1 else 0

/-- The units of a compiled MLP, placed: `wi[:, f] += weights` for each input on feature
`at[x]`, `wi[:, 0] += b`, and `wo[outs] = m.wo` on the features `new[y]` of its outputs. -/
def mlpUnits (W : ℕ) (before row : Row) (m : MLP) : List Hidden :=
  let c := m.compiled
  (List.range c.n).map fun j =>
    { gate := fun f =>
        (∑ k ∈ range c.ins.length, if featIn W before (.sig (c.ins.getD k 0)) = f then c.g j k else 0)
          + (if f = 0 then c.gb j else 0)
      val := fun f =>
        (∑ k ∈ range c.ins.length, if featIn W before (.sig (c.ins.getD k 0)) = f then c.v j k else 0)
          + (if f = 0 then c.vb j else 0)
      out := fun f =>
        ∑ i ∈ range c.outs.length, if featIn W row (.sig (c.outs.getD i 0)) = f then c.wo i j else 0 }

/-- `m.outputs[0] in new`: an MLP is at the depth where its outputs first appear. -/
def mlpIsNew (W : ℕ) (before row : Row) (m : MLP) : Bool :=
  match m.outs.head? with
  | some y => inRow W row (.sig y) && !inRow W before (.sig y)
  | none => false

/-- Python `layer(mlps, before, row, "geglu", skip=True, norm=True)`. -/
def layerUnits (W : ℕ) (mlps : List MLP) (before row : Row) : List Hidden :=
  oneUnit W before row ::
    ((mlps.filter (mlpIsNew W before row)).flatMap (mlpUnits W before row) ++
      ((List.range W).filter (fun f => decide (1 ≤ f) && before f != none && before f != row f)).map
        eraseUnit)

/-- The network of `emit(t, grid, "geglu", skip=True, norm=True)`. `one` is what the stream holds
`ONE` as: `2 * len(grid[0]) ** 0.5`. -/
structure Network where
  W : ℕ
  grid : List Row
  layers : List (List Hidden)
  one : ℝ

noncomputable section

/-- `EPS`, under the norm's square root. -/
def EPS : ℝ := 1 / 1000000

/-- Python `emit(t, grid, "geglu", skip=True, norm=True)`. -/
def emit (t : Trace) (g : Grid) : Network where
  W := g.width
  grid := g.rows
  layers := (g.rows.zip g.rows.tail).map fun br => layerUnits g.width t.mlps br.1 br.2
  one := 2 * Real.sqrt g.width

/-- The cells of `row` after `ONE` that hold a signal: `sum(s is not None for s in row[1:])`. -/
def count (W : ℕ) (row : Row) : ℕ :=
  ((List.range W).filter (fun f => decide (1 ≤ f) && (row f).isSome)).length

/-- The weight of the norm before a layer that reads `row`. -/
def rmsScale (W : ℕ) (row : Row) (one : ℝ) : ℝ :=
  Real.sqrt ((one ^ 2 + count W row) / W + EPS)

/-- Python `rmsnorm(x, row, one)`: `x` over its root mean square, times `rmsScale`, and feature 0
`one` times smaller. -/
def rmsnorm (W : ℕ) (row : Row) (one : ℝ) (x : ℕ → ℝ) : ℕ → ℝ := fun f =>
  let z := rmsScale W row one * x f / Real.sqrt ((∑ g ∈ range W, x g ^ 2) / W + EPS)
  if f = 0 then z / one else z

/-- `w . z` over the features. -/
def dot (W : ℕ) (w : ℕ → ℚ) (z : ℕ → ℝ) : ℝ := ∑ f ∈ range W, (w f : ℝ) * z f

/-- `w_out @ (act(w_gate @ z) * (w_val @ z))`, at feature `f`. -/
def layerApply (act : ℝ → ℝ) (W : ℕ) (us : List Hidden) (z : ℕ → ℝ) (f : ℕ) : ℝ :=
  (us.map fun u => (u.out f : ℝ) * (act (dot W u.gate z) * dot W u.val z)).sum

/-- `x + layer(rmsnorm(x, before, one))` -/
def netStep (act : ℝ → ℝ) (W : ℕ) (one : ℝ) (before : Row) (us : List Hidden) (x : ℕ → ℝ) :
    ℕ → ℝ :=
  fun f => x f + layerApply act W us (rmsnorm W before one x) f

/-- Python `Network.run`: the stream at every depth. `layers[d]` reads `grid[d]`. -/
def Network.run (act : ℝ → ℝ) (N : Network) (x : ℕ → ℝ) : List (ℕ → ℝ) :=
  (N.grid.zip N.layers).scanl (fun x bl => netStep act N.W N.one bl.1 bl.2 x) x

/-- The stream after the last layer. -/
def Network.final (act : ℝ → ℝ) (N : Network) (x : ℕ → ℝ) : ℕ → ℝ :=
  (N.grid.zip N.layers).foldl (fun x bl => netStep act N.W N.one bl.1 bl.2 x) x

/-- Python's `gelu`: `z / 2 * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 z^3)))`. -/
def gelu (z : ℝ) : ℝ :=
  z / 2 * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (z + 0.044715 * z ^ 3)))

end

end Reifier

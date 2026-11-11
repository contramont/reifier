module

public import Reifier.Grid

@[expose] public section

/-!
# Stage 3, the weights: `src/reifier/emit.py`

Layer `d` is one MLP on features, with no biases: `w_out @ act(w_in @ x)`. Its `x` is the stream,
the values of the features, which hold row `d - 1` of the grid, and it makes row `d`. What is
around the layers decides the rest:

    a stack:  x = layer(x)
    skip:     x = x + layer(x)
    norm:     layer(norm(x))      a bit is held as -1 or 1, not 0 or 1

`convert`, `copy`, `signed`, `layer`, `rmsnorm`, `Network` and `emit` are here as Python has
them. The weights are made over any field, and a network runs on reals.
-/

namespace Reifier

variable {K : Type} [Field K]

/-- `SHARPNESS = {"swiglu": 128.0, "geglu": 32.0}` -/
def sharpness : Act → Option K
  | .swiglu => some 128
  | .geglu => some 32
  | _ => none

/-- `EPS = 1e-6`, under the RMSNorm's square root. -/
noncomputable def EPS : ℝ := 1e-6

/-- `convert(m, activation)`: `m` as an MLP of `activation`: step -> relu -> reglu -> swiglu or
geglu, as far as needed. Python then asserts `MLP.HasForm`. -/
def convert (m : MLP K) (activation : Act) : MLP K :=
  -- [z >= 0], for an integer z, is relu(4 z + 5/2) - relu(4 z + 3/2): the units u+ and u-
  let m := if m.activation = .step ∧ activation ≠ .step then
      { m with
        wi := m.wi.map (·.map (4 * ·)) ++ m.wi.map (·.map (4 * ·))
        b := m.b.map (4 * · + 2 + 1 / 2) ++ m.b.map (4 * · + 2 - 1 / 2)
        activation := .relu
        wo := List.zipWith (· ++ ·) m.wo (m.wo.map (·.map (- ·)))
        hidden := m.hidden.map (· ++ "+") ++ m.hidden.map (· ++ "-") }
    else m
  -- relu(z) is relu(z) * 1
  let m := if m.activation = .relu ∧ activation.gated then
      { m with
        wi := m.wi ++ m.wi.map (·.map (0 * ·))
        b := m.b ++ m.b.map (fun _ => 1)
        activation := .reglu }
    else m
  -- relu(g) * v is act(s g) * v / s
  match (if m.activation = .reglu then sharpness activation else none : Option K) with
  | some s =>
    let scale := List.replicate m.hidden.length s ++ List.replicate m.hidden.length (1 / s)
    { m with
      wi := List.zipWith (fun weights c => weights.map (· * c)) m.wi scale
      b := List.zipWith (· * ·) m.b scale
      activation := activation }
  | none => m

/-- `assert m.activation == activation, f"{m.name} has no {activation} form"` -/
def MLP.HasForm (m : MLP K) (activation : Act) : Prop :=
  (convert m activation).activation = activation

/-- `copy(activation, name, weight)`: an MLP of `activation` whose one unit, `name`, gives
`weight` times the one thing it reads: a bit, as `[x >= 1/2]` or `relu(x)`, if `activation` is
step or relu, and anything, as `relu(1) * x`, if it is gated. -/
def copy (activation : Act) (name : String) (weight : K := 1) : MLP K :=
  convert
    (if activation.gated then
      ⟨name, ["x"], ["y"], [name], [[0], [1]], [1, 0], .reglu, [[weight]]⟩
    else
      ⟨name, ["x"], ["y"], [name], [[1]], [if activation = .step then -(1 / 2) else 0],
        activation, [[weight]]⟩)
    activation

/-- `signed(m)`: `m` on bits held as -1 or 1: it reads `2 v - 1` for each bit `v`, and gives
twice each bit it makes, which is that bit as held, plus 1. -/
def signed (m : MLP K) : MLP K :=
  { m with
    wi := m.wi.map (·.map (· / 2))
    b := List.zipWith (fun b weights => b + weights.sum / 2) m.b m.wi
    wo := m.wo.map (·.map (2 * ·)) }

/-- `{s: f for f, s in enumerate(row) if s}.get(name)`: the last feature of `row` that holds
`name`, a name that Python takes for true. -/
def Row.at? (row : Row) (name : String) : Option ℕ :=
  if name = "" then none
  else (List.range row.length).reverse.find? fun f => row[f]? = some (some name)

/-- `new.get(name)`, for `new = {s: f for f, s in enumerate(row) if s and s not in at}`: the
feature of `row` in which `name` is made, if the row before does not hold it. -/
def newAt? (before row : Row) (name : String) : Option ℕ :=
  if (before.at? name).isSome then none else row.at? name

/-- `f in new.values()` -/
def isNew (before row : Row) (f : ℕ) : Bool :=
  match row.getD f none with
  | some s => newAt? before row s == some f
  | none => false

/-- A part of a layer: an MLP, the features it reads, the features it writes. -/
structure Part (K : Type) where
  m : MLP K
  ins : List ℕ
  outs : List ℕ

/-- The parts of a layer, in order: ONE, if a stack writes it again or, with norms, each new
bit needs its -1; the MLPs that make the signals new in `row`; then the copies, by feature. -/
def parts (mlps : List (MLP K)) (before row : Row) (activation : Act) (skip norm : Bool) :
    List (Part K) :=
  -- the unit ONE gives 1: to ONE, unless a skip connection keeps it
  (if norm || !skip then [⟨copy activation ONE (if skip then 0 else 1), [0], [0]⟩] else [])
  -- an MLP is at the depth where its outputs first appear
  ++ mlps.filterMap (fun m =>
    if (m.outputs.head?.bind (newAt? before row)).isSome then
      some ⟨if norm then signed (convert m activation) else convert m activation,
        m.inputs.map fun x => (before.at? x).getD 0,
        m.outputs.map fun y => (newAt? before row y).getD 0⟩
    else none)
  ++ (List.range' 1 (row.length - 1)).flatMap (fun f =>
    -- a skip connection would keep it
    (if skip && truthy (before.getD f none) && before.getD f none != row.getD f none then
      [⟨copy activation ("erase " ++ (before.getD f none).getD "") (-1), [f], [f]⟩] else [])
    -- and a stack would lose it
    ++ (if !skip && ((row.getD f none).bind before.at?).isSome then
      [⟨copy activation ("keep " ++ (row.getD f none).getD ""), [f], [f]⟩] else []))

/-- A part's `wi`, on `width` features: `wi[:, f] += weights` for each feature `f` it reads,
an input read twice adding up, then `wi[:, 0] += m.b`, as biases are weights on ONE. -/
def Part.wi (p : Part K) (width : ℕ) : List (List K) :=
  List.zipWith
    (fun weights b => (List.range width).map fun f =>
      (((p.ins.zip weights).filter (·.1 = f)).map (·.2)).sum + if f = 0 then b else 0)
    p.m.wi p.m.b

/-- A part's `wo`, on `height` features: `wo[outs] = m.wo`. -/
def Part.wo (p : Part K) (height : ℕ) : List (List K) :=
  (p.outs.zip p.m.wo).foldl (fun wo q => wo.set q.1 q.2)
    (List.replicate height (List.replicate p.m.hidden.length 0))

/-- `layer(mlps, before, row, activation, skip, norm)`: one layer, the MLP on features that
makes `row` from the row before. -/
def layer (mlps : List (MLP K)) (before row : Row) (activation : Act) (skip norm : Bool) :
    MLP K :=
  let ps := parts mlps before row activation skip norm
  let wis := ps.map (·.wi before.length)
  let w_in :=  -- all gates, then values
    if activation.gated then
      wis.flatMap (fun w => w.take (w.length / 2)) ++ wis.flatMap (fun w => w.drop (w.length / 2))
    else wis.flatten
  let w_out := (List.range row.length).map fun f => ps.flatMap fun p => (p.wo row.length).getD f []
  let w_out :=  -- the unit ONE, the first, gives each new bit its -1
    if norm then
      w_out.zipIdx.map fun q => if isNew before row q.2 then q.1.set 0 (-1) else q.1
    else w_out
  { name := "layer"
    inputs := before.zipIdx.map fun q => if truthy q.1 then q.1.getD "" else s!"f{q.2}"
    outputs := (List.range row.length).map fun f => s!"f{f}"
    hidden := ps.flatMap (·.m.hidden)
    wi := w_in
    b := w_in.map fun _ => 0
    activation := activation
    wo := w_out }

/-- `rmsnorm(x, row, one)`: the RMSNorm before a layer that reads `row`: `x` over its root mean
square, times a weight. The weight is the root mean square of a stream that holds `row`, with
ONE as `one` and every other signal as -1 or 1, so such a stream comes out as it went in: but
for ONE, whose weight is `one` times smaller, so that the layer reads it as 1. -/
noncomputable def rmsnorm (x : List ℝ) (row : Row) (one : ℝ) : List ℝ :=
  let rms := Real.sqrt ((one ^ 2 + ((row.drop 1).filter (·.isSome)).length) / row.length + EPS)
  (x.map fun v => rms * v / Real.sqrt ((x.map fun v => v * v).sum / x.length + EPS)).modifyHead
    (· / one)

/-- `class Network` -/
structure Network where
  trace : Trace ℝ
  /-- the name of every feature at every depth -/
  grid : Grid
  /-- `layers[d - 1]` makes row `d` from row `d - 1` -/
  layers : List (MLP ℝ)
  skip : Bool
  norm : Bool
  /-- what the stream holds ONE as: 1, or more with skip and norm (see `emit`) -/
  one : ℝ

/-- `Network.stream(values, d)`: what the stream should hold at depth `d`, for the values of
the signals of row `d`: a bit `v` as `2 v - 1` with norms, else as `v`, and 0 where the row has
no signal. -/
def Network.stream (net : Network) (values : String → ℝ) (d : ℕ) : List ℝ :=
  (net.grid.getD d []).map fun
    | none => 0
    | some s => if s = ONE then net.one else if net.norm then 2 * values s - 1 else values s

/-- One turn of the loop of `Network.run`: the stream after `layer`, which reads `before`. -/
noncomputable def Network.step (net : Network) (layer : MLP ℝ) (before : Row) (x : List ℝ) :
    List ℝ :=
  let y := layer.eval (if net.norm then rmsnorm x before net.one else x)
  if net.skip then List.zipWith (· + ·) x y else y

/-- `Network.run(x)`: the stream at every depth, from the stream `x` at depth 0. -/
noncomputable def Network.run (net : Network) (x : List ℝ) : List (List ℝ) :=
  (net.layers.zip net.grid).scanl (fun x q => net.step q.1 q.2 x) x

/-- `Network.__call__`: the outputs on inputs `x`. -/
noncomputable def Network.call (net : Network) (x : List ℝ) : List ℝ :=
  let last := (net.run (net.stream (assign (fun _ => 0) net.trace.inputs x) 0)).getLastD []
  let outs := net.trace.outputs.map fun y => last.getD ((net.grid.getLastD []).idxOf (some y)) 0
  if net.norm then outs.map fun v => (v + 1) / 2 else outs

/-- `emit(t, grid, activation, skip, norm)`: a layer of `activation` for each depth of `grid`
after the first. `skip` puts a skip connection around each layer, and `norm` an RMSNorm before
each, which needs a gated activation. With both, the stream holds ONE as `2 sqrt(features)`. -/
noncomputable def emit (t : Trace ℝ) (grid : Grid) (activation : Act) (skip norm : Bool) :
    Network where
  trace := t
  grid := grid
  layers := (grid.zip grid.tail).map fun q => layer t.mlps q.1 q.2 activation skip norm
  skip := skip
  norm := norm
  one := if skip && norm then 2 * Real.sqrt ((grid.headD []).length) else 1

end Reifier

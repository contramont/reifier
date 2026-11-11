import ReifierProofs.Basic

/-!
# Stage 2: the grid (`src/reifier/grid.py`, `token_grid` of `transformer.py`)

`grid[d][f]` is the name of the signal that feature `f` holds at depth `d`, or none. A row is a
function from features to names: Python's list `row` is `row f` for `f < len(row)`, and none
beyond, which is the padding `layout` gives every row to the width of the last.

`layout` is Python's algorithm, step for step: dicts keep their order of insertion (inputs, then
the outputs of each MLP in turn), and the list `free` is a stack, its last element popped first.
Here `free` is kept with that last element first.
-/

namespace Reifier

/-- What a cell can hold: `ONE`, a signal, or bit `i` of a token's own code (`code[i]`). -/
inductive Name
  | one
  | sig (s : Sig)
  | code (i : ℕ)
  deriving DecidableEq, Repr

/-- A row of the grid: what each feature holds; none beyond the width. -/
abbrev Row := ℕ → Option Name

/-- A grid: its width, and its rows, depth 0 first. -/
structure Grid where
  width : ℕ
  rows : List Row

namespace Layout

/-- `max(last.get(x, 0), d)`, where `none` is `inf`. -/
def maxLast : Option ℕ → ℕ → Option ℕ
  | none, _ => none
  | some a, d => some (max a d)

/-- Python's first loop, over the MLPs: the depth of each signal (inputs 0, an MLP's outputs
1 + the deepest of its inputs), and the depth of the last MLP that reads each signal: `none` for
an output (`inf`), and `some 0` for a signal that nothing has read yet (`last.get(x, 0)`). -/
def scan (t : Trace) : (Sig → ℕ) × (Sig → Option ℕ) :=
  t.mlps.foldl
    (fun st m =>
      let d := 1 + (m.ins.map st.1).foldl max 0
      (fun s => if s ∈ m.outs then d else st.1 s,
       fun s => if s ∈ m.ins then maxLast (st.2 s) d else st.2 s))
    (fun _ => 0, fun s => if s ∈ t.outputs then none else some 0)

/-- The depth of each signal (Python `depth`). -/
def depth (t : Trace) : Sig → ℕ := (scan t).1

/-- The depth of the last MLP that reads each signal (Python `last`). -/
def last (t : Trace) : Sig → Option ℕ := (scan t).2

/-- Where each signal is freed: `last.get(x, d + 1)` for its depth `d`; `none` for never. No
MLP is at depth 0, so `last x = some 0` means that no MLP reads `x`. -/
def freedAt (t : Trace) (x : Sig) : Option ℕ :=
  match last t x with
  | some 0 => some (depth t x + 1)
  | l => l

/-- The keys of `depth`, in the order they were inserted. -/
def order (t : Trace) : List Sig := t.inputs ++ t.mlps.flatMap (·.outs)

/-- `made[d]`: the signals of depth `d`, in order. -/
def made (t : Trace) (d : ℕ) : List Sig := (order t).filter (fun x => depth t x = d)

/-- `freed[d]`: the signals freed at depth `d`, in order. -/
def freed (t : Trace) (d : ℕ) : List Sig := (order t).filter (fun x => freedAt t x = some d)

/-- `max(depth.values(), default=0)` -/
def maxDepth (t : Trace) : ℕ := ((order t).map (depth t)).foldl max 0

/-- The state of Python's second loop: `len(row)`, `row`, `at` (for signals; `ONE` is feature 0
and is never freed), and `free`, its last element first. -/
structure State where
  width : ℕ
  row : Row
  feat : Sig → ℕ
  free : List ℕ

/-- `free.append(at[x]); row[at[x]] = None` -/
def freeOne (st : State) (x : Sig) : State :=
  { st with free := st.feat x :: st.free, row := Function.update st.row (st.feat x) none }

/-- `if not free: free.append(len(row)); row.append(None)`, then
`at[y] = free.pop(); row[at[y]] = y`. A row holds none beyond its width, so appending none
changes nothing but the width. -/
def place (st : State) (y : Sig) : State :=
  let st := if st.free = [] then { st with free := [st.width], width := st.width + 1 } else st
  match st.free with
  | f :: rest =>
      { st with free := rest, feat := Function.update st.feat y f,
                row := Function.update st.row f (some (.sig y)) }
  | [] => st

/-- One depth of the second loop: free what it frees, then place what it makes. -/
def stepDepth (t : Trace) (st : State) (d : ℕ) : State :=
  (made t d).foldl place ((freed t d).foldl freeOne st)

/-- `row = [ONE, *t.inputs]` and `at = {x: f for f, x in enumerate(row)}`. -/
def init (t : Trace) : State where
  width := t.inputs.length + 1
  row := fun f => if f = 0 then some .one
    else if f ≤ t.inputs.length then some (.sig (t.inputs.getD (f - 1) 0)) else none
  feat := fun s => t.inputs.idxOf s + 1
  free := []

/-- The state after each depth: `init` for depth 0, then one for each depth to the deepest. -/
def states (t : Trace) : List State :=
  (List.range (maxDepth t)).scanl (fun st d => stepDepth t st (d + 1)) (init t)

end Layout

/-- Python `layout(t)`: row 0 is `ONE` and the inputs, and each MLP goes right after the last of
the MLPs it reads. All rows as wide as the last. -/
def layout (t : Trace) : Grid where
  width := ((Layout.states t).getLast?.map (·.width)).getD 1
  rows := (Layout.states t).map (·.row)

/-- `[s or old for s, old in zip(row, rows[-1])]` -/
def heldRow (prev r : Row) : Row := fun f =>
  match r f with
  | some s => some s
  | none => prev f

/-- The rows after the first, each held from the one before. -/
def heldRows : Row → List Row → List Row
  | _, [] => []
  | prev, r :: rs => heldRow prev r :: heldRows (heldRow prev r) rs

/-- Python `held(grid)`: each signal left in its feature until another is written there. -/
def held (g : Grid) : Grid where
  width := g.width
  rows := match g.rows with
    | [] => []
    | r :: rs => r :: heldRows r rs

/-- `[ONE, *code, *row[1:]]`: features `1 .. b` hold `code[0] .. code[b-1]`. -/
def tokenRow (b : ℕ) (r : Row) : Row := fun f =>
  if f = 0 then some .one else if f ≤ b then some (.code (f - 1)) else r (f - b)

/-- `if y in row: row[row.index(y)] = None; row[1 + i] = y` -/
def moveOut (W i : ℕ) (y : Sig) (r : Row) : Row :=
  match (List.range W).find? (fun f => r f == some (.sig y)) with
  | some f => Function.update (Function.update r f none) (1 + i) (some (.sig y))
  | none => r

/-- Python `token_grid(t)`: layout's, with features `1, 2, ...` for the bits of the token's own
code, each output `i` moved to feature `1 + i`, and held. -/
def tokenGrid (t : Trace) : Grid :=
  let L := layout t
  let b := t.outputs.length
  let W := L.width + b
  let rows := L.rows.map (tokenRow b)
  let rows := t.outputs.zipIdx.foldl (fun rows yi => rows.map (moveOut W yi.2 yi.1)) rows
  held { width := W, rows := rows }

end Reifier

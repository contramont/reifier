module

public import Reifier.Trace

@[expose] public section

/-!
# Stage 2, the grid: `src/reifier/grid.py`

A grid is a list of rows: `grid[d][f]` is the name of the signal that feature `f` holds at depth
`d`, or `none`. `layout` and `held` are here, statement by statement as Python has them: a dict
is the list of its items in the order they were first set, a list used as a stack has its top
first, and `inf` is `⊤`.
-/

namespace Reifier

variable {K : Type}

/-- `ONE = "ONE"`: the constant, feature 0 of every row. -/
def ONE : String := "ONE"

/-- A row of a grid: for each feature, the name of the signal it holds, or `none`. -/
abbrev Row := List (Option String)

/-- `Grid = list[list[str | None]]` -/
abbrev Grid := List Row

/-- Python's truth of a cell `s`, as in `if s`: false for `None`, and for the empty string. -/
def truthy : Option String → Bool
  | some s => s != ""
  | none => false

/-- A Python dict with strings for keys: its items, in the order they were first set. -/
abbrev Dict (α : Type) := List (String × α)

/-- `d[k] = v` -/
def Dict.set {α : Type} (d : Dict α) (k : String) (v : α) : Dict α :=
  if d.any (·.1 == k) then d.map (fun p => if p.1 == k then (k, v) else p) else d ++ [(k, v)]

/-- `d | dict.fromkeys(keys, v)` -/
def Dict.setAll {α : Type} (d : Dict α) (keys : List String) (v : α) : Dict α :=
  keys.foldl (fun d k => d.set k v) d

/-- `{x: f for f, x in enumerate(row)}`, as a function: 0 at a name that is not in `row`. -/
def featureOf (row : List String) : String → ℕ :=
  row.zipIdx.foldl (fun at' p => Function.update at' p.1 p.2) (fun _ => 0)

/-- What the last loop of `layout` holds: the row it is making, the feature of each signal, and
the features freed and not taken again, the last freed first. -/
structure LayoutState where
  row : Row
  at' : String → ℕ
  free : List ℕ

/-- One depth of the last loop of `layout`: from the row before, free the features of `freed`,
then give each signal of `made` the feature freed last, or else a new one. -/
def layoutStep (freed made : List String) (s : LayoutState) : LayoutState :=
  let s := freed.foldl
    (fun s x => { s with free := s.at' x :: s.free, row := s.row.set (s.at' x) none }) s
  made.foldl
    (fun s y =>
      let s := if s.free.isEmpty then  -- a new feature
        { s with free := [s.row.length], row := s.row ++ [none] } else s
      let f := s.free.headD 0
      { row := s.row.set f (some y), at' := Function.update s.at' y f, free := s.free.tail })
    s

/-- The first loop of `layout`: the depth of each signal, and the depth of the last MLP that
reads each signal, `⊤` for an output. -/
def depths (t : Trace K) : Dict ℕ × (String → Option (WithTop ℕ)) :=
  t.mlps.foldl
    (fun (depth, last) m =>
      let d := 1 + (m.inputs.map fun x => (depth.lookup x).getD 0).foldl max 0
      (depth.setAll m.outputs d,
        fun x => if x ∈ m.inputs then some (max ((last x).getD 0) d) else last x))
    (Dict.setAll [] t.inputs 0, fun x => if x ∈ t.outputs then some ⊤ else none)

/-- `layout(t)`: a grid for `t`. Row 0 is ONE and the inputs, and each MLP goes right after the
last of the MLPs it reads. A signal's feature is freed at the depth of the last MLP that reads
it, which reads the row before, and an output's never is. A signal that is made takes the
feature freed last, or else a new one. -/
def layout (t : Trace K) : Grid :=
  let (depth, last) := depths t
  -- the signals each depth makes, and frees: what nothing reads is freed after its own depth
  let made (d : ℕ) : List String := (depth.filter fun p => p.2 = d).map (·.1)
  let freed (d : ℕ) : List String :=
    (depth.filter fun p => (last p.1).getD ((p.2 + 1 : ℕ) : WithTop ℕ) = (d : WithTop ℕ)).map (·.1)
  let row : Row := some ONE :: t.inputs.map some
  let start : LayoutState := ⟨row, featureOf (ONE :: t.inputs), []⟩
  let rows := ((List.range' 1 ((depth.map (·.2)).foldl max 0)).foldl
    (fun (s, rows) d =>
      let s := layoutStep (freed d) (made d) s
      (s, rows ++ [s.row]))
    (start, [row])).2
  rows.map fun row => row ++ List.replicate ((rows.getLastD []).length - row.length) none

/-- `held(grid)`: grid with each signal left in its feature until another is written there, as
a skip connection leaves it. -/
def held (grid : Grid) : Grid :=
  match grid with
  | [] => []
  | first :: rest =>
    rest.scanl (fun before row => List.zipWith (fun s old => if truthy s then s else old) row before)
      first

end Reifier

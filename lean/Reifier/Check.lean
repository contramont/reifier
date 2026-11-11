module

public import Reifier.Emit

@[expose] public section

/-!
# The check of a grid: `check_grid` of `src/reifier/check.py`

`checkGrid t grid` is true where Python's `check_grid(t, grid)` returns, and false where one of
its asserts fails. A set is the list of its members.
-/

namespace Reifier

variable {K : Type}

/-- `made = {y: m for m in t.mlps for y in m.outputs}`, at a name: the last MLP that makes it. -/
def madeBy (t : Trace K) (y : String) : Option (MLP K) :=
  t.mlps.reverse.find? fun m => y ∈ m.outputs

/-- The cells of `row` that are new, as `check_grid` sees them:
`[s for s, old in zip(row, before) if s and s != old]`. -/
def changed (before row : Row) : List (Option String) :=
  ((row.zip before).filter fun q => truthy q.1 && q.1 != q.2).map (·.1)

/-- The asserts of `check_grid` on one depth, for the signals `seen` so far: the signals seen
after it, or `none` if an assert fails. Where a signal first appears, an MLP makes it: this row
holds all that the MLP makes, and the row before all that it reads. -/
def checkRow (t : Trace K) (before row : Row) (seen : List (Option String)) :
    Option (List (Option String)) :=
  (changed before row).foldlM
    (fun seen s =>
      match s.bind (madeBy t) with
      | some m =>
        if !seen.contains s && m.inputs.all (fun x => before.contains (some x))
            && m.outputs.all (fun y => row.contains (some y)) then
          some (s :: seen)
        else none
      | none => none)
    seen

/-- `check_grid(t, grid)`: ONE is feature 0 of every row. Row 0 holds `t`'s inputs, and the
last row its outputs. A signal keeps its feature. Where one first appears, an MLP makes it:
that row holds all that the MLP makes, and the row before all that it reads. -/
def checkGrid (t : Trace K) (grid : Grid) : Bool :=
  -- ONE is feature 0 of every row
  grid.all (fun row => (row.take 1).contains (some ONE))
  -- the rows do not differ in width
  && (grid.map List.length).dedup.length == 1
  -- depth 0 holds every input
  && t.inputs.all (fun x => (grid.headD []).contains (some x))
  -- the last depth holds every output
  && t.outputs.all (fun y => (grid.getLastD []).contains (some y))
  -- depth 0 holds no signal that an MLP makes
  && (grid.headD []).all (fun s => (s.bind (madeBy t)).isNone)
  && ((grid.zip grid.tail).foldlM (fun seen q => checkRow t q.1 q.2 seen) (grid.headD [])).isSome

end Reifier

"""Write small.lean (the eight small programs: grids and units) and adder_grid.lean (the adder's
grids) from Programs.lean, which export.py writes. Run each with `lake env lean difftest/<f>.lean`
from the project root; each writes lean_<f>.json there, which compare.py reads."""
import re
from pathlib import Path

here = Path(__file__).parent
progs = (here / "Programs.lean").read_text().replace("end Reifier.DiffTest\n", "")
head, *defs = re.split(r"\n(?=def [a-z0-9]+ : Trace where)", progs)
defs = {re.match(r"def ([a-z0-9]+)", d).group(1): d for d in defs}
CHECK = r'''
def cellStr : Option Name → String
  | none => "null"
  | some .one => "[\"one\"]"
  | some (.code i) => s!"[\"code\", {i}]"
  | some (.sig s) => s!"[\"sig\", {s}]"

def arr (xs : List String) : String := "[" ++ ", ".intercalate xs ++ "]"

def gridJson (g : Grid) : String := arr (g.rows.map fun r => arr ((List.range g.width).map (cellStr ∘ r)))

def qs (x : ℚ) : String := "\"" ++ toString x ++ "\""

/-- `emit`'s layers, computably (`one` is a real, so `emit` is not). -/
def layersOf (t : Trace) (g : Grid) : List (List Hidden) :=
  (g.rows.zip g.rows.tail).map fun br => layerUnits g.width t.mlps br.1 br.2

theorem layersOf_eq (t : Trace) (g : Grid) : layersOf t g = (emit t g).layers := rfl

def layersJson (t : Trace) : String :=
  let g := tokenGrid t
  arr ((layersOf t g).map fun us => arr (us.map fun u =>
    arr [arr ((List.range g.width).map (qs ∘ u.gate)), arr ((List.range g.width).map (qs ∘ u.val)),
         arr ((List.range g.width).map (qs ∘ u.out))]))
'''
LAYERS = r''' ++ ", \"layers\": " ++ layersJson t'''


def make(names, out, layers):
    prog = (r'''
def progJson (t : Trace) : String :=
  "{\"layout\": " ++ gridJson (layout t) ++ ", \"token\": " ++ gridJson (tokenGrid t)'''
            + (LAYERS if layers else "") + r''' ++ "}"
''')
    items = ", ".join(r'"\"' + n + r'\": " ++ progJson ' + n for n in names)
    tail = (f'\ndef allJson : String := "{{" ++ ", ".intercalate [{items}] ++ "}}"\n\n'
            f'#eval IO.FS.writeFile "lean_{out}.json" allJson\n\nend Reifier.DiffTest\n')
    (here / f"{out}.lean").write_text(head + "\n" + "\n".join(defs[n] for n in names) + CHECK + prog + tail)


make(["xor3", "add22", "add33", "gluadd33", "odd", "par4", "soa", "chain"], "small", True)
make(["adder"], "adder_grid", False)

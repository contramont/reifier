import ReifierProofs.Transformer

/-!
# Checks of the model against what the Python prints and its tests assert

`xor3` is the README's `trace(xor, 3)`: inputs `xs0 xs1 xs2` are signals 0, 1, 2, and the gates
`atleast1`, `atleast2`, `atleast3`, `odd` are signals 3, 4, 5, 6.
-/

namespace Reifier.Examples

/-- `threshold_gate(xs, weights, threshold)`: `mlp(xs, [weights], [-threshold], "step", [[1]])` -/
def gate (ins : List Sig) (out : Sig) (w : List ℚ) (θ : ℚ) : MLP where
  ins := ins
  outs := [out]
  act := .step
  hidden := 1
  wi := fun _ k => w.getD k 0
  b := fun _ => -θ
  wo := fun _ _ => 1

def xor3 : Trace where
  inputs := [0, 1, 2]
  mlps := [gate [0, 1, 2] 3 [1, 1, 1] 1, gate [0, 1, 2] 4 [1, 1, 1] 2, gate [0, 1, 2] 5 [1, 1, 1] 3,
           gate [3, 4, 5] 6 [1, -1, 1] 1]
  outputs := [6]

/-- the grid as a list of rows, each a list of cells -/
def cells (g : Grid) : List (List (Option Name)) := g.rows.map fun r => (List.range g.width).map r

/-- The README prints, for `layout(trace(xor, 3))`:
```
feature  depth 0  depth 1       depth 2
0        ONE      "             "
1        xs0      xor/atleast3  xor/odd
2        xs1      xor/atleast2  .
3        xs2      xor/atleast1  .
```
-/
example : cells (layout xor3) =
    [[some .one, some (.sig 0), some (.sig 1), some (.sig 2)],
     [some .one, some (.sig 5), some (.sig 4), some (.sig 3)],
     [some .one, some (.sig 6), none, none]] := by decide

/-- `held(layout(t))` keeps `atleast2` and `atleast1` where nothing writes over them. -/
example : cells (held (layout xor3)) =
    [[some .one, some (.sig 0), some (.sig 1), some (.sig 2)],
     [some .one, some (.sig 5), some (.sig 4), some (.sig 3)],
     [some .one, some (.sig 6), some (.sig 4), some (.sig 3)]] := by decide

/-- `test_every_hidden_unit_has_a_name`: with `held(g)`, `emit(..., "geglu", skip=True, norm=True)`
has 10 hidden units in its first layer (`ONE`, `u+` and `u-` of each of three gates, three
erasures), and 4 in its second: `ONE`, `xor/odd+`, `xor/odd-`, `erase xor/atleast3`. -/
example : ((emit xor3 (held (layout xor3))).layers.map List.length) = [10, 4] := by
  decide +kernel

/-- `show(kept).endswith("erase xor/atleast3 = gelu(32 ONE) * (0.03125 xor/atleast3) -> -f1")`:
the last unit of the second layer reads `ONE` with 32, feature 1 with 1/32, writes -1 into
feature 1. -/
example :
    let u := ((emit xor3 (held (layout xor3))).layers.getD 1 []).getLast?.getD ⟨0, 0, 0⟩
    (List.range 4).map u.gate = [32, 0, 0, 0] ∧ (List.range 4).map u.val = [0, 1 / 32, 0, 0] ∧
      (List.range 4).map u.out = [0, -1, 0, 0] := by
  decide +kernel

/-- The gate `odd = [atleast1 - atleast2 + atleast3 >= 1]` in layer 2, unit `xor/odd+`: in
`signed(convert(m, "geglu"))` its gate row is `32 * 4 * w / 2 = 64 w` on the features of its
inputs (`atleast3` on 1, `atleast2` on 2, `atleast1` on 3), and its bias on `ONE` is
`32 * (4 * (-1) + 5/2) + 64 * (1 - 1 + 1) = 16`; its value is `1/32` on `ONE`; it writes 2 into
feature 1 (where `odd` goes). -/
example :
    let u := ((emit xor3 (held (layout xor3))).layers.getD 1 []).getD 1 ⟨0, 0, 0⟩
    (List.range 4).map u.gate = [16, 64, -64, 64] ∧ (List.range 4).map u.val = [1 / 32, 0, 0, 0] ∧
      (List.range 4).map u.out = [0, 2, 0, 0] := by
  decide +kernel

/-- `token_grid` of the README's program `count(bits) = add(bits[:2], bits[2:])` is beyond this
file; here the token grid of `xor3` as a one-bit program on 3 tokens: code bit 0 at feature 1,
the output `odd` moved there at depth 2. -/
example : cells (tokenGrid xor3) =
    [[some .one, some (.code 0), some (.sig 0), some (.sig 1), some (.sig 2)],
     [some .one, some (.code 0), some (.sig 5), some (.sig 4), some (.sig 3)],
     [some .one, some (.sig 6), some (.sig 5), some (.sig 4), some (.sig 3)]] := by decide

end Reifier.Examples

# Notes for the agents proving ReifierProofs

The Python compiler is this repository's `src/reifier` (branch `simple-refactored`).
You may READ its source (`src/reifier/*.py`) for reference. Do NOT execute or import it.

## Layout of the Lean project (`lean/`)

Definitions (frozen; never edit):

| file | models | key defs |
|---|---|---|
| `ReifierProofs/Basic.lean` | `trace.py` | `MLP` (`pre`, `unit`, `out`, `read`, `apply`), `Trace` (`values`, `env`, `outVal`, `window`), `relu`, `heaviside`, `IsBit` |
| `ReifierProofs/Grid.lean` | `grid.py`, `token_grid` | `Name`, `Row := ℕ → Option Name`, `Grid`, `Layout.*` (`scan`, `depth`, `last`, `freedAt`, `order`, `made`, `freed`, `maxDepth`, `State`, `freeOne`, `place`, `stepDepth`, `init`, `states`), `layout`, `heldRow`, `heldRows`, `held`, `tokenRow`, `moveOut`, `tokenGrid` |
| `ReifierProofs/Emit.lean` | `emit.py` (geglu, skip, norm) | `Gated`, `MLP.toReglu`, `Gated.sharpen`, `Gated.signed`, `MLP.compiled`, `Hidden` (a unit: `gate`, `val`, `out`), `featIn`, `inRow`, `isNewFeat`, `oneUnit`, `eraseUnit`, `mlpUnits`, `mlpIsNew`, `layerUnits`, `Network`, `emit`, `EPS`, `count`, `rmsScale`, `rmsnorm`, `dot`, `layerApply`, `netStep`, `Network.run`, `Network.final`, `gelu` |
| `ReifierProofs/Transformer.lean` | `transformer.py` | `PAIRS = 8`, `CONTEXT = 512`, `MARGIN = 30`, `turn`, `meanCos`, `runnerUp`, `attnScale`, `lookback`, `rope`, `Transformer`, `transformer`, `visible`, `Transformer.{embedded, matvec, zvec, key, query, score, attn, value, afterAttn, final, logit}`, `windowBits` |
| `ReifierProofs/BF16.lean` | bf16 forward pass | `maxBF16`, `IsBF16`, `Rounding` (`fl`, `fl_mem`, `fl_nearest`), `BF16.{one, norm, layerApply, netStep, final, embed, zvec, matvec, key, query, score, attn, value, head, afterAttn, finalAt, logit}` |
| `ReifierProofs/Spec.lean` | well-formedness | `Trace.inVals`, `Trace.WellFormed`, `Trace.IsProgram`, `CodesOK`, `MLP.rows`, `MLP.slope`, `MLP.Bounded`, `Trace.RealOK`, `Gated.{gOn, vOn, gateAt, valAt}`, `Trace.BF16OK` |
| `ReifierProofs/Interfaces.lean` | the proof's steps | `ideal`, `valAt`, `ValOK`, `RowOK`, `PairOK`, `Row0OK`, `GridOK`, every `Stmt_*`, `Close`, `PosOK`, `WidthOK`, `logitOf`, `blogitOf`, `CodeIs`, `unitSum`, `gateDot`, `valDot` |
| `ReifierProofs/Main.lean` | the theorems | `MainReal`, `MainBF16`, `Stmt_rounding_exists` |

Proofs: `ReifierProofs/Proofs/*.lean`, one per step, each proving `stmt_*` or `*_of` (which take
other `Stmt_*` as hypotheses). `ReifierProofs/Final.lean` plugs them together.
`ReifierProofs/Examples.lean` checks the model against the Python by `decide` (e.g. `layout xor3`).

## Rules

* Edit only your own file(s). You may create helper modules `ReifierProofs/Proofs/<YourFile>Aux*.lean`
  and import them from your file; nobody else touches them.
* Never change a definition file, an interface, or the statement of your theorem. If you are sure a
  statement is false, stop and report a concrete counterexample.
* Check with `cd lean && lake env lean ReifierProofs/Proofs/<File>.lean`.
  If you made helper modules: `lake build ReifierProofs.Proofs.<YourHelper>` (only yours) before that.
  NEVER run a bare `lake build` or build any module that is not yours: other agents edit their files
  at the same time, and the definitions are already built.
* Scratch files: put them in a directory of your own outside the repository
  and run them with `lake env lean <path>` from the project directory.
* `lake env lean` does not apply the lakefile's `leanOptions` (`autoImplicit = false`): a file that
  passes it can still fail `lake build`. Declare every variable.
* The final file has no `sorry`, `admit`, `axiom`, `native_decide`, `implemented_by`, or `@[extern]`.
* Keep the file compiling while you work (`sorry` on open subgoals), and finish with none.
* Many small lemmas beat one big proof. `simp only [...]` with the definitions, `unfold`, `show`,
  `Finset.sum_comm`, `Finset.sum_ite_eq'`, `Finset.sum_range_succ`, `List.sum_cons`,
  `List.map_append`, `List.sum_append`, `List.flatMap`, `abs_sub_le`, `abs_le`, `nlinarith`,
  `positivity`, `field_simp`, `ring`, `norm_num`, `gcongr`, `linarith`, `omega`, `decide`.
  Use `exact?`/`apply?`/`rw?`/`simp?` to find names.

## Facts checked numerically (by an independent script, not the repo's code)

* `runnerUp ≈ 0.93565`, attained at offset `e = 1`; every other `meanCos e ≤ 0.806` for `2 ≤ e ≤ 511`.
  For every `2 ≤ e ≤ 511` some `i ≤ 4` has `cos (e * turn i) ≤ 0.52` (bands: `e ∈ [2,5]` with `turn 0 = 1`,
  `[6,16]` with `turn 1 = 10^-1/2`, `[17,52]` with `turn 2 = 0.1`, `[53,166]` with `turn 3 = 10^-3/2`,
  `[167,511]` with `turn 4 = 0.01`; each puts `e * turn i` in `[1.03, 5.25]` where `cos ≤ cos 1.03 ≤ 0.5177`).
* `attnScale ≈ 58.27`; the target scores `8 attnScale ≈ 466`.
* `1022 * exp(-30) ≈ 9.56e-11 < 1e-10`; `exp 1 > 2.7182818283` (`Real.exp_one_gt_d9`).
* `|gelu z - relu z| ≤ |z| exp(-2 sqrt(2/π) (|z| + 0.044715 |z|^3))`, which is `≤ 3.2e-21` for `|z| ≥ 8`.

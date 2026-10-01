# cheap-gates: one gated unit, or none, for step gates that need no more

Two general compiler passes. No Keccak code is touched.

1. **Cheap gates** (`tensors/swiglu.py`, on by default). This is the avenue. It is a
   SwiGLU-level conversion in `SwiGLU.from_matrix(..., cheap=True)`. It handles copies,
   NOT, AND, inhib and constants, the BOS row, duplicate gates and dead units.
2. **Simplify** (`compile/simplify.py`, `TreeCompiler(simplify=True, tighten=True)`, off
   by default). This is an extension. Once copies cost one unit, the next step is
   copies that cost nothing. It folds levels of copies, NOTs and constants into their
   neighbours, folds constants into biases, merges equal nodes and drops unread nodes.
   With tighten, it also bounds gates exactly by enumerating their grandparents. It
   overlaps with dce-const and leveling, so take whichever version composes best. Cheap
   gates apply on top of either.

All numbers come from the harness at log_w=6 with `ok: true`. `xofbench.py` is used
unchanged, except for runs marked (kw). Those set from_matrix options and use
`xofbench_kw.py`, a copy of the harness with a `--kw` flag. The simplify runs use
`cheap_variants:*_simplified`, which patch `Tree.from_root(simplify=True, tighten=True)`
the same way `variants.py` patches keccak.

## Results, log_w=6

| config | xof depth | depth | dense | sparse | hidden | margin |
|---|---|---|---|---|---|---|
| baseline (brief) | 3 | 20 | 3,305,445,348 | 2,034,788 | 187,752 | 3.6e-4 |
| baseline + cheap | 3 | 20 | 2,927,188,704 | 1,920,001 | 149,868 | 2.4e-7 |
| baseline + cheap, two=1 (kw) | 3 | 20 | 2,772,535,904 | 1,891,201 | 140,268 | 2.4e-7 |
| baseline + cheap + simplify (tighten) | 3 | 15 | 1,355,885,786 | 1,109,356 | 89,309 | 3.0e-7 |
| baseline + cheap two=1 + simplify (tighten) (kw) | 3 | 15 | 1,267,427,538 | 1,088,932 | 82,501 | 2.4e-7 |
| glu_chi_iota (brief) | 3 | 8 | 247,067,600 | 541,724 | 43,152 | 3.7e-4 |
| **glu_chi_iota + cheap** | 3 | 8 | 214,170,480 | 521,437 | 36,768 | 2.4e-7 |
| glu_chi_iota + cheap, form=relu (kw) | 3 | 8 | 214,170,480 | 515,117 | 36,768 | 1.8e-5 |
| **glu_chi_iota + cheap + simplify** | 3 | **6** | **101,534,934** | **300,502** | 22,393 | 3.5e-5 |
| same, form=relu (kw) | 3 | 6 | 101,534,934 | 297,814 | 22,393 | 3.6e-5 |
| glu_chi_unit + cheap (kw) | 3 | 11 | 244,452,636 | 554,269 | 42,243 | 2.4e-7 |
| glu_chi_unit + cheap + simplify (kw) | 3 | 6 | 101,534,934 | 300,550 | 22,393 | 3.9e-5 |
| glu_xor_everywhere + cheap (kw) | 3 | 14 | 334,979,592 | 615,913 | 52,518 | 2.4e-7 |
| glu_xor_everywhere + cheap + simplify (kw) | 3 | 9 | 157,514,154 | 344,884 | 29,783 | 9.5e-7 |
| glu_clean_chi_iota + cheap (kw) | 3 | 8 | 356,834,480 | 1,390,237 | 60,768 | 2.4e-7 |
| glu_chi_iota, cheap=0 (kw) | 6 | 14 | 577,139,258 | 1,094,738 | 88,860 | 3.7e-4 |
| glu_chi_iota + cheap | 6 | 14 | 498,611,400 | 1,056,289 | 76,422 | 2.4e-7 |
| glu_chi_iota + cheap, relu (kw) | 6 | 14 | 498,611,400 | 1,037,873 | 76,422 | 1.6e-5 |
| glu_chi_iota + cheap + simplify | 6 | 12 | 358,295,502 | 831,322 | 61,375 | 3.2e-5 |
| baseline, cheap=0 (kw) | 6 | 38 | 7,268,023,266 | 4,153,442 | 394,188 | 3.7e-4 |
| baseline + cheap (kw) | 6 | 38 | 6,263,146,536 | 3,901,801 | 310,686 | 2.4e-7 |
| baseline + cheap + simplify, no tighten | 6 | 30 | 4,609,592,476 | 3,052,357 | 245,355 | 2.7e-4 |
| baseline + cheap + simplify (tighten) | 6 | 30 | 4,323,543,734 | 3,027,784 | 237,164 | 3.6e-7 |

The brief's references without cheap: glu_chi_unit 11 / 307,626,437 / 590,981 / 54,102;
glu_xor_everywhere 14 / 473,310,074 / 683,450 / 74,652; glu_clean_chi_iota 8 /
389,731,600 / 1,410,524 / 67,152.

Gains as depth / dense / sparse (hidden units in brackets):
- **cheap alone**
  - glu_chi_iota: 1.00x / 1.15x / 1.04x (1.17x). The margin is about 1500x smaller.
  - baseline: 1.00x / 1.13x / 1.06x (1.25x); with two=1, 1.19x / 1.08x (1.34x).
  - glu_xor_everywhere: 1.41x dense (1.42x hidden). glu_chi_unit: 1.26x dense.
- **cheap + simplify**
  - glu_chi_iota: **1.33x / 2.43x / 1.80x** (1.93x). Against the baseline: 3.3x / 32.6x
    / 6.8x.
  - baseline: 1.33x / 2.44x / 1.83x (2.10x); with two=1, 1.33x / 2.61x / 1.87x (2.28x).

## 1. Cheap gates (`SwiGLU.from_matrix`, `cheap_rows`, `merge_units`)

A step row of the folded matrix computes `[z >= 1]` for `z = w @ (1, bits)`. Its weights
are integers, so `z` is an integer. `zmax = w0 + sum(max(0, wi))` and
`zmin = w0 + sum(min(0, wi))` bound `z` over all Boolean inputs. By those bounds:
- `zmax <= 0`: the row is constant 0 and gets no unit.
- `zmin >= 1`: the row is constant 1 and reads the BOS unit, so it only costs a `wo` entry.
- `zmax <= 1`: the row gets **one unit, `max(0, 2z-1) * (3-2z)`**. Its gate row is
  `2w - e0` and its value row is `3e0 - 2w`. It gives 0 at every `z <= 0` and 1 at
  `z = 1`. This covers copies (z = x), NOT (1-x), AND of k (sum-(k-1)), inhib, a
  2-input xor's final gate, and any gate with `sum(max(0, wi)) <= threshold`.
- With `two=True`, `zmax == 2` gets one unit, `max(0, 2z-1) * (5-2z) / 3` (1 at z = 1
  and z = 2). This covers OR2, majority-3, NAND2 and similar gates. It is off by
  default, see numerics.
- The **BOS row** is a constant-1 row, so it is now one unit, `silu(16r)*r/16`, instead
  of two. A unit with gate 1 then has exactly the BOS scale: a clean unit at z = 1
  computes the same float as BOS.
- `merge_units` runs on every layer.
  - Hidden units with equal gate and value rows compute the same function, so they share
    one unit and their `wo` columns are added. This catches duplicate gates, such as
    L1's 1144 duplicate message copies: 2288 step units became 1144 units.
  - It also drops units whose gate can never be positive on bit inputs, and units nobody
    reads. After constants are folded into biases, this is how the xor units past the
    largest reachable sum disappear. In round 1 of theta that leaves about 4.2 units per
    node instead of 6.

**Why it is exact.** A unit computes `silu(16 r g) * r v / 16`. That is
`r^2 max(0, g) v` up to `r^2 |g v| sigmoid(-16 r |g|)`. For the clean form, `g = 2z-1`
is an odd integer and never 0, so the error is below 3e-7 (r >= 1). Steps are off by
3.4e-4 at every integer, because their slope sits 0.375 from it.

**Cleaning.** At `z = 1+d`, the clean unit gives `1 - 4d^2`. For `z <= 0` it stays 0 for
any `d < 1/2`, because the gate is at most `-1+2d`. So a clean copy maps an input error
`d` to `4d^2`, while a step maps any error below about 0.3 to a fixed 3.4e-4. For
`d < 0.009`, the clean unit is the better cleaner.

**Why `two` is off by default.** No 1-unit form of a `z <= 2` row can be flat at both
`z = 1` and `z = 2`: on the positive-gate side, `g*v` would have to be 1 with zero slope
at both points, so it would be the constant 1, and then the gate cannot cross 0. The
`two` form has slope 4/3 at both points, so it passes errors on.

**The relu form.** `form="relu"` is `max(0, z) * 1`. It uses fewer nonzeros (3 instead
of 5 for a copy) but passes errors on: the margin is 1.8e-5 instead of 2.4e-7. It saves
only about 1% of sparse, so `clean` is the default.

**Assumption.** Cheap gates assume every input feature is a bit, with BOS = 1. That holds
for every circuit this pipeline compiles, since each feature is a gate or glu output.
For other features, such as linear-fold's counts up to 11, pass
`from_matrix(..., bounds=(low, high))`: integer bounds per input column, BOS included.
Both `cheap_rows` and the dead-unit test in `merge_units` then use those bounds.
Without them, a theta unit `max(0, s - 2j)` on a count `s` would look dead and be
dropped. `tests/cheap_test.py::test_bounds_for_inputs_that_are_not_bits` covers this.
Merging equal units and the single BOS unit are safe for any inputs.

**Checks.**
- With `cheap=False`, the weights are bit-equal to the base's.
  `scripts/same_as_base.py` checks this on baseline, glu_chi_iota and glu_clean at
  log_w 0-2.
- `validate_bench.py` shows equal weights and passes for the default pipeline,
  glu_chi_iota, and both simplified variants.
- All 70 repo tests pass (57 base tests plus 13 new ones in `tests/cheap_test.py`).
  `tests/glu_test.py` now expects 1 BOS unit, not 2.
- `scripts/errs.py` checks the simplified glu_chi_iota at log_w=6 on 256 random
  messages. Every layer matches the exact simulation within 8e-5, and the outputs match
  eager xof.

## 2. Numerics: errors do not build up, so no rule is needed

`scripts/errs.py` measures, layer by layer, the gap to an exact simulation of the leveled
graph. At log_w=6, glu_chi_iota, xof depth 6, over 16 messages, the max error per layer
is:

| layer | steps (cheap=0) | cheap, clean | cheap, relu |
|---|---|---|---|
| L1 (copies, consts) | 3.5e-4 | 2.4e-7 | 0 |
| L2 theta / L3 chi | 3.6e-3 / 6.8e-3 | 2.9e-6 / 9.2e-6 | 3.1e-6 / 1.2e-5 |
| L5 chi | **1.0e-2** | 3.8e-5 | 3.2e-5 |
| L7 .. L13 | 7.4e-3 .. 3.4e-4 | 3.5e-5 .. 5.3e-5 | 3.6e-5 .. 4.5e-5 |
| L14 outputs | 3.7e-4 | 2.4e-7 | 1.6e-5 |

The L1 step copies were the main source of error. The 11-input theta (slope 2 at even
sums) and chi amplified their 3.5e-4 to 1e-2 inside the chain, halfway to the 0.02
read-out tolerance. With cheap gates, the chain holds a steady 3-5e-5 over 6 rounds and
does not grow. A longer run confirms it: at log_w=4, 24 xof steps (48 chain layers),
over 64 messages:
- cheap: the chain max is 1.2e-4, the same in the first and second half, and the output
  margin is 2.4e-7.
- steps: the chain max is 1.0e-2 (4.4e-4 in the second half), and the margin is 3.8e-4.
- cheap + simplify: the chain max is 9.1e-5, and the margin is 3.4e-5.

The xof chain has no cleaning, and the theta/chi units are not flat. Even so, no "step
every k layers" rule is needed at depth 3, 6 or 24. With simplify, outputs come straight
from the chi units, so the margin is the chain's 3-4e-5, still 500x below the
tolerance.

If a longer chain ever needs cleaning, insert copies: each one is now a clean single
unit that maps an error `d` to `4d^2`. Use `two=True` only where `z <= 2` gates do not
form long unbroken chains, since each one passes errors on with a gain of 4/3.

## 3. Extension: `compile/simplify.py` (tree level, off by default)

These are exact rewrites of the leveled graph, applied until nothing changes. The input
level and the output order are kept.
- `merge_equal`: equal nodes of a level are merged, and their readers' weights are
  added. This covers duplicate copies, and theta outputs that constants made equal.
- `fold_constants`: a constant node (by its z bounds) is folded into its readers' biases
  and unit biases.
- `fold_affine_levels`: a level made only of copies, NOTs and constants is affine
  (`c + a*x`), so it is substituted into the next level's gate and unit weights. RMSNorm
  keeps bits relative to BOS, so the next level can read the level below directly. This
  removes L1. In the baseline it also removes the iota levels.
- `fold_outputs`: the outputs level is only copies, so the level below it computes the
  outputs directly. A NOT output becomes a negated node; for units that means `1 - sum`,
  and the extra `max(0,1)*1` unit merges into BOS.
- `drop_unread`: nodes nobody reads are removed, top down. After `fold_outputs`, the
  last round keeps only 224 chi nodes and the 320 theta nodes they need.
- `tighten` (only with `tighten=True`, since it makes gated units for MLP_SwiGLU):
  - It enumerates up to 2^12 values of a gate's grandparents to bound `z` exactly,
    beyond what the weight bounds show.
  - Gates with `z <= 1` become one clean unit, and gates with a fixed output become
    constants.
  - Example: the xor's final gate sums monotone counters with signs +-1, so its weights
    allow z up to 6, but it only takes the values 0 and 1.
  - On the baseline, this saves 7% of dense and makes the margin 3e-7.

On glu_chi_iota at log_w=6, the widths become [1145,6100,1468] [1468,1601,1601]
[1601,9825,1825] [1825,1825,1825] [1825,2369,769] [769,673,673].
- Round 1's theta reads the message directly: 1467 distinct nodes with about 4.2 units
  each.
- The last round has 320 theta nodes and 224 chi nodes.
- Theta layers are now 82% of dense, 88% of sparse and 82% of hidden units. Theta
  round 2 alone is 49% of dense.

Why L1 was 2752 wide:
- 1144 live message copies and 456 live constants come from `msg_to_state`.
- 1144 dead copies and 8 dead pad constants come from `bitlist_to_msg`. Its pad is empty
  (the message is full length), but its `const()` call still creates 8 gates. That makes
  the block one level tall, so its pass-through outputs get copied.

**Checks.**
- A fuzzer (`scripts/fuzz_simplify.py`) builds random circuits from gates, glu xor,
  chi units, constants, copies and NOTs, with NOT outputs. It ran 1300 circuits, of up
  to 14 or 30 ops. Every one matches eager evaluation on all inputs, with and without
  simplify+tighten. The 713 step-only circuits also match through MLP_Step with
  simplify on. Layer counts went from 8379 to 2273.
- With `TreeCompiler.simplify` on by default (no tighten), every repo test passes except
  the ones in `cheap_test.py` that count unsimplified layers. So it could be switched
  on.
- tighten must stay opt-in: MLP_Step and FlatCircuit reject gated units.

## Composes with

- **Tree-level avenues.** Cheap gates work on whatever matrix they receive. Any copy
  another avenue adds (for example a two-layer theta that carries the state) costs 1
  unit, not 2. Any constant another avenue folds into a bias removes the dead xor units
  automatically, through the gate-range pruning in `merge_units`.
- **dce-const and leveling.** Simplify overlaps with them (constants, dead nodes, L1, the
  output layer). If their passes are used instead, cheap gates still apply on top.
- **linear-fold.** It found much of simplify independently, at the matrix level
  (folding copy levels and the output level, prune and dedupe). Its `unit_copies` is
  the relu form of the cheap copy. Its best result is 6 / 91.8M / 179.6K. To combine:
  - Keep its fold.
  - Replace `unit_copies` with `cheap_rows`. That also covers NOT, AND, inhib and
    constants, with the clean form (flat, 1e-7 accuracy), and makes BOS one unit
    instead of two.
  - **Pass its level bounds** to `from_matrix(bounds=...)`: lower 0 (1 for BOS), upper
    = its `blist`. Its theta layers read counts, and without bounds the dead-unit test
    would drop live theta units.
  - Expected gain on top of its best: small (about 1 unit per layer, plus precision).
    Its circuits have no step gates left.
- **Theta restructuring (theta-structure, linear-fold).** Theta is now the only large
  cost left: 82% of dense and 88% of sparse after simplify.
- **Theta's input sums (linear-fold's `theta_count`).** This pushes each theta node's
  input sum into the chi layer's `wo`. The sum feature costs no hidden units, because
  the chi units are shared, and each theta unit's gate then reads 2 entries instead of
  12. I estimated it would take the simplified circuit's sparse from 300k to about
  195k. linear-fold measured more with its packing (179.6K). Those features are counts,
  so they need the `bounds` hook above.

## Tried or considered, and not used

- **relu form as the default.** It is 1% sparser but passes errors on. Kept as
  `form="relu"`.
- **`two=True` as the default.** It is exact and saves 7% of hidden units on the
  baseline, but it is not flat (see above). Kept as an option.
- **Sharing hidden units across outputs beyond exact duplicates.** For theta, the five
  outputs a_y ^ D share one D. A rank argument shows each output still needs its own
  units when the gates are linearly independent functions of the sum. No gain.
- **A 1-unit clean form for z <= 2, or a 1-unit OR3.** Neither exists. OR3's multilinear
  form has a cubic term, which one piecewise quadratic cannot match.
- **Carrying two digest bits in one feature (x1 + 2x2).** Packing is free, but unpacking
  costs as many units as it saves over only 3 carry layers. At most 2-3% of dense.
- **Merging always-on units (gates that are always >= 0) into one product unit.** Rare
  here. The round-1 constants are mostly capacity zeros, which do not shift the sums.
- **Collapsing two threshold levels into one gated level by synthesizing units from a
  node's truth table.** An example is the baseline's counters plus alternating sum, which
  would become glu_xor. It only helps step-based circuits, and it is unit-synthesis or
  theta-structure territory, so it was not done.

## Files

- `repo/`: the working copy.
- `patch.diff`: `git diff` of the repo, plus the new files `compile/simplify.py`,
  `tests/cheap_test.py`, and `tests/glu_test.py` from base.
- `patch_vs_base.diff`: only this avenue's changes against `$S/xof/base`: `swiglu.py`,
  `tree.py`, `glu_test.py`, `simplify.py` and `cheap_test.py`.
- `cheap_variants.py`: `*_simplified` variants for the unchanged harness.
- `xofbench_kw.py`: the harness copy with `--kw` for from_matrix options.
- `scripts/`:
  - `errs.py`: per-layer error against an exact simulation.
  - `census.py`: nodes per level by z range.
  - `per_layer.py`: per-layer dense, sparse and hidden.
  - `same_as_base.py`: checks that cheap=False gives the base's weights.
  - `fuzz_simplify.py`: the fuzzer.
- `runs/`: all raw harness outputs.
- `results.jsonl`: the log_w=6, depth 3 lines.

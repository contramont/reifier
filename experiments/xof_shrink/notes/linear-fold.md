# linear-fold: linear nodes, and folding affine levels into the layers around them

All numbers: log_w=6, 3 XOF steps, from `xofbench_fold.py` (a copy of the harness
that builds layers through the repo's own fold, see below), `ok: true`.

| variant (`fold_variants.py`) | flags | depth | dense | sparse | hidden | margin |
|---|---|---|---|---|---|---|
| glu_chi_iota (reference, plain harness) | | 8 | 247,067,600 | 541,724 | 43,152 | 3.7e-4 |
| variants:glu_chi_iota | --fold | 6 | 117,040,923 | 333,901 | 26,446 | 3.7e-4 |
| variants:glu_chi_iota | --fold --unit-copies | 6 | 111,718,235 | 327,181 | 25,102 | 4.2e-5 |
| lin_theta | --fold | 6 | 113,784,923 | 217,037 | 26,446 | 3.7e-4 |
| lin_theta | --fold --unit-copies | 6 | 109,544,603 | 210,317 | 25,102 | 1.6e-5 |
| lin_theta_consts | --fold | 6 | 103,627,049 | 187,688 | 23,743 | 3.7e-4 |
| lin_theta_consts | --fold --unit-copies | 6 | 99,386,729 | 180,968 | 22,399 | 1.5e-5 |
| **lin_theta_consts_pack** | --fold --unit-copies | **6** | **91,805,897** | **179,624** | 21,951 | 3.4e-5 (64 msgs) |
| lin_theta_consts_pack_last1 | --fold --unit-copies | 5 | 105,686,874 | 222,741 | 29,669 | 5.1e-5 |

Best (lin_theta_consts_pack): vs glu_chi_iota 1.33x depth, 2.69x dense, 3.02x sparse;
vs baseline (20 / 3,305,445,348 / 2,034,788) 3.33x, 36.0x, 11.3x. The depth-5 variant:
1.6x / 2.34x / 2.43x vs glu_chi_iota (a Pareto alternative: -1 layer for +15% dense,
+24% sparse). The general fold alone (glu_chi_iota --fold) is 1.33x / 2.11x / 1.62x.
Layers (in, hidden, out), BOS included:
`[1145, 6101, 1468] [1468, 1602, 1713] [1713, 9714, 1713] [1713, 1714, 545] [545, 2146, 545] [545, 674, 673]`
= theta1, chi1, theta2, chi2, theta3, chi3. The two layers that only copied (L1: message
copies and constants; L8: output copies) are gone, theta reads one feature per unit,
and the last round computes only what the 224-bit digest needs.

Run: `PYTHONPATH=$S/xof/av/linear-fold/repo/src:$S/xof/av/linear-fold:$S/xof
$S/venv/bin/python xofbench_fold.py --log-w 6 --depth 3 --variant
fold_variants:lin_theta_consts_pack --fold --unit-copies --widths`.
Bit-equality with the dense pipeline (`Compiler(fold=True, unit_copies=True)` ->
`MLP_SwiGLU`): `validate_fold.py <variant> --fold --unit-copies` (log_w 0-2,
weights_equal=True for every variant above). The original harness and
validate_bench are unchanged by the patch when fold is off (checked: default xof and
glu_chi_iota, weights_equal=True).

## Method (compiler: general; files in `repo/src/reifier`)

1. `neurons/core.py` `lin(incoming, weights=1s, bias=0, inline=False)`: a linear node,
   `weights . incoming + bias` with no threshold, an integer such as a count of bits.
   Gates and glu units read it like any Signal. `operations.py`: `count_parity(s)`,
   the glu_xor units on one count (range taken from the lin node, shifted to an even
   lower end), and `lin_range`.
2. Tracing: `lin` is a creator like `gate` and `glu` (`blocks.CREATORS`). `Origin` gets
   `linear` (and `inline`). A copy or output of a linear node is linear too (the
   identity, not a step at 1/2, which would threshold the integer): `Tree._copy_origin`.
3. `Matrices.layer_to_affine(level, size_in, prev)`: if every node of a level is affine
   in the level before (linear nodes, and threshold gates on at most one bit: copies,
   outputs, nots, constants, since any function of one bit is affine on 0/1), the level
   is an affine map M (BOS row/column), `fold.Affine`.
4. `tensors/fold.py` `fold_levels(items, bounds)`: walks the layers and affine levels
   and keeps a "read" map from the current features to the current level's values.
   An affine level after a layer: each row with 2+ inputs becomes a feature, i.e. a new
   row `M_row @ wo` of the layer before (identical rows shared); rows with one input
   (copies, constants) are read through. A layer reads the current level through
   `wg @ read`, `wv @ read`. Levels before the first layer are read through (they
   become the first layer's gates and biases: message copies and constants cost
   nothing), and the output level becomes rows of the last wo (no copy layer).
   Features no longer read are dropped. Consecutive affine levels compose.
5. Norm scale: RMSNorm divides every feature by the rms, so integer features up to B
   (counts up to 11) would flatten every gate of the layer by up to B. A layer reading
   features bounded by B gets RMSNorm weight `2^ceil(log2 B)` (16 for theta), so its
   BOS stays >= 1 after the norm and its gates are at least as steep as with bits.
   Bounds come from `Matrices.level_ranges` (bits (0,1); linear nodes: sums of weighted
   parent ranges). The norm weight is counted anyway, so it is free. Without it the
   integer features break the circuit (measured: margin 1.05, 256 wrong bits).
6. `fold.prune`: dedupe (hidden units with equal gate and value rows are merged, their
   wo columns added; features with equal wo rows are merged, their reader columns
   added), then drop hidden units whose wo columns are zero for all needed outputs and
   features that no unit reads (not the first layer's inputs). This is where the last
   round shrinks to the digest's cone (theta3 1600 -> 320 bits, chi3 1600 -> 224).
7. `unit_copies` (Matrices.from_graph / Compiler option): a copy of a bit is one gated
   unit `max(0, x) * 1` instead of a two-unit step; with ranges, a nonnegative integer
   linear node (e.g. a copy of a packed pair) is one unit too. Linear nodes left in a
   level with gates are computed exactly by the pair `silu(z) - silu(-z) = z`.
   One-unit copies are also more precise (margin 3.7e-4 -> 1.5e-5): the step copies
   were the main error source.
8. `Compiler(fold=True, unit_copies=True)` and `MLP_SwiGLU.from_matrices(fold=True)`
   run 4-6; `xofbench_fold.py` mirrors it layer by layer (`--fold --unit-copies`).

## Keccak-specific circuit changes (`fold_variants.py`)

- `theta_count`: theta bit = `count_parity(lin(11 bits))`. The count s of each theta
  bit is one feature that the chi layer before outputs (11 wo entries), and the 6
  theta units read 1 feature each instead of 11 (about 94 -> 24 entries per theta bit;
  theta2 is now 40.9K sparse).
- `theta_count_consts`: the traced xof records the constant state bits (suffix,
  capacity); round 1's counts add them to the bias and span only the variable bits,
  so they need fewer units (ceil(n_var/2)), and theta bits whose own bit is a zero
  capacity bit are the same function, which dedupe merges (theta1 9602 -> 6101 units).
- `xof_pack`: the digests of steps 1 and 2 are packed in pairs `x = d0 + 2 d1` (a lin
  node: one feature from the chi layer's wo, one unit per layer to carry, instead of
  two bits) and unpacked in the last layer: `d1 = max(0, x - 1)(2 - x/2)`,
  `d0 = x - 2 d1` (2 units per pair). The unpacking glu reads a last-round theta count
  with weight 0 so that the layout places it in the last layer. Saves ~8% dense.
- `xof_pack_last1` (depth 5): the last round as one layer, chi on the counts by
  `chi(a,b,c) = (a - (a^b) + (a^c) + (a^b^c))/2` with the parities of Sa, Sa+Sb, Sa+Sc,
  Sa+Sb+Sc (iota: Sa+1): 45 units per digest bit.

## Why it is exact

The fold only multiplies weight matrices: a feature `M_row @ wo @ h` equals
`M_row @ (features)`, and `wg @ read @ x` equals `wg @ (level values)`; M holds small
integers and wo dyadic fractions, so float32 products are exact and the harness and the
dense pipeline agree bit for bit. The identity for copies and outputs holds on bits,
and copies of integers stay linear. RMSNorm scales all features of a layer alike, and
the output is read relative to BOS; the norm weight keeps the approximation
`silu(k g)/k ~ max(0, g)` as tight as with bits. Pruning removes only units with no
path to an output and features no unit reads; dedupe merges only equal rows.
Structured messages (all 0, all 1, alternating, one-hot, 5%/95% density) stay within
2e-5 at log_w=4 and 1.4e-4 at log_w=6 (all ones is the worst; `probe/stress.py`).

## Compared with the old linear_out (branch glu, e0f5d02, removed in 1b27243)

linear_out marked a level linear only if all its blocks were linear, and folded only
trailing linear levels into the last wo (`new_wo = linear_m @ wo`). A linear level in
the middle became a SwiGLU of steps on integer values; copies of linear values were
steps at 1/2 (thresholding the integer); a level mixing linear nodes and gates made
steps of them; and nothing kept gates steep under RMSNorm with integer features. Here:
the layout is untouched (lin nodes level like gates), folding works on the matrices
after leveling at any position (backward into wo, forward into wg/wv, composed), copies
of linear nodes are linear, linear nodes in mixed levels get exact unit pairs, and
norm weights follow value bounds. Tests: `tests/fold_test.py` (36 tests, all inputs,
fold on/off, unit copies on/off, counts to 64); full suite passes.

## Composes with

Anything that changes units (cheaper theta or chi units, other gate tricks): the fold
only touches linear structure, and the gain per count grows with its fan-in. The
general parts (fold of input/output copy levels, prune/dedupe, unit copies) apply to
any compiled circuit: e.g. glu_chi_iota with only `--fold` is 8 -> 6 layers, 2.11x dense.
Round-1 constant folding generalizes if the tracer marked inputs (const() inputs and
constants look alike at trace time); here the variant knows the state layout.

## On other circuits, and a caveat on precision

The fold is not Keccak-XOF specific (`probe/general.py`, SHA3-like Keccak, 16 random
messages, all correct):

| circuit | fold | unit copies | depth | dense | sparse | margin |
|---|---|---|---|---|---|---|
| threshold xor, log_w=1, 3 rounds | no | no | 20 | 3,034,510 | 62,390 | 6.7e-4 |
| same | yes | no | 15 | 1,471,362 | 33,766 | 3.7e-4 |
| glu_xor, log_w=1, 3 rounds | no | no | 14 | 389,968 | 20,498 | 5.9e-4 |
| same | yes | yes | 9 | 149,185 | 10,008 | 5.4e-3 |
| glu_xor, log_w=2, 2 rounds | no | no | 10 | 1,113,130 | 27,906 | 6.1e-4 |
| same | yes | yes | 6 | 300,072 | 10,383 | 2.5e-3 |

(Single-bit threshold gates such as iota's `not_` are affine on bits and fold too.)
Caveat: a two-unit step is imprecise (~1e-3) but flat, so it cleans small errors;
glu units are precise (~e^-16) but pass input errors on, and glu_xor amplifies them
(slope up to ~2n). Folding removes the step copies and nots between glu layers, so in
circuits that still mix threshold gates with glu units the margin grows with depth
(5e-3 to 1e-2 above, still < 0.02). The Keccak variants here have no step gates left
(unit copies), and stay at ~3e-5 (1.4e-4 worst on structured messages at log_w=6).
For deep mixed circuits, keep copies as steps (no fold of pure copy levels) or use
clean glu units.

## Why L1 of glu_chi_iota was 2752 wide

Level 1 holds 1144 copies of the message bits inside `bitlist_to_msg` (dead: the pad
const() there is truncated to 0 bits, 8 dead constants), 1144 copies inside
`msg_to_state`, and the 456 suffix/capacity constants; level 2 reads 1600 of the 2752.
Copies appear because a const() inside a block puts a level-1 node there, and the
tracer copies the message to that level. With the fold this level is read through
(copies become direct reads, constants become biases) and prune drops the dead ones.

## Tried, and why it did not help

- chi of xors: theta computes only the 320 column-pair parities P (5 units each, shared
  by 5 bits) and copies A; chi reads A, P and computes chi(A1^P1, A2^P2, A3^P3). One
  unit is impossible (exhaustive integer search), and 2, 3, 4 units fail (gradient
  search, best max error ~0.52); at 5+ units (plus 2 per bit in theta) it is no cheaper
  than 6 theta units + 1 chi unit per bit, and needs 320 more features.
  (`search/k1.py`, `search/gd.py`.)
- Fewer units for 11-bit parity: 1D pieces on s need ceil(n/2) units, also with
  non-integer kinks (grid search, `search/parity1d.py`), and 2D gates on (A, D) do not
  help (4 units for D in 0..10: error 0.51, `search/gd2d.py`).
- Folding chi's u = 2a - b + c into theta's wo: a u row costs 18 wo entries (3 bits x 6
  units) against 6 per theta bit, and saves 4 entries per chi unit: +12.8K sparse per
  round.
- (A, D) split (D = 10 column bits as 320 features, s = A + D inline): log_w=4 dense
  7.05M vs 6.81M, sparse 52.1K vs 52.0K: the extra 320 features cost more than the
  shorter wo rows save.
- Depth 5 by merging the last round (45 units per digest bit): measured +15% dense,
  +24% sparse at log_w=6 (105.7M vs 91.8M, 222.7K vs 179.6K) for 1.2x depth; kept as a
  variant, not the best.
- (estimated, not run) An extra layer that materializes round 1's counts: theta1 would
  read 1 feature per unit (about -36K sparse) but +1 layer and ~+9% dense.
- (estimated, not run) A whole round in one layer (Walsh chi on counts for all 1600
  bits): ~45 units per bit (~32 in round 1), dense several times larger.
- (estimated, not run) Packing digests 4 bits per feature instead of 2: unpacking
  takes ~15 units per 4 bits in the last layer; net about -2% dense.
- (analysis) Theta needs 1600 independent counts in and 1600 bits out, and 6 units
  per bit, so theta2 >= 3 x 1601 x 9602 = 46M dense in this two-layers-per-round
  design; it is at 49.9M. Further 2x gains need fewer units per theta bit, which the
  searches above did not find.

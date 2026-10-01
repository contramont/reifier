# dce-const: dead code elimination and constant propagation in the tree compiler

General compiler change (no Keccak-specific code). Files: `compile/simplify.py` (new),
`compile/blocks.py`, `compile/tree.py`, `tests/simplify_test.py` (new).
`patch.diff` applies on top of `$S/xof/base` (`patch -p1`); `git_diff_full.diff` is
`git -C repo diff` plus the new files (it also contains base's own uncommitted changes).

## Results (log_w=6, 3 XOF steps, harness `ok: true`)

| circuit | depth | dense | sparse | hidden | margin |
|---|---|---|---|---|---|
| glu_chi_iota, base compiler | 8 | 247,067,600 | 541,724 | 43,152 | 0.00037 |
| **glu_chi_iota, this compiler** | **6** | **106,883,049** | **304,552** | 23,743 | 0.00037 |
| baseline (threshold xor), main | 20 | 3,305,445,348 | 2,034,788 | 187,752 | 0.00036 |
| **baseline, this compiler** | **15** | **1,643,110,919** | **1,180,571** | 113,034 | 0.00021 |

Gains: glu_chi_iota 1.33x depth, 2.31x dense, 1.78x sparse. Baseline 1.33x depth,
2.01x dense, 1.72x sparse. Nothing gets worse. Layers of glu_chi_iota, as (in, hidden, out) with BOS:
`[1145,6101,1468] [1468,1602,1601] [1601,10050,1825] [1825,2050,1825] [1825,2818,769] [769,1122,673]`
(theta1, chi1, theta2, chi2, theta3 on 320 live bits, chi3 on the 224 digest bits; the last
level is the outputs, so no copy-only output layer).

## Why glu_chi_iota's first layer was 2752 wide, and why it existed

Level 1 (checked by `scripts/why_l1.py`) was:
- 456 live constants: `msg_to_state` calls `const()` inside the traced function for the
  suffix (8) and capacity (448) bits. `const` is `gate([], [], t)`, so each is a traced
  gate with no inputs, and gates cannot sit on level 0 (inputs only).
- 8 dead constants: `bitlist_to_msg` builds the pad `const(pad_bitstr * 1)` and then
  truncates it to 0 bits, as the message is full length.
- 1144 live copies of the message, made inside `msg_to_state`: the layout requires a
  block's outputs at its top, and the constants give that block height 1, so the 1144
  message bits it passes through are copied.
- 1144 dead copies of the message, made inside `bitlist_to_msg` for the same reason (its
  8 dead constants give it height 1). Theta reads the `msg_to_state` copies.

So 456 + 8 + 1144 + 1144 = 2752. The layer exists only because constants were nodes: once
they fold into theta1's gate and value biases, theta1 reads the message directly, the
pass-through blocks have height 0, and no copies are made. In the last step, 1280 theta
and 1376 chi nodes were dead (only the 224 digest bits are outputs; they need the 320 theta
bits that rho-pi maps to row y=0), and the last layer only copied the 672 outputs.

## Method

1. `compile/simplify.py`, run on the traced bits before layout (`blocks.simplify_blocks`,
   which replaces `fold_untraced_bits`). Bits are processed in creation order. Each bit
   created by a gate or glu is resolved to either:
   - a constant: all inputs are constants, or a threshold gate saturates
     (`min(z) >= 0` or `max(z) < 0` over the 0/1 cube once constants are folded), or all
     of a glu's units are dead;
   - an alias `offset + sign * base` (copy or negation of one non-constant bit), for
     threshold gates and glus with a single non-constant input (flag `forward`, default on);
   - itself, with a simplified neuron on the remaining bits: constants and aliases folded
     into the weights and bias (threshold) or into both the gate and the value of every
     unit (glu), repeated bases summed, zero weights dropped. Glu units whose gate is never
     positive on the cube are dropped. Glu units with the same gate are merged (their
     values add).
   Untraced bits (created before tracing) are still constants, now with a real check that
   they do not depend on the inputs (the old check only looked one level up).
   Liveness: backward from the outputs through the simplified neurons. Gates of dead,
   constant and aliased bits are removed from the block tree. Block outputs are replaced
   by what consumers see (bases, live only). So pass-through blocks become height 0 and
   need no copies.
2. Outputs that are aliases or constants (in `simplify` and `add_output_blocks`):
   - a copy of a node: the output reads the node;
   - a negation of a node: computed beside that node from the node's own inputs (threshold
     gate: `-w`, `-b-1`; glu: `1 - sum(units)`, i.e. one more unit `max(0,1)*1` and negated
     values). The node dies if nothing else uses it. This keeps the negation on the node's
     level. Keccak's last iota then costs no layer in the baseline;
   - an alias of an input keeps its own node (at level 1, cheaper if it is a glu);
   - a constant becomes an output node with only a bias.
3. `compile/tree.py`, on the leveled graph:
   - `_merge_duplicate_nodes`: nodes on the same level with the same parents, weights,
     bias and units are merged, cascading upward. This includes duplicate copies. It
     never changes depth, unlike merging before layout, which can reorder blocks. Keccak:
     the 133 theta1 bits of all-constant lanes that share a column (x=3,4 capacity lanes,
     5 suffix bits).
   - `_remove_dead_nodes`: backward liveness per level (dead copies, merged duplicates).
   - `redundant_outputs_layer_order`: the output layer is dropped whenever it only copies
     the level below. The level below is then permuted into output order. Before, only the
     identity order counted. Repeated outputs duplicate their node, and constant outputs
     move down as bias-only nodes.

## Why it is exact

On 0/1 inputs, a threshold gate is `step(w.x + b)` and a gated unit is `max(0, g.x + b)
* (v.x + c)`. Both are affine in x inside the nonlinearity. Substituting a constant, or
`1 - y` for a negation, gives the same function with folded weights and biases. SwiGLU
computes these same affine gates and values (`silu(16 r g) r v / 16`). A dropped unit is
`max(0, g) = 0` on every input (in SwiGLU it only added about `-1e-6`). Merged units:
`max(0,g)v1 + max(0,g)v2 = max(0,g)(v1+v2)`, also exactly in SwiGLU. Integer threshold
negation: `not (z >= 0)` is `-z-1 >= 0`. Dead nodes do not reach the outputs. Merged level
nodes have identical rows, so they compute identical floats. The dropped output layer only
copied nodes. Checks:
- `validate_bench.py`: weights are bit-equal to the dense Compiler -> MLP_SwiGLU pipeline,
  log_w 0-2, baseline and glu_chi_iota.
- Full repo test suite: 57 existing tests plus 30 new tests pass.
- `scripts/edge_cases.py` and `tests/simplify_test.py` check every input on SwiGLU and on
  MLP_Step: constant, negated, duplicate and pass-through outputs, saturated gates, glu
  with dead units, deep dead code, with and without forwarding.
- The harness at log_w=6: margin 0.00037 with 16 random messages (same as before) and
  0.00043 with 64.
- `validate_bench.py` again on the final code, also for glu_chi_unit, which uses aliases
  and glu negations.

## Ablations (log_w=6, harness `ok: true`, `ablate.py`, lines in `ablations.jsonl`)

Each row switches one part off (or, for `foldonly`, everything but constant folding).
The "all off" rows are the brief's harness numbers for the base compiler.
`ablate:glu_none` and `ablate:base_none` reproduce the base compiler exactly: same widths,
dense, sparse and margin, checked at log_w 2 and 3.

| glu_chi_iota | depth | dense | sparse | note |
|---|---|---|---|---|
| all on | 6 | 106,883,049 | 304,552 | |
| all off (`none` = base compiler) | 8 | 247,067,600 | 541,724 | |
| only constant folding (`foldonly`) | 7 | 176,846,223 | 450,760 | L1 gone, theta1 4.14 units/bit |
| no constant folding (`nofold`) | 7 | 124,961,764 | 354,667 | L1 back: 1144 copies, 456 consts merged into one 0 and one 1; theta1 6 units/bit; margin 0.0075 |
| no permuted output-layer drop (`nodrop`) | 7 | 109,601,296 | 310,607 | |
| no level merge (`nomerge`) | 6 | 110,171,304 | 310,721 | |
| no level-wise DCE (`nolevel`) | 6 | 110,171,304 | 310,721 | merged duplicates stay |
| no block-level DCE (`noprune`) | 6 | 106,883,049 | 304,552 | level DCE catches it here |

| baseline (threshold xor) | depth | dense | sparse | note |
|---|---|---|---|---|
| all on | 15 | 1,643,110,919 | 1,180,571 | |
| all off (`none` = main) | 20 | 3,305,445,348 | 2,034,788 | |
| no forwarding (`nofwd`) | 18 | 1,680,685,940 | 1,217,368 | iota's NOT layers stay |
| no constant folding (`nofold`) | 16 | 1,947,977,115 | 1,334,635 | |
| no permuted output-layer drop (`nodrop`) | 16 | 1,645,829,166 | 1,186,626 | |
| no level merge (`nomerge`) | 15 | 1,712,083,473 | 1,204,273 | |

The margin of `nofold` (0.0075 instead of 0.00037) shows why the old circuits looked
precise. The old last layer copied the outputs with step units, and a step unit is flat
around 0 and 1, so it hid the error of the glu layers below it. With constants folded, theta1
reads exact inputs instead of step outputs, and the final glu layer stays at 0.00037
without that cleanup.

Block-level DCE is still needed in general. Dead code deeper than the live code puts the
outputs on a later level, and level DCE cannot undo that
(`test_dead_code_adds_no_depth`: 5 layers before, 1 now).

## Other variants (log_w=6, harness `ok: true`; base compiler numbers from the brief's `glu_w6.jsonl`)

| variant | base compiler | this compiler | margin |
|---|---|---|---|
| glu_xor_everywhere | 14 / 473,310,074 / 683,450 | 9 / 208,795,543 / 371,192 | 0.0015 |
| glu_chi_unit (iota as NOT gates) | 11 / 307,626,437 / 590,981 | 6 / 107,089,613 / 304,704 | 0.00037 |
| glu_chi_iota | 8 / 247,067,600 / 541,724 | 6 / 106,883,049 / 304,552 | 0.00037 |
| glu_clean_chi_iota | 8 / 389,731,600 / 1,410,524 | 6 / 175,184,185 / 796,790 | 0.00037 |

Forwarding makes the hand-fused iota of glu_chi_iota unnecessary: glu_chi_unit compiles to
nearly the same circuit. Its few extra units are the `1 - chi` negations of flipped digest bits.

## Cross-check with the hand-written Keccak compiler

`examples/keccak_compile.py` folds the suffix and capacity bits and fuses iota by hand, for
threshold Keccak digests. The general tree compiler now builds smaller circuits than it
(`scripts/vs_direct.py`, level sizes incl. inputs):
- log_w=3, n=2: direct `[176, 2200, 200, 400, 400, 200, 2200, 200, 400, 400, 200, 8]`, tree
  `[176, 1936, 200, 400, 400, 200, 264, 24, 16, 16, 8]`.
- The tree compiler also drops the counters that constants saturate, the dead part of the
  last round, and the separate output level.

## Composes with

Everything that changes the circuit or the realization: theta restructuring, unit
synthesis, cheaper copies or BOS units, XOF restructuring. The last-round DCE shrinks
any theta/chi form automatically. Constant folding and unit pruning apply to any first
round. A two-layer theta, for example, would keep only the column parities the 320 live
bits need. What is left in glu_chi_iota: theta layers are 80% of dense (theta2 alone 47%).
Digest copies (1344 copy nodes, 2 units each) are about 15%. So the next factor has to come
from theta or from cheaper copies, not from DCE.

## Extra, not in the patch: one-unit clean gates (`cleancopy.py`, `extra.jsonl`)

This is cheap-gates territory, but it composes with this patch and is the same kind of
trick as the compact xor, so it is measured here. A threshold node that fires only at the
top of its range (`max z = 0` on the cube: copies, NOTs, AND, inhib, the top xor counter)
is exactly the one gated unit `max(0, 2z+1) * (1-2z)`. That is 1 at z=0 with zero slope, and
0 for z<=-1, so it is flat at both bits like the two-unit step. A copy is
`max(0, 2x-1) * (3-2x)`. It is applied after `Tree.from_root` (log_w=6, harness `ok: true`):
- glu_chi_iota: 6 / 101,560,361 / 300,520 (margin 4.6e-5, lower than the step's 3.7e-4);
  all 1344 digest copies become one unit each. Total vs base compiler: 2.43x dense.
- baseline: 15 / 1,455,595,791 / 1,119,662 (all copies, inhibs and AND counters). Total vs
  main: 2.27x dense.
Gates that fire on more than one z value (OR-like xor counters) still need two units.

## Tried, did not work, or left out

- Keeping aliases that are outputs as nodes in place: iota's NOT on a digest bit gives
  the `iota` block height 1. The block layout then starts every later sibling (the next
  hash step) a level later (baseline log_w=3: 18 layers instead of 15).
- Hoisting such nodes to the root: better (16), but a negation of a node on the top level
  then needs an extra level. Computing it beside its base from the base's inputs gives 15.
- Computing aliases in the output blocks: then the output layer is not a pure copy layer
  and cannot be dropped, and a glu identity (1 unit) becomes a 2-unit step copy.
- Common subexpressions before layout (on bits): exact, but a consumer can then depend
  on a block that is placed later, which can add depth. Merging per level cannot.
- Negation twins in merging: theta1 bits of the suffix lane are NOT of a capacity-lane
  bit, but two glus that are negations of each other cannot be recognized from their
  units. It is 3 nodes.
- Unit pruning beyond dead gates: theta1 after folding is within 15 units of
  `ceil(n'/2)` per bit (6099 units in all), so re-synthesizing folded xors gains nothing.
- Beside-negation places the negation early (ASAP). If its base is also used later, the
  old NOT-node placement needs one copy fewer (`scripts/edge_cases.py`, "not of gate, also
  used"). That is irrelevant for the XOF, where the counts are equal.

## Reproduce

```
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
export PYTHONPATH=$S/xof/av/dce-const/repo/src:$S/xof/av/dce-const:$S/xof
$S/venv/bin/python $S/xof/xofbench.py --log-w 6 --depth 3 --variant variants:glu_chi_iota --widths
$S/venv/bin/python $S/xof/xofbench.py --log-w 6 --depth 3 --widths          # baseline
$S/venv/bin/python $S/xof/xofbench.py --log-w 6 --depth 3 --variant ablate:glu_nodrop
$S/venv/bin/python $S/xof/validate_bench.py variants:glu_chi_iota
```
`TreeCompiler(forward=False)` switches alias forwarding off. `Tree.from_root(root,
remove_redundant_outputs_layer=False)` keeps the output layer.

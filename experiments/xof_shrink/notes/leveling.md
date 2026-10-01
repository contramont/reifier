# leveling: layer assignment and output structure

Everything here is a general compiler change (`src/reifier/compile/blocks.py`,
`src/reifier/compile/tree.py`) plus tests (`tests/leveling_test.py`, 11 tests).
No Keccak-specific code: the circuit is the unchanged `variants:glu_chi_iota`, and the
harness is used as is (it calls `TreeCompiler()` with defaults, which now include all
passes below).

## Result (log_w=6, 3 XOF steps, harness `ok: true`)

| | depth | dense | sparse | hidden | margin |
|---|---|---|---|---|---|
| baseline (main, threshold xor) | 20 | 3,305,445,348 | 2,034,788 | 187,752 | 3.6e-4 |
| `glu_chi_iota` on base compiler | 8 | 247,067,600 | 541,724 | 43,152 | 3.7e-4 |
| **`glu_chi_iota` with this patch** | **6** | **96,861,513** | **297,832** | 22,175 | 8.7e-5 |

Gains vs glu_chi_iota: depth 1.33x, dense 2.55x, sparse 1.82x.
vs baseline: depth 3.33x, dense 34.1x, sparse 6.83x.

Layers (in, hidden, out), BOS included:
`[1145,6101,1468] [1468,1602,1601] [1601,9714,1713] [1713,1714,1713] [1713,2146,545] [545,898,673]`
= theta1, chi1, theta2 (+digest-1 pairs), chi2 (+d1), theta3 (only the 320 bits the
digest needs, +d1,d2 pairs), chi3 (only 224 bits) writing all 672 outputs in order.

The same compiler on the other circuits (log_w=6, ok):

| circuit | before | after |
|---|---|---|
| default xof (threshold gates) | 20 / 3,305,445,348 / 2,034,788 | 15 / 1,937,204,428 / 1,290,666 |
| glu_xor_everywhere | 14 / 473,310,074 / 683,450 | 9 / 170,513,331 / 343,201 |
| glu_chi_unit (iota as NOT gates) | 11 / 307,626,437 / 590,981 | 6 / 96,907,269 / 297,856 |
| glu_clean_chi_iota | 8 / 389,731,600 / 1,410,524 | 6 / 163,549,849 / 790,070 |

Note `glu_chi_unit`: the generic NOT collapse (below) fuses iota into chi by itself,
matching the hand-fused `glu_chi_iota` to within 4 units per chi layer.

## Why glu_chi_iota had 8 layers

- **L1 [1145 -> 2752]** did no work: 456 constants (suffix and capacity: `const()`
  inside the traced function makes traced gates, which take a level of their own and
  push everything that reads them one level up), 8 dead pad constants
  (`bitlist_to_msg` makes 8 pad bits and truncates them to 0 bits), and the 1144
  message bits copied **twice**: once inside `bitlist_to_msg` and once inside
  `msg_to_state`. A block that contains a gate is at least one level tall, so bits it
  passes through get copies up to its top; the first set of copies is dead.
- **L8** was output copies: `has_redundant_outputs_layer` only dropped the outputs
  level if it copied the level below exactly (same width, same order).
- The last round computed all 1600 bits; the digest needs 224 chi outputs, which need
  the 320 theta outputs of lanes (x, x) (the rho-pi preimages of row y=0).

## Changes, and why each is exact

1. **`prune_blocks` (blocks.py), before `assign_inputs`**: a traced gate that does not
   depend on the inputs (constant propagation in uid = creation order, so it also
   catches gates on constants only, glu included) is removed, and its consumers fold
   it into their bias through the existing untraced-bit path of `Tree._gate_origin`
   (which uses the traced activation, the true constant). Gates that no output
   depends on (backward reachability from the outputs) are removed; block outputs are
   filtered to needed bits so no copies are made for them. Constant outputs, traced or
   untraced, get an output node with only a bias (untraced constant outputs used to
   crash with "io created outside of the tree"; they are now supported).
   Option `TreeCompiler(prune=...)`, and `prune_blocks(drop_dead=False)`.
2. **`Tree.collapse_copy_levels`** (replaces `has_redundant_outputs_layer`): a level
   whose nodes each copy or negate one node of the level below, or are constants,
   merges into that level. Every node of a level is only read by the next level, so the
   merged level computes `src`, `1 - src` or the constant directly: a copy takes the
   origin of its source; a NOT takes the negated origin, `[w.x+b >= 0]` ->
   `[-w.x-b-1 >= 0]` (exact for integer pre-activations), or for gated units a
   constant unit `max(0,1)*1` plus the units with negated values; a node read twice
   is repeated. The outputs level is always such a level, so the last computing level
   writes the outputs in their order (repeats and pass-through inputs work). It also
   removes each NOT+copy iota level of the threshold circuit and of glu_chi_unit.
   Option `collapse_copies`.
3. **Gated-unit copies** (`COPY_UNIT = max(0, x) * 1`): one hidden unit instead of the
   two-unit step, used automatically when the tree has gated units (the target is then
   SwiGLU anyway; `MLP_Step` rejects gated units). In SwiGLU it is
   `silu(16 r x)/16 * r ~ r^2 x`: exact at x=0, relative error e^-16r at x=1. Unlike a
   step it does not clean input errors, but it does not amplify them either (slope 1 at
   1, ~1/2 near 0); margins went down, not up (3.7e-4 -> 4e-5), because the steps'
   silu tails were the largest error source. Option `glu_copies`.
4. **`Tree.simplify_levels`**: merges nodes of a level with equal
   (parents, weights, bias, units), then removes nodes the next level does not read.
   After constants fold in, theta1 outputs t[x][y][z] = a[x][y][z] ^ D[x][z] whose own
   bit a is the same constant for several y (capacity lanes) are identical: 1600 -> 1467
   theta1 nodes at log_w=6. This pass also removes the dead last-round nodes, so
   block-level dead-gate removal adds nothing for xof.
5. **Never-firing units** (`Tree._unit_is_live`): after constants fold into a gated
   unit, its gate may be <= 0 on every Boolean input (the upper units of an 11-input
   compact xor when only 7-9 inputs are live); such a unit, or one with value
   identically 0, contributes 0 and is dropped. theta1: 8802 -> 6099 units.
6. **`Tree.pack_copy_chains`**: two bits copied along the same levels are carried as
   one feature p = (a+2b)/3: packed by `max(0,1)*(a/3+2b/3)`, carried by `max(0,1)*p`,
   unpacked on the last level of the run with q = 3p: b = max(0,q-1)(2-q/2) (0,0,1,1
   at q=0..3), a = q - 2b; every gate is integer-valued at q in {0..3}, as for other
   units. 2n units become n+2 for a run of n levels, with half the features. p is kept
   in [0,1] on purpose: with p = a+2b (up to 3) the RMSNorm scale fell below 1, silu
   was a worse ReLU, and the margin rose to 9e-4; scaled, it is 4e-5..9e-5.
   Requires copies to carry bits, as all traced Bits do (glu asserts 0/1). Default: on
   with gated copies; option `pack_copies`.

The harness equals the dense pipeline (`validate_bench.py`, weights bit-equal, log_w
0-2) for glu_chi_iota and for the default circuit (which exercises the NOT collapse).
Repo tests: 68 passed (57 existing + 11 new).

Precision: the outputs are no longer cleaned by a final layer of step copies, so the
margin is that of the last computing level. It does not grow with more XOF steps
(log_w=3, 3/6/10 steps): glu_chi_iota 4.0e-5 / 4.6e-5 / 4.5e-5 (base compiler ~4.5e-4);
glu_xor_everywhere, whose chi mixes step gates into glu xors, 2.6e-3 / 3.0e-3 / 3.0e-3
(base ~4.5e-4), still far below 0.02. Depth there: glu_chi_iota 20 vs 22 at 10 steps
(the first and last layers), glu_xor_everywhere 30 vs 42 (plus one iota level a round).

## Ablation (log_w=6, glu_chi_iota; all ok) -- see results.jsonl

| passes on | depth | dense | sparse |
|---|---|---|---|
| none (= base) | 8 | 247,067,600 | 541,724 |
| gated copies only | 8 | 224,663,648 | 520,204 |
| collapse only | 7 | 230,641,641 | 523,285 |
| constant folding only | 7 | 188,418,057 | 482,798 |
| prune (constants + dead gates) | 7 | 124,461,385 | 348,814 |
| + collapse | 6 | 121,743,138 | 342,759 |
| + gated copies | 6 | 116,420,450 | 336,039 |
| + simplify | 6 | 111,718,235 | 327,181 |
| + unit pruning | 6 | 101,560,361 | 297,832 |
| + pair packing (= defaults) | 6 | 96,861,513 | 297,832 |

Depth: constant folding removes L1, the collapse removes the outputs layer. Dense:
the first layer (-59M) and the dead last-round gates (-64M) dominate, then unit pruning
(-10M), gated copies (-5M), merged theta1 nodes (-5M), packing (-5M), collapse (-3M).

## ASAP/ALAP and the digest copies

`scripts/slack.py`: in the final leveled graph every one of the 6,811 computing nodes
has zero slack (ASAP level = ALAP level), so no re-leveling can move work. The only
copies are the digests: d1 (224 bits, made on level 2, output on 6) and d2 (made on 4),
1,344 copy units unpacked, 1,120 packed; they are forced, since d1 is part of state 1,
which theta2 reads on level 3. Recomputing d1 late would need 320 theta1 bits carried
instead of 224 digest bits.

## Composes with

The passes are generic, so they apply to any circuit variant: better theta/chi units
(unit-synthesis, theta-structure) compose directly, as do linear folds. Overlaps:
`prune_blocks` is constant folding + DCE (the dce-const avenue); gated copies may
overlap cheap-gates. If another avenue introduces non-Boolean traced values, turn
`pack_copies` off or restrict it (it assumes copies carry 0/1).

## Tried, or considered, and not done

- Merging theta and chi into one layer (depth 5 or 3): chi of three 11-bit parities
  needs a sum of hinge-times-linear units fitting a checkerboard in 3 sums; chi of
  three 2-bit xors already needs ~6-8 units (search below), so tens of units per bit,
  more dense than the layer saves.
- Packing 3+ bits per feature: unpacking k bits needs ~2^k units, so pairs are the
  sweet spot (a run of n >= 3 levels gains n-2 units per pair; n = 2 only narrows).
- Pairing chains with overlapping but unequal level ranges: no such chains in xof.
- Sharing one hidden unit between the two unpack nodes (their `max(0,q-1)` units are
  proportional): -112 units, ~0.2% dense, needs a Matrices/SwiGLU change. Not done.
- BOS as one gated unit instead of two: 6 units total, ~0.03%. Not done (changes hidden
  sizes asserted by existing tests).
- Block-level dead-gate removal on top of the level cleanup: redundant for xof (same
  numbers), kept because dead gates can otherwise add levels in the layout.

## Where the rest of the cost is

Dense per layer: theta2 47.7M (49%), theta1 22.9M (24%), chi2 8.8M, theta3 8.5M, chi1
7.3M, chi3 1.6M. Sparse is ~85% theta units (each of the 6 units of an 11-input xor
reads all 11 inputs: ~94 nonzeros per theta bit vs 8 per chi bit). Leveling has no
slack left, so further 2x gains need fewer or narrower theta units. One direction
for other avenues, estimated only (not measured): a chi layer that also outputs the
column sums of its bits (free: extra `wo` rows, but non-Boolean nodes, which glu()
rejects today) lets theta's gates read 3 features instead of 11: sparse of a theta
layer ~2.5x smaller, its dense ~7% larger.

Checked and rejected: splitting theta across the two layers of a round (layer A: the
320 column terms D[x][z], 10-input xors, 1600 units, plus copies of the state; layer B:
chi(a^Da, b^Db, c^Dc) directly). A round would cost ~16.4M + k*8.7M dense vs 54M now,
so it needs k <= 3 units for chi of pair-xors. `scripts/units_search.py` (batched
restarts of Adam, silu annealed to the exact relu form; it recovers the known 1-unit
chi to 2e-3) finds best max errors 0.50 / 0.50 / 0.46 / 0.38 / 0.10 / 0.043 / 0.004 for
k = 1 / 2 / 3 / 4 / 5 / 6 / 8: k is about 6-8, so the split would cost more.

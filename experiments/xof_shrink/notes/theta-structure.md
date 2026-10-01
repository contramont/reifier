# theta-structure: restructuring theta and chi for 1-round Keccak XOF

All numbers: `xofbench.py --log-w 6 --depth 3`, `ok: true` (every output within 0.02 of
its bit, relative to BOS). Reference: `variants:glu_chi_iota` = depth 8, dense 247,067,600,
sparse 541,724 (hidden 43,152); main baseline = 20 / 3,305,445,348 / 2,034,788.

Code: the construction is `repo/src/reifier/examples/keccak_theta.py` (new file, in the
patch); `tsx.py` in this directory holds the xofbench variants (it sets the compiler flags
and the round layout). Run e.g.

    A=$S/xof/av/theta-structure
    PYTHONPATH=$A/repo/src:$A:$S/xof $S/venv/bin/python $S/xof/xofbench.py \
        --log-w 6 --depth 3 --variant tsx:lazy_middle_pack

`tsv.py` has the earlier exploration variants (same ideas, step by step; its rows below
were measured with the code of that time), `compiler_only.py` the compiler-only rows.
`patch.diff` is `git diff` of the repo (incl. the base's uncommitted changes) plus new
files; `patch_vs_base.diff` is only this avenue's change against `$S/xof/base`.

## Results (log_w=6, 3 XOF steps)

| variant (round layouts) | depth | dense | sparse | hidden | vs glu_chi_iota depth/dense/sparse |
|---|---|---|---|---|---|
| **tsx:lazy_middle_pack** [2,L,2] + packed digests | **6** | **78,505,505** | **168,264** | 20,561 | 1.33x / 3.15x / 3.22x |
| tsx:lazy_middle [2,L,2] | 6 | 82,904,753 | 167,816 | 20,561 | 1.33x / 2.98x / 3.23x |
| tsx:two_layer [2,2,2] | 6 | 100,566,062 | 181,466 | 22,384 | 1.33x / 2.46x / 2.99x |
| tsx:three_first_lazy_pack [3,L,2] | 7 | 71,933,215 | 122,078 | 18,496 | 1.14x / 3.43x / 4.44x |
| tsx:three_first_lazy [3,L,2] | 7 | 76,332,463 | 121,630 | 18,496 | 1.14x / 3.24x / 4.45x |
| tsx:three_middle [2,3,2] | 7 | 75,067,509 | 156,193 | 17,586 | 1.14x / 3.29x / 3.47x |
| tsx:three_first_middle_pack [3,3,2] | 8 | 63,818,099 | 109,559 | 15,297 | 1.00x / 3.87x / 4.94x |
| tsx:three_first_middle [3,3,2] | 8 | 68,495,219 | 110,007 | 15,521 | 1.00x / 3.61x / 4.92x |
| *ablations of [2,L,2]* | | | | | |
| tsx:lazy_middle_base (base compiler, no fold) | 8 | 160,450,451 | 274,986 | 31,825 | 1.00x / 1.54x / 1.97x |
| tsx:lazy_middle_fold (+ constants folded) | 7 | 88,120,152 | 178,351 | 22,803 | 1.14x / 2.80x / 3.04x |
| tsx:lazy_middle_fold_perm (+ output permutation) | 6 | 85,401,905 | 172,296 | 21,457 | 1.33x / 2.89x / 3.14x |
| tsx:two_layer_base (free sums + last-round DCE only) | 8 | 179,854,928 | 290,876 | 34,096 | 1.00x / 1.37x / 1.86x |
| tsx:three_middle_base | 9 | 154,356,375 | 265,603 | 29,298 | 0.89x / 1.60x / 2.04x |
| tsv:sums_chi_iota (glu_chi_iota + free sums only) | 8 | 243,383,888 | 338,236 | 41,776 | 1.00x / 1.02x / 1.60x |
| tsv:colsum_chi_iota (320 column sums, superseded) | 8 | 261,387,600 | 366,364 | 43,152 | 1.00x / 0.95x / 1.48x |
| *compiler changes alone* | | | | | |
| compiler_only:glu_chi_iota_perm | 7 | 230,641,641 | 523,285 | 40,430 | 1.14x / 1.07x / 1.04x |
| compiler_only:glu_chi_iota_perm_gc | 7 | 206,291,057 | 502,837 | 35,654 | 1.14x / 1.20x / 1.08x |

vs the main baseline: lazy_middle_pack 3.33x / 42.1x / 12.1x; three_first_middle_pack
2.50x / 51.8x / 18.6x. Margins 2e-5..7e-3 (lazy_middle_pack: 0.0023, also with 128 messages).
lazy_middle_pack widths (in, hidden, out):
`[1145,6086,1601] [1601,1602,1921] [1921,3202,1713] [1713,4915,657] [657,3858,545] [545,898,673]`
= first theta, chi, X (D + E), lazy chi, last theta (320 bits), last chi + unpacking.

## Why theta dominates, and the methods

`theta[x][y][z] = A[x][y][z] ^ D[x][z]`, `D[x][z]` = parity of the 10 bits of columns x-1
(at z) and x+1 (at z+1). glu_chi_iota spends 6 units per theta bit (11-input glu_xor,
each unit reading 11 inputs) and 1 unit per chi bit: 7 units per state bit per round.
Parity of an integer in [0, top] needs ceil(top/2) units in one layer (glu_xor's
construction; searches below found nothing cheaper), so the lever is what is summed.

### 1. Free sums (sparse; the enabler for everything else)
A layer's outputs are `wo @ units`, so any linear combination of one layer's units is a
free output: no hidden units, only wo nonzeros. The chi layer outputs, per theta bit of
the next round, `s = A + sum(column x-1) + sum(column x+1)` (a sum of 11 chi units,
scaled by 1/8), and theta = `parity(s)`, s in [0, 11]: the same 6 units, now reading 1
input instead of 11 (theta layer nonzeros 95 -> 24 per bit).
- `glu_sum` / `glu_terms` (operations.py) build a non-boolean glu node (`glu(...,
  boolean=False)`) from the units of other glu nodes or from (inputs, units) terms, so no
  node is created for the terms (no dead outputs).
- `Matrices.layer_to_units` shares one hidden unit among units that are equal up to a
  positive gate scale and a value scale (the scale goes to wo), so the sum costs nothing.
- Exact: s is an exact sum of exact 0/1 unit outputs. RMSNorm divides all features by
  their RMS and the silu-to-relu approximation needs that scale not to be small, so free
  sums are scaled down (1/8; lazy sums 1/16; packed digests 1/4). Unscaled sums of 11 bits
  gave wrong outputs (margin 1.05).
- Per-bit sums beat 320 column sums (`tsv:colsum_chi_iota`: theta reads 3 inputs, but
  the 320 extra features widen the theta layer: dense 0.95x).

### 2. Last-round DCE
The last step outputs only its 224 digest bits, which need 320 theta bits (the 5 lanes of
row 0 after rho-pi): the last theta layer computes 320 theta bits instead of 1600, and the
chi layer before it outputs 320 sums (`needed_theta`). Chi layers create nodes only for
digest bits; other chi outputs exist only inside free sums.

### 3. Constants folded at the source, constant-aware first theta
The suffix and capacity bits are untraced constants (`free_const`), so the first theta
reads the message directly (no copy/const layer) and `parity_of` spends units only on the
variable bits, ceil(n_var/2) (an odd constant flips one input: same count). First theta:
6,084 units for 1600 bits instead of 9,600. (Overlaps dce-const; here it is the theta side.)

### 4. D once per column pair, E = A + D as a free output: layer X
X computes `D[x][z] = parity(P)`, P the free sum of the 10 column-pair bits from the chi
layer (5 units reading 1 input, once per column pair = 1 unit per bit), plus a copy unit of
A per bit; `E = A + D` in {0,1,2} is a free sum of the copy and D's units, and
theta = [E == 1]. Three-layer round (kind 3): X, then Y = `parity(E) = max(0,E)(2-E)`
(1 unit), then chi: 3 units per theta bit instead of 6, for one more layer.

### 5. Lazy chi, exact only mod 2 (kind "L", the round before the last): 2 layers
The next theta is a parity of a sum, so it needs its inputs only mod 2, and
`chi = a ^ q = a + q - 2aq` is congruent to `a + q`. After X, one layer computes
`o' = [E_a == 1] + q(E_b, E_c)` in {0,1,2}, q = [E_b != 1][E_c == 1], straight from E:
- `[E_a == 1] = max(0, E_a)(2 - E_a)`: 1 unit;
- `q = max(0, 3 - 4E_b - 2E_c) E_c + max(0, 4E_b - 2E_c - 5) E_c`: 2 units (derived by
  hand, checked on all 9 points; each unit is 1 on one of (0,1), (2,1) and 0 elsewhere; one
  unit cannot do both, as their midpoint (1,1) must stay 0);
- iota: `1 - [E_a == 1]`, with one shared constant unit.
3 units per bit instead of Y + chi (2 units, but 2 layers) or an exact chi of xor pairs
(4 units, see below). The last theta then takes parities of sums in [0, 22] (11 units,
but only for 320 bits in a 657-wide layer), and this round's digest bits get their exact
value one layer later as `parity(o')`, 1 unit each. Mod-2 variants with fewer units exist
(`o' = E_a + q`, `q = E_c(1 - E_b)` in 1 unit) but widen the sums the last theta must
reduce (range 33..56) and cost more there than they save.

### 6. Digests carried two bits per feature (XOF-specific, composes with xof-structure)
Earlier digests ride through every later layer (no residual stream). Packed as
`p = (b0 + 2 b1)/4` (a free sum of the two units that make the bits), a pair costs one copy
per layer, and 3 units in the last layer to unpack (`b1 = max(0,4p-1) - max(0,4p-2)`,
`b0 = 4p - 2 b1`; the unpack units read a node of the layer before with weight 0 so that
the tracer puts them in the last layer). -5.3% dense for lazy_middle (82.9M -> 78.5M).

### 7. General compiler changes this needed (also useful alone)
- unit sharing in `Matrices.layer_to_units` (above);
- `Tree.from_root`: the outputs layer is dropped when it copies distinct nodes of the
  level below in any order (the level below then emits them in output order; nodes it
  no longer emits were dead). The old rule only dropped an identity copy of the whole
  level (flag `tree.PERMUTE_OUTPUTS`). Caveat: the last layer's outputs are no longer
  cleaned by threshold gates; glu_chi_iota then has margin 0.019 at log_w=0 (0.0074 at
  log_w=6), the tsx variants 2e-5..7e-3;
- `tree.GLU_COPIES`: copies as one gated unit `max(0, 1) * x` (exact for any value, gate
  on BOS) instead of a two-unit threshold gate; off by default, the tsx variants set it;
- `glu_parity(x, top, weights, bias)`: parity of an integer weighted sum.
validate_bench (weights bit-equal to the dense Compiler pipeline at log_w 0-2) passes for
tsx:lazy_middle_pack, lazy_middle, three_first_lazy_pack, three_first_middle_pack,
two_layer, lazy_middle_base and the tsv variants. New tests `tests/keccak_theta_test.py`
(free sums cost no units; all layouts compute the reference XOF, compiled and eager); the
repo's test suite passes (66 tests, 11 of them new; hash_long_test not run). Both patches
apply cleanly (patch.diff to HEAD, patch_vs_base.diff to $S/xof/base).

## What composes
- dce-const / leveling: method 3 is the theta side of constant folding; the ablation rows
  show the split (lazy_middle_base 160M -> + fold 88M -> + output permutation 85M ->
  + glu copies 83M).
- xof-structure: method 6 (packing) and last-round DCE are XOF-level; the digests of steps
  1 and 2 still cost ~7M of 78.5M (carried features, their copies and parity/unpack units).
- unit-synthesis: any cheaper parity of an integer in [0, top] lowers every theta cost here
  directly (top 11: 6 units, top 10: 5, top 22: 11, first round top 6-9).
- cheap-gates: GLU_COPIES is the same idea for copies.

## Searches and ideas that did not pay off
Tool: `vp.py` (variable projection: gates by Adam with a softened relu, values by least
squares, 160-200 restarts, then snap to integers and check exactly); `synth.py` an
exhaustive pair search (too slow for 6 inputs on the loaded machine).
- chi of xor pairs, `F6 = chi(A1^D1, A2^D2, A3^D3)` ("defer the xor with A into chi"): no
  2- or 3-unit decomposition (best losses 0.092, 0.014); 4 units fit (loss 1e-8, real
  gates). With T = D + copies that is 6 units/bit instead of 7, but the chi layer then
  reads 1920 features with 4x the units: only -6..-12% dense. In E = A + D space (27
  points) also 4 (3: loss 0.0056). The lazy form (3 units) is what works.
- `Q4 = ~(A2^D2) & (A3^D3)`: 2 units (exact), 1 impossible.
- Sharing theta units among the 5 bits of a column (s_y = A_y + T): the per-bit part
  `A_y (1 - 2 P(T))` alternates in T, ~5 units per bit, no gain over 6 (unit-synthesis
  reaches the same conclusion).
- Parity on 2-D grids (s given as two or three free sums): no fewer units than 1-D
  (2 units on [0,3]^2: loss 0.125; 4 units on [0,5]^2: loss 0.076).
- Carries: AND of two chi outputs of a column takes 3 units (XOR: more than 3); two carries
  per column cut theta from 6 to 4 units but cost 640 x 3 chi-layer units: ~-8% dense,
  more sparse.
- Deferring the D of chi's `a` input to the next layer; a lazy first round ([L,L,2]: 94M);
  a lazy round before a three-layer round: the next D then reduces sums of 20 bits
  (10 units instead of 5), which costs more than it saves.
- First round as X + exact chi of pairs (4 units) or X + lazy chi: 35-42M vs 32M.
- Column parities instead of D (3 units per column, then a 3-bit xor per bit): 3.6 units
  per bit instead of 3.
- One-layer rounds (chi of three parities, or last theta + chi): far too many units;
  depth 6 = 2 layers per round is the floor here.
- Lane complementing: NOT is free (affine) in this model, nothing to save.

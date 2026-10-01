# xof-structure: XOF- and Keccak-specific simplifications

All numbers: `xofbench.py --log-w 6 --depth 3`, `ok: true`. References:
glu_chi_iota 8 / 247,067,600 / 541,724; baseline 20 / 3,305,445,348 / 2,034,788
(depth / dense / sparse).

## Pareto front of this avenue (one best circuit per depth)

| depth | variant | dense | sparse | vs glu_chi_iota (depth / dense / sparse) | vs baseline |
|---|---|---|---|---|---|
| **4** | **`xs4:d4a_m4k3`** (rounds 2 and 3 each in ONE layer) | 162,246,830 | 420,226 | **2.00** / 1.52 / 1.29 | 5.00 / 20.4 / 4.84 |
| 5 | `xs4:d5_m3k2` (column-parity lazy chi + Walsh last round + packed digests) | 91,133,210 | 224,883 | 1.60 / 2.71 / 2.41 | 4.00 / 36.3 / 9.05 |
| 6 | `xs4:d6_m3k2` (= lazy_middle + packed digests) / `xs3:lazy_middle` | 75,259,962 / 76,805,192 | 169,549 / 166,687 | 1.33 / 3.28 / 3.20, 1.33 / 3.22 / 3.25 | 3.33 / 43.9 / 12.0 |
| 7 | `xs3:split_middle` / `xs3:split_first_lazy` | 68,688,604 / 70,156,703 | 154,268 / 120,238 | 1.14 / 3.60 / 3.51, 1.14 / 3.52 / 4.51 | 2.86 / 48.1 / 13.2 |
| 8 | `xs3:split_first_middle` | 62,040,115 | 107,819 | 1.00 / 3.98 / 5.02 | 2.50 / 53.3 / 18.9 |

**New in this session: a depth-4 circuit** (`xs4.py`), 2x shallower than glu_chi_iota
with fewer dense (1.52x) and sparse (1.29x) parameters, i.e. a >= 2x gain on depth at no
cost on the other two metrics. `d4a_m4k3`: harness margin 1.5e-4 (1.7e-4 on 128 and
2.7e-4 on 512 random messages); edge cases (63 messages against the unpatched reference) margin 1.4e-3, eager
function equal to the reference; validate_bench bit-equal at log_w 0-2. Against the best
depth-6 circuits (lazy_middle) it is 1.5x shallower at 2.1x dense and 2.5x sparse, so it is
a new Pareto point, not a replacement.
Layers (in, hidden, out): `[1145,6074,1465] [1465,3202,1657] [1657,24059,452] [452,21649,673]`
= theta1 | chi1 + one AND unit per theta2 count (emits the 1600 theta2 counts, range 10)
| round 2 in one layer | round 3 in one layer. Dense by layer: 22.8M, 14.7M, 90.6M, 34.1M.
Steps: `xs4:d4` 4 / 181,841,962 / 421,488 (edge 2.3e-4) -> packed digests and the row
bound on counts `d4_m4k3` 166,958,830 / 427,875 (edge 1.4e-3; also 128 random messages)
-> AND units `d4a_m4k3` 162,246,830 / 420,226. At log_w=5: `d4_m4k3` 4 / 41.7M / 213K
against glu_chi_iota 8 / 61.8M / 271K (same ratios).
`d5_m3k2` and `d6_m3k2`: harness margins 1.2e-4 / 1.1e-4 (2.1e-4 / 2.1e-4 on 128 messages),
edge cases 7.2e-4 / 7.1e-4, validate_bench bit-equal. Repo test suite: 55 passed
(hash_long_test not run); the repo is unchanged in this session (all new code is in xs4.py,
decode.py and xs3.py).
The depth-5 point improves the previous depth-5 best (`direct_walsh`, 105.5M) by 13.7%;
without packed digests it is `xs3:lazy_middle_cols_walsh` 5 / 94,888,241 / 221,201
(edge margin 3.2e-4). Layers of `d5_m3k2`:
`[1145,6074,1465] [1465,1602,1921] [1921,3203,1676] [1676,5357,508] [508,13141,673]`.

Run (about 1 min each; `S` = the scratchpad):
`A=$S/xof/av/xof-structure; PYTHONPATH=$A/repo/src:$A:$S/xof $S/venv/bin/python
$S/xof/xofbench.py --log-w 6 --depth 3 --variant xs4:d4a_m4k3 --widths` (also
`xs4:d5_m3k2`, `xs4:d6_m3k2`); edge cases: `PYTHONPATH=... $S/venv/bin/python
$A/adv/adv_check.py $A/adv/ref_w6.pt xs4:d4a_m4k3`; eager equality with the reference
xof at log_w 0-3: `python -c "import check_ref; check_ref.check('xs4:d4a_m4k3', (0,1,2,3), (3,))"`.
At log_w=5, `d4a_m4k3` is 4 / 40,511,081 / 209,529 (glu_chi_iota 8 / 61,802,992 / 270,870).

### A new one-unit trick: AND of two adjacent chi bits (`xs4.AND_CHI2`)

Two adjacent chi bits of a row, chi(t1,t2,t3) and chi(t2,t3,t4), share two inputs, and
their AND is ONE gated unit (exhaustive gate search, `s2/and2chi.py`; exact on all 16
inputs):
    chi(t1,t2,t3) & chi(t2,t3,t4) = max(0, 4t1 + 2t2 - t3 + t4 - 4) * (0.75 + 0.75t1 - t2 + 0.5t3 - 0.5t4).
So their XOR, a + b - 2 AND, costs one unit on top of the two chi units, which exist
anyway. A theta count S = own + (column x-1) + (column x+1) always holds the own bit and
its left neighbour (x-1, same row), so the chi layer emits S - 2 AND(left, own): congruent
to S mod 2, range 10 instead of 11. In the one-layer round that saves one unit per single
(6 -> 5) and one per pair parity (11 -> 10) for one extra unit in the chi layer: -2.8%.
(The AND of two independent chi bits is not one unit: gates in [-3,3]^7, `s2/and_indep.py`.
In the two-layer layouts the counts are column-pair sums of chi bits from 10 different
rows, so the trick does not apply there; for a direct theta layer it moves one unit per bit
from the theta layer to the chi layer, about -1%.)

### One-layer middle round (lazy round, `xs4.lazy_round`) -- why it is exact

A theta bit is t = parity(S) (xor a known flag) where S, an integer count in [0, 11], is a
feature the layer before emits for free (a sum of its chi units). The next theta takes
parities of sums of chi bits, so chi is needed only mod 2, and
    chi(a, b, c) = t_a ^ (~t_b & t_c) = t_a + (1 - t_b) t_c  (mod 2),
    (1 - t_b) t_c = (t_c - t_b + (t_b ^ t_c)) / 2          (exact, in {0, 1}),
    t_b ^ t_c    = parity(S_b + S_c).
So o' = t_a + (t_c - t_b + parity(S_b + S_c)) / 2 is an integer in {0, 1, 2}, congruent to
chi mod 2, and it is a LINEAR combination of glu_xor units: the singles parity(S_x)
(6 units on one feature, shared by the three chi bits that read t_x) and one pair parity
parity(S_b + S_c) (11 units on two features) per chi bit: 17 units per state bit, all
flat at the lattice (knots at integers). The layer emits, for free, the counts
S3 = sum of 11 o' of the 320 theta3 bits the digest needs, plus digest 2 as o'.
Iota flips are parity flags of the counts. No layer in between: round 2 costs one layer.
Range: a count holds the own bit p and its left neighbour, two adjacent chi bits of one
row, and o'(x-1) + o'(x) = t_{x-1} + (t_x | t_{x+1}) + (1 - t_{x+1}) t_{x+2} <= 3
(t_x + (1 - t_x) t_{x+1} = t_x | t_{x+1}), so S3 <= 21, not 22: the Walsh terms need
11 + 21 + 21 + 32 = 85 units instead of 88 (the same bound gives 13 instead of 14 for
the column-parity counts below).

### One-layer last round (Walsh, as linear-fold/brainstorm-critic, here on lazy counts)

chi = (p(S_a) + p(S_a + S_c) - p(S_a + S_b) + p(S_a + S_b + S_c)) / 2 with p = parity:
with counts of range 21 that is 11 + 21 + 21 + 32 = 85 units per digest bit (19.0K units).

### Digest packing for wide layers (`xs4.pack`, `decode.py`)

In the depth-4 layout a carried feature costs ~2 x 27K + 27K + 2 x 22K parameters (it
crosses the two widest layers), so the digests of steps 1 and 2 are packed more densely:
digest 1 as 4 bits per feature (binary), digest 2 as 3 lazy values o' in {0,1,2} per
feature (base 3); features are p / 2^e (dyadic, <= 1). `decode.py` builds the exact
one-layer decoders: a continuous piecewise quadratic with knots at integers (units
relu(p - k)(a + b p)), minimum number of units by a DP over knots (a segment must fit one
quadratic): 4-bit binary 20 units, 3-digit ternary 20 units, after sharing equal units.
-7.6% dense against pairs (181.8M -> 168.0M), sparse +2.4%.

### Column-parity lazy chi (`xs3.lazy_chi_cols`, `xs4` "d5"): counts <= 13 instead of 21

In the lazy chi from E = a + D in {0,1,2} (t = [E == 1], congruent to E mod 2), the 5 chi
bits of a column sum mod 2 to parity(sum_y E_a) + sum_y Q_y (Q = [E_b != 1][E_c == 1],
exact). The parity of the 5 E values takes 5 glu_xor units on one sum, the same number
as the 5 [E_a == 1] units it replaces, but the column now contributes 1 + 5 instead of
5 + 5 to a count: theta counts of range 14 (own bit 2 + two columns 6 + 6). The own bits
of the 320 needed theta bits and the digest bits keep an [E_a == 1] unit (+480 units).
With the row bound (own + left neighbour's Q <= 2) the counts are <= 13. Worth it before a
Walsh round (85 -> 53 units per digest bit): depth 5 94.9M (91.1M with packed digests)
vs 105.5M. Before a two-layer last round it is a wash (`lazy_middle_cols`: 76,385,512 /
170,687).

### Depth 3 is out of reach, and why depth 4 costs what it costs

Depth 3 needs round 1 in one layer from raw message bits: its pair parities read ~16
message bits each, and every unit feeds ~11 theta2 counts, so layer 1 alone has ~650K
nonzeros, and round 2 then reads counts of range 21 (singles 11 units, pairs 21):
estimated ~300-340M dense and > 1M sparse, beyond 1.25x of glu_chi_iota on sparse (and
at the limit on dense). In depth 4 the one-layer round 2 (15 units per bit with the AND
units, 24K units) is 56% of dense; it is minimal for units that are parities of sums of
counts: a lazy chi bit needs t_a and the product (1 - t_b) t_c, whose Fourier support
contains {b, c}, so each chi bit needs its own pair parity (range 20: 10 units), and each
theta bit its single (range 10: 5 units). Likewise Walsh needs {a}, {a,b}, {a,c}, {a,b,c}.
A one-layer round costs about 3x a two-layer one, which is the price of the missing layer.

## Previous results (first session)

Ratios are depth / dense / sparse. "edge" = brainstorm-critic's `adv_check.py` (copied to
`adv/`): 63 messages (all 0, all 1, alternating, one-hot, one-cold, blocks, random at
density 0.05/0.5/0.95) against reference digests of the unpatched keccak.py; every
variant listed with a value passes and its eager function matches the reference.

| variant (`xs3.py`) | depth | dense | sparse | hidden | harness margin | edge margin | vs glu_chi_iota | vs baseline |
|---|---|---|---|---|---|---|---|---|
| **`lazy_middle`** (+ theta-structure's lazy chi) | **6** | **76,805,192** | **166,687** | 20,326 | 8.3e-5 | 3.4e-4 | 1.33 / 3.22 / 3.25 | 3.33 / 43.0 / 12.2 |
| `split_middle` (+ theta-structure's 3-layer theta) | 7 | 68,688,604 | 154,268 | 17,127 | 5.0e-5 | 2.9e-4 | 1.14 / 3.60 / 3.51 | 2.86 / 48.1 / 13.2 |
| `split_first_lazy` | 7 | 70,156,703 | 120,238 | 18,136 | 5.7e-5 | 1.1e-3 | 1.14 / 3.52 / 4.51 | 2.86 / 47.1 / 16.9 |
| `split_first_middle` | 8 | 62,040,115 | 107,819 | 14,937 | 3.2e-5 | 6.3e-4 | 1.00 / 3.98 / 5.02 | 2.50 / 53.3 / 18.9 |
| `direct` (this avenue alone) | 6 | 91,678,357 | 179,541 | 21,925 | 2.7e-5 | 2.4e-4 | 1.33 / 2.69 / 3.02 | 3.33 / 36.1 / 11.3 |
| `direct_bits` (digests as bits) | 6 | 99,259,637 | 180,885 | 22,373 | 2.0e-5 | | 1.33 / 2.49 / 2.99 | 3.33 / 33.3 / 11.3 |
| `direct_walsh` (+ linear-fold's one-layer last round) | **5** | 105,545,230 | 222,599 | 29,635 | 4.4e-5 | 3.2e-4 | 1.60 / 2.34 / 2.43 | 4.00 / 31.3 / 9.1 |
| `split_middle_walsh` | 6 | 82,555,477 | 197,326 | 24,837 | 5.9e-5 | 3.3e-4 | 1.33 / 2.99 / 2.75 | 3.33 / 40.0 / 10.3 |
| `split_first_middle_walsh` | 7 | 75,906,988 | 150,877 | 22,647 | 5.0e-5 | | 1.14 / 3.25 / 3.59 | 2.86 / 43.6 / 13.5 |
| `shared_middle` (new trick, noise-sensitive) | 6 | 79,921,256 | 176,343 | 18,726 | 1.6e-3 | **1.3e-2** | 1.33 / 3.09 / 3.07 | 3.33 / 41.4 / 11.5 |

The same round layouts built by the other avenues: theta-structure `lazy_middle_pack`
6 / 78,505,505 / 168,264, `three_first_lazy_pack` 7 / 71,933,215 / 122,078,
`three_first_middle_pack` 8 / 63,818,099 / 109,559; linear-fold `lin_theta_consts_pack`
6 / 91,805,897 / 179,624 and `_last1` 5 / 105,686,874 / 222,741. This builder is 1-2.5%
smaller on each (first layer 1465 outputs instead of 1601; 2-unit pair decode with one
shared constant; digests carried free in the split layer).

Layers (in, hidden, out), BOS included:
- `lazy_middle`: `[1145,6074,1465] [1465,1602,1921] [1921,3203,1713] [1713,4914,657] [657,3858,545] [545,675,673]`
- `direct`: `[1145,6074,1465] [1465,1602,1713] [1713,9714,1713] [1713,1714,545] [545,2146,545] [545,675,673]`
- `split_middle`: `[1145,6074,1465] [1465,1602,1921] [1921,3202,1713] [1713,1714,1713] [1713,1714,545] [545,2146,545] [545,675,673]`
- `split_first_middle`: `[1145,2418,1465] [1465,1466,1601] [1601,1602,1921] [1921,3202,1713] [1713,1714,1713] [1713,1714,545] [545,2146,545] [545,675,673]`
- `direct_walsh`: `[1145,6074,1465] [1465,1602,1713] [1713,9714,1713] [1713,1714,545] [545,10531,673]`

Run: `A=$S/xof/av/xof-structure; PYTHONPATH=$A/repo/src:$A:$S/xof $S/venv/bin/python
$S/xof/xofbench.py --log-w 6 --depth 3 --variant xs3:lazy_middle --widths`.
Edge cases: `... $S/venv/bin/python adv/adv_check.py adv/ref_w6.pt xs3:lazy_middle`.

## What each part does, and why it is exact

The circuit is built by `xs3.build` (helpers in `xs.py`), not by patching keccak.py:
**one traced function call per SwiGLU layer**, so every gated unit sits exactly one
level above its inputs, the compiler adds no copies, and the last layer emits exactly
the 672 outputs in order (the redundant output layer is dropped by
`Tree.has_redundant_outputs_layer`). A state bit is a *literal*: a Python int (constant)
or `(Bit, neg)`; constants and negations are folded into the weights of whatever unit
reads them (`xs.fold`: a weight w on 1-x adds w to the bias and flips the sign).

1. **First round, constants (Keccak/XOF-specific).** The suffix byte and the 448
   capacity bits are Python constants, so theta1 xors only the live message bits:
   7-9 inputs instead of 11 (4-5 units instead of 6), and it reads the message directly
   (no layer that copies the message and creates constants: glu_chi_iota's L1,
   `[1145,5506,2753]`, is 1144 x 2 message copies + 456 + 8 constants). Equal xors are
   built once: the theta bit of a zero lane is D[x][z]; columns 3 and 4 have two zero
   lanes each and the 8 suffix bits of lane (2,3) are D[2] or its negation, so theta1
   has 1464 distinct outputs, not 1600 (`[1145,6074,1465]`). Suffix ones and iota are
   literal flags: an xor of a negated input is the negated xor, and a chi unit reading
   1-x is still one unit.
2. **Last round (Keccak/XOF-specific).** Only the 224 digest bits are outputs: chi of
   row 0 (x = 0..2 and the upper half of x = 3). They read the 5 lanes of post-pi row 0,
   i.e. the theta bits of the 5 diagonal lanes (x, x) (pi maps (x, y) to (y, 2x+3y)):
   320 theta bits instead of 1600, and 224 chi units instead of 1600.
3. **Column sums are free features (Keccak-specific algebra).** A theta bit is
   parity(s), s = a + (the 10 bits of columns x-1 at z and x+1 at z+1), an integer in
   [0, 11]. The chi layer before emits s itself as one feature: a sum of its chi units,
   which costs no hidden unit (see the compiler change) and 11 wo entries. The theta
   units (glu_xor units on one input) then read one feature instead of 11: theta2's
   layer goes from ~150K to ~40K nonzeros. Before the last theta, the chi layer emits
   only the 320 counts it needs (plus the digests), not 1600 state bits (`out` 545 vs
   1825). Counts are emitted as s/16 so that RMSNorm's scale stays >= 1; readers use
   weight 16 (all dyadic, floats exact). Iota flips of chi outputs are summed into the
   count's parity flag.
4. **Digests of earlier steps (XOF-specific).** Carried two per feature, p = d0 + 2 d1
   (emitted free by the layer that makes the digest bits), one unit per layer
   (`max(0, 4f) * 1/4` on f = p/4; p >= 0 so relu is exact), decoded in the last layer
   with two units per pair: `hi = max(0, p-1)(4-p)/2`, `lo = max(0, p) - max(0, p-1)(4-p)`
   (the shared unit `max(0, p-1)` appears in both, CSE merges it). Iota flips of digest
   bits are applied there, `1 - x` with one constant unit shared by the whole layer.
   Against one-unit bit copies this is -7.6% dense (`direct_bits` 99.3M vs 91.7M).
5. **Round layouts** (compositions with the theta-structure ideas): `split` computes a
   middle round's theta in two layers, X: `E = a + D` (a copy unit of a plus the 5
   units of D = parity(column-pair count), which the 5 bits of a column share, so 2
   units per bit) and Y: `theta = [E == 1] = max(0,E)(2-E)`, one unit; E is emitted as
   E/2 (unscaled E breaks dense messages, as brainstorm-critic found for theta-structure).
   The digest of the previous step is carried free in X (pairs of the copy units).
   `lazy` (theta-structure's lazy chi): after X (with the flags folded into E, so E is
   exact), chi is computed only mod 2, `o' = [Ea == 1] + [Eb != 1][Ec == 1]` (3 units),
   because the next theta only needs parities of sums; its counts range to 22 (11 units
   per last-theta bit) and this step's digest bits are made exact one layer later,
   directly as pairs. `walsh` merges the last theta and chi into one layer (linear-fold's idea):
   `chi(a,b,c) = (a - a^b + a^c + a^b^c)/2` and each xor is the parity of a sum of
   counts, 6 + 11 + 11 + 17 = 45 units per digest bit.

Correctness: `check_ref.py` compares each variant's eager function with the reference
xof (unpatched keccak.py) on random messages, log_w 0..6, XOF depth 1..3 (all layouts).
The harness verifies the compiled network against that function. `validate_bench.py`:
harness weights bit-equal to `Compiler().get_mlp_from_tree` at log_w 0-2 (all layouts).
The repo test suite passes (57 tests; 55 without hash_long_test, rerun this session).

## Compiler changes (general, in `repo/src/reifier`)

- `neurons/core.py`: `glu(..., numeric=True)` makes a node whose value may be any number
  (a count), so later units can read a linear combination of this layer's units.
- `tensors/matrices.py`, `Matrices.layer_to_units`: **hidden-unit CSE**. Units with equal
  gate rows and proportional value rows (over the layer input) share one hidden unit;
  the value scale moves to wo. Units whose value or gate is identically 0 are dropped.
  This is what makes a count (a sum of other nodes' units) free. Unchanged circuits
  compile to the same weights (validate_bench on glu_chi_iota: bit-equal).

## Shared-column theta: exact, 4.2 units per theta bit instead of 6, but noise-sensitive

The 5 theta bits of a column, `b_y = parity(a_y + T)` (T = column-pair count, 0..10),
can share units: 3 per-bit units `u_k(a_y, T)` whose knots move with a_y, chosen so that
`sum_k u_k(1,T) - u_k(0,T) = (-1)^T`, plus 5 units on T alone (and one constant for the
layer) for `S(T) = parity(T) - sum_k u_k(0,T)`; then `b_y = S(T) + sum_k u_k(a_y, T)`
exactly (`shared_theta.py` checks it with exact fractions). 20 units per column instead
of 30 (theta2 layer 9714 -> 6515 hidden, `shared_middle` 79.9M at depth 6). theta-structure
and unit-synthesis concluded "~5 units per bit" for this sharing; 3 works because both
branches of a per-bit unit are active, with different knots. But the per-bit units have
large coefficients on a (sensitivity d b / d a up to ~80 at the lattice points, against
~2 for glu_xor): the small float errors of the chi outputs are amplified, and dense
messages reach 1.3e-2 on the edge-case check (1.8e-2 on a 95%-ones message in `stress.py`)
against 2.4e-4 for `direct`. Searches (`search/delta_*.py`): 2 per-bit units do not
exist (exact search, knots on the half grid, n = 7..10; they do for n <= 6); 3 exist only
with knots between integers (none with integer knots, where silu would average the kinks);
the least sensitive of 40 solutions has sensitivity 79. Use it only on exact inputs, or
find a better-conditioned 3-unit solution.

## What did not work

- Gate sharpening (gates x4, values /4, exact in theory): no effect on errors, which come
  from float32 rounding of glu_xor's large cancelling terms (~60 at s = 9), not from silu;
  it made the shared variant worse.
- Round 1 with its xors deferred into chi (F-variants): chi with one (a, D) input takes 2
  units (exact search), with two inputs no 2-unit solution; since every chi unit feeds 11
  packed counts, the extra chi units cost as many wo entries and hidden units as theta1
  saves (net about 0).
- Round-1 shared theta (T in 0..7-8, inputs exact): 3 per-bit units needed, <= 4% fewer
  units in the first layer.
- Packing digests 3-4 bits per feature: decoding needs ~2^m/2 units per feature; <= 2%.
- A 2-unit lazy chi on E values (o' congruent to chi mod 2, o' in {0,1,2}): not found
  (integer gates in [-2,2], `search/lazy2.py`); using E itself (congruent to [E == 1]) as a
  linear term saves the [E == 1] units but widens the next counts to 33: net ~0.
- Merging theta and chi of rounds 1-2 into one layer (Walsh): 35-45 units per state bit.
  (Superseded for round 2 by the lazy one-layer round, 17 units per bit: depth 4.)
- Depth 3 (all three rounds one layer each): layer 1 would read raw message bits in
  ~16-input pair parities and feed ~11 counts per unit (~650K nonzeros in layer 1 alone),
  and round 2 would read counts of range 21 (singles 11, pairs 21 units): estimated
  ~340M dense and > 1M sparse, over 1.25x of glu_chi_iota on both.
- Depth 4 with the one-layer round first (round 1 from raw bits, then theta2 on range-21
  counts, chi2, Walsh): ~189M and much higher sparse than `d4` (pair parities of raw bits).
- (a, P) instead of S = a + P as the one-layer round's input (to use the shared-column
  singles, 4.2 instead of 6 units): +320 input features cost 2 x 27K parameters each in the
  27K-unit layer, more than the 3.2K saved units.
- One-unit mod-2 representative of the lazy q-term [E_b != 1][E_c == 1] with values in a
  window of width 2: none (exact search, gates on the half-integer grid in [-6, 6],
  `s2/q1x.py`). Width 3 exists (e.g. [E_c == 1](1 + E_b)) but widens the counts so much
  that the next theta or Walsh layer pays more than the 1600 units saved.
- XOR of two adjacent chi bits of a row (to cut a count's range by one): no 1-unit form
  (gates in [-3,3]^5, exact), no 2-unit form in the first 3M pairs (`s2/xor2chi.py`).
- 2-D units for the AND of two parities given the singles for free (`s2/and2d.py`): the
  gradient search finds 2 units on [0,3]^2 but fails even where 1-D solutions exist
  (n = 5, K = 5), so it is inconclusive; analytically, knots between integers with units
  opening both ways save 1 of the 11 pair units, and give up flatness at the lattice.
- Column-parity lazy chi before a two-layer last round, and digest packing in the depth-6
  layout: about -0.5M each (the layers they cross are narrow there).
- One-unit parity with non-integer knots (brainstorm-critic's MIN_PARITY) was not used:
  it has slope at lattice points too (their edge-case margin 0.0265 in this design).

## Composes with

- Cheaper theta/chi units (unit-synthesis, brainstorm-critic MIN_PARITY where inputs are
  clean): the layouts here only fix what is computed where.
- theta-structure's layouts (split, lazy): measured above on this builder (`lazy_middle`,
  `split_first_lazy`, `split_middle`, `split_first_middle`).
- linear-fold's fold pass: it produces the same first/last-round structure from the
  compiler side; this builder produces it from the circuit side.
- The depth-4 layout with any cheaper parity of a count: every unit of layers 3 and 4 is a
  glu_xor unit on counts (singles on range 10, pairs on 20, Walsh terms on 21/42/63). A
  5-for-6-style min-parity (brainstorm-critic) would cut roughly 10% there, at the price of
  slope at the lattice points; layer 1 reads exact message bits, so min-parity with sharpened
  gates is safe there (about -500 units, -1.8M dense, in every layout).
- `decode.py` is general: any bits carried across layers can be packed m per feature and
  decoded exactly in one layer with the DP-minimal piecewise-quadratic units (a compiler
  pass could do this for leveling's pair packing).
- The one-layer lazy round is the general form "a product of two parities of counts in one
  layer" = (t_c - t_b + parity(S_b + S_c)) / 2, for any xor-and circuit whose next stage
  needs only parities.

## Files

- `xs4.py` (this session): the depth-4 (`d4*`, `d4a*` with the AND units), depth-5
  (`d5*`) and depth-6 (`d6*`) layouts: one-layer lazy round, AND_CHI2, Walsh last round on
  lazy counts, column-parity lazy chi, packed digests;
  `decode.py`: minimum exact one-layer decoders of packed integers (DP over knots);
  `s2/`: searches of this session (`q1x.py`, `xor2chi.py`, `and2chi.py`, `and_indep.py`,
  `and2d.py`), `collect.py`
  (rebuilds results.jsonl), `mkpatch.sh`; `runs2/`: harness output of this session.
- `xs3.py`: the builder and all variants; `xs.py`: literals, folding, maps, first version
  (`xs:variant`, 6 / 101.4M / 297.7K with bit copies and no packing); `xs2.py`: packed
  counts only (superseded); `shared_theta.py`: the shared-column theta units.
- `check_ref.py` (eager vs reference), `stress.py`, `layer_err.py`, `layer_stats.py`,
  `adv/` (edge-case check), `search/` (unit searches), `final/` and `runs/` (harness output).
- `patch.diff`: `git -C repo diff` (includes the base's uncommitted changes) plus the new
  files; `patch_vs_base.diff`: this avenue's compiler change only.

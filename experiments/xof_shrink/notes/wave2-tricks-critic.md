# tricks-critic (wave 2): packing tricks, and how far the untied SwiGLU MLP can go

Avenue: new SwiGLU tricks of the "XOR in one unit" kind that wave 1 did not use, plus lower
bounds. All numbers: log_w=6, 3 XOF steps.
- Checks: `xofbench.py` harness plus `audit/adv_check.py` on BOTH reference sets
  (`ref777_w6.pt`, 69 msgs; `combined-1/adv/ref_w6.pt`, 63 msgs). Each result has
  `ok: true`, 0 wrong bits and an empty eager mismatch.
- Every line is in `results.jsonl`: 26 variants x 3 lines.

## Summary

Two new exact tricks. Both put the value path of units that were only copies to work:
- **cp, column packing.** The chi layer before an X layer emits 3 features per column
  instead of 6. The X layer decodes them with the units it already spends on copies:
  -8.6% to -14% dense at depths 5-8.
- **sp, packed round-1 split.** The first X layer passes the message bits as pairs, one
  unit per pair instead of two copies. Y1 computes theta = bit ^ D straight from the
  packed pair with 3 units per 2 bits: another -2.5M at depths 7 and 8, with sparse down
  too.

The Y1 dedupe (dd) is a small cleanup.

Depth 8 goes from 61.08M to **49.71M dense (-18.6%)**, and depth 7 from 63.65M to
**52.83M (-17.0%)**. Depth 8 is now below wave 1's estimated floor of ~50M. The lowest
sparse also improves: d8 106,979 and d7 119,398.

The other tricks in the brief were examined and give nothing, each with a short proof or
a measured reason:
- products for decoding;
- silu identities;
- RMSNorm;
- one unit feeding many outputs;
- the 224-of-1600 structure.

Depths 7 and 8 are now within ~1-3% of the floor of their layout family (every stage at
its rank, lemma or exhaustive-search minimum). A further 2x needs a primitive the
searches rule out, or an architecture change.

### Verified Pareto points (new, non-dominated at their depth among wave 1 and this avenue)

| depth | variant | dense | sparse | vs wave-1 frontier (dense / sparse) | audit margin ref777 / ref63 |
|---|---|---|---|---|---|
| 5 | `xc:d5_m3k2_mp_cp` | **81,555,805** | 232,333 | -8.6% / +1.1% | 8.5e-4 / 1.7e-3 |
| 6 | `xs3:lazy4c_middle_m3_mp_cp` | **61,679,613** | 180,087 | -11.1% / +1.4% | 8.1e-4 / 9.4e-4 |
| 6 | `xs3:lazy4c_middle_m3_cp` | 63,568,378 | 175,101 | -8.4% / -1.4% | 5.7e-4 / 6.6e-4 |
| 6 | `xs3:lazy4c_middle_cp` | 64,276,040 | 173,695 | -7.3% / -2.2% | 3.4e-4 / 4.1e-4 |
| 6 | `xc:d6_m3k2_cp` | 67,571,322 | 172,013 | -2.6% / -3.2% | 8.0e-4 / 6.6e-4 |
| 6 | `xs3:lazy_middle_cp` | 69,116,552 | 169,151 | -0.4% / -4.8% | 4.3e-4 / 4.7e-4 |
| 7 | `xs3:split_first_lazy4c_m3_mp_cp_sp` | **52,827,268** | 129,094 | -17.0% / +1.3% | 2.5e-3 / 1.9e-3 |
| 7 | `xs3:split_first_lazy4c_m3_cp_sp` | 53,656,273 | 127,812 | -15.7% / +0.3% | 1.8e-3 / 1.8e-3 |
| 7 | `xs3:split_first_lazy4c_cp_sp` | 54,363,935 | 126,406 | -14.6% / -0.8% | 9.8e-4 / 6.8e-4 |
| 7 | `xs3:split_first_lazy_cp_sp` | 59,204,447 | 121,862 | -7.0% / -4.4% | 1.6e-3 / 1.1e-3 |
| 7 | `xs3:split_first_lazy_sp` | 66,893,087 | **119,398** | +5.1% / -6.3% | 5.7e-4 / 2.3e-4 |
| 8 | `xs3:split_first_middle_m3m2_mp_cp_sp` | **49,713,174** | 111,983 | -18.6% / +2.6% | 2.5e-3 / 1.9e-3 |
| 8 | `xs3:split_first_middle_mp_cp_sp` | 50,260,774 | 110,725 | -17.7% / +1.5% | 2.1e-3 / 1.8e-3 |
| 8 | `xs3:split_first_middle_m3m2_cp_sp` | 50,542,179 | 110,701 | -17.3% / +1.5% | 1.8e-3 / 1.8e-3 |
| 8 | `xs3:split_first_middle_cp_sp` | 51,089,779 | 109,443 | -16.4% / +0.3% | 9.8e-4 / 1.3e-3 |
| 8 | `xs3:split_first_middle_sp` | 58,776,499 | **106,979** | -3.8% / -1.9% | 2.4e-4 / 2.2e-4 |

Wave-1 frontier: d5 89,244,445 / 229,869; d6 69,368,253 / 177,623; d7 63,651,004 / 127,470;
d8 61,082,590 / 109,101. Wave 1's lowest sparse values were d6 166,687 (lazy_middle,
76.8M, still Pareto), d7 120,238 and d8 107,819.

Depth 4 is unchanged (`xc:d4a_m4k3_mp`, 160,358,065 / 425,212): its layout has no X layer.

Against the baseline (depth 20, 3,305,445,348 / 2,034,788):
- d8 is 66.5x dense and 18.2-19.0x sparse;
- d7 is 62.6x / 15.8-17.0x;
- d6 is 53.6x / 11.3-12.0x.

Ten intermediate variants were also verified: cp without sp, dd only, and the m3 digest-2
packing. They are all dominated now; their lines are in `results.jsonl`, e.g.
`split_first_middle_m3_mp_cp_sp` 49,884,891 / 113,537.

`m3m2` packs digest 1 three bits per feature (it is carried through 5 layers) and digest 2
in pairs (it is carried through 1 layer, so the cheaper pair decoder wins). Against m3 for
both digests (49,884,891 / 113,537) that is -0.17M dense and -1.6K sparse. It is the new
`m2` argument of `xs3.build`.

Also checked with the harness at log_w 4 and 5 (64 random messages), all `ok`, margins
2.6e-4 to 2.3e-3 (`runs/lw/`):
- the cp headliners;
- `split_first_middle_m3_mp_cp_sp`, `split_first_lazy4c_m3_mp_cp_sp` and
  `split_first_middle_cp_sp`.

More random messages at log_w 6, all `ok`:

| variant | messages | margin |
|---|---|---|
| `split_first_middle_m3m2_mp_cp_sp` | 400 | 2.2e-3 |
| `split_first_middle_m3_mp_cp_sp` | 128 | 2.5e-3 |
| `split_first_middle_m3_mp_cp_sp` | 400 | 3.4e-3 |
| `split_first_lazy4c_m3_mp_cp_sp` | 400 | 2.2e-3 |

`split_first_middle_m3m2_mp_cp_sp` is also `ok` at log_w 4 and 5 (margin 1.6e-3).

`validate_bench.py` (`runs/validate_bench.txt`) was run on
`split_first_middle_m3_mp_cp_sp`, `lazy4c_middle_m3_mp_cp` and `d5_m3k2_mp_cp`. At log_w
0-2 the harness weights are bit-equal to `Compiler().get_mlp_from_tree`, and depth, dense
and sparse are equal.

## Trick 1: column packing (cp)

**What the X layer did.** In the split/lazy layouts, round 2's theta starts with the X
layer: E = a + D.
- a is the chi1 bit;
- D is the parity of the column-pair count P, shared by the 5 bits of a column;
- the layer has one copy unit per state bit, plus 5 glu_xor units per column pair.

With no skip connection, the chi1 layer had to emit 1600 bits plus 320 counts P
(`[..,1602,1921] -> [1921,3202,..]`). At 2 x 3202 per input feature, X was the widest
layer (17.7M).

**Pair decode at copy cost.** A pair p = a + 2b in {0,1,2,3} is decoded exactly by two
gated units, the same count as two copy units:
- b = relu(p - 1) (4 - p) / 2, with values 0, 0, 1, 1;
- a = relu(p) - 2 b, where relu(p) = p because p >= 0.

**Fifth bit from the column count.** The D units need the column sums anyway, so the chi1
layer emits 3 free linear features per column (x, z):
- c = sum_y a_y;
- p1 = a_0 + 2 a_1;
- p2 = a_2 + 2 a_3.

The fifth bit follows:
- a_4 = (c - p1 - p2) + b_1 + b_2, with one BOS-gated unit relu(1) (c - p1 - p2).
- That makes 5 units per column, exactly the 5 copies it replaces.
- D reads P = c(x-1, z) + c(x+1, z+1).

**Effect.** chi1 emits 961 features instead of 1921, and X reads 961; neither layer gains
a unit.
- X: 17.7M -> 11.5M. chi1: 7.8M -> 6.2M (`[1465,1602,961] [961,3203,1676]`).
- Digest 1 (the y=0 bits a_0) is packed from the same units, with no new unit.
- It works both with the flags folded (lazy layouts, E = (a ^ af) + (D ^ pf)) and with
  flags kept (split layouts).
- Implemented in xs3 (`chi_to_split_cp`, `cp_decoders`, `cp_d_specs`,
  `theta_split_x_cp`) and xs4 (`x_exact_cp`, for d5).

## Trick 2: packed round-1 split (sp)

In the split-first layouts, X1 copies every message bit into E1 = a + D1 (1144 copy units
on the 1145-wide input), and Y1 computes theta = [E1 == 1] (1 unit per bit). With sp:
- **X1** emits per column (x, z):
  - the live message bits in pairs, p = a_0 + 2 a_1 (one unit relu(a_0 + 2 a_1));
  - a lone bit where the column has 3 live bits;
  - D1, the parity of the column pair (unchanged).

  That is 640 pack units instead of 1144 copies, and 961 outputs instead of 1465.
- **Y1** decodes theta = bit ^ D from (p, D) with 3 units per pair:
  - u1 = relu(p) (1 - 2D) (= +-p);
  - u2 = relu(p - 1 - 4D) (4 - p)/2 (= hi(p) on D = 0, 0 on D = 1);
  - u3 = relu(4D - 2 - p) (1 + p)/2 (= 0 on D = 0, hi(3 - p) = 1 - hi(p) on D = 1).

  Then theta_1 = u2 + u3 and theta_0 = u1 - 2 u2 - 2 u3 + 3 D. The D term is the unit
  relu(D) that the column's capacity-lane theta (= D) needs anyway. Every column has a
  capacity lane, so the D term is free. Lone bits use one XOR unit on a + D.
- **Checked on all 8 points:**
  - D = 0: theta_1 = hi(p), theta_0 = p - 2 hi(p) = lo(p).
  - D = 1: theta_1 = hi(3 - p) = 1 - hi(p), theta_0 = -p - 2 hi(3 - p) + 3 = lo(3 - p) = 1 - lo(p).
- **Cost:**
  - X1 `[1145,2163,1465] -> [1145,1659,961]`: 8.12M -> 5.39M.
  - Y1 `[1465,1466,1465] -> [961,1970,1465]`: 6.44M -> 6.67M.
  - Net -2.5M, and sparse -0.6K. Code: `theta1_split_packed` in xs3, layout kind "splitp".
- **Why 3 per pair:** 2 units per pair are impossible. With 2 units plus the free
  {1, D}, each unit must lie in span{theta_0, theta_1, 1, D} (4-dim on the 8 points).
  An exhaustive check (`search/pair2.py`) covered 120,572 gates
  g = al p + be D + ga (al, be on the half grid in [-8,8] x [-12,12], ga on the quarter
  grid in [-12,12]). Only one direction (alpha : beta) = (1 : 2) of single-unit members
  exists, the u1 = +-p type; two units would need two independent directions.
  (Consistently, the lemma below makes theta_0 alone need 2 units on the D = 0 slice.)
- **Round 2 does not pay** (cost model, not built): the same idea at X2/Y2 comes to
  about +2.1M. X2's "copies" are already decode units, so X2 saves only outputs. Y2 would
  need 8 units per column instead of 5 (3 + 3 for the pairs, 1 for a_4 ^ D, 1 for the D
  term, since round 2 has no capacity lane).

**Why both tricks are exact.**
- Every decoded value is an integer identity on the lattice, with integer knots.
- Gates are never 0 where a unit must be on:
  - relu(p) at p = 0 is silu(0) = 0;
  - the other gates are integers, and they are <= 0 exactly where the unit must be 0.
- Features are dyadic (c/8, p/4), so eager values are exact and RMSNorm's scale stays
  <= 1.
- Eager circuits equal the reference xof on the audited edge messages (empty
  `eager_mismatch`).

**Noise.** Decode units are not flat at the lattice points the way glu_xor is (slopes
1/2 to 3/2). So the cp margins are 2-3x the non-cp ones (<= 1.7e-3). sp with min-parity
D1 reaches 4.0e-3 on the edge set: the D errors of min-parity pass through the +-4D
gates. That is the same range as wave 1's accepted lazy4c variants (3.6e-3 to 5.1e-3),
5x below 0.02. Without min-parity, sp stays <= 1.3e-3.

**dd (Y1 dedupe).** `theta_split_y` made one output per position. The 136 capacity-lane
positions whose E1 node is shared had duplicate outputs. Now there is one output per
distinct node: -0.63M in the split-first layouts without sp.

## Where the dense goes now

| layer | d8 `split_first_middle_m3m2_mp_cp_sp` | d7 `split_first_lazy4c_m3_mp_cp_sp` | d6 `lazy4c_middle_m3_mp_cp` |
|---|---|---|---|
| 1 | X1 [1145,1659,961] 5.39M | X1 5.39M | theta1 [1145,5571,1465] 20.92M |
| 2 | Y1 [961,1970,1465] 6.67M | Y1 6.67M | chi1 [1465,1602,961] 6.23M |
| 3 | chi1 [1465,1602,961] 6.23M | chi1 6.23M | X2 [961,3203,1676] 11.53M |
| 4 | X2 [961,3202,1676] 11.52M | X2 11.53M | lazy chi2 [1676,2861,620] 11.37M |
| 5 | Y2 [1676,1677,1676] 8.43M | lazy chi2 11.37M | theta3 [620,5645,508] 9.87M |
| 6 | chi2 [1676,1677,508] 6.47M | theta3 9.87M | chi3 [508,1045,673] 1.77M |
| 7 | theta3 [508,2109,508] 3.21M | chi3 1.77M | |
| 8 | chi3 [508,1045,673] 1.77M | | |

Sparse:
- d8: X2 24K, X1 20K, chi2 17K, chi1 16K, Y1 12K.
- d6: theta1 alone is 83K of 180K, because min-parity units read all 7-9 message bits in
  both gate and value.

## The avenue's candidate tricks, one by one

1. **Products for multi-bit decoding.** A pair (m = 2) decodes at 1 unit per bit, and
   that is what cp and sp use. m = 3 needs 8 units for 3 bits, and m = 4 needs 23 for 4
   (DP minima, `decode.py`). So only pairs are free, and only in front of copy units. For
   digests carried through several layers, m3 is already the optimum.
2. **One hidden unit feeding many outputs through wo.**
   - Already used: counts are free sums, D is shared by 5 E's, and CSE merges equal
     units.
   - cp and sp add two more cases: one b unit feeds 3 outputs; u2 and u3 feed both theta
     outputs.
   - What cannot share: each chi bit needs its own unit. The 5 outputs of a chi row are
     linearly independent. And a unit that is always active is a product of two affine
     forms (degree <= 2 on the cube), while chi has an abc term. So one unit cannot give
     chi_0 + chi_1 of two rows: fixing row 1 so that chi_1 = 1 forces an always-active
     unit to equal chi_0 + 1.
3. **silu(z) - silu(-z) = z.**
   - It gives an exact bilinear g * v for g of either sign in 2 units; a gate that is
     >= 0 already gives it in 1 (sp's u1 is such a product).
   - In the +-1 domain, parity becomes a tree of products: 2 units per multiplication
     and log n layers, worse than glu_xor.
   - Using silu's curvature near 0 would give signals ~k z^2, far below float32
     resolution at this size. Rejected.
4. **RMSNorm effects.** With relu, a layer is positively homogeneous of degree 2. The
   norm multiplies every unit and BOS by the same 1/r^2, and outputs are read relative to
   BOS, so the norm is inert. It only sets the effective steepness k/r, and that is why
   features are kept <= 1. The per-feature weight is a column scaling of wg and wv.
   Nothing to exploit.
5. **Value-path tricks that halve units.**
   - cp and sp are exactly this: copies whose value path decodes.
   - Also considered: absorbing the lazy chi's pass (linear) terms into the product unit.
     Over GF(2), a product of two affine forms cannot be congruent to Ea + Ec + Eb Ec:
     the Ea Eb and Ea Ec terms vanish only if Ea's coefficient is even.
   - A lazy chi without a pass term would also lose lazy4c's column reduction. Counts
     would go from 32 to 44: +3.4M against -2.2M saved.
6. **The digest only needs 224 of 1600 bits.** Wave 1 already computes only 320 theta3
   and 224 chi3 bits.
   - The next step would be the column parities of chi2 through the one-unit mod-2 inner
     product (two AND terms in one unit, unit-synthesis): 3 units per column instead
     of 5.
   - It fails: the own bits and digest 2 still need exact chi2, and lazy column values
     have width 13 instead of 5, so theta3 needs 14 units per bit instead of 6.
7. **Packing other interfaces** (the generalization of cp and sp).
   - **Lemma:** a single unit relu(g) v (g, v affine) cannot compute a function that
     depends on bit t but not on bit s from a feature p = t + 2s.
   - **Proof:** fix the other features. Along P = t + 2s in {0,1,2,3} the unit is
     phi(P) = relu(A + aP) (B + dP), and it must take the pattern (u, w, u, w) with
     u != w.
     - All 4 points active: q(0) = q(2) and q(1) = q(3) make the quadratic symmetric
       about both 1 and 2, hence constant.
     - Otherwise the inactive points are a prefix or a suffix.
     - Inactive {0}: u = 0, so B + 2d = 0 (active at 2), and q(1) = q(3) gives A = -2a.
       Active at 1 needs a < 0, which makes P = 0 active: contradiction.
     - Inactive {3}: w = 0, so B + d = 0, and q(0) = q(2) gives A = -a. Active at 2 needs
       a > 0, and inactive at 3 needs 2a <= 0: contradiction.
     - Longer prefixes or suffixes set both u and w to 0.

     With base-3 packing of E values the pattern has period 3, and the same argument
     applies.
   - **Consequence:**
     - Readers with one unit per output need unpacked inputs: Y units ([E == 1]), CHI
       units, and the lazy product units (which need Eb and Ec linearly). This covers
       the inputs of Y1, chi1, Y2, chi2 and lazy chi2.
     - Parity readers (theta layers on counts) need one feature per GF(2)-independent
       parity, since a linear form of integer features mod 2 has rank <= the number of
       features.
     - The only interfaces left for packing were the ones cp and sp use: into copy
       units, and into Y1 through the 3-unit decode.
8. **Measured or computed rejects:**
   - D from column parities instead of pair parities: X2 -2.3M, but Y2 +8.2M (E in
     [0,3] needs 2 units).
   - A lazy chi2 on 4-valued E: width >= ~9.
   - Packing lazy digest-2 values (base 5 or base 3): decoders cost more than the
     features save.
   - A 4-unit exact chi for digest 2: +0.5M, and no integer form is known.
   - Splitting X2 into two layers: +3.3M.
   - A depth-9 X3/Y3 last round: X3 needs as many units as theta3 direct.
   - sp at round 2 (X2 passes pairs, Y2 decodes): about +2.1M (cost model).
   - A 4-bit column packed as one number: 23 decode units.
   - Column parities instead of D1 in X1: Y1 +2.2M.
   - sp at depth 6: its theta1 is a single layer, so there is no X1 to pack.
   - Min-parity on count features (theta3; the d4 pair parities): -1.4M at d6/7 and -9M
     at d4. Knots between integers amplify the noise of count inputs, and wave 1 measured
     an edge margin of 0.0265. Not used.

## Lower bounds: how far can each metric go?

**(a) Unconditional**, for any exact network of this architecture (silu as relu, outputs
exactly 0/1 relative to BOS).
- The last layer's outputs are linear combinations of its hidden units.
- The 673 output functions (BOS plus 672 digest bits) are linearly independent: the real
  rank over 1500 random messages is 673 (`scripts/rank_check.py`, sigma_min 1.13).
- So **h_L >= 673**: dense >= 673 (2 in_L + 673), sparse >= 3 x 673 + 1144.
- Depth 8 without m3 meets this layer bound (675 units).

Nothing else is provable without a model of the features. In exact arithmetic a single
real feature can carry any number of bits, so the other widths have no floor.
- Unconditional floors are ~0.5-1M dense and a few K sparse, 50-100x below every
  construction. They do not constrain.
- With the 0.02 tolerance, even the rank bound needs the error matrix to have spectral
  norm < 1.13. That holds for these circuits (errors ~1e-3), but not in general.

**(b) Lattice model** (features are integers up to scale; gates and knots are integers).
This covers every integer-knot construction of both waves.
- Inputs pinned by the lemma and GF(2) rank:
  - chi1: >= 1464 (the distinct theta1 bits);
  - Y2, chi2 and lazy chi2: >= 1600;
  - theta3: >= 320 + digest features;
  - Y1: sp reads 961. That works because the decode spends 1.5 units per bit; with
    1 unit per bit the lemma pins 1464.
- Units pinned by rank:
  - every chi layer: >= 1600 (see 2 above);
  - Y2: >= 1600;
  - X2: >= 1600 for the a parts, plus 5 per column pair for D. Parity of a range-10
    count needs 5 units; 4 is impossible for knots with denominator <= 6 (exhaustive
    1-D, wave 1).
- Not pinned:
  - the X2 input (cp: 3 features per column, conjectured minimal, since m >= 3 decoders
    exceed the copy budget);
  - Y1's 3 units per pair: 2 per pair is impossible for gates on the searched grid
    (`search/pair2.py`), so 3 is minimal there.

**(c) Family floor per depth** (the stage decomposition fixed, every layer at its minimum
in (b), digests at the m3 optimum).

| depth | best measured | floor of its family | slack that remains, and why it is not taken |
|---|---|---|---|
| 8 | 49.71M | ~49.2M | theta3 min-parity (n=11: 5 units, not 6): -0.5M, noise risk on counts |
| 7 | 52.83M | ~51.1M | theta3 min-parity on range-32 counts: -1.7M (noise) |
| 6 | 61.68M | ~60.0M | the same -1.7M. theta1 (20.9M) is at its one-layer minimum: 1464 parities of 7-9 raw bits need 3-4 units each (1-D exhaustive), and shared-column units give no gain for T = 6-8 (wave 1) |
| 5 | 81.56M | ~79M | min-parity in the Walsh terms (noise) |
| 4 | 160.36M | ~151M | min-parity on the pair parities of the one-layer round (-9M, noise). 15 units per state bit is minimal for parities of sums of counts (Fourier support {b,c} per chi bit) |

Sparse, in the same family: d8 107K and d7 119K. The largest sparse items are forced:
- theta1/X1 parity units read 6-9 raw bits each;
- CHI units need 7-8 nonzeros plus wo;
- D units fan out to 5 E's.

I estimate the sparse floor at ~100K (d8) and ~112K (d7).

**(d) What a further 2x would have to break.** Each item would remove a Y layer or halve
a stage:
- exact chi from one-feature E codes in <= 2 units: impossible (exhaustive, wave 1);
  3 units impossible with |w| <= 3;
- parity of range 10 in < 5 units: impossible;
- < 1 unit per chi bit: impossible (rank and degree);
- packed inputs for one-unit-per-output readers: impossible (the lemma).

**Headroom in this architecture:**
- dense: ~1% below 49.7M at depth 8 in this family, and ~49-50M overall unless a new
  primitive appears; ~51M at depth 7, ~60M at depth 6, ~79M at depth 5, ~150M at
  depth 4;
- sparse: within ~5-10% of 107K (d8) and 119K (d7).

The 100x-vs-baseline target (~33M) needs the architecture change from wave 1 (weights
tied across XOF steps, a residual stream).

## Composes with

- cp changes only the interface between a chi layer and an X layer, so it applies to any
  layout with an X layer.
- sp changes only the first X/Y pair. It needs a column D shared by the pair; in round 1
  every column also has a capacity lane, which provides the D term.
- Min-parity (in D1), m3 digest packing, lazy4c and the Walsh last round compose
  unchanged; the table uses them.
- Other wave-2 avenues that change the theta3/chi3 tail or the lazy layer should stack
  additively.

## Files (in this directory)

- `patch.diff`: `git -C repo diff`. It touches only `experiments/xof_shrink/` files:
  - `xs3.py`: cp, sp, dd, m2 and the new variants;
  - `xs4.py`: cp for d5/d6;
  - `xc.py`: `d5_m3k2_mp_cp`, `d6_m3k2_cp`, `d6_m3k2_mp_cp`.

  The reifier `src/` is unchanged against the wave-2 base.
- `results.jsonl`: harness plus both audits for each variant (kind = harness /
  audit_ref777 / audit_ref63).
- `runs/`: raw harness and audit JSON, per-layer breakdowns (`layers_*.txt`), log_w 4/5
  checks (`lw/`), and the rank check.
- `scripts/`:
  - `audit.sh` (harness plus both audits, appends to results.jsonl);
  - `layers.py` (per-layer dense/sparse);
  - `rank_check.py`;
  - `edit_splitp.py` (the sp edit, as applied).
- Run: `PYTHONPATH=repo/src:repo/experiments/xof_shrink $S/venv/bin/python
  repo/experiments/xof_shrink/xofbench.py --log-w 6 --depth 3 --variant
  xs3:split_first_middle_m3m2_mp_cp_sp --widths`

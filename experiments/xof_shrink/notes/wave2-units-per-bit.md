# units-per-bit (wave 2): fewer gated units per output bit in the frontier blocks

Avenue: cut the units per output bit of the building blocks of the wave-1 frontier
(round-1 theta on raw bits, the X layer, the lazy chi, the last theta), with exact
synthesis, and re-test min-parity on count features now that q = 8. Architecture unchanged
(plain `MLP_SwiGLU`, no residuals, untied, attention = identity), default steepness
c = 4, q = 8.

Every number below comes from `xofbench.py --log-w 6 --depth 3` plus `audit/adv_check.py`
on BOTH reference sets (`xof/final/ref777_w6.pt`, 69 messages; `xof/av/combined-1/adv/
ref_w6.pt`, 63 messages): `ok: true`, 0 wrong bits, empty eager mismatch on both.
`run.sh module:fn` runs all three and appends a line to `results.jsonl`; `verified: true`
only if all pass. `validate_bench.py` (harness weights bit-equal to
`Compiler().get_mlp_from_tree` at log_w 0-2) passes for the new layouts
(`runs/validate_small.txt`), and the headline variants are ok with 0 wrong bits at log_w 3,
4 and 5 on 64 random messages (`runs/other_logw*.txt`).

## Result in short

<!-- table -->
| depth | variant | dense | sparse | vs frontier dense / sparse | audit margins (69 / 63 msgs) | status |
|---|---|---|---|---|---|---|
| 4 | `xc:d4a_m4k3_mp_mpc_th` | 158,265,974 | 430,458 | -1.30% / +1.23% | 2.6e-03 / 1.8e-03 | trade (Pareto) |
| 4 | `xc:d4a_m4k3_mp_mpc` | 159,651,569 | 425,212 | -0.44% / +0.00% | 5.1e-04 / 8.1e-04 | **dominates frontier** |
| 5 | `xc:d5_m3k2_mp_nop_mpc_th` | 84,539,298 | 248,875 | -5.27% / +8.27% | 2.3e-03 / 1.8e-03 | trade (Pareto) |
| 5 | `xc:d5_m3k2_mp_nop_mpc` | 85,924,893 | 243,629 | -3.72% / +5.99% | 3.4e-04 / 8.3e-04 | trade (Pareto) |
| 5 | `xc:d5_m3k2_mp_mpc_th` | 87,102,178 | 235,115 | -2.40% / +2.28% | 2.3e-03 / 1.8e-03 | trade (Pareto) |
| 5 | `xc:d5_m3k2_mp_mpc` | 88,487,773 | 229,869 | -0.85% / +0.00% | 3.2e-04 / 8.5e-04 | **dominates frontier** |
| 6 | `xs3:lazy4c_middle_m3_mp_nop_z11_lz5_th` | 63,532,738 | 198,533 | -8.41% / +11.77% | 1.7e-03 / 1.7e-03 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_nop_z11_th` | 64,148,738 | 194,389 | -7.52% / +9.44% | 7.1e-04 / 7.0e-04 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_nop_z11_lz5` | 64,918,333 | 193,287 | -6.41% / +8.82% | 1.0e-03 / 1.0e-03 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_nop_c5_lz5` | 65,406,013 | 192,007 | -5.71% / +8.10% | 4.9e-04 / 4.5e-04 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_nop_z11` | 65,534,333 | 189,143 | -5.53% / +6.49% | 1.1e-03 / 1.1e-03 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_nop_c5` | 66,093,693 | 187,863 | -4.72% / +5.77% | 4.6e-04 / 5.8e-04 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_z11_th` | 66,711,618 | 180,629 | -3.83% / +1.69% | 7.1e-04 / 7.0e-04 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_z11_lz5` | 67,481,213 | 179,527 | -2.72% / +1.07% | 1.0e-03 / 1.0e-03 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_c5_lz5` | 67,968,893 | 178,247 | -2.02% / +0.35% | 4.2e-04 / 5.0e-04 | trade (Pareto) |
| 6 | `xs3:lazy4c_middle_m3_mp_z11` | 68,097,213 | 175,383 | -1.83% / -1.26% | 1.0e-03 / 1.0e-03 | **dominates frontier** |
| 6 | `xs3:lazy4c_middle_m3_mp_c5` | 68,656,573 | 174,103 | -1.03% / -1.98% | 5.3e-04 / 5.3e-04 | **dominates frontier** |
| 6 | `xs3:lazy4c_middle_mp_z11` | 68,769,355 | 173,977 | -0.86% / -2.05% | 1.0e-03 / 1.0e-03 | **dominates frontier** |
| 6 | `xs3:lazy4c_middle_mp_c5` | 69,364,235 | 172,697 | -0.01% / -2.77% | 4.3e-04 / 5.8e-04 | **dominates frontier** |
| 7 | `xs3:lazy4_middle_m3_mp_nop_th_rp` | 58,892,322 | 181,704 | -7.48% / +42.55% | 8.3e-03 / 6.6e-03 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_nop_z11_lz5` | 59,201,084 | 143,134 | -6.99% / +12.29% | 3.3e-03 / 3.3e-03 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_nop_c5_lz5` | 59,688,764 | 141,854 | -6.22% / +11.28% | 9.6e-04 / 9.4e-04 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_nop_z11` | 59,817,084 | 138,990 | -6.02% / +9.04% | 3.3e-03 / 3.3e-03 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_nop_c5` | 60,376,444 | 137,710 | -5.14% / +8.03% | 5.2e-04 / 4.4e-04 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_z11_lz5` | 61,763,964 | 129,374 | -2.96% / +1.49% | 3.3e-03 / 3.3e-03 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_c5_lz5` | 62,251,644 | 128,094 | -2.20% / +0.49% | 9.2e-04 / 9.1e-04 | trade (Pareto) |
| 7 | `xs3:split_first_lazy4c_m3_mp_z11` | 62,379,964 | 125,230 | -2.00% / -1.76% | 3.3e-03 / 3.3e-03 | **dominates frontier** |
| 7 | `xs3:split_first_lazy4c_m3_mp_c5` | 62,939,324 | 123,950 | -1.12% / -2.76% | 5.4e-04 / 6.0e-04 | **dominates frontier** |
| 7 | `xs3:split_first_lazy4c_mp_z11` | 63,052,106 | 123,824 | -0.94% / -2.86% | 3.3e-03 / 3.3e-03 | **dominates frontier** |
| 7 | `xs3:split_first_lazy4c_mp_c5` | 63,646,986 | 122,544 | -0.01% / -3.86% | 4.4e-04 / 3.7e-04 | **dominates frontier** |
| 8 | `xs3:split_first_lazy4_m3_mp_nop_rp` | 54,560,668 | 126,305 | -10.68% / +15.77% | 5.6e-03 / 3.9e-03 | trade (Pareto) |
| 8 | `xs3:split_first_lazy4_mp_nop_rp` | 55,030,161 | 125,047 | -9.91% / +14.62% | 5.6e-03 / 3.9e-03 | trade (Pareto) |
| 8 | `xs3:split_first_lazy4_m3_mp_rp` | 57,123,548 | 112,545 | -6.48% / +3.16% | 5.8e-03 / 4.0e-03 | trade (Pareto) |
| 8 | `xs3:split_first_lazy4_mp_rp` | 57,593,041 | 111,287 | -5.71% / +2.00% | 5.8e-03 / 4.1e-03 | trade (Pareto) |
| 8 | `xs3:split_first_lazy4_rp` | 58,550,566 | 110,005 | -4.15% / +0.83% | 1.5e-03 / 1.4e-03 | trade (Pareto) |
| 8 | `xs3:split_first_middle_mp_mpc` | 60,559,390 | 108,781 | -0.86% / -0.29% | 3.0e-03 / 2.1e-03 | **dominates frontier** |
| 8 | `xs3:split_first_middle_mpc` | 61,516,915 | 107,499 | +0.71% / -1.47% | 1.2e-03 / 1.4e-03 | trade (Pareto) |
<!-- /table -->

- **Every depth has a point that dominates the wave-1 frontier** (lower dense, lower or
  equal sparse): d4 -0.4%, d5 -0.9%, d6 -1.8% (and -1.3% sparse), d7 -2.0% (-1.8%),
  d8 -0.9% (-0.3%) dense; at unchanged dense, C5 with pair digests cuts sparse by 2.8% (d6)
  and 3.9% (d7).
- **Largest dense cuts** (trade points, Pareto-optimal, more sparse): d8 **54.6M**
  (-10.7%), d7 58.9M (-7.5%), d6 63.5M (-8.4%), d5 84.5M (-5.3%), d4 158.3M (-1.3%).
- Depth 8 has a new layout (fold + parity, section 3): 57.1M / 112.5K (-6.5% / +3.2%) and
  58.6M / 110.0K (-4.2% / +0.8%).
- Wave-1 assumptions this breaks: min-parity on counts is fine at q = 8 (but saves a unit
  only for odd ranges); lazy4c's 2-unit column reduction is not needed (1 zigzag per count
  with an odd range, or none with a fold layer); round-1 theta bits CAN share units across
  a column (exact 3 + 2/5 per bit form); "every layer except X is at ~1 unit per output
  bit" at depth 8 held for the split-middle layout only, a fold layer uses the 8 layers better.
- Also fixed: a tracer bug (functions that exit by raising unbalance its stack).

## What was done, and why each piece is exact

### 1. Min-parity on count features passes at q = 8 (MPC), and exactly when it helps

brainstorm-critic's MIN_PARITY forms (knots between integers) failed the tolerance on
count features at q = 4 (0.0265 edge margin). At q = 8 they pass: on the depth-8 last theta
(MIN_PARITY[11] on counts of 11 exact chi bits) the audit margins are 3.0e-3 / 2.1e-3
(glu_xor: 2.8e-4 / 2.6e-4), 7x below 0.02.

What min-parity can buy is small and exact: parity of an integer s in [0, n] takes
floor(n/2) units for 7 <= n <= 11 (n = 7: 3, 8: 4, 9: 4, 10: 5, 11: 5; exhaustive over
knots with denominators <= 6, wave 1 and brainstorm-critic), glu_xor needs ceil(n/2):
**one unit, and only for odd n.** (For n > 11 floor(n/2) is what the construction below
reaches; a lower bound there is not proven.)
For larger odd n the n = 11 form extends exactly (`minpar_counts.py`): its last piece is
the parabola (s - 10)^2 on [8.73, oo) (it holds 9, 10, 11), and glu_xor-style ramps
-4 max(0, s - 11 - 2j) continue it, (s - 10)^2 - 4 (s - 11) = (s - 12)^2, ..., so every
odd n >= 7 takes (n - 1)/2 units (checked exactly up to 63). The ramps' knots are on
integers, where silu averages the kinks (flat, as in glu_xor); only the MIN_PARITY[11]
head has slope at lattice points.

Applied where count ranges are odd: depth-8 last theta (n = 11: 6 -> 5 units per bit),
depth-5 Walsh layer (singles n = 13: 7 -> 6, triples n = 39: 20 -> 19), depth-4 Walsh layer
(singles 21: 11 -> 10, triples 63: 32 -> 31). The X layers' D (n = 10), the depth-4 round-2
parities (10, 20) and lazy4c's counts (32) are even: nothing to gain there directly,
which led to sections 2 and 3.

### 2. Lazy chi: one zigzag unit per count instead of two units per column (Z11; C5)

In lazy4c (unit-synthesis) a count is own lazy chi + two columns of lazy chi values, each
column's linear part c = sum_y Ea_y in [0, 10] folded into [-5, -1] by 2 units shared by
the 2 counts that read it: counts in [0, 32], 16 parity units per last-theta bit. A lazy
chi unit costs ~4.0K dense, a last-theta unit ~1.8K, so moving units from the former to
the latter pays whenever the ratio is below ~2.2.

- Exhaustive MILP (`syn/t_win_milp.py`, every gate with |a| <= 6): with ONE unit plus a
  pass term the smallest window for c in [0, 10] is 5 (e.g. the zigzag
  c - 2 max(0, c - 5)). **C5**: one unit per column; counts in [0, 34], 17 parity units.
  320 lazy-chi units out, 320 last-theta units in: -0.7M dense, and sparse drops too.
- **Z11** (better, same unit count): no column units at all, and ONE zigzag per count on
  the count's whole linear part L = own Ea + c_L + c_R in [0, 22]:
  z(L) = L - 2 max(0, L - 11) in [0, 11]. The count z(L) + (sum of 11 Q' in {0,1,2}) lies
  in [0, 33], and 33 is odd, so with MPC the last theta needs 16 units, not 17. 33 is the
  exact range (attained with Ea_own = 2, Eb_own = 0, Ec_own = 2 and L = 11).
- A 2-unit zigzag (window 7-8 on [0, 22]) would trade 1 lazy-chi unit for 2 last-theta
  units: a loss at the 2.2 cost ratio. No reduction at all (lazy4, n = 44) is worse too.

Exactness: z(L) - L = -2 max(0, L - 11) is an even integer (knot on an integer, so the
unit is exact under silu), so each count keeps the parity of the lazy4 count, which
unit-synthesis proved congruent to the next theta bit; the count is an integer in
[0, 33] whose parity the last theta takes exactly.

### 2b. Round-1 theta with units shared across a column (TH1S)

A round-1 theta bit is parity(A + T): A the own message bit, T the count of the live message
bits of the column pair (T <= 7 for 4 of the 5 columns), shared by the 5 bits of the
column (the zero lanes have A = 0). glu_xor / min-parity need 4 units for a live bit
(n = 8 is even) and 3 for a zero lane: 4 x 4 + 3 = 19 per column. A sampled search
over wave 1's per-bit triples with two shared units (`syn/t_d_q2.py`, 1 hit in 36K
samples), solved exactly with sympy (`syn/t_d_exact.py`, unique rational solution):

    parity(A + T) = -4 max(0, A + T - 3)
                    + max(0, 2A + T - 6) (-8A + 2T - 6)
                    + max(0, 5A + 2T - 1) (17A/4 - 13T/10 - 4)          (3 units per bit)
                    + max(0, 16 - 3T) (26/5 - 6T/5) + (75T/2 - 416/5)  (2 units per column)

exact on {0,1} x [0, 7] (all 16 points; integer gates, every lattice gate value is 0 or
at least 1 in magnitude, so silu is exact to exp(-32)). Zero lanes use the three per-bit
units at A = 0. Per column: 4 x 3 + 3 + 2 = 17 units instead of 19 (3 live bits:
14 instead of 15). Inputs are raw message bits (exact), so the slope at lattice points
does not matter. Columns with T = 8 (x = 1) keep the old units (no T = 8 form found).
Measured: round-1 theta 5571 -> 5202 units, -1.39M dense at depths 4-6, but +5K sparse
(the per-bit values read the 7 column bits, where glu_xor's mostly read none): a trade, not
a domination (`lazy4c_middle_m3_mp_z11_th` 66.71M / 180.6K against `_z11` 68.10M / 175.4K).
A search for a form whose per-bit values do not read T (`syn/t_d_q2s.py`, 40K sampled
triples) or read it in only one unit (`syn/t_d_q2m.py`, 54K) found nothing; neither did a
T = 8 form (`syn/t_d_q2.py` on wave 1's T = 8 triples, 30K samples; a structured grid
around the T = 7 solution, `syn/t_d_t8grid.py`, stopped unfinished). These are samples,
not proofs.

### 3. Fold, then parity: the last theta in two narrow layers (RP, new depth-8 layout)

With a spare layer, the parity of a wide count is cheaper in two steps: a narrow layer
folds each count s in [0, n] into r(s) = s - 2 max(0, s - 11) + 2 max(0, s - 22) - ...
in [0, 11] (ceil(n/11) units, knots on integers, r = s mod 2), and the next takes
parity(r) with MIN_PARITY[11] (5 units). With the fold, the lazy chi needs no linear-part
reduction at all (lazy4, counts in [0, 44]): 4 + 5 units per last-theta bit over two layers
of width ~510-620, instead of 16 (lazy4c) in one plus the 640 reduction units.
Layout `split_first_lazy4_*_rp`: X1 | Y1 | chi1 | X2 | lazy chi | fold | parity | chi
(depth 8) against the frontier's X1 | Y1 | chi1 | X2 | Y2 | chi2 | theta | chi:
57.1M / 112.5K (m3 digests, min-parity round 1), 58.6M / 110.0K (pairs, glu_xor round 1),
54.6M / 126.3K with NOP. The same fold after a direct round 1 gives a depth-7 point
(`lazy4_middle_m3_mp_nop_th_rp`, 58.9M / 181.7K): the lowest depth-7 dense, but with the
largest audit margin of this avenue (8.3e-3, still 2.4x below 0.02) and +43% sparse.
Exactness: r(s) - s is an even integer (each ramp moves by 2 times an integer, knots on
integers), so parity(r) = parity(s); r lies in [0, 11], where MIN_PARITY[11] is exact.
The per-layer sparse cost of the lazy chi's pass units (~22 E values per count) is what
keeps the fold layout ~2K sparse above the split-middle one.

### 4. Digest bits of the step before the last packed two per feature (LZ5, trade)

The lazy chi layer makes each digest-2 bit a lazy value o in [0, 4] with its own pass unit
and feature (224 each). Packing F = o1 + 5 o2 (one pass unit and one feature per pair) and
decoding F in [0, 24] into the usual pair p = d1 + 2 d2 in the next layer with the
DP-minimal decoder of `decode.py` (12 units, knots on integers, exact on all 25 values):
-112 units and -112 features, -0.6M dense, +4K sparse. A trade (at depth 8 with the fold
it loses: the decoders then sit in the fold layer).

### 5. X layers without the column-pair count features (NOP, trade)

The chi layer before an X layer emitted 320 count features P (sums of 10 chi bits) only so
that the X layer's D units could gate on one feature. The chi bits are features anyway,
so the D units gate on the 10 chi-bit features directly (same units, same values, glu_xor
on 10 inputs): 320 fewer features in two layers, -2.56M dense, +14K sparse (each D gate
reads 10 features). Applies to every layout with an X layer (depths 5-8).

## Negative results (exact synthesis, tools in `syn/`)

- **Lazy-chi product units cannot be shared across rows.** One gated unit plus a free
  affine pass, congruent mod 2 to Q(Eb1, Ec1) + Q(Eb2, Ec2) on E-encoded inputs
  (E in {0,1,2}, the X layer's output): MILP over every distinct gate with |g| <= 3
  (30,369 gates, `syn/t_ipmilp.py`, `milp_Q2_E_3.log`) is infeasible at ANY window width.
  (On exact bits a one-unit mod-2 inner product exists, wave 1; the MILP reproduces it.)
  Algebraic reason for always-active units: E^2 = 2 C(E,2) + E with C(E,2) = [E == 2] not
  linear mod 2 forces u_i v_i to be integers, which kills the half-integer trick that makes
  the bit version work (the product of two linear forms is then rank <= 2 mod 2, IP2 needs
  4). So the lazy chi keeps one product unit per chi bit.
- **Round-1 theta: which column-sharing forms do not exist.** theta_y = parity(T) +
  a_y (-1)^T with parity(T) shared: per-bit units that vanish at a_y = 0 and sum to (-1)^T
  on T in [0, n] need K >= 4 for n = 7, 8 (exact pattern search with real knots,
  `syn/t_delta_pat.py`, brainstorm-critic's par1d method without a constant; K = 3 works
  for n <= 6). With general per-bit units (wave 1's 9,539,013 feasible triples for T = 7)
  and ONE shared affine unit per column: 0 solutions (`syn/t_d_q1.py`). With one shared
  hinge plus one affine unit (Q = 2): found, see section 2b.
- **Min-parity saves one unit only for odd ranges** (see 1): nothing for n = 8, 10, 32.
- 1-unit window reductions: minimum width 5 on [0, 10] (MILP); the per-count zigzag has
  width 11 on [0, 22].

## Headroom (my view, with the floors that remain)

Where the dense of the best layouts sits now (depth 6, `lazy4c_middle_m3_mp_nop_z11_lz5_th`
and depth 8, `split_first_lazy4_m3_mp_nop_rp`), per layer, against the unit floors
established by exact synthesis in wave 1 and here:

| block | units per output bit now | floor (and why) |
|---|---|---|
| chi on exact bits | 1 | 1: the 5 bits of a row are independent functions |
| X layer (E = a + D) | 1 copy + 5/5 D | same: 1 copy per bit + parity of an 11-value count per column pair, n = 10 even, so 5 |
| lazy chi | 1 product + 2 per count (pass, zigzag; 1 with a fold layer) | products cannot be shared across rows (IP2 on E codes infeasible at any width); the pass is per count |
| round-1 theta (live bit) | 3 + 2/5 shared (T <= 7) | 3 per bit: 2 per-bit units cannot make the alternating A-difference for T >= 7 (wave 1, (d)); the shared part needs 2 (1 affine is infeasible, section 2b) |
| last theta | 5 + fold 4 (depth 8), 16 (depth 6/7) | parity of an integer range n needs floor(n/2) (exhaustive for n <= 11) |

So every full-width layer is at its unit floor for the primitives we know, and dense is
fixed by the layer widths: a layer carrying the 1600-bit state costs about
units x (2 x 1600 + 1600) = 7.7M per unit-per-bit. What is left inside these layouts:
1-2% (e.g. the T = 8 columns of round 1, digest packing trades).

Lower-bound view: depth 8 (fold layout) has 4 full-width layers (Y1, chi1, X2 at 2
units/bit, lazy chi) plus the round-1 X layer on 1145 inputs and three ~510-wide layers.
Summing the per-layer unit floors at these widths gives about 53.8M; the NOP variant is at
54.6M. Depth 6 (theta on raw bits, chi1, X, lazy chi, last theta, chi) is already at the sum
of its floors (63.5M for the NOP + Z11 + LZ5 + TH1S point); what is left there is the T = 8
round-1 columns (about -0.7M with a T = 8 form, not found yet) and packing trades.

How far each metric can go, my estimate:
- **dense**: ~53M at depth 8 and ~62.5M at depth 6 with these primitives (another 1-3%).
  A real step (2x) needs fewer full-width layers or narrower state layers, which
  the untied, residual-free MLP cannot give: every state-carrying layer has >= 1600
  independent outputs and hence >= 1600 units, and its inputs are the previous layer's
  1600+ features. Wave 1's estimate of ~50M for depth 8 stands as the practical floor.
- **sparse**: the dominating points are 0-3.9% below the frontier; the trade points pay
  1-43%. A sparse floor is roughly 3 nonzeros per unit plus norms, about 60K at depth 8;
  what keeps it at ~110K is gates that read many features (raw-bit parities read 7-9 bits,
  lazy-chi pass units ~22 E values, NOP's D units 10 chi bits). Count features trade
  dense for sparse at about 190-380 dense per sparse parameter (NOP, TH1S, mp), so any
  point on the dense end costs sparse.
- **depth**: unchanged (4 is the minimum at bounded cost, wave 1); the fold layer is a new
  way to spend the eighth layer: 57.1M against 60.6M for the split-middle layout (both with
  the same first four layers).


## Bug fixed: the tracer's stack breaks on functions that exit by raising

`reifier/compile/monitor.py` pushes a node on every `PY_START` and pops on `PY_RETURN`,
but a Python function that exits by raising emits `PY_UNWIND` instead, so any exception
raised and caught inside a traced function (also the ones importlib raises and catches
while importing a module for the first time) left the stack unbalanced and
`TreeCompiler().run` failed with `assert len(self.stack) == 1` in `Tracer.root`. I hit it
when a builder imported a helper module inside the traced function. Fix: register a
`PY_UNWIND` callback that pops like a return (`on_unwind`); regression test
`tests/tracer_unwind_test.py` (fails before, passes after). Normal traces never unwind, so
compiled circuits are unchanged (checked: `runs/regress_tracerfix_*.json` give the same
dense/sparse and margin as before for `split_first_middle_mp_mpc` and the depth-6 frontier
variant; the repo test suite passes, 62 tests without hash_long_test,
`runs/pytest_after_fix.txt`).

## Reproduce

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
D=$S/xof2/av/units-per-bit
./run.sh xs3:split_first_lazy4_m3_mp_rp        # harness + both audits -> results.jsonl
# or by hand:
PYTHONPATH=$D/repo/src:$D/repo/experiments/xof_shrink $S/venv/bin/python \
  $D/repo/experiments/xof_shrink/xofbench.py --log-w 6 --depth 3 --variant xs3:lazy4c_middle_m3_mp_z11 --widths
python3 mdtable.py                              # Pareto table of verified points
```
Other log_w (harness, 64 random messages): `runs/other_logw.txt` (`other_logw.sh`).

## Files

- `repo/experiments/xof_shrink/xs3.py`: switches `NOP`, `MPC`, `C5`, `Z11`, `LZ5`, `RP`,
  `TH1S` and the variants (`with_nop`, `with_mpc`, `with_c5`, `with_z11`, `with_lz5`,
  `with_rp`, `with_th1s`; `theta1_shared_lits` builds the column-shared round-1 theta);
  `minpar_counts.py` (new): min-parity units for any odd count range; `xs4.py`: NOP and MPC
  for the depth-4/5 layouts; `xc.py`: the depth-4/5 variants. Default variants are
  unchanged (validated: same dense/sparse as the frontier when the switches are off).
- `repo/src/reifier/compile/monitor.py`: the PY_UNWIND fix; `repo/tests/tracer_unwind_test.py`.
- `patch.diff`: `git -C repo diff` including the new files.
- `results.jsonl`: one line per variant (harness + both audits); `runs/`: raw outputs,
  `validate_small.txt`, per-layer stats `ls_*.txt`.
- `run.sh` (harness + both audits), `table.py` / `mdtable.py` (tables), `layerstats.py`.
- `syn/`: `t_ipmilp.py` (IP2 MILP), `t_win_milp.py` (window reductions), `t_delta_pat.py` /
  `t_delta.py` (per-bit forms), `t_d_q1.py` / `t_d_q2.py` / `t_d_q2s.py` / `t_d_q2m.py`
  (column-shared theta searches), `t_d_exact.py` (exact solve), `t_ramp_q.py`,
  `t_tracer_bug.py` (tracer repro), `minpar.py`, and their logs.

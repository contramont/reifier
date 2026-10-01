# What is still promising, ranked (after wave 2 + combined-2)

**Current bests (dense / sparse):**

| depth | dense | sparse |
|---|---|---|
| d3 | 253.8M | 1.29M |
| d4 | 134.7M | 548K |
| d5 | 79.4M | 238K |
| d6 | 57.3M | 192K |
| d7 | 49.4M | 137K |
| d8 | 45.07M (73x the baseline) | 116K |

Sparse-best points: d8 96.0K, d9 91.8K.

Every full-width layer of the d6-d8 points is at its unit floor for the known unit forms. The
per-layer numbers are in FRONTIER.md section 4. So the items below are ranked by expected
dense or sparse gain times the odds of success, with the cheapest decisive experiment named
for each.

## 1. A cheaper exact AND of two parities (depth 3-4; biggest single lever)

**What.** Every fused round needs (1 - p_B) p_C per chi bit.
- Today that costs a pair parity over B xor C: 13-17 raw bits, 6-8 units.
- A restriction argument allows about 4 units.
- Count-symmetric gates were searched exhaustively and found nothing: 3+3 bits with 2 units,
  4+4 bits with 2 or 3 units (depth-low `andsearch2.py`).
- Gates that weight B and C differently, or that read individual bits, were never searched.

**Worth.**
- d4x's L1 holds 12,956 pair-parity units. At 4 units instead of 8 per pair: about -24M at
  depth 4, from 134.7M to about 111M.
- A similar cut at depth 3.

**First experiment.** A MILP like `search/d2d_milp.py` with gates on (b, c) counts plus
per-bit offsets, for |B| = |C| = 4.
- Target: (1 - parity(b)) parity(c), plus free singles.
- Asymmetric integer gates, abs(w) <= 3, K = 3 units.

## 2. Fewer units for the column-pair parity D in the X layers (depths 5-8, about 1.1M per unit saved per column)

**What.** X2 spends 5 units per column on D = parity(C_L + C_R), 1600 of its 3203 units.
- That is about 5.8M of the 45M at depth 8, and about the same at depths 5-7.
- The 1-D bound (range 10 needs 5 units with integer knots) does not cover units whose gates
  read C_L and C_R separately (2-D gates on two features).
- This synthesis ruled out gate directions in {-1, 0, 1}^2. Steeper integer directions are
  still open.
- One fewer unit per column is -320 units in X2: about -1.15M at each of depths 5-8.
- `search/d2d_milp.py` is the exact MILP for this, with integer gates abs(a), abs(b) <= A and
  exact equality on the 36 points. Its result is at the end of this file (K = 4).
- If a 4-unit form exists, check its slope at the lattice points: it feeds E, then Y or the lazy
  chi, so a steep form would cost margin. Build it into `cp_d_specs` (xs3) and xs4's
  `x_exact_cp`.

## 3. A 2-unit, width-2, mod-2 double AND for the lazy chi (depths 6-7, about -1.9M; less at depth 8)

**What.** Mod 2, the lazy chi of E-coded bits is Ea + Ec + Eb Ec.
- One unit for two AND terms is infeasible: exhaustive search (first-last, units-per-bit).
- Two units for two AND terms with an output window of 2, instead of 4, would cut the lazy count
  range from 33 to about 24. That removes 4-5 of the 16 theta3 units per bit at depths 6 and 7.
- At depth 8 it would narrow the counts that enter the fold, so fewer fold units are needed.
- Never searched: the gate-pair space was too big for a brute-force enumeration.

**First experiment.** A MILP over pairs of gates on (Eb1, Ec1, Eb2, Ec2) with E in {0,1,2} and
integer weights up to 2.

## 4. Round-1 theta for the T = 8 columns (depths 5 and 6, about -0.7M)

**What.** TH1S (3 units per bit + 2 per column) is exact only for T <= 7. The x = 1 columns
(T = 8) keep 4 units per bit.
- Sampled searches (units-per-bit) found no form.
- A structured grid around the T = 7 solution was left unfinished (`syn/t_d_t8grid.py`).

**Worth.** About -0.7M at depths 5 and 6, and also at depth 4 in the wave-1 d4a layout.

## 5. Noise-robust min-parity on counts (depth 8: -0.46M at the same margins)

**What.** MPC saves one unit for odd count ranges. It is used in the depth-8 parity layer
(n = 11), the depth-6/7 last theta (n = 33) and the Walsh layers. Its knots between integers
raise the depth-8 worst margin from 5.0e-3 to 8.6e-3.

**Options.**
- A 5-unit parity of [0, 11] that is flat at every integer would get the saving back at
  glu_xor-level noise. Flat means zero slope, or kinks symmetric about the integer, as in
  glu_xor.
  - It probably does not exist: every unit interval then needs a vertex or a symmetric kink,
    and glu_xor already spends 6 units on exactly that.
  - It is cheap to settle with a MILP like `search/d2d_milp.py`, using a 1/2 knot grid and
    slope constraints.
  - Expected gain if it exists: -0.46M at depth 8 at the robust margin.
- Folding to a different window does not help:
  - folding to [0, 9] or [0, 10] costs one more fold unit per count, since the fold is a pass
    unit plus one ramp per window;
  - in exchange, the parity layer can use a 5-unit glu_xor or MIN_PARITY[9] (4 units);
  - net: +0.09M to +0.54M against the current MIN_PARITY[11].

## 6. Sparse-side stacking (sparse-focus x packing)

**What.** sparse-focus's g-fold and D-sep save 6.4K and about 2.8K per round. They are
independent of cp, u1 and RP, but no point combines them:
- g-fold: chi reads one feature g = 2a - b + c, 3 nonzeros instead of 7;
- D-sep: D is one feature, and Y computes a xor D.

**Worth, as estimated before the measurement below:**
- depth 8: about 100-105K sparse at about 50M dense (today 107-111K at 47-51M);
- depth 9: col1 round 1 + cp + RP, possibly below 91.8K sparse.
- The family floor is about 85-88K (sparse-focus accounting).

**Measured here, and it loses on top of u1.** The switch `GF1` in xs3 (`c2:d8_rp_cp_u1_g_gf`)
lands at 48,404,742 / 114,917, against 47,632,398 / 112,765 without it.
- Each u1 theta bit is a sum of 3-4 decode units, so a g feature collects about 10 wo entries.
  A chi unit saves only 4.
- g-fold pays only where every theta bit is one unit, as in sparse-focus's D-sep Y layer
  (t = a xor D).
- So the sparse-side stack is sparse-focus's own layout plus cp into X2 and the fold layout,
  not u1 plus g-fold.

## 7. Digest routing at depths 7-8 (about 0.5-1M)

**What.** Digest 1 is 56 features at m4s4. It crosses X2, lazy4, fold and parity. Digest 2 is
224 lazy values through the fold, then pairs.
- About 2M of the depth-8 total is digest carrying.
- s6 (6 bits per feature) fails float32.
- Two things are untried:
  - a two-stage split (m4 -> m2 in the fold instead of the parity layer);
  - digest 2 in LZ5 pairs decoded in the parity layer instead of the fold.

**Caveat.** Both are small, with the cost model to price them first: `overlay_opt/cost_model.py`
(optimizer) is exact for the xs3 kinds but does not model RP, cp, u1 or s4 yet. Extending it is
cheap and makes 2-7 quick to price.

## 8. Extend the optimizer's exact cost model to the combined option space

**What.** The optimizer's `cost_model.py` matched the harness to the parameter on 24 variants
and searched its whole option space.
- Adding RP, cp, u1, s4, Z11/C5, LZ5, TH1S and s_last would confirm that the combined points are
  optimal within the union of the avenues.
- It would also price items 2-7 before anyone builds them.

**Worth.** Expected gain is small (0-1%), but the value is certainty about the floor.

## 9. Depth 4 and 5 tails

**What.** d4x's L3 + L4 and d5's last two layers are the xs4 lazy round plus the Walsh layer:
- 42M at depth 4;
- 42M at depth 5.

depth-low thinks the tail is about 2x its floor.

**Possible routes.**
- A Walsh layer on narrower counts (Z11-like zigzags in the lazy layer).
- cp-style packing of the digests entering the Walsh layer.

**Worth.** Uncertain: perhaps -3 to -8M at depths 4 and 5.

## Not promising (measured or proved)

- **Packing into Y, chi or lazy-chi inputs.** The one-unit lemma (tricks-critic) rules it out
  at one unit per bit, and packed readers with 2 or more units per bit lose. Measured by
  first-last, packing, round-structure and optimizer.
- **Depth 2.** Ruled out.
- **Depth 3 below about 231M** without item 1.
- **Depth 9 or 10 for dense.** No extra layer pays for itself.
- **u pairs in round 2 next to cp.** The 5th bit then needs 2 more hi units per column. It pays
  only in the split-middle layout, which RP beats.
- **The fold with Z11 at depth 8, and LZ5 at depth 8.** Both lose.
- **m4 without s4.** Loses.
- **6-bit digest packing (s6).** Fails float32.
- **Sharper steepness.** The margins are float32 rounding, not silu.
- **2x in this architecture.** It needs residual or tied weights (wave 1: 56.7K sparse with a
  gated-residual tied variant), which the task excludes.

## Result of the D search (item 2), run during synthesis

The search is an exact MILP (HiGHS, `search/d2d_milp.py`) over every distinct integer gate
(a C_L + b C_R + c), with value coefficients bounded by 64 and exact equality on the 36 lattice
points. Logs are in `search/d2d_*.log`.

| run | gates | limit | result |
|---|---|---|---|
| sanity, K = 5, abs(a), abs(b) <= 2 | 528 | 300 s | finds a 5-unit form (all gates along a = b, i.e. glu_xor on the sum) |
| K = 4, abs(a), abs(b) <= 2 | 528 | 900 s | no feasible point found; infeasibility not proven |
| K = 4, abs(a), abs(b) <= 3 | 1584 | 1200 s | no feasible point found; infeasibility not proven |
| K = 4, abs(a), abs(b) <= 2, value coefficients up to 8 | 528 | 1300 s | no feasible point found; infeasibility not proven |
| K = 4, abs(a), abs(b) <= 1, value coefficients up to 16 and up to 256 | 118 | < 1 min | **infeasible (proved)** |

Gates with directions in {-1, 0, 1}^2 are the natural ones on two count features. For those,
no 4-unit exact D exists, even with value coefficients up to 256. Steeper directions (abs(a),
abs(b) >= 2) are still open, and so are knots between integers.

So the question stays open. To settle it:
- break the symmetry: order the selected gates and fix one gate direction per unit class;
- or enumerate gate 4-sets with a row/column prefilter: each of the 6 rows and 6 columns
  restricts to a 1-D parity of range 5, so it needs several knots strictly inside the grid.
  That prunes the 4-sets whose knot lines miss rows or columns;
- then solve the 13-unknown linear system per surviving set.

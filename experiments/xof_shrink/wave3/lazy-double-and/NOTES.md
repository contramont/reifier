# lazy-double-and (wave 3, NEXT.md item 3): a two-unit mod-2 double AND for the lazy chi

**Result: no construction; the item is closed negatively inside a well-defined space.** With two
gated units plus a free affine pass term on the four E values of two same-column chi bits, no
mod-2 double AND has an output window of 3 or less (so none of window 2). Checked exhaustively
for integer gate weights |w| <= 2 with every combination of integer and half-integer knots
(9.3M gate pairs up to symmetry), and for |w| <= 3 with integer knots (67.5M pairs, 18.2M exact
checks after a sound prefilter). No quadratic polynomial works at all, and a relaxation over
every real knot offset (|w| <= 2, window 2) leaves only spurious hits. No new Pareto point;
nothing was built into the frontier variants, and the repo copy is unchanged. The planned -1.9M
at d6/d7 (count range 33 -> about 24) is not available from pair units.

## 1. The question

The lazy chi layer (d6/d7: lazy4c + Z11 + MPC) emits, per next-theta bit, a count
C = z(L) + sum_i h_i over the own bit and the two neighbouring columns (11 chi bits):

- z(L) = L - 2 max(0, L - 11), L = sum of the 11 Ea (one zigzag unit per count), range 11;
- h_i = 2Eb + (Eb + Ec)(1 - Eb) (one product unit per chi bit), h = Q(Eb, Ec) (mod 2),
  Q(b, c) = [b != 1][c == 1], window 2 (values in {0, 1, 2});
- so C lies in [0, 33] and the last theta spends MPC(33) = 16 units per bit.

Item 3 asks for two units per two AND terms whose sum has window 2 instead of 4: pairing the four
non-own bits of each column (the own bit y = x of column (x, z) must stay single, because the
count at (x, x, z) reads it alone) gives C in [0, 25], MPC 12 units (-4 per bit, about -1.8M at
d6 and d7), or C in [0, 29] with window 3 (-2 per bit, about -0.9M).

Precisely: find F(E) = sum_{k=1,2} max(0, w_k.E - tau_k + beta_k) (v_k0 + v_k.E) + (l0 + l.E) on
E = (b1, c1, b2, c2) in {0,1,2}^4 with F integer, F = Q(b1,c1) + Q(b2,c2) + flip (mod 2) and
F in [0, W], on all 81 points. flip = 1 covers windows that start at an odd value. v, l are free
reals (the count's pass unit takes any affine term), so a pair is decided by its two gates.

Why all 81 points (no don't-cares): E = a + D, and for every pairing of the four non-own rows
of every column (1,920 pairs), the four E sit at distinct positions with distinct D columns, and
the four D (each a xor of two column parities) are GF(2)-independent (`search/indep.py`). Why
the model is not too narrow: the four E of a pair are independent of the zigzag's Ea and of every
other product in the count, so extra inputs to the pair's two gates or values restrict to this
problem (fix them; they fold into the biases); and a unit shared by the two counts that read the
column is only more constrained than a unit used by one count (output weights are per count and
are absorbed into v). Not covered: pair units that also take over the zigzag (three units mixing
L with the pair), which is a different problem.

## 2. Method

- **Gates.** Every distinct relu profile max(0, w.E - tau + beta) on the 81 points, integer w in
  [-R, R]^4, integer tau, with beta = 0 (knots on lattice points; R = 2: 5,024 gates, R = 3:
  27,968) or beta = 1/2 (knots between lattice points; R = 2: 6,000 gates). Directions with a
  common factor are included, so beta = 0 with R = 2 also contains the half-integer knots of the
  directions in {-1, 0, 1}^4.
- **Symmetry.** T is invariant under E_i -> 2 - E_i for each of the 4 coordinates and under
  swapping the two AND terms (32 elements). Every gate orbit contains a gate with w >= 0, so
  gate 1 runs over the w >= 0 gates only (a gate counts as w >= 0 if any of its representations
  is), and pairs of two such gates are taken once.
- **Exact per-pair check** (`search/dand4.py`, `feasible`). For fixed gates the 15 unknowns
  (v1, v2, l) enter linearly. Rows are grouped by gate-activity region (both off first), pivot
  rows are chosen blockwise by QR, and a breadth-first enumeration assigns each pivot one of
  its allowed values ({0,2} / {1} for W = 2, {t, t + 2} for W = 3), pruned after every pivot by
  all rows the pivots so far determine. It is exhaustive in F (no LP relaxation). A population
  cap (2^18) returns "skip"; it cannot trigger at W = 3 (at most 15 two-valued pivots, 2^15)
  and did not trigger in any reported run (skips = 0 where counted).
- **R = 3 prefilter** (`search/dand7.py`, sound). On a sub-box B where neither gate changes
  sign, F|B is affine (both off), l + g_k v_k with g_k known (one on), or a quadratic (both on).
  If that class cannot meet the constraints on B, the pair is out for that flip. Boxes: all 6^4
  interval products for the affine test (576 of 1296 boxes are affine-obstructed, e.g. any
  2 x 2 face {0,1}^2 of an AND pair), the 81 products of [0,1], [1,2], [0,2] for the other two
  (25 of them are quadratic-obstructed). Survivors get the exact check.
- **Validation.** Sanity targets reproduce the known facts: the single AND has exactly the known
  one-unit window-2 forms (18 gate/offset hits at R = 2) and none of window 1; two units give
  the exact single AND (774 pairs); one unit gives no double AND at W = 2, 3, 4 (as first-last
  found). A planted pair (the trivial h(b1,c1) + h(b2,c2), window 4) is found at W = 4 and
  rejected at W = 3 (`search/planted.py`). The R = 3 prefilter loses nothing on the W = 4, R = 1
  test (332 feasible pairs, the same as the unfiltered search), and the null-space prefilter of
  `dand5.py` loses nothing on the single-AND test (3,533 = 3,533). An independent MILP (HiGHS,
  F = t + 2k with integer k) agrees with the solver's verdict on 900 random gate pairs at W = 3
  and W = 4 (`search/milpcross.py`).

## 3. Results

(W = 3 infeasible implies W = 2 infeasible: F in [0, 2] is a special case of F in [0, 3].)

| search | gates | unordered pairs (up to symmetry) | W | feasible | log |
|---|---|---|---|---|---|
| R = 2, integer knots | 5,024 | 2,473,035 | 2 | **0** | `search/w2_r2_int.log` |
| R = 2, integer knots | 5,024 | 2,473,035 | 3 | **0** | `search/w3_r2_int.log` |
| R = 2, both knots at +1/2 | 6,000 | 3,678,372 | 3 | **0** | `search/w3_r2_half_half.log` |
| R = 2, one knot at 0, one at +1/2 | 5,024 x 6,000 | 3,114,000 | 3 | **0** | `search/w3_r2_int_half.log` |
| R = 3, integer knots (prefilter + exact) | 27,968 | 67,534,416 (18,211,536 pair/parity survivors checked exactly) | 3 | **0** | `search/d7_w3_r3.log` |
| R = 2, every real knot offset (linear relaxation) | 5,216 | 2,647,715 | 2 | 28 relaxation hits, none real | `search/w2_r2_rel.log`, `search/beta12_w2.log` |

The relaxation replaces (s - tau + beta) v on the active set {s >= tau} by p + s q with p, q
independent. Its 28 hits are all separable pairs (each gate reads one AND pair), and the relaxed
"units" there are step functions [A] p that make the exact AND. None is realizable: with the true
units, knot offsets beta1, beta2 in every fraction with denominator <= 12 give nothing. At W = 3
the relaxation is too loose to use (spurious step-function hits), so the half-integer knots were
searched directly instead.

## 4. Structural facts found on the way

- **No quadratic works** (`search/quadcheck.py`). No quadratic polynomial P on {0,1,2}^4 (any real
  coefficients) has P = T (mod 2) with window 2 or 3; window 4 has 16 solutions, non-flipped
  only. So any window-3 double AND needs a knot through the grid, and every quadratic-obstructed
  box must be cut by a knot.
- **Obstructions** (`search/obstr.py`). For W = 2, all 24 boxes of shape 3 x 3 x 2 x 2 and the six
  3-D slices b1 = 0, b1 = 2, c1 = 1, b2 = 0, b2 = 2, c2 = 1 are quadratic-obstructed; for
  W = 3, 16 of the 24 boxes and no 3-D slice. An affine function cannot even carry one AND on a
  2 x 2 face (the parity step in c has to differ between b = 0 and b = 1), so the region where
  both gates are off cannot contain any such face.
- **The one E-sharing pair in a count ("chain").** The own bit (x, x, z) and the column bit
  (x-1, x, z) share E(x+1, x, z) (Ec of one, Eb of the other). Keeping one product unit per bit
  (each a valid single-AND unit, as the other counts that read it need) and a free joint pass
  term, their sum can have window 3 instead of 4, but not 2 (`search/chain2.py`, `chain3.py`,
  `chain4.py`: 16 distinct single-AND units mod affine, gates |w| <= 3 with integer or
  half-integer knots; 24 window-3 solutions at |w| <= 2, each with at least one half-integer
  knot). Moving the own Ea from the zigzag into the chain gives window 4 at best. So the count
  range drops from 33 to 32, which saves nothing: min-parity saves a unit only for odd ranges
  (MPC(33) = 16 = glu_xor(32) units). The chain helps only where the count range is even
  (C5: 34 -> 33, lazy4c: 32 -> 31, -1 theta3 unit per bit with MPC), and neither then beats
  Z11 + MPC: C5 + chain + MPC has Z11's unit counts (same dense) with more nonzeros than C5
  (MPC units read more weights than glu_xor units), and lazy4c spends 320 more lazy units
  (+1.2M) to save 320 theta3 units (-0.46M). Z11 + chain with glu_xor instead of MPC keeps the
  dense exactly; by my count its sparse changes by about +0.6K (-1.6K from glu_xor units, +2.2K
  from the chain units' extra gate/value weights and pass entries), so it is a margin-only
  option (flat parity knots, but the chain units have knots between integers). Not built.
- **One unit per AND is the floor per bit**: a single product unit with any linear parity moved
  into the zigzag cannot reach window 1 (integer knots, |w| <= 4; `search/single_w1.py`).

## 5. What this means for the lazy count, and what is left

With one product unit per chi bit and one zigzag per count, the lazy count range at d6/d7 is at
its floor for everything searched here: 11 (zigzag) + 3 (the chained pair) + 9 x 2 = 32, and 33
is what MPC needs anyway. The planned -1.9M (d6, d7) does not exist in this family; at d8 the
fold needs count <= 33 to drop a fold unit, and even window-2 pairs would only give 36 there.
FRONTIER.md's d7 floor estimate (about 47.5M) counted this item; without it, theta3 at d7 stays
at 16 units per bit on a 489-wide input (9.8M), and the remaining d6/d7 levers are NEXT.md
items 2 (2-D gates for D), 4 (T = 8 theta1, d6) and 7 (digest routing).

Still open (not searched, much larger spaces):
- three or more AND terms with as many units (6 or 8 E inputs; a window-5 triple would give
  count 27, 13 units, about -1.4M);
- gate directions beyond |w| <= 3, real knot offsets other than 0, 1/2 at R = 2 (for W = 3),
  and non-integer directions. An argument for all real gates is still missing; the facts in
  section 4 (no quadratic, affine-obstructed faces) are the ingredients such a proof would use.

## 6. What went wrong along the way (for anyone rerunning)

- `dand2.py` took gate 1 from the gates whose stored representative has w >= 0; a profile can
  have several (w, tau) representations, so some orbits could be missed. `dand3.py`/`dand4.py`
  mark a gate as w >= 0 if any representation is; the W = 2 run was redone with that (same
  result, 0).
- numpy's BLAS threads inside 28 pool workers oversubscribed the (shared, load 50-100) machine;
  the scripts pin OMP/OPENBLAS threads to 1.
- The real-offset relaxation is loose (a relaxed unit can be a step function times an affine
  function, which gives the exact AND for free), so it only certifies absence where it is
  infeasible; hits need the direct check. At W = 3 it produced many spurious hits and population
  skips, so it was replaced by the direct beta in {0, 1/2} searches.
- The W = 2 linear prefilter of `dand5.py` is useless for the non-flipped case: the constant pass
  term already satisfies "F = 1 on the fixed rows", so only the null-space version is meaningful.

## 7. Files

- `search/dand4.py`: the exhaustive search (int knots, or the linear relaxation for real knots);
  `dand6.py`: fixed knot offsets; `dand7.py`: R = 3 with the sound box prefilter;
  `dand2.py`, `dand3.py`: earlier versions (dand2's symmetry reduction was incomplete; its run
  is superseded by dand3's `w2_r2_int.log`); `dand5.py`: W = 2 null-space prefilter (sound but
  weak, unused).
- `search/quadcheck.py`, `obstr.py`, `chain*.py`, `single_w1.py`, `betacheck.py`, `planted.py`,
  `cmp45.py`, `milpcross.py`, `indep.py` (needs the venv and the repo on PYTHONPATH): the checks
  of sections 1-4. `search/*.log`: the raw runs.
- `results.jsonl`: one line per search and check (`python3 collect.py` rebuilds it from the logs).
- Run with the system python3 (scipy + numpy; the venv has no scipy), e.g.
  `python3 search/dand4.py 3 2 --jobs 28`.

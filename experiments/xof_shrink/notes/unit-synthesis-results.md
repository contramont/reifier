# unit-synthesis: minimum gated units for Keccak pieces (live results, updated as they come in)

Model: one layer, output = sum_k max(0, g_k(x)) * v_k(x), g_k integer affine (integer
weights and bias), v_k rational affine; a constant costs a unit (gate on BOS). Multi-output:
hidden units shared through wo.

Tools (all in this directory, run with `svenv/bin/python`, a private venv with scipy/ortools):
- `synth.py`: exact search. For FIXED gates the target is linear in the value weights, so
  representability is a column-span test. Gates are enumerated as distinct ReLU vectors
  (integer weights |w| <= W), K=2/3 sets are searched with a prefix projection + batched
  normal equations, hits are re-solved exactly (sympy rationals).
- `symm.py`: orbit-canonical first gates (e.g. parity: coordinate permutations x even
  complements, 1920 elements for n=5), so K=2 searches are exhaustive but cheap.
- `lazy.py` (first version) and `lz.py`: "mod 2" synthesis: outputs only need y = target
  (mod 2) inside a window [lo, lo+R]; `lz.py` enumerates the allowed values on pivot rows of
  the fixed-gate design matrix and checks the rest (exact), for the encodings E/F/G/AD.
- `t_k3.py` (exhaustive 3-unit), `t_ext.py` / `t_ext_pass.py` (restriction-extension),
  `t_par1d*.py` (1-D parity), `vpx.py` (numeric fits, evidence only); see NOTES.md.
- `cpsat.py`: CP-SAT model (bounded integer gates and values); works for small cases,
  times out on the 6-input ones (the bilinear products are hard for it).

## (a) n-bit parity: fewer than ceil(n/2) units?

| n | ceil(n/2) | result |
|---|---|---|
| 2 | 1 | 1 (xor = max(0,a+b)(2-a-b)) |
| 3 | 2 | 1 unit impossible (exhaustive, all gates with abs(w) <= 3) -> 2 |
| 4 | 2 | 1 impossible -> 2 |
| 5 | 3 | **2 impossible**: exhaustive over ALL gate pairs with abs(w) <= 3 (129,001 distinct ReLU vectors, 547 orbit-canonical first gates) -> 3 |
| 6 | 3 | 2 impossible (fixing x6=0 in a 2-unit 6-bit solution gives a 2-unit 5-bit one, same weights) -> 3 |
| 7 | 4 | **3 suffice** with knots between integers (count s = sum of the bits, i.e. symmetric weights): `t_par1d.py`, gates -6s+18, -5s+27, 2s-3 |
| 8 | 4 | 3 impossible for knots p/q with q <= 6 (exhaustive 1-D) -> 4 |
| 9 | 5 | **4 suffice** (knots between integers), e.g. gates -6s+7, -6s+30, -5s+37, 2s-7 |
| 10 | 5 | 4 impossible for knots p/q with q <= 6 (exhaustive 1-D) -> 5 |
| 11 | 6 | **5 suffice** (brainstorm-critic, knots between integers) |

**CORRECTION (supersedes the "1-D theorem" that stood here).** The claim "parity of an
integer s in [0, top] needs exactly ceil(top/2) units" is FALSE. Its proof sketch wrongly
assumed that after a 3-point piece every following piece covers 2 points and curves the
same way. With knots strictly between integers and units that open to the left
(max(0, b - a s)(p s + q)), a continuous piecewise quadratic can alternate 3-point and
2-point pieces (two adjacent 3-point pieces are still impossible), about 2.5 points per
unit: brainstorm-critic's exact search (`av/brainstorm-critic/exp/par1d.py`) found minima
n=3: 2, 5: 3, 9: 4, 11: 5, and this avenue's `t_par1d.py` confirms n=7: 3, n=9: 4
(exhaustive over knots with denominator <= 6). What remains true: with every knot ON an
integer (glu_xor's structure), ceil(top/2) is needed (brainstorm-critic, n=11), and only
the integer-knot form is flat at the lattice points (silu averages the kink slopes), so
the cheaper forms amplify input noise (brainstorm-critic measured an edge-case margin of
0.0265 in a count-based design, fine on raw message bits). For Keccak: min-parity is safe
in the first theta (raw bits: n=7 -> 3, n=9 -> 4 units) and risky on counts.
Asymmetric weights (non-count gates) gave nothing below these for n <= 6.

## (b) chi on xor inputs f(A1,D1,A2,D2,A3,D3) = (A1^D1)^(~(A2^D2)&(A3^D3))

- Lower bound 2: fixing A2=D2=0 leaves 4-bit parity of (A1,D1,A3,D3).
- 2 units: impossible in symmetric features (s1=A1+D1, u=A2+D2, s3=A3+D3; abs(w) <= 3,
  exhaustive), impossible in E-space (E_i = A_i + D_i, 27 points, same thing), and in
  E' = A - D space. Full 6-bit exhaustive K=2 (abs(w) <= 2, 110,313 ReLU vectors, 4,927
  canonical first gates under the 64-element symmetry group): 0 hits (complete).
- 2 units, larger weights, by restriction (`t_ext.py`): on the slice A2 = D2 = 0 chi is
  4-bit parity of (A1, D1, A3, D3), so the two units restricted there must solve it. All
  12,252 two-unit (+ constant) solutions of 4-bit parity with gate weights abs(w) <= 4
  (first gate canonical under the chi symmetries) were extended with every (A2, D2) gate
  weight pair in [-3, 3] per unit: none gives chi on the 64 points; with abs(w) <= 5 on
  the slice (74,361 solutions), none either. => 2 units impossible. With a free affine
  pass term as well (`t_ext_pass.py`, slice abs(w) <= 2, (A2,D2) weights in [-2,2]): none
  (slice abs(w) <= 4: none in the first 50 of 2407 first gates, 19,374 slice pairs; stopped).
- 3 units, exhaustive K=3 (+ constant) in the free one-feature encodings of the X layer
  (`t_k3.py`): E = A + D (27 points) abs(w) <= 3: none; F = A - D and G = A + 2D (64
  points, bijective) abs(w) <= 2: none. With a free affine pass term: E abs(w) <= 3 and
  F abs(w) <= 2: none (F abs(w) <= 3 + pass: none in the first 80 of 499 canonical first
  gates, stopped).
- 4 units: theta-structure's optimizer fits 4 with real gates; its integer snaps fail and
  CP-SAT (abs(w) <= 3, values in halves) times out at 600 s, so an exact integer-gate
  4-unit chi is not established.

## (c) whole chi row (5 outputs, units shared through wo): exactly 5 units

Lower bound: the 5 row outputs are linearly independent functions on {0,1}^5, and every
output is a linear combination of the hidden units, so at least 5 units. Upper bound: one
unit per bit (the existing CHI/NCHI units). Sharing cannot help an exact chi row.

## (d) parity(A + t), A a bit, t in 0..T (theta: t = the 10 column bits, shared by 5 bits)

Split: per-output units (gate/value on A and t) + shared units (on t only, one copy per
column, reused by the 5 bits of the column through wo; unit sharing in layer_to_units
merges them automatically).
- Only per-output units change the A-difference Delta(t) = f(1,t) - f(0,t) = (-1)^t.
  A per-output unit contributes at most 2 breakpoints to Delta (one per A-slice) and is
  linear in t where both slices are active, so the per-output count is about T/4, half the
  standalone parity.
- Exhaustive over gates alpha*A + beta*t + gamma (abs(alpha) <= 12, abs(beta) <= 3):

| T | 2..6 | 7, 8, 9 | 10 |
|---|---|---|---|
| per-output units P | 2 | 3 (2 impossible) | 3 (needs half-integer breakpoints, beta = +-2; 2 impossible) |

  e.g. T=10: gates `t + 2 - 7A`, `2t - 3 - 10A`, `2t - 7 - 10A` (459 triples found).
- Shared units Q for those triples (exact subset search over shared gates at the slice
  breakpoints, `t_d_shared2.py`): Q = 5 for one triple, 6 for the other 458. Values are
  large (up to ~44), see `t_d_exact.py`.
- Cost for theta with t in 0..10 as a free feature: 3 + Q/5 units per bit instead of 6.
  For theta-structure's layouts this does NOT beat "X layer (D + copy, 2/bit) + lazy chi":
  B directly via (d) = ~4.2/bit + exact chi 1/bit vs 2 + 3. In round 1 the constant-A bits
  need their own parity(t) units, which eats the gain.

## (e) other

### Parity on 2-D count grids does not help
parity(s1 + s2) with gates affine in (s1, s2) (the theta layer reading two partial counts
instead of one): exhaustive (abs(w) <= 6) on [0,3]x[0,3] and [0,4]x[0,2]: 2 units
impossible, i.e. no better than the 1-D count (ceil((a+b)/2) = 3). `t_grid.py`.

### Lazy (mod 2) chi: units vs output width (tool `lz.py`, `t_lz_k1.py`, `t_lz_k2.py`)
The lazy chi output o only has to be congruent to chi mod 2; the next theta then needs
the parity of a sum of 11 such outputs, which costs ceil(11 w / 2) units for a window
of width w. A free affine "pass" term (any linear function of the E's) costs one unit per
next-theta count, not per bit, because counts are free sums.
- 1 unit + pass, E or F encoding: minimum width 4 (widths 1-3 impossible, abs(w) <= 4,
  exhaustive). Example (E): `o = Ea + 2 Eb + max(0, Eb + Ec) (1 - Eb)`
  (= Ea + 3Eb - Eb^2 + Ec - Eb Ec, the gate is always active, so it is a pure product)
  is congruent to chi and in [0, 4].
- G = A + 2D: no 1-unit + pass solution of width <= 5 (G mod 2 = A, not theta).
- AD (A and D as features): no 1-unit + pass solution of width <= 2 (abs(w) <= 2).
- One-unit mod-2 AND: `q = max(0, Eb + Ec - 1) (2 - Eb)` is congruent to [Eb==1][Ec==1]
  and in {0,1,2} (the center of the 3x3 grid is the only odd point; (0,2) and (1,2) give 2).
  By contrast [Eb!=1][Ec==1] (what chi needs) has no 1-unit width-2 form (midpoint argument).
- **One-unit Q (the AND part of lazy chi):** `Q' = 2 Eb + max(0, Eb + Ec) (1 - Eb)` is
  congruent to `[Eb != 1][Ec == 1]` (= theta_c + theta_b theta_c mod 2) and lies in {0,1,2}
  (table over (Eb,Ec): 00:0 01:1 02:2 10:2 11:2 12:2 20:2 21:1 22:0). The gate Eb + Ec is
  never negative, so the unit is the exact product; `2 Eb` is a free pass term (it merges
  into the one pass unit of whatever count the bit is summed into). It replaces the 2-unit
  exact Q = max(0, 3-4Eb-2Ec) Ec + max(0, 4Eb-2Ec-5) Ec wherever only counts are needed.
  lazy chi `o = Ea + Q'` (range [0,4]) is the 1-unit + pass solution above.
- **Measured (implemented as layout kind `lazy4` in `impl/xs3.py`, a copy of xof-structure's
  builder):** `xs3:lazy4_middle` = depth 6, dense **72,920,840**, sparse 171,231 (vs
  lazy_middle 76,805,192 / 166,687: dense -5.1%, sparse +2.7%); `xs3:split_first_lazy4` =
  depth 7, 66,272,351 / 124,782 (vs split_first_lazy 70,156,703 / 120,238). Both ok,
  edge-case check (63 messages) margin 0.0076, validate_bench weights bit-equal.
  L4 drops from 4914 to 2258 units, L5 grows from 3858 to 7602 (the last theta now takes
  the parity of a count in [0, 44]: 22 units instead of 11).
- **lazy4c** (lazy4 + a 2-unit reduction of each column's linear part c = sum_y Ea_y in
  [0,10]: `r(c) = max(0, 9-3c)(-2/3 - c/3) + 2 max(0, c-7) + (2 - c)`, congruent to c, in
  [-5,-1]; count range 32 instead of 44): `xs3:lazy4c_middle` depth 6, dense **71,964,680**,
  sparse 171,231 (-6.3% / +2.7% vs lazy_middle); `xs3:split_first_lazy4c` depth 7,
  **65,316,191** / 124,782. Edge-case margins 0.002, validate_bench bit-equal.
- 2 units, no pass, width 2: none in E (abs(w) <= 4) and F (abs(w) <= 4), exhaustive over
  canonical first gates x all second gates, windows [0,2] and [-1,1]. G (abs(w) <= 3): none.
  E with abs(w) <= 6: none in the first 1500 of 4616 canonical first gates (stopped).
- **2 units + pass, width 2: exists** (E: 50 gate pairs with abs(w) <= 3; F too), e.g.
  `o = max(0, 8 - Ea - 3Eb - Ec)(-1 - Eb/2) + max(0, Ea + 3Eb - Ec)(2 - Eb/2) + 8 - 2Ea - 5Eb`,
  in [0,2] and congruent to chi. In the depth-6 cost model it ties with lazy4c (L4 3744
  units + L5 11/bit vs 2898 + 16/bit: 22.9M vs 22.4M), so it is not used.
- Exact chi in E-space: 3 units + constant impossible for abs(w) <= 3 (exhaustive, 499
  canonical first gates); 3 units + pass impossible for abs(w) <= 3.
- Reducing a count mod 2 into a window (1-D, P in [0,10], the X layer's D): width 1: 5 units;
  width 2: 4 units + pass; width 3: 3 + pass; width 4: 2 + pass (`t_win1d.py`). In the X layer
  the pass term is free (it merges into the copy unit of A), so a wider D is cheaper, but
  the wider E then widens every lazy chi and the last theta; no net gain found.
- Mod-2 inner product on exact bits: `max(0, b1+c1+b2+c2-1) (b1+c1-b2-c2)/2` is congruent to
  b1 c1 + b2 c2 and in {-1,0,1}: two AND terms in ONE unit (`t_ip2.py`). No symmetric-gate
  one-unit IP_3. Only useful where mod-2 sums of products of exact bits are consumed; in the
  current layouts chi1 must be exact per bit (the X layer copies it), so it is not used.
- Cost model for the depth-6 layout (L4 unit ~4.1K dense, L5 unit ~1.9K): per chi2 bit,
  lazy4 saves 2 L4 units and adds width 2 to ~2.2 counts (~2.2 L5 units), so lazy4 wins
  everywhere; exact column-parity tricks (xof-structure's lazy_chi_cols) combined with Q'
  give at most ~1-2% more.

### Parity of several chi bits in one layer (would remove the X layer)

The X layer exists because D2 = parity of the 10 chi1 bits of a column pair needs those bits
first. If the chi1 layer could compute D2 straight from the 30 theta1 bits with K_D units per
column pair, E2 = chi1 + D2 would be a free sum there, the X layer and its 1600 copy units
would disappear, and a round would take ONE layer of work plus a lazy chi: cost model
L2 = 4643 x (1600 + 320 K_D) dense, depth 5 total ~ 53.8M + 1.49M x K_D (break-even with
the depth-6 lazy4c at K_D ~ 12; with a Walsh last round, depth 4 at ~76M + 1.49M x K_D,
against xof-structure's d4 at 168M).
Lower bound: K_D >= 10 (b = 0 gives parity of 20 bits). Measured for XOR of m chi bits
(3m inputs): m = 2: 2 units impossible (L-space abs(w) <= 6 and bit space abs(w) <= 1,
exhaustive, with constant); numeric fits (vpx.py, 256 restarts): 3 units residual 0.14,
4 units 0.0065, 5 units 0.0032 (not exact), i.e. at least ~4-5 units for m = 2 where parity
of 4 bits needs 2. chi is not a threshold function (chi(a,0,c) = a ^ c), so no affine form
counts chi bits and glu_xor's trick does not apply. m = 3 (512 points): 6 units residual
0.73, 7 units 0.65, 8 units 0.40, i.e. far from exact; the count grows much faster than 2
per chi, so this route does not pay.

### Min-parity in the first theta (composition with brainstorm-critic's MIN_PARITY)
The first theta reads raw message bits (exact inputs), where the noise gain of knots between
integers does not matter. With 3-unit parity for n=7 (this avenue's exact form:
`max(0,18-6s)(-25/36 - 19s/72) + max(0,2s-3)(-47/6 + 4s/3) + max(0,5s-23)(-s/3) + 25/2`)
and brainstorm-critic's 4-unit n=9 entry, theta1 drops from 6074 to 5571 units:
`xs3:lazy4c_middle_mp` = depth 6, dense **70,075,915**, sparse 176,217 (edge-case margin
0.0018, validate_bench bit-equal). Dense -2.6% but sparse +2.9% against lazy4c (the
min-parity values read all n inputs, glu_xor's mostly read none), so it is a trade, not a win.
n=8 and n=10 gain nothing (exhaustive 1-D: 3 resp. 4 units impossible).

### Numeric cross-checks of (a) (vpx.py, 128 restarts, hard-relu residual with LS values)
parity of 7 bits: 4 units exact (0.0), 3 units 0.0088 (an exact 3-unit form exists, found by
the 1-D search); 8 bits: 4 exact, 3 units 0.098; 10 bits: 4 units 0.087 (5 needed, as the
1-D search says). The fitter finds exact solutions when they are easy (parity 6 in 3 units:
0.0), so large residuals are evidence, not proof.

### Last round through shared column parities (checked, no gain)
lazy4c's last theta takes the parity of each count (range 32, 16 units per theta bit). Each
column sum is read by two counts, so its parity (range 14: 7 units) could be shared, with
theta3 = own ^ C_left ^ C_right carried as the free sum E3 = own + C_L + C_R in {0..3} and
the last chi computed exactly from 4-valued codes. Units drop (L5 5682 -> ~3440), but L4 must
then emit the 320 own values as well as the 320 column sums (657 -> 977 features, +1.0M),
and the last chi from 4-valued codes costs several units per digest bit (numeric fits of an
exact chi on codes E in {0..3}, theta = E mod 2: 3 units residual 0.59, 4 units 0.51, far
from exact): no gain.

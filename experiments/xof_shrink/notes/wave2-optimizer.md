# optimizer (wave 2): cost-model-driven layout search, 1-round Keccak XOF (log_w=6, 3 steps)

Architecture unchanged: plain `MLP_SwiGLU`, no residual, no weight tying, attention = identity;
input BOS + 1144 message bits, output BOS + 672 bits. Every number below comes from
`xofbench.py` (harness, 16 random messages) plus `audit/adv_check.py` on BOTH reference sets
(`$S/xof/final/ref777_w6.pt`, 69 messages; `$S/xof/av/combined-1/adv/ref_w6.pt`, 63 messages;
edge cases + random, against the unpatched reference xof): `ok: true`, 0 wrong bits, eager
function equal to the reference. `results.jsonl` holds the harness and audit lines
(`kind: harness` / `kind: audit`, `stage: final_code` for the re-run with the final code);
`python table.py final_code` prints the summary. `runs/final/` holds harness + both audits
for every reported variant with the final code (md5 of the builder files in
`runs/final_code.md5`; `runs/final/regress_*` reproduce the wave-1 frontier exactly).

## Result: new Pareto points at depths 5-8 (depth 4 unchanged)

| depth | variant (`xo.py`) | dense | sparse | vs wave-1 frontier (dense / sparse) |
|---|---|---|---|---|
| 5 | `d5_m3k2_mp_pk` | 82,837,245 | 231,431 | -7.2% / +0.7% |
| 5 | `d5_m3k2_pk` | 84,726,010 | 226,445 | -5.1% / -1.5% (dominates) |
| 6 | `lazy4c_middle_m3_mp_pks` | 62,220,049 | 183,827 | -10.3% / +3.5% |
| 6 | `lazy4c_middle_m3_mp_pk` | 62,961,053 | 179,185 | -9.2% / +0.9% |
| 6 | `lazy4c_middle_m3_pk` | 64,849,818 | 174,199 | -6.5% / -1.9% (dominates) |
| 7 | `split_first_lazy4c_m3_mp_pksud` | 52,960,192 | 133,650 | -16.8% / +4.8% |
| 7 | `split_first_lazy4c_m3_mp_pkud` | 53,701,196 | 129,008 | -15.6% / +1.2% |
| 7 | `split_first_lazy4c_m3_pkud` | 54,448,601 | 127,726 | -14.5% / +0.2% |
| 7 | `split_first_lazy4c_pkud` | 55,156,263 | 126,246 | -13.3% / -1.0% (dominates) |
| 8 | `split_first_middle_m3_mp_pksud` | **48,792,168** | 113,293 | **-20.1%** / +3.8% |
| 8 | `split_first_middle_mp_pkud` | 49,708,462 | 109,285 | -18.6% / +0.2% |
| 8 | `split_first_middle_pkud` | 50,455,867 | **108,003** | -17.4% / -1.0% (dominates) |
| 8 | `split_first_middle_m3_mp_pksud_mc` (flagged) | 48,370,728 | 112,621 | -20.8% / +3.2% |
| 8 | `split_first_middle_pkud_mc` (flagged) | 49,932,667 | 107,683 | -18.3% / -1.3% |

The two `_mc` rows use min-parity on count features (see "Small ones"); they pass both
audits but with margins 0.0079-0.0119 against the 0.02 tolerance (all other rows <= 2.2e-3),
so the recommended depth-8 points are the ones above them.

Pareto set over (depth, dense, sparse) of all verified points (these + wave 1): depth 4
`xc:d4a_m4k3_mp` (wave 1); depth 5 both rows; depth 6 the three rows plus wave 1's
`xs3:lazy4c_middle_m3` (71,257,018 / 172,637, still the sparsest at depth 6: pk costs ~1.5K
sparse because a decoded pair needs 3 wo entries and a 2-weight value); depth 7 all four
rows; depth 8 the three unflagged rows, or, if the flagged ones are admitted, `mp_pkud` and
both `_mc` rows (`pksud_mc` dominates `pksud`, `pkud_mc` dominates `pkud`). Every other
wave-1 frontier point is dominated on all three metrics.

Wave-1 frontier: d5 89,244,445 / 229,869; d6 69,368,253 / 177,623; d7 63,651,004 / 127,470;
d8 61,082,590 / 109,101 (d4 160,358,065 / 425,212 is unchanged: none of the constructions
applies there, see below). At every depth 5-8 one of the new points is smaller on all three
metrics than the wave-1 point ("dominates"). Against the baseline (3,305,445,348 / 2,034,788)
depth 8 is 67.7x dense / 18.0x sparse (`pksud`) and 65.5x / 18.8x (`pkud`, no min-parity).
Suffixes: `mp` min-parity in the first layer (raw bits only, as wave 1), `pk` pass-through
pairs, `s` s_last, `u`/`ud` u pairs (u3 in round 1), `m3` 3-bit digest packing.

The wave-1 floor ("about 50M at depth 8: a state-carrying layer costs >= 3 x 1600^2") is
broken: it assumed 1600-wide boundaries. Depth-8 `pksud` layers (in, hidden, out):
`[1145,1659,641] [641,2474,1465] [1465,1602,1121] [1121,2562,1036] [1036,2957,1676]
[1676,1677,471] [471,2073,375] [375,1415,673]` (4.9, 6.8, 6.5, 8.4, 11.1, 6.4, 2.7, 2.0 M);
wave-1 depth 8: `[1145,2163,1465] [1465,1466,1601] [1601,1602,1921] [1921,3202,1713]
[1713,1714,1713] [1713,1714,545] [545,2146,545] [545,675,673]`.

## Method: an exact cost model, read for marginal costs, then exhaustive search

`repo/experiments/xof_shrink/cost_model.py`. Dense of a SwiGLU layer with i inputs (BOS
included), h hidden units, o outputs is i + h (2 i + o). The model gives (i, h, o) of every
layer kind the builders can emit as a function of the options (round-1 layout, round-2 layout,
min-parity, pk, dedupe, s_last, digest packing m1, u options, the xs4 depth-5 layout) and
matches the harness TO THE PARAMETER on all 24 measured variants (`python cost_model.py
--check`). `enumerate_designs()` searches the whole option space per depth (exhaustive, it is
small) and returns exactly the measured winners above, so within the modelled space these
layouts are optimal. `best_partitions()` enumerates how a column pair's five positions are
grouped at the X -> Y boundary (E singles, u pairs, u1/u3 groups with the constant positions)
and confirms the chosen groupings (depth 8, cost per column pair: round 1 `u3+u` 43.8K vs
47.8K next; `u3+E` 35.6K vs 38.3K; round 2 `u+u+E` 69.0K vs 73.3K). `what_if()` prices
hypothetical constructions (headroom below).

The model prints marginal costs: a boundary feature costs h_L + 2 h_{L+1} + 1 (5K-14K
parameters in the middle rounds; 51K-67K at depth 4), a unit of layer L costs 2 i_L + o_L.
For every wave-1 layout the state-carrying boundaries of the middle rounds were the
expensive part, so the search targeted narrower boundaries at (nearly) equal unit count.
Three exact constructions came out of it:

### 1. Pass-through pairs (`pack_a`, "pk"): -6.4M at every depth 5-8

The X layer of a split theta (E = a + D) only COPIES the chi bits a. Emit two chi bits as
ONE feature f = (c1 + 2 c2)/4 (a free sum of their two chi units) and decode them in X:
  U1 = max(0, p) = p,  U2 = max(0, p - 1)(2 - p/2) = c2,  c1 = U1 - 2 U2   (p = 4f in {0..3}),
two units for two bits, which is what the two copies cost; knots on integers (flat). The
chi layer's output and the X layer's input shrink from 1921 to 1121 features for free
(chi1 -1.3M, X2 -5.1M). Pairs only: for 3-bit p no unit other than the copy lies in
span{1, p, bits of p} (every knot position checked by hand), so triples need >= 5 units.

### 2. u pairs (a shared XOR mask): round 2 -1.5M (depth 8), round 1 -2.9M (depths 7, 8)

The five theta bits of a column pair (x, *, z) share the mask D. For two of them
  (a1 ^ D) + 2 (a2 ^ D) = |u|,   u = (a1 + 2 a2) - 3 D  in [-3, 3]
(D = 0: u = a1 + 2 a2; D = 1: -u = (1 - a1) + 2 (1 - a2)). So X emits ONE feature per pair
(the copy of the packed pair minus 3 D: one unit instead of two copies) and Y recovers both
theta bits with 4 units whose knots sit on lattice points:
  t2 = [|u| >= 2] = max(0, u - 1)(2 - u/2) + max(0, -u - 1)(2 + u/2),
  t1 = |u| - 2 t2 = max(0, u) + max(0, -u) - 2 t2.
Round 2 (depth 8): per column pair 2 u pairs + the y = 0 bit as an E single (a digest bit,
decoded from a pk pair anyway): X2 8 units / 3 outputs instead of 10 / 5, Y2 9 units instead
of 5; X2 + Y2 = 20.98M -> 19.48M. Round 1 (message bits: the pair copy is ONE unit
max(0, a1 + 2 a2) on raw inputs). `upair1d` also folds the constant-own positions of a
round-1 column pair (capacity/suffix lanes, whose theta bit is D itself) into a pair:
  u = 2 (a1 + 2 a2) - 7 D in {0,2,4,6} (D = 0) or {-7,-5,-3,-1} (D = 1)   (injective),
5 lattice-knot units give all three: A = max(0,u), B = max(0,-u), B' = max(0,-u-1),
C = max(0,u-2)(1-u/8), C' = max(0,-u-3)(9/8+u/8); D = B - B', t2 = C + C',
t1 = (A + B')/2 - 2 t2 (all 8 cases checked by hand, and by the audits).
Column pairs with 4 live own bits: `u3 + u` (2 features, 9 units), with 3: `u3 + E`
(2 features, 6 units). X1 + Y1: 14.56M (pk only) -> 11.66M.
u pairs cannot feed the lazy chi of depths 6/7 (its products of u values would have range ~50
and the counts would explode), so there round 2 keeps E values.

### 3. Small ones
- `s_last`: the last theta layer emits per digest bit the one linear form its chi unit reads,
  s = 2a - b + c (224 features instead of the 320 theta bits): -0.47M (d8) to -0.74M (d6/7),
  sparse +1% to +3% (each s reads the parity units of 3 counts).
- `dedupe_y`: the Y layer of the split first round emits the 1464 distinct theta bits only.
- digest packing re-optimized with the model (m1 = 3 dense-best, m1 = 2 sparse-best);
  min-parity (raw bits) is a dense-for-sparse trade (-0.75M dense, +1.3K sparse at d8).
- `_mc` (flagged, noise-sensitive): brainstorm-critic's MIN_PARITY[11] (5 units, knots between
  integers, gates scaled so every integer sits >= 1 from a knot) on the range-11 COUNTS of the
  depth-8 last theta: -421,440 dense exactly as the model's what-if predicted, sparse -0.7K,
  but the slope at the integers (2.85) amplifies input noise: audit margin 0.0087 / 0.0119
  (95%-ones random messages) against 0.002 without it. It passes (ok, 0 wrong bits on both
  sets) at c = 4, q = 8, but it uses 60% of the tolerance, so it is reported separately and
  not recommended for larger circuits. (Wave 1 saw 0.0265 with it on counts at q = 4.)

## Why it is exact

Each construction is an identity on the lattice of exact bit/integer inputs, with knots on
lattice points (silu(0) = 0 at a knot; every other gate value has magnitude >= 1 after the
c*q = 32 scaling, so the silu error is ~e^-32). The u / u3 formulas are identities over all
8 (a1, a2, D) cases, the pk decode is exact on {0..3}. The only knots between integers are
wave 1's min-parity units on raw message bits (exact inputs), and, in the flagged `_mc` rows
only, MIN_PARITY[11] on the last-theta counts. The eager builder functions equal
the reference xof at log_w 2, 3, 4 (8 messages each incl. all-0/all-1, `check_ref_small.py`)
and at log_w 6 (audit `eager_mismatch: []`); the compiled networks pass the harness at
log_w 3 and 4 (64 messages, `runs/smallw/`) and the two audits at log_w 6 (margins <= 2.9e-3
against the 0.02 tolerance; worst messages are dense random ones). Stress run with 256 random
messages at log_w 6 (`runs/stress/`): `pksud` margin 3.7e-3, `pkud` 7.5e-4, `pksud_mc` 0.0125,
0 wrong bits each.

## What did not work (priced with the model or searched exactly)
- 3-unit decode of a u pair (1-D gates on u): no solution (exact search over all unit
  triples with knots on the 1/4 and 1/12 grids, `search/k3exact*.py`); priced at -3.4M.
- A joint decode of a column pair's u pair and E single with 2-D gates on (u, E): needs >= 5
  units, i.e. no saving. Proof sketch: on the row E = 1 both values of D occur, so there the
  units are 1-D units in u; t0 = [E == 1] is 1 on that row and 0 elsewhere, so some unit's
  restriction to the row is a combination of 1 and the others, and the remaining K - 1
  restricted units must decode (t1, t2) from u, which needs 4 (no 3-unit form, above).
  (A numeric search was started and dropped once this argument settled it.)
- Two features (pp, D) instead of u: each theta bit needs >= 2 units (XOR-shaped): break-even.
- Base-3 packed E values for Y: [E1 == 1], [E2 == 1] from E1 + 3 E2 need 7 units per pair.
- Pair triples, u triples (3 bits sharing D: ~10 units), two-mask pairs (~8 units), 4-bit
  |u| encodings of two pairs: all lose (decoding grows ~2^k/2 with k bits per feature).
- Column parities C (3 units per column) instead of D (5 per column pair) in X: Y then needs
  parity of a 0..3 value (2 units per bit) or a 10-point u decode: +5M.
- Window-reduced D (fewer units, E in a wider range): breaks the u decode, widens E.
- AND trick on the last counts at depth 8 (range 10): +0.8M (a chi2 unit costs 3x a theta3 unit).
- Lazy digest-2 values packed in pairs, exact digest-2 in the lazy layer (3-unit lazy +
  base-3 packing), digest m1 = 4 or mixed m1: all within +-0.2M or worse.
- Depth 9: no layer whose split pays; depth 6/7 with a Walsh last round after u pairs:
  62-74M; depth 6 with a split first round needs a Walsh round on range-32 counts (100M).
- Depths 4 and 5: their one-layer rounds read 1600 counts in parities (no pass-through, no
  shared-mask boundary), so only pk applies (d5, via its X2); the d4 carries cost 51-67K per
  feature and are already packed as far as the decoders allow.
- Recompute instead of carry: the digest-1 bits' inputs (theta1 bits) are gone after chi1.

## Headroom (lower-bound reasoning)

- Depth 8, 48.8M. The Y layers now hold 18M: they must emit 1465/1600 theta bits for chi,
  whose units read each bit in a gate, so the Y -> chi boundary cannot be packed, and the u
  decode is at its minimum (4 units, and a joint decode with the E single cannot save one).
  Remaining levers are unit constructions: D in 4 units per column pair (-1.0M; P has range
  10, where 4-unit parity does not exist even with knots between integers), range-11 parity
  in 5 units (-0.4M: realized by the flagged `_mc` rows, at a noise cost). I put the floor of
  exact, noise-robust constructions in this family at about 46-47M. A loose bound if every
  state boundary could be packed 2 bits per feature: about 30M (4 full layers of ~1600 units
  x (2 x 800 + 800) + first/last layers).
- Depth 7, 53.0M: min-parity on the range-32 last-theta counts would give -1.6M.
- Depths 5-6 (82.8M, 62.2M) are bound by the one-layer theta1 on raw bits (20.9M, ~3.8 units
  per bit with min-parity) and depth 4-5 by one-layer rounds on counts (lazy round 90.6M,
  Walsh 22-34M): only unit-level constructions (parity of counts in fewer units, an AND of
  two parities) can move them; the layout search found nothing there.
- Sparse: 108K at depth 8 (`pkud`, no min-parity). A loose floor is ~3-4 nonzeros per
  hidden unit (~16K units) plus norms: 60-70K; the units-per-bit count is what must drop.
- Depth: 4 is the minimum found (depth 3 estimated > 300M by wave 1).

## Composes with
Any cheaper unit construction for the parities (D, theta3, Walsh, lazy round), the
first-round/last-round ideas of other avenues (pk/u only touch the middle boundaries), and any
other layout that has a layer copying bits (pk) or an XOR with a shared mask (u). As a
compiler pass, pk is general: whenever a layer's only use of some input bits is COPY units,
pair them at the producer and decode with (U1, U2).

## Files
- `patch.diff`: `git -C repo diff` incl. new files (`xs3.py` options `pack_a`, `dedupe_y`,
  `s_last`, `upair1`, `upair1c`, `upair1d`, `upair2` and the `MINPAR_COUNTS` switch; `xs4.py`
  option `pack_a`; new `xo.py` (variants) and `cost_model.py`). Old variants are unchanged (regression runs in
  `runs/final/regress_*`).
- `results.jsonl`, `table.py`, `log_result.py`, `runs/` (harness/audit outputs; `runs/final/`
  final-code re-verification; `runs/final_prev_code/` the same before the `_mc` option was
  added (identical numbers); `runs/smallw/` log_w 3/4; `runs/stress/` 256 messages),
  `search/` (exact searches), `check_ref_small.py`, `audit.sh`, `harness.sh`,
  `final_verify.sh`. (The base's own uncommitted docstring edit in `operations.py` is not
  part of this patch.)
- Run: `PYTHONPATH=repo/src:repo/experiments/xof_shrink python repo/experiments/xof_shrink/xofbench.py
  --log-w 6 --depth 3 --variant xo:split_first_middle_m3_mp_pksud --widths`.

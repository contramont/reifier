# depth-low: shallow SwiGLU circuits for 1-round Keccak XOF (log_w=6, 3 steps)

Architecture unchanged: plain `MLP_SwiGLU` (RMSNorm, `wo(silu(wg x) * wv x)`), no residual,
untied, attention = identity, no embedding/readout. Every number below comes from
`xofbench.py` (harness, 16 random messages) plus `audit/adv_check.py` on BOTH reference sets
(`$S/xof/final/ref777_w6.pt`, 69 messages; `$S/xof/av/combined-1/adv/ref_w6.pt`, 63
messages). A row is claimed only with `ok: true`, 0 wrong bits and no eager mismatch on all
three; `results.jsonl` has the raw lines (including the failing ones, marked below).

## Summary

- Depth 4: `d4x` fuses round 1 on the raw message bits (lazy chi1 in {0,1,2}) and turns the
  lazy values into exact chi1 bits with one unit per bit in the next (X) layer; the rest is
  wave 1's depth-5 tail. 135,400,953 dense (-15.6% vs 160,358,065) at 547,787 sparse (+29%).
- Depth 3 (new): three fused rounds, `d3c1_mp17`, 253,823,869 dense / 1,292,483 sparse. The
  plain build is exact but fails float32 on dense messages (glu_xor's ~n^2 cancelling terms
  in 40-bit raw parities); a centred parity construction with the same unit count, used for
  the raw (exact-input) parities only, fixes it.
- Depth 4 at <= 80M is out of reach in this primitive family (floor ~116M: every fused round
  needs a pair parity per chi bit); depth 2 is out (a layer would need chi then theta).

## Results

| depth | variant (`dl3.py`) | dense | sparse | margin harness / ref777 / c1 | vs frontier |
|---|---|---|---|---|---|
| 4 | `d4x_m4k2_mpx` | **135,400,953** | 547,787 | 4.2e-4 / 2.8e-3 / 1.7e-3 | dense -15.6%, sparse +29% vs `xc:d4a_m4k3_mp` 160,358,065 / 425,212 |
| 4 | `d4x_mpx` | 135,500,786 | 545,309 | 4.2e-4 / 2.8e-3 / 1.7e-3 | -15.5% / +28% |
| 4 | `d4x` (no min-parity) | 140,356,754 | 533,861 | 4.0e-4 / 1.1e-3 / 1.4e-3 | -12.5% / +26% |
| 3 | `d3c1_mp17` (centred raw parities, min-parity for odd raw sets of 7-17 bits) | **253,823,869** | **1,292,483** | 4.0e-3 / 6.2e-3 / 9.8e-3 | first depth-3 circuits (no earlier point); dominates d3c1 |
| 3 | `d3c1` (centred raw parities) | 258,773,437 | 1,301,763 | 4.3e-3 / 9.4e-3 / 8.5e-3 | verified, dominated by d3c1_mp17 |
| 3 | `d3c1t` (d3c1 + zero-on-lattice top-end units on even-range count parities) | 265,613,757 | 1,314,147 | - / 9.4e-3 / 8.5e-3 | identical margins to d3c1: no gain, dominated (harness not run) |
| 3 | `d3c` (centred raw AND count parities) | 258,773,437 | 1,306,289 | - / **2.7e-2** / **2.7e-2** | FAILS on sparse messages (onehotL); not claimed |
| 3 | `d3c_mp17` | 253,823,869 | 1,297,009 | - / 1.4e-2 / **2.2e-2** | FAILS on c1 (rand1_0.5); not claimed |
| 3 | `d3` (plain glu_xor) | 258,773,437 | 1,297,295 | 4.3e-4 / **3.1e-2** / **3.1e-2** | FAILS the audit (float32 rounding, see below); not claimed |
| 3 | `d3_mpx` | 253,014,805 | 1,295,939 | 1.3e-3 / **4.2e-2** / **4.4e-2** | FAILS the audit; not claimed |

Extra check (not required): harness with 64 random messages, `d4x_m4k2_mpx` margin 4.8e-4
and `d3c1_mp17` margin 8.1e-3 (0 wrong bits in both).

Widths (in, hidden, out):
- `d4x_m4k2_mpx`: `[1145,17783,1601] [1601,4803,1657] [1657,5338,489] [489,13667,673]`
  (digest 1 packed 4 exact bits per feature, digest 2 two lazy values per feature);
  dense by layer 69,194,798 / 23,339,378 / 20,302,071 / 22,564,706, sparse 370,307 /
  65,557 / 39,783 / 72,140 (`layerstats.py`).
- `d4x_mpx` (m1=3, k2=2): `[1145,17783,1601] [1601,4803,1676] [1676,5357,508]
  [508,13141,673]`, dense by layer 69.2M / 23.4M / 20.7M / 22.2M; `d4x`: first layer 19031
  units (74.0M). Harness-only packings: m4k3 135,475,730, m3k3 135,644,716.
- `d3c1_mp17`: `[1145,24113,1676] [1676,32078,471] [471,22015,673]`, dense 95.6M / 122.6M /
  35.6M; `d3c1` (same units as `d3`): layer 1 has 25361 units (100.6M).

New Pareto points: the three d4x rows (lower dense than any earlier depth-4 circuit; wave
1's `d4a_m4k3(_mp)` stay on the front through lower sparse), and `d3c1_mp17`, the first depth-3 circuit (`d3c1` is verified too but dominated).
Depth 2 is not reachable (lower-bound section).

## d4x: fuse round 1 on raw bits, recover EXACT chi1 bits one layer later

Wave 1's depth-4 circuit fuses round 2 (theta2 + chi2 in one layer on counts of range 10:
15 units per state bit on a 1657-wide input = 90.6M, 56% of it). Its depth-5 circuit
(89.2M) is `theta1 | chi1 | X2 | lazy chi2 with column parities | Walsh round 3`. d4x is
that depth-5 circuit with theta1 and chi1 fused into ONE layer on raw message bits:

- **L1 (round 1, lazy, on raw bits):** theta1 of every position is `flip ^ parity(R)` for a
  set R of raw message bits (capacity and suffix bits are constants folded into flips). For
  each state position q with post-pi row inputs a, b, c the layer emits (as o'/2)
  `o'(q) = t_a + (t_c - t_b + (t_b ^ t_c)) / 2  =  t_a + (1 - t_b) t_c`,
  where t_a, t_b, t_c are parities of raw sets (1464 distinct theta1 sets of 6-9 bits:
  6072 glu_xor units, shared by the 3 chi bits that read each) and `t_b ^ t_c =
  parity(B xor C)` (symmetric difference, 13-17 raw bits: 12956 units for the 1600
  adjacent pairs). o' is in {0,1,2} and o' = chi1 (mod 2). 1600 features, as wide as the
  state.
- **L2 (theta2 as an X layer, from the lazy values):** `A = max(0, o')(2 - o') = [o' == 1]`
  is the EXACT chi1 bit (one unit); `D = parity(P)`, P = the sum of the column pair's 10 o'
  inputs (a linear form of the inputs, range 20; 10 glu_xor units shared by the 5 bits of
  the column). `E2 = A + D` with iota/column flags folded (emitted as E2/2), so
  theta2 = [E2 == 1] exactly as in wave 1's X layer. Digest 1 = the exact A units of row 0,
  packed binary (free sums).
- **L3, L4:** wave 1's d5 tail unchanged (`lazy_chi_cols`: lazy chi2 with exact column
  parities, theta3 counts <= 13; `walsh_last`: round 3 in one layer, 53 units per digest
  bit; digest decoders).

The point is that a lazy value in {0,1,2} becomes an exact bit with ONE unit in the next
layer while its column sums stay free linear features, so the X layer costs about what
wave 1's does (4803 vs 3203 units; D has range 20 instead of 10). Saving d5's layer this
way costs +46M (fused L1 69.2M against d5's theta1 + chi1 28.7M, X layer 23.3M against
17.7M); saving it by fusing round 2 on counts (wave-1 d4) costs +71M. (Wave-1 synthesis item 6 listed a "hybrid depth-4
layout" of this shape, estimated at -8%; measured -12.5% plain, -15.6% with mpx/m4k2.)

`mpx` = min-parity on the raw parities: unit-synthesis' 3-unit MINPAR7 (knots between
integers) extended to every odd n >= 7 by integer-knot units `-4 max(0, s - 7 - 2j)` (the
piece (s-6)^2 on [5,7] continues as (s-8)^2, (s-10)^2, ...): (n-1)/2 units instead of
(n+1)/2, exact in rational arithmetic for n = 7..59 (`mpcheck.py`). Inputs are exact
message bits, so the slope at lattice points does not matter for n <= 17 (d4x); L1 17783
instead of 19031 units (-4.9M dense, +11K sparse).

**Why d4x is exact:** t_x are parities of exact bits (glu_xor or min-parity units, exact at
integer points); (1 - t_b) t_c = (t_c - t_b + t_b ^ t_c)/2 is an identity on bits; so o' in
{0,1,2} and o' = chi1 mod 2; [o' == 1] = chi1; P = sum of 10 o' = XOR of the column pair's
10 chi1 bits mod 2; D = parity(P) exactly (glu_xor, knots on integers); E2 in {0,1,2},
[E2 == 1] = own ^ C[x-1] ^ C[x+1] = theta2; the rest is wave 1's verified d5 tail.

## d3: three fused rounds

Depth 3 must fuse every round (lower bounds), so the layout is forced:
- **L1:** round 1 as in d4x, but emitting the theta2 COUNTS S2 of every theta2 bit (own o'
  + two columns). Each column contributes `parity(sum of its 5 a-terms)` (ONE raw parity of
  35-41 bits, 6330 units for the 320 columns, shared by the 10 counts that read it) plus its
  5 exact ANDs: <= 6 instead of <= 10; with the row bound (own t_x + the left neighbour's
  AND (1 - t_x) t_{x+1} <= 1) S2 <= 13 instead of 21. Digest 1 = lazy o' of row 0, packed
  base 3, decoded in L3 as [o' == 1].
- **L2:** round 2 fused on S2 (xs4 `lazy_round`): singles 7 units, pairs 13 units per chi2
  bit; emits the theta3 counts (<= 21) and digest 2 (lazy, base 3).
- **L3:** Walsh round 3 on counts <= 21 (85 units per digest bit) + decoders.

**Why d3 is exact:** each chi1 bit is congruent to its lazy value t_a + AND (mod 2); a
column's value parity(sum of its a-terms) + sum of its ANDs is an integer congruent to the
XOR of its 5 chi1 bits; so S2 (own o' + two column values) is an integer in [0, 13]
congruent to theta2 (flags folded), and every parity downstream is exact on integers. The
counts' bound 13 is the row bound above (checked by the eager function against the
reference on the audit messages as well).
**Float32 failure of the plain build and the fix.** `d3` is exact (float64 forward: max
error 1.2e-6 on all 63 c1 messages) but float32 gives 0.031 on `onecold0` (digest 3;
digest 2 0.015) and up to 0.009 on 95%-dense random messages (`diag.py`,
`runs/diag_d3.txt`). The cause is cancellation: glu_xor for a count s in [0, n] sums terms
of size ~n^2 (s(2 - s) against 4 sum relu(s - 2j)), and L1's column parities have n = 35-41
(terms ~1500); their float32 rounding (~1e-4 each, the RMSNorm scale is not a power of 2
unless every input is 1) lands in the counts S2 and is passed on by L2 and L3 (with gain 2
at the ends of even ranges, and x9 in the base-3 digest packing). `centered(n)`: the same
parity with its pieces centred at c = 2 floor(n/4), `(s + 1)(2c + 3 - s) - (c+1)(c+3)` plus
4 max(0, s - k) to the right and 4 max(0, k - s) to the left (knots on even integers, flat
at the lattice points), has exactly ceil(n/2) units as glu_xor (checked exactly for
n = 6..69, `ctrcheck.py`) but terms of size ~(n/2)^2 for every s, where glu_xor's are ~s^2:
the worst case (dense inputs, s near n) drops from 1520 to 484 at n = 40 and from 3843 to
1024 at n = 63, while sparse inputs (s near 0, tiny terms in glu_xor) get the same ~(n/2)^2
that random inputs already see in glu_xor (the constant is one shared BOS unit per layer).
Measured (edge margins in `runs/adv*_d3c*.json`):
- `d3c1_mp17`: as d3c1, with min-parity (knots between integers, fine on exact inputs)
  for the odd raw sets of 7-17 bits and centred pieces for the rest: -4.9M dense, and
  smaller errors on sparse messages (onehotL 0.0036); worst 0.0098 (rand49_0.5). Passes.
- `d3c1`, centred pieces for the raw parities of layer 1 only (the count parities of L2/L3
  keep glu_xor): `onecold0` 0.031 -> 0.0015, and the sparse cases stay small (onehotL
  0.0085, zeros 0.0039); worst 0.0094 (a 95%-dense random message). Passes both audits.
- `d3c`, centred count parities as well: dense cases fixed (onecold0 0.0013) but sparse
  cases fail (onehotL 0.027, zeros 0.010): for a count near 0, glu_xor's terms are tiny and
  its knot at 0 averages the slope to 1, while the centred form has ~(n/2)^2 terms and slope
  2 there. `d3c_mp17` (plus min-parity for raw sets of 7-17 bits) fails the same way (0.022).
So the rule is: centre the parities whose inputs are exact (raw bits), keep glu_xor where
the inputs carry errors. Steepness is not the lever here: with the same weights (c=4, q=8)
the float64 forward is exact to 1.2e-6, so the failure is float32 rounding of large
cancelling terms, and sharper silu would not reduce it.

## Lower bounds (why depth 4 cannot reach 80M here, and depth 2 is out)

Primitive family of all wave-1/wave-2 circuits: features are exact bits, counts (sums of
lazy values) or E codes; parity of a count of range R costs ceil(R/2) integer-knot units
(minimal with knots on integers; knots between integers save one unit for odd R, safe only
on exact raw bits); chi on exact bits is one unit.
1. **No layer can hold a theta after a chi.** A theta2 bit is the XOR of 11 chi1 bits =
   linear ^ inner product of 11 pairs of theta1 bits. Unit-synthesis measured XOR of 2 chi
   bits > 3 units and XOR of 3 chi bits far from exact with 7 units; inner product needs
   2^Omega(n) units in depth-2 threshold circuits (Hajnal et al.). So every layer is theta,
   chi or a fused theta+chi, and the 6 half-rounds in d layers need 6 - d fused layers.
   Depth 2 puts 3 consecutive half-rounds in one layer, which always contains chi then
   theta: no construction is known and the smallest cases already blow up (2 chi bits:
   no exact form with <= 3 units; 3 chi bits: no exact fit with 7; a theta bit needs 11),
   so depth 2 is out in practice.
   Depth 3 fuses all three rounds; depth 4 exactly two: `th1|ch1|th2ch2|th3ch3` (wave-1
   d4), `th1ch1|th2|ch2|th3ch3` (d4x) or `th1ch1|th2ch2|th3|ch3`.
2. **A fused round needs a pair parity per chi bit.** Lazy chi needs (1 - t_b) t_c exactly
   (the Fourier support of the AND contains chi_{b xor c}), i.e. the parity over the union
   of b's and c's inputs: R units on counts of range R (R >= 10 for 11 exact chi bits even
   with the AND trick), >= 6 units on 13-17 raw bits, plus the singles. Measured: 15 units
   per state bit on counts (wave-1 d4 L3), 11.1 per bit on raw bits (d4x_mpx L1). A window
   of width 2 does not help 1-D parities (the triangle wave 0,1,2,1,0,... also needs one
   unit per 2 points).
3. Per layout, at those unit counts:
   - fused round 2 on counts: >= 24,000 units on a >= 1601-wide input: >= 84M alone;
   - fused round 1 on raw bits: >= 17,780 units x (2 x 1145 + 1601) = 69.2M (d4x_mpx's
     L1 is at exactly this count), plus an X layer >= 23M (1600 A + 3200 D units on 1601
     inputs), a lazy chi2 >= ~11M and a Walsh round 3 >= ~13M (exact theta3 counts):
     >= ~116M for d4x's layout, >= ~140M for wave-1's, more for the third.
   So in this family depth 4 >= ~116M; <= 80M would need a fused round at < ~4 units per
   state bit, i.e. an exact AND of two parities far cheaper than a pair parity.
   Depth 3 >= ~93M (the 17.8K round-1 units plus ~6.1K column units that bring the
   counts down to <= 13) + ~113M (fused round 2 on counts <= 13: 20 units per bit) + ~25M
   (Walsh on counts <= 21) = ~231M; d3c1_mp17 is at 253.8M (the rest is digest carries,
   the 1676-wide L2 input and unit-count rounding).

## Headroom (my estimate)
- depth 4 dense: 135.4M now. The tail (L3 + L4, wave 1's d5 tail, 43M) is ~2x its floor
  but was tuned in wave 1; D's range 20 in L2 could drop to 12 only with exact column
  parities in L1, which cost more than they save. ~125M is plausible with more tuning;
  below ~116M needs a new AND primitive.
- The open question that decides depth 4: the exact AND (1 - p_B) p_C of two parities of
  disjoint raw sets (|B|, |C| ~ 8). Writing it as (1 + chi_B - chi_C - chi_B chi_C)/4, the
  singles are free (shared), so the units must carry chi_{B xor C}; restricting C so that
  p_C is fixed shows any exact form needs >= ~max(|B|, |C|)/2 ~ 4 units, while the pair
  parity uses (|B| + |C|)/2 ~ 8. A 4-5 unit form would take ~6K units out of d4x's L1
  (-24M, d4x ~112M) and ~5K out of d3's. Exact search on the count grid (`andsearch2.py`:
  units max(0, a b + e c + g)(u + v b + w c) with b, c the counts of the two sets, singles and
  a constant free): |B| = |C| = 3, 2 units (|a|,|e| <= 3, |g| <= 9): none (the pair parity
  needs 3); |B| = |C| = 4, 2 units (<= 3, <= 12): none; 3 units (<= 2, <= 8, 4.2M triples):
  none (the pair parity needs 4). So within count-symmetric gates the pair parity is minimal
  at these sizes, which supports the ~116M floor; asymmetric gates remain unsearched.
- depth 4 sparse: layer 1 holds 370,307 of the 547,787 nonzeros (its pair parities' gates
  read 13-17 message bits each; min-parity values read all of them); the other three layers
  hold 65,557 + 39,783 + 72,140. Wave-1 d4a (425K) stays the sparse-optimal depth-4 point.
- depth 3: 253.8M built vs ~231M floor in this family; the fused round 2 (L2, 123M) is at
  its unit floor for counts <= 13, so more needs a cheaper AND of parities (next item).
- depth 2: no.

## Tooling changes (speed only; compiled weights identical)
With the stock code one eager evaluation of the depth-3 layer 1 (1600 numeric nodes of
~220 units over ~180 raw inputs each) took minutes on the heavily loaded machine.
- `neurons/core.py` `glu`: dot products over the nonzero weights only
  (`compress`/`filter`, same terms in the same order; zero terms add nothing).
- `compile/tree.py` `_gate_origin.fold`: skip the per-weight fold when every input is a
  traced parent (nothing to fold, identical result).
- `tensors/matrices.py` `layer_to_units`: skip zero weights (they were filtered later).
- Builder in `experiments/xof_shrink/depth_low/lib/python/` (`dl3.py`; renamed copies
  `xs_u.py`, `xs3_u.py`, `decode_u.py`, `shared_theta_u.py` of the wave-1 helpers): the
  tracer does not record calls made from files under a `/lib/python` path, so helper calls
  do not become call-tree blocks (the glu gates still do); layer-1 nodes are built once and
  reused across evaluations (`cached_node`).
Checked: identical widths, dense, sparse and margin to the plain builder at log_w=2
(d3: 938,979 / 64,673 / 1.34e-4); repo test suite (all but hash_long_test and legacy_tests): 57 passed.

## What did not work / was not pursued (and why)
- min-parity in d3's layer 1 (`d3_mpx`): the column parities have n = 35-41, where the
  non-dyadic MINPAR7 coefficients times ~n^2 terms make float32 errors reach 0.044 on
  dense messages (fails the audit). Fine for n <= 17 (d4x).
- Exact chi1 in the fused round 1 (Walsh on raw bits): +3 parities per bit (~18 units) to
  cut the next count range by one; never pays.
- Column parities for d4x's D (range 20 -> 12): +6330 L1 units and +320 L1 outputs (+33M)
  to save 1280 L2 units (-6M).
- Column parities of round 2 inside a fused round (smaller theta3 counts for Walsh):
  parity of 5 counts of range 13 = 33 units per column, +40M for -12M.
- A lazy (window-2) AND of two raw parities cheaper than the pair parity: products of
  counts are congruent but have ranges ~100; windowed 1-D parities need as many units.
- E' = o' + D (skipping d4x's A units): the 4-valued codes make lazy chi2 and its column
  parities cost ~+3K units in L3 (+12M) for -7.8M in L2.
- Min-parity on counts: saves one unit only for odd ranges (~2%), with slope at lattice
  points.
- Depth 5 from the fused round 1 + wave-1's depth-6 tail: ~116M > 89.2M (d5).

## Reproduce
```bash
S=<scratchpad>; D=$S/xof2/av/depth-low; R=$D/repo; E=$R/experiments/xof_shrink
PP=$R/src:$D/lib/python:$D:$E   # dl3.py and its helper copies live in $D/lib/python
PYTHONPATH=$PP $S/venv/bin/python $E/xofbench.py --log-w 6 --depth 3 --variant dl3:d4x_m4k2_mpx --widths
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP $S/venv/bin/python $E/audit/adv_check.py $S/xof/final/ref777_w6.pt dl3:d4x_m4k2_mpx
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP $S/venv/bin/python $E/audit/adv_check.py $S/xof/av/combined-1/adv/ref_w6.pt dl3:d4x_m4k2_mpx
```
(`runall.sh <module:fn> <tag>` runs all three in parallel into `runs/`; `collect.py` builds
results.jsonl; in the patch the files are under `experiments/xof_shrink/depth_low/`.)

# first-last (wave 2): the first and last rounds of 1-round Keccak XOF, and the state packing they led to

All numbers: `xofbench.py --log-w 6 --depth 3` (harness, 16 random messages, `ok: true`) plus
`audit/adv_check.py` against the unpatched reference xof on BOTH reference sets
(`ref777_w6.pt`, 69 messages; `combined-1/adv/ref_w6.pt`, 63 messages): `ok: true`,
`wrong_bits: 0`, no eager mismatch. Raw lines: `results.jsonl` (kind `harness` / `audit`);
harness JSON in `runs/`, audit JSON in `audits/`.


## Results (log_w=6, 3 XOF steps; every row: harness ok + both audits ok, wrong_bits 0)

New Pareto points (depth / dense / sparse), against the wave-1 frontier at the same depth:

| depth | variant | options | dense | sparse | vs frontier dense / sparse | audit margin (worst of 2 sets) |
|---|---|---|---|---|---|---|
| 4 | `fl4:d4a_m4k3_mp_mpl` | xc d4a + min-parity in the Walsh layer | 159,651,569 | 426,108 | -0.4% / +0.2% | 8.1e-04 |
| 5 | `fl4:d5_col` | pack_col, mp_last (Walsh) | 80,799,133 | 233,229 | -9.5% / +1.5% | 1.7e-03 |
| 5 | `fl4:d5_best` | pack_a, mp_last | 82,080,573 | 232,327 | -8.0% / +1.1% | 1.1e-03 |
| 5 | `fl4:d5_m3k2_mp_pk` | pack_a | 82,837,245 | 231,431 | -7.2% / +0.7% | 1.1e-03 |
| 6 | `fl:d6_g` | pack_col, d2p5, gate_out, m3, mp | 60,200,529 | 188,873 | -13.2% / +6.3% | 1.3e-03 |
| 6 | `fl:d6_col` | pack_col, d2p5, m3, mp | 61,027,773 | 184,231 | -12.0% / +3.7% | 1.3e-03 |
| 6 | `fl:d6_col_nop5` | pack_col, m3, mp | 61,679,613 | 180,087 | -11.1% / +1.4% | 9.4e-04 |
| 6 | `fl:d6_col_m2` | pack_col, m2, mp | 62,387,275 | 178,681 | -10.1% / +0.6% | 4.1e-04 |
| 6 | `fl:d6_col_sp` | pack_col, m3, no mp | 63,568,378 | 175,101 | -8.4% / -1.4% | 6.6e-04 |
| 6 | `fl:d6_col_sp_m2` | pack_col, m2, no mp | 64,276,040 | 173,695 | -7.3% / -2.2% | 3.3e-04 |
| 6 | `fl:d6_pk_sp_m2` | pack_a, m2, no mp | 65,557,480 | 172,719 | -5.5% / -2.8% | 4.0e-04 |
| 7 | `fl:d7_g` | pack_col, dedupe_y, pack_d1, d2p5, gate_out, m3, mp | 53,032,664 | 138,321 | -16.7% / +8.5% | 5.4e-03 |
| 7 | `fl:d7_col` | pack_col, dedupe_y, pack_d1, d2p5, m3, mp | 53,859,908 | 133,679 | -15.4% / +4.9% | 5.4e-03 |
| 7 | `fl:d7_g_m2` | pack_col, dedupe_y, pack_d1, gate_out, m2, mp | 54,546,005 | 132,771 | -14.3% / +4.2% | 1.1e-03 |
| 7 | `fl:d7_col_p5_m2` | pack_col, dedupe_y, pack_d1, d2p5, m2, mp | 54,642,162 | 132,273 | -14.2% / +3.8% | 5.4e-03 |
| 7 | `fl:d7_col_m2` | pack_col, dedupe_y, pack_d1, m2, mp | 55,219,410 | 128,129 | -13.2% / +0.5% | 1.1e-03 |
| 7 | `fl:d7_col_sp_m2` | pack_col, dedupe_y, pack_d1, m2, no mp | 56,136,135 | 126,974 | -11.8% / -0.4% | 6.8e-04 |
| 7 | `fl:d7_sp_m2` | pack_a, dedupe_y, pack_d1, m2, no mp | 57,417,575 | 125,998 | -9.8% / -1.2% | 3.7e-04 |
| 8 | `fl:d8_g` | pack_col, dedupe_y, pack_d1, mp_last, gate_out, m3 + m2 (m_last), mp | 50,538,922 | 113,194 | -17.3% / +3.8% | 7.1e-03 |
| 8 | `fl:d8_col_ml2` | pack_col, dedupe_y, pack_d1, mp_last, m3 + m2 (m_last), mp | 50,909,974 | 112,424 | -16.7% / +3.0% | 6.6e-03 |
| 8 | `fl:d8_col_m2` | pack_col, dedupe_y, pack_d1, mp_last, m2, mp | 51,422,054 | 111,166 | -15.8% / +1.9% | 6.6e-03 |
| 8 | `fl:d8_g_sp` | pack_col, dedupe_y, pack_d1, gate_out, m2, no mp | 52,527,806 | 111,133 | -14.0% / +1.9% | 6.8e-04 |
| 8 | `fl:d8_col_sp2` | pack_col, dedupe_y, pack_d1, m2, no mp | 52,861,979 | 110,011 | -13.5% / +0.8% | 6.8e-04 |
| 8 | `fl:d8_sp` | pack_a, dedupe_y, pack_d1, m2, no mp | 54,143,099 | 109,035 | -11.4% / -0.1% | 5.7e-04 |
| 8 | `fl:d8_dd` | dedupe_y only (= xs3:split_first_middle_mp + the fix) | 60,447,334 | 108,829 | -1.0% / -0.2% | 2.8e-04 |
| 8 | `fl:d8_sp_dd` | dedupe_y, no mp | 61,404,859 | 107,547 | +0.5% / -1.4% | 1.3e-04 |

Frontier (wave 1): d4 160,358,065 / 425,212; d5 89,244,445 / 229,869; d6 69,368,253 / 177,623;
d7 63,651,004 / 127,470; d8 61,082,590 / 109,101. Points that beat the frontier on BOTH
metrics at the same depth: `fl:d8_sp` (-11.4% / -0.1%), `fl:d8_dd`, `fl:d7_sp_m2`
(-9.8% / -1.2%), `fl:d7_col_sp_m2`, `fl:d6_col_sp` (-8.4% / -1.4%), `fl:d6_col_sp_m2`,
`fl:d6_pk_sp_m2`. Against the main baseline (20 / 3,305,445,348 / 2,034,788):
`fl:d8_g` is 2.50x / 65.4x / 18.0x, `fl:d7_g` 2.86x / 62.3x / 14.7x, `fl:d6_g`
3.33x / 54.9x / 10.8x, `fl4:d5_col` 4.00x / 40.9x / 8.7x.

Options: m2/m3 = digest bits per carried feature; mp = wave 1's min-parity in the first theta
(raw bits); no mp = glu_xor there (fewer nonzeros). Per-layer widths are in `runs/*.json`
(`widths`); every measured variant (with its audit status) is in `table.md`.

Depth 8 `fl:d8_g` layers (in, hidden, out): `[1145,2163,1305] [1305,1466,1465]
[1465,1602,961] [961,3202,1676] [1676,1677,1676] [1676,1677,508] [508,1790,412] [412,1045,673]`
= X1 (E = a + D, pure D values in pairs) | Y1 | chi1 -> pairs + column sums | X2 (unpack + D)
| Y2 | chi2 -> 320 counts | last theta (5-unit min-parity) -> 224 chi gates | last chi +
digest decoders. Dense by layer: 7.78, 5.98, 6.23, 11.52, 8.43, 6.47, 2.56, 1.56 (M).

## 1. Main result: the chi layer before an X layer emits the state packed (pack_a, pack_col)

The X layer of a split theta (`E = a + D`, wave 1's theta-structure / xof-structure) is the
widest layer of every depth-5..8 design (17.7M dense, 1921 inputs = 1600 chi bits + 320
column-pair counts + BOS). Its 3202 units read the 1600 state bits `a` ONLY through one-unit
copies `max(0, 1) * a`: the X layer consumes the state LINEARLY. A copy costs the same as an
exact unpack, so the state does not have to arrive one bit per feature:

* **pairs (pack_a).** Two bits packed as `p = a0 + 2 a1` (a free sum of the two chi units that
  make them, emitted as p/4) come back exactly with two units, knots on integers (flat at the
  lattice, like glu_xor):

      a1 = max(0, p - 1) * (2 - p/2)        # 0, 0, 1, 1 at p = 0, 1, 2, 3
      a0 = max(0, p) - 2 a1                 # 0, 1, 0, 1

  `E = a + D` stays a free sum of these units and D's parity units, and the digest of the step
  (packed m3 from the recovered bits) is still free. 1600 bit features -> 800 pair features;
  the X layer keeps 3202 units. X-layer input 1921 -> 1121.
* **pairs + column sums (pack_col).** D needs the column-pair count P = C[x-1][z] +
  C[x+1][z+1] as an affine form of the X layer's inputs, and wave 1 emitted the 320 P counts on
  top of the 1600 bits. Per column (5 bits) emit instead: two pairs (y0, y1), (y2, y3) and the
  column sum C = sum_y a_y (count, C/8). Then `a4 = C - (a0 + a1) - (a2 + a3)` is a free sum
  (`a0 + a1 = max(0,p) - max(0,p-1)(2-p/2)`, plus one pass unit `max(0, 1) * C`), and D's five
  glu_xor units read `C_left + C_right` (two features, range 10) instead of P. 3 features per
  column instead of 6 (5 bits + one P count per column) or 3.5 (pairs + P): X-layer input
  1921 -> **961**, still 3202 units (5 per column for the bits, 5 per column pair for D).

Cost change (depth 8): X layer `[1921,3202,1713]` 17.79M -> `[961,3202,1676]` 11.52M; chi layer
`[1601,1602,1921]` 8.21M -> `[1465,1602,961]` 6.23M. About -8M dense at every depth 5..8, for
+1..2.5K sparse (a recovered bit reads two units in wo, a4 five).

Why it is exact: p and C are exact integers (sums of exact chi units, which are 0/1 up to float
rounding); the unpack units have their knots on integers, so at every lattice point silu is 0
(gate 0) or within e^-32 of relu (|gate| >= 1); a4 is an exact integer combination. E, and
everything downstream, is the same number as before. Checked by the harness and both audits.

Why 3 features per column is the minimum for this layer: the X layer must hold the 5 bits of a
column as 5 linearly independent functions (5 units, the copy count), and D's gate must be an
affine form of the inputs, so C must be affine in the features; a single bit cannot be paired
across columns (then C is not affine), three bits in one feature cannot be unpacked by 3
units (each unit would have to be affine in the bits; decode.py's minimum is 9), and two
features for 5 bits + C would need a 3-bit feature. So 2 pairs + C.

Why only the X layer: everywhere else a state bit is read by units that need it as a single
affine form (chi reads 2a - b + c; Y reads [E == 1] of a 3-valued E; the lazy chi's product
units read Eb + Ec and 1 - Eb). With one unit per output bit in these forms each bit needs its
own affine form, i.e. one feature per bit (not a proof for arbitrary unit sets, but every
packed alternative we tried costs more units than it saves). Checked the alternative for the Y
layer (search/ypair.py): a pair sharing D, (a0, a1, D) -> (a0^D, a1^D) from ONE feature
p = al a0 + be a1 + ga D (integer al, be, ga in [-4, 4], knots on a quarter grid) has no exact
2-unit decoder and no 3-unit decoder whose units are shared by both outputs (62K exact 3-knot
fits, none rank-1 per unit); 4 units per pair = 2 per bit does not pay.

## 2. Smaller first-round items

* **dedupe_y (a waste, not a trick).** Round 1's Y layer (d7/d8) created one theta node per
  position although 136 positions share their E feature with another (the two zero lanes of
  columns 3 and 4, suffix positions): 1601 outputs for 1465 distinct values. One node per
  distinct E: -0.64M dense, -272 sparse, same circuit. (`fl:d8_dd` 60,447,334 / 108,829 beats
  the depth-8 frontier on both metrics by itself.)
* **pack_d1.** The theta bits of the 7 zero lanes (and the 8 suffix positions) are the D values
  themselves (bits). X1 emits them as 160 pairs instead of 320 features, and Y1 unpacks them
  (2 units per pair = the 2 [E == 1] units it had): X1 output 1465 -> 1305, -0.8M at d7/d8.
* **Constants.** Capacity and suffix are already folded by wave 1 (theta1 reads 7-9 live bits;
  the 1464 distinct theta1 values). Nothing else constant survives theta1: every chi1 input is
  message-dependent (post-pi column 4 is made of D values only, but that does not reduce units).

## 3. Last-round items

* **mp_last: min-parity for odd tops in the last theta / Walsh layer.** brainstorm-critic's
  exact 4-unit parity on [0, 9] (two knots on integers, two in gaps 0.378 from the lattice,
  effective slope at the lattice 2 as glu_xor) extends to any odd top n >= 9 with
  `-4 max(0, s - k)` for k = 9, 11, ..., n - 2 (each shifts the last parabola by 2):
  (n - 1)/2 units instead of (n + 1)/2 (checked exactly for n = 9..63). Used where the count top
  is odd: d8's last theta (top 11: 6 -> 5 units per bit, -0.52M), d5's and d4's Walsh layer
  (singles 13/21 and triples 39/63; pairs have even tops). Effect on margins: d8 1.1e-3 harness,
  6.6e-3 audit (still 3x inside 0.02). The margins are float32 rounding, not the constructions:
  on the 69 audit messages (search_fl/diag64.py), float32 vs float64 margins are
  2.8e-4 / 2.7e-5 for `fl:d8_dd` (no packing), 1.8e-3 / 1.2e-4 for `fl:d8_col_ml2_nompl`
  (packing; worst output a step-1 digest bit, which goes through the unpack, the C - 4 bits
  difference and the m3 decoder) and 6.6e-3 / 5.1e-4 for `fl:d8_col_ml2` (plus mp_last; worst
  output a step-3 digest bit). A sharper silu (q = 16) leaves the harness margin at 1.1e-3. Without mp_last (`fl:d8_g_nompl`, 50,995,882 / 113,546, +0.46M against `fl:d8_g`,
  audited on both sets; also `fl:d8_col_ml2_nompl` 51,397,654 / 112,424) the margins are
  2.7e-4 (harness) and 1.8e-3 (audit); prefer it if float32 headroom matters more than 1% dense.
* **d2p5 (lazy designs d6/d7).** The lazy chi layer emitted the 224 step-2 digest values
  o in [0, 4] one per feature into the theta3 layer (5645 units: each input costs 11.3K dense).
  Pack two per feature, `(o0 + 5 o1)/32` (one pass unit per pair; the product units are shared
  with the counts), and decode `parity(o0) + 2 parity(o1)` exactly in the theta3 layer
  (12 units, integer knots, decode.py's DP): -0.65M, +4K sparse.
* **gate_out: the last theta layer emits the chi gates, not the theta bits.** The last chi
  unit of a digest bit reads its three theta bits only through its gate
  `G = 2 t_a - t_b + t_c` (value `(3 - G)/2`; iota flips t_a). G is a free sum of the last
  theta layer's parity units (plus its constant unit), so that layer emits the 224 G values
  (G/4, integers in [-1, 3]) instead of the 320 theta bits, and the last layer's chi unit is
  `max(0, 4g)(1.5 - 2g)` on one feature. Exact (same gate value, same unit). 96 fewer features
  between the last two layers: -0.37M dense at depth 8 (+0.8K sparse: a G row sums the parity
  units of three theta bits), -0.83M at depth 6/7 (+4.6K sparse, 16 parity units per theta
  bit there). This only works where the consumers are fewer than the producers (224 digest chi
  bits on 320 theta bits); in rounds 1-2 the 1600 chi units would need 1600 G features, no
  fewer than the theta bits.
* **m_last (depth 8): pack each digest by the layers it crosses.** Digest 1 crosses Y2, chi2
  and the last theta (three carries: 3 bits per feature pays for its 9-unit decoder); digest 2
  is born in the chi2 layer and crosses only the last theta (one carry: pairs, 3-unit decoder,
  are cheaper). m3 + m2 instead of m3 + m3: -0.21M dense AND -1.5K sparse (`fl:d8_col_ml2`).
* **Tried, no gain** (details in "Failed ideas"): a one-layer last round from the step-2 state
  (Walsh on range-11 counts: +13M at d7/d8), digest carrying at higher density or with a
  two-stage decode (< 0.3M), the last theta through shared column parities (wave 1: no gain),
  carrying D2 instead of digest 1, a one-unit double-AND (mod 2) for the lazy chi layer.


## 4. Failed ideas (measured or searched) and why

* **Packing the Y layer's input** (X -> Y interface, 1676 features at depth 8, 11M dense).
  Y needs `[E == 1]` for 1600 three-valued E; one unit per bit needs each E as an affine form
  of the inputs, so one feature per bit. Pairs cost more: a pair sharing D,
  (a0, a1, D) -> (a0 ^ D, a1 ^ D) from one feature `p = al a0 + be a1 + ga D`
  (search/ypair.py: integer al, be, ga in [-4, 4], knots on a quarter grid): no exact 2-unit
  decoder at all; 62,112 exact 3-knot fits, none with each unit shared (rank 1) by both
  outputs; 4 units per pair = 2 per bit loses (Y is as wide as X's output). Ternary E pairs
  `E0 + 3 E1` need about 7 units. A same-column pair with a separate D feature costs 4 units
  per pair (u1, u2, f*D, hi*D), again 2 per bit.
* **One-unit double AND for the lazy chi layer (d6/d7).** Mod 2, the lazy chi of E-coded bits
  is `Ea + Ec + Eb Ec`, so a column's 5 product terms could share units if one unit (plus the
  count's free pass term) gave `Q(b1,c1) + Q(b2,c2) (mod 2)`, Q = [Eb != 1][Ec == 1], in a
  window of width W: 4 prod units -> 2 per column, about -2.5M (W = 4) to -3.5M (W = 3).
  search/qip2.py (exhaustive over integer gates |w| <= 2, |b| <= 6, v and pass solved exactly
  by pivot enumeration; sanity-checked: it finds the known single-AND form): none for W = 2,
  3, 4; the faster search_fl/qip2f.py (QR pivoting) extends it to |w| <= 3, |b| <= 9: none for
  W = 2 and W = 4 (so none for W = 3). The exact-bit version (IP2) exists, but E = 2 must
  behave as 0 mod 2 and breaks it. (A 2-unit form of width 2 for two AND terms would cut the
  count range 32 -> 24, about -1.9M at d6/d7; not searched: the gate-pair space is too large
  for the time left.)
* **no_p2** (D reads the 10 chi bits directly instead of a count feature): -2.56M dense,
  +14K sparse (`fl:d8_dd_np` 57,885,094 / 122,589). Superseded by pack_col, which removes the
  count features' cost without reading 10 inputs per unit.
* **A one-layer last round from the step-2 state (Walsh).** From exact range-11 counts:
  singles 5 (min-parity) + pairs 11 + 11 + triple 16 = 38 units per digest bit + 1600 singles
  in a 471-wide layer: about 18M against 4.8M for theta3 + chi3, so a d7 of about 63M (worse
  than the lazy d7). From lazy range-32 counts it is 30K units. The lazy one-layer middle round
  of xs4 (d4) is the only one-layer round that pays, and only at depth 4.
* **Round 1 in two layers without theta1 direct**: X1 + exact chi from E (>= 4 units per bit,
  no integer form known) is 34.5M against 27.2M for theta1 direct + chi1; X1 + lazy chi1 with
  X2 recovering `[o' == 1]` (same unit as the unpack) doubles D2's count range (5 -> 10 units
  per column pair) and triples the chi1 units: +18M.
* **Packing the first layer's input**: an extra layer that packs the message into pairs and
  column sums (for D1) saves 1.1M in X1 and costs 2.8M (depth 9, no gain).
* **Digest carrying at higher density**: m4 (23-unit decoders) or two-stage decoding
  (m6 split into two m3 in the theta3 layer) change the depth-7/8 totals by -0.3M..+0.3M; m3
  for digest 1 (three crossings) and m2/m3 for digest 2 are within 0.1M of the optimum.
  Carrying D2 instead of digest 1 (digest1 = theta2 ^ D2) is the same number of bits.
* **d2p5 alternatives**: exact step-2 digest bits from a 4-unit exact chi in the lazy layer
  (+0.85M), exact theta2 of the 5 diagonal lanes then exact chi (+1.6M), 3 lazy values per
  feature (decoder of 125 values), width-2 lazy values in base 3 (+4K per bit): all worse.
* **AND_CHI2 in the depth-8 chi2 layer** (count range 11 -> 10 for the last theta): the AND
  units sit in a 1677-unit layer (4K each), the saved parity units in a 471-wide one (1.6K).

## 5. Headroom (my view, with the lower-bound reasoning)

Cost model: a layer costs `n + 2 n h + m h` (inputs n, hidden h, outputs m). A layer whose units
need one input bit as a single affine form (Y: [E == 1]; chi: 2a - b + c; lazy chi: Eb + Ec)
needs one input feature per bit, and one unit per output bit needs h >= bits. Only layers that
consume the state linearly (X, carries) can take it packed, at 3 features per 5 bits.

* **Depth 8 (50.5M):** the interfaces are input 1145, E1 1305, theta1 1465, packed chi1 961,
  E2 1676, theta2 1676, counts 508, chi gates 412; hidden 2163, 1466, 1602, 3202, 1677, 1677,
  1790, 1045. Every hidden width is at 1 unit per bit except X (2 per bit: 1 for the bit, 1 for the
  shared D) and the first layer's D1 units. With these unit counts the floor of the split
  layout is about 49-50M; the remaining slack is the digest carries (about 4M, near their
  decode-cost optimum) and D1/D2 (glu_xor/min-parity counts are minimal for their ranges:
  range 10 -> 5 units, 4 impossible). Going below ~48M needs fewer than one feature per state
  bit on an interface that is consumed non-linearly (E -> Y, theta -> chi); every packed form
  we tried there costs more units than it saves (section 4).
* **Depth 7 (53.0M):** lazy chi + theta3 on range-32 counts (about 20M) is the price of the missing
  Y layer. Levers: fewer product units or a narrower count (a one-unit double AND mod 2: not
  found; a two-unit one of width 2: -1.9M, not searched). Fusing X2 and Y2 into one theta2
  layer on the packed state (wave 1's shared-column trick with per-pair units on (f, T),
  f = a0 + 2a1) does not pay: the C-derived fifth bit of a column is not affine, so the fused
  layer needs 4 features per column (1281 inputs), and to beat the lazy depth 7 it would need
  under 2.7 units per bit including the shared ones, while the per-bit version already needs 3
  per bit plus shared units. Floor about 50-52M.
* **Depth 6 (60.2M):** the first layer (theta1 on raw bits, 20.9M, 3.8 units per theta bit)
  dominates. Shared-column theta on exact bits is worth <= 4% (wave 1), 3-unit parity of
  8 bits is impossible; floor about 58-60M.
* **Depth 5 (80.8M), depth 4 (159.7M):** the Walsh last layer and the one-layer lazy round
  dominate; the last round is at 38-46 units per digest bit (Fourier support {a}, {a,b},
  {a,c}, {a,b,c} of chi forces three multi-count parities per digest bit). Depth 4's middle
  round (90M) is not a first/last-round problem.
* **Sparse:** my points reach 107.5K at depth 8 (no min-parity, no packing, dedupe only) and
  126.0K at depth 7; packing costs +1..5K sparse (the recovered bits read two to five units).

## 6. Checks beyond the two audits

* `validate_bench.py` (harness weights bit-equal to `Compiler().get_mlp_from_tree` at
  log_w 0-2, forward difference, verify): `fl:d8_g`, `fl:d7_g`, `fl:d8_col`, `fl:d7_col`,
  `fl:d6_col`, `fl4:d5_col` all `weights_equal=True`, `ok` (runs/val_*.txt).
* `fl:d8_col`, `fl:d7_col`, `fl:d6_col` at log_w 3, 4, 5 and `fl:d8_g`, `fl:d7_g`, `fl:d6_g` at
  log_w 4, 5, with 64 random messages each: all `ok`, margins 3.3e-4..2.1e-3 (runs/lw*.json).
* The repo's src/ is unchanged (all new code is in experiments/xof_shrink/fl.py, fl4.py and
  search_fl/), so the test suite is unaffected.

## 7. Files, how to run

* `repo/experiments/xof_shrink/fl.py`: copy of xs3.py with the options (`fl_vs_xs3.diff` shows
  only the changes): `pack_a`, `pack_col` (chi layer -> X layer packing), `dedupe_y`,
  `pack_d1` (round 1), `d2p5` (lazy step-2 digest), `mp_last` (odd-top min-parity in the last
  theta), `gate_out` (last theta emits the chi gates), `m_last` (bits per feature of the last
  digest), `no_p2` (superseded); variants at the end of the file (`fl:d8_g`, `fl:d7_g`, ...).
* `repo/experiments/xof_shrink/fl4.py`: copy of xs4.py (`fl4_vs_xs4.diff`) with `pack_a`,
  `pack_col` for the depth-5 layout and `MPL` (min-parity in the Walsh layer):
  `fl4:d5_col`, `fl4:d5_best`, `fl4:d4a_m4k3_mp_mpl`.
* `repo/experiments/xof_shrink/search_fl/ypair.py` (Y-layer pair decoders),
  `search_fl/qip2.py`, `qip2f.py` (one-unit mod-2 double AND on E values; qip2f needs scipy,
  e.g. wave 1's `$S/xof/av/unit-synthesis/svenv`), `diag64.py` (float32 vs float64 forward).
* `patch.diff` = `git -C repo diff` (new files only; the base is the wave-2 base commit).
* `results.jsonl`, `runs/`, `audits/`; `table.py` builds `table.md` (all 45 measured
  variants) from them; `collect.py` builds results.jsonl.

Run (S = the scratchpad, A = this directory):

    PYTHONPATH=$A/repo/src:$A/repo/experiments/xof_shrink $S/venv/bin/python \
      $A/repo/experiments/xof_shrink/xofbench.py --log-w 6 --depth 3 --variant fl:d8_g --widths
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<same> $S/venv/bin/python \
      $A/repo/experiments/xof_shrink/audit/adv_check.py $S/xof/final/ref777_w6.pt fl:d8_g
    (and $S/xof/av/combined-1/adv/ref_w6.pt)

# round-structure (wave 2): how rounds map to layers at depths 5-8

Target: 1-round Keccak XOF, `log_w=6`, 3 steps, compiled to plain `MLP_SwiGLU` (no
residual, attention = identity, untied layers). Every number below comes from
`xofbench.py` (harness, 16 random messages) **and** `audit/adv_check.py` on both reference
sets (`xof/final/ref777_w6.pt`, 69 msgs; `xof/av/combined-1/adv/ref_w6.pt`, 63 msgs):
`ok: true`, `wrong_bits 0`, no eager mismatch, on both. Raw lines: `results.jsonl`
(harness JSON in `runs/*.hb.json`, audits in `runs/*.adv_*.json`, per-layer tables in
`runs/lstats_*.txt`). `validate_bench.py` (harness weights bit-equal to the dense
`Compiler` pipeline at log_w 0-2, verify ok) passes for the headline variants
(`runs/*.val.txt`). No compiler change: the patch adds two builder modules (`rs.py`, a fork
of `xs3.py`; `rs4.py`, a fork of `xs4.py`) and a few variant lines in `xc.py`.

## Headline (best dense per depth, and best sparse at 7-8)

| depth | variant | dense | sparse | vs wave-1 frontier row (dense / sparse) |
|---|---|---|---|---|
| 5 | `xc:d5_m3k2_mp_xp` | 82,837,245 | 231,431 | -7.2% / +0.7% |
| 5 | `rs4:d5_m3k2_xp` | 84,726,010 | 226,445 | -5.1% / -1.5% (dominates) |
| 6 | `rs:lazy4c_middle_m3_mp_xpg3` | 62,220,049 | 183,827 | -10.3% / +3.5% |
| 6 | `rs:lazy4c_middle_xp` | 65,557,480 | 172,719 | -5.5% / -2.8% (dominates) |
| 7 | `rs:split_first_lazy4c_m3_mp_xpydg3_x1p` | 53,367,704 | 133,338 | -16.2% / +4.6% |
| 7 | `rs:split_first_lazy4c_m3_xpyd_x1p` | 54,937,713 | 127,414 | -13.7% / -0.0% (dominates) |
| 7 | `rs:split_first_lazy_xpg_x1s` | 63,754,034 | 115,944 | +0.2% / -9.0% |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1p_x2pc_d2p` | 49,816,267 | 110,995 | -18.4% / +1.7% |
| 8 | `rs:split_first_middle_xpyd_x1p_x2pc` | 51,355,587 | 107,259 | -15.9% / -1.7% (dominates) |
| 8 | `rs:split_first_middle_xpg_x12sc` | 57,023,849 | 97,335 | -6.6% / -10.8% |

The full audited list with Pareto flags (within this avenue) is at the end of this file (`table.md`). Against the baseline
(depth 20, 3,305,445,348 / 2,034,788): depth 8 is 66x dense (49.8M) or 20.9x sparse (97.3K),
depth 7 62x dense, depth 6 53x dense, depth 5 40x dense.

## 1. The X layer reads the state two bits per feature (`xpairs`, -6.4M at depths 5-8)

Wave 1's floor argument was "a layer that carries the 1600-bit state costs at least
3 x 1600^2 = 7.7M dense" (hidden ~ in ~ out ~ 1600). It assumes every layer reads one
feature per state bit. That is needed for layers whose units compute a nonlinear function of
*individual* bits, but not for a layer whose units only *copy* a bit.

The middle round of the depth 5-8 layouts starts with an X layer (theta-structure's split
theta): E = a + D per state bit, i.e. one copy unit of a (a = the previous chi bit) plus the
5 units of D = parity(P) that the 5 bits of a column share (P = column-pair count, a free
sum of the chi layer). A copy can decode a packed input at no extra unit cost:
- the chi layer before X emits its 1600 bits **two per feature**, p = a0 + 2 a1 (as p/4, a
  free sum of its two chi units: wo only);
- X decodes a pair with two units, u0 = relu(p), u1 = relu(p - 1)(4 - p):
  a0 = u0 - u1, a1 = u1 / 2, exact on p in {0,1,2,3} (integer knots, like glu_xor);
- X keeps one unit per bit, but its input shrinks from 1921 to 1121 features (800 pairs +
  320 counts + BOS), and the chi layer's output from 1921 to 1121.
X2 [1921, 3203, 1676] 17.7M -> [1121, 3203, 1676] 12.6M; chi1 7.8M -> 6.5M: **-6.4M dense
at every depth with an X layer**, +1.6K sparse. Pairs are (x, y, z), (x, y, z+1), z even, so
the digest-1 bits pair with each other and the digest pair X carries on is u0/4 itself.
Triples lose (a 3-bit decoder needs ~8 units per 3 bits, not 3).

## 2. The Y layer can read pairs too, if both bits share D (`x1pairs`, `x2pairs`)

Y computes theta = a ^ D (one unit per bit). A packed pair p = a1 + 2 a2 with an
*arbitrary* partner cannot be decoded and xored in 2 units (`search/pairD.py`,
`pairD2.py`: no single gated unit, gates on the half grid |w| <= 8, has any non-constant
function of span{t1, t2, 1} in its range). But if both bits are in the **same column**,
they share D, and D is a feature Y needs anyway (round 1: the zero lanes' theta bits are D;
middle round: X emits D per column). With sigma = 1 - 2D:
    t1 = a1 ^ D = D + a1 sigma,  t2 = a2 ^ D = D + a2 sigma,  a1 sigma + 2 a2 sigma = p sigma,
    A = p sigma = relu(p)(1 - 2D)                      (one unit: gate p, value 1 - 2D)
    a2 sigma = u + w,  u = relu(p - 1 - 4D)(4 - p)/2,  w = relu(p + 4D - 5)(p - 4)/2
    t2 = D + u + w,    t1 = D + A - 2(u + w)
**3 units per pair plus the column's D copy** (shared by the column's pairs and zero lanes;
`search/check_pair_decode.py` checks all 8 points exactly; integer gates, exactly 0 at
the knots). Two units plus a free D are impossible: p sigma is the only one-unit direction
of span{a1 sigma, a2 sigma} modulo span{D, 1} (a2 sigma alone would need one affine value
whose slope in p has opposite signs on the two D slices); `search/pairD_freeD2.py` confirms
it for the encodings p = a1 + k a2, k in {2, 3, 4, -2, -3, 1/2, 3/2, -1/2, -3/2, 1/3, 2/3}
(integer gates in [-6, 6], half-integer offsets): one direction each, never two. Triples
are worse ([p >= 4] on 4 active points needs a 2-unit ramp per D slice).
So Y pays 1.5 units per bit instead of 1, but X passes a pair with ONE unit instead of two
copies and emits half as many features:
- **x1pairs** (round 1, depth 7/8): X1 passes the live message bits of each column two per
  feature ((a1 + 2 a2)/4, a pass unit on two raw bits) next to the column's D (504 pairs,
  136 singles). X1 [1145, 2163, 1465] 8.12M -> [1145, 1659, 961] 5.39M; Y1 [1465, 1466,
  1465] 6.44M -> [961, 1970, 1465] 6.67M: **-2.5M** at depth 7 and 8.
- **x2pairs** (middle round, depth 8): the chi layer pairs y = 1, 2 and y = 3, 4 of each
  column; the y = 0 bits stay z-pairs, decoded in X2 and emitted as a. X2 passes the column
  pairs with one unit and emits D; Y2 decodes. X2 3202 -> 2562 units, Y2 1677 -> 2638:
  -0.2M; with **x2carry** (Y2 packs digest 1 from the a features it already reads, so X2
  emits no digest features) **-0.8M**.

## 3. Smaller changes

- **ydedup**: Y1 emitted 1600 theta bits, only 1465 distinct (zero lanes share D). -0.6M.
- **g features (gfeat, gmid)**: the chi unit max(0, 2a - b + c)(3 - 2a + b - c)/2 depends
  only on g = 2 t_a - t_b + t_c. A Y layer can emit g per chi bit (a free sum of three of
  its units): the chi unit reads 1 feature instead of 3. -3.2K sparse per Y/chi pair;
  +0.6M dense after Y1 (1600 g's vs 1465 distinct t's), free after Y2 (gmid). Not with
  x2pairs (the decode units then feed 6 g's each: +2.9K).
- **g3**: the same for the last theta layer: it emits the 224 chi gates of the digest (iota
  folded) instead of the 320 theta bits: -0.74M at depth 6/7, -0.3M at 8, +2..5K sparse.
- **x1sep / x2sep**: X emits a and D as separate features instead of E = a + D, and Y
  computes t = a ^ D = max(0, a + D)(2 - a - D) on two features. X1: same width (1144 +
  320 = the 1464 distinct E's), -2.6K sparse for free. X2: +320 features: -3.4K sparse
  for +2.1M dense (sparse points only).
- m3 digest packing and min-parity in the first theta (mp) are kept as options; `m2`
  packs the digest made by the chi layer before the last theta (depth 8) separately: two bits
  per feature there (its decoders sit in the narrow last layer) and three for digest 1: -0.1M.

## 4. Why it is exact

- Pair features: p/4 with p an exact sum of two exact 0/1 units; X decoders (u0, u1) and Y
  decoders (A, u, w, D copy) have integer gates that are exactly 0 at their knots, so
  silu(0) = 0 there and the lattice values are exact (as for glu_xor). Checked exactly in
  `search/check_pair_decode.py`.
- g features: g/4 is a linear combination of exact Y units (or of glu_xor parity units on
  exact counts, g3), with iota and negation flags folded as constants; CHI on g is the
  same function (checked in the same script).
- All features stay <= 1 in magnitude (p/4 <= 3/4, g/4 in [-1/4, 3/4], D and a bits), so
  RMSNorm's scale stays >= 1 as in wave 1.
- Every listed variant: harness ok and both audits ok with 0 wrong bits and no eager
  mismatch. Worst audit margins: 2.3e-3 for the x1pairs variants (dense 95%-ones random
  messages; 1.0e-3 without x1pairs), 10x inside the 0.02 tolerance. 128-message harness
  runs of the headline dense variants: `runs/*.hb128.json`.

## 5. What did not work / was ruled out

- Y reading a pair with an arbitrary partner (different D): no 2-unit decoder; with two D's
  the decode costs 5 units per pair (more than the X side saves). Same for pairing the 136
  round-1 singles across columns.
- Packed E values (E1 + 3 E2): decoding [E1 == 1], [E2 == 1] costs 7 units per pair.
- Lazy chi (depth 6/7) cannot read pairs: its product units need E_b, E_c as linear forms.
- Pairs into the chi layers (chi needs 1600 independent gates g) or the theta3 layer (the
  320 needed theta3 bits are one per column, nothing to pair).
- Lazy digest-2 values packed two per feature (base 5): 19 decoder units per pair.
- Two-stage digest decode (6 bits per feature through the wide layers, split in the last
  theta layer): the staircase split costs ~15 units per feature, a wash.
- Column parities C instead of D in X (3 units per column instead of 5 per column pair):
  E = a + C_L + C_R is 4-valued and Y needs 2 units per bit (+7.6M at depth 8).
- Range reduction of the P counts in the chi layer (AND units, inner-product units): more
  chi units than the D units they save.
- Round 1 in three layers (column-count pass layer, X1', Y1), a third-round X/Y split (each
  needed theta3 bit has its own column pair), depth 9: no gain.
- Depth 6 with the cheap first round (X1/Y1 = 12.1M instead of 20.9M) needs an extra layer;
  the only place to take it back is a one-layer last round (Walsh), which costs far more
  (76M at depth 6). Depth 5 keeps d5's tail (range-13 counts are the floor of a one-layer
  lazy chi: exact linear column parity + 5 exact Q's per column).
- lazy4c counts really reach 0..32 (32 is attainable), so no theta3 unit can be dropped.

## 6. Headroom (my view)

Dense, per depth (current -> my estimate of the floor of this layout family):
- **Depth 8: 49.8M -> ~45M.** Layers: X1 5.4 | Y1 6.7 | chi1 6.5 | X2 9.0 | Y2 11.2 |
  chi2 6.4 | theta3 2.7 | chi3 2.0 (`runs/lstats_d8best.txt`). Lower bounds: each chi layer
  needs 1600 units and ~1465-1600 input features (1600 independent chi gates, one unit per
  exact chi bit), ~12M for the two; D needs 5 units per column pair (parity of a range-10
  count, minimal with integer knots); Y needs >= 1 unit per bit; the digests cost ~3M of
  carries. What is left above ~45M is Y2's 1.6 units per bit and the digest carries.
- **Depth 7: 53.4M -> ~50M** (lazy4c + theta3 are 20.7M; the theta3 parity of range-32
  counts and the 224 lazy digest-2 values it must make exact are the fixed part).
- **Depth 6: 62.2M -> ~58M.** The first theta on raw bits is 20.9M (34%) and needs the
  extra layer of the split to shrink; nothing in this avenue beats it in one layer.
- **Depth 5: 82.8M -> ~78M.** The one-layer last round (Walsh, 22M) on range-13 counts is the
  floor of a one-layer lazy chi; the digest inputs of that layer alone cost ~5M.
Beyond that, cheaper parities (fewer units per theta bit, e.g. min-parity on counts if it
can be made flat at the lattice) or architecture changes (residual/tying) are needed;
wave 1's ~50M floor for depth 8 is broken by packing, but the next 10% needs new units.

Sparse: depth 8 97.3K, depth 7 115.9K. Per-bit nonzero counts are already near their minimum
(chi unit on g: 3 + wo; Y unit: 3-5 + wo; D units on a count: ~15 per column pair); the X1
layer (~19.5K, D units reading 7-8 raw bits) is the largest single item. I expect at most
~5-10% more from this family.

## Files

- `repo/experiments/xof_shrink/rs.py`: builder with the options xpairs, ydedup, gfeat,
  gmid, g3, x1sep, x2sep, x2carry, x1pairs, x2pairs (see `build`'s docstring) and all
  variants; `rs4.py`: the depth-4/5 builder with xpairs; `xc.py`: min-parity wrappers.
- `search/`: pair-decode searches and the exact check; `hb.sh`, `audit.sh`, `collect.py`,
  `table.py`, `impl/lstats.py`: harness, audit, results and per-layer tools.
- Run: `PYTHONPATH=repo/src:repo/experiments/xof_shrink $S/venv/bin/python
  repo/experiments/xof_shrink/xofbench.py --log-w 6 --depth 3 --variant
  rs:split_first_middle_m3_mp_xpydg3_x1p_x2pc_gm --widths`

## Appendix: all audited variants of this avenue (regenerate with `python3 table.py`)

| depth | variant | dense | sparse | vs frontier row (dense / sparse) | audit margin (worst of 2 sets) | Pareto |
|---|---|---|---|---|---|---|
| 5 | `xc:d5_m3k2_mp_xp` | 82,837,245 | 231,431 | -7.2% / +0.7% | 1.1e-03 | yes |
| 5 | `rs4:d5_m3k2_xp` | 84,726,010 | 226,445 | -5.1% / -1.5% | 7.0e-04 | yes |
| 6 | `rs:lazy4c_middle_m3_mp_xpg3` | 62,220,049 | 183,827 | -10.3% / +3.5% | 1.1e-03 | yes |
| 6 | `rs:lazy4c_middle_m3_mp_xp` | 62,961,053 | 179,185 | -9.2% / +0.9% | 1.1e-03 | yes |
| 6 | `rs:lazy4c_middle_xp` | 65,557,480 | 172,719 | -5.5% / -2.8% | 4.0e-04 | yes |
| 6 | `rs:lazy_middle_xp` | 70,397,992 | 168,175 | +1.5% / -5.3% | 6.4e-04 | yes |
| 7 | `rs:split_first_lazy4c_m3_mp_xpydg3_x1p` | 53,367,704 | 133,338 | -16.2% / +4.6% | 2.3e-03 | yes |
| 7 | `rs:split_first_lazy4c_m3_mp_xpyd_x1p` | 54,108,708 | 128,696 | -15.0% / +1.0% | 2.3e-03 | yes |
| 7 | `rs:split_first_lazy4c_m3_xpyd_x1p` | 54,937,713 | 127,414 | -13.7% / -0.0% | 1.7e-03 | yes |
| 7 | `rs:split_first_lazy4c_m3_mp_xpydg3_x1s` | 55,867,544 | 130,818 | -12.2% / +2.6% | 1.0e-03 |  |
| 7 | `rs:split_first_lazy4c_m3_mp_xpyd` | 56,608,548 | 128,760 | -11.1% / +1.0% | 1.4e-03 |  |
| 7 | `rs:split_first_lazy4c_m3_xpg` | 58,205,860 | 124,552 | -8.6% / -2.3% | 4.0e-04 | yes |
| 7 | `rs:split_first_lazy4c_xpg` | 58,913,522 | 123,072 | -7.4% / -3.5% | 3.5e-04 |  |
| 7 | `rs:split_first_lazy4c_xpg_x1s` | 58,913,522 | 120,488 | -7.4% / -5.5% | 3.2e-04 | yes |
| 7 | `rs:split_first_lazy_xpg_x1s` | 63,754,034 | 115,944 | +0.2% / -9.0% | 4.7e-04 | yes |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1p_x2pc_d2p` | 49,816,267 | 110,995 | -18.4% / +1.7% | 2.3e-03 | yes |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1p_x2pc` | 49,920,385 | 112,549 | -18.3% / +3.2% | 2.3e-03 |  |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1p_x2pc_gm` | 49,924,623 | 115,751 | -18.3% / +6.1% | 2.3e-03 |  |
| 8 | `rs:split_first_middle_mp_xpydg3_x1p_x2pc` | 50,192,409 | 109,663 | -17.8% / +0.5% | 9.0e-04 | yes |
| 8 | `rs:split_first_middle_mp_xpydg3_x1p_x2pc_gm` | 50,196,684 | 112,865 | -17.8% / +3.5% | 9.0e-04 |  |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1p_x2p_gm` | 50,512,548 | 115,863 | -17.3% / +6.2% | 2.3e-03 |  |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1p_gm` | 50,701,668 | 111,063 | -17.0% / +1.8% | 2.3e-03 |  |
| 8 | `rs:split_first_middle_xpyd_x1p_x2pc` | 51,355,587 | 107,259 | -15.9% / -1.7% | 8.9e-04 | yes |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1s` | 53,196,480 | 111,741 | -12.9% / +2.4% | 1.0e-03 |  |
| 8 | `rs:split_first_middle_m3_mp_xpydg3_x1s_gm` | 53,201,508 | 108,543 | -12.9% / -0.5% | 1.0e-03 |  |
| 8 | `rs:split_first_middle_mp_xpydg3_x1s` | 53,707,561 | 108,855 | -12.1% / -0.2% | 8.9e-04 |  |
| 8 | `rs:split_first_middle_mp_xpydg3_x1s_gm` | 53,712,700 | 105,657 | -12.1% / -3.2% | 8.9e-04 | yes |
| 8 | `rs:split_first_middle_mp_xpyd` | 54,041,734 | 110,317 | -11.5% / +1.1% | 1.0e-03 |  |
| 8 | `rs:split_first_middle_mp_xpg` | 54,686,660 | 104,193 | -10.5% / -4.5% | 1.0e-03 |  |
| 8 | `rs:split_first_middle_mp_xpg_x1s` | 54,686,660 | 101,609 | -10.5% / -6.9% | 8.8e-04 | yes |
| 8 | `rs:split_first_middle_xpg` | 55,644,185 | 102,911 | -8.9% / -5.7% | 7.2e-04 |  |
| 8 | `rs:split_first_middle_xpg_x1s` | 55,644,185 | 100,327 | -8.9% / -8.0% | 5.0e-04 | yes |
| 8 | `rs:split_first_middle_mp_xpg_x12sc` | 56,066,324 | 98,617 | -8.2% / -9.6% | 8.8e-04 | yes |
| 8 | `rs:split_first_middle_xpg_x12sc` | 57,023,849 | 97,335 | -6.6% / -10.8% | 5.0e-04 | yes |
| 8 | `rs:split_first_middle_xpg_x12s` | 57,766,745 | 97,447 | -5.4% / -10.7% | 5.0e-04 |  |

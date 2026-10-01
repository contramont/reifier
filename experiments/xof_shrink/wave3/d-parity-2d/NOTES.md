# d-parity-2d (wave 3): the column-pair parity D in 4 units, and every parity of range n ≡ 2 (mod 4) in n/2 - 1

Item: NEXT.md 2. In X2, D = parity(C_L + C_R), with C_L, C_R in [0, 5], costs 5 glu_xor units per
column: 1600 of X2's 3203 units. The question was whether 4 units can do it, using 2-D gates that
read C_L and C_R separately.

## Result

1. **D takes 4 units, and 4 is optimal.**
   - 2-D gates are not needed. The gates stay on s = C_L + C_R.
   - The knots must sit at **irrational** points (odd ± 2√2). Every earlier search used integer or
     small-rational knots and so could not find it:
     - wave 2's "range 10 needs 5, 4 is impossible";
     - `search/d2d_milp.py`;
     - my exhaustive integer-direction search (below).
   - 3 units is impossible for **any** real 2-D gates of this unit form (lemma N1).
2. **The same "parabola bump" form does every count parity with n ≡ 2 (mod 4) in n/2 - 1 units**
   (glu_xor: n/2). Where it applies here:
   - X2's D (n = 10);
   - the lazy chi's column parities at d4/d5 (n = 10);
   - the Walsh pair parities at d4/d5 (n = 26);
   - the last theta after a fold into [0, 10] at d8;
   - C5's theta3 (n = 34);
   - raw-bit parities of 6, 10 or 14 bits.
3. **The price is margin.**
   - glu_xor is flat at every lattice point, because silu averages its integer kinks. A bump has
     slope ±2 at the even points, so it passes on 2x the error of its input counts.
   - Most stacks absorb that.
   - The ones that do not all stack a bump on TH1S (round-1 column-shared theta) plus a second
     error source: TH1S + LZ5 at d6, TH1S + the n = 34 bump at d6, and TH1S at d5 (X2 bump). The
     other failure is the X2 bump next to d8's MPC parity layer.

**New verified bests.** Each passed the harness and adv_check on ref777 and ref63: ok, 0 wrong
bits, empty eager mismatch.

| depth | combined-2 best dense | new best dense (variant) | change | sparse | worst margin |
|---|---|---|---|---|---|
| 4 | 134,661,305 | **132,023,684** (`dl3:d4x_m4k2_mpx_wmpc_bumpr`) | -1.96% | 567,567 | 0.0031 |
| 5 | 79,361,129 | **77,408,324** (`xc:d5_m4k2_mp_cp_mpc_th_l4w`) | -2.46% | 250,690 | 0.0088 |
| 6 | 57,250,043 | **56,870,395** (`c2:d6_m4s4_mp_cp_z11_th_sl_d4`) | -0.66% | 188,813 (was 192,041: this point dominates) | 0.0066 |
| 7 | 49,375,781 | **48,207,053** (`c2:d7_m4s4_mp_cp_u1_z11_lz5_sl_d4_r`) | -2.37% | 137,526 | 0.0079 |
| 8 | 45,071,004 | **43,986,756** (`c2:d8_rp_m4s4_mp_cp_u1_sl_g_d4_w10_r`) | -2.41% | 116,923 (was 115,841) | 0.0172 |
| 8 (robust) | | 44,358,916 (`c2:d8_rp_m4s4_mp_cp_u1_sl_g_d4_r`) | -1.58% | 117,421 | 0.0051 |

The d8 margin:
- The d8 dense-best point (`_w10`: RP fold into [0, 10] plus a 4-unit bump parity) passes at
  0.0172. It dominates the other d8 points of this avenue on dense and sparse, but it has the
  thinnest margin.
- The robust d8 point beats the old best with a lower margin (0.0051 against 0.0086). Its
  256-message stress run gives 0.0043.
- Bumps add about 1K nonzeros per wide layer, because their values read the count feature and
  glu_xor's do not. So the sparse count rises slightly wherever a bump replaces glu_xor.

## The construction (`minpar_counts.bump_units`, `search/bump.py`)

A bump centred on an odd c is

  P_c(s) = 8 - (s - c)^2 = (c + r - s)(s - c + r),   r = 2√2.

With the constant -7:
- -7 + P_c is 1 at c and 0 at c ± 1;
- two bumps 4 apart meet at c + 2 with 4 + 4 - 7 = 1.

Take bumps at c = 1, 5, ..., n - 1. A gated unit max(0, g)·v with g, v affine is 0 where g = 0, so it
cuts a bump **continuously** at one of its roots c ± r. For n = 10:

```
D(s) = -7 + max(0, 1+r - s)(s - 1+r)        # bump 1, cut at 3.83 (the domain ends at 0)
          + max(0, s - 5+r)(5+r - s)        # bump 5, cut at 2.17 ...
          + max(0, s - 5-r)(s - 5+r)        #   ... its tail beyond 7.83 cancelled
          + max(0, s - 9+r)(9+r - s)        # bump 9, cut at 6.17 (the domain ends at 10)
```

Why it is exact:
- the pieces are {0,1,2}, {3}, {4,5,6}, {7} and {8,9,10}, and the error at every integer is about
  1e-14;
- end bumps take 1 unit and inner bumps 2, so the count is n/2 - 1;
- the constant goes to the layer's shared BOS unit: X2 hidden 3203 → 2883, exactly -320.

The parameters are forced. Write a bump as h - a(s - c)^2 and require 1 at c, 0 at c ± 1, and 1 at
the meeting point. That gives a = 1 and h = 8, so the knots are the irrational roots c ± 2√2. No
rational knot grid contains them.

For odd n the form ties MIN_PARITY. For n ≡ 0 (mod 4) it ties glu_xor.

**Slopes and margin (the check the item asked for).**
- Every knot is 3 - 2√2 = 0.1716 from the nearest integer. Gates are scaled by 1/(3 - 2√2) = 5.83,
  so |gate| ≥ 1 at every integer and the silu error is at the e^-32 level.
- The slope is F'(s) = 0 at odd s and ±2 at even s, for every n.
- The largest unit value at a lattice point is 17 for n = 10, 433 for n = 26 and 833 for n = 34.
  Inner bumps cancel their own tails.
- With 1-D gates this cannot be avoided at 4 units. There are 4 knots off the lattice and 11
  points, so some piece holds 3 lattice points, and the quadratic through (0, 1, 0) has slopes ±2.
  Integer-knot forms do not exist (exhaustive). glu_xor is flat because its kinks sit on the
  integers.
- A 2-D continuation from the bump (`search/grad2d.py`) finds no exact form with smaller
  gradients. The bump is locally rigid.

## All runs (harness + both audits; `results.jsonl`, table generated by `gen_table.py`)

See `results_table.md` for the full list: 50-odd variants, with margins, verified or failed, and
the Pareto flag against combined-2's table plus these points. The switches:
- `_d4`: X2 D by bumps;
- `_d4l`: plus the lazy column parities (xs4);
- `_l4`: lazy column parities only;
- `_l4w`: lazy + Walsh pairs;
- `_bump`: d5: X2 + lazy + Walsh; dl3: count parities;
- `_bumpr`: dl3 count and raw parities;
- `_bc`: count parities with n ≡ 2 (mod 4) in xs3 (C5's n = 34);
- `_w10`: RP fold into [0, 10] with a 4-unit bump parity;
- `_r`: raw-bit parities in xs3.

Failed the margin (> 0.02), so these do not count:

| variant | dense | worst margin |
|---|---|---|
| `c2:d8_rp_m4s4_mp_cp_u1_sl_d4` (MPC parity layer) | 43,925,724 | 0.0253 |
| `c2:d6_m4s4_mp_cp_z11_lz5_th_sl_d4` (TH1S + LZ5) | 56,104,763 | 0.0206 |
| `c2:d6_m4s4_mp_cp_c5_th_sl_d4_bc` | 56,870,395 | 0.0286 |
| `xc:d5_m4k2_mp_cp_mpc_th_d4` | 78,215,849 | 0.0205 |
| `xc:d5_m4k2_mp_cp_mpc_th_bump` | 76,263,044 | 0.0235 |

Common factor: a bump stacked on TH1S plus a second error source (LZ5, the n = 34 bump, or at d5
the X2 bump over TH1S), or next to the d8 MPC parity layer. The same bumps without TH1S (d5, d6)
or with glu_xor/C5 neighbours verify with room to spare.

Extra checks (`runs/extra/`):
- `validate_bench` for the d4, d5, d6, d7, d8 and d8-w10 points: weights are bit-equal to
  `Compiler().get_mlp_from_tree` at log_w 0-2, and depth, dense and sparse match. This covers every
  builder touched (xs3, xs4, dl3);
- the harness at log_w 4 and 5 with 64 messages, all ok with 0 wrong bits, for the d4, d5, d6, d7,
  d8 and d8-w10_r bests: margins 0.0008-0.0081;
- 256-message stress runs at log_w 6, all ok with 0 wrong bits:
  - d8 `_w10_r`: 0.0098;
  - d8 `_g_d4_r`: 0.0044;
  - d8 m3 `_g_d4`: 0.0016;
  - d7 `_d4_r`: 0.0060;
  - d6 `z11_th_sl_d4`: 0.0031;
  - d5 `_th_l4w`: 0.0038;
  - d4 `_bump`: 0.0017.
- Regression: with every switch off, the patched repo rebuilds combined-2's d4, d5, d6 and d8
  bests with the same dense, sparse and harness margin to every printed digit (`runs/regress/`).

## Negative results and how this was found (`search/`)

1. **Structure lemmas.** They hold for any real gates, with units max(0, g)·v, g and v affine in
   (C_L, C_R), plus a free constant.
   - **N1.** Every row and every column of the 6x6 grid needs at least 2 distinct interior knots.
     Proof: each piece between knots is one quadratic, which can fit at most 3 consecutive
     alternating points. With a single knot the pieces must be {0,1,2} and {3,4,5}, and
     q2 - q1 = ±2(x² - 5x + 8) has no real root.
   - Hence each open edge of [0,5]² needs 2 distinct crossings: 8 in all. A line gives at most 2,
     so **3 units is impossible**. With 4 units, every line crosses exactly two open edges and no
     corner, which leaves 6 configurations. The bump is 2 lines cutting the lower-left corner and 2
     cutting the upper-right corner (BL/RT), all on the anti-diagonal.
   - **N3.** For two edge-adjacent unit cells that no line cuts strictly, and whose shared edge
     lies on no line, the mixed differences are equal. The target alternates ±2, so such pairs are
     impossible. Checked on 492K random cases (`test_n3.py`).
   - Local-region test: a corner line is constant on a fixed half-grid.
   - A lemma I first used, "F3: every edge needs a fully active non-crossing line", is **false**.
     Two facing knots cover the row (`f3where.py`), so I dropped it and reran everything.
     `old_withF3/` holds the superseded runs.
2. **Exhaustive integer-direction search** (`search6.py`; `search5.py` is a plain cross-check).
   - Space: gates a x + b y + c with (a, b) primitive, |a|, |b| ≤ A, both orientations,
     c ∈ Z/Q, real affine values and a free constant.
   - **No exact 4-unit D** for (A, Q) = (2,2), (2,3), (2,4), (3,1), (3,2), (4,1)
     (search5: (1,2), (2,1), (2,2)).
   - This settles the MILP's open cases A = 2 and A = 3 (integer (a,b,c) with |a|,|b| ≤ 3 lies in
     (3,1) ∪ (1,2) ∪ (1,3)). It is consistent with the irrational knots.
   - Validation: 12/12 planted random 4-unit targets were recovered across the 6 configurations.
   - (3,3), (3,4), (4,2) and (5,1) were stopped after the construction was found.
3. **Real gates.**
   - `gen_active.py` enumerates all 836 active sets of generic lines that cross two open edges.
     The list is complete: directions up to |a|, |b| ≤ 10 are the mediants of all critical
     directions.
   - `search_rel.py` relaxes each unit to "a quadratic on its active set". This leaves 18.5K
     survivors in the BL/RT configuration.
   - `surv_opt.py` optimises the real gates from those survivors. It hits residual 1e-13 at
     once, and the solutions have all lines on (1,1). That is how the irrational knots showed up.
   - `oned4*.py` map the 1-D real-knot families: 395 solutions, 16 orientation patterns, most
     with slopes above 100. `bump.py` is the well-conditioned member.

## Headroom and lower bounds

- **D in X layers.** 4 units is optimal for this unit form (N1).
  - With 1-D gates, slope 2 at 4 units is forced.
  - With 2-D gates, no flatter form was found near the bump.
  - The only further lever is D not computed on its own, for example E = a + C_L + C_R with a
    wider lazy window. That is out of scope here, and it moves cost downstream.
- **Margin.** The failing stacks miss by 0.0005-0.009:
  - d6 with LZ5 would reach 56.10M (-0.77M more);
  - d5 with the X2 bump would reach 76.26M;
  - d8 with MPC would reach 43.93M.
  A quieter TH1S, LZ5 or k2 decoder, or glu_xor D only in the columns that feed the worst bits
  (untried), would get them. The mirrored bump form (`BUMPMIR`, s → n - s) gives identical
  margins: |F'| is 2 at every even s either way.
- **d3.** Bumps in L2's pair parities and in Walsh (n = 26) give `dl3:d3c1_mp17_bump`:
  246,983,549 (-6.8M, -2.7%), and 246,142,757 with raw bumps. Both **fail the margin**
  (0.037-0.050); d3 already sits at 9.8e-3. `dl3:d3c1_mp17_bump_ctr` (smaller cancelling tails)
  fails the same way (0.0376 / 0.0489), so the cause is the slope: d3's L2 inputs carry the largest
  count errors. d3 stays at combined-2's 253.8M.
- **Precision.** Inner bumps cancel tails of size up to (n - c)^2. `BUMPCTR` orients each inner
  bump's tail toward the nearer end: exact, and max |unit| drops 433 → 161 (n = 26) and 833 → 281
  (n = 34). On the d6 C5 + TH1S + n = 34 stack it does **not** rescue the margin (0.0283 against
  0.0286). So those failures come from the slope, not from cancellation.

## Files

- `patch.diff`: the diff against combined-2's repo:
  - `minpar_counts.bump_units` / `check_bump`;
  - switches `xs3.D4 / BUMPC / BUMPRAW / RPW`, `xs4.D4 / D4L / BUMPALL`, `dl3.BUMP / BUMPR`, and
    `minpar_counts.BUMPCTR / BUMPMIR` (all off by default);
  - the variants in `c2.py`, `xc.py` and `dl3.py`.
- `repo/`: combined-2 plus the patch.
- `run.sh <mod:fn>`: runs the harness and both audits into `runs/`.
- `collect.py`: rebuilds `results.jsonl` (runs + search summaries).
- `gen_table.py`: writes `results_table.md`.
- `pareto.py`: writes `pareto.md` (merged front with combined-2).
- `runs/extra/`: `validate_bench`, log_w 4/5 and 256-message stress runs.
- `search/`: all searches and their logs.

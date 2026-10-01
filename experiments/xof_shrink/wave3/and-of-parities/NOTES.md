# and-of-parities (wave 3): a cheaper exact AND of two parities

Item: NEXT.md 1, the exact AND (1 - p_B) p_C of two raw-bit parities in the fused rounds (d4x L1
holds 12,956 pair-parity units). Work copy: `repo/` = combined-2's repo + this patch
(`patch.diff`: `dl3.py`, `xs3.py`, `xs4.py`, `xc.py`, `c2.py`, new `depth_low/lib/python/p1d_forms.py`;
apply with `patch -p3` inside a copy of combined-2's `repo/`).
Every row claimed below passed the harness (16 msgs) and `audit/adv_check.py` on BOTH reference
sets (ref777_w6, 69 msgs; combined-1 ref_w6, 63 msgs) with `ok: true`, 0 wrong bits and an empty
eager mismatch. Raw JSON: `runs/<variant>.{h,a777,a63}.json`; `results.jsonl` has one line per
check, including the failing runs (`ok: false`).

## 1. Result

**The AND itself is not cheaper than "free singles + one pair parity" in any space I searched
(section 5), but the pair parity is.** Parity of an integer count s in [0, n] needs only
**floor((n-1)/2) gated units for every n >= 6, including even n**: one unit fewer than glu_xor
(ceil(n/2)) and than every earlier form. Min-parity had saved the unit only for odd n; waves 1-2
recorded "even n gain nothing (exhaustive for n = 8, 10)" and "range 10 (D) needs 5; 4 is
impossible". Those searches used integer or grid knots. The new forms need knots at *irrational*
points between integers. The key piece is n = 6 with 2 units instead of 3 (d = s - 3,
r = 2 sqrt 2 - 2 = 0.8284, a = 2 + 2 sqrt 2):

    1 - parity(s) = 8 + max(0, d + r)(d - a) + max(0, r - d)(-d - a)

That is (d + 2)^2 for d <= -r, 2 d^2 for |d| < r and (d - 2)^2 for d >= r: two unit-curvature
parabolas through 3 alternating points each, glued by a steeper parabola through the middle point.
The glue points +-r are the roots of d^2 + 4d - 4 = 0. The form's end pieces are unit parabolas
through 3 lattice points, so glu_xor ramps 4 max(0, +-(s - k)) with k on even integers continue it
on either side. That gives **n/2 - 1 units for every even n in closed form** (`p1d_forms.ext`,
`ext_c`, `ext_t`). Beyond the 7-point core the forms are flat under silu, with dyadic weights.

It applies wherever an even-range parity sits in the frontier circuits:
- the fused round-1 raw-bit parities (d3, d4);
- X2's column-pair parity D (range 10, depths 5-8), which is NEXT.md item 2;
- the even count parities of the later layers of d3 (fused round 2, Walsh), d4x and d5, and
  sparse-focus's layouts;
- the even round-1 raw parities of d5-d8;
- odd ranges n, through the even form of range n + 1: fewer units where glu_xor was used (d3),
  and the same count as min-parity but sparser elsewhere.

New verified frontier points (bold: better than combined-2's dense-best point of that depth):

| depth | variant | dense | sparse | before (combined-2 best) | change | x baseline | margins harness / ref777 / c1 |
|---|---|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt1_xca` | **231,671,853** | **1,222,721** | 253,823,869 / 1,292,483 | -22.15M (-8.7%), sparse -70K | 14.3 | 4e-4 / 1.21e-2 / 8.1e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | **124,087,564** | **526,518** | 134,661,305 / 547,787 | -10.57M (-7.9%), sparse -21.3K | 26.6 | 9.3e-4 / 5.6e-3 / 5.6e-3 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` | **74,073,879** | **226,480** | 79,361,129 / 240,057 | -5.29M (-6.7%), sparse -13.6K | 44.6 | 6e-4 / 1.4e-3 / 2.1e-3 |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_sl_p1dct_ra` | **53,915,598** | **179,991** | 57,250,043 / 192,041 | -3.33M (-5.8%), sparse -12.1K | 61.3 | 5e-4 / 3.8e-3 / 3.8e-3 |
| 7 | `c2:d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dct_ra` | **48,066,365** | **134,662** | 49,375,781 / 136,618 | -1.31M (-2.7%), sparse -2.0K | 68.8 | 1.8e-3 / 5.9e-3 / 4.4e-3 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra` | **43,761,588** | **114,731** | 45,071,004 / 115,841 | -1.31M (-2.9%), sparse -1.1K | **75.5** | 3.2e-3 / 1.46e-2 / 1.33e-2 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` (robust) | 44,218,228 | 114,557 | 45,527,644 / 116,513 (`_g`) | -1.31M, sparse -2.0K | 74.8 | 1.5e-3 / 4.5e-3 / 3.4e-3 |
| 4 (sparse end) | `xc:d4a_m4k3_mp_mpc_p1dxt_ra` | 143,319,113 | **400,884** | `xc:d4a_m4k3_mp_mpc` 159,651,569 / 425,212 | -16.33M, sparse -24.3K | 23.1 | 4e-4 / 1.2e-3 / 1.3e-3 |
| 6 (sparse-leaning) | `c2:d6_cp_c5_p1dct_ra` | 56,937,635 | **168,357** | `c2:d6_cp_c5` 63,564,360 / 170,175 | -6.63M, sparse -1.8K | 58.1 | 4e-4 / 5e-4 / 5e-4 |
| 7 (sparse-leaning) | `c2:d7_cp_u1_c5_p1dct_ra` | 51,088,402 | 123,028 | `c2:d7_cp_u1_c5` 53,163,143 / 123,702 | -2.07M, sparse -674 | 64.7 | 4e-4 / 1.2e-3 / 1.8e-3 |
| 7 (sparse-leaning) | `c2:d7_cp_sp_c5_p1dct_ra` | 51,477,994 | **122,212** | `c2:d7_cp_sp_c5` 53,652,255 / 122,886 | -2.17M, sparse -674 | 64.2 | 6e-4 / 7e-4 / 9e-4 |
| 8 (robust) | `c2:d8_rp_cp_u1_g_p1dct_ra` | 45,557,657 | 112,091 | `c2:d8_rp_cp_u1_g` 47,632,398 / 112,765 | -2.07M, sparse -674 | 72.6 | 4e-4 / 1.2e-3 / 1.8e-3 |
| 8 | `c2:d8_rp_cp_sp_p1dct_pf_ra` | 45,425,684 | 111,917 | `c2:d8_rp_cp_sp` 47,598,310 / 111,629 | -2.17M, sparse +288 | 72.8 | 4.4e-3 / 7.2e-3 / 8.8e-3 |
| 7 (sparse end) | `c2:sp_lazy2_l3_p1d` | 67,786,082 | **113,772** | `sp:lazy2_l3` 70,156,703 / 114,454 | -2.37M, sparse -682 | 48.8 | 2e-4 / 2.7e-3 / 3.0e-3 |
| 8 (sparse end) | `c2:sp_base_p1d` | 62,287,910 | **95,637** | `sp:base` 64,162,035 / 95,955 | -1.87M, sparse -318 | 53.1 | 1.0e-4 / 1.0e-3 / 6.0e-4 |
| 9 (sparse end) | `c2:sp_col1b_p1d` | 66,359,285 | **91,460** (fewest nonzeros of any depth) | `sp:col1b` 68,233,410 / 91,778 | -1.87M, sparse -318 | 49.8 | 1.0e-4 / 1.0e-4 / 1.0e-4 |
| 9 | `c2:sp_col1b_x2e_p1d` | 64,339,445 | 94,340 | `sp:col1b_x2e` 66,111,490 / 94,658 | -1.77M, sparse -318 | 51.4 | 1.0e-4 / 1.0e-4 / 1.0e-4 |

The sparse-end rows at depths 7-9 are sparse-focus's layouts (`sp.py`) with their even count
parities on the top-core forms (`c2.with_sp_p1d`). The sparse-leaning rows at depths 6-8 are
combined-2's c5/cp points with the new D and round-1 forms. The depth-4 sparse end is xc's d4a
layout with the new count and round-1 forms. Each of these beats the point it came from on both
dense and sparse, except `d8_rp_cp_sp` (+288 nonzeros). They are the new sparse-best points at
depths 4, 7, 8 and 9. At depth 6, round-structure's `rs:lazy_middle_xp` (70.4M / 168,175)
keeps 182 fewer nonzeros than `d6_cp_c5_p1dct_ra` (56.9M / 168,357). At depth 5, `rs4:d5_m3k2_xp`
(84.7M / 226,445) keeps 35 fewer than the new d5 headline (74.1M / 226,480).

- Every headline point beats combined-2's dense-best point of the same depth on dense AND sparse
  (the sparse drop comes from the constant-value ramps of the `ext` forms, section 6).
- Depth 8 is now 43.76M, 75.5x below the baseline (3,305,445,348).
- The margin-robust depth-8 point (44.22M, worst audit margin 4.5e-3) also beats every
  combined-2 depth-8 point on dense.
- Worst audit margins of the dense-best rows, now and in combined-2 (all below the 0.02
  tolerance; see section 4):

  | depth | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|
  | now | 1.2e-2 | 5.6e-3 | 2.1e-3 | 3.8e-3 | 5.9e-3 | 1.5e-2 |
  | combined-2 | 9.8e-3 | 2.8e-3 | 6.2e-3 | 6.6e-3 | 5.0e-3 | 8.6e-3 |
- Re-verification: every dense-best row, the `_r` / `_kt_xc` / d3 `_kt1_xco` versions, and the
  d4a, d6 c5, d7 u1 c5, d8 u1 g and sp base / col1b rows were rebuilt with the final code in fresh
  processes (`runs/final/`), with identical dense, sparse and
  margins, all `ok` on both audits.

Other verified rows (dominated now or alternatives), all in `runs/` and `results.jsonl`:

| depth | variant | dense | sparse | margins h / ref777 / c1 |
|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt1_xco` | 231,671,853 | 1,237,257 | 3e-4 / 1.10e-2 / 7.2e-3 |
| 3 | `dl3:d3c1_mp17_p1d_kt_xca` | 238,512,173 | 1,235,283 | 3e-4 / 4.5e-3 / 4.3e-3 |
| 3 | `dl3:d3c1_mp17_p1d_kt_xco` | 238,512,173 | 1,249,819 | 2.4e-4 / 4.0e-3 / 3.8e-3 |
| 3 | `dl3:d3c1_mp17_p1d_kt_xc` (odd raw columns on centred glu_xor) | 239,321,237 | 1,252,267 | 2.2e-3 / 6.3e-3 / 1.09e-2 |
| 3 | `dl3:d3c1_mp17_p1d_xc` (raw forms only, centred ext) | 246,161,557 | 1,258,283 | 2.1e-3 / 3.8e-3 / 1.08e-2 |
| 3 | `dl3:d3c1_mp17_p1d` (raw forms only, VP table) | 246,621,613 | 1,331,739 | 2.9e-3 / 4.7e-3 / 1.14e-2 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xc` (odd raw sets on min-parity) | 124,087,564 | 541,054 | 8.6e-4 / 4.6e-3 / 4.6e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_ke_xc` (count core at the bottom) | 124,087,564 | 541,054 | 6.6e-4 / 1.55e-2 / 8.5e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_ke_x` | 124,087,564 | 541,054 | 9.2e-4 / 1.80e-2 / 9.2e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_ke` (VP raw forms) | 124,087,564 | 613,118 | 1.6e-3 / 9.7e-3 / 9.6e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_k` | 126,382,092 | 617,571 | 5.2e-4 / 3.6e-3 / 2.2e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d` | 127,595,249 | 615,009 | 4.6e-4 / 3.6e-3 / 1.8e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1ds_k` | 129,712,788 | 554,835 | 5.9e-4 / 3.6e-3 / 1.9e-3 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1ds` (singles only) | 130,925,945 | 552,273 | 5.9e-4 / 3.6e-3 / 1.2e-3 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_th_p1dxt_ra` (TH1S round 1) | 76,056,519 | 234,304 | 3.1e-3 / 1.60e-2 / 1.04e-2 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_th_p1dxt_r` | 76,056,519 | 238,112 | 3.3e-3 / 1.54e-2 / 1.01e-2 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_th_p1dxt` | 76,263,044 | 238,166 | 3.3e-3 / 1.51e-2 / 1.01e-2 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_th_p1dxe` | 76,263,044 | 238,166 | 3.1e-3 / 1.53e-2 / 1.26e-2 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_th_p1dxf` / `xc:d5_m3k2_mp_cp_mpc_th_p1dxf` | 77,002,692 / 77,030,838 | 243,535 / 241,057 | up to 1.89e-2 |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_th_sl_p1dct_ra` (TH1S round 1) | 55,898,238 | 187,815 | 3.9e-3 / 1.14e-2 / 1.79e-2 |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_th_sl_p1dct_r` | 55,898,238 | 191,623 | 4.4e-3 / 1.15e-2 / 1.82e-2 |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_th_sl_p1dce_r` | 55,898,238 | 191,623 | 4.2e-3 / 1.13e-2 / 1.91e-2 |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_th_sl_p1dce` / `_p1dcf` | 56,104,763 | 191,677 / 192,957 | up to 1.93e-2 |
| 7 | `c2:d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dct_r` | 48,066,365 | 136,198 | 3.9e-3 / 5.8e-3 / 7.7e-3 |
| 7 | `c2:d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dce_r` | 48,066,365 | 136,198 | 3.4e-3 / 1.01e-2 / 8.2e-3 |
| 7 | `c2:d7_m4s4_mp_cp_u1_z11_lz5_sl_p1dce` / `_p1dcf` / `_p1dc` | 48,230,501 | 136,254 / 137,534 / 137,534 | 6.5e-3 to 1.21e-2 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_r` / `_g_p1dct_r` | 43,761,588 / 44,218,228 | 116,267 / 116,093 | 1.20e-2 / 1.33e-2; 4.7e-3 / 2.9e-3 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dce_pf_r` | 43,761,588 | 116,267 | 6.6e-3 / 1.34e-2 / 1.59e-2 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dce_pf` | 43,925,724 | 116,323 | 6.3e-3 / 1.21e-2 / 1.49e-2 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dce_r` / `_g_p1dce` / `_g_p1dcf` | 44,218,228 / 44,382,364 / 44,382,364 | 116,093 / 116,149 / 117,429 | <= 5.0e-3 |

## 2. The parity forms: how they were found and why they are exact

Unit model (that of every wave-1/2 construction): parity(s) for an integer s in [0, n] as
c + sum_k max(0, g_k s + h_k)(v_k s + w_k), where s is the count of a raw bit set or a count
feature. Between consecutive knots the function is one quadratic. Consecutive quadratics differ by
(s - knot)(affine), so they must intersect at the knot.
- glu_xor puts the knots on even integers, so neighbouring 3-point parabolas share a lattice
  point: ceil(n/2) units.
- Min-parity puts knots between integers: (n-1)/2 for odd n.
- Two alternating 3-point parabolas never meet strictly between their lattice points (for example
  1 - (s-1)^2 and (s-4)^2 give s^2 - 5s + 8 > 0). That is what made even n look stuck.
- The fix is a steeper parabola through one lattice point in between (the 2 d^2 piece above). Its
  intersections with the neighbours are irrational, so no integer, half-integer or grid search
  finds it.

**Search** (`search/par1d_vp.py`, `search/best1d.py`, `search/polish1d.py`) uses variable
projection:
1. For fixed knots and directions, the value coefficients are a linear least-squares problem, so
   only the K knots are optimised: Adam, softplus annealed to relu, 2048-4096 random restarts per
   (n, K).
2. Exact candidates are polished by Levenberg-Marquardt to a residual below 1e-12.
3. They are ranked by knot distance to the integers, the size of the cancelling terms, and the
   slope at the integers.

Fewest units found (with one batch of 4096 restarts; "no" = the batch found nothing at K - 1):

| n | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| found | 2 | 3 | 3 | 4 | 4 | 5 | 5 | 6 (mp) | 6 | 7 (mp) | 7 | 8 (mp) | 9 |
| K - 1 | - | - | - | no | - | no | no | no | - | no | - | no | - |
| glu_xor / min-parity before | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 | 8 | 8 | 10 |

So K(n) = floor((n-1)/2) for n >= 6 in every case searched, and the gain is exactly one unit for
every even n. A piece-counting argument supports it as the true floor for this unit family, but it
is not a proof:
- a piece covers at most 3 lattice points;
- two 3-point pieces cannot be adjacent;
- a 2-point piece next to a "cap" (0, 1, 0) must be strongly concave (curvature < -1), and one
  next to a "cup" strongly convex, so cap-2-cup and cap-2-2-cup chains are infeasible;
- the feasible patterns (3-1-3, 2-3-2, ...) average about 2 points per unit.

**Exactness.**
- `search/mpcheck_forms.py` evaluates every table form at every integer in 50-digit arithmetic:
  the max error is <= 4e-13 for the float64 coefficients, and the same with silu(32 g)/32 in place
  of relu.
- Gates are scaled so that every integer is >= 1 away from each knot in gate units (silu error
  ~ exp(-32)).
- The n = 6 form is exact in closed form (knots 3 -+ (2 sqrt 2 - 2)).
- `ext`, `ext_c` and `ext_t` add only glu_xor ramps, which are exact on the lattice by the same
  identity as glu_xor: the core's last piece is a unit parabola through 3 lattice points.

**The table** (`p1d_forms.py`):
- `FORMS` holds the VP forms for n = 6, 8, 10, 14 and 16, each with the smallest cancelling terms
  found.
- `FORMS_FLAT` holds the flattest VP forms for n = 10 and 11 (max slope at the integers 2.2 and
  2.0; MIN_PARITY[11] has 2.85).
- `FORMS_EXT`, `FORMS_EXTC` and `FORMS_EXTT` hold the n = 6 core plus ramps, with the core at the
  bottom, the middle or the top of [0, n].

## 3. Where the forms are used (switches in the patch)

| switch | where | effect |
|---|---|---|
| `P1D` (dl3 `raw_units`), `P1D["ext"] = "c"`, `P1D["odd"]` | raw-bit parities of the fused round-1 layer (d3, d4); `odd`: d3's odd column parities of 35-41 bits use the even form of range n + 1 | theta1 singles of 6 and 8 bits (8 + 952 sets) and pair parities of 14 and 16 bits (168 + 688 pairs) lose one unit each: -1816 L1 units. `"c"` uses the centred core + constant-value ramps (sparser, cancelling terms ~ (n/2)^2). |
| `P1DK` (dl3 `parity_specs`), `ext = "t"`, `odd = "t1"` | counts in d4x: L2's D (range 20), L3's column parities (range 10), L4's Walsh pairs (range 26); d3's round-2 pairs (26) and Walsh pairs (42); with `t1` also d3's odd ranges (round-2 singles 13, Walsh singles 21 and triples 63) via the even top-core form of range n + 1 | -1 unit per even parity (and per odd parity with `t1`, against glu_xor). `"t"` puts the core's non-flat integers at the top of the range, where counts are rare. |
| `P1DC` (xs3 `cp_d_specs`: `with_p1dc`, `_flat`, `_ext`, `_top`) | X2's D = parity(C_L + C_R), range 10, depths 6-8 | 5 -> 4 units: -320 X2 units, -1.15M. This settles NEXT.md item 2: the d2d MILP had only integer gate offsets. |
| `P1DX` (xs4 `parity_specs`: `_p1dx`, `_p1dxf`, `_p1dxe`, `_p1dxt`) | d5 layout: X2's D, the lazy chi's column parities (10), the Walsh pairs (26) | -1 unit each: -3.10M at depth 5 |
| `P1DR` (xs3 `par_units`, `with_p1dr`, `with_p1dr_all`) | round-1 raw parities of d5-d8: the X1 D parities (d7, d8) and, without TH1S, the whole theta1 layer (d5, d6) | even sets one unit fewer (-64 X1 units at d7/d8); with `_ra` odd sets on the range n + 1 form. In theta1 direct (d5, d6) this beats TH1S: 4,619 units against 5,202, and the margins drop from 1.6-1.8e-2 to 2-4e-3 |
| `P1DP` (xs3 `count_parity_units`, `with_p1dp_flat`) | the depth-8 parity layer (range 11) | same 5 units as MIN_PARITY[11] but flatter (slope 2.0 against 2.85). This is what lets the non-`_g` depth-8 point pass with the new D. |

Variant suffixes:

| suffix | meaning |
|---|---|
| `_p1d` | P1D on |
| `_p1ds` | P1D on singles (<= 9 bits) only |
| `_k` | P1DK with the flat forms |
| `_ke` | P1DK with the bottom core |
| `_kt` | P1DK with the top core |
| `_kt1` | `_kt` + odd count ranges on the range n + 1 top-core form |
| `_x` | raw forms `ext` |
| `_xc` | raw forms `ext_c` |
| `_xco` | `_xc` + odd raw sets > 17 bits on the range n + 1 form |
| `_xca` | `_xc` + every odd raw set >= 7 bits on the range n + 1 form (same units as min-parity, sparser) |
| `_p1dc` | P1DC with the VP form |
| `_p1dcf` | P1DC with the flat form |
| `_p1dce` | P1DC with the bottom core |
| `_p1dct` | P1DC with the top core |
| `_pf` | P1DP |
| `_r` | P1DR (even round-1 raw parities) |
| `_ra` | P1DR + odd round-1 raw parities (7, 9 bits) on the range n + 1 form: same units, sparser |

## 4. Float32: what the forms cost in margin, and what failed

**Raw-bit parities (exact inputs).** Only the float32 rounding of the irrational weights times the
cancelling terms matters.
- A first n = 14/16 pair table with terms up to 500-950 failed the audit on the all-zero message
  (0.020).
- Forms with terms <= 128 pass with the old margins (d4 3.6e-3).
- The centred ext forms pass as well, and so do the odd raw sets on the even form of range
  n + 1 (`_xca`, `_xco`, `_ra`). At depth 3 they even lower the margin (section 4 below).

**Count parities (inputs carry the upstream float error).** Knots between integers leave a slope
at the integers. glu_xor's kinks sit on integers, where silu averages them to a flat point, so
glu_xor does not amplify input noise and these forms do.
- The VP-found n = 10 forms (slope 2.2-3.3 at many integers) pass at depth 7, fail at depth 8
  (0.025-0.037) and are marginal at depths 5-6 (0.018-0.019).
- The core + ramps forms are non-flat only at 3 points. With the core at the top of the range
  (`t`) those points sit where counts are rare: best margins everywhere (depth 4: 4.6e-3 against
  1.55e-2 with the core at the bottom).
- Depth 8 needs both the top-core D and the flatter range-11 parity (`_pf`). With MIN_PARITY[11]
  it fails at 0.021-0.022.
- At depths 5-6 the D forms looked marginal (1.6-1.9e-2) on top of TH1S's column-shared round-1
  theta. With theta1 computed directly on the new raw forms instead (`c2:d6_m4s4_mp_cp_z11_lz5_sl_p1dct_ra`,
  `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra`), the same D gives 3.8e-3 and 2.1e-3, with fewer units. TH1S's
  2-D forms were the larger noise source.
- Depth 3 shows the effect most clearly. The fused round 2 on counts <= 13 (pairs of range 26)
  and the Walsh pairs (range 42):
  - with the bottom-core forms: fails at 0.021 / 0.036 (`dl3:d3c1_mp17_p1d_ke`, 239.8M dense),
    the float32 sensitivity depth-low documented;
  - with the top-core forms: passes at 6.3e-3 / 1.09e-2 (`dl3:d3c1_mp17_p1d_kt_xc`, 239.3M dense);
    moving layer 1's odd raw column parities (35-41 bits, centred glu_xor) to the even form of
    range n + 1 (`_xco`) saves 1 unit per odd column and *improves* the margins to 4.0e-3 / 3.8e-3
    (238.5M), because those forms cancel smaller terms than the centred glu_xor;
  - with the odd count ranges (round-2 singles, range 13) also on min-parity, reflected so its
    non-flat points sit at the top: fails badly at 0.064 / 0.055 (`dl3:d3c1_mp17_p1d_kto_xc`,
    232.5M dense). Min-parity has 7 non-flat integers and counts of range 13 hit them;
  - with the odd ranges on the even top-core form of range n + 1 (valid on [0, n]): passes ref777
    (8.2e-3) but fails c1 at 0.029 (`dl3:d3c1_mp17_p1d_kt1_xc`, 232.5M). On top of `_xco`
    (the quieter layer 1) the same forms pass: 1.10e-2 / 7.2e-3 (`dl3:d3c1_mp17_p1d_kt1_xco`,
    231.7M); `_xca` (odd raw sets on the sparser forms) gives the depth-3 headline.

**Failing runs**, kept in `runs/` and `results.jsonl` with `ok: false`:
- `c2:d8_rp_m4s4_mp_cp_u1_sl_p1dc`, `_p1dcf`, `_p1dce` and `_p1dcf_pf`;
- `c2:d6_m4s4_mp_cp_z11_lz5_th_sl_p1dc`;
- `xc:d5_m4k2_mp_cp_mpc_th_p1dx`;
- `dl3:d3c1_mp17_p1d_ke`, `dl3:d3c1_mp17_p1d_kto_xc` and `dl3:d3c1_mp17_p1d_kt1_xc`;
- the first `dl3:d4x_m4k2_mpx_wmpc_p1d` table with big terms, which was overwritten in `runs/`
  (reported above: margin 0.0201 on zeros).

## 5. The AND itself: what was searched, and what it shows

Target: T = (1 - p_B) p_C on disjoint raw sets B and C, with the singles p_B, p_C and a constant
free (they are shared units of the layer). In d4x, |B| and |C| lie in 6..9, the sets are always
disjoint, and |B xor C| = 13..17 (`overlap.py`). Any cheaper form would replace the pair parity
p(B xor C), which now costs floor((|B| + |C| - 1)/2) units.

| search space | sizes, units K | pair parity needs | result |
|---|---|---|---|
| **exhaustive**: gates on individual bits, integer weights in [-2, 2], offsets in Z/2; values real (exact least squares); symmetry-reduced (bit flips, permutations inside B and C, B <-> C) with an exact 3-face cut prefilter | 3+3, K = 2 | 2 (new n = 6 form; 3 before) | **0 solutions** in 267,850 surviving gate pairs (699 canonical first gates x 249,688 gates). Sanity 2+2: K = 1 none, K = 2 found. The gap to the real-knot solution shows that integer offsets are the blind spot. |
| variable projection, real gates on individual bits (1024 restarts x 3000 steps) | 4+4, K = 2 | 3 | not found (best max error 0.25) |
| variable projection, 2-D count model: gates a b + e c + g with real a, e, g, values affine in (b, c) (2048-4096 restarts x 2000 steps) | 3+3 K=2; 4+4 K=2; 5+5 K=3; 7+7 K=5; 7+8 K=6; 8+8 K=6; 8+9 K=7 | 2; 3; 4; 6; 7; 7; 8 | 3+3 found (the pair parity itself); all others not found (best max error 0.25-0.28) |
| the same, with every function of b alone and of c alone free (per-set shared units at no cost) | 4+4 K=2; 8+8 K=6 | 3; 7 | not found (0.29; 0.31) |
| parity of n individual bits, real gates (512-1024 restarts) | n = 4, 5 with K = 1; n = 7, 8 with K = 2 | 2, 2, 3, 3 | not found (asymmetric gates do not beat the count forms) |

- **Restriction lower bound.** Fix x_C with p_C = 1; the units must then compute parity(B) plus a
  constant. So K >= the parity cost of max(|B|, |C|) bits with a free constant: 2 for 3 bits (3+3
  is now tight at 2), 2 for 4 and 5 bits (1 unit fails numerically), 3 for 8 bits.
- For 8+8 the bound is 3 against the 7 used. Every search up to 8+8 found the pair parity to be
  the cheapest exact AND.
- The separable-free runs show that sharing units per set does not help. Only the cross mixed
  differences Delta_i Delta_j (i in B, j in C) matter, and separable functions have none.

**Other decompositions of the fused round** (exhaustive, a few lines of Python in this session):
- The layer must emit an integer lift o' of chi1 = t_a xor (not t_b and t_c) with a small range.
  L2 reads o' with one unit for the exact bit and sums it into D.
- I enumerated every lift with values in a window of 3 consecutive integers (o' in {0,1,2} or
  {-1,0,1}), written in the Walsh basis of (p_A, p_B, p_C).
- Only two lifts need a single non-single parity, and both use p(B xor C): today's
  o' = t_a + (1 - t_b) t_c and its mirror.
- Every other lift needs 2-4 non-single parities. For example p(A xor B) + p(A xor C) +
  p(A xor B xor C) avoids B xor C but costs three bigger parities.
- A wider window costs at least one more unit per bit in L2 plus a wider D: about +15.6M at
  depth 4 for window 4.

So within this layout the AND is exactly the pair parity, and the lever is the parity cost, which
the new forms cut.

## 6. Lower bounds and headroom (my view)

- **Parity of a count range n.** floor((n-1)/2) is the fewest found for n = 6..17 and 20, and
  K - 1 was never found. With the piece-counting argument (section 2), I expect it to be the floor
  for these units.
- **The fused rounds (d3, d4).** Every raw parity is now at floor((n-1)/2). What remains in d4x's
  L1 (15,967 units) is 1464 singles and 1600 pairs at that floor: about 4.0 + 6.9 units per bit.
  Only a cheaper AND than the pair parity would move it, and section 5 found none.
- **d4x's later layers.** No even count parity is left. Redone with floor((n-1)/2) parities,
  depth-low's d4x floor (about 116M) becomes about 110M, against 124.1M built. The built layers
  are [1145, 15967, 1601] [1601, 4483, 1657] [1657, 5019, 489] [489, 12771, 673].
- **Depth 3.** Every parity is now at floor((n-1)/2): raw ones by the forms, and count ones by
  the top-core forms (range 13 via the range-14 form). The rest is the fused round 2's pair
  structure (range-26 pair parities at 12 units per chi bit), which again only a cheaper AND
  would cut. Widths are now [1145, 21977, 1676] [1676, 28878, 471] [471, 21119, 673]
  (round 2: 32,078 -> 28,878 units). Depth-low's family-floor estimate (about 231M) assumed
  ceil(n/2)-unit parities. Redone with floor((n-1)/2) (round 1 about 84M, round 2 about 102M,
  Walsh about 24M) it is about 210M, against 231.7M built.
- **Depths 5-6, round 1.** theta1 direct with floor((n-1)/2)-unit parities (sets of 7-9 bits:
  3, 3, 4 units) costs 4,619 units against TH1S's 5,202, so TH1S no longer pays. The
  T = 8 columns are no longer special (NEXT.md item 4 is moot).
- **Depths 5-8.** X2 is now 1600 decode + 1280 D units. D is at floor((10-1)/2) = 4 per column
  pair, which answers NEXT.md item 2 positively. The other layers are unchanged, so combined-2's
  floor estimates move down by about 1.3M: depth 8 about 42.2M, depth 7 about 46.2M.
- **Sparse.**
  - The VP-found raw forms carry value weights on all n bits, so the first depth-4 points had
    +67K sparse. The core + constant-value-ramp forms fix that, and using them for the odd raw
    sets too (`_xca`, same unit count as min-parity) brings the depth-4 point to 526,518
    nonzeros, 21K fewer than combined-2's 547,787.
  - At depth 3 the same forms bring sparse down as well: 1,222,721 against 1,292,483.
  - At depths 5-8 sparse drops by 1-2.4% (`_ra`: odd round-1 raw parities on the sparser
    forms). The sparse-focus points drop by 318 nonzeros each.

## 7. Reproduce, files

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
A=$S/xof3/and-of-parities; R=$A/repo; E=$R/experiments/xof_shrink
cd $A && ./run.sh dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca  # harness + both audits -> runs/
# or by hand:
PYTHONPATH=$R/src:$E:$E/depth_low/lib/python $S/venv/bin/python $E/xofbench.py --log-w 6 --depth 3 \
  --variant c2:d8_rp_m4s4_mp_cp_u1_sl_p1dct_pf_ra --widths
python3 collect.py                                   # rebuilds results.jsonl from runs/
$S/venv/bin/python search/mpcheck_forms.py           # 50-digit exactness check of the forms
python3 repo/experiments/xof_shrink/depth_low/lib/python/p1d_forms.py   # float64 check, all tables
```

Files:
- `repo/experiments/xof_shrink/depth_low/lib/python/p1d_forms.py`: the forms. `FORMS`, `FORMS_FLAT`,
  `ext`, `ext_c`, `ext_t` (do not regenerate it with `search/mkforms.py`: that script rewrites only
  `FORMS` and drops the hand-added parts).
- Switches and variants: `dl3.py` (P1D, P1DK and the `_p1d*` variants), `xs3.py` (P1DC, P1DP,
  P1DR, `with_p1dc*`, `with_p1dp_flat`, `with_p1dr`), `xs4.py` (P1DX), `xc.py` (d5 variants),
  `c2.py` (d6-d8 variants).
- `search/`:
  - `par1d_vp.py`, `best1d.py`, `best1d_r.py`, `polish1d.py`, `mkforms.py`: the 1-D parity-form
    search;
  - `vp.py`: variable projection on the cube (parity / AND, individual-bit gates);
  - `vp2d.py`: the AND in the 2-D count model;
  - `and_int.py`: the exhaustive integer-gate AND search;
  - `mpcheck_forms.py`: the high-precision check;
  - `logs/`.
- `overlap.py`: sizes and overlaps of B and C in d4x. `l1diag.py`: float64 layer-1 lattice
  distances (used to find the float32 weight-rounding issue).
- `runs/`: harness and audit JSON for every variant run, plus `p1d_forms_v*.py` snapshots of the
  table at each stage. `runs/final/`: the fresh-process re-runs of the headline rows
  (`run_final.sh`). `results.jsonl` covers both, with the field `rerun_final_code`.

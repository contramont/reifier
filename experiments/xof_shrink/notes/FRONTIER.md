# Wave 2 frontier: 1-round Keccak XOF as a plain SwiGLU MLP (combined-2)

> **Superseded (final check, 2026-09-30).** Wave 3 (`../wave3/FRONTIER3.md`) and the float32 stress sets
> in `../README.md` replace these tables. For example, `c2:d8_rp_m4s4_mp_cp_u1_sl` reaches 1.6e-2 on the mixed
> stress set, while `c2:d8_rp_m4s4_mp_cp_u1_sl_g` stays at 6.7e-3 or below.

**Target.** `xof(msg, depth=3, k)` with `k = Keccak(log_w=6, n=1, c=448, pad_char="_")`, compiled
to reifier's `MLP_SwiGLU`:
- no residual stream, attention = identity, untied layers, no embedding or readout;
- input: BOS + 1144 message bits; output: BOS + 672 digest bits;
- default steepness c = 4, q = 8.

**What counts as verified.** A row counts only if all three checks pass, with `ok: true`,
0 wrong bits and an empty eager mismatch:
- the harness (`xofbench.py`, 16 random messages);
- `audit/adv_check.py` on `ref777_w6.pt` (69 messages);
- `adv_check.py` on `combined-1/adv/ref_w6.pt` (63 messages).

Rows marked "combined-2" were built and verified in this directory's `repo/`. The raw lines are
in `results.jsonl` and `runs/`. Rows credited to an avenue were verified in that avenue's own
repo. The key ones (marked "re-verified") were rerun here with identical numbers.

Reference points:
- baseline: depth 20, dense 3,305,445,348, sparse 2,034,788;
- wave-1 frontier:

| depth | dense | sparse |
|---|---|---|
| d4 | 160,358,065 | 425,212 |
| d5 | 89,244,445 | 229,869 |
| d6 | 69,368,253 | 177,623 |
| d7 | 63,651,004 | 127,470 |
| d8 | 61,082,590 | 109,101 |

## 1. Headline: best circuit per depth

| depth | variant | dense | sparse | x baseline (dense / sparse) | vs wave-1 dense | vs best single wave-2 avenue (dense) | worst audit margin |
|---|---|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17` (depth-low) | 253,823,869 | 1,292,483 | 13.0 / 1.6 | (new depth) | same point | 9.8e-3 |
| 4 | **`dl3:d4x_m4k2_mpx_wmpc`** | **134,661,305** | 547,787 | 24.5 / 3.7 | -16.0% | -0.55% (d4x_m4k2_mpx 135.40M) | 2.8e-3 |
| 4 | `xc:d4a_m4k3_mp_mpc` (sparse end, units-per-bit) | 159,651,569 | 425,212 | 20.7 / 4.8 | -0.4% | same point | 8.1e-4 |
| 5 | **`xc:d5_m4k2_mp_cp_mpc_th`** | **79,361,129** | 240,057 | 41.7 / 8.5 | -11.1% | -1.8% (fl4:d5_col 80.80M) | 6.2e-3 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_th` (m3 digests, lower margin) | 79,413,538 | 237,579 | 41.6 / 8.6 | -11.0% | -1.7% | 3.9e-3 |
| 6 | **`c2:d6_m4s4_mp_cp_z11_lz5_th_sl`** | **57,250,043** | 192,041 | 57.7 / 10.6 | -17.5% | -4.3% (xp3 pkc54 59.85M) | 6.6e-3 |
| 6 | `c2:d6_cp_c5` (sparse-leaning) | 63,564,360 | 170,175 | 52.0 / 12.0 | -8.4% | | 4.8e-4 |
| 7 | **`c2:d7_m4s4_mp_cp_u1_z11_lz5_sl`** | **49,375,781** | 136,618 | 66.9 / 14.9 | -22.4% | -4.8% (xp3 pkg54 51.87M) | 5.0e-3 |
| 7 | `c2:d7_cp_u1_c5` (sparse-leaning) | 53,163,143 | 123,702 | 62.2 / 16.4 | -16.5% | | 8.1e-4 |
| 8 | **`c2:d8_rp_m4s4_mp_cp_u1_sl`** | **45,071,004** | 115,841 | **73.3** / 17.6 | **-26.2%** | -7.6% (xo pksud 48.79M) | 8.6e-3 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g` (glu_xor last parity: robust) | 45,527,644 | 116,513 | 72.6 / 17.5 | -25.5% | -6.7% | 5.0e-3 |
| 8 | `c2:d8_rp_m3_mp_cp_u1_sl_g` (most robust of the d8 dense points) | 45,978,208 | 116,427 | 71.9 / 17.5 | -24.7% | -5.8% | 2.5e-3 |
| 8 | `sp:base` (sparse end, sparse-focus) | 64,162,035 | 95,955 | 51.5 / 21.2 | +5.0% | | 1.1e-4 |
| 9 | `sp:col1b` (sparse-focus; sparse only) | 68,233,410 | **91,778** | 48.4 / **22.2** | | same point | 5.0e-5 |
| 10 | none: no depth-10 construction beats depth <= 9 on either metric (see section 5) | | | | | | |

**Depth 8 is now 45.07M dense, 73.3x below the baseline.** Wave 1's estimated floor was about
50M, and the best single wave-2 avenue reached 48.79M. The gain over the avenues comes from
stacking tricks they found separately. Three of them had never been combined:
- units-per-bit's **fold layout** (RP);
- the column packing that the tricks-critic, packing and first-last avenues each found (cp);
- the optimizer's **round-1 u pairs** (u1).

The depth-8 margins are the largest in the table (8.6e-3 against the 0.02 tolerance). They come
from float32 rounding: min-parity knots between integers amplify the float32 error of the
decoded packed features. `_sl_g` replaces the MIN_PARITY[11] parity layer with glu_xor, which
is flat at the lattice points. It costs +0.46M and brings the margin down to 5.0e-3, or 2.5e-3
without the s4 staircase.

Extra checks for every bold row (`runs/`):
- `validate_bench.py`: the harness weights are bit-equal to `Compiler().get_mlp_from_tree` at
  log_w 0-2, and depth, dense and sparse match;
- the harness at log_w 4 and 5 with 64 random messages: all `ok`, margins 5e-4 to 5.4e-3;
- 256-message stress runs at log_w 6 for the depth-6/7/8 points (`*.stress256.json`): all
  `ok`, 0 wrong bits, worst margin 7.4e-3;
- the repo test suite (`runs/pytest.txt`): 58 passed, excluding hash_long_test and
  legacy_tests;
- regression: seven avenue variants rebuilt in the merged repo reproduce their dense and
  sparse exactly (`runs/regress/`):
  - wave-1 `xs3:split_first_middle_mp`;
  - tricks-critic `split_first_middle_m3m2_mp_cp_sp` and `xc:d5_m3k2_mp_cp`;
  - units-per-bit `split_first_lazy4_m3_mp_rp` and `lazy4c_middle_m3_mp_z11`;
  - packing `xp3:split_first_lazy4c_m4_mp_pkg54`;
  - round-structure `rs:split_first_middle_xpg_x12sc`.

The most margin-robust fold-layout point is `c2:d8_rp_cp_u1_g` at 47,632,398 / 112,765, with
worst audit margin 8.0e-4. It uses no min-parity anywhere and keeps the digests in pairs. It
is dominated in (dense, sparse) by `c2:d8_rp_cp_sp` (6.6e-3), but it is the depth-8 point to
use if float32 headroom matters more than 1.5M.

## 2. Verified Pareto table (all avenues + combined-2)

- The table below is the 2-D Pareto front (dense, sparse) at each depth, over every verified
  point of wave 2 and the wave-1 frontier rows.
- "3-D Pareto" marks the rows that also survive against shallower depths.
- The margin column is filled only for rows run here. Avenue rows keep their avenue's
  margins: at most 3.8e-3 for the rows in this table, except the optimizer's
  `split_first_middle_pkud_mc`.
- The optimizer's `_mc` rows pass both audits, but that avenue flagged them for margins of
  0.008-0.012.
- Generated by `pareto.py` from `avenue_points.txt` and `runs/`.

| depth | variant | dense | sparse | baseline / this (dense / sparse) | this / wave-1 frontier (dense / sparse) | 3-D Pareto | source | worst audit margin (combined-2 runs) |
|---|---|---|---|---|---|---|---|---|
| 3 | `dl3:d3c1_mp17` | 253,823,869 | 1,292,483 | 13.0x / 1.6x | - | yes | depth-low, re-verified | 9.8e-03 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc` | 134,661,305 | 547,787 | 24.5x / 3.7x | 0.840 / 1.288 | yes | combined-2 | 2.8e-03 |
| 4 | `dl3:d4x_mpx` | 135,500,786 | 545,309 | 24.4x / 3.7x | 0.845 / 1.282 | yes | depth-low | - |
| 4 | `dl3:d4x` | 140,356,754 | 533,861 | 23.6x / 3.8x | 0.875 / 1.256 | yes | depth-low | - |
| 4 | `xc:d4a_m4k3_mp_mpc_th` | 158,265,974 | 430,458 | 20.9x / 4.7x | 0.987 / 1.012 | yes | units-per-bit | - |
| 4 | `xc:d4a_m4k3_mp_mpc` | 159,651,569 | 425,212 | 20.7x / 4.8x | 0.996 / 1.000 | yes | units-per-bit, re-verified | 8.1e-04 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_th` | 79,361,129 | 240,057 | 41.7x / 8.5x | 0.889 / 1.044 | yes | combined-2 | 6.2e-03 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_th` | 79,413,538 | 237,579 | 41.6x / 8.6x | 0.890 / 1.034 | yes | combined-2 | 3.9e-03 |
| 5 | `fl4:d5_col` | 80,799,133 | 233,229 | 40.9x / 8.7x | 0.905 / 1.015 | yes | first-last, re-verified | 1.7e-03 |
| 5 | `xp4:d5_m3k2_mp_pc` | 81,555,805 | 232,333 | 40.5x / 8.8x | 0.914 / 1.011 | yes | packing | - |
| 5 | `xc:d5_m3k2_mp_cp` | 81,555,805 | 232,333 | 40.5x / 8.8x | 0.914 / 1.011 | yes | tricks-critic | - |
| 5 | `fl4:d5_best` | 82,080,573 | 232,327 | 40.3x / 8.8x | 0.920 / 1.011 | yes | first-last | - |
| 5 | `xp4:d5_m3k2_mp_pa` | 82,837,245 | 231,431 | 39.9x / 8.8x | 0.928 / 1.007 | yes | packing | - |
| 5 | `xc:d5_m3k2_mp_xp` | 82,837,245 | 231,431 | 39.9x / 8.8x | 0.928 / 1.007 | yes | round-structure | - |
| 5 | `fl4:d5_m3k2_mp_pk` | 82,837,245 | 231,431 | 39.9x / 8.8x | 0.928 / 1.007 | yes | first-last | - |
| 5 | `xo:d5_m3k2_mp_pk` | 82,837,245 | 231,431 | 39.9x / 8.8x | 0.928 / 1.007 | yes | optimizer | - |
| 5 | `rs4:d5_m3k2_xp` | 84,726,010 | 226,445 | 39.0x / 9.0x | 0.949 / 0.985 | yes | round-structure | - |
| 5 | `xo:d5_m3k2_pk` | 84,726,010 | 226,445 | 39.0x / 9.0x | 0.949 / 0.985 | yes | optimizer | - |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_th_sl` | 57,250,043 | 192,041 | 57.7x / 10.6x | 0.825 / 1.081 | yes | combined-2 | 6.6e-03 |
| 6 | `c2:d6_m3_mp_cp_z11_lz5_th_sl` | 57,579,734 | 191,879 | 57.4x / 10.6x | 0.830 / 1.080 | yes | combined-2 | 4.8e-03 |
| 6 | `c2:d6_m3_mp_cp_z11_lz5_th` | 58,406,978 | 187,237 | 56.6x / 10.9x | 0.842 / 1.054 | yes | combined-2 | 4.8e-03 |
| 6 | `xp3:lazy4c_middle_m4_mp_pc54` | 60,639,235 | 184,393 | 54.5x / 11.0x | 0.874 / 1.038 | yes | packing | - |
| 6 | `fl:d6_col` | 61,027,773 | 184,231 | 54.2x / 11.0x | 0.880 / 1.037 | yes | first-last | - |
| 6 | `xp3:lazy4c_middle_m4_mp_pc4` | 61,354,467 | 180,249 | 53.9x / 11.3x | 0.884 / 1.015 | yes | packing | - |
| 6 | `xp3:lazy4c_middle_m3_mp_pc` | 61,679,613 | 180,087 | 53.6x / 11.3x | 0.889 / 1.014 | yes | packing | - |
| 6 | `fl:d6_col_nop5` | 61,679,613 | 180,087 | 53.6x / 11.3x | 0.889 / 1.014 | yes | first-last | - |
| 6 | `xs3:lazy4c_middle_m3_mp_cp` | 61,679,613 | 180,087 | 53.6x / 11.3x | 0.889 / 1.014 | yes | tricks-critic | - |
| 6 | `fl:d6_col_m2` | 62,387,275 | 178,681 | 53.0x / 11.4x | 0.899 / 1.006 | yes | first-last | - |
| 6 | `c2:d6_cp_c5` | 63,564,360 | 170,175 | 52.0x / 12.0x | 0.916 / 0.958 | yes | combined-2 | 4.8e-04 |
| 6 | `xs3:lazy_middle_cp` | 69,116,552 | 169,151 | 47.8x / 12.0x | 0.996 / 0.952 | yes | tricks-critic | - |
| 6 | `rs:lazy_middle_xp` | 70,397,992 | 168,175 | 47.0x / 12.1x | 1.015 / 0.947 | yes | round-structure | - |
| 7 | `c2:d7_m4s4_mp_cp_u1_z11_lz5_sl` | 49,375,781 | 136,618 | 66.9x / 14.9x | 0.776 / 1.072 | yes | combined-2 | 5.0e-03 |
| 7 | `c2:d7_m3_mp_cp_u1_z11_lz5_sl` | 49,705,472 | 136,456 | 66.5x / 14.9x | 0.781 / 1.070 | yes | combined-2 | 3.5e-03 |
| 7 | `c2:d7_m3_mp_cp_sp_z11_lz5_sl` | 50,112,984 | 135,640 | 66.0x / 15.0x | 0.787 / 1.064 | yes | combined-2 | 2.8e-03 |
| 7 | `c2:d7_m3_mp_cp_sp_z11_lz5` | 50,940,228 | 130,998 | 64.9x / 15.5x | 0.800 / 1.028 | yes | combined-2 | 2.8e-03 |
| 7 | `xs3:split_first_lazy4c_m3_mp_cp_sp` | 52,827,268 | 129,094 | 62.6x / 15.8x | 0.830 / 1.013 | yes | tricks-critic | - |
| 7 | `c2:d7_cp_u1_c5` | 53,163,143 | 123,702 | 62.2x / 16.4x | 0.835 / 0.970 | yes | combined-2 | 8.1e-04 |
| 7 | `c2:d7_cp_sp_c5` | 53,652,255 | 122,886 | 61.6x / 16.6x | 0.843 / 0.964 | yes | combined-2 | 9.8e-04 |
| 7 | `rs:split_first_lazy4c_xpg_x1s` | 58,913,522 | 120,488 | 56.1x / 16.9x | 0.926 / 0.945 | yes | round-structure | - |
| 7 | `rs:split_first_lazy_xpg_x1s` | 63,754,034 | 115,944 | 51.8x / 17.5x | 1.002 / 0.910 | yes | round-structure | - |
| 7 | `sp:lazy2_l3` | 70,156,703 | 114,454 | 47.1x / 17.8x | 1.102 / 0.898 | yes | sparse-focus | - |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl` | 45,071,004 | 115,841 | 73.3x / 17.6x | 0.738 / 1.062 | yes | combined-2 | 8.6e-03 |
| 8 | `c2:d8_rp_m3_mp_cp_u1_sl` | 45,521,248 | 115,755 | 72.6x / 17.6x | 0.745 / 1.061 | yes | combined-2 | 8.5e-03 |
| 8 | `c2:d8_rp_m3_mp_cp_u1` | 45,892,300 | 114,985 | 72.0x / 17.7x | 0.751 / 1.054 | yes | combined-2 | 8.5e-03 |
| 8 | `c2:d8_rp_m3_mp_cp_sp_sl` | 45,928,760 | 114,939 | 72.0x / 17.7x | 0.752 / 1.054 | yes | combined-2 | 9.6e-03 |
| 8 | `c2:d8_rp_m3_mp_cp_sp` | 46,299,812 | 114,169 | 71.4x / 17.8x | 0.758 / 1.046 | yes | combined-2 | 9.6e-03 |
| 8 | `c2:d8_rp_m3_cp_sp_sl` | 46,757,765 | 113,657 | 70.7x / 17.9x | 0.765 / 1.042 | yes | combined-2 | 6.4e-03 |
| 8 | `c2:d8_rp_cp_u1_sl` | 46,805,745 | 113,215 | 70.6x / 18.0x | 0.766 / 1.038 | yes | combined-2 | 8.5e-03 |
| 8 | `c2:d8_rp_cp_sp` | 47,598,310 | 111,629 | 69.4x / 18.2x | 0.779 / 1.023 | yes | combined-2 | 6.6e-03 |
| 8 | `xo:split_first_middle_mp_pkud` | 49,708,462 | 109,285 | 66.5x / 18.6x | 0.814 / 1.002 | yes | optimizer | - |
| 8 | `xo:split_first_middle_pkud_mc` | 49,932,667 | 107,683 | 66.2x / 18.9x | 0.817 / 0.987 | yes | optimizer(flagged-margin) | - |
| 8 | `rs:split_first_middle_xpyd_x1p_x2pc` | 51,355,587 | 107,259 | 64.4x / 19.0x | 0.841 / 0.983 | yes | round-structure | - |
| 8 | `rs:split_first_middle_mp_xpydg3_x1s_gm` | 53,712,700 | 105,657 | 61.5x / 19.3x | 0.879 / 0.968 | yes | round-structure | - |
| 8 | `rs:split_first_middle_mp_xpg_x1s` | 54,686,660 | 101,609 | 60.4x / 20.0x | 0.895 / 0.931 | yes | round-structure | - |
| 8 | `rs:split_first_middle_xpg_x1s` | 55,644,185 | 100,327 | 59.4x / 20.3x | 0.911 / 0.920 | yes | round-structure | - |
| 8 | `rs:split_first_middle_mp_xpg_x12sc` | 56,066,324 | 98,617 | 59.0x / 20.6x | 0.918 / 0.904 | yes | round-structure | - |
| 8 | `rs:split_first_middle_xpg_x12sc` | 57,023,849 | 97,335 | 58.0x / 20.9x | 0.934 / 0.892 | yes | round-structure | - |
| 8 | `sp:base_mp` | 63,204,510 | 97,237 | 52.3x / 20.9x | 1.035 / 0.891 | yes | sparse-focus | - |
| 8 | `sp:base` | 64,162,035 | 95,955 | 51.5x / 21.2x | 1.050 / 0.880 | yes | sparse-focus | - |
| 9 | `sp:col1b_x2e` | 66,111,490 | 94,658 | 50.0x / 21.5x | - | yes | sparse-focus | - |
| 9 | `sp:col1b` | 68,233,410 | 91,778 | 48.4x / 22.2x | - | yes | sparse-focus, re-verified | 5.0e-05 |

## 3. What each combined point uses (and where each trick comes from)

Every trick is an identity on the lattice of exact bits and integer counts. Knots sit on lattice
points, except where marked "min-parity" (knots between integers). The avenue NOTES.md files
hold the exactness arguments and the searches.

### Layouts

Round kinds: "split" = X (E = a + D) then Y ([E == 1]); "lazy4/lazy4c" = X then a lazy chi
whose counts are congruent mod 2.

| depth | layers | source |
|---|---|---|
| 8 | X1 \| Y1 \| chi1 \| X2 \| lazy4 chi \| **fold** \| parity \| chi3 | units-per-bit (RP) |
| 7 | X1 \| Y1 \| chi1 \| X2 \| lazy4c chi \| theta3 \| chi3 | wave-1 `split_first_lazy4c` |
| 6 | theta1 \| chi1 \| X2 \| lazy4c chi \| theta3 \| chi3 | wave-1 `lazy4c_middle` |
| 5 | theta1 \| chi1 \| X2 \| lazy chi with column parities \| Walsh round 3 | wave-1 xs4 d5 |
| 4 | fused lazy round 1 on raw bits \| X2 recovering [o' == 1] \| lazy chi \| Walsh | depth-low `d4x` |
| 3 | three fused rounds with centred raw parities | depth-low `d3c1_mp17` |

### Tricks in the combined points

| flag | trick | found by | used in (combined-2) |
|---|---|---|---|
| `cp` | The chi layer before an X layer emits 3 free linear features per column: c = sum a_y, p1 = a0 + 2a1, p2 = a2 + 2a3. X decodes them with the 5 units the copies cost: b = relu(p-1)(4-p)/2, a = relu(p) - 2b, a4 = (c - p1 - p2) + b1 + b2. D gates on c(x-1,z) + c(x+1,z+1). The X input drops from 1921 to 961. | tricks-critic `cp` = packing `pa`+`pc` = first-last `pack_col` (optimizer and round-structure found the pairs-only part) | d5-d8 |
| `u1` | Round-1 u pairs. X1 emits u = (a1 + 2a2) - 3D per same-column pair of live message bits (1 unit instead of 2 copies), and u3 = 2(a1 + 2a2) - 7D, which folds in the constant-own (capacity) positions. Y1 decodes with 4 or 5 lattice-knot units. X1 has 641 outputs instead of 1465. | optimizer `upair1` + `upair1d` (ported as `UP1`) | d7, d8 |
| `sp` | Alternative to u1: message-bit pairs + D, 3 units per pair in Y1. | tricks-critic `sp` = round-structure `x1pairs` | ablations |
| `rp` | Fold layer r(s) = s - 2 max(0, s-11) + 2 max(0, s-22) - ... into [0, 11] (4 units per count), then MIN_PARITY[11] (5 units). The lazy chi needs no reduction. | units-per-bit `RP` | d8 |
| `z11` | One zigzag per lazy count, z(L) = L - 2 max(0, L - 11). Counts lie in [0, 33], and odd 33 lets MPC save a unit. | units-per-bit | d6, d7 |
| `c5` | One reduction unit per column (window 5, MILP-minimal); fewer nonzeros. | units-per-bit | sparse-leaning d6, d7 |
| `mpc` | Min-parity on odd count ranges, (n-1)/2 units (minpar_counts.py): the parity layer (n = 11), last theta (n = 33) and Walsh singles/triples (n = 13, 39). | units-per-bit `MPC` = first-last `mp_last` | d5-d8. `_g` = without it in the d8 parity layer |
| `wmpc` | The same in depth-low's Walsh layer. | combined-2 (port) | d4 |
| `lz5` | The lazy digest-2 values travel two per feature, F = o0 + 5 o1, with a 12-unit exact decoder. | units-per-bit `LZ5` = first-last/packing `d2p5` | d6, d7 |
| `th` | Column-shared round-1 theta: 3 units per bit + 2 per column, for T <= 7. | units-per-bit `TH1S` | d5, d6 |
| `sl` | The last theta layer emits, per digest bit, the chi gate s = 2a - b + c (224 features instead of 320). | optimizer `s_last` = first-last `gate_out` = packing `lf` = round-structure `g3` (ported as `SLAST`) | d6-d8 |
| `m3` / `m4s4` | Digest 1 at 3 bits per feature (combined-1), or at 4 bits split into pairs in the last theta by a 6-unit DP staircase. | combined-1 / packing `s4` (ported as `S4`) | d5-d8 |
| `mp` | Min-parity (MINPAR7/9) on raw message bits in round 1. | wave 1 | d5-d8 |
| `dd` | One Y1 output per distinct E node. | tricks-critic `dd` = first-last/optimizer `dedupe_y` = packing `dy` | d7, d8 |
| `mpx` | d4x's min-parity for every odd raw set of 7-17 bits. | depth-low | d4 |

### Variant names

- `c2:d8_rp_m4s4_mp_cp_u1_sl` = u1 + mp in round 1, cp into X2, lazy4, fold, MPC parity with
  `sl`, digest 1 m4s4.
- `c2:d7_m4s4_mp_cp_u1_z11_lz5_sl` = u1 + mp, cp, lazy4c with Z11 + MPC, LZ5, `sl`, m4s4.
- `c2:d6_m4s4_mp_cp_z11_lz5_th_sl` = TH1S + mp theta1, cp, Z11 + MPC, LZ5, `sl`, m4s4.
- `xc:d5_m4k2_mp_cp_mpc_th` (and `_m3k2_`) = TH1S + mp theta1, cp, xs4's lazy chi with
  column parities, Walsh with MPC, digest 1 at 4 (or 3) bits per feature, digest 2 as lazy-value
  pairs.
- `dl3:d4x_m4k2_mpx_wmpc` = depth-low d4x + MPC in the Walsh layer.

### Ablations measured here (depth 8, dense / sparse)

The ablations are cumulative along each path and read as a chain of +/- steps:
- **Base** (RP + sp + cp + mp + m3): 46.30M / 114.2K.
- **Adding `sl`**: -0.37M.
- **u1 instead of sp**: -0.41M, reaching 45.52M.
- **m4 without s4**: +0.20M.
- **m4 with s4**: -0.45M, reaching 45.07M.
- **Without mp**: +0.8M to +1.3M, and -1.3K to -2.5K sparse.
- **Without MPC in the parity layer (`_g`)**: +0.46M, with the margin 8.6e-3 -> 5.0e-3.

At depth 7, from the tricks-critic point `split_first_lazy4c_m3_mp_cp_sp` (52.83M):
- Z11 + LZ5: -1.89M (units-per-bit measured them separately at -1.27M and -0.62M);
- `sl`: -0.83M;
- u1: -0.41M;
- m4s4: -0.33M.

That reaches 49.38M.

## 4. Where the dense goes now (per layer, [in, hidden, out] dense in M)

| layer | wave-1 d8 `split_first_middle_mp` (61.08M) | **d8 `c2:d8_rp_m4s4_mp_cp_u1_sl` (45.07M)** | d7 `c2:d7_m4s4_..._sl` (49.38M) | d6 `c2:d6_m4s4_..._sl` (57.25M) |
|---|---|---|---|---|
| 1 | X1 [1145,2163,1465] 8.12 | X1 (u1) [1145,1659,641] 4.86 | X1 (u1) 4.86 | theta1 (TH1S) [1145,5202,1465] 19.53 |
| 2 | Y1 [1465,1466,1601] 6.64 | Y1 [641,2474,1465] 6.80 | Y1 6.80 | chi1 [1465,1602,961] 6.23 |
| 3 | chi1 [1601,1602,1921] 8.21 | chi1 (cp out) [1465,1602,961] 6.23 | chi1 6.23 | X2 [961,3203,1657] 11.46 |
| 4 | X2 [1921,3202,1713] 17.79 | X2 [961,3203,1657] 11.46 | X2 11.46 | lazy4c (Z11) [1657,2410,489] 9.17 |
| 5 | Y2 [1713,1714,1713] 8.81 | lazy4 [1657,2202,601] 8.62 | lazy4c (Z11) [1657,2410,489] 9.17 | theta3 [489,6859,449] 9.79 |
| 6 | chi2 [1713,1714,545] 6.81 | fold [601,1786,489] 3.02 | theta3 [489,6859,449] 9.79 | chi3 [449,675,673] 1.06 |
| 7 | theta3 [545,2146,545] 3.51 | parity (MPC, sl, s4 split) [489,2107,449] 3.01 | chi3 [449,675,673] 1.06 | |
| 8 | chi3 [545,675,673] 1.19 | chi3 [449,675,673] 1.06 | | |

Other headline points:
- d5 `xc:d5_m3k2_mp_cp_mpc_th`: theta1 19.53 | chi1 6.23 | X2 11.53 | lazy chi cols 20.68 |
  Walsh 21.44.
- d4 `dl3:d4x_m4k2_mpx_wmpc`: L1 69.19 | X2 23.34 | lazy chi 20.30 | Walsh 21.83.

The d8 total is 45.07M.

## 5. Lower bounds and headroom, across the avenues

### 5.1 What is proven or exhaustively checked

**(a) Unconditional (tricks-critic).**
- The last layer needs at least 673 units: the 673 output functions have real rank 673 over
  1500 random messages, with sigma_min 1.13.
- Nothing else about widths is unconditional. One real feature can carry any number of bits,
  so any floor needs a model of the features.

**(b) Lattice model (features are integers up to scale; integer gates and knots).**
- A single unit relu(g)v, with g and v affine, cannot compute from p = t + 2s a function that
  depends on t but not on s (tricks-critic lemma, proved; the base-3 version holds too).
- So a reader with one unit per output bit needs its inputs unpacked:
  - chi layers (gate 2a - b + c);
  - Y layers ([E == 1]);
  - lazy-chi products (Eb, Ec linear).
- This pins chi1 >= 1464 inputs and X2 -> lazy >= 1600 E features. Packing pays only into
  layers that consume bits linearly, which is exactly where cp and u1 act.
- Every interface of the best designs is full rank for its readers (packing, `ranks.py`).

**(c) Unit floors.** Established by exact search. Each is a floor only for its unit form, with
integer or small-rational knots.

| block | floor | how established |
|---|---|---|
| chi on exact bits | 1 unit per bit | rank + degree argument |
| parity of a count range n | floor(n/2) units | exhaustive for n <= 11. Range 10 (D) needs 5; 4 is impossible |
| u-pair decode | 4 units | no 3-unit form (optimizer) |
| sp pair decode | 3 units | 2 impossible over 120K gates |
| lazy chi on E codes | 1 product per chi bit | one-unit double inner product (IP2) infeasible at any window (MILP, units-per-bit) |
| one-unit mod-2 double AND on E | none | none with abs(w) <= 3, abs(b) <= 9 (first-last) |
| round-1 theta | 3 units per bit + 2 per column | 1 shared affine unit infeasible |
| pair decoders | pairs are the only free packing | triples need 8 units for 3 bits |
| digest packing | about 4 bits per feature | 6-bit packing fails float32 (0.026-0.030) |

**(d) Depth (depth-low).**
- No single layer can hold a theta after a chi, so depth 2 is out and depth 3 is forced to three
  fused rounds.
- A fused round needs a pair parity per chi bit. Exact search over count-symmetric gates finds
  no cheaper AND of two parities for 3+3 and 4+4 bit sets.
- Family floors: depth 4 about 116M, depth 3 about 231M.

### 5.2 Headroom per depth (dense), current against my estimated floor for these layouts

| depth | now | estimated floor | what the rest is |
|---|---|---|---|
| 8 | 45.07M | ~43.5M | See below. |
| 7 | 49.38M | ~47.5M | theta3 on range-33 counts (16 units per bit on a 489-wide input, 9.8M) is the price of the missing Y layer. A 2-unit width-2 double AND (unsearched, first-last) would give about -1.9M. |
| 6 | 57.25M | ~55.5M | theta1 on raw bits (19.5M) is at its TH1S minimum except for the T = 8 columns (about -0.7M, no form found). |
| 5 | 79.41M | ~77M | Walsh (21.4M, three multi-count parities per digest bit, forced by chi's Fourier support) and xs4's lazy round (20.7M). |
| 4 | 134.66M | ~116M | See below. |
| 3 | 253.8M | ~231M | L2 (fused round 2 on counts <= 13, 122.6M) is at its unit floor. |

Depth 8 in detail:
- Every full-width layer is at its unit floor:
  - chi1: 1600 units, 1465 inputs pinned by (b), 961 outputs = 3 per column;
  - X2: 1600 decode + 1600 D units, 1657 outputs pinned;
  - lazy4: 1600 products + 320 pass units.
- The slack is:
  - the digest carries: digest 1 crosses X2, lazy4, fold and parity; digest 2's 224 lazy
    values cross the fold. About 2M in all;
  - MPC in the parity layer;
  - the u1/Y1 decode.

Depth 4 in detail:
- L1 is at the pair-parity floor.
- The 116M family floor needs a cheaper exact AND of two parities to go lower. Search over
  asymmetric gates is still open; a restriction argument allows about 4 units against 8 now,
  about -24M.

**Wave 1's "about 50M" floor is broken at depths 7 and 8.** The argument ("every state layer is
about 3 x 1600^2") assumed that every layer reads one feature per state bit. Two layers now
read far fewer:
- X2 reads 961 features for 1600 bits. X2 consumes the state linearly, so the decode costs no
  extra units.
- Y1 reads 641 features for 1464 theta bits. It spends 1.7 units per bit, and that pays because
  X1 emits less than half as many features.

The layout also changed. At depth 8 the fold layer replaces the Y2 + chi2 + theta3 route. It
carries the state through a lazy chi, whose products need no exact chi2 bits.

**Sparse.**
- Best points: depth 8 95,955 (`sp:base`) and depth 9 91,778 (`sp:col1b`).
- sparse-focus's per-bit accounting puts the floor of this family at about 85-88K: about 22-25
  nonzeros per state bit per round. Norms, copies and Keccak's fan-out (3 chi bits per theta
  bit, 11 bits per theta) make up about 9 of those, and no untied, residual-free layout can
  avoid them.
- Packing costs sparse: decoded pairs need 2-5 wo entries, and min-parity values read all
  their inputs. The dense-best depth-8 point has 115.8K nonzeros, +6% over wave 1's depth-8
  point and +21% over the sparse-best depth-8 point.

**2x / 100x.**
- 100x the baseline is 33.1M. That needs about -27% at depth 8.
- Every full-width layer is at its floor for the known unit forms. Getting there needs one of:
  - fewer units per bit in X2 (D) or in the lazy layer;
  - a Y/chi reader that takes packed inputs, which the lemma rules out at one unit per bit;
  - an architecture change (residual or tied weights), which is outside the fixed
    architecture.
- My estimate for this architecture is about 43M dense at depth 8 (77x) and about 88K sparse.

### 5.3 Depths 9 and 10

- No depth-9 or depth-10 layout beats 45.07M dense. An extra layer pays only if it lets a wide
  layer shrink by more than its own cost: about 0.7M at the narrowest (489) boundary, 7.7M at
  full width. The avenues found none:
  - optimizer: no depth-9 layer whose split pays;
  - first-last: packing the first layer's input costs 2.8M to save 1.1M;
  - sparse-focus: col1 is +4.1M dense.
- The only deeper Pareto points are sparse ones:
  - `sp:col1b`, depth 9: round 1 from column parities; each message bit is read once;
  - `sp:col1b_x2e`, depth 9.
- A depth-10 point would need round 2 from column parities as well. sparse-focus priced that
  at +1.8K sparse.

## 6. Float precision, steepness, bugs

**Steepness.**
- Everything runs at the base's c = 4, q = 8. The sizes do not depend on q.
- The margins come from float32 rounding, not from silu:
  - first-last: q = 16 leaves the depth-8 margin unchanged;
  - depth-low: float64 with the same weights is exact to 1.2e-6 for depth 3.
- So a sharper steepness buys nothing here, and I did not change it.

**What drives the margins.**
- Min-parity units have knots between integers, so their slope at the lattice points
  amplifies the float32 error of their inputs. This covers MPC on counts and mp on raw bits
  through the u1/sp D gates.
- Rule used here: min-parity on exact raw bits is safe. Min-parity on counts is safe at q = 8
  up to about 8.6e-3 worst margin at depth 8.
- Where that matters, `_g` variants replace it at +0.46M.
- depth-low's centred parity construction fixes the depth-3 glu_xor cancellation (terms of
  size n^2 at n = 35-41).
- The 6-bit digest packing (packing's `s6`) fails float32 (0.026-0.030) and is not used.

**Bugs fixed in `repo/`.**
- The tracer's PY_UNWIND stack imbalance (units-per-bit): a function that exits by raising
  broke `Tracer.root`. The fix is in `monitor.py`, with a regression test in
  `tests/tracer_unwind_test.py`.
- The Y1 duplicate outputs (`dd`).
- depth-low's compiler speed-ups (`glu` sums only nonzero weights; tree fold and
  `layer_to_units` skip no-ops) leave the compiled weights identical. `validate_bench` gives
  bit-equal weights, and the test suite passes.

## 7. Reproduce, files

Set up the environment:

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
C=$S/xof2/av/combined-2
E=$C/repo/experiments/xof_shrink
```

Run the harness plus both audits (writes `runs/<name>.{h,a777,a63}.json`):

```bash
$C/run.sh c2:d8_rp_m4s4_mp_cp_u1_sl
```

Or run the harness by hand:

```bash
PYTHONPATH=$C/repo/src:$E:$E/depth_low/lib/python $S/venv/bin/python $E/xofbench.py \
  --log-w 6 --depth 3 --variant c2:d8_rp_m4s4_mp_cp_u1_sl --widths
```

Rebuild the results and the table:

```bash
python3 $C/collect.py    # rebuilds results.jsonl
python3 $C/pareto.py     # the table in section 2
```

Files:

- `repo/`: the wave-2 base plus every avenue patch (`patch.diff` = `git -C repo diff`,
  including new files):
  - **Merged builders:** `xs3.py`, `xs4.py` and `xc.py` are the 3-way merge of tricks-critic
    (cp, sp, dd, m2) and units-per-bit (NOP, MPC, C5, Z11, LZ5, RP, TH1S). combined-2 then
    added the switches `UP1` (optimizer's round-1 u pairs), `SLAST` (s_last) and `S4`
    (packing's s4). It also added `GF1` (g-fold after the u1 Y1), which is measured and loses;
    see NEXT.md item 6.
  - **Combined variants:** `c2.py`.
  - **Avenue builders, new files, unchanged:** `xp3.py`, `xp4.py` (packing); `rs.py`, `rs4.py`
    (round-structure); `fl.py`, `fl4.py` (first-last); `sp.py` (sparse-focus);
    `depth_low/lib/python/dl3.py` (depth-low, plus the `WMPC` switch added here).
  - **Optimizer overlay:** `overlay_opt/` holds the optimizer's own `xs3.py`, `xs4.py`, `xo.py`
    and `cost_model.py`. Run it with `run.sh xo:<v> overlay_opt`.
  - **Caveat:** the switches are module globals set by the variant wrappers. Build one variant
    per process, as the harness and the audit do.
- `runs/`:
  - raw harness and audit JSON;
  - `*.val.txt` (validate_bench) and `*.lw4/lw5.json` (other log_w);
  - `*.stress256.json`;
  - `pytest.txt`.
- `search/d2d_milp.py`: an exact MILP for the column-pair parity D with 2-D gates. See
  NEXT.md.
- `avenue_points.txt`: every verified avenue point used in section 2.

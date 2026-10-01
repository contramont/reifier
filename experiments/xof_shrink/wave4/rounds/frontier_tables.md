
**log_w 4, 1 step x 2 rounds (T = 2)** - threshold baseline: depth 14, dense 131,240,010, sparse 334,762

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 4 | `rv:T2_direct_ra_sl_mpc` | 1,450,906 | 25,064 | 3.5 / 90.5 / 13.4 | 7.3e-04 / 1.3e-03 / 1.2e-03 | not run |
| 5 | `rv:T2_split_u1_ra_sl_mpc` | 1,124,457 | 14,439 | 2.8 / 116.7 / 23.2 | 9.7e-04 / 1.7e-03 / 2.4e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 4 | `rv:W_alld_sl_m4` | 1,792,399 | 25,582 | 73.2 | 9.3e-05 | 9.4e-05 |
| 5 | `rv:W_T2_split_sl` | 1,379,950 | 14,621 | 95.1 | 2.4e-04 | 2.5e-04 |

**log_w 6, 1 step x 2 rounds (T = 2)** - threshold baseline: depth 14, dense 2,097,727,098, sparse 1,338,874

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 4 | `rv:T2_direct_ra_sl_mpc` | 24,097,954 | 103,496 | 3.5 / 87.1 / 12.9 | 9.3e-04 / 9.3e-04 / 1.2e-03 | not run |
| 5 | `rv:T2_split_u1_ra_sl_mpc` | 18,248,721 | 58,167 | 2.8 / 115.0 / 23.0 | 2.3e-03 / 3.1e-03 / 4.1e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 4 | `rv:W_alld_sl_m4` | 29,838,919 | 105,622 | 70.3 | 3.3e-04 | 2.1e-04 |
| 5 | `rv:W_T2_split_sl` | 22,555,174 | 58,901 | 93.0 | 4.0e-04 | 2.7e-04 |

**log_w 4, 1 step x 3 rounds (T = 3)** - threshold baseline: depth 20, dense 193,413,652, sparse 498,804

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 2,843,243 | 40,651 | 3.3 / 68.0 / 12.3 | 1.0e-03 / 8.9e-04 / 1.6e-03 | not run |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` | 2,516,794 | 30,026 | 2.9 / 76.8 / 16.6 | 1.6e-03 / 3.3e-03 / 5.1e-03 | not run |
| 8 | `rv:F_rp_nop_ra_u1_cp2_p1` | 2,349,281 | 26,310 | 2.5 / 82.3 / 19.0 | 1.3e-04 / 6.2e-04 / 5.2e-04 | not run |

**log_w 4, 1 step x 4 rounds (T = 4)** - threshold baseline: depth 26, dense 255,587,294, sparse 662,846

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 8 | `rv:F_alld_ra` | 8,216,454 | 60,557 | 3.2 / 31.1 / 10.9 | 2.6e-05 / 5.5e-05 / 2.2e-05 | not run |
| 9 | `rv:F_d1_rp_nop_ra_cp2` | 6,120,384 | 54,704 | 2.9 / 41.8 / 12.1 | 2.6e-05 / 4.4e-05 / 4.1e-05 | not run |
| 10 | `rv:F_l4c_z11_nop_ra_u1_cp2` | 4,520,655 | 45,359 | 2.6 / 56.5 / 14.6 | 2.6e-04 / 6.4e-04 / 7.5e-04 | not run |
| 11 | `rv:F_rp_ra_u1_cpall` | 4,032,022 | 38,326 | 2.4 / 63.4 / 17.3 | 2.8e-04 / 1.1e-03 / 2.6e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 8 | `rv:W_alld_sl_m4` | 8,540,427 | 60,907 | 29.9 | 2.7e-05 | 4.2e-05 |
| 9 | `rv:W_d1_rp_nop_sl_m4` | 6,765,797 | 57,934 | 37.8 | 4.6e-05 | 6.7e-05 |
| 11 | `rv:W_rp_nop_sl_m4` | 4,912,555 | 44,100 | 52.0 | 4.5e-05 | 4.8e-05 |

**log_w 6, 1 step x 4 rounds (T = 4)** - threshold baseline: depth 26, dense 4,085,035,982, sparse 2,650,958

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 8 | `rv:F_alld_ra` | 132,007,422 | 245,247 | 3.2 / 30.9 / 10.8 | 2.1e-05 / 3.2e-05 / 3.4e-05 | not run |
| 9 | `rv:F_d1_rp_nop_ra_cp2` | 98,509,512 | 221,834 | 2.9 / 41.5 / 12.0 | 3.0e-05 / 5.7e-05 / 5.1e-05 | not run |
| 10 | `rv:F_l4c_z11_nop_ra_u1_cp2` | 72,283,959 | 181,625 | 2.6 / 56.5 / 14.6 | 3.7e-04 / 1.2e-03 / 9.0e-04 | not run |
| 11 | `rv:F_rp_ra_u1_cpall` | 64,492,606 | 153,472 | 2.4 / 63.3 / 17.3 | 1.5e-03 / 8.0e-03 / 5.0e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 8 | `rv:W_alld_sl_m4` | 137,470,947 | 246,701 | 29.7 | 3.2e-05 | 4.5e-05 |
| 11 | `rv:W_rp_nop_sl_m4_cp2` | 73,646,099 | 165,054 | 55.5 | 8.1e-05 | 2.8e-04 |

**log_w 4, 3 steps x 2 rounds (T = 6)** - threshold baseline: depth 38, dense 408,416,850, sparse 1,009,970

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 12 | `rv:F_alld_ra` | 15,756,633 | 98,246 | 3.2 / 25.9 / 10.3 | 1.9e-04 / 3.0e-04 / 5.3e-04 | fails (103425 wrong) |
| 15 | `rv:F_d1_rp_nop_ra` | 10,851,049 | 89,619 | 2.5 / 37.6 / 11.3 | 2.9e-04 / 5.2e-04 / 3.8e-04 | not run |
| 16 | `rv:F_d0_rp_nop_ra_cp2` | 9,082,096 | 83,922 | 2.4 / 45.0 / 12.0 | 2.4e-03 / 4.8e-03 / 5.3e-03 | not run |
| 17 | `rv:F_rp_nop_ra_cp2_p1` | 8,852,967 | 73,139 | 2.2 / 46.1 / 13.8 | 1.9e-03 / 3.7e-03 / 7.0e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 12 | `rv:W_alld_sl_m4` | 16,080,606 | 98,596 | 25.4 | 5.4e-04 | 5.4e-04 |
| 16 | `rv:W_d0_rp_nop_sl_m4` | 9,734,229 | 87,096 | 42.0 | 7.7e-04 | 8.2e-04 |
| 17 | `rv:W_rp_nop_sl_m4_cp2` | 8,993,620 | 73,311 | 45.4 | 6.8e-03 | 7.0e-03 |

**log_w 6, 3 steps x 2 rounds (T = 6)** - threshold baseline: depth 38, dense 6,527,735,970, sparse 4,039,202

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 12 | `rv:F_alld_ra` | 252,273,501 | 395,190 | 3.2 / 25.9 / 10.2 | 2.5e-04 / 7.3e-04 / 5.0e-04 | not run |
| 14 | `rv:F_d2_rp_nop_ra` | 198,118,638 | 372,206 | 2.7 / 32.9 / 10.9 | 2.6e-04 / 7.3e-04 / 5.0e-04 | not run |
| 15 | `rv:F_d1_rp_nop_ra` | 173,821,045 | 360,693 | 2.5 / 37.6 / 11.2 | 3.1e-04 / 6.4e-04 / 1.7e-03 | not run |
| 16 | `rv:F_d0_rp_nop_ra_cp2` | 145,544,572 | 337,884 | 2.4 / 44.9 / 12.0 | 1.6e-03 / 4.1e-03 / 4.9e-03 | not run |
| 17 | `rv:F_rp_nop_ra_cp2_p1` | 141,357,507 | 291,943 | 2.2 / 46.2 / 13.8 | 1.7e-03 / 4.9e-03 / 4.8e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 16 | `rv:W_d0_rp_nop_sl_m4` | 156,241,377 | 350,634 | 41.8 | 9.8e-04 | 3.2e-03 |
| 17 | `rv:W_rp_nop_sl_m4_cp2` | 143,724,352 | 292,617 | 45.4 | 6.6e-03 | 6.9e-03 |

**log_w 4, 3 steps x 3 rounds (T = 9)** - threshold baseline: depth 56, dense 610,030,560, sparse 1,511,168

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 18 | `rv:F_alld_ra` | 26,338,995 | 151,523 | 3.1 / 23.2 / 10.0 | 1.5e-04 / 2.5e-04 / 3.4e-04 | not run |
| 23 | `rv:F_d2_rp_nop_ra` | 18,454,413 | 137,338 | 2.4 / 33.1 / 11.0 | 3.6e-04 / 6.8e-04 / 6.1e-04 | not run |
| 24 | `rv:F_d1_rp_nop_ra` | 17,013,620 | 134,465 | 2.3 / 35.9 / 11.2 | 6.4e-04 / 1.2e-03 / 1.1e-03 | not run |
| 25 | `rv:F_l4c_z11_nop_ra_u1_cp2` | 15,139,099 | 122,128 | 2.2 / 40.3 / 12.4 | 4.9e-03 / 4.7e-03 / 4.7e-03 | not run |
| 26 | `rv:F_rp_nop_ra_cp2` | 15,083,938 | 118,031 | 2.2 / 40.4 / 12.8 | 9.1e-04 / 1.4e-03 / 2.6e-03 | not run |

**log_w 4, 3 steps x 4 rounds (T = 12)** - threshold baseline: depth 74, dense 811,644,270, sparse 2,012,366

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 24 | `rv:F_alld_ra` | 36,921,357 | 204,893 | 3.1 / 22.0 / 9.8 | 2.3e-04 / 3.1e-04 / 2.0e-04 | not run |
| 30 | `rv:F_d4_rp_nop_ra` | 27,576,942 | 188,003 | 2.5 / 29.4 / 10.7 | 4.9e-04 / 6.1e-04 / 6.3e-04 | not run |
| 32 | `rv:F_d2_rp_nop_ra` | 24,616,956 | 182,257 | 2.3 / 33.0 / 11.0 | 1.0e-03 / 1.0e-03 / 1.1e-03 | not run |
| 33 | `rv:F_d1_rp_nop_ra` | 23,176,163 | 179,384 | 2.2 / 35.0 / 11.2 | 8.6e-04 / 2.5e-03 / 1.9e-03 | not run |
| 34 | `rv:F_d0_rp_nop_ra_cp2` | 21,400,490 | 173,631 | 2.2 / 37.9 / 11.6 | 1.3e-03 / 2.2e-03 / 2.9e-03 | not run |
| 35 | `rv:F_rp_nop_ra_cp2_p1` | 21,169,121 | 162,859 | 2.1 / 38.3 / 12.4 | 1.5e-03 / 6.1e-03 / 6.9e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 24 | `rv:W_alld_sl_m4` | 37,245,330 | 205,243 | 21.8 | 2.9e-04 | 4.3e-04 |
| 34 | `rv:W_d0_rp_nop_sl_m4` | 22,059,343 | 176,861 | 36.8 | 4.5e-03 | 2.2e-03 |
| 35 | `rv:W_rp_nop_sl_m4` | 21,646,894 | 165,900 | 37.5 | 4.9e-03 | 5.2e-03 |

**log_w 6, 3 steps x 4 rounds (T = 12)** - threshold baseline: depth 74, dense 12,972,317,214, sparse 8,048,030

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 24 | `rv:F_alld_ra` | 589,879,665 | 821,115 | 3.1 / 22.0 / 9.8 | 2.6e-04 / 6.6e-04 / 2.8e-04 | not run |
| 32 | `rv:F_d2_rp_nop_ra` | 393,083,196 | 730,423 | 2.3 / 33.0 / 11.0 | 1.0e-03 / 1.8e-03 / 2.7e-03 | not run |
| 33 | `rv:F_d1_rp_nop_ra` | 370,040,003 | 718,910 | 2.2 / 35.1 / 11.2 | 8.8e-04 / 3.4e-03 / 2.2e-03 | not run |
| 34 | `rv:F_d0_rp_nop_ra_cp2` | 341,656,010 | 695,877 | 2.2 / 38.0 / 11.6 | 1.6e-03 / 5.4e-03 / 7.7e-03 | not run |
| 35 | `rv:F_rp_nop_ra_cp2` | 338,667,985 | 650,300 | 2.1 / 38.3 / 12.4 | 1.4e-03 / 3.9e-03 / 6.2e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 35 | `rv:W_rp_nop_sl_m4` | 345,176,590 | 662,130 | 37.6 | 3.2e-03 | 5.6e-03 |

**log_w 3, 1 step x 24 rounds (T = 24)** - threshold baseline: depth 144, dense 374,839,360, sparse 1,968,720

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 48 | `rv:F_alld_ra` | 18,983,546 | 207,184 | 3.0 / 19.7 / 9.5 | 2.2e-05 / 2.3e-05 / 2.2e-05 | not run |
| 70 | `rv:F_l4c_z11_nop_ra_u1_cp2_yf2` | 12,069,591 | 179,379 | 2.1 / 31.1 / 11.0 | 2.3e-04 / 4.6e-04 / 3.5e-04 | not run |
| 71 | `rv:F_rp_nop_ra_u1_cp2_yf4` | 11,424,438 | 173,306 | 2.0 / 32.8 / 11.4 | 1.5e-04 / 1.3e-03 / 9.2e-04 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 48 | `rv:W_alld_sl_m4` | 19,058,863 | 207,350 | 19.7 | 2.8e-05 | 2.6e-05 |
| 71 | `rv:W_rp_nop_sl_m4_yf2` | 12,163,187 | 178,752 | 30.8 | 5.4e-05 | 4.6e-05 |

**log_w 3, 3 steps x 24 rounds (T = 72)** - threshold baseline: depth 428, dense 1,210,834,652, sparse 6,007,308

| depth | variant | dense | sparse | vs baseline (depth / dense / sparse) | worst f32 error (ref / mixed / dense) | bf16 weights (w16) |
|---|---|---|---|---|---|---|
| 144 | `rv:F_alld_ra` | 62,375,478 | 637,638 | 3.0 / 19.4 / 9.4 | 1.8e-04 / 2.5e-04 / 3.3e-04 | not run |
| 215 | `rv:F_rp_nop_ra_cp2_yf3` | 39,230,940 | 551,256 | 2.0 / 30.9 / 10.9 | 1.4e-03 / 1.0e-03 / 2.7e-03 | not run |

bf16-weight-correct (w16) points:

| depth | variant | dense | sparse | vs baseline dense | worst w16 error | worst f32 error |
|---|---|---|---|---|---|---|
| 215 | `rv:W_rp_nop_sl_m4_yf2` | 40,697,601 | 561,526 | 29.8 | 9.7e-04 | 6.1e-04 |

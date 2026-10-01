# theta1-t8 (wave 3): round-1 theta with a column pool, down to its floor

Item 4 of `xof2/av/combined-2/NEXT.md` (round-1 theta for the T = 8 columns).
- TH1S (3 units per bit + 2 per column) was exact only for T <= 7, so the x = 1 columns (T = 8)
  kept 4 units per bit.
- This avenue replaces TH1S with a different column-shared structure: the column pool. It gives:
  - **F1:** a form exact on {0,1} x [0, 8]: 16 units per 4-live column, for every column
    (T = 8 was 20, TH1S 17). It also has *fewer* nonzeros than TH1S.
  - **F2:** a form exact on {0,1} x [0, 7]: 15 units per 4-live column and 12 per 3-live column
    (TH1S 17 / 14). It uses the layer's shared constant unit, and it costs nonzeros.
  - **F3:** a point of the same solution family as F2 in which one per-bit value does not read T.
    Same units as F2, 6.4K fewer nonzeros, so F13 replaces F12.
  - **F13 = F3 where T <= 7, F1 where T = 8:** the round-1 layer drops from 5202 units (TH1S)
    to 4451. That is the floor of this structure (4448 theta units, counted below) plus the
    shared constant and 2 non-theta units that the layer has in every variant (F1: 4712 + 2 =
    4714). F12 (F2 + F1) has the same units and 6.4K more nonzeros.

Architecture unchanged: plain `MLP_SwiGLU`, no residuals, untied, attention = identity,
c = 4, q = 8.

**What counts as verified.** Every row below comes from `xofbench.py --log-w 6 --depth 3` plus
`audit/adv_check.py` on BOTH reference sets (`xof/final/ref777_w6.pt`, 69 messages;
`xof/av/combined-1/adv/ref_w6.pt`, 63 messages). All rows have `ok: true`, 0 wrong bits and an
empty eager mismatch in all three runs (`results.jsonl`; raw JSON in `runs/`).

## Result: new Pareto points (depths 4-6)

| depth | variant (repo in this dir) | dense | sparse | replaces (dense / sparse) | margins harness / a777 / a63 |
|---|---|---|---|---|---|
| 4 | `xc:d4a_m4k3_mpc_tr` (F13) | 155,445,969 | 419,868 | | 4.9e-4 / 1.0e-3 / 9.9e-4 |
| 4 | `xc:d4a_m4k3_mpc_tp` (F1) | 156,433,534 | **413,978** | `xc:d4a_m4k3_mp_mpc` 159,651,569 / 425,212 (old sparse end), `_th` 158,265,974 / 430,458 | 5.5e-4 / 7.5e-4 / 7.9e-4 |
| 5 | **`xc:d5_m4k2_mp_cp_mpc_tr`** (F13) | **76,541,124** | 229,467 | `xc:d5_m4k2_mp_cp_mpc_th` 79,361,129 / 240,057 | 1.3e-3 / 5.4e-3 / 4.4e-3 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_tr` (F13) | 76,593,533 | 226,989 | `xc:d5_m3k2_mp_cp_mpc_th` 79,413,538 / 237,579 | 1.3e-3 / 3.6e-3 / 2.0e-3 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_tp` (F1) | 77,528,689 | 223,577 | | 9.3e-4 / 5.3e-3 / 2.7e-3 |
| 5 | `xc:d5_m3k2_mp_cp_mpc_tp` (F1) | 77,581,098 | **221,099** | old sparse end `rs4:d5_m3k2_xp` 84,726,010 / 226,445 | 7.3e-4 / 2.2e-3 / 2.1e-3 |
| 6 | **`c2:d6_m4s4_mp_cp_z11_lz5_tr_sl`** (F13) | **54,430,038** | 181,451 | `c2:d6_m4s4_mp_cp_z11_lz5_th_sl` 57,250,043 / 192,041 | 1.2e-3 / 7.0e-3 / 4.4e-3 |
| 6 | `c2:d6_m3_mp_cp_z11_lz5_tr_sl` (F13) | 54,759,729 | 181,289 | `c2:d6_m3_mp_cp_z11_lz5_th_sl` 57,579,734 / 191,879 | 1.1e-3 / 3.4e-3 / 2.1e-3 |
| 6 | `c2:d6_m4s4_mp_cp_z11_lz5_tp_sl` (F1) | 55,417,603 | 175,561 | | 1.8e-3 / 5.3e-3 / 2.6e-3 |
| 6 | `c2:d6_m3_mp_cp_z11_lz5_tp_sl` (F1) | 55,747,294 | 175,399 | | 1.1e-3 / 2.2e-3 / 1.7e-3 |
| 6 | `c2:d6_cp_z11_lz5_tp_sl` (F1) | 56,561,627 | 173,993 | | 1.1e-3 / 2.1e-3 / 1.7e-3 |
| 6 | `c2:d6_m3_mp_cp_c5_tr` (F13) | 56,762,333 | 171,223 | | 7.2e-4 / 2.6e-3 / 1.7e-3 |
| 6 | `c2:d6_cp_c5_tr` (F13) | 57,469,995 | 169,817 | | 4.3e-4 / 1.8e-3 / 1.0e-3 |
| 6 | `c2:d6_m3_mp_cp_c5_tp` (F1) | 57,749,898 | 165,333 | | 7.0e-4 / 2.2e-3 / 1.5e-3 |
| 6 | `c2:d6_cp_c5_tp` (F1) | 58,457,560 | 163,927 | `c2:d6_cp_c5` 63,564,360 / 170,175 | 4.4e-4 / 1.2e-3 / 7.7e-4 |
| 6 | `c2:d6_lazy_cp_tp` (F1) | 64,009,752 | **162,903** | old sparse end `rs:lazy_middle_xp` 70,397,992 / 168,175 and `xs3:lazy_middle_cp` 69,116,552 / 169,151 | 8.4e-4 / 1.2e-3 / 1.1e-3 |

- Every row is on the 2-D Pareto front of its depth, over this avenue's runs plus the
  combined-2 frontier rows (`pareto.py`).
- The whole d5 and d6 fronts are now made of these points. At d4, depth-low's `d4x` points keep
  the dense end, and these two are the new sparse end.
- The F12 (`_tq`) runs have the same dense as F13 and 6.4K more sparse, so F13 dominates them
  (`results.jsonl` keeps them, e.g. d6 54,430,038 / 187,867).

Summary of the gains:
- **Dense-best points:**
  - d6: 57.25M -> **54.43M** (-4.9%), at 181.5K sparse, below the old 192.0K;
  - d5: 79.36M -> **76.54M** (-3.6%), at 229.5K sparse, below the old 240.1K;
  - d4: the depth-low `d4x` point (134.66M) is unchanged, since its round 1 is fused. The d4a
    layout goes 158.27M (TH1S) -> 155.45M.
- **Sparse:** F1 dominates every old point it replaces (lower dense AND lower sparse):
  - the new d4, d5 and d6 sparse ends are 413,978, 221,099 and 162,903, against 425,212,
    226,445 and 168,175.
- **Per-layer:** the round-1 theta layer goes from 5202 units (TH1S) to 4714 with F1 and 4451 with
  F13 (-1.83M / -2.82M dense).
- **Every new point dominates the old point it replaces** (rows with an entry in the "replaces"
  column). The F13 points have lower dense AND lower sparse than the TH1S points they replace.
  Rows without an entry are new trade points between them.
- **Depths 7-8 are unchanged:** their round 1 is X1 | Y1 with u pairs, not a direct theta. A
  depth-7 fold layout with this theta1 would be about 50.1M, above 49.38M.
- **F1 vs F13:** F1 has fewer nonzeros than TH1S (its per-bit values mostly read only A). F13
  trades +5.9K sparse for -1.0M dense (two of its per-bit values and its slices read T), so the F1
  and F13 rows are both on the front.
- **`mp` (min-parity on raw message bits) is now a no-op:** the pool forms take every round-1 bit
  with 2 <= T <= 8, so `_mp_` and non-`mp` variants build identical networks (checked).
- **Knots between integers in round 1:** all three forms have some, as min-parity (`mp`) did in
  round 1 before. F1's are at half-integers; F2 and F3 have some irrational ones. After gate
  scaling, every lattice gate value is 0 or at least about 0.9, and the inputs are exact message
  bits, so the slope at lattice points does not matter.

**Extra checks** (`extra_checks.sh`: `runs/*.val.txt`, `*.lw4/lw5.json`, `*.stress256.json`), run
for the bold rows plus `d6_m3_..._tr_sl`, `d4a_m4k3_mpc_tr`, the F1 rows `d6_m4s4_..._tp_sl`,
`d6_cp_c5_tp`, `d6_lazy_cp_tp`, `d5_m3k2_mp_cp_mpc_tp`, `d5_m4k2_mp_cp_mpc_tp`, `d4a_m4k3_mpc_tp`, and
the F12 rows:
- `validate_bench.py`: the harness weights are bit-equal to `Compiler().get_mlp_from_tree` at log_w
  0-2, and depth, dense and sparse match;
- the harness at log_w 4 and 5 with 64 random messages;
- 256 random messages at log_w 6.

All are `ok` with 0 wrong bits (`extra_checks.jsonl`, from `collect_extra.py`).

**Regression.** With the switches off, combined-2's variants rebuild with identical numbers
(`runs/regress/`):
- `c2:d6_m4s4_mp_cp_z11_lz5_th_sl` 57,250,043 / 192,041;
- `c2:d8_rp_m4s4_mp_cp_u1_sl` 45,071,004 / 115,841;
- `xc:d5_m4k2_mp_cp_mpc_th` 79,361,129 / 240,057.

## The structure: a column pool made of the zero lane's units

**Setup.** Round-1 theta bit (x, y, z) = parity(A + T), where:
- A is the own message bit (0 for the zero lanes);
- T is the count of live message bits in the column pair (x-1, z), (x+1, z-1), shared by the
  column's 5 bits.

**Pool and outputs.**
- **The pool.** Every column has a zero lane (y = 4, capacity) whose units are functions of T
  alone. `Matrices.layer_to_units` merges units with equal gates and proportional values, so a
  live bit can read the zero lane's units for one `wo` entry each. The zero lane's units are
  therefore a free pool for the column.
- **Live bit:** three own units u_j(A, T) = max(0, g_j(A, T)) v_j(A, T), with g_j and v_j affine,
  plus sum_j beta_j P_j over the pool P.
- **Zero lane:** sum_j gamma_j P_j.
- Both may add the layer's shared constant unit (gate = BOS, one unit for the whole layer).

**Exactness conditions.** Write U_A = sum_j u_j(A, .). The form is exact iff:
- **(D)** U_1 - U_0 = (-1)^T;
- **(P)** parity(T) is in span(P) + constant;
- **(U)** U_0 is in span(P) + constant.

The live coefficients then follow: sum beta_j P_j = parity - U_0 - const. Generically (U) forces
the pool to contain the A = 0 slices z_j(T) = u_j(0, T) of the per-bit units (as separate,
A-independent units). So pool = {z_j} + extras, and (U) holds by construction.

**Cost per column:** 3 per live bit + |pool|. Lower bounds for this structure:
- 3 per live bit: 2 per-bit units cannot make the alternating A-difference for T >= 7 (wave 1);
- |pool| >= the parity unit count on [0, T]: 3 for T = 7 (MINPAR7 + constant), 4 for T = 8
  (floor(n/2), exhaustive for n <= 11; no 3-unit + constant form exists on a 1/10 knot grid,
  `syn/logs/q0grid_8_d10c.log`).

So the floor is 15 / 12 (T <= 7, 4 / 3 live bits) and 16 (T = 8). F12 and F13 reach it on every
column.

**Unit count per column type (c = 448, 8 pad bits at the end of lane 17):**
- x = 0: 4 live, T = 7: 64 columns x 15;
- x = 1: 4 live, T = 8 for 56 columns (x 16) and T = 7 for 8 (x 15);
- x = 2: 4 live, T = 7: 56 columns x 15, plus 8 columns with 3 live (x 12);
- x = 3 and x = 4: 3 live: 128 columns x 12 (T = 7, or 6 at the pad columns).

Total 4448 (F1 alone: 4712).

**How TH1S fits.** TH1S is this structure with gamma = 1 and a pool of 3 slices + hinge + affine
(5 units). Giving the live bits free coefficients on the zero lane's units is what frees the pool
units.

### F1 (`xs3.TH1P_FORMS["F1"]`, exact on {0,1} x [0, 8])

    u_1 = max(0, -2A +  T -  4) (-19A      + 8)
    u_2 = max(0,  7A + 2T -  4) (  3A      + 4)
    u_3 = max(0, 15A + 2T - 13) (        T - 10)
    X   = max(0, T) (2 - T)                          (extra pool unit = glu_xor's base unit)
    zero lane = z_1/2 + z_2/2 - (4/3) z_3 + X,   z_j = u_j at A = 0
    live bit  = u_1 + u_2 + u_3 - z_1/2 - z_2/2 - (7/3) z_3 + X

- **Exactness:** checked in exact rational arithmetic on all 27 conditions, by sympy
  (`syn/pbfull.py`) and independently by fractions (`syn/verify_form.py`).
- **silu:** gates are integer forms in (A, T), so every lattice gate value is 0 or at least 1, and
  silu at k = 32 is exact to about e^-32.
- **Nonzeros:** F1 is the sparsest of 494 exact forms (`syn/pbrank.py`), about 237 nonzeros per
  4-live column against TH1S's 275.
- **F0:** `TH1P_FORMS["F0"]`, the first form found, gives the same dense and +22K sparse
  (`runs/old_F0/`).

### F2 (`xs3.TH1P_FORMS["F2"]`, exact on {0,1} x [0, 7], used for T <= 7)

**Pool: the zero lane = MINPAR7's knots plus the shared constant.**

    z_1 = max(0, T - 3/2)(8T/3 - 47/3),  z_2 = max(0, T - 23/5)(-5T/3),  z_3 = max(0, 3 - T)(-19T/12 - 25/6)
    zero lane = z_1 + z_2 + z_3 + 25/2

**Per-bit units.** Unit j has A = 0 knot tau0_j (the pool's knot) and A = 1 knot tau1_j:

    u_j = max(0, s_j (T - tau0_j) + s_j (tau0_j - tau1_j) A) (lambda_j (C0_j + C1_j T) + d_j A)
    tau1   = (9/2, 35417/11172 - sqrt(23870282185)/55860, 113777/42560 + sqrt(26231841889)/42560)
           ~ (4.5, 0.40431, 6.47884)
    lambda = (16/27, 38/27, 640/567)
    d      = (-12664/1701, (177085 + sqrt(23870282185))/23814, 75259/7938 - sqrt(26231841889)/23814)
    live bit = u_1 + u_2 + u_3 + sum_j (1 - lambda_j) z_j + 25/2

**Checks.**
- **Exactness:** checked by sympy with the surds (`syn/f2_verify.py`: 24 conditions).
- **silu:** gates are scaled so that the nearest nonzero lattice value is about 1 (factors 2, 2.5
  and 2). The one exact lattice zero (unit 3 at A = 0, T = 3) stays an exact float zero.
- **Precision:** the builder uses the surds to 25 digits.

**How F2 was found:**
1. `syn/q0grid.py`: all 3-unit + constant parity pools on [0, 7] with knots on a 1/10 grid. There
   are 4: MINPAR7 (3/2 and 23/5 up, 3 down), (3/2 up, 3 and 27/5 down), and their mirrors. The
   1/30 grid gives the same 4; the 1/12 grid gives none.
2. `syn/q0D.py`: for each pool and each cell of the three A = 1 knots, solve (D) with continuous
   knots. (D) is linear in (lambda, d) given the knots. Solutions exist for all 4 pools, as
   1-parameter families (117 cell hits).
3. `syn/q0select.py`: pick the solution with knots farthest from the lattice and smallest unit
   outputs.
4. `syn/q0exact.py`: fix one knot at 9/2 and solve the rest exactly (quadratic surds).

Notes on this search:
- **The free constant matters.** Without it no 3-unit pool spans parity on [0, 7] (`q0grid`
  without the constant: 0 pools).
- **Why the earlier searches missed it.** The earlier "3 slices only" searches (units-per-bit, and
  this avenue's `pb.py` q0 on wave 1's 9.5M integer triples, with or without the constant) found
  nothing. Their gate space has knots at multiples of 1/2 only, while every pool of 3 needs a knot
  at 23/5 or 27/5, and the A = 1 knots of the (D) solutions are generically irrational.
- **Bug found and fixed on the way.** A value-order bug in the first versions of
  q0grid/q0D gave spurious hits. The fixed versions were re-run; `q0dim.py` evaluates the
  residual independently with the exact pool values.

### F3 (`xs3.TH1P_FORMS["F3"]`, exact on {0,1} x [0, 7], used for T <= 7 in F13)

**Construction.** F3 uses the same pool as F2. The (D) solutions for a fixed pool form a
1-parameter family, so one extra condition can be imposed. F3 imposes lambda_2 = 0 in the cell
where lambda_2 is near 0 (`syn/q0exact.py` with `{"1": 0}`):

    tau1   = ((104311 - sqrt(9033204049))/17024, 142/41, (1429 + sqrt(4134929))/532) ~ (0.54440, 3.46341, 6.50837)
    lambda = (-8/13, 0, 24/19)
    d      = ((4295 + sqrt(9033204049))/10374, -328/19, (2829 - sqrt(4134929))/266)

**Effect.** The second per-bit unit is then max(0, T - 23/5 + (23/5 - 142/41) A) (-328/19) A. Its
value reads only A, which saves 7 nonzeros per live bit.

**Checks.**
- **Exactness:** checked by sympy (`syn/f3_verify.py`).
- **silu:** gate scales 2, 2.5 and 2. The nearest nonzero lattice gate values are then 0.91, 1.0
  and 0.98.
- **Margins:** F3's larger values raise the worst audit margin to 7.0e-3 at depth 6 (m4s4 digests;
  3.4e-3 with m3). That is still under the 0.02 tolerance.

## How F1 was found (exhaustive over wave 1's per-bit space)

1. **`syn/pb.py`, the relaxed filter.**
   - Input: wave 1's 4,256,720 (D)-feasible T = 8 per-bit triples (`xof/av/unit-synthesis/t_d_p3.py`:
     gates beta T + alpha A + gamma, beta in {+-1, +-2}, knots at multiples of 1/|beta|).
   - Test: parity in span{z_1, z_2, z_3} + span{one extra hinge}, for every triple and every
     hinge interval, with the hinge knot solved exactly (linear in (a, b, a t, b t) plus a
     consistency check).
   - Relaxation: (D)-null directions are included.
   - Result: 61,465 candidates.
2. **`syn/pbx.py`, exact solve.** sympy solves the bilinear system for each candidate (gamma, X
   values, knot t, (D) null parameters) and keeps real t inside its interval. Result: 494 exact
   forms, 0 timeouts.
3. **`syn/pbrank.py`:** rank the forms by nonzeros, then by coefficient size.

## Negative results

- **T = 8 at 15 units per 4-live column** (pool of 3): impossible in this structure. Parity on
  [0, 8] needs 4 pool units (see the bounds above).
- **Pool = 2 slices + 1 extra for T = 7, integer-gate space.** The only ways are a per-bit unit
  that vanishes at A = 0, or two merged slices. Both, in wave 1's integer-gate space and without
  the constant, have 0 exact solutions:
  - vanishing slice: 2,945 candidates, exact sympy (`syn/logs/pbx7_*.log`);
  - merged slices: 631 candidates, checked numerically (`syn/t7mn.py`; the sympy variant
    `t7mx.py` crashed with NotImplementedError). Superseded by F2.
- **Numeric random-restart searches** (`syn/vp.py`, `q0num.py`, `q1num.py`, `q0vp.py`) failed even
  on the positive control (T = 8 with one extra, where 494 exact forms exist). The solutions sit
  at special knot positions, so the structured and exact searches above are the ones that work.

## Other NEXT.md items (5+)

**Item 5 (a 5-unit parity of [0, 11] flat at every integer): impossible.**

What "flat" means here:
- Zero silu-smoothed slope at an integer: zero derivative there, or a knot on it whose one-sided
  derivatives average to 0 (a symmetric kink).
- glu_xor is flat in this sense at 1..11. At 0 it has slope 1, because relu(s)(2 - s) kinks
  there asymmetrically.

The proof, for a free constant plus 5 units max(0, a s + b)(p s + q):
1. Between knots f is one quadratic. A quadratic with zero slope at both ends of [k, k+1] is
   constant, but parity changes by 1 there.
2. So every unit interval with two flat endpoints needs a knot strictly inside it, or a
   symmetric-kink knot on one of its endpoints.
3. A knot inside an interval serves 1 interval; a knot on an integer serves at most 2.
4. Flat at 1..11 (glu_xor's set): the 10 intervals [1,2] .. [10,11] need 5 knots, all on integers
   and paired. That forces exactly one unit at each of 2, 4, 6, 8, 10.
5. `syn/flat11.py` solves the linear system in the values exactly (sympy) for all 2^5 direction
   patterns: 0 feasible.
6. Hence impossible, and by mirroring also for flat at 0..10.
7. Flat at all of 0..11: 11 intervals need at least 6 knots by the counting alone.

The MPC saving in the depth-8 parity layer therefore cannot be had at glu_xor-level noise with 5
units.

**Items 6-9** are not quick (6 was measured as a loss in combined-2; 7-9 need the cost model or
new layouts). Not attempted.

**Pointer for item 1 (untested).** The pool idea may also apply to depth-low's fused L1. There, each
chi1 bit is (1 - p_B) p_C, with p_B = A_b xor parity(T_b) and p_C = A_c xor parity(T_c), and the 5
rows of a column pair share (T_b, T_c). Walsh-expanding in (A_b, A_c):

    4 f = 1 + s_b (1 - 2A_b) - s_c (1 - 2A_c) - s_b s_c (1 - 2A_b)(1 - 2A_c),   s = (-1)^T

The A-free terms (1, s_b, s_c, s_b s_c) could come from units shared by the 5 rows, as the zero
lane does here. Only the A-dependent terms need per-row units:
- A_b (-1)^{T_b} and A_c (-1)^{T_c};
- A_b (-1)^{T_b+T_c}, A_c (-1)^{T_b+T_c} and A_b A_c (-1)^{T_b+T_c}.

Whether this beats the 8 pair-parity units per chi bit that d4x uses now is open.

## Headroom (round 1)

- **F12 and F13 are at the floor of the pool structure** (per-bit >= 3, pool >= parity unit
  count), 4448 units.
- **Going lower needs one of:**
  - a per-bit form with 2 units (ruled out for T >= 7 by wave 1's A-difference argument);
  - a pool shared across columns (impossible: T differs per column);
  - a different decomposition altogether, e.g. reading A through the next layer. The chi layer
    needs exact bits, so the lemma in FRONTIER.md 5.1(b) rules that out at one unit per bit.
- **Dense floor:** at depths 4-6 the round-1 layer is now 16.7M dense (4451 units x 3755), about 3.0
  units per output feature (4451 / 1465).
- **Sparse:** F13 is still 5.9K above F1. A pool-of-3 form whose slices and per-bit values do not
  read T would close that gap. All four pools found on the 1/10 knot grid have nonzero T-slopes,
  and the 1/12 grid has no pool at all.
- The 1/30 grid gives exactly the same 4 pools (`syn/logs/q0grid_7_d30_pools.log`, exhaustive over
  all 2e7 gate triples with knots in [-2, 7.5]). So on these grids the 3-unit + constant parity
  forms on [0, 7] are MINPAR7, one other and their mirrors, and every pool-of-3 theta form has
  slices that read T.
- F13's remaining sparse gap to F1 is therefore structural for pools on these grids.

## Files

- **`repo/`:** combined-2's repo plus this avenue's changes.
  - `patch.diff` = `git -C repo diff` (against the wave-2 base, all avenues included).
  - `patch_vs_combined2.diff` = this avenue only (xs3.py, xc.py, c2.py).
  - `xs3.py`: `TH1P`, `TH1P_FORMS` (F1, F2, F3, F0), `_theta1_pool`, `_theta1_pool_knots` and
    `with_th1p(variant, form)`; called from `theta1_shared_lits`, the TH1S path of the xs3 and xs4
    round-1 builders. `xc.py`: `_th1p`.
  - Variant suffixes: `_tp` (F1), `_tr` (F13), `_tq` (F12), `_tp0` (F0).
- **Scripts:**
  - `run.sh`: harness + both audits;
  - `pareto.py`: the per-depth front;
  - `extra_checks.sh`;
  - `collect.py` -> `results.jsonl`; `collect_extra.py` -> `extra_checks.jsonl`.
- **`runs/`:** raw JSON; `runs/old_F0/` holds the F0 runs.
- **`syn/`:**
  - F1 search: `pb.py`, `pbx.py`, `pbcheck.py`, `pbfull.py`, `pbrank.py`, `verify_form.py`;
  - F2 / F3 search: `q0grid.py`, `q0D.py`, `q0select.py`, `q0exact.py`, `f2_verify.py`,
    `f3_verify.py`, `q0dim.py`;
  - negative results: `pbx7.py`, `t7merge.py`, `t7mn.py`, `t7mx.py`, `flat11.py` (item 5);
  - numeric attempts: `vp.py`, `q0num.py`, `q1num.py`, `q0vp.py`;
  - logs in `syn/logs/`.

## Reproduce

Set up the environment:

```bash
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
T=$S/xof3/theta1-t8
```

Build and check:

```bash
$T/run.sh c2:d6_m4s4_mp_cp_z11_lz5_tr_sl   # harness + both audits -> runs/
$T/extra_checks.sh xc:d5_m4k2_mp_cp_mpc_tr  # validate_bench, log_w 4/5, 256 messages
python3 $T/collect.py; python3 $T/collect_extra.py; python3 $T/pareto.py
```

Re-verify the forms exactly (system python3 with sympy):

```bash
cd $T/syn && python3 verify_form.py && python3 f2_verify.py && python3 f3_verify.py && python3 flat11.py
```

# unit-synthesis: exact synthesis of gated units for 1-round Keccak XOF

Avenue: build exact synthesis tools for "minimum number of gated units max(0, g) * v"
(integer gates, rational values, one layer, shared units through wo, a constant through
the BOS unit, optionally a free affine "pass" term), answer the questions (a)-(e) of the
task, and turn useful results into smaller Keccak variants. Per-question details and the
exact constructions are in `RESULTS.md` (kept live for the theta-structure agent); this
file is the summary.

All harness numbers: `xofbench.py --log-w 6 --depth 3`, `ok: true`.

## Result in short

No 2x gain comes out of unit-level synthesis. The primitives the best circuits use are
minimal or close to it, and the one new primitive found, a one-unit mod-2 AND for lazy chi,
is worth -5% dense, -6.3% with a second small reduction (lazy4c). The exhaustive and
numeric results also close several doors: exact chi after the X layer costs more than 3
units in every one-feature encoding (so the lazy chi wins), chi rows cannot share units,
parity of several chi bits in one layer (which would remove the X layer) costs 2x more
units than it saves, and splitting a count into two features does not help parity.
**Correction:** my earlier "1-D theorem" (parity of a count needs exactly ceil(top/2)
units) was wrong. With knots between integers, n=7 takes 3 units and n=9 takes 4
(brainstorm-critic found this first, with n=11 in 5 units). Those forms are noise-sensitive.

| variant | depth | dense | sparse | vs xof-structure layout (dense / sparse) | vs glu_chi_iota | vs baseline |
|---|---|---|---|---|---|---|
| **`xs3:lazy4c_middle`** | **6** | **71,964,680** | 171,231 | lazy_middle 76,805,192 / 166,687: -6.3% / +2.7% | 1.33 / 3.43 / 3.16 | 3.33 / 45.9 / 11.9 |
| `xs3:lazy4c_middle_mp` | 6 | 70,075,915 | 176,217 | -8.8% / +5.7% (min-parity in theta1) | 1.33 / 3.53 / 3.07 | 3.33 / 47.2 / 11.5 |
| `xs3:lazy4_middle` | 6 | 72,920,840 | 171,231 | -5.1% / +2.7% | 1.33 / 3.39 / 3.16 | 3.33 / 45.3 / 11.9 |
| **`xs3:split_first_lazy4c`** | **7** | **65,316,191** | 124,782 | split_first_lazy 70,156,703 / 120,238: -6.9% / +3.8% | 1.14 / 3.78 / 4.34 | 2.86 / 50.6 / 16.3 |
| `xs3:split_first_lazy4` | 7 | 66,272,351 | 124,782 | -5.5% / +3.8% | 1.14 / 3.73 / 4.34 | 2.86 / 49.9 / 16.3 |

All pass the 63-message edge-case check (margins 0.0018-0.0076) and validate_bench
(harness weights bit-equal to the dense Compiler pipeline at log_w 0-2); lazy4c is also ok
at log_w 3, 4, 5 (64 messages). lazy4c_middle widths: `[1145,6074,1465] [1465,1602,1921]
[1921,3203,1713] [1713,2898,657] [657,5682,545] [545,675,673]`, harness margin 0.0014.

## The new primitive: a one-unit mod-2 AND (lazy chi in one unit plus a free pass term)

After the X layer (theta-structure's method 4) each theta bit arrives as E = A + D in
{0,1,2}, theta = [E == 1]. The next theta only needs chi mod 2, and
chi = theta_a ^ (~theta_b & theta_c) is congruent to theta_a + Q, Q = [Eb != 1][Ec == 1].
- Exact Q needs 2 units, even with a free affine term (exhaustive, `t_q.py`); a one-unit Q
  of width 2 without help is impossible ((1,1) is the midpoint of (0,1) and (2,1)).
- With a free affine term: `Q' = 2 Eb + max(0, Eb + Ec) (1 - Eb)` is congruent to Q and lies
  in {0,1,2}. The gate Eb + Ec is never negative, so the unit is the exact product
  (Eb + Ec)(1 - Eb) (silu(0) = 0 at Eb = Ec = 0; gates are integers). Mod 2:
  Eb + Ec - Eb^2 - Eb Ec + 2Eb = Ec + Eb Ec (Eb^2 = Eb mod 2) = theta_c + theta_b theta_c = Q.
- `o = Ea + Q'` (range [0,4]) is congruent to chi. The affine part (Ea + 2Eb) of all bits
  summed into one next-theta count is ONE unit (gate on BOS), because counts are free sums;
  so a count costs one unit plus one shared product unit per chi bit.
- Exhaustive: with one unit plus a free affine term, width 4 is the minimum (widths 1-3
  impossible, E and F encodings, integer gates abs(w) <= 4).

Where it pays: the depth-6 layout [theta1 | chi1 | X | lazy chi | last theta | last chi].
The lazy chi layer (1713 in) drops from 4914 to 2258 units (-10.9M dense) and the last
theta (657 in, 320 bits) grows from 11 to 22 units per bit (parity of [0, 44]; +6.9M).
Per chi bit: lazy4 saves 2 units of a 4.1K-dense layer and widens ~2.2 counts by 2
(~2.2 units of a 1.9K-dense layer), so it wins for every bit. Step 2's digest bits are
made exact one layer later with two units each (parity of [0, 4]).

Why it is exact: o is an integer congruent to chi at every one of the 27 E points (checked
by enumeration); each count is an exact sum of such integers; the last theta takes its
parity with glu_xor units (knots on integers, flat at the lattice points).

### Second step: reduce each column's linear part (lazy4c)

A count is own o + column(x-1) + column(x+1), and a column's linear part c = sum_y Ea_y is
in [0, 10]. Exhaustive 1-D window search (`t_win1d.py`, `t_win_exact.py`): with a free
pass term, a value congruent to c of width 4 takes 2 units,
`r(c) = max(0, 9 - 3c)(-2/3 - c/3) + 2 max(0, c - 7) + (2 - c)` in [-5, -1] (width 1: 5
units, width 2: 4, width 3: 3). The two units of a column are shared by the two counts that
read it (CSE), the count range drops from 44 to 32 (offset +10 to keep it non-negative),
the last theta from 22 to 16 units per bit: L4 2258 -> 2898 units, L5 7602 -> 5682.
Cost model (w = width kept, k = units): +320 k units at 4.1K dense, -320 (10 - w) units at
1.9K: best at w = 4, k = 2 (-1.0M); w = 3: -0.3M; w = 1 (exact parity, xof-structure's
lazy_chi_cols idea) +1.1M on top of lazy4.

### Optional: min-parity in the first theta (lazy4c_mp)

The first theta reads raw message bits (exact inputs), so knots between integers are
safe there: n=7 in 3 units (this avenue's exact form, RESULTS.md) and n=9 in 4
(brainstorm-critic's MIN_PARITY[9]); theta1 6074 -> 5571 units. Dense -2.6%, sparse +2.9%
(the min-parity values read all n inputs, glu_xor's mostly read none): a trade, kept as a
separate variant. It needs glu's eager check to accept sums within 1e-6 of 0/1
(`neurons/core.py`, the same change as brainstorm-critic's).

## What composes

- lazy4/lazy4c is a drop-in replacement for theta-structure's lazy chi (layout "L") in any
  layout (xof-structure's builder is used here; theta-structure's tsx works the same way).
- xof-structure's newer `lazy_chi_cols` (exact parity of each column's linear part, range 13)
  attacks the same trade-off from the other side; for the Walsh depth-5 design their range 13
  is better (Walsh costs ~4R units per digest bit), for depth 6 lazy4c is better.
- Min-parity (brainstorm-critic) composes where inputs are exact (theta1).
- Packed digests, constant folding, last-round DCE, free sums, hidden-unit CSE: kept.

## Answers to (a)-(e) (details, constructions and tools in RESULTS.md)

(a) n-bit parity with fewer than ceil(n/2) units: n <= 6 no (exhaustive; 5 bits over all
gate pairs with abs(w) <= 3, 129,001 ReLU vectors). n >= 7: yes, with the plain count as the
feature and knots between integers: n=7 -> 3, n=9 -> 4 (exhaustive 1-D over rational knots
with denominator <= 6, `t_par1d.py`), n=11 -> 5 (brainstorm-critic); n=8 and n=10 gain
nothing (3 resp. 4 impossible). These forms have slope at the lattice points and amplify
input noise; with knots on integers ceil(n/2) holds. Two partial counts (2-D grids) do not
help.

(b) chi on xor pairs f(A1,D1,A2,D2,A3,D3): 2 units impossible: full 6-bit exhaustive search
with abs(w) <= 2, and a restriction proof search for larger weights (on A2 = D2 = 0 it is
4-bit parity; all 74,361 two-unit 4-bit-parity solutions with abs(w) <= 5 fail to extend
with (A2, D2) weights in [-3, 3]; with a free affine term too, abs(w) <= 2 / [-2,2]).
3 units: impossible in the one-feature encodings E = A + D (abs(w) <= 3), F = A - D and
G = A + 2D (abs(w) <= 2), also with a free affine term (E abs(w) <= 3, F <= 2). 4 units:
real-valued fits exist (theta-structure); no integer-gate form found. So an exact chi
after the X layer costs at least 4 units in these encodings, and the lazy form wins.

(c) a 5-bit chi row as a 5-output function: exactly 5 units (the 5 outputs are linearly
independent functions, one unit per bit suffices). Sharing cannot help.

(d) parity(A + t), t in 0..10 shared by the 5 bits of a column: per-bit units change only
Delta(t) = f(1,t) - f(0,t) = (-1)^t; 2 per-bit units impossible, 3 need half-integer knots
(459 triples) plus 5-6 shared units per column: 4-4.2 units per bit instead of 6, with
value coefficients ~40 (noise gain; xof-structure measured edge margin 1.3e-2). Not used.

(e) other, all in RESULTS.md: the one-unit mod-2 AND Q' (above); lazy chi with 2 units +
pass in width 2 (exists, ties with lazy4c); a one-unit mod-2 AND of two E values,
`max(0, Eb + Ec - 1)(2 - Eb)` (congruent to theta_b theta_c, in {0,1,2}); a one-unit mod-2
inner product of two AND terms on exact bits, `max(0, b1+c1+b2+c2-1)(b1+c1-b2-c2)/2`;
window reductions of a count mod 2; parity of several chi bits in one layer (below).

## Tried and did not work (and why)

- **Removing the X layer by computing D2 = parity of the 10 chi1 bits of a column pair
  straight from the 30 theta1 bits in the chi1 layer** (E2 = chi1 + D2 would then be a free
  sum; depth 5 at ~54M + 1.49M x K_D, and depth 4 with a Walsh last round). K_D >= 10, and
  already XOR of 2 chi bits needs more than 3 units (exhaustive, L-space abs(w) <= 6 and
  bit space abs(w) <= 1) and has no exact 4-8 unit fit numerically (residuals 6e-3..3e-4);
  XOR of 3 chi bits is far from exact with 7 units (residual 0.65).
  chi is not a threshold function (chi(a,0,c) = a ^ c), so no affine form counts chi bits
  and glu_xor's trick does not transfer. Not viable.
- Exact chi from the X layer's encodings in 2 units (a 4-unit round, ~-10M): impossible.
- A 2-unit width-2 lazy chi without a pass term (~-6.5M): none (E, F abs(w) <= 4; G <= 3).
- Lazy chi with one unit and width 3: impossible (E, F exhaustive).
- A cheaper one-layer lazy round (xof-structure's d4 uses 17 units per bit): products of
  counts are congruent to the AND of their parities, but their range (up to 121) makes
  every later parity unaffordable.
- Round 1 as X layer + chi from E: chi from E costs >= 4 units, more than theta1 saves.
- Wider D in the X layer (window reduction): its saving is eaten by wider lazy chi outputs
  (lazy chi on 4-valued codes needs width > 5 with one unit).
- Depth 5 with exact chi from E (4 units) + Walsh: ~93M (vs direct_walsh 105.5M) but no
  integer 4-unit chi was found; xof-structure's cols + Walsh (96.2M) is the depth-5 point.
- The last theta through exact column parities (each shared by 2 bits) + a Walsh chi layer
  on 3-bit xors: fewer units but a wider last layer (+2.4M).

## Tools (this directory)

- `synth.py`: exact span search (fixed gates -> linear), gate enumeration as distinct ReLU
  vectors, batched projections; `symm.py`: orbit-canonical first gates.
- `lz.py`: encodings E/F/G/AD of theta after the X layer; window (mod 2) solver (pivot rows
  + enumeration of allowed values); exact rational values via sympy.
- `t_lz_k1.py`, `t_lz_k2.py`: lazy chi with 1 unit + pass / 2 units (+ pass), all windows.
- `t_k3.py`: exhaustive K=3 exact (optionally + pass) with a monochromatic-inactive filter.
- `t_ext.py`, `t_ext_pass.py`: restriction-extension search for 2-unit exact chi on (A,D).
- `t_par1d.py`, `t_par1d_all.py`, `t_par1d_vals.py`: exact 1-D parity minima and values.
- `vpx.py`: numeric variable-projection fits (evidence only).
- `t_grid.py`, `t_win1d.py`, `t_win_exact.py`, `t_ip2.py`, `t_ipk.py`, `t_q.py`, `t_fix.py`,
  `t_chixor.py`, `t_chixor2.py`, `t_e4.py`, `t_k2p_show.py`: the smaller questions;
  `cpsat.py`, `t_cp_*.py`: CP-SAT models (time out beyond 3 inputs).
- `impl/xs3.py`: xof-structure's builder (copied with xs.py, shared_theta.py) plus the
  layout kinds `lazy4`, `lazy4c` (`lazy_chi4`, `lazy_chi4c`, lazy4 digests in
  `carry_layer`) and `par_units` (min-parity in theta1); variants `xs3:lazy4_middle`,
  `xs3:lazy4c_middle`, `xs3:lazy4c_middle_mp`, `xs3:split_first_lazy4`,
  `xs3:split_first_lazy4c`.
- `runs/`: harness JSON, edge-case checks (`adv_*.json`), validate_bench outputs;
  `adv/`: copy of the edge-case checker and reference digests; `bench/layerstats.py`.
- Run: `U=$S/xof/av/unit-synthesis; PYTHONPATH=$U/repo/src:$U/impl:$S/xof $S/venv/bin/python
  $S/xof/xofbench.py --log-w 6 --depth 3 --variant xs3:lazy4c_middle --widths`

`patch.diff`: `git -C repo diff` (the base's uncommitted changes, xof-structure's compiler
patch that the builder needs: numeric glu nodes and hidden-unit CSE in
`Matrices.layer_to_units`, and this avenue's one-line tolerance in glu's eager check)
followed by the new files `impl/xs.py`, `impl/shared_theta.py`, `impl/xs3.py`. The repo
test suite passes (55 tests, hash_long_test not run).

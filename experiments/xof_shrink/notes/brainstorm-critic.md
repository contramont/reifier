# brainstorm-critic: avenues the others may miss, feasibility checks, and an adversarial audit

Avenue: think adversarially about ideas outside the other eight avenues, estimate them,
rank them, and implement the top ones nobody covers. Target: 1-round Keccak XOF,
3 steps, log_w=6, final `MLP_SwiGLU`. Numbers are (depth / dense / sparse). References:
baseline 20 / 3,305,445,348 / 2,034,788; glu_chi_iota 8 / 247,067,600 / 541,724.

## TL;DR

1. **The brief's glu_chi_iota computes wrong digests, and so do four theta-structure
   variants (the three-layer-first family). The harness cannot see it.** `xofbench.verify` uses 16 random, *balanced*
   messages and compares the network with the variant's *own* eager function.
   `adv/adv_check.py` compares with the *unpatched reference xof* on 63 messages: all
   zeros, all ones, alternating bits, alternating 00/FF bytes, one-hot, one-cold, and
   random at density 0.05, 0.5 and 0.95. At log_w=6:
   - glu_chi_iota on the base compiler: **1294 wrong bits** in total. All-ones,
     all-ones-but-one and alternating 00/FF bytes are each off by 1.05, while balanced
     random messages pass. The cause is base's layer 1, whose 2-unit step copies of the
     message err by about e^-6r with the same sign for every 1 bit. All-ones pushes that
     layer's RMS scale to r = 1 (error 3.6e-3), and runs of ones make the errors coherent.
     The rounds then amplify them: a layer-by-layer trace of all-ones at log_w=3 grows
     0.004 -> 0.005 -> 0.014 -> 0.08 -> 0.23 -> 0.49. Every improved compiler (no layer 1,
     1-unit copies) is immune.
   - theta-structure `tsx:three_first_lazy` (7 / 76.3M / 121.6k) and `_pack` (7 / 71.9M /
     122.1k): **1 wrong bit** each on all-ones, margin 0.54. `tsx:three_first_middle`
     (8 / 68.5M / 110.0k) and `_pack` (8 / 63.8M / 109.6k): margin 0.17-0.18. All four
     leave the features E = A + D in {0,1,2} unscaled (the other free sums are scaled by
     1/8). On dense messages most E are 2, and the next layer's normalized BOS falls to
     0.62. **Fix, tested on a private copy: scale E by 1/2 (`E_SCALE`), with its readers
     using weight 2.** Unit and parameter counts do not change, and all four then pass
     (margins 1e-3 to 4e-3). Results are in §5.
   - Every other best circuit passes with margin ≤ 3.8e-3: tsx lazy_middle(_pack) /
     three_middle / two_layer, linear-fold `lin_theta_consts_pack` and `_last1`, leveling, dce-const,
     cheap-gates, and the architecture avenue's four programs.

   **Recommendation:** add these edge cases to the harness and compare against the
   reference xof, not the variant's own function.
2. **Lower bounds: the composed ~6 / 83-100M / 168-181k designs are within about 3x
   (dense) of what `MLP_SwiGLU` allows for this XOF, and 100x is out of reach on every
   metric** (§4). Any layer that outputs N independent functions needs at least N hidden
   units, and each state layer carries at least 1600 features, so dense is at least
   about 34M. Depth is at least 6 with 2-layer rounds and 5 at bounded cost.
3. **Fewer units for parity: exact but noise-sensitive.** An exact search over knot patterns
   (knots between integers, units opening both ways) needs about 2(n+1)/5 gated units where
   glu_xor needs ceil(n/2): **11-bit parity with 5 units (glu_xor 6), 9-bit with 4 (glu_xor
   5).** glu_xor is robust for a reason nobody had written down. Its knots sit on integers,
   and silu's smoothing (slope 1/2 at 0) averages each kink's slopes (-2, +2) to 0, so it is
   flat at every lattice point. Knots between integers leave slope at some lattice points
   (2.5 at best with every knot in a gap, 2.0 with some knots on integers, 36 for the
   ramp-rich solution). Measured, with constants folded and all three thetas at 5 units (2 of
   them pure ramps, slope 2.85): **7 / 108.8M / 375.4k** against the 7 / 124.5M / 348.8k
   control (-12.6% dense, +7.6% sparse), passing every edge case at log_w=6 (margin 1.6e-3).
   Inside linear-fold's best design (a private copy), the flattest entry (effective slope
   2.0, knots on integers where possible, §6) gives **6 / 82.2M / 183.4k**, harness-ok. That
   is -10.5% dense against their 91.8M, but the edge-case margin at log_w=6 is 0.0265, just
   over 0.02. **Use it only where the parity inputs are clean** (raw-bit designs), unless a
   flatter 5-unit solution is found (§6).
4. **One-layer last round (Walsh expansion of chi):** depth -1. The identity is exact:
   `chi(A,B,C) = (p(A) + p(A^C) - p(A^B) + p(A^B^C))/2`, where each p is the parity of a
   linear form of the round input. Linear-fold found the same idea independently
   (`_last1`: 5 / 105.7M / 222.7k). Mine, on the raw-bit design: 6 / 157.0M / 610.7k
   against the 7 / 124.5M / 348.8k control.
5. Dead ends, with reasons (§3): smooth-silu parity with exact scale (Polya bound; the
   solutions are ill-conditioned), RMSNorm division as a nonlinearity, sharing theta units
   across outputs through wo (proven impossible), ±1 encoding for size (none; its one real
   use is robustness, see §3.3), complemented lanes, changing c and q, one-layer rounds
   for rounds 1-2.

## 1. Measured results

### Mine (log_w=6, xof depth 3, `xofbench.py`, `ok: true`, and the edge-case check)

| variant (`bc_variants.py`) | depth | dense | sparse | margin (harness / edge cases) | what |
|---|---|---|---|---|---|
| `c_ccc_trunc` (control) | 7 | 124,461,385 | 348,814 | 3.8e-4 / 6.5e-4 | constants folded (theta1 reads the message), last step computes only what the digest needs; base compiler (its output copy layer remains) |
| **`c_mmm_trunc`** | 7 | **108,778,505** | **375,445** | 4.1e-4 / 1.6e-3 | + 5-unit 11-bit parity in all 3 thetas (`glu_xor_min`; 2 of 5 units pure ramps, slope 2.85) |
| `c_mmm_trunc`, earlier table entry | 7 | 108,778,505 | 442,853 | 4.0e-4 / 9.7e-4 | the same with the slope-2.50, 0-ramp solution: every value reads all 11 inputs |
| `c_ccc_walsh` | **6** | 157,019,610 | 610,715 | 3.8e-4 / 6.5e-4 | + last step as ONE layer (Walsh chi) |
| `walsh_last` (superseded) | 7 | 215,669,153 | 669,641 | 4.1e-4 / **fails** | on base's layer 1: inherits its dense-message failure |
| `trunc_last` (superseded) | 8 | 183,110,928 | 407,740 | 4.3e-4 / **fails** | same |

Against glu_chi_iota, `c_mmm_trunc` is 1.14x depth, **2.27x dense** and 1.44x sparse, and
unlike glu_chi_iota it is correct on every edge case. `validate_bench.py` passes for
`c_mmm_trunc` (both table entries), `c_ccc_walsh` and `walsh_last`: weights bit-equal, identical dense and
sparse at log_w 0-2. Layers of c_mmm_trunc (in, hidden, out):
[1145,8002,1601] [1601,1602,1601] [1601,8450,1825] [1825,2050,1825] [1825,2498,769]
[769,1122,673] [673,1346,673]. Theta goes from 9602 to 8002 hidden units.

### Audit of the other avenues' best circuits (log_w=6, `adv/summary_w6.jsonl`)

63 messages, reference = the unpatched `reifier.examples.keccak.xof`. The eager function
of every variant matches the reference on the first 12 messages (the 9 edge cases and 3
random). The failures are numerical.

| circuit | depth / dense / sparse | edge-case margin | verdict |
|---|---|---|---|
| glu_chi_iota, base compiler (brief) | 8 / 247.1M / 541.7k | 1.06, 1294 wrong bits | **wrong** on ones, ones-but-one, 00/FF bytes |
| theta-structure `tsx:three_first_lazy` | 7 / 76.3M / 121.6k | 0.54, 1 wrong bit | **wrong** on all-ones |
| theta-structure `tsx:three_first_middle` | 8 / 68.5M / 110.0k | 0.17 | over tolerance (ones-but-one) |
| theta-structure `tsx:three_first_lazy_pack` | 7 / 71.9M / 122.1k | 0.54, 1 wrong bit | **wrong** on all-ones |
| theta-structure `tsx:three_first_middle_pack` | 8 / 63.8M / 109.6k | 0.18 | over tolerance (ones-but-one) |
| theta-structure `tsx:lazy_middle_pack` | 6 / 78.5M / 168.3k | 2.3e-3 | ok |
| theta-structure `tsx:three_middle` | 7 / 75.1M / 156.2k | 3.8e-3 | ok |
| theta-structure `tsx:lazy_middle` | 6 / 82.9M / 167.8k | 1.2e-3 | ok |
| theta-structure `tsx:two_layer` | 6 / 100.6M / 181.5k | 3.5e-4 | ok |
| linear-fold `lin_theta_consts_pack` | 6 / 91.8M / 179.6k | 2.2e-4 | ok |
| linear-fold `lin_theta_consts_pack_last1` | 5 / 105.7M / 222.7k | 3.3e-4 | ok |
| leveling (glu_chi_iota, all passes) | 6 / 96.9M / 297.8k | 6.1e-4 | ok |
| dce-const (glu_chi_iota) | 6 / 106.9M / 304.6k | 5.5e-4 | ok |
| cheap-gates `glu_chi_iota_simplified` | 6 / 101.5M / 300.5k | 4.6e-5 | ok |
| architecture `gated_split_taps` / `tied_split_taps` | 9 / 30.3M / 56.7k, 9 / 41.4M / 58.6k | 7.1e-5, 9.2e-5 | ok |
| architecture `gated_compact_taps` / `tied_compact_taps` | 6 / 56.0M / 168.0k, 6 / 63.7M / 171.2k | 3.3e-5, 3.2e-5 | ok |

E-scaling fix for the four theta-structure failures, on a private copy (`ts/repo`, E_SCALE
= 1/2): see §5 for the log_w=6 numbers.

How to audit any circuit, which only reads the other avenue's code:
```
S=...scratchpad; A=$S/xof/av/brainstorm-critic
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repo>/src:<avenue dir>:$S/xof \
  $S/venv/bin/python $A/adv/adv_check.py $A/adv/ref_w6.pt module:function \
  [--harness xofbench_fold --kw fold=1,unit_copies=1]
```
`adv/ref_gen.py` regenerates the reference in a fresh process. `adv/adv_check_res.py`
covers the architecture avenue's residual harness.

## 2. Ranked list

Gains are my estimates against the composed designs unless marked measured. "Covered"
means another avenue owns it.

| rank | idea | gain | cost | status |
|---|---|---|---|---|
| 1 | **edge-case audit against the reference xof** | correctness | none | **done**: 5 failing circuits found (glu_chi_iota, 4 theta-structure), fix for the 4 |
| 2 | 5-unit 11-bit / 4-unit 9-bit parity (`glu_xor_min`) | dense -12.6% (measured, raw-bit design, edge-ok); -10.5% in linear-fold's design (82.2M, harness-ok, edge 0.0265) | +7.6% sparse on raw bits (2-ramp solution); noise gain about 10x glu_xor's, fails dense edge cases on count features | **implemented**, keep only where inputs are clean |
| 3 | one-layer last round (Walsh chi) | depth -1 | dense +26% raw-bit (measured), +15% with count features (linear-fold measured) | implemented; covered by linear-fold too |
| 4 | scale every integer feature so RMS stays near 1 on all inputs (E = A + D) | correctness of 3-layer rounds | none | tested fix (§5) |
| 5 | 2-bit packing of carried digests | dense ~-7% | none | covered: linear-fold and theta-structure `pack` |
| - | sum features in chi's wo, D split, lazy chi | ~2x sparse, 1.2-1.5x dense | | covered: linear-fold, theta-structure |
| - | 1-unit copies/BOS, L1 removal, last-round truncation | ~2.3x dense | | covered: cheap-gates, leveling, dce-const, xof-structure |
| x | smooth silu with exact scale | none in practice | | infeasible numerically (§3.5) |
| x | RMSNorm division as a nonlinearity | none | | infeasible (§3.4) |
| x | sharing theta units across outputs via wo | none | | proven impossible (§3.6) |
| x | ±1 encoding for size / complemented lanes / c, q | none | | §3.3, §3.9 |
| x | one-layer rounds 1-2 | depth | dense 4-6x | infeasible (§3.1) |

## 3. Details

### 3.1 One-layer rounds via the Walsh expansion of chi

As a ±1 function, chi's Fourier expansion is
`s_o = (s_A + s_A s_C - s_A s_B + s_A s_B s_C)/2`. In 0/1 terms that is
`o = (p(A) + p(A^C) - p(A^B) + p(A^B^C))/2`. A, B and C are theta outputs (XORs of 11 input
bits), so each term is a parity of 11, 22, 22 or 33 input bits: the symmetric difference,
and on the diagonal lanes no column parities cancel. That is glu_xor units, 45 per digest
bit. Iota flips A, which is in every term, so each sum gets +1 (`p(s+1) = 1-p(s)`) and
needs no extra unit. The A^B^C term needs a ridge direction with odd coefficients on all
three sums, so its range of 34 values cannot be avoided. It pays only for the last step
(224 bits): a full round would be 72k units against 11.2k. The eager `glu` sums are
exactly 0/1, and `bc_variants.check_against_reference` matches the unpatched xof at
log_w 0-3.

### 3.2 Parity with fewer gated units, and why glu_xor's form is robust

glu_xor's pieces are parabolas over [2j, 2j+2] with knots on even sums, so 2 new points per
unit. With knots strictly between integers and units that open to the left
(`max(0, b-t)(p t + r)`), a C0 piecewise quadratic can alternate 3- and 2-point pieces.
Two adjacent 3-point pieces are impossible (their difference has discriminant 36-64 < 0).
`exp/par1d.py` enumerates activation patterns exactly: fixed pattern, linear solve, then the
root-in-gap check on the nullspace. The minima are **n=3: 2, n=5: 3, n=9: 4, n=11: 5**.
`exp/par1d_ramps.py` maximizes pure ramps (an 11-bit solution with 3 ramps has slope 36).
`exp/par1d_flat.py` minimizes the slope at the integers (11-bit: 2.50, 9-bit: 2.45, both
with 0 ramps). `exp/par1d_flatramps.py` maximizes ramps under a slope cap of 4 (11-bit: 2 ramps,
slope 2.85: the table entry behind the final numbers).

Noise: a single 11-input xor through the compiled SwiGLU, with input noise sigma 1e-4 /
1e-3, errs by 0.044 / 0.37 (slope-36 solution), 0.002 / 0.025 (slope 2.45, 9-bit) and
0.0005 / 0.0019 (glu_xor). glu_xor's knots are on integers, and `silu(z) ~ z/2` near 0
averages the -2/+2 kink slopes to 0, so its output is flat at every lattice point. That is
the "clean" property, obtained for free, and why every glu_xor chain in this project
tolerates float32 noise. A knot between integers gives up this flatness at the adjacent
integers, and a 2-point piece always has slope ≥ 1 at one end. So "fewer units" and "flat
at the lattice" conflict. The practical rule: use `glu_xor_min` where its inputs are clean
(raw message bits, precise glu chi outputs). Counts carrying coherent errors (dense
messages in linear-fold's scaled sums) push it over tolerance. **For unit-synthesis: the
objective must include the slope at the lattice points, not only exactness.**

`exp/par1d_int.py` searches that objective directly. It allows knots exactly on integers,
which count half their one-sided slope there because silu flattens them, and minimizes the
largest *effective* slope over the lattice. glu_xor's effective slope is 0 everywhere
except 1 at t=0, where all inputs are exact zeros. Results: n=5 with 3 units reaches 0.40
(one knot in a gap, two on even integers). **n=9 with 4 units reaches 2.0** (two gap
knots), against 2.45 with gap knots only. n=11 with 5 units: see §6. A slope of about 2
at high counts is still too much for coherent count errors, which is why min-parity stays
a raw-bit-only option.

### 3.3 ±1 encodings

A ±1 product costs 2 gated units, against 1 for the 0/1 xor2 `max(0,a+b)(2-a-b)`, and chi
is 1 unit either way, so there is no size gain. But ±1 features make RMSNorm's scale
exactly constant, and that is the variable behind every failure in §1: dense inputs, or
unscaled integer features, change r and with it the silu/ReLU fidelity of every unit. The
cheap version is the rule of §1: keep every feature's magnitude at or below about 1
relative to BOS for all inputs (scale integer features, as theta-structure does with its
1/8, but everywhere).

### 3.4 RMSNorm's division as a nonlinearity

One data-dependent scalar divides every feature of a layer. ReLU-limit units are positively
homogeneous, so it cancels in every ratio. It cannot implement 1600 per-bit
nonlinearities. It only matters as a precision hazard (§1).

### 3.5 Smooth silu units with an exactly known scale

Clearing the sigmoid denominators turns `f(t) = sum_i silu(l_i t + m_i)(a_i t + b_i)` into
an exponential polynomial. Polya's bound then gives `f - 1/2` at most `3(2^k - 1)` zeros:
k=2 cannot do 12-point parity (9 < 11 sign changes), k=3 might. In practice exact 1-unit
3-bit parity is impossible (`f = q*sigmoid` has only q's zeros). The optimizer's
approximate 1-unit solution (error 1.9e-4) multiplies silu's tail at -9.7 by a value of
1655 and has slope 8000. Random-restart fits (`exp/smoothfit2.py`) for 11-bit with k ≤ 4
stay at error 0.55. The first layer's scale depends on the message anyway. Dead end.

### 3.6 Sharing hidden units across outputs through wo

For theta in one layer, `theta_y = par(S) + a_y (-1)^S` (S the shared 10-bit column-pair
sum). The per-y part needs `sum_k relu(S - t_k) w_k beta_k^T = (-1)^S I_5`, i.e. a rank-5
update at every knot of the zigzag, so at least 5 units per knot. Putting a_y in the gate
makes the a_y=1 branch a global quadratic, which cannot zigzag. Chi's 1600 outputs are
linearly independent, so it needs at least 1 unit per bit. The only free sharing is linear
(sums through wo), which linear-fold and theta-structure do.

### 3.7 Column-parity (D) split, fusing theta into the previous chi

D is a parity of chi outputs: in the chi layer that is a Walsh expansion with 4^10 terms.
"D + copies, then chi of 3 xor2s" needs 4 units for the second part (theta-structure's
exact F6 search; my numeric fits found nothing with 3 or fewer), so 6 units per bit
against 7. Theta-structure's lazy chi (exact mod 2, 3 units) is the better form.

### 3.8 Packing several bits per feature

Carried digests as `F = d0 + 2 d1`: 1 unit per pair to carry, 2 units to unpack
(`d1 = max(0,F-1)(4-F)/2`, `d0 = F - 2 d1`), about -7% dense. Linear-fold and
theta-structure implemented it (`pack`). Packing state bits costs decode units in chi,
for no net gain.

### 3.9 Other items from the brief

- `silu(z) - silu(-z) = z`: an exact 2-unit linear pass-through. The 1-unit
  `max(0,BOS)*x` is exact to e^-16, and cheap-gates and leveling use it.
- Complemented lanes: NOT is free (a sign in the weights and a bias through BOS).
- c, q: they scale gates and values only; unit counts do not change. Precision is set by
  the RMS scale r (§1), not by c·q.
- Literature: ReLU depth-2 parity needs about n units, glu_xor about n/2, and the exact 1-D
  minimum is about 2(n+1)/5 (§3.2). Product (bilinear) layers do parity in log n depth.
  Keccak bitslicing tricks target NOT count and memory, which are free here.

## 4. Lower bounds

- Hidden: outputs = wo · hidden, so N linearly independent output functions need at least
  N units. Theta's 1600 outputs and chi's 1600 are each independent, so a round needs at
  least 3200 units.
- Width: each state layer carries at least 1600 features (bits or integer sums: theta needs
  1600 distinct sums, chi 1600 distinct gates).
- Dense is about the sum of hidden x (2 in + out), at least about 7.7M per full layer, so
  **at least about 34M** for this XOF: 7x below glu_chi_iota and 2.5-3x below the composed
  designs. Reaching it needs theta at 1 unit per bit, which needs D as an input feature,
  which needs another layer.
- Depth: 2 nonlinear stages per round, so at least 6; at least 5 with the Walsh last step.
  At most 1.6x against glu_chi_iota.
- Sparse: at least about 3 nonzeros per unit, so at least about 40k. At most 13x against
  glu_chi_iota; about 4.5x is already reached in `MLP_SwiGLU`.
- These bounds are for untied `MLP_SwiGLU` weights. The architecture avenue's weight tying
  (every XOF step is the same round with the same constant, so its layers are counted
  once) and its residual stream (no copies) change the accounting. Its best,
  `gated_split_taps` at 9 / 30.3M / 56.7k, passes my audit, and is still far from 100x.

## 5. E-scaling fix for theta-structure's three-layer first round (private copy)

`ts_e_scale.diff` (against theta-structure's repo; applied only in my copy `ts/repo`):
`E_SCALE = 1/2`. E = A + D is built with `glu_terms(..., scale=E_SCALE)`, and its readers
use weight `1/E_SCALE`: `theta_from_e` and the lazy-chi units XOR_E, NXOR_E, Q_E. Unit and
nonzero counts are unchanged, as are the eager functions (the same integers, rescaled).

| circuit (log_w=6) | depth / dense / sparse | harness margin | edge-case margin |
|---|---|---|---|
| `tsx:three_first_lazy`, original | 7 / 76,332,463 / 121,630 | ok | **0.54, 1 wrong bit** (all ones) |
| `tsx:three_first_lazy`, E scaled | 7 / 76,332,463 / 121,630 | 1.0e-4 | 1.4e-3 |
| `tsx:three_first_middle`, original | 8 / 68,495,219 / 110,007 | ok | **0.17** (ones but one) |
| `tsx:three_first_middle`, E scaled | 8 / 68,495,219 / 110,007 | - | 9.8e-4 |
| `tsx:three_first_lazy_pack`, original | 7 / 71,933,215 / 122,078 | ok | **0.54, 1 wrong bit** (all ones) |
| `tsx:three_first_lazy_pack`, E scaled | 7 / 71,933,215 / 122,078 | - | 4.2e-3 |
| `tsx:three_first_middle_pack`, original | 8 / 63,818,099 / 109,559 | ok | **0.18** (ones but one) |
| `tsx:three_first_middle_pack`, E scaled | 8 / 63,818,099 / 109,559 | - | 2.0e-3 |

The general lesson: RMSNorm divides by the RMS of *all* features. Any feature whose value can
exceed about 1 (relative to BOS) for some inputs lowers r for those inputs, which softens
every silu (and every step) in the next layer. Scale integer features so that their
largest value stays at about 1 or below.

## 6. Effective-slope search for 11-bit parity with 5 units

(`exp/int_n11_k5.log`; at most 2 knots in gaps, the rest on integers)
With every knot on an integer, 5 units are infeasible (that is glu_xor's structure, 6
units). With 1 knot in a gap it is also infeasible. With **2 knots in gaps** the best
effective slope is **2.0**, with gap-knot margin 0.38 (the all-gap solutions have
0.07-0.085): knots at 2.38 (gap), 4, 5, 6.56 (gap) and 9. It is the 9-bit solution
(knots 2.38, 4, 5, 6.56) extended by one ramp at 9. Units (sig, b, p, r):
(1, 2.378, -1.5, 10.933), (-1, 6.562, -2.0, 4.877), (1, 5, 2.5, -18), (1, 9, 0, -4),
(-1, 4, 3, -8).

In linear-fold's best design (private copy `lf/`, `count_parity` using these n=9 and
n=11 entries): log_w=3 passes every edge case, margin 0.0146. The slope-2.5 all-gap entry
failed there (0.05). log_w=6: the harness (linear-fold's `xofbench_fold.py`, their flags)
gives **6 / 82,173,409 / 183,364, margin 2.4e-3, ok**, against their 6 / 91,805,897 /
179,624 (dense -10.5%, sparse +2.1%; theta2 [1713, 8114, 1713] instead of 9714 hidden). But
the edge-case check gives margin **0.0265** (95%-ones random message) and 0.023 (ones but
one), just over the tolerance, with no wrong bits. Per lattice point this solution's
effective slopes are [2, 0, -2, 1.5, -1.5, -1.25, 2, -2, 0, 0, 0, **2**]. The slope of 2 at
t = 11, the all-ones count that dense messages hit with coherent errors, is what hurts.
glu_xor is flat there. A search weighted toward the top counts
(`exp/par1d_int_top.py`: slope 0.68 at t = 9-11 but 3.46 at low t) is worse, with log_w=3
edge margin 0.052. Later rounds spread counts over the whole range, so the uniform
slope-2.0 entry (kept in `lf_min_parity.diff`) is the best found. **Verdict:** in
count-based designs, min-parity is about 0.5% over the edge tolerance at log_w=6. A
5-unit 11-bit parity with effective slope ≤ ~1.5 everywhere would make it a clean -10%
dense for linear-fold's and theta-structure's two-layer designs. None was found with at most 2
gap knots: every activation pattern was enumerated, and the continuous parameters were
sampled on each pattern's solution space. Allowing 3 gap knots is the open next step.

## Files

- `bc_variants.py`: `c_ccc_trunc`, `c_mmm_trunc`, `c_ccc_walsh`, `xof_variant(xors,
  last)`, the superseded base-layout `walsh_last` / `trunc_last`, `glu_min_chi_iota`, and
  `check_against_reference`.
- `repo/`: base plus `neurons/core.py` (`glu` accepts real-valued units up to rounding
  1e-6; exact units behave as before) and `neurons/operations.py` (`MIN_PARITY` for n=9
  and n=11, `glu_xor_min`).
- `adv/`: `ref_gen.py`, `adv_check.py`, `adv_check_res.py`, `ref_w3.pt`, `ref_w6.pt`,
  `summary_w6.jsonl` and the per-circuit `w6_*.json`.
- `lf/`, `ts/`: private copies of linear-fold's and theta-structure's repos with my
  patches (min-parity in `count_parity`; `E_SCALE`). Their `git diff` against base
  includes the owners' changes.
- `exp/`: parity searches (`par1d*.py`), smooth fits, gradient fits, `estimate.py`.
- `runs/`: harness JSON at log_w=6. `results.jsonl`: the harness lines behind §1 and §5.
- `patch.diff`: `git -C repo diff` (base's own changes included) plus the new files.
  `patch_vs_base.diff`: only my repo changes (`core.py`, `operations.py`); the new test is
  `repo/tests/min_parity_test.py`. `ts_e_scale.diff`: the E fix against theta-structure's
  repo. `lf_min_parity.diff`: min-parity in linear-fold's `count_parity` (fails edge
  cases; for the record).

# Avenue "architecture": residual stream, weight tying, linear readout

All numbers: log_w=6, 3 XOF steps (1 round each), 672 output bits, verified against the
reference `xof` of the unpatched keccak module on 16 random messages (the 4 best also on
128), margin = max |out/BOS - bit|. Baseline for ratios: `variants:glu_chi_iota`
(depth 8, dense 247,067,600, sparse 541,724). Harness: `resbench.py` (my copy of
xofbench for the new architecture) and `nonres.py` (non-residual comparison points).
Depth counts SwiGLU layer applications; the linear embedding and readout are counted in
dense/sparse but not in depth.

## Results

| variant | architecture | depth | dense | sparse | margin | vs glu_chi_iota (depth/dense/sparse) |
|---|---|---|---|---|---|---|
| gated_split_taps | gated residual, tied, per-step readout, split theta | 9 | 30,340,274 | 56,652 | 1.0e-4 | 0.89x / **8.14x** / **9.56x** |
| tied_split_taps | residual, tied, per-step readout, split theta | 9 | 41,387,945 | 58,569 | 8.4e-5 | 0.89x / **5.97x** / **9.25x** |
| gated_compact_taps | gated residual, tied, per-step readout | 6 | 56,017,389 | 168,008 | 2.5e-5 | 1.33x / **4.41x** / **3.22x** |
| tied_compact_taps | residual, tied, per-step readout | 6 | 63,694,184 | 171,208 | 1.9e-5 | 1.33x / **3.88x** / **3.16x** |
| tied_compact | residual, tied, digest shift register | 6 | 85,189,224 | 175,240 | 4.0e-5 | 1.33x / 2.90x / 3.09x |
| res_compact | residual only (unrolled) | 6 | 153,321,900 | 338,698 | 2.5e-5 | 1.33x / 1.61x / 1.60x |
| res_plain | residual only, glu_chi_iota circuit unchanged | 6 | 172,002,600 | 369,478 | 6.2e-5 | 1.33x / 1.44x / 1.47x |
| nonres_lin_compact_glu | MLP_SwiGLU + linear embed/readout (no L1/L8) | 6 | 182,557,540 | 504,627 | 5.4e-5 | 1.33x / 1.35x / 1.07x |
| nonres_tied_compact_glu | MLP_SwiGLU + linear embed/readout + tying | 6 | 78,107,880 | 171,823 | 3.5e-5 | 1.33x / 3.16x / 3.15x |
| nonres_tied_split_glu | same, split theta | 9 | 53,850,799 | 63,222 | 3.7e-5 | 0.89x / 4.59x / 8.57x |

More lines (unrolled split, gated unrolled, shift-register variants) are in
`results.jsonl`, and robustness checks are in `results_extra.jsonl`: 128 messages give
the same margins, and 12 XOF steps with the same (tied) weights pass with margins
3.4e-5 (compact) and 2.1e-3 to 2.4e-3 (split).

Best under the brief's rule (>=2x on one metric, <=1.25x cost on the others):
**gated_split_taps**, which gains 8.1x dense and 9.6x sparse and costs 1.125x depth
(9 vs 8). With the standard residual `x + SwiGLU(x)` it is **tied_split_taps**: 6.0x,
9.3x, 1.125x. If depth may not grow: **tied_compact_taps** (depth 6, 3.9x, 3.2x) or,
gated, gated_compact_taps (4.4x, 3.2x).

## Where the gains come from (attribution)

1. **Linear embedding and readout** instead of the copy layers L1 and L8: depth 8 -> 6.
   The embedding also places the 456 constant state bits (from BOS). Dense 1.30x
   (nonres_lin: 189.6M, 511K), or 1.35x with 1-unit glu copies instead of 2-unit step
   copies.
2. **Weight tying across the XOF steps** is the big one. With n=1 every step is the
   same round with the same round constant, so one block of layers is applied 3 times
   and counted once, as `model.parameters()` counts shared modules once. Without
   residuals it gives 78.1M (from 182.6M). The earlier digests move through a shift
   register in the loop state: each step moves the digest of its input state into
   register 1 and register 1 into register 2. The gain grows with the number of XOF
   steps: at 12 steps the parameter count is unchanged.
3. **Residual stream** `x + SwiGLU(x)`: persistent bits and copies are free, and
   updates can be in place. For the direct round this is about neutral against tying
   without residuals (85.2M vs 78.1M): copies disappear, but in a tied block the
   rho-pi permutation has to move every bit back to its slot, and a move into an
   occupied slot needs a clear. What the residual stream adds:
   - **Readout taps**: the stream has scale exactly 1 (see below), so digests can be
     read after each step with one tied readout head (225 x d). The shift register
     and its 448 slots go away (d 2049 -> 1601): 85.2M -> 63.7M. Without residuals
     this fails: each layer output is scaled by a data-dependent r^2, so digests from
     different depths cannot share one BOS reference (tried: margin 0.29, bits right).
   - **Gated residual** `alpha * x + SwiGLU(x)`, with a fixed 0/1 alpha per feature
     per layer (a static highway gate; alpha is counted as parameters): an overwrite
     drops the old value instead of clearing it with a unit. 63.7M -> 56.0M, and
     41.4M -> 30.3M for split theta. It is not the transformer residual, so I report
     it separately.
4. **Split theta** (a circuit change that the residual stream makes cheap): a D layer
   computes D[x][z] = C[x-1][z] ^ C[x+1][z-1] as a 10-bit xor (5 units, 320 values),
   and the next layer applies A ^= D with one unit per bit. The state persists across
   the D layer for free. Per round this is 3 layers and 3 units per bit (gated; 4.2
   with the standard residual's clears), against 2 layers and 7 (gated) or 8 units
   per bit for the direct 11-input xor plus chi. Depth 6 -> 9, dense
   63.7M -> 41.4M, sparse 171K -> 59K. Without residuals the same split costs 1600
   copies per D layer (53.9M), which is the same count as the permutation clears in
   the residual version; gated residual has neither.

Dense breakdown of gated_split_taps: layers 3 x 1921 x 4806 = 27.70M (units per
layer: D 1601, apply 1601, chi 1604), embedding 1921 x 1145 = 2.20M, readout 225 x
1921 = 0.43M, norms and alphas 11.5K. Sparse: D layer 24.3K, apply 12.8K, chi 6.4K,
norms 5.8K, embedding 3.1K, alphas 3.8K, readout 0.4K. d = 1601 (BOS + state) + 320
(D). All three layers are at their unit minimum for this factorization: 5 units per
10-bit xor, 1 per 2-bit xor, 1 per chi bit.

## New code (general compiler, not Keccak-specific)

- `compile/residual.py`: compiles any Bit circuit of threshold gates and glu units
  into a residual SwiGLU program.
  - `Graph`: folds constants. Copies and nots are resolved into (variable, sign)
    aliases, which are free: a not is a sign flip that readers fold into their weights.
    Nodes get ASAP levels.
  - `Allocator`: register allocation of stream slots. Slot choice, in order: a
    forced slot (a loop-state output), a hinted slot (where a forced in-place update
    needs its target), the target's slot for an in-place update, a slot holding a
    known constant, a dead slot, a new slot. It also emits one layer per level.
  - `compile_unrolled(fn, n_in)`: every level gets its own layer. Nodes that no
    output needs are never compiled, so the last round computes only the 224 digest
    bits and their 320 theta inputs.
  - `compile_recurrent(init, step, readout, n_in, n_steps)`: tied weights. `step` is
    compiled once into a block whose output bit i ends in the slot of input bit i.
    `readout(states)` picks outputs from the states after each step, and steps that
    read the same slots share one readout matrix.
  - Options: `gated` (gated residual), `clear="linear"|"relu"`, `prefer_new`.
- `tensors/residual.py`: `ResSwiGLU` (`x + SwiGLU(x)`), `GatedResSwiGLU`, and
  `MLP_ResSwiGLU.from_program`: a Linear embedding, unique layers run in schedule
  order, and Linear readouts at taps.
- `neurons/core.py`: `glu(..., inplace=i)`: the units add up to `new - incoming[i]`.
  `compile/tree.py` supports it by adding the copy unit `max(0, x_i) * 1`, so
  non-residual pipelines stay correct.
- `neurons/operations.py`: `glu_xor(x, inplace=True)` (`glu_xor_update`) and
  `glu_xor(x, flat=True)` (`glu_xor_flat`), see below.
- `tests/residual_test.py`: unrolled and recurrent programs, with and without gating,
  checked against eager evaluation on all inputs, including threshold gates, nots,
  constants and in-place units.
- `validate_res.py`: `MLP_ResSwiGLU.from_program` equals the harness at log_w 0-2
  (weights bit-equal, dense and sparse counts equal, forward diff <= 3e-5) for
  tied_split_taps, gated_split_taps, tied_compact_taps, tied_compact and res_compact.

## Why it is exact

- **Spin encoding makes RMSNorm the identity.** Every stream feature holds +-1:
  BOS = 1, and bits are stored as s = 2b - 1, including unused and junk slots, which
  the embedding sets to -1. So mean(x^2) = 1 on every valid input and
  norm(x) = x / sqrt(1 + eps). With 0/1 bits the norm would rescale by a factor that
  depends on the input, and the residual update (degree 2 in the normalized input)
  would not match the stream's scale.
- Units are the same gated units as before: silu(16 g) v / 16 with integer gates at
  valid inputs. They are exact where g = 0 and within ~e^-16 elsewhere. A unit's
  bit-space gate or value `w.b + c` becomes `(w/2).s + (c + sum(w)/2)` on spins, and
  its output is scaled by 2, since a spin moves by 2 per bit.
- A write of bit v into a slot holding bit y adds 2(v - y):
  - in place (glu inplace, or chi `(1-2a) max(0, c-b)`): the units already add up to
    v - y;
  - otherwise one **linear unit** max(0, 1) (-y + ...) clears y. It is exact and
    cancels y's error completely. (Relu clears `max(0, y)` have a kink at y = 0 and
    only halve the error there.) Constant-gate units of a node, like the flat xor's
    1/4, fold into that same linear unit;
  - gated: alpha = 0 on the slot and the write adds 2v - 1.
- Threshold gates become two step units with gates scaled x4. The norm no longer
  sharpens them (r = 1), and `silu(-6)` would leave ~4e-3 error.

## Numerics (what does and does not blow up)

- The compact 11-input xor with a linear clear is error-correcting to first order. Its
  slope is 0 at every integer sum except the asymmetric kink at 0, which has slope 1.
  So errors do not accumulate over rounds: margins stay at 2e-5 to 4e-5 even at 12
  XOF steps.
- **In-place compact theta** (`glu_xor_update`, 6 units instead of 7): values
  `4(1-2a)` make d(out)/da up to -8 * sum_j max(0, t-2j) (-160 at t=10). The a-gated
  unit `max(0, t+11a-11)(2t-4)` does not cancel this at a = 0. The worst-case error
  gain is about 160 per round. Measured margins: 0.011 at log_w=6 (res_update, ok but
  close) and 0.005-0.01 at small log_w, against about 2e-5 for the 7-unit version. Not
  used. Within the family "one quadratic unit plus corrections on
  u = a + t", no 6-unit representation is flat in both a and t: the flat one needs the
  7th, linear unit.
- Split theta: the non-in-place 2-input xor `max(0,s)(2-s)` has slope -2 at s = 2, so
  errors grow about 1.4x per round: 1e-4 after 3 steps, 2.4e-3 after 12. That is fine
  here. For long XOFs, use the direct theta or make the apply step flat (2 units).
- `glu_xor_flat`: `s^2 - 4 sum_(odd k<n) max(0, s-k)`, with the first unit
  `max(0,s+1/2)(s-1/2)` plus 1/4. Its kinks at odd s are symmetric and it has zero
  slope at every integer s (except s = n for odd n), with the same unit count as
  compact. In practice compact with the linear clear was as good (tied_regs flat
  8.5e-5 vs compact 4.0e-5), so the results use compact.

## Composes with

- **Any per-layer circuit improvement** from other avenues: cheaper theta or chi
  units, better xor synthesis. The compiler takes any glu or threshold circuit, and
  tying multiplies whatever the block costs by 1/n_steps. If another avenue finds
  cheaper theta units, the depth-6 tied_compact_taps variant benefits directly, and a
  cheaper 10-bit xor helps the split variants.
- **DCE and constant folding** (dce-const) are built into compile_unrolled.
  Constants in the tied loop state are placed by the embedding.
- It does not compose with the last-round DCE when tied: all steps share the full round.

## Tried or considered, did not work or was not worth it

- **In-place theta updates** save 1 unit per bit but amplify errors (worst case about
  160x per round, above).
- **Relu clear units** `max(0, y)`: kink at y = 0 (the error is only halved there),
  and the two digest moves per register bit need 3 shared relu units, against 2
  linear units.
- **Non-residual taps**: the per-layer r^2 scale differs between taps, so one BOS
  cannot reference them all.
- **Merging apply (A ^= D) into chi**: chi(a^Da, b^Db, c^Dc) is a 6-input function
  that needs about 9 units per bit, against 1 + 1 for two layers.
- **Computing D[r+1] in the chi layer of round r**: D is a parity of 10 chi outputs,
  i.e. the parity of 10 ANDs (inner product mod 2), which depth-2 threshold circuits
  cannot do in small size.
- **Column parities C first** (3-unit 5-bit xors), then A ^= C1 ^ C2 with 2 units per
  bit: 7680 vs 6724 units per round.
- **Avoiding the permutation clears in the tied residual block**: the block's output
  layout must equal its input layout, so each round every bit has to land in a slot
  that holds something else. Double buffering, doing the permutation in the D layer or
  in chi, and hints all cost 1 unit per bit, unless some write is not in place
  anyway. In the direct design it is free, because theta's write needs a clear
  anyway. The gated residual makes it free.
- **Tying theta and chi into one layer** with a phase bit: same unit count, no gain.
- **A narrower embedding**: the D and junk slots must hold +-1 for the norm, so the
  embedding covers all of d.

## Outlook (not implemented, rough estimate only)

Beyond minimal changes, the remaining factor is weight sharing *inside* a layer. The
round is equivariant under z-translation, except for iota's 4 bits. A model over 64
tokens (one per z) with 25 lane spins plus 5 D values each could run the split round
as follows:
- D: a per-token MLP that also reads token z-1 (a width-2 conv), 25 units;
- apply: per token, 25 units;
- rho-pi: a fixed per-lane circular shift plus a channel permutation, e.g. a
  depthwise conv or attention with fixed offsets;
- chi: per token, 25 units;
- iota: a per-token bias.
Rough count: about 1e5 dense parameters, mostly the kernel-64 conv if rho is done that
way, against 3e7 here. Depth stays about 9. This is no longer an MLP on a flat vector,
so I only note it.

## Files

- `patch.diff`: `git -C repo diff` plus the new files (includes the base's uncommitted
  changes). `patch_vs_base.diff`: only this avenue's changes against `$S/xof/base`.
- `resbench.py`: harness copy for residual programs. `nonres.py`: non-residual
  comparisons (Tree compiler, glu copies, tying, taps). `resvariants.py`: the XOF
  variants. `validate_res.py`: dense pipeline equality (log in `validate_res.log`).
  `treevariants.py`: in-place glu through the original pipeline (passes the original
  validate_bench at log_w 0-2). `scripts/`: small-log_w runner and a stream tracer
  for debugging.
- The repo's test suite passes (72 tests, including the new `tests/residual_test.py`).
  The original harness still gives glu_chi_iota 8 / 247,067,600 / 541,724 with this repo.
- `results.jsonl` (log_w=6, 3 steps), `results_extra.jsonl` (128 messages, 12 steps),
  and the raw runs in `final/`, `runs/`, `runs2/`.

Run: `PYTHONPATH=$A/repo/src:$A:$S/xof $S/venv/bin/python $A/resbench.py --log-w 6
--depth 3 --variant resvariants:gated_split_taps --widths` (about 10 s).

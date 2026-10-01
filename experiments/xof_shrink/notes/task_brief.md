# Shrinking the compiled SwiGLU circuit for 1-round Keccak XOF (reifier)

`S = /tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad`

## Task
reifier (contramont/reifier) compiles Boolean circuits into a SwiGLU MLP (`MLP_SwiGLU`: a stack
of layers `wo(silu(wg(norm(x))) * wv(norm(x)))`, RMSNorm, no residual connections). Target
circuit: `xof(msg, depth, k)` from `reifier.examples.keccak` with `k = Keccak(log_w, n=1,
c=CAPACITY[log_w], pad_char="_")`, i.e. ONE Keccak round per XOF step, and 3 XOF steps.
Headline config: log_w=6 (1600-bit state, c=448, 224-bit digests, 1144 message bits); use
smaller log_w to iterate fast.

Make the final SwiGLU circuit smaller in **depth** (number of layers), **dense params** (numel
of all weights) and **sparse params** (nonzeros). Look for improvements that gain at least 2x
on one metric, without costing more than ~1.25x on either of the other two. Output
correctness is required. The circuit must compute exactly the same function (all digests of
all steps) and pass the harness check: every output read relative to the BOS feature is
within 0.02 of the right bit. Keep the target architecture `MLP_SwiGLU`, unless your
avenue says otherwise (then report it separately).

## Numbers so far (log_w=6, 3 XOF steps)
- baseline (main, threshold-gate xor): depth 20, dense 3,305,445,348, sparse 2,034,788, hidden 187,752
- best so far, `variants.py:glu_chi_iota` (theta via glu_xor; chi and iota as one gated unit
  per bit): depth 8, dense 247,067,600, sparse 541,724, hidden 43,152. That is 2.5x / 13.4x / 3.8x.
  Layers as (in, hidden, out), with BOS included:
  [1145,5506,2753] [2753,9602,1601] [1601,1602,1601] [1601,10050,1825] [1825,2050,1825]
  [1825,10498,2049] [2049,2498,2049] [2049,1346,673]
  - L1 only copies the 1144 message bits and creates the 456 constant state bits (suffix and
    capacity, `const()` inside the traced function), plus more. Why is it 2752 wide?
  - L2, L4, L6 are theta: 6 units per bit (11-input compact xor), plus 2-unit copies of the
    digests of earlier steps.
  - L3, L5, L7 are chi+iota, one unit per bit.
  - L8 is only output copies (3 x 224 digest bits).
  - The last round computes all 1600 bits although only 224 are output.
  - All results are in `$S/xof/results/` (baseline.jsonl for log_w 3-6, glu_w6.jsonl).

## How the compiler works (short)
Tracing (`compile/blocks.py`: calls of functions named `gate` or `glu` become creator blocks),
leveling into a leveled graph (`compile/tree.py`, `compile/levels.py` `Origin`), per-layer
matrices with the bias folded into column 0, which carries a BOS feature that is always 1
(`tensors/matrices.py`), and SwiGLU weights (`tensors/swiglu.py` `SwiGLU.from_matrix`).
- A threshold gate is TWO hidden units approximating a step: `(silu(16(z-.375)) - silu(16(z-.625)))/4`.
- A gated unit (`neurons/core.py` `glu`/`Unit`) is ONE hidden unit, `silu(16*gate)*value/16`,
  which approximates `max(0, gate)*value`. A node may sum several units.
- Copies carry bits across levels as two-unit step gates, because there are no residuals.
- The BOS row costs 2 hidden units per layer.
- RMSNorm scales all features of a layer by the same r >= 1, so outputs are ~r^2 * bit.

This session's key trick: SwiGLU's value path lets one hidden unit compute products, e.g.
2-bit xor = max(0, a+b)(2-a-b) in one unit, n-bit xor in one layer with ceil(n/2) units, and
chi a^(~b&c) = max(0, 2a-b+c)(3-2a+b-c)/2 in one unit. Look for more tricks of that kind.

## Tools
- `$S/xof/base`: reifier main plus the gated-unit changes, uncommitted. This is your starting point.
- `$S/xof/xofbench.py`: the harness.
  `PYTHONPATH=<your repo>/src:$S/xof $S/venv/bin/python $S/xof/xofbench.py --log-w 6 --depth 3 --variant mod:fn --widths`
  It builds the layers with the repo's own Matrices/SwiGLU code, one at a time, as sparse
  tensors, reports the metrics, and verifies correctness with a sparse forward pass on 16
  random messages. log_w=6 takes about 1-2 min for the baseline.
- `$S/xof/validate_bench.py [mod:fn]`: checks that the harness equals the real dense pipeline
  (Compiler -> MLP_SwiGLU) at log_w 0-2. Weights must be bit-equal. Run it whenever you change
  the pipeline.
- `$S/xof/variants.py`: circuit variants that patch keccak.py (examples: glu_chi_iota and others).

## Rules
- Work ONLY in your own copy: `mkdir -p $S/xof/av/<name> && cp -r $S/xof/base $S/xof/av/<name>/repo`.
  Put your variant modules and scripts in `$S/xof/av/<name>/`. Run with
  `PYTHONPATH=$S/xof/av/<name>/repo/src:$S/xof/av/<name>:$S/xof`.
- If you modify the harness, copy it into your dir first.
- Never use git worktrees. Never touch `/home/ubuntu/eualethic` (neither the reifier checkout
  there nor the outer repo), `$S/xof/base`, or files of other agents. Do not commit or push.
- Every claimed number must come from the harness with `ok: true` at log_w=6. If you changed
  the pipeline (Matrices/SwiGLU/tree), also run validate_bench (or your adapted copy) for
  bit-equality with the dense pipeline at small log_w.
- Keep code clean and general where you can: a compiler improvement beats a Keccak-only hack,
  but both are welcome. Say which it is.

## Deliver
- `$S/xof/av/<name>/patch.diff`: `git -C $S/xof/av/<name>/repo diff` plus any new files.
- `$S/xof/av/<name>/results.jsonl`: harness lines for your best variants (log_w=6, depth 3).
- `$S/xof/av/<name>/NOTES.md`: the method, why it is exact, what it composes with, and what
  you tried that did not work and why.

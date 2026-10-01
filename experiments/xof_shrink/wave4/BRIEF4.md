# Wave 4: which XOF size reductions are robust, across configurations and in bfloat16? (reifier)

`S = /tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad`

## Background
reifier compiles boolean circuits into a Transformer with identity attention and no skip
connections: `MLP_SwiGLU`, a stack of `y = wo(silu(wg n) * (wv n))` with `n = RMSNorm(x)`.
Every layer has its own weights, and there are no embedding or readout layers. Readout is
relative to a BOS feature: reifier's `boolify` reads bit 1 iff out/BOS is within 0.02 of 1.

**Waves 1-3** shrank the compiled 1-round Keccak XOF: `xof(msg, steps=3, k)` with
`k = Keccak(log_w=6, n=1, c=448, pad_char="_")`, 1144 message bits, 3 x 224 output bits.

**Baseline** (main's threshold gates): depth 20, dense 3,305,445,348, sparse 2,034,788.

**Robust frontier** at log_w 6, 3 steps, 1 round. All points are in float32; worst error is
at most 0.01 on 2 audits + 3040 stress messages:

| depth | variant | dense | sparse |
|---|---|---|---|
| 3 | `dl3:d3c1_mp17_p1d_kt_xca` | 238,512,173 | 1,235,283 |
| 4 | `dl3:d4x_m4k2_mpx_wmpc_p1d_kt_xca` | 124,087,564 | 526,518 |
| 5 | `xc:d5_m4k2_mp_cp_mpc_p1dxt_ra` | 74,073,879 | 226,480 |
| 6 | `c2:d6_m4s4_mp_cp_c5_lz5_sl_p1dct_ra_bt` | 53,915,598 | 178,277 |
| 7 | `c2:d7_m4s4_mp_cp_u1_c5_sl_p1dct_ra_bt` | 48,831,997 | 128,804 |
| 8 | `c2:d8_rp_m4s4_mp_cp_u1_sl_g_p1dct_ra` | 44,218,228 | 114,557 |
| 9 | `c2:sp_col1b_p1d` (sparse end) | 66,359,285 | 91,460 |

Read first: `experiments/xof_shrink/README.md` in the base. It covers the strategy catalog,
float32 margins and floors. `notes/FRONTIER.md` section 3 has a table of every builder flag,
and `wave3/FRONTIER3.md` the wave-3 flags.

**The strategies, in short:**
- **S1 gated units:** one-unit xor2 and chi (with iota), and glu_xor.
- **S2 compiler passes:** constants folded into biases, unit sharing, no re-threshold layer
  after pure units.
- **S3 Keccak layouts:**
  - shared column parities;
  - count features (`glu(numeric=True)`);
  - lazy chi (parity inside chi's units);
  - split rounds (X then Y);
  - the fold layout (`rp`);
  - fused rounds (d3/d4, `dl3`);
  - the Walsh last round (`xs4`);
  - the sparse-first layouts (`sp`).
- **S4 packing:**
  - column packing into X2 (`cp`);
  - round-1 u pairs (`u1`, `sp`);
  - digest packing (`m3`, `m4s4`, `lz5`, `k2`);
  - chi's gate as the last theta's output (`sl`).
- **S5 parity forms:**
  - min-parity (`mp`, `mpc`; knots between integers);
  - wave 3's floor((n-1)/2) forms with irrational knots (`p1d*`, `_t` top-core);
  - reduction units (`c5`, `z11`).
- **S6 round-1 theta pool:** `_tr` / `_tp` (F13 / F1).

## Base and tools (all verified to work)
- `$S/xof4/base`: reifier at main 21683d0 plus `experiments/xof_shrink/`. It is a small git
  repo, so `git diff` gives your patch.
- Python: `$S/venv/bin/python`. Set `PP=<repo>/src:<E>:<E>/depth_low/lib/python` with
  `E=<repo>/experiments/xof_shrink`.
- **Harness:**
  `PYTHONPATH=$PP python $E/xofbench.py --log-w W --depth STEPS --rounds R --variant mod:fn --widths`.
  - It prints depth / dense / sparse and checks the network against the variant's own eager
    function.
  - Without `--variant`, it builds the threshold baseline.
  - `--rounds` sets Keccak rounds per XOF step (`k.n`). Every existing builder assumes 1 round
    and fails on more: `(rc,) = k.get_round_constants()`. `xs4`, `dl3` and `sp` also assert
    3 steps.
- **Reference sets** (run in a fresh process, from the unmodified reference keccak):
  - `python $E/audit/ref_gen.py W STEPS N_RANDOM out.pt [ROUNDS]` writes 9 edge cases (zeros,
    ones, alternating, one-hot, ...) plus random messages at densities 0.05 / 0.5 / 0.95;
  - `python $E/audit/ref_stress.py {mixed,dense} SEEDS REPS out.pt --log-w W --steps STEPS --rounds R`
    writes random messages at 10-11 densities plus lane/z patterns. `mixed 8 16` is 1792
    messages; `dense 8 12` is 1248.
- **float32 audit**, against reference outputs:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python $E/audit/adv_check.py [--eager 0] ref.pt mod:fn`.
  It reports margin = worst |out/BOS - bit| and wrong_bits.
- **bfloat16 audit:** `PYTHONPATH=$PP python $E/audit/bf16_check.py ref.pt mod:fn|baseline`.
  - `w16`: weights rounded to bfloat16, float32 activations.
  - `b16`: weights and activations in bfloat16. This is what
    `Compiler(mlp_dtype=t.bfloat16)` / `MLP_SwiGLU(dtype=bfloat16)` run, since
    `SwiGLU.forward` casts its input to the layer dtype.
  - It reports margin, wrong_nearest (err > 0.5) and wrong_boolify, and `correct` if both
    wrong counts are 0.
- **Threshold-baseline sizes for the configuration grid** are being computed into
  `$S/xof4/baselines.jsonl`, one JSON line per (log_w, xof_depth=steps, rounds). The grid:
  - log_w 0-5, 3 steps;
  - steps 1, 2, 4, 6 at log_w 4 and 6;
  - rounds 2, 3, 4 at log_w 4 (steps 1 and 3);
  - rounds 2 and 4 at log_w 6, 1 step;
  - rounds 24 at log_w 3.

  Read it rather than recomputing. A log_w-6 baseline needs about 9 GB per 3.3B dense
  parameters; build one yourself only if the grid lacks it.

## Scouting facts (measured by the lead)
- **In float32 (w16),** circuits made only of integer or half-integer units are exact: glu
  xor, chi, and the split/direct layouts, e.g. `xs3:split_first_middle`. Rounding the
  weights to bf16 breaks:
  - min-parity (`mp`);
  - the u1/sp/cp decoders;
  - the irrational-knot forms;
  - the lazy-chi layouts, to a lesser degree: margin 2.5e-2.

  The robust d8 point gets 604 wrong bits at log_w 2 and errors of 1e15 at log_w 4.
- **In full bf16 (b16),** every gated-unit circuit tried fails, even `variants:glu_xor_everywhere`
  (42 wrong bits of 5376 at log_w 4). **The threshold baseline fails too** on audit edge
  cases: 2 wrong bits at log_w 4, 7 at log_w 3 with 2 rounds.
  - Its per-layer bf16 deviation tracks the gate pre-activation magnitude. Layers with
    |wg x| up to 2822 (wide threshold gates times c*q = 32) deviate by 0.2-0.3. Near 50 they
    deviate by 0.02.
  - In bf16, silu(a) - silu(b) with a - b = 8 loses everything once |a| is in the thousands,
    where bf16's spacing is 16.
- The float32 lessons from wave 3 (README "Float32 margins"):
  - errors come from rounding amplified by forms with large cancelling terms, not from silu,
    so q does not help;
  - worst errors grow with message density and count, so always stress-test with dense
    messages.

## Robustness criterion (use it for every claim)
- **Correct:** 0 wrong bits (nearest) and 0 boolify errors.
- **float32-robust:** worst error at most 0.01 on:
  - the `ref_gen` audit, with at least 60 random messages;
  - a stress set of at least 500 messages, dense ones included, for that configuration.
- **bf16-correct in mode M:** `bf16_check` correct in mode M on the same two sets. Report the
  margin as well.
- Report sizes as (depth, dense, sparse) and as ratios to the threshold baseline of the same
  configuration.

## Rules
- **Work in your own copy:** `mkdir -p $S/xof4/av/<name> && cp -r $S/xof4/base $S/xof4/av/<name>/repo`.
  Never use git worktrees. Never touch `/home/ubuntu/eualethic`, `$S/xof4/base`, or other
  agents' directories. Do not push or commit to any real repository.
- **Shared machine** (30 cores, 222 GB): at most 5 concurrent python processes, with
  `OMP_NUM_THREADS=2`. At most one process above 20 GB at a time.
- **Time box:** finish by the deadline in your prompt. Check `date -u` often, and write
  NOTES.md and results.jsonl as you go, so partial work survives.
- Prefer exact constructions verified on the lattice and on audits over tuning.
- Say clearly what you measured and what you estimate.

## Deliver in `$S/xof4/av/<name>/`
- `NOTES.md`:
  - the method and why each construction is exact;
  - every table, with what failed and why;
  - a per-strategy verdict (robust / breaks / stops paying, with numbers);
  - remaining ideas.
- `results.jsonl`: one line per verified run (harness, audits, bf16 checks).
- `patch.diff`: `git -C repo diff` plus new files (`git -C repo add -N . && git -C repo diff`).
- Your final reply: the structured summary the prompt asks for.

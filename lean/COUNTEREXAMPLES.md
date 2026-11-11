# Counterexamples, and the patches they led to

The goal was: for every well-formed trace `t` on bits, the transformer compiled from `t` computes
it. Each entry below is a trace for which that fails, and the condition added to exclude it. The
conditions of `ReifierProofs/Spec.lean` and `Interfaces.lean` that came from a counterexample are
marked `PATCH (CE-n)` there; CE-12's patch is in the bf16 model itself (`BF16.one`), and CE-13 is
not patched. The other conditions of `WellFormed` (distinct names, each MLP reading only what came
before it, every signal a bit) are what `check_trace` and the Python's docstrings already ask.

How each counterexample is backed:

* **Lean**: a theorem in `ReifierProofs/Counterexamples.lean`, with no `sorry`, depending only on
  `propext`, `Classical.choice` and `Quot.sound`. "End to end" means: the trace meets every other
  hypothesis of the main theorem it refutes, and the compiled transformer's output is proved wrong.
* **Real compiler**: `simulations/real_compiler.py` traces the program with `reifier`, compiles it
  with `reifier.transformer.transformer`, and runs the compiler's own forward pass (numpy,
  float64), with `reifier.check.check_transformer` as the judge where it applies. For bf16, it
  replays `Transformer.run` on the compiled weights, rounding every weight and every result with
  `gemma.round_to_bf16`. Its output is in `simulations/real_compiler.out`.
* **Simulation**: `simulations/sim.py`, a model of the Lean definitions in high-precision arithmetic
  (mpmath, 60 digits), written from scratch. It agrees with the real compiler wherever both ran
  (CE-7 at 58 GLUs: `-0.056875` in both).

| id | what fails | patch | evidence |
|---|---|---|---|
| CE-1 | A step unit whose sum on bits is strictly between -3/4 and 0, e.g. `[x/2 - 1 >= 0]` (sum `-1/2` at `x = 1`). `convert` makes `[z >= 0]` into `relu(4z + 5/2) - relu(4z + 3/2)`, which is `1/2` at `z = -1/2`, not 0: the output feature is 0 instead of -1. | `WellFormed.step_margin`: each step sum is `>= 0` or `<= -3/4`. The conversion itself is exact but on `(-5/8, 0)`; `-3/4` keeps both gates at least 1/2 from 0, where `gelu` is `relu`. Integer weights, as `threshold_gate` asks, give `>= 0` or `<= -1` | Lean `ce1`, end to end; real compiler: output `0.0000`, `check_transformer` fails |
| CE-2 | A `relu` or GLU gate of `1/32` on bits, e.g. `relu(1/32) * 32` (= 1). Its compiled gate is `32 * 1/32 = 1`, where `gelu 1 < 0.85`, not `relu 1 = 1`. | `WellFormed.gate_margin`: each gate is 0 or at least 1/2 from 0, the invariant `emit.py` states of its own units ("wherever g is 0 or at least 1/2 from 0") | Lean `ce2_bf16`, end to end for every `Rounding` (output `<= 47/64` instead of 1): this is what shows the condition is needed. Lean `ce2` in real arithmetic (output `< 0.7`), where the trace also breaks `GateMargin` (CE-2b). Real compiler: `0.6824` in float64, `0.6797` in bf16 |
| CE-2b | In real arithmetic only: a gate exactly 0 on bits, which `gelu` does not cut off. `y = 2 relu(2x - 1) - relu(4x - 3) + C relu(a(x - 1))` is `x` on bits and flat, but its last gate is 0 at `x = 1`, where `gelu` passes on `gelu'(0) = 1/2` of the attention's leakage, times `C a`. With `C = 2^15`, `a = 1`, the output is off by `3.1e-9`; two of them with `a = 2^12` give `-345` instead of 1. | `Trace.GateMargin` (in `RealOK` and `RealCore`): gates at least 1/2 from 0. Not needed in bf16, where `gelu 0 = 0` exactly | Lean `ce2b`: the trace meets every other hypothesis of `MainReal` and `MainRealDeep`. The wrong output: real compiler, `0.9999999969` and `-345.158` |
| CE-3 | The same signal returned twice. `token_grid` moves output 0 to feature 1, then again to feature 2, and `held` gives feature 1 back to `code[0]`: output 0 is the current token's own code bit. | `IsProgram.outputs_nodup` | Lean `ce3`, end to end (the grid by `decide`); real compiler: `check_transformer` fails |
| CE-4 | An output that is an input, e.g. `lambda b: [b[0]]`. `token_grid` moves the input to feature 1, where the embedding writes `code[0]` and the attention adds the input: the output feature is 2. | `IsProgram.outputs_made`: each output is made by an MLP | Lean `ce4`, end to end; real compiler: `2.0000`, `check_transformer` fails |
| CE-5 | A program that reads more than `CONTEXT = 512` tokens. The head for the token 512 back cannot see it (the mask is `p - p' < CONTEXT`), and copies a mean of the bits of other tokens. | `IsProgram.window_le` | Lean `ce5`, through the attention (the stream after it is wrong); real compiler: output `-1` instead of `+1` |
| CE-6 | Two tokens of one code. They get one embedding, so they tie for the largest logit, and argmax returns the lower id. | `CodesOK.inj` | Lean `ce6`, real and bf16; real compiler: logits `4.464102` twice, `check_transformer` fails |
| CE-7 | A `relu` or GLU MLP that is not flat at bits, e.g. `relu(y + 1/2) * (2/3) y` (the identity on bits, slope 5/3 at 1), chained. The attention leaks `~1.9e-13` of the neighbouring token into each input, and the chain multiplies it by about 5/3 a layer: off by `1.1e-9` at 17 GLUs, the wrong bit at 58 (`-0.0569`). | Either `RealOK.flat`, for `MainReal` at any depth; or a bound on depth, `MainRealDeep`: slope sums at most `σ` (`RealCore.slope`) and `D` layers with `(1 + σ)^D ≤ 10^4`, each bit then within `(1 + σ)^D / 10^10`. For this GLU, `σ = 5/3` and `D ≤ 9` | Real compiler and simulation agree: `0.999999998893` at 17, `-0.056875` at 58, `-0.999999` at 70. Lean (local): `glu7_iterate`, the error grows by at least 3/2 a layer |
| CE-8 | A step gate `[w x - w >= 0]` (the identity) with `w >= 10^13`: the leakage times `w` pushes its sum past `-1/2`, and the bit flips. | `RealOK.bounded` (weight sums `<= 2^16`) and `RealOK.glu_gain` (a GLU's second-order gain `Σ_j \|wo_ij\| ‖g_j‖₁ ‖v_j‖₁ <= 10^8`; `10^5` in `MainRealDeep`, whose error grows to `10^-6`): bounds the attention's error cannot grow past | real compiler: `+1` at `w = 10^12`, `0.259` at `5 * 10^12`, `-1` at `10^13`; Lean (local): `ce8_local` |
| CE-9 | In bf16, compiled weights that bf16 cannot hold: two always-on step units with `wo = 1027/2` and `-1025/2` (output 1). Their compiled weights 1027 and -1025 round to `±1024`, the units cancel, and the output is `<= 0` instead of 1, for every rounding to nearest. | `BF16OK.weights` | Lean `ce9`, end to end for every `Rounding`; real compiler (bf16 replay): `-1` |
| CE-10 | In bf16, a gate whose sums bf16 cannot hold: `[128 x - 64 >= 0]` at `x = 1` gives `u+` and `u-` the sums 8272 and 8240, which every rounding to nearest sends to 8256. They cancel, and the output is `<= 0` instead of 1. `gemma.py` warns of this: "bf16 holds both sums only below 4096". | `BF16OK.sums`: on bits, each gate above 0, its value and their product are bf16 numbers. A gate at most 0 passes 0 however it rounds, and need not be one | Lean `ce10`, end to end for every `Rounding`; real compiler: `+1` in float64, `-1` in bf16 |
| CE-11 | A grid so wide (more than `maxBF16^2 / 4`, about `2.9e76` features) that `ONE = 2 sqrt(width)` is beyond the largest bf16 number. | `WidthOK` | argument; Lean lemmas `ce11`, `ce11_embed` (`ONE` then reads as 0) |
| CE-12 | In bf16, the norm read literally: Python's `rmsnorm` computes its scale from the float `one = 2 sqrt(W)`, while a bf16 stream holds `fl(one)`. At width 21, `fl(2 sqrt 21) = 147/16` for every rounding, and the norm gives each bit as `255/256`, not 1. In majority-of-19 (width 21; `IsProgram` and `BF16OK` hold) the output is then `1.015625` at ten ones: inexact, though never the wrong bit. | The compiler's: round `one` once, `one = round_to_bf16(2 * sqrt(W))`, and use that number in the embedding and in every norm. `BF16.one` is that `one`: the bf16 model is the compiler so patched. (`gemma.py` meets the same problem, "for one count in nine what comes out is ±0.996", with pads.) | Lean `fl_two_sqrt21`, `ce12_ratio`, `ce12`: the norm step, for every `Rounding`. The whole network: real compiler (bf16 replay), worst error `0.015625` read literally, `0` with `one` rounded once |
| CE-13 | RoPE in float32: the angle `pos * turn` loses precision as the position grows. The worst loss of a head's margin of 30 is under 0.035 below position `2^16`, 0.07 by `2^17` (past the `10^-10` budget of the real-arithmetic attention), 1 to 10 in `[2^20, 2^24)` (which bf16 survives: it needs 13.2), and from `2^24`, where float32 positions collide, a head reads the wrong token. | Not patched. The model computes RoPE's angles and trigonometry exactly, and rounds only the rotated query and key; past about `2^16` (real) or `2^24` (bf16) positions, a float32 implementation is outside the model | simulation (`simulations/ce13.py`, the scores in float32) |

## Two conditions found by checking statements, not by a trace

* `Stmt_layer_gain` first claimed that a layer keeps the stream within `(1 + σ)` times its error
  for every `σ`. It needs `σ E ≤ 1/100`: the norm's own error, `0.41 E`, scales what a GLU passes
  on, so `c^2 σ E` can pass `(1 + σ) E` when `σ E` is large. `MainRealDeep` meets it at every depth
  it uses (`σ E_d ≤ E_{d+1} ≤ 10^-6`).
* The logits in bf16 need `2 sqrt(width) + bits ≤ 250`: above 256 bf16 numbers are 2 apart, and the
  logits of two codes one bit apart, which differ by 2, can round to one number.

## What the theorems do cover

* Step gates of integer weights meet `step_margin`. In real arithmetic `MainReal` covers them while
  their weight sums are at most `2^16`; in bf16 `MainBF16` covers them while their compiled gates
  above 0 are bf16 numbers. With integer weights those are multiples of 16, so that holds below
  4096, which is `gemma.py`'s own check. `and_` of 33 bits, whose gate is `-4144` (not bf16, but at
  most 0) at all zeros, is covered (`NonVacuous.lean`). On the real compiler it runs exactly in
  bf16, and so does the decimal adder of `examples/adder.py` (width 109): every output feature
  exactly `±1`, every sum right (`simulations/real_compiler.out`).
* `lib.py`'s `parity` is flat by design, with weight sums `4n + 2` and second-order gain `4 n^3`:
  `RealOK` holds up to about 290 inputs. `sum_of_ands` is flat too.
* `Examples.xor3`, the README's `xor`, meets every hypothesis of `MainReal`, `MainBF16` and
  `MainRealDeep` at once, with codes for two tokens (`hypotheses_satisfiable`,
  `deep_hypotheses_satisfiable`).
* `relu(x + 1/2) * (2/3) x`, a GLU that is not flat, meets the hypotheses of `MainRealDeep` (with
  `σ = 5/3`, depth 1) but not `RealOK` (`deep_beyond_realOK`): the bound on depth covers what
  flatness does not.
* A `relu` gate exactly 0 on bits, such as AND as `relu(x0 + x1 - 1)`, meets `WellFormed`, so
  `MainBF16` covers it; only real arithmetic asks for `GateMargin` (CE-2b).

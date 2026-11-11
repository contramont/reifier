# Reifier's compiler in Lean 4

A model of [Reifier](https://github.com/contramont/reifier/tree/simple-refactored)'s compiler
(branch `simple-refactored`) in Lean 4 with Mathlib, and a proof that for every well-formed trace
on bits, the transformer compiled from it computes it.

```
lake exe cache get && lake build      # Lean v4.35.0, Mathlib v4.35.0; builds the proofs and checks
```

`lake build` checks everything: the model, the proofs, the checks against the Python
(`Examples.lean`), the counterexamples (`Counterexamples.lean`) and that the hypotheses can all
hold at once (`NonVacuous.lean`). No file has a `sorry`, an `axiom` or `native_decide`, and

```
#print axioms Reifier.main_real        -- [propext, Classical.choice, Quot.sound]
#print axioms Reifier.main_bf16        -- [propext, Classical.choice, Quot.sound]
#print axioms Reifier.main_real_deep   -- [propext, Classical.choice, Quot.sound]
```

## The theorems (`Main.lean`; proved in `Final.lean`)

For every trace `t` that is a well-formed program on bits (`IsProgram`), every coding of tokens
(`CodesOK`), every token sequence and every position `p` where the program's window ends and whose
tokens have codes (`PosOK`), the transformer `transformer t codes`:

* **`main_real`**, in real arithmetic, with `gelu` and softmax attention: holds each output bit of `t`
  as -1 or 1 within `10^-9`, and gives the largest logit to the token whose code `t` gives. Needs
  `RealOK`: gates at least 1/2 from 0, flat `relu` and GLU MLPs, weight sums at most `2^16`, and
  each GLU's second-order gain at most `10^8`.
* **`main_real_deep`**, in real arithmetic, for GLUs that are not flat, to a bound on depth: if every
  `relu` or GLU MLP's slope sum is at most `σ` and the grid has `D` layers with `(1 + σ)^D ≤ 10^4`, it holds each
  bit within `(1 + σ)^D / 10^10` and gives the right token. Needs `RealCore (10^5) σ`.
* **`main_bf16`**, in bf16: holds each output bit exactly, and, if `2 sqrt(width) + bits ≤ 250`,
  gives the largest logit to the right token. Needs `BF16OK`: every compiled weight is a bf16
  number, and on bits so is every gate above 0, its value and their product (for gates of integer
  weights, `gemma.py`'s check that no gate sums to 4096 gives this); and `WidthOK`
  (`ONE = 2 sqrt(width)` is within bf16's finite range).

`IsProgram` is: no two signals of one name; each MLP gives a signal and reads only inputs and
earlier MLPs; on bits every signal is a bit; a step unit's sum is never strictly between -3/4 and 0;
a `relu` or GLU gate is 0 or at least 1/2 from 0; the outputs are distinct, nonempty and made by
MLPs; the inputs are the bits of whole tokens, at most `CONTEXT = 512` of them.

The margins, distinct outputs made by MLPs, the window of at most 512 tokens, `CodesOK.inj`, and
the numeric conditions `RealOK`, `RealCore`, `BF16OK` and `WidthOK` were each added for a
counterexample, logged in `COUNTEREXAMPLES.md`: a trace that meets every other hypothesis and on
which the compiled transformer is wrong. Most are proved in Lean end to end, and all but CE-11 (a
width of 10^76) and CE-13 (RoPE in float32) are reproduced on the real compiler. The other conditions are what `check_trace` and the Python's docstrings ask.

## The model

Each file models one stage of the Python:

| Lean | Python | What |
|---|---|---|
| `Basic.lean` | `trace.py` | `MLP` (`wo @ act(wi @ x + b)`, activations `step`, `relu`, `reglu`), `Trace`, `Trace.values` |
| `Grid.lean` | `grid.py`, `token_grid` | `layout`, step by step (dict order, the free list as a stack), `held`, `tokenGrid` |
| `Emit.lean` | `emit.py` | `convert` to geglu, `signed`, `copy`, `layer`, `emit` with skip connections and norms, `rmsnorm`, `Network.run`, `gelu` |
| `Transformer.lean` | `transformer.py` | `transformer(t, codes)`: embedding, RoPE lookback heads (`runner_up`, `lookback_query`), the layers, tied logits |
| `BF16.lean` | | the same forward pass in bf16 |
| `Examples.lean` | README, tests | by `decide`: the README's grid for `xor`, and the units and weights `test_every_hidden_unit_has_a_name` asserts |

Where it is not the Python's code to the letter:

* Signals are natural numbers, not strings: a well-formed trace has no two of one name, as
  `check_trace` asserts. Where Python's dicts keep the last of two equal names and Lean the first,
  they agree once there are none.
* A row is a function from features to names, with a width: Python's `row[f]` for `f < len(row)`,
  and `None` beyond, which is the padding `layout` gives.
* Weights are rationals. `one = 2 sqrt(width)` and the attention's weights are reals.
* Only the network `transformer` uses: `emit(t, grid, "geglu", skip=True, norm=True)`.
* Real arithmetic is exact, not float64; the softmax omits the shift by the largest score.

## The model against the real compiler

`difftest/` checks the model against the Python itself. `export.py` traces programs with
`reifier` (`xor`, `add` on 2+2 and 3+3 bits, `add` with GLU parities, the tests' `odd` with an
input and a constant as outputs and an input read twice, `parity`, `sum_of_ands`, a chain of NOTs,
and the decimal adder of `examples/adder.py`) and writes them as Lean, with Python's `layout`,
`token_grid` and every unit of `emit(t, token_grid(t), "geglu", skip=True, norm=True)` as exact
fractions. For the eight small programs, the Lean model gives the same grids, cell for cell, and
the same 7626 weights of their 237 units (`lean_small.json`; regenerate with `export.py`,
`make_checks.py`, then `lake env lean difftest/small.lean` and `compare.py`). The adder (773 units
on 109 features) was not checked this way: Lean's interpreter did not finish even its grid within
50 minutes.

`simulations/real_compiler.py` runs each counterexample on the real compiler (float64), and in bf16
by replaying `Transformer.run` on its weights with `gemma.round_to_bf16` after every operation:
each fails as `COUNTEREXAMPLES.md` says, and the decimal adder and `and_` of 33 bits run exactly in
bf16 (`simulations/real_compiler.out`).

## The forward pass in bf16

bf16 numbers are `m 2^e` with `|m| < 2^8` and `-133 ≤ e ≤ 120`. A `Rounding` is any function that
rounds each real of finite range to a nearest bf16 number, whatever it does with ties: IEEE
round-to-nearest-even (Python's `round_to_bf16`) is one, and `rounding_exists` shows there is one.
The theorem holds for every such rounding. Every weight is rounded, and so is the result of every
operation of `Transformer.run`, each computed exactly and rounded once, as a matmul accumulates in
float32 and stores bf16: the embedding, each norm, each matmul, RoPE, the scores, the softmax,
`gelu`, each product of gate and value, each addition to the stream.

Two choices of this model matter, and the theorem in bf16 is about the model so chosen:

* `one` is rounded to bf16 once, and that one number is both what the embedding holds and what
  each norm computes its scale from. Read literally, Python computes the scale from the float `one`
  while the stream holds its rounding; then at width 21 the norm gives each bit as `255/256`
  (proved in Lean for every rounding), and in a bf16 run on the real compiler's weights the outputs
  are not exact (CE-12). Rounding `one` once is a one-line change to the compiler:
  `one = round_to_bf16(2 * sqrt(width))`.
* RoPE's angles and trigonometry are exact, and only the rotated query and key are rounded. In
  float32, angles lose precision with the position: past about `2^24` a head reads the wrong token
  (CE-13, numeric). The norm's scale is computed, not a stored bf16 weight (`gemma.py`'s norms are
  weights, and it adds pads to make them exact).

## The proof

`Interfaces.lean` states each step as a proposition (`Stmt_*`); each file of `Proofs/` proves one,
taking others as hypotheses where it builds on them; `Final.lean` puts them together.

| Step | Statement |
|---|---|
| `Stmt_grid` | `token_grid` gives a grid the layers can follow: rows of distinct names, `ONE` in feature 0, each cell that changes gets an output of an MLP new at that depth, whose inputs the row before holds; the outputs end at features `1 .. bits` |
| `Stmt_trace` | an MLP's output in the trace is what it gives on its inputs' values |
| `Stmt_decomp`, `Stmt_unit_dots` | what a layer writes into each feature, unit by unit |
| `Stmt_compiled` | what `signed(convert(m, "geglu"))` computes, against its reglu form |
| `Stmt_norm_exact`, `Stmt_norm_close` | the norm on the stream a row describes, and near it |
| `Stmt_gelu`, `Stmt_round` | `gelu` is `relu` away from 0; what rounding to bf16 does |
| `Stmt_layer_gain`, `Stmt_layer_real`, `Stmt_layer_bf16` | one layer keeps the stream within `(1 + σ)` times its error of the one its row describes (in bf16: equal to it) |
| `Stmt_attn_real`, `Stmt_attn_bf16` | after the attention, the stream is row 0's: within `10^-10`, and exactly in bf16 |
| `Stmt_logit_real`, `Stmt_logit_real_E`, `Stmt_logit_bf16` | the right token has the largest logit |

The attention's margin needs `runner_up < 1`: in real arithmetic because `π` is irrational
(`cos e < 1` for a whole number `e ≠ 0`); in bf16, `runner_up ≤ 0.94` from bounds on cosines,
which leaves a margin of more than 14 after rounding the scores.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier</h1>

<p align="center">Compile algorithms into neural network circuits that stay interpretable, in a few hundred lines.</p>

Reifier needs Python 3.14 and numpy. It is one small compiler in three stages, from threshold
gates to a transformer. Each stage can be printed, and each is checked against the one before.

```python
from reifier import check_net, check_trace, circuit, emit, layout, threshold_gate, trace


@circuit
def xor(xs):
    n = len(xs)
    counts = [threshold_gate(xs, [1] * n, i + 1, f"atleast{i + 1}") for i in range(n)]
    return threshold_gate(counts, [(-1) ** i for i in range(n)], 1, "odd")


t = trace(xor, 3)  # stage 1: a flat list of MLPs, named by scope
check_trace(t, lambda x: [sum(x) % 2])  # it is xor, on all 8 inputs
print(t)
```

```
xor(xs0 xs1 xs2): 4 MLPs
  xor/atleast1 = [xs0 + xs1 + xs2 >= 1]
  xor/atleast2 = [xs0 + xs1 + xs2 >= 2]
  xor/atleast3 = [xs0 + xs1 + xs2 >= 3]
  xor/odd = [xor/atleast1 - xor/atleast2 + xor/atleast3 >= 1]
  return xor/odd
```

A threshold gate prints as `[what it sums >= its threshold]`.

```python
g = layout(t)  # stage 2: which signal each feature holds at each depth
print(g)
```

```
feature  depth 0  depth 1       depth 2
0        ONE      "             "
1        xs0      xor/atleast3  xor/odd
2        xs1      xor/atleast2  .
3        xs2      xor/atleast1  .
```

A feature is one of the numbers that a network holds between two layers, and the depth says
between which: the features at one depth are a row of the grid, printed here as a column. `"`
is the signal of the depth before, and `.` no signal. `ONE` is the constant 1.

```python
net = emit(t, g, "step")  # stage 3: weights, once g is checked against t
check_net(net)  # every feature holds, at every depth, what g names there
print(net)
print(net("101"))
```

```
xor: stack, 4 features, ONE = 1
layer 1: 4 hidden units of step
  ONE = step(0.5 ONE) -> f0
  xor/atleast1 = step(-ONE + xs0 + xs1 + xs2) -> f3
  xor/atleast2 = step(-2 ONE + xs0 + xs1 + xs2) -> f2
  xor/atleast3 = step(-3 ONE + xs0 + xs1 + xs2) -> f1
layer 2: 2 hidden units of step
  ONE = step(0.5 ONE) -> f0
  xor/odd = step(-ONE + xor/atleast3 - xor/atleast2 + xor/atleast1) -> f1
[0.0]
```

Each line is a hidden unit: its name, what it reads, by the names of the row before, and the
features it writes. Biases are weights on `ONE`, so the gate `[xs0 + xs1 + xs2 >= 2]` is
`step(-2 ONE + xs0 + xs1 + xs2)`. A stack writes every feature again, so the unit `ONE` copies
`ONE`: `step(0.5 ONE)` is `[ONE >= ONE / 2]`.

The same grid gives other networks. `emit(t, g, "relu", skip=True)` has a skip connection around
each layer, and its relu units `u+` and `u-` are the gate `u`. `emit(t, g, "geglu", skip=True,
norm=True)` has an RMSNorm before each layer too. And a program on tokens gives a transformer, in
numpy:

```python
from reifier import trace, transformer
from reifier.programs.adder import CODES, SYMBOLS, WINDOW, next_token

program = trace(next_token, len(CODES) * WINDOW)  # the bits of 8 tokens -> the bits of the next
m = transformer(program, {i: CODES[c] for i, c in enumerate(SYMBOLS)})
tokens = m.generate([SYMBOLS.index(c) for c in "123+246="], 3)
print("".join(SYMBOLS[i] for i in tokens))
```

```
123+246=369
```

## Design

**Stage 1, the trace** (`trace.py`). A signal is a bit on a wire: while tracing it has a name
and no value. The only operation is the MLP, `wo @ act(wi @ x + b)`: a threshold gate is an MLP
of activation `"step"`, and a GLU an MLP of a gated activation such as `"reglu"`, which
multiplies relu of the left half of `wi @ x + b` by its right half. `trace(fn, *sizes)` runs a
Python function once and keeps, in order, the MLPs it makes that its outputs depend on: a flat
list. `@circuit`, or `scope`, names what is made inside it, so `add/xor.1/odd` is the gate `odd`
of the second `xor` of `add`. `convert` turns step MLPs into relu MLPs, relu into reglu, and
reglu into swiglu or geglu: exactly, for gates of whole weights and for the GLUs of `lib.py`.
`check_trace` compares a trace with a Python function, or with the trace before a pass.

**Stage 2, the grid** (`grid.py`). `rows[d][f]` is the name of the signal that feature `f` holds
at depth `d`. ONE, the constant, is feature 0 of every row: a layer reads it as 1, and its
weights on it are its biases. `layout` puts each MLP right after the last it reads, and gives
each signal a feature from the depth that makes it to the last that reads it: the feature freed
last, or a new one. A grid holds no weights, and `check_grid` says what makes any grid, however
built, a grid of its trace.

**Stage 3, the weights** (`emit.py`). `emit(t, g, activation, skip, norm)` makes each row of
the grid from the row before, with one MLP per layer: the trace's MLPs of that depth side by
side, under their own names, and copies for the rest. A stack must write all that a row holds,
so `keep s` copies a signal that stays. With skip connections a layer writes only what changes,
so `erase s` subtracts a signal that ends; `held(g)` is the grid in which a signal stays until
another is written over it, so that less is erased. With norms a bit is held as -1 or 1, so that
the norm of a row is known from the grid and an RMSNorm is a fixed scaling; with skip connections
too, the stream holds ONE as 2 sqrt(features), which keeps what is a little off from growing, and
the norm gives it to the layer as 1. `check_net` runs the network and compares every feature at
every depth with the trace's value of the signal that the grid names there.

**The transformer** (`transformer.py`) is those layers, with skip connections and norms, on
tokens. Each token carries a code of bits, and its embedding is ONE and its code. One attention
layer makes row 0 of the grid: head h attends to the token h back, through RoPE, and copies its
code into the features of the program's inputs. The program's outputs are pinned over the
token's own code, so the logits, tied to the embedding, are largest for the token whose code
the program gives. `check_transformer` compares its stream with the grid at every position.

**Outside the core**, on the same trace and grid: `fold` works out what constants decide;
`tiles` is a layout that keeps each circuit together, as a rectangle of the grid; `gemma.py`
writes the transformer into the weights of google/gemma-4-E4B, with the norms, the shared heads
and the `<bos>` that Gemma adds, and the names of its features and hidden units beside them.

| File | Contents |
|---|---|
| `src/reifier/trace.py` | signals, the MLP and its activations, `trace`, `scope`, `convert`, `check_trace` |
| `src/reifier/grid.py` | the grid, `layout`, `held`, `check_grid` |
| `src/reifier/emit.py` | stacks, skip connections and norms: `emit`, `check_net` |
| `src/reifier/transformer.py` | a transformer in numpy with lookback attention, `check_transformer` |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor`, `parity`, `add`, `sum_of_ands` |
| `src/reifier/fold.py` | `fold`, a pass that works out what constants decide |
| `src/reifier/tiles.py` | `tiles`, a layout of rectangles |
| `src/reifier/gemma.py` | a program in the weights of google/gemma-4-E4B, from the last tokens' codes to the next token's |
| `src/reifier/programs/keccak.py` | Keccak-p and TurboSHAKE128, each bit one GLU on 3 or 5 bits |
| `src/reifier/programs/adder.py` | decimal addition on tokens: after "123+246=" come "3", "6" and "9" |
| `examples/gemma_keccak.py` | TurboSHAKE128 with Keccak-p[1600, 6] compiled into google/gemma-4-E4B |
| `examples/gemma_adder.py` | that addition compiled into google/gemma-4-E4B |

Run the tests with `uv run pytest`.

Not here yet: PyTorch export, layers as sparse matrices for very large circuits, and signals
that are not bits.

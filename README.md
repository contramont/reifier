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
from reifier.check import check_grid, check_net, check_trace
from reifier.emit import emit
from reifier.grid import layout
from reifier.show import show
from reifier.trace import circuit, threshold_gate, trace


@circuit
def xor(xs):
    n = len(xs)
    counts = [threshold_gate(xs, [1] * n, i + 1, f"atleast{i + 1}") for i in range(n)]
    return threshold_gate(counts, [(-1) ** i for i in range(n)], 1, "odd")


t = trace(xor, 3)  # stage 1: a flat list of MLPs, named by the circuits that make them
check_trace(t, lambda x: sum(x) % 2)  # it is xor, on all 8 inputs
print(show(t))
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
check_grid(t, g)  # each gate comes after what it reads
print(show(g))
```

```
feature  depth 0  depth 1       depth 2
0        ONE      "             "
1        xs0      xor/atleast3  xor/odd
2        xs1      xor/atleast2  .
3        xs2      xor/atleast1  .
```

A feature is one of the numbers that a network holds between two layers, and the depth says
between which. `g[d]`, row d of the grid, lists what each feature holds at depth d: it prints
as a column, with a line for each feature. `"` is the signal of the depth before, and `.` no
signal. `ONE` is the constant 1.

```python
net = emit(t, g, "step")  # stage 3: weights
check_net(net)  # every feature holds, at every depth, what g names there
print(show(net))
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

Each line is a hidden unit: its name, what it reads, by the names of the depth before, and
the features it writes. Biases are weights on `ONE`, so the gate `[xs0 + xs1 + xs2 >= 2]` is
`step(-2 ONE + xs0 + xs1 + xs2)`. A stack writes again every feature that holds a signal, and
a step unit copies a bit s as `[s >= 1/2]`, which for `ONE` itself is `step(0.5 ONE)`.

The same grid gives other networks. `emit(t, g, "relu", skip=True)` has a skip connection around
each layer, and each gate `u` is two relu units, `u+` and `u-`, whose difference is the gate.
`emit(t, g, "geglu", skip=True, norm=True)` has an RMSNorm before each layer too. With `held(g)`
for `g`, either leaves a signal in its feature until another is written there, and erases less.
And a program on tokens gives a transformer, in numpy, here from an adder of threshold gates:

```python
from reifier.lib import add
from reifier.transformer import transformer


def count(bits):  # the bits of the last two tokens -> the bits of the next: their sum, mod 4
    return add(bits[:2], bits[2:])


m = transformer(trace(count, 4), {n: [n >> 1, n & 1] for n in range(4)})  # each token, its bits
print(m.generate([1, 1], 6))
```

```
[1, 1, 2, 3, 1, 0, 1, 1]
```

## Reading the code

The files are in `src/reifier`. The first four are the compiler, about 520 lines to be read
in this order. Each opens with what its stage is, and holds nothing but what makes it: printing
and checking are in the next two. The rest is outside the compiler, on the same trace and grid.

| File | Contents |
|---|---|
| `trace.py` | stage 1: signals, the MLP and its activations, `circuit`, `trace` |
| `grid.py` | stage 2: `layout`, and `held`, the grid as skip connections hold it |
| `emit.py` | stage 3: `convert`, `layer` and `emit`, for stacks, skip connections and norms |
| `transformer.py` | tokens: lookback attention in front of those layers, in numpy |
| `show.py` | how each stage prints |
| `check.py` | each stage checked against the one before |
| `lib.py` | circuits: `not_`, `and_`, `or_`, `xor`, `parity`, `add`, `sum_of_ands` |
| `fold.py` | `fold`, a pass on traces that moves constants into the biases of what reads them |
| `tiles.py` | `tiles`, a layout that keeps each circuit together, as a rectangle of the grid |
| `gemma.py` | the transformer written into the checkpoint of google/gemma-4-E4B |

The two examples are programs on tokens, each with the lines that compile it into
google/gemma-4-E4B and write the names of its features and hidden units beside the weights:
`examples/adder.py` is decimal addition, where after "123+246=" come "3", "6" and "9", and
`examples/keccak.py` is a sponge on Keccak-p[400, 6], each bit one GLU on at most 5 bits. Run the
tests with `uv run pytest`.

Not here yet: PyTorch export, layers as sparse matrices for very large circuits, and signals
that are not bits.

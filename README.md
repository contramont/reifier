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
With [uv](https://docs.astral.sh/uv/), `uv run python` runs the code below, and `uv run pytest`
the tests.

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
`step(-2 ONE + xs0 + xs1 + xs2)`. A stack, of layers with no skip connections around them,
writes again every feature that holds a signal, and a step unit copies a bit s as `[s >= 1/2]`,
which for `ONE` itself is `step(0.5 ONE)`.

The same grid gives other networks. `emit(t, g, "relu", skip=True)` has a skip connection around
each layer, and each gate `u` is two relu units, `u+` and `u-`, whose difference is the gate.
`emit(t, g, "geglu", skip=True, norm=True)` has an RMSNorm before each layer too. Given
`held(g)` in place of `g` (it is in `reifier.grid`), either leaves a signal in its feature until
another is written there, and erases less.

A program on tokens gives a transformer, in numpy. Its layers are not emit's: it is shaped as
Gemma is, with an RMSNorm before and after the attention and each block, and its weights and
activations are of the sizes of a trained Gemma's. A bit is 2 in its stream, where a feature of
a token's embedding in Gemma is about 1, and written into Gemma 3 270M, the examples below have
no weight larger than the largest of Gemma's own of its kind. A `table` of `reifier.lib` is any
function of a few bits as one GLU, whose units pass on only the square of what the bits they
read are off by: a circuit of them is not much moved by weights that are a little off, up to a
point, and `examples/measure.py`, below, says where. Here, from an adder of threshold gates:

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

The files are in `src/reifier`. The first four are the compiler, about 800 lines to be read
in this order. Each opens with what its stage is, and holds nothing but what makes it: printing
and checking are in the next two. The rest is outside the compiler, on the same trace and grid.

| File | Contents |
|---|---|
| `trace.py` | stage 1: signals, the MLP and its activations, `circuit`, `trace` |
| `grid.py` | stage 2: `layout`, and `held`, the grid as skip connections hold it (`rest`, `first` and `home` are for `transformer.py`) |
| `emit.py` | stage 3: `convert`, `layer` and `emit`, for stacks, skip connections and norms |
| `transformer.py` | tokens: a transformer shaped as Gemma is, in numpy: lookback attention, then a block of GLUs for each row, each between two RMSNorms |
| `show.py` | how each stage prints |
| `check.py` | each stage checked against the one before |
| `lib.py` | circuits: `not_`, `and_`, `or_`, `xor`, `parity`, `add`, `sum_of_ands`, `sum_of_parities`, `table`, `ands`, `odd` |
| `fold.py` | `fold`, a pass on traces that moves constants into the biases of what reads them |
| `relay.py` | `relay`, a pass that passes each signal on, so that an MLP reads only the layer before it |
| `tiles.py` | `tiles`, a layout that keeps each circuit together, as a rectangle of the grid |
| `gemma.py` | the transformer written into a checkpoint of Gemma 3 or Gemma 4, which has a norm wherever it does |
| `beside.py` | the transformer and Gemma 3's own weights in one checkpoint, twice as wide: its blocks in front of Gemma's layers, and layers that carry what they give to the logits |

The examples are programs on tokens, each with the lines that compile it into
unsloth/gemma-3-270m, or beside it, and write the names of its features and hidden units beside
the weights. Each file's docstring says how to run it; the first run fetches from huggingface.co.

- `examples/adder.py` is decimal addition: after "123+246=" come "3", "6" and "9".
- `examples/keccak.py` is a sponge on Keccak-p[50, 2], of `table`s on at most 3 bits each.
- `examples/alongside.py` is a sponge on Keccak-p[200, 7] beside Gemma itself, in one bf16
  model twice as wide and 53 layers deep: the sponge's 30 in front of Gemma's 18, four among
  those that lift what it carries, as Gemma's stream grows, and one after them that makes it
  loud for the logits. Every token has a code of 144 bits, and any two codes differ in a
  quarter of their bits or more. Where the sponge gives, for the last two tokens, the code of a
  token, half of the time, the next token is that one, and elsewhere what Gemma gives. It
  compiles in about 11 seconds.
- `examples/measure.py` says in a minute, on a GPU, how well the two of alongside.py get on:
  how often each is right, how large the sponge's weights and activations are beside Gemma's
  own, and how much noise on the weights each takes.

On an H100, on the text of `src/reifier`, it says: where the sponge speaks, at half of the
positions, its token is the model's at every one, by 36 or more in the logits. Where it is
silent, the model's token is that of Gemma alone, as wide, at 98 positions in 100: no further
from it than Gemma alone is from Gemma itself, whose weights it has, rounded to bf16 again. No
weight of the sponge's is larger than the largest of Gemma's own of its kind: the nearest are
those of a head's value, at 0.93 of it, and of a gate, at 0.85. Nor is its stream, what its
blocks read, or a value or a hidden unit of theirs larger than the largest in Gemma's layers.
A gate of its last layer is, 54.5 against 47.8, and so is what that layer gives, 164 against
129, where Gemma's stream is small: on random tokens both are under Gemma's. And the logit of
the sponge's token is larger by far, 157 as a rule and up to 290, where those of Gemma alone
are 63 at most: the token of a code must win over every other, the nearest of which has half
of its logit, by more than Gemma's own logits differ.

Noise on every weight leaves the sponge exact up to a point, and past it wrong everywhere,
where Gemma fades. With every weight times 1 + 5% noise, Gemma alone keeps its own token at
seven positions in ten, and the sponge's token is the model's at 9,994 in 10,000; at 7%, at a
third, where Gemma keeps six in ten; at 10%, nowhere. With noise added to every projection and
the embedding, of 5% of the root mean square of Gemma's own of its kind, the sponge keeps all
of its tokens and Gemma eight in ten, and of 10%, nine in ten and six. With those rounded to 6
bits it keeps all, and Gemma three in ten. But some noise breaks it sooner than Gemma. Added to
the weights of norms, noise of 0.03% of the root mean square of Gemma's own leaves all of its
tokens, and of 0.1% none, where Gemma keeps eight in ten: the weights of the norms after its
blocks are about 1, where Gemma's are 21 as a rule and up to 1,660. And with one weight in
1,000 of a projection or the embedding set to 0, it keeps 96 in 100 of its tokens and Gemma 88
of its own; with one in 100, almost none, and Gemma six in ten. Where its token is lost,
Gemma's is too, more often than in Gemma alone.

Not here yet: PyTorch export, layers as sparse matrices for very large circuits, and signals
that are not bits.

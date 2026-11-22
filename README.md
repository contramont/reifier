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
most of its activations are of the sizes of a trained Gemma's. A bit is 2 in its stream, where
a feature of a token's embedding in Gemma is about 1, and written into Gemma 3 270M, the
examples below have no weight larger than the largest of Gemma's own of its kind, but those of
one norm in `examples/alongside.py`. A block writes a new bit where the stream held nothing,
or over a bit that it reads for the last time, so the stream needs few features: for Keccak,
little more than the state. A `table` of `reifier.lib` is any function of a few bits as one
GLU, whose units pass on only the square of what the bits they read are off by: a circuit of
them is not much moved by weights that are a little off, up to a point, and
`examples/measure.py`, below, says where. A `cover` and `steps` make functions of more bits in
fewer units, and take less noise. Here, from an adder of threshold gates:

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

The files are in `src/reifier`. The first four are the compiler, about 950 lines to be read
in this order. Each opens with what its stage is, and holds nothing but what makes it: printing
and checking are in the next two. The rest is outside the compiler, on the same trace and grid.

| File | Contents |
|---|---|
| `trace.py` | stage 1: signals, the MLP and its activations, `circuit`, `trace` |
| `grid.py` | stage 2: `layout`, and `held`, the grid as skip connections hold it (`rest`, `first` and `home` are for `transformer.py`) |
| `emit.py` | stage 3: `convert`, `layer` and `emit`, for stacks, skip connections and norms |
| `transformer.py` | tokens: a transformer shaped as Gemma is, in numpy: lookback attention, then a block of GLUs for each row, each between two RMSNorms, which writes its bits where the row before held nothing, or over bits that it has read |
| `show.py` | how each stage prints |
| `check.py` | each stage checked against the one before |
| `lib.py` | circuits: `not_`, `and_`, `or_`, `xor`, `parity`, `add`, `sum_of_ands`, `sum_of_parities`, `table`, `ands`, `odd`, `cover`, `steps` |
| `fold.py` | `fold`, a pass on traces that moves constants into the biases of what reads them, and takes out the MLPs that are left as wires |
| `relay.py` | `relay`, a pass that passes each signal on, so that an MLP reads only the layer before it |
| `tiles.py` | `tiles`, a layout that keeps each circuit together, as a rectangle of the grid |
| `gemma.py` | the transformer written into a checkpoint of Gemma 3 or Gemma 4, which has a norm wherever it does |
| `beside.py` | the transformer and Gemma 3's own weights in one checkpoint, wider by the transformer's features: its blocks in front of Gemma's layers, and one layer after them that makes what they gave loud for the logits |

The examples are programs on tokens, each with the lines that compile it into
unsloth/gemma-3-270m, or beside it, and write the names of its features and hidden units beside
the weights. Each file's docstring says how to run it; the first run fetches from huggingface.co.

- `examples/adder.py` is decimal addition: after "123+246=" come "3", "6" and "9".
- `examples/keccak.py` is a sponge on Keccak-p[50, 2], of `table`s on at most 5 bits each, in
  7 of Gemma's layers and 66 of its features.
- `examples/alongside.py` is a sponge on Keccak-p[200, 7] beside Gemma itself, in one bf16
  model of 34 layers and 896 features: the sponge's 15 layers, two a round and one for the
  code of its token, on 256 features, in front of Gemma's 18, and one after them that makes
  what the sponge gave loud for the logits. Every token has a code of 72 bits, and any two
  codes differ in 16 bits or more. Where the sponge gives, for the last two tokens, the code
  of a token, half of the time, the next token is that one, and elsewhere what Gemma gives.
  It compiles in about 7 seconds, into 1.37 GB.
- `examples/measure.py` says in a minute, on a GPU, how well the two of alongside.py get on:
  how often each is right, how large the sponge's weights and activations are beside Gemma's
  own, and how much noise on the weights each takes.

On an H100, on the text of `src/reifier`, it says: where the sponge speaks, at half of the
positions, its token is the model's at every one, by 14 or more in the logits. Where it is
silent, the model's token is that of Gemma alone, as wide, at 98 positions in 100: about as far
from it as Gemma alone is from Gemma itself, whose weights it has, rounded to bf16 again. The
logit of the sponge's token is about 68 as a rule and under 85 at most, where Gemma's top logit
is under 50: the token of a code must win over every other, the nearest of which is four
ninths of its logit behind, by more than Gemma's own logits differ. No weight of the sponge's
is larger than the largest of Gemma's own of its kind, but those of the norm after the block
of its last layer, 3,200, 1.9 times Gemma's largest, which keep that logit nearly the same at
every token: the nearest of the others are those of a head's value, at 0.93 of Gemma's
largest, and of a gate, at 0.92. Its stream, and what its blocks read and give, are under the
largest in Gemma's layers, but some of what they hold is not: the gates of its `steps` reach
nearly three times Gemma's largest, and a value and a hidden unit are 1.2 and 1.1 times.

Noise on every weight leaves the sponge exact up to a point, and past it wrong everywhere, where
Gemma fades. With every weight times 1 + 1% noise, the sponge's token is the model's at all but
one position in 10,000, and Gemma alone keeps its own token at 93 in 100; at 1.5%, at half,
where Gemma keeps nine in ten; at 2%, nowhere. With noise added to every projection and the
embedding, of 1% of the root mean square of Gemma's own of its kind, the sponge keeps all of its
tokens, and of 2%, three in four, where Gemma keeps 95 in 100 and nine in ten; of 3%, none.
With those rounded to 6 bits it keeps all, where Gemma keeps three in ten, and to 5 bits none.
Two kinds of noise break it far sooner than Gemma. With one weight in 10,000 of a projection or
the embedding set to 0, it keeps three in ten of its tokens, where Gemma keeps more than 95 in
100 of its own; with one in 1,000, none. And added to the weights of norms, noise of 0.01% of
the root mean square of Gemma's own leaves all of its tokens, and of 0.03% four in ten, where
Gemma keeps 94 in 100: the weights of the norms after its blocks, the last aside, are 2 at
most, where Gemma's are 21 as a rule and up to 1,660. Where its token is lost, Gemma's is too,
more often than in Gemma alone.

The model is made to be small, and gave up much of what a larger one had. Here it is against
the same sponge in the design of commit 7e2804a, and in rounds of three layers, which
`KINDS = "3333333"` in alongside.py still gives, each as `examples/measure.py` measures it:

| | commit 7e2804a | rounds of three layers | now |
|---|---|---|---|
| layers | 53 | 40 | 34 |
| the sponge's features, of the stream's | 601 of 1,280 | 256 of 896 | 256 of 896 |
| file | 2.69 GB | 1.53 GB | 1.37 GB |
| right with every weight times 1 + noise of | 5%, all but 6 in 10,000 | 4%, all but 4 in 1,000 | 1%, all but 1 in 10,000 |
| its largest weight, of Gemma's largest of its kind | 0.93 | 1.9, of one norm | 1.9, of one norm |
| its largest gate, of Gemma's largest | 1.1 | 1.2 | nearly 3 |

The design of 7e2804a also had four layers that made what the sponge carries past Gemma's
layers larger on the way. This one has none: with noise in the stream itself, of 0.1% of its
size after every layer, the sponge loses one token in seven, where one such layer more keeps
them all. And noise on the weights of the norms loses Gemma more of its own tokens than it does
in Gemma alone (`reifier/beside.py` has the numbers). Rounds of two layers need about 20 hidden
units a layer for each bit of the state: Keccak-p[200] fits in 4,096, and Keccak-p[400] would
need more than 8,000.

Not here yet: PyTorch export, layers as sparse matrices for very large circuits, and signals
that are not bits.

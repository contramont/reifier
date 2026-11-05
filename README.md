<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier, simple</h1>

<p align="center">Compile algorithms into neural network circuits, in a few hundred lines.</p>

This is the `simple` branch: a from-scratch rewrite of reifier. It needs Python 3.14 and numpy.
See the stages of compiling in the browser at
[contramont.org/reifier/simple](https://contramont.org/reifier/simple/), which runs this
branch's code.

```python
from reifier import bits, circuit, gate, network

@circuit
def xor(xs):
    n = len(xs)
    counts = [gate(xs, [1] * n, i + 1) for i in range(n)]  # at least i + 1 ones
    return gate(counts, [(-1) ** i for i in range(n)], 1)

graph = xor.trace(bits(3))
print(graph.eval("101"))             # [0.0]
net = network(graph, "geglu", norm=True)
print(net.eval([1, 1, 0, 1]))        # [b, 0.0]: BOS first, outputs relative to it
```

## Design

**The core** (`core.py`) is abstract: a `Signal` is a wire with a kind (`"bit"`, `"int"`,
`"real"`, ...) but no value, and a `Circuit` maps signals to signals. A `Graph` is made of
circuits; `@circuit` traces a function into one, once per argument structure, and later calls
reuse it. `flatten` and `lower` are passes from graphs to graphs, and `lower` uses a circuit's
hand-written version for a dialect where it has one (`@xor.impl("reglu")`).

**The leaf** (`units.py`) is the `Unit`, `wo @ f(wi @ x + b)`: linear, a function, linear. A
gate is a unit with `f = "step"`; a GLU is a unit with a gated `f` such as `"reglu"`, which
multiplies relu of the left half of `wi @ x + b` by its right half. `cast` turns step units
into relu units, relu into reglu, and reglu into swiglu or geglu.

**Local structure** (`place.py`): `place(graph, f)` puts units on a grid of layers and
features. Each circuit is placed once, as a block: a rectangle as wide as its widest level, its
inputs included, with the features it does not use left empty; a unit is as wide as its inputs or
its outputs, whichever are more. Each use of a circuit puts its block in the layers after
its inputs, at the first features free on all of its levels, so what a circuit holds stays
together and in place. Units keep their weights: placing one only wires it to the level below.
Copies carry signals straight up to the levels that read them. The result is a chain of layer
graphs, each as wide as the others.

**Global structure** (`target.py`): `network(graph, f, norm, ff)` builds a target architecture
around the placed units. Each layer holds its units side by side as one unit, `ff` times as wide
inside as outside (4 by default), with each unit's hidden units under its features. Global parts
act on whole layers: BOS = 1 leads every layer and carries the biases, and an RMSNorm can come
before each layer. The matrices of a block's layer are built once and copied into each use.
Targets include MLPs of step or relu units and GLU layers (reglu, swiglu, geglu) with or without
RMSNorms.

| File | Contents |
|---|---|
| `src/reifier/core.py` | signals, circuits, graphs, tracing, `flatten`, `lower`, `show` |
| `src/reifier/units.py` | the unit, its functions, gates, GLUs, `cast` |
| `src/reifier/place.py` | placing units in layers |
| `src/reifier/target.py` | architectures: BOS, RMSNorm, one unit per layer |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor` (with relu and reglu versions), `add`, `pick`, `popcount`, `at_least` |
| `src/reifier/gemma.py` | Gemma 4 as a target: attention that gathers the prompt, a program as feed-forward blocks |
| `examples/gemma_ffn.py` | a circuit on reals: Gemma's feed-forward block as three units |
| `examples/gemma_adder.py` | an adder of n-bit numbers in the weights of google/gemma-4-E4B |
| `examples/gemma_keccak.py` | Keccak-p[200, 6] in the weights of google/gemma-4-E4B |

**Gemma** (`gemma.py`): `weights(program, ...)` compiles a program into google/gemma-4-E4B,
for prompts of `<bos>`, other tokens and bit tokens. Attention gives each token what the last
tokens are, 8 at a time and then 8 of those windows at a time, up to 512 back; a gate marks
the prompt's last bit by them, and attention gathers all the prompt's bits from there. The
program, in the feed-forward blocks, gives the next bit token: each of its layers writes its
bits into one of two banks of columns and erases the bank it read. These write a 4-bit and a
16-bit adder and Keccak-p[200, 6] to `solution`, `solution2` and `solution3`:

```
uv run --with torch==2.15.0 --with transformers==5.22.1 --with safetensors==0.8.0 \
    examples/gemma_adder.py 4 solution    # or 16 solution2; or examples/gemma_keccak.py
```

Run the tests with `uv run pytest`.

Not here yet: skip connections and attention as global parts (`gemma.py` lays them out by
hand, for one model), PyTorch export, and fast compilation of very large circuits such as
SHA-3 (all but the first are on `main`).

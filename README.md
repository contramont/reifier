<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier, simple</h1>

<p align="center">Compile algorithms into neural network circuits, in a few hundred lines.</p>

This is the `simple` branch: a from-scratch rewrite of reifier around `Signal` and `Circuit`.
It needs Python 3.14 and numpy. Try it in the browser at
[contramont.org/reifier/simple](https://contramont.org/reifier/simple/), which runs this
branch's code.

```python
from reifier import bits, circuit, gate, show, to_swiglu

@circuit
def xor(xs):
    n = len(xs)
    counts = [gate(xs, [1] * n, i + 1) for i in range(n)]  # at least i + 1 ones
    return gate(counts, [(-1) ** i for i in range(n)], 1)

graph = xor.trace(bits(3))
print(show(graph))             # the circuit as text, each sub-circuit once
print(graph.run("101"))        # [[0.]]
net = to_swiglu(graph)         # 2 SwiGLU layers, also a circuit
print(net.run("1101").round()) # [[1. 0.]]: BOS = 1 first, in and out
```

## Design

- **A `Signal` is a wire with a kind but no value** (`"bit"`, `"int"`, `"real"`, ...).
  Branching on a bit raises an error, so a circuit's structure depends only on the shapes of
  its arguments. Values come from `graph.run`, on whole batches at once.
- **A `Circuit` maps signals to signals.** A leaf circuit is a class with `kinds` (one per
  output) and `eval` (numpy rows in, rows out); a `Graph` is made of other circuits. The core
  (`core.py`) knows no leaf types, so a new kind of circuit, such as an RMSNorm, is a few lines
  outside it ([examples/gemma_ffn.py](examples/gemma_ffn.py) builds Gemma's feed-forward block).
- **`@circuit` traces a function once per argument structure.** Later calls with the same
  shapes, kinds and static arguments add one node that reuses the trace. Undecorated functions
  are inlined into their callers.
- **Passes interpret a graph into a new graph.** `flatten` inlines sub-circuits; `lower(graph,
  dialect)` rewrites leaves with `RULES[dialect, type]` and uses a circuit's hand-written version
  for the dialect where it has one:

  ```python
  @xor.impl("relu")
  def xor_relu(xs): ...  # one layer of units instead of two layers of gates
  ```

## One leaf: the GLU

`glu.py` has the only leaf the compiler needs, `GLU`: `wo @ (act(wg @ x + bg) * (wv @ x + bv))`,
the feed-forward block of transformers. Its activation is its dialect:

| Activation | What it is |
|---|---|
| `"step"` | linear threshold gates, which `gate()` makes: the circuits you write |
| `"relu"` | what `layers` stacks: a step is two relus, exact on integer sums |
| `"silu"` | SwiGLU: `silu(kz) / k` is `relu(z)` to within `exp(-k\|z\|)` |

Every stage of compiling is a circuit, so it runs and shows like the program:

```python
lower(graph, "relu")  # gates as relu units, hand-written versions where there are some
layers(graph)         # one relu GLU per layer; copies carry signals, BOS carries biases
to_swiglu(graph)      # the same layers with silu, as in a transformer
```

## Layout

| File | Contents |
|---|---|
| `src/reifier/core.py` | signals, circuits, graphs, tracing, `flatten`, `lower`, `show` |
| `src/reifier/glu.py` | the GLU, gates, dialect rules, `layers` and `to_swiglu` |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor` (with a one-layer version), `add`, `popcount`, `at_least` |
| `examples/gemma_ffn.py` | a circuit on reals: a new leaf (RMSNorm) and Gemma's FFN block |

Run the tests with `uv run pytest`.

Not here yet, and on `main`: optimization levels, PyTorch and `.safetensors` export, RMSNorm
layers in compiled networks, and fast compilation of very large circuits such as SHA-3.

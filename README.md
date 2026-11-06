<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier, simple</h1>

<p align="center">Compile algorithms into neural network circuits, in a few hundred lines.</p>

This is the `simple` branch: a from-scratch rewrite of reifier. It needs Python 3.14 and numpy.

```python
from reifier import bits, circuit, network, threshold_gate

@circuit
def xor(xs):
    n = len(xs)
    counts = [threshold_gate(xs, [1] * n, i + 1) for i in range(n)]  # at least i + 1 ones
    return threshold_gate(counts, [(-1) ** i for i in range(n)], 1)

graph = xor.trace(bits(3))
print(graph.eval("101"))             # [0.0]
net = network(graph, "geglu", pre_norm=True)
print(net.eval([1, 1, 0, 1]))        # [b, 0.0]: ONE first, outputs relative to it
```

## Design

**The core** (`core.py`) is abstract: a `Signal` is a wire with a kind (`"bit"`, `"int"`,
`"real"`, ...) but no value, and a `Circuit` maps signals to signals. A `Graph` is made of
circuits; `@circuit` traces a function into one, once per argument structure, and later calls
reuse it. `flatten` and `lower` are passes from graphs to graphs, and `lower` uses a circuit's
hand-written version for a dialect where it has one (`@xor.impl("reglu")`).

**The leaf** (`mlp.py`) is the `MLP`, `wo @ act(wi @ x + b)`: linear, an activation, linear. A
threshold gate is an MLP with activation `"step"`; a GLU is an MLP with a gated activation such
as `"reglu"`, which multiplies relu of the left half of `wi @ x + b` by its right half. `convert`
turns step MLPs into relu MLPs, relu into reglu, and reglu into swiglu or geglu.

**Local structure** (`place.py`): `place(graph, activation)` puts MLPs on a grid of layers and
features. Each circuit is placed once, as a tile: a rectangle as wide as its widest row of
features, its inputs included, with the features it does not use left empty; an MLP is as wide as
its inputs or its outputs, whichever are more. Each use of a circuit puts its tile in the layers
after its inputs, at the first features free at all of its depths, so what a circuit holds stays
together and in place. MLPs keep their weights: placing one only wires it to the depth below.
Passthroughs carry signals straight up to the depths that read them. The result is a chain of
layer graphs, each as wide as the others.

**Global structure** (`network.py`): `network(graph, activation, pre_norm, mlp_ratio)` builds a
network around the placed MLPs. Each layer holds its MLPs side by side as one MLP, `mlp_ratio`
times as wide inside as outside (4 by default), with each MLP's hidden units under its features.
Global parts act on whole layers: ONE = 1 leads every layer and carries the biases, and an
RMSNorm can come before each layer. The matrices of a tile's layer are built once and copied into
each use. Networks include stacks of step or relu MLPs and GLU layers (reglu, swiglu, geglu) with
or without RMSNorms.

| File | Contents |
|---|---|
| `src/reifier/core.py` | signals, circuits, graphs, tracing, `flatten`, `lower`, `format_graph` |
| `src/reifier/mlp.py` | the MLP, its activations, threshold gates, GLUs, `convert` |
| `src/reifier/fold.py` | `fold_constants`, an optional pass that works out what constants decide |
| `src/reifier/place.py` | placing MLPs in layers |
| `src/reifier/network.py` | networks: ONE, RMSNorm, one MLP per layer |
| `src/reifier/gemma.py` | a program in the weights of google/gemma-4-E4B, from the last tokens' bits to the next token's |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor` (with relu and reglu versions), `add`, `popcount`, `at_least` |
| `src/reifier/keccak.py` | Keccak-p and TurboSHAKE128, each bit one GLU on 3 or 5 bits |
| `examples/gemma_ffn.py` | a circuit on reals: Gemma's feed-forward block as three MLPs |
| `examples/gemma_keccak.py` | TurboSHAKE128 with Keccak-p[1600, 6] compiled into google/gemma-4-E4B |

Run the tests with `uv run pytest`.

Not here yet: skip connections and attention as global parts, PyTorch export, and fast
compilation of very large circuits such as SHA-3 (all but the first are on `main`).

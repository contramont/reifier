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

**Local structure** (`place.py`): `place(graph, f)` puts each unit in the layer after its
inputs, as it is: placing a unit only wires it to the layer below. Copies carry signals to the
layers that read them. The result is a chain of layer graphs.

**Global structure** (`target.py`): `network(graph, f, norm)` builds a target architecture
around the placed units. Each layer holds its units side by side as one unit, and global parts
act on whole layers: BOS = 1 leads every layer and carries the biases, and an RMSNorm can come
before each layer. Targets include MLPs of step or relu units and GLU layers (reglu, swiglu,
geglu) with or without RMSNorms.

| File | Contents |
|---|---|
| `src/reifier/core.py` | signals, circuits, graphs, tracing, `flatten`, `lower`, `show` |
| `src/reifier/units.py` | the unit, its functions, gates, GLUs, `cast` |
| `src/reifier/place.py` | placing units in layers |
| `src/reifier/target.py` | architectures: BOS, RMSNorm, one unit per layer |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor` (with relu and reglu versions), `add`, `popcount`, `at_least` |
| `examples/gemma_ffn.py` | a circuit on reals: Gemma's feed-forward block as three units |

Run the tests with `uv run pytest`.

Not here yet: skip connections and attention as global parts, PyTorch export, and fast
compilation of very large circuits such as SHA-3 (all but the first are on `main`).

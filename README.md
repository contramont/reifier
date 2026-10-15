<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier, simple</h1>

<p align="center">Compile algorithms into neural network circuits, in under 500 lines.</p>

This is the `simple` branch: a from-scratch rewrite of reifier around two units, `Signal` and
`Circuit`. It needs Python 3.14 and numpy. Try it in the browser at
[contramont.org/reifier/simple](https://contramont.org/reifier/simple/), which runs this
branch's code.

```python
from reifier import bits, circuit, gate, run_swiglu, show, to_swiglu

@circuit
def xor(xs):
    n = len(xs)
    counts = [gate(xs, [1] * n, i + 1) for i in range(n)]  # at least i + 1 ones
    return gate(counts, [(-1) ** i for i in range(n)], 1)

graph = xor.trace(bits(3))
print(show(graph))             # the circuit as text, each sub-circuit once
print(graph.run("101"))        # [[0.]]
layers = to_swiglu(graph)      # 2 SwiGLU layers as numpy matrices
print(run_swiglu(layers, [[1, 0, 1]]).round())  # [[0.]]
```

## Design

- **Signals have no values.** A `Signal` is a wire with a kind (`"bit"`, `"int"` or
  `"real"`). Branching on a bit raises an error, so a circuit's structure can only depend on
  its arguments' shapes. Values come from `graph.run`, on whole batches at once.
- **`@circuit` traces once per argument structure.** The first call to a decorated function
  traces it into a `Graph`; later calls with the same shapes, kinds and static arguments add one
  node that points to that graph, without running the function. Undecorated functions are inlined
  into their callers.
- **Every pass is one interpreter.** `interpret(graph, args, prim)` walks a graph and calls
  `prim` on each op: evaluation, `flatten` and `lower` differ only in `prim`.
- **Circuit types are dialects.** A dialect is the set of primitives a graph may contain:
  `"threshold"` allows `Gate`, `"glu"` allows `GLU`. `lower(graph, "glu")` rewrites each gate
  with a rule (`@rule("glu", Gate)`), unless the circuit has a hand-written implementation for
  the dialect, as `xor` has in one layer instead of two:

  ```python
  @xor.impl("glu")
  def xor_glu(xs): ...
  ```

`to_swiglu` lowers to `"glu"`, flattens, gives each gate the earliest layer it can run in, adds
copies for signals that later layers read and folds constants into biases. Each layer maps
`h` to `wo @ (silu(wg @ h) * (wv @ h))`, where `h[0] = 1` carries the biases.

## Layout

| File | Contents |
|---|---|
| `src/reifier/core.py` | signals, primitives, circuits, passes, `show`, `verify` |
| `src/reifier/swiglu.py` | compilation to SwiGLU layers and a numpy forward pass |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor` (with its GLU implementation), `add`, `popcount`, `at_least` |

Run the tests with `uv run pytest`.

Not here yet, and on `main`: optimization levels, PyTorch and `.safetensors` export, RMSNorm
layers, and fast compilation of very large circuits such as SHA-3.

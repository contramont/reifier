<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier</h1>

<p align="center">Compile algorithms into neural network circuits, in a few hundred lines.</p>

Reifier needs Python 3.14 and numpy.

```python
from reifier import bits, circuit, network, threshold_gate


@circuit
def xor(xs):
    n = len(xs)
    counts = [threshold_gate(xs, [1] * n, i + 1) for i in range(n)]  # at least i + 1 ones
    return threshold_gate(counts, [(-1) ** i for i in range(n)], 1)


graph = xor.trace(bits(3))
print(graph.eval("101"))  # [0.0]
net = network(graph, "geglu", pre_norm=True)
print(net.eval([1, 1, 0, 1]))  # [b, 0.0]: ONE first, outputs relative to it
```

A program can read earlier tokens with `lookback`, an attention head, and compile into a
transformer. The compiler places the heads and the MLPs in the layers:

```python
from reifier import Architecture, bits, lookback, trace, transformer, xor
from reifier.decoder import predict


def changes(token):  # which of a token's 4 bits differ from the token before
    return [xor([a, b]) for a, b in zip(token, lookback(token, 1))]


program = trace(changes, bits(4))
print(program.eval_sequence(["1010", "1000"]).tolist())  # [[1, 0, 1, 0], [0, 0, 1, 0]]
small = Architecture(
    hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, head_dim=32, layer_types=("sliding_attention",) * 2,
    sliding_window=32, rope_theta=1e4, vocab_size=32, bos_token_id=2, token_offset=16,
)  # a small Gemma 4
weights = transformer(program, small)
print(predict(weights, [0b0101, 0b0001]))  # [5, 4]: the same, bit i of a token being 2**i
```

## Design

**The core** (`core.py`) is abstract: a `Signal` is a wire with a kind (`"bit"`, `"int"`,
`"real"`, ...) but no value, and a `Circuit` maps signals to signals. A `Graph` is made of
circuits; `@circuit` traces a function into one, once per argument structure, and later calls
reuse it. `flatten` and `lower` are passes from graphs to graphs, and `lower` uses a circuit's
hand-written version for a dialect where it has one (`@xor.impl("reglu")`), or for one it falls
back to (reglu's for geglu). A graph runs at every position of a sequence: `eval_sequence` gives
its outputs at each.

**The leaves** are the two parts of a transformer. The `MLP` (`mlp.py`), `wo @ act(wi @ x + b)`,
reads its own position. A threshold gate is an MLP with activation `"step"`; a GLU is an MLP
with a gated activation such as `"reglu"`, which multiplies relu of the left half of
`wi @ x + b` by its right half. `convert` turns step MLPs into relu MLPs, relu into reglu, and
reglu into swiglu or geglu. `Attention` (`attention.py`) is a head that reads earlier
positions: `lookback(xs, d)` gives `xs` as they were `d` tokens back. A sequence starts with a
zero token, whose bits are all 0 (`<bos>`, in a transformer), and before the start a head reads
its inputs as they are at the zero token, so `lookback([not_(x)], 1)` is 1 at the first token.

**The compiler** (`compiler.py`) places leaves in layers, so that a program never says where
they go. A stack of layers alternates attention and MLP sublayers, and `schedule` puts each
leaf of a flat graph in the first sublayer of its kind after those that give its inputs, and
that has room for it: a target says how many layers and features there are, and what room
each sublayer has, such as hidden units or heads. Each signal holds a feature from the sublayer
that gives it to the last that reads it, and features are used again once free. MLPs keep
their weights: placing one only wires it to the features it reads and writes.

**The targets** turn a schedule into weights.
- `network(graph, activation, pre_norm, mlp_ratio)` (`network.py`) builds a plain stack of
  layers, each one MLP, `mlp_ratio` times as wide inside as outside (4 by default), with no
  attention. Each layer holds its MLPs side by side, and a passthrough for each signal a later
  layer reads. Global parts act on whole layers: ONE = 1 leads every layer and carries the
  biases, and an RMSNorm can come before each layer. Networks include stacks of step or relu
  MLPs and GLU layers (reglu, swiglu, geglu) with or without RMSNorms.
- `transformer(program, architecture)` (`decoder.py`) builds the weights of a Gemma 4
  decoder of any size. A token carries the bits a program reads at its position, and greedy
  decoding gives the bits it computes as the next token. The residual stream holds bits as ±1,
  so Gemma's norms are fixed scalings; a head attends to the token `d` back with RoPE, or to
  `<bos>` if there is none, and copies its bits. `forward` runs the weights in numpy as
  transformers runs Gemma 4.
- `gemma.save(program, out_dir)` (`gemma.py`) writes a program in the weights of
  google/gemma-4-E4B, as a checkpoint transformers loads.

| File | Contents |
|---|---|
| `src/reifier/core.py` | signals, circuits, graphs, tracing, evaluation, `flatten`, `lower`, `format_graph` |
| `src/reifier/mlp.py` | the MLP, its activations, threshold gates, GLUs, passthroughs, `convert` |
| `src/reifier/attention.py` | the attention head, `lookback` |
| `src/reifier/fold.py` | `fold_constants`, an optional pass that works out what constants decide |
| `src/reifier/compiler.py` | `schedule`: leaves in layers, signals in features |
| `src/reifier/network.py` | plain stacks: ONE, RMSNorm, one MLP per layer |
| `src/reifier/decoder.py` | Gemma 4 decoders: `transformer`, and a numpy forward pass |
| `src/reifier/gemma.py` | google/gemma-4-E4B: its checkpoint's shapes and config, `save` |
| `src/reifier/lib.py` | `not_`, `and_`, `or_`, `xor` (with relu and reglu versions), `add`, `popcount`, `at_least` |
| `src/reifier/keccak.py` | Keccak-p and TurboSHAKE128, each bit one GLU on 3 or 5 bits |
| `examples/gemma_keccak.py` | TurboSHAKE128 of the last 8 tokens, with Keccak-p[1600, 6], in google/gemma-4-E4B |

Run the tests with `uv run pytest`.

Not here yet: heads that look up tokens by content, attention in Gemma's full-attention layers,
PyTorch export, and fast compilation of very large circuits.

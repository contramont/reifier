<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/contramont/reifier/refs/heads/main/assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="https://raw.githubusercontent.com/contramont/reifier/refs/heads/main/assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier</h1>

<p align="center">Compile algorithms into neural network circuits.</p>

Learn it in the browser with the [interactive tutorial](https://contramont.org/reifier/tutorial/):
build threshold circuits gate by gate, then download them as neural networks. The
[demo](https://contramont.org/reifier/demo/) has the xor below, a 4-bit adder and a sandbox for your own circuits.

Installation:
```bash
uv pip install reifier
```

See a demo Google Colab notebook [here](https://colab.research.google.com/drive/196UXK9fwExQI07u0ZDQKMr25YbZNPilA?usp=sharing).

Circuit visualization from the [demo](https://contramont.org/reifier/demo/): xor of 3 bits
on input `101`, the circuit that the example below builds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/contramont/reifier/refs/heads/main/assets/xor-circuit-dark.svg">
  <img alt="Xor of 3 bits on input 101: three inputs feed three counter gates, at least 1 to at least 3, which feed one output gate with weights +1, -1, +1 and threshold 1; the output is 0" src="https://raw.githubusercontent.com/contramont/reifier/refs/heads/main/assets/xor-circuit.svg" width="500">
</picture>

Inputs are on the left and the output on the right; filled nodes are 1, and each node is
named after the variable that holds it. Counter `count[i]` fires when at least i + 1 inputs
are 1, and the output gate `odd` adds the counters with weights +1, −1, +1 (gray positive,
red negative), so it reaches its threshold of 1 exactly when an odd number of inputs are 1.
Here two inputs are 1: `count[0]` and `count[1]` fire and cancel, and the output is 0. An
interactive block visualization of a compiled circuit is [here](http://draguns.me/circuit.html),
with inputs at the bottom and outputs at the top.

Simple example calculating xor of 3 bits, the demo's program (`r.xor` builds the same circuit):
```python
import reifier as r

def program(inputs):
    # 1 when an odd number of inputs are 1
    n = len(inputs)
    # count[i]: at least i + 1 inputs are 1
    count = [r.gate(incoming=inputs, weights=[1] * n, threshold=i + 1)
             for i in range(n)]
    # count[0] - count[1] + count[2] - ... is 1 for an odd number of ones, else 0
    odd = r.gate(incoming=count, weights=[(-1) ** i for i in range(n)], threshold=1)
    return odd

inputs = r.const('101')
outputs = program(inputs)
print('xor of', r.Bits(inputs).bitstr, '=', r.Bits(outputs).bitstr)  # xor of 101 = 0
```

Fast compilation with `reifier.fast`, which compiles circuits into
numpy-backed leveled graphs. `fast_compile` works on any circuit function:
```python
from reifier.examples.keccak import Keccak
from reifier.fast import fast_compile
from reifier.utils.format import Bits

k = Keccak(log_w=6, n=24, c=448, pad_char="_", stamp=True)
circuit = fast_compile(k.digest, Bits("0" * k.msg_len))
msg = k.format("Rachmaninoff")
print(Bits([int(v) for v in circuit.run(msg.ints)]).hex)
```
`Keccak(stamp=True)` compiles each round's structure once and stamps it for
all 24 rounds (see `reifier.fast.stamp`).

For Keccak specifically, `compile_keccak` skips bit-level execution entirely
and stacks the round template with array arithmetic, compiling full SHA3-224
in ~40ms (>10,000x faster than the tracing compiler):
```python
from reifier.examples.keccak_compile import compile_keccak

circuit = compile_keccak(Keccak(log_w=6, n=24, c=448, pad_char="_"))
```
Benchmark all paths with `python benchmarks/bench_keccak.py`.

Optimization levels with `reifier.opt`, an optional package beside the core compiler. A
level trades size for robustness, from `"ultra"` (bfloat16 or float16 on GPUs, next to a
host model's norm scales, LayerNorm shifts and noise) through `"hardened"`, `"robust"` and
`"O2"` to `"O3"` (float32). Each level compiles several recipes and keeps the smallest:
```python
import torch as t
from reifier.neurons.operations import add
from reifier.opt import Compiler
from reifier.utils.format import Bits

comp = Compiler("robust", mlp_dtype=t.bfloat16)
mlp = comp.run(lambda x: add(x[:16], x[16:]), x=Bits("0" * 32).bitlist)
print(comp.chosen, sum(p.numel() for p in mlp.parameters()))  # robust 31678
```
The core's `reifier.tensors.compilation.Compiler(mlp_dtype=t.bfloat16)` builds 156,115
parameters for this adder. Each level's guarantees are in `reifier.opt`'s docstring.

## Models from the tutorial

The Download model button of the tutorial and the demo compiles the program in the editor
into the network that `reifier.tensors.compilation.Compiler().run(program, inputs)` builds,
and saves it as `.safetensors`: a stack of float32 SwiGLU layers that reads a leading 1,
then the input bits. Plain PyTorch runs it:
```python
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

w = load_file("xor-3bit.safetensors")
x = torch.tensor([1.0, 1, 0, 1])  # a leading 1, then the input bits 101
for i in range(len(w) // 4):
    g, v, o = (w[f"layers.{i}.{k}.weight"] for k in ("wg", "wv", "wo"))
    x = F.rms_norm(x, x.shape[-1:], w[f"layers.{i}.norm.weight"])
    x = o @ (F.silu(g @ x) * (v @ x))
print((x[1:] / x[0]).round().int().tolist())  # the output bits: [0]
```
The file's metadata keeps the program it was compiled from.

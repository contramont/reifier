# Reifier

Compile algorithms into neural network circuits.

Installation:
```bash
uv pip install reifier
```

See a demo Google Colab notebook [here](https://colab.research.google.com/drive/196UXK9fwExQI07u0ZDQKMr25YbZNPilA?usp=sharing).

Circuit visualization:

<img src="https://raw.githubusercontent.com/contramont/reifier/refs/heads/main/src/reifier/examples/example_circuit.png" width="400">

Interactive visualization [here](http://draguns.me/circuit.html)

The visualization has inputs at the bottom and outputs at the top.

Simple example calculating xor of 5 bits:
```python
from reifier.neurons.core import const
from reifier.neurons.operations import xor
from reifier.utils.format import Bits

inputs = const('01101')
output = xor(inputs)
print(f"{Bits(inputs)} -> {Bits(output)}")
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

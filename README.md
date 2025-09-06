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

Simple example calculating xor of 5 bits:
```python
from reifier.neurons.core import const
from reifier.neurons.operations import xor
from reifier.utils.format import Bits

inputs = const('01101')
output = xor(inputs)
print(f"{Bits(inputs)} -> {Bits(output)}")
```

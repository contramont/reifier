<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Reifier logo: the letter R built from square bits and a wire" src="assets/logo.svg" width="128">
  </picture>
</p>

<h1 align="center">Reifier</h1>

<p align="center">Compile algorithms into transformer circuits.</p>

Reifier turns a Python function on bits into the weights of a transformer that computes the same
function on tokens. Each feature of the residual stream and each hidden unit keeps the name of
the part of the function that it computes. The transformer runs in numpy, or as a checkpoint
of Gemma 3 or Gemma 4, Google's open language models. Reifier needs Python 3.14 or later and
numpy: with [uv](https://docs.astral.sh/uv/), `uv run python` runs the example below and
`uv run pytest` runs the tests.

```python
from reifier.circuit import trace
from reifier.lib import add
from reifier.transformer import transformer


def fibonacci(bits):  # the bits of the last two tokens, the older first -> those of the next token
    return add(bits[:2], bits[2:])  # their sum, modulo 4


circuit = trace(fibonacci, 4)  # step 1: run it once on 4 bits, and record a circuit of small MLPs
codes = {token: [token >> 1, token & 1] for token in range(4)}  # the code of a token: two bits
model = transformer(circuit, codes)  # steps 2 and 3: a feature for each bit, then the weights
print(model.generate([1, 1], 6))  # 2 tokens, as many as the circuit reads, then 6 more
```

```text
[1, 1, 2, 3, 1, 0, 1, 1]
```

While `fibonacci` is traced, its bits have no values, so Python's `+`, `and` and `if` do not work
on them. Combine them with the functions of `lib.py`, or with MLPs of your own (`circuit.mlp`).

`model` is an ideal transformer: it computes every bit exactly, and it has no normalization. To
run the same computation in a real model, continue the example: give the codes to tokens of Gemma,
here "0" to "3", and save a checkpoint.

```python
from reifier import gemma  # needs huggingface_hub: uv run --with huggingface_hub python

repo = "unsloth/gemma-3-270m"  # or "google/gemma-4-E2B"
vocabulary = gemma.vocabulary(repo)  # {text of a token: its id in Gemma}
model = transformer(circuit, {vocabulary[str(token)]: code for token, code in codes.items()})
tokens = model.generate([vocabulary["1"], vocabulary["1"]], 6)
gemma.save(model, "out", repo, check_on=tokens)  # checks the weights on `tokens` first
```

Hugging Face's `AutoModelForCausalLM.from_pretrained("out")` loads the directory `out`. It holds
the weights and no tokenizer: use the tokenizer of `repo`.

## Reading order

Read `src/reifier` in this order. Steps 1 to 3 are the core: they make the ideal transformer.
Steps 8 and 9 lower it: they turn it into layers of a real model that compute the same.

| Step | File | What it adds |
|---|---|---|
| 1 | `circuit.py` | `trace` runs a function once and records a circuit of small MLPs |
| 2 | `layout.py` | `layout` chooses a feature of the residual stream for each bit of the circuit |
| 3 | `transformer.py` | `transformer` makes the embedding, the attention heads and the MLP blocks |
| 4 | `lib.py` | parts to build circuits with: threshold gates, `add`, `sum_of_ands`, `table` |
| 5 | `fold.py` | `fold` turns constants into biases, and removes MLPs that copy or negate a bit |
| 6 | `check.py` | checks: a circuit against a Python function, a transformer against its circuit |
| 7 | `show.py` | `show` gives a circuit or a layout as text, to print |
| 8 | `lower_stream.py` | lowering, part 1: the stream, the embedding and the attention with RoPE |
| 9 | `lower.py` | lowering, part 2: the MLP blocks with RMSNorm and GELU, and `lower` |
| 10 | `checkpoint.py` | the files of a checkpoint, as Hugging Face `transformers` reads them |
| 11 | `gemma.py` | the lowered transformer as Gemma's tensors; `save` writes them as a checkpoint |
| 12 | `beside.py` | a circuit and Gemma 3 270M's own weights in one wider, deeper model |

`examples/adder.py` adds decimal numbers: after "123+246=" come "3", "6", "9".
`examples/keccak.py` chooses each token by a hash of the last three. `examples/alongside.py` puts
a larger hash and Gemma 3 270M in one model, which answers as Gemma where the hash gives no output.

`uv run --with torch --with transformers --with huggingface_hub python -m pytest` also runs the
tests on a real Gemma. They need a GPU.

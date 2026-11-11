"""The counterexamples of COUNTEREXAMPLES.md, on the real compiler.

Each program is traced with reifier, compiled with reifier.transformer.transformer, and run by its
own forward pass (numpy, float64); reifier.check.check_transformer is the judge where it applies.
For bf16, `bf16_run` replays Transformer.run on the compiled weights, rounding every weight and the
result of every operation with gemma.round_to_bf16 (to nearest, ties to even), as BF16.lean does;
`literal=True` computes each norm's scale from the float ONE, as rmsnorm reads literally.

Run from the repository root: uv run --python 3.14 python lean/simulations/real_compiler.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from reifier.check import check_transformer
from reifier.emit import EPS
from reifier.gemma import round_to_bf16
from reifier.lib import and_
from reifier.trace import glu, mlp, threshold_gate, trace
from reifier.transformer import CONTEXT, rope, transformer

BITS1 = {0: [0], 1: [1]}


def r16(x):
    a = np.asarray(x, np.float64)
    return round_to_bf16(a).astype(np.float64).reshape(a.shape)  # ascontiguousarray makes 0-d 1-d


def check(m, tokens):
    try:
        check_transformer(m, tokens)
        return "check_transformer passes"
    except AssertionError as e:
        return f"check_transformer FAILS: {e}"


def final_out(m, tokens, i=0):
    """The output feature 1 + i at the last position, after the last layer."""
    return float(m.run(tokens)[0][-1][-1, 1 + i])


def bf16_rmsnorm(x, row, one, one_norm):
    rms = np.sqrt((one_norm**2 + sum(s is not None for s in row[1:])) / len(row) + EPS)
    z = rms * x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)
    z[..., 0] /= one_norm
    return r16(z)


def bf16_run(m, tokens, literal=False):
    """Transformer.run in bf16: the stream at the last depth, and the logits."""
    net, grid = m.net, m.net.grid
    one = float(r16(net.one))  # what the bf16 embedding holds
    one_norm = net.one if literal else one  # what each norm's scale is computed from
    pos = np.arange(len(tokens))
    x = r16(m.embed)[np.asarray(tokens)]
    embedded = [s if f <= len(m.v) else None for f, s in enumerate(grid[0])]
    z = bf16_rmsnorm(x, embedded, one, one_norm)
    seen = (pos[:, None] >= pos) & (pos[:, None] - pos < CONTEXT)
    keys = r16(rope(r16(z @ r16(m.k).T), pos, m.turns))
    values = r16(z @ r16(m.v).T)
    for q, o in zip(m.q, m.o):
        query = r16(rope(r16(z @ r16(q).T), pos, m.turns))
        scores = r16(np.where(seen, query @ keys.T, -np.inf))
        a = np.exp(scores - scores.max(1, keepdims=True))
        a = r16(a / a.sum(1, keepdims=True))
        x = r16(x + r16(r16(a @ values) @ r16(o).T))
    for layer, before in zip(net.layers, grid):
        zz = bf16_rmsnorm(x, before, one, one_norm)
        n = len(layer.hidden)
        g = r16(zz @ r16(layer.wi[:n]).T)
        v = r16(zz @ r16(layer.wi[n:]).T)
        gel = r16(g / 2 * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3))))
        y = r16(r16(gel * v) @ r16(layer.wo).T)
        x = r16(x + y)
    logits = r16(bf16_rmsnorm(x, grid[-1], one, one_norm) @ r16(m.embed).T)
    return x, logits


def report(name, line):
    print(f"{name:7s} {line}")


# CE-1: a step gate whose sum is -1/2 on a bit: [x/2 - 1 >= 0], always 0
m = transformer(trace(lambda b: [mlp(b, [[0.5]], [-1], "step", [[1]], "half")[0]], 1), BITS1)
report("CE-1", f"output feature {final_out(m, [1]):+.4f} (trace: 0, so -1); {check(m, [0, 1])}")

# CE-2: a GLU gate of 1/32 where its value is 32: relu(1/32) * 32 = 1, of no inputs
m = transformer(trace(lambda b: [glu([], [([], 1 / 32, [], 32)], "g")], 1), BITS1)
report("CE-2", f"output feature {final_out(m, [1]):+.4f} (trace: 1, so +1); {check(m, [0, 1])}")
x, _ = bf16_run(m, [1])
report("CE-2 bf16", f"output feature {x[-1, 1]:+.6f}")

# CE-3: the same signal returned twice
codes2 = {n: [n >> 1, n & 1] for n in range(4)}
m = transformer(trace(lambda b: (lambda g: [g, g])(threshold_gate(b[:1], [-1], 0, "not")), 2), codes2)
report("CE-3", f"output features {final_out(m, [2], 0):+.4f} {final_out(m, [2], 1):+.4f} "
       f"(token 2 = bits 10; trace: not 1 = 0 twice, so -1 -1); {check(m, [3, 2])}")

# CE-4: an output that is an input
m = transformer(trace(lambda b: [b[0]], 1), BITS1)
report("CE-4", f"output feature {final_out(m, [1]):+.4f} (trace: 1, so +1); {check(m, [0, 1])}")

# CE-5: a window of 513 tokens: the oldest token's bit, which no head can see
m = transformer(trace(lambda b: [threshold_gate(b[:1], [1], 1, "oldest")], 513), BITS1)
toks = [1] + [0] * 512
report("CE-5", f"output feature {final_out(m, toks):+.4f} (oldest token 1, so +1)")

# CE-6: two tokens with one code
m = transformer(trace(lambda b: [threshold_gate(b, [1], 1, "g")], 1), {0: [1], 1: [1]})
_, logits = m.run([0])
report("CE-6", f"logits {logits[-1][0]:.6f} {logits[-1][1]:.6f}: tied; {check(m, [0, 1])}")


# CE-7: a chain of the GLU relu(y + 1/2) * (2/3) y: the identity on bits, not flat
def chain(n):
    def program(b):
        y = b[0]
        for i in range(n):
            y = glu([y], [([1], 1 / 2, [2 / 3], 0)], f"glu{i}")
        return [y]
    return program


for n in (1, 17, 40, 58, 70):
    m = transformer(trace(chain(n), 1), BITS1)
    report("CE-7", f"{n:3d} GLUs: output feature {final_out(m, [0, 1]):+.12f} (trace: 1)")

# CE-8: [w x - w >= 0], the identity, with a huge w
for w in (1e12, 5e12, 1e13):
    m = transformer(trace(lambda b: [mlp(b, [[w]], [-w], "step", [[1]], "id")[0]], 1), BITS1)
    report("CE-8", f"w = {w:.0e}: output feature {final_out(m, [0, 1]):+.6f} (trace: 1)")

# CE-2b: in real arithmetic, a relu gate exactly 0 on bits, which gelu does not cut off:
# zg(C, a) = 2 relu(2x - 1) - relu(4x - 3) + C relu(a (x - 1)), the identity on bits
def zg(y, C, a, name):
    return mlp([y], [[2], [4], [a]], [-1, -3, -a], "relu", [[2, -1, C]], name)[0]


for n, a in ((1, 1), (2, 2**12)):
    def program(b, n=n, a=a):
        y = b[0]
        for i in range(n):
            y = zg(y, 2**15, a, f"zg{i}")
        return [y]
    m = transformer(trace(program, 1), BITS1)
    report("CE-2b", f"{n} unit(s), a = {a}: output feature {final_out(m, [0, 1]):+.10f} (trace: 1)")

# CE-9: two always-on step units of weights 1027/2 and -1025/2 (output 1): bf16 cannot hold 1027
m = transformer(trace(lambda b: [mlp([], np.zeros((2, 0)), [0, 0], "step", [[1027 / 2, -1025 / 2]], "w")[0]], 1), BITS1)
x, _ = bf16_run(m, [1])
report("CE-9", f"float64 output {final_out(m, [1]):+.4f}; bf16 output {x[-1, 1]:+.4f} (trace: 1)")

# CE-10: [128 x - 64 >= 0] in bf16: at x = 1 its sums 8272 and 8240 both round to 8256
m = transformer(trace(lambda b: [threshold_gate(b, [128], 64, "big")], 1), BITS1)
x, _ = bf16_run(m, [1])
report("CE-10", f"float64 output {final_out(m, [1]):+.4f}; bf16 output {x[-1, 1]:+.4f} (trace: 1)")

# CE-12: majority of 19 one-bit tokens (width 21), bf16, literal reading vs one rounded once
m = transformer(trace(lambda b: [threshold_gate(b, [1] * 19, 10, "maj")], 19), BITS1)
assert len(m.net.grid[0]) == 21
rng = np.random.default_rng(0)
for literal in (True, False):
    worst = 0.0
    for ones in range(20):
        toks = [1] * ones + [0] * (19 - ones)
        x, _ = bf16_run(m, list(rng.permutation(toks)), literal)
        worst = max(worst, abs(abs(x[-1, 1]) - 1))
    report("CE-12", f"{'literal' if literal else 'one rounded once'}: worst |output| - 1 = {worst:.6f}")

# And what the theorems cover, on the real compiler: and_ of 33 bits (gate -4144 at all zeros),
# and the decimal adder of examples/adder.py, in bf16 with ONE rounded once
m = transformer(trace(lambda b: [and_(b)], 33), BITS1)
outs = []
for ones in (0, 32, 33):
    x, _ = bf16_run(m, [1] * ones + [0] * (33 - ones))
    outs.append(x[-1, 1])
report("and33", f"bf16 outputs at 0, 32, 33 ones: {outs} (want -1, -1, +1)")

import adder  # noqa: E402

symbols = adder.SYMBOLS
m = transformer(adder.program(), dict(enumerate(adder.CODES.values())))
exact = True
for a, b in [(123, 246), (999, 999), (0, 0), (5, 995), (608, 397)]:
    toks = [symbols.index(c) for c in f"{a:03}+{b:03}="]
    for _ in range(3):
        x, logits = bf16_run(m, toks)
        want = np.array(adder.CODES[f"{(a + b) % 1000:03}"[len(toks) - 8]]) * 2 - 1
        exact &= bool(np.array_equal(x[-1, 1 : 1 + len(want)], want))
        toks.append(int(logits[-1].argmax()))
    report("adder", f"{a:03}+{b:03}= -> {''.join(symbols[i] for i in toks[-3:])} in bf16")
report("adder", f"every output feature exactly +-1 in bf16: {exact}; width {len(m.net.grid[0])}")

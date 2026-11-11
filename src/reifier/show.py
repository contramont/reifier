"""How each stage prints: show(x) is a trace, a grid, a network or a transformer as text."""

from collections.abc import Iterator
from itertools import pairwise

import numpy as np

from .emit import Network
from .trace import GATED, MLP, Trace
from .transformer import Transformer


def show(x) -> str:
    kinds = {Trace: trace, list: grid, Network: network, Transformer: transformer}
    return "\n".join(kinds[type(x)](x))


def linear(weights, names, bias=0.0) -> str:
    """A weighted sum as text: 4 a0 - a1 + 0.5"""
    text = ""
    for w, n in [(w, n) for w, n in zip(weights, names) if w] + [(bias, "")] * bool(bias):
        size = "" if abs(w) == 1 and n else f"{abs(w):g}"
        text += (" - " if w < 0 else " + ") + " ".join(filter(None, [size, n]))
    return ("-" + text[3:] if text.startswith(" - ") else text[3:]) or "0"


def units(m: MLP) -> Iterator[str]:
    """A line for each hidden unit: what it reads, and what it writes"""
    n, act = len(m.hidden), GATED.get(m.activation, m.activation)
    for j, h in enumerate(m.hidden):
        unit = f"{act}({linear(m.wi[j], m.inputs, m.b[j])})"
        if m.activation in GATED:
            unit += f" * ({linear(m.wi[n + j], m.inputs, m.b[n + j])})"
        yield f"{h} = {unit} -> {linear(m.wo[:, j], m.outputs)}"


def trace(t: Trace) -> Iterator[str]:
    """A line for each threshold gate, [what it sums >= its threshold], and for each hidden unit
    of any other MLP"""
    yield f"{t.name}({' '.join(t.inputs)}): {len(t.mlps)} MLPs"
    for m in t.mlps:
        if m.activation == "step" and m.wo.tolist() == [[1]]:
            yield f"  {m.name} = [{linear(m.wi[0], m.inputs)} >= {0 - m.b[0]:g}]"
        else:
            yield f"  {m.name}: {m.activation}, {len(m.hidden)} hidden"
            yield from ("    " + line for line in units(m))
    yield f"  return {' '.join(t.outputs)}"


def grid(rows) -> list[str]:
    """A line for each feature, a column for each depth: " for a signal that stays, . for no
    signal"""
    lines = [["feature", *(f"depth {d}" for d in range(len(rows)))]]
    for f, names in enumerate(zip(*rows)):
        stays = [False, *(new == old for old, new in pairwise(names))]
        lines.append([str(f), *('"' if s and same else s or "." for s, same in zip(names, stays))])
    widths = [max(map(len, column)) for column in zip(*lines)]
    return ["  ".join(map(str.ljust, line, widths)).rstrip() for line in lines]


def network(net: Network) -> Iterator[str]:
    """Each hidden unit of each layer: what it reads, by the names of the row before, and the
    features it writes"""
    kind = ("skip" if net.skip else "stack") + " + norm" * net.norm
    yield f"{net.trace.name}: {kind}, {len(net.grid[0])} features, ONE = {net.one:g}"
    for d, layer in enumerate(net.layers, 1):
        yield f"layer {d}: {len(layer.hidden)} hidden units of {layer.activation}"
        yield from ("  " + line for line in units(layer))


def transformer(m: Transformer) -> Iterator[str]:
    """Each head, by the features it writes, then the network"""
    yield f"{len(m.q)} heads on {len(m.codes)} tokens"
    for h, o in enumerate(m.o):
        names = " ".join(m.net.grid[0][f] for f in np.flatnonzero(o.any(1)))
        yield f"head {h}: the code of the token {h} back -> {names}"
    yield from network(m.net)

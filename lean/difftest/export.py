"""Differential test of the Lean model against the real compiler.

For each program: export its trace as Lean (signals renamed to naturals: inputs in order, then the
outputs of each MLP in turn), and Python's layout, token_grid, and the units of
emit(t, token_grid(t), "geglu", skip=True, norm=True), every weight as an exact fraction.
Run from the repository root: uv run --python 3.14 python lean/difftest/export.py
"""
import json, sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import numpy as np
from reifier.trace import trace, mlp, const, threshold_gate
from reifier.grid import layout
from reifier.transformer import token_grid
from reifier.emit import emit
from reifier.lib import add, xor, parity, not_, sum_of_ands

def glu_add(a, b): return add(a, b, parity)
def odd(x):
    deep = xor([not_(x[1]), x[0]])
    two = mlp([x[0], x[1], x[0]], [[1, 1, 1], [0, 1, 0]], [-2, -1], "step", np.eye(2), "two")
    return [x[1], const("1")[0], two[1], deep]
def par4(x): return [parity(x)]
def soa(x): return [sum_of_ands(x, [([1, -1, 0], 1), ([0, 1, 1], 1)])]
def chain_not(x):
    for _ in range(5): x = [not_(x[i - 1]) for i in range(len(x))]
    return x

PROGRAMS = {"xor3": (xor, (3,)), "add22": (add, (2, 2)), "add33": (add, (3, 3)), "gluadd33": (glu_add, (3, 3)),
            "odd": (odd, (2,)), "par4": (par4, (4,)), "soa": (soa, (3,)), "chain": (chain_not, (4,))}
import adder
PROGRAMS["adder"] = None

def q(x): return F(float(x))
def lq(fr):  # Lean rational literal
    fr = F(fr)
    return f"({fr.numerator} : ℚ)" if fr.denominator == 1 else f"(({fr.numerator} : ℚ) / {fr.denominator})"
def llist(xs): return "[" + ", ".join(xs) + "]"

out_lean, expect = [], {}
for name, spec in PROGRAMS.items():
    t = adder.program() if name == "adder" else trace(spec[0], *spec[1])
    ids = {s: i for i, s in enumerate(t.inputs)}
    for m in t.mlps:
        for y in m.outputs: ids.setdefault(y, len(ids))
    def cell(s):
        if s is None: return None
        if s == "ONE": return ["one"]
        if s.startswith("code["): return ["code", int(s[5:-1])]
        return ["sig", ids[s]]
    mlps = []
    for m in t.mlps:
        act = {"step": "step", "relu": "relu", "reglu": "reglu"}[m.activation]
        n = len(m.hidden)
        wi = llist(llist(lq(q(w)) for w in row) for row in m.wi)
        b = llist(lq(q(v)) for v in m.b)
        wo = llist(llist(lq(q(w)) for w in row) for row in m.wo)
        mlps.append(f"  {{ ins := {llist(str(ids[x]) for x in m.inputs)}, outs := {llist(str(ids[y]) for y in m.outputs)}, "
                    f"act := .{act}, hidden := {n}, wi := fun j k => ({wi}.getD j []).getD k 0, "
                    f"b := fun j => {b}.getD j 0, wo := fun i j => ({wo}.getD i []).getD j 0 }}")
    out_lean.append(f"def {name} : Trace where\n  inputs := {llist(str(ids[x]) for x in t.inputs)}\n  mlps := [\n" + ",\n".join(mlps)
                    + f"]\n  outputs := {llist(str(ids[y]) for y in t.outputs)}\n")
    lay = layout(t); tg = token_grid(t)
    rec = {"layout": [[cell(s) for s in r] for r in lay], "token": [[cell(s) for s in r] for r in tg]}
    if name != "adder" or True:
        net = emit(t, tg, "geglu", skip=True, norm=True)
        layers = []
        for L in net.layers:
            n = len(L.hidden); W = L.wi.shape[1]
            layers.append([[[str(q(w)) for w in L.wi[j]], [str(q(w)) for w in L.wi[n + j]], [str(q(L.wo[f, j])) for f in range(L.wo.shape[0])]] for j in range(n)])
        rec["layers"] = layers
    expect[name] = rec
Path(__file__).with_name("Programs.lean").write_text(
    "import ReifierProofs.Transformer\n\nnamespace Reifier.DiffTest\n\n" + "\n".join(out_lean) + "\nend Reifier.DiffTest\n")
Path(__file__).with_name("expected.json").write_text(json.dumps(expect))
print({k: (len(v["token"]), len(v["token"][0]), sum(len(l) for l in v["layers"])) for k, v in expect.items()})

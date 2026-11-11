"""The Lean model against the Python it mirrors: on small programs, what layout, held, check_grid
and emit make here is written as Lean terms, and Lean checks that its own definitions give the
same: every grid, and every weight and name of every layer of every network.

Run, from the root of the repository: uv run python lean/test/compare.py
It writes lean/test/Compare.lean and runs Lean on it: no output means that all agree.
"""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from reifier.check import check_grid
from reifier.emit import emit
from reifier.grid import held, layout
from reifier.lib import add, not_, parity, xor
from reifier.trace import GATED, const, mlp, trace
from reifier.transformer import token_grid

NETWORKS = [
    (activation, skip, norm)
    for activation in ("step", "relu", "reglu", "swiglu", "geglu")
    for skip in (False, True)
    for norm in (False, True)
    if activation in GATED or not norm
]


def glu_add(a, b):
    return add(a, b, parity)


def odd(x):  # as in tests/test_core.py: outputs that are an input and a constant, an input
    deep = xor([not_(x[1]), x[0]])  # read twice, an MLP of two outputs, one unread
    two = mlp([x[0], x[1], x[0]], [[1, 1, 1], [0, 1, 0]], [-2, -1], "step", np.eye(2), "two")
    return [x[1], const("1")[0], two[1], deep]


def ones():
    return const("11")


def empty(x):  # a signal named "": Python takes its name for false
    return mlp(x, [[1, 1]], [-1], "step", [[1]], "")


def count(bits):
    return add(bits[:2], bits[2:])


def num(x) -> str:
    q = Fraction(float(x))  # every weight made here is a float that is a small fraction
    return str(q.numerator) if q.denominator == 1 else f"({q.numerator}/{q.denominator})"


def lst(xs, item=json.dumps) -> str:
    return "[" + ", ".join(item(x) for x in xs) + "]"


def matrix(w) -> str:
    return lst(np.asarray(w).tolist(), lambda row: lst(row, num))


def lean_mlp(m) -> str:
    fields = [json.dumps(m.name), lst(m.inputs), lst(m.outputs), lst(m.hidden), matrix(m.wi)]
    fields += [lst(m.b.tolist(), num), "." + m.activation, matrix(m.wo)]
    return "⟨" + ", ".join(fields) + "⟩"


def lean_grid(grid) -> str:
    cell = lambda s: "none" if s is None else f"some {json.dumps(s)}"  # noqa: E731
    return lst(grid, lambda row: lst(row, cell))


def passes(t, grid) -> str:
    try:
        check_grid(t, grid)
    except AssertionError:
        return "false"
    return "true"


def main():
    out = ["import Reifier", "open Reifier", "set_option maxRecDepth 100000", ""]
    programs = [(xor, 3), (add, 2, 2), (glu_add, 2, 2), (odd, 2), (ones,), (empty, 2), (count, 4)]
    for i, (program, *sizes) in enumerate(programs):
        t = trace(program, *sizes)
        name = f"t{i}"
        mlps = ",\n   ".join(lean_mlp(m) for m in t.mlps)
        out.append(f"def {name} : Trace ℚ := ⟨{json.dumps(t.name)}, {lst(t.inputs)}, ")
        out.append(f"  {lst(t.outputs)},\n  [{mlps}]⟩\n")
        g = layout(t)
        out.append(f"#guard layout {name} = {lean_grid(g)}")
        out.append(f"#guard held (layout {name}) = {lean_grid(held(g))}")
        if program is count:
            out.append(f"#guard tokenGrid {name} = {lean_grid(token_grid(t))}")
        grids = {"layout": g, "held": held(g), "padded": [*g, g[-1]], "twice": [r + ["ONE"] for r in g]}
        for d, f, cell in [(0, 1, [None]), (len(g) - 1, 1, [None]), (0, 0, [None, "ONE"])]:
            if len(g) > 1 and len(g[0]) > 1:
                wrong = [list(row) for row in g]
                wrong[d][f : f + 1] = cell
                grids[f"wrong{d}{f}{len(cell)}"] = wrong
        if program is add:  # the grids that tests/test_core.py expects check_grid to refuse
            for d, f, cell in [(0, 5, ["add/and"]), (2, 6, []), (2, 1, [None]),
                               (4, 5, ["add/and"]), (4, 5, ["add/xor.1/odd"])]:
                wrong = [list(row) for row in layout(trace(add, 2, 2))]
                wrong[d][f : f + 1] = cell
                grids[f"wrong{d}{f}{len(cell)}{cell and cell[0]}"] = wrong
        for kind, grid in grids.items():
            out.append(f"-- {t.name}, {kind}")
            out.append(f"def {name}_{len(out)} : Grid := {lean_grid(grid)}")
            gname = out[-1].split()[1]
            ok = passes(t, grid)
            out.append(f"#guard checkGrid {name} {gname} = {ok}")
            if ok == "false" or kind == "twice" or kind.startswith("wrong"):
                continue
            for activation, skip, norm in NETWORKS:
                if program is glu_add and activation not in GATED:
                    continue  # a GLU has no step or relu form
                net = emit(t, grid, activation, skip, norm)
                layers = ",\n   ".join(lean_mlp(layer) for layer in net.layers)
                flags = f".{activation} {str(skip).lower()} {str(norm).lower()}"
                out.append(f"#guard ({gname}.zip {gname}.tail).map")
                out.append(f"    (fun q => layer {name}.mlps q.1 q.2 {flags}) =\n  [{layers}]")
        out.append("")
    here = Path(__file__).parent
    (here / "Compare.lean").write_text("\n".join(out))
    subprocess.run(["lake", "env", "lean", "test/Compare.lean"], cwd=here.parent, check=True)


if __name__ == "__main__":
    main()

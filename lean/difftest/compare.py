"""Compare the Lean model's grids and units (lean_*.json, from `lake env lean difftest/<name>.lean`
run in the project root, then moved here) with the real compiler's (expected.json, from export.py)."""
import json, sys
from fractions import Fraction as F
from pathlib import Path

here = Path(__file__).parent
exp = json.loads((here / "expected.json").read_text())
grid = lambda g: [[tuple(c) if c else None for c in r] for r in g]
agree = total = 0
for path in sys.argv[1:] or sorted(here.glob("lean_*.json")):
    for name, got in json.loads(Path(path).read_text()).items():
        want = exp[name]
        for key in ("layout", "token"):
            ok = grid(got[key]) == grid(want[key])
            total += 1; agree += ok
            print(f"{name:9s} {key:7s} {'OK' if ok else 'MISMATCH'}  {len(want[key])} rows x {len(want[key][0])}")
        if "layers" in got:
            ok, n = len(got["layers"]) == len(want["layers"]), 0
            for lu, pu in zip(got["layers"], want["layers"]):
                ok &= len(lu) == len(pu)
                for a, b in zip(lu, pu):
                    for part in range(3):
                        n += len(b[part])
                        ok &= all(F(x) == F(y) for x, y in zip(a[part], b[part]))
            total += 1; agree += ok
            print(f"{name:9s} layers  {'OK' if ok else 'MISMATCH'}  {n} weights")
print(f"{agree}/{total} agree")

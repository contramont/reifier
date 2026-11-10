"""Stage 2, the grid: which signal each feature holds at each depth.

rows[d][f] is the name of the signal that feature f holds at depth d, or None. Row 0 is what the
network is given, the last row holds the outputs, and an MLP is at the depth where its outputs
first appear: it reads the row before. A grid holds names only, no weights, and every emitter
reads the same one: it makes whatever it takes for each row to follow from the row before.
layout builds a grid, and check_grid says what makes any grid, however built, a grid of its trace.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from .trace import MLP, Trace

ONE = "ONE"  # the constant, feature 0 of every row: a layer reads it as 1, for its biases


@dataclass
class Grid:
    rows: list[list[str | None]]

    def __str__(self):
        """A line for each feature, a column for each depth: " for a signal that stays, . for
        no signal"""
        lines = [["feature", *(f"depth {d}" for d in range(len(self.rows)))]]
        for f, names in enumerate(zip(*self.rows)):
            stays = [False, *(new == old for old, new in pairwise(names))]
            lines.append(
                [str(f), *('"' if s and same else s or "." for s, same in zip(names, stays))]
            )
        widths = [max(map(len, column)) for column in zip(*lines)]
        return "\n".join("  ".join(map(str.ljust, line, widths)).rstrip() for line in lines)


def layout(t: Trace, row0: list | None = None, pins: dict | None = None) -> Grid:
    """A grid for t. Each MLP goes right after the last of the MLPs it reads. A signal takes a
    feature when it is made: the feature freed last, or else a new one. Its feature is free from
    the depth of the last MLP that reads it, and an output's never is. row0 is the first row
    (default: ONE, then the inputs): it may hold names that are not t's, which stay. pins
    {signal: feature}: a signal that an MLP makes takes the feature of such a name instead"""
    row, pins, outputs = list(row0 or [ONE, *t.inputs]), pins or {}, set(t.outputs)
    depth, last = dict.fromkeys(t.inputs, 0), {}  # the depth of each signal, the last to read it
    for m in t.mlps:
        d = 1 + max((depth[x] for x in m.inputs), default=0)
        depth |= dict.fromkeys(m.outputs, d)
        last |= {x: max(last.get(x, 0), d) for x in m.inputs}
    for y, f in pins.items():
        taken = row[f] in (None, ONE, *depth) if 0 < f < len(row) else True
        if not depth.get(y) or taken or list(pins.values()).count(f) > 1:
            raise ValueError(
                f"{y} cannot take feature {f}: a pin puts a signal that an MLP makes "
                f"over a name of row0 that is not {t.name}'s"
            )
    made, frees = defaultdict(list), defaultdict(list)  # the signals each depth makes, and frees
    for x, d in depth.items():
        made[d].append(x)
        if x not in outputs:  # a signal that nothing reads is free after its own depth
            frees[last.get(x, d + 1)].append(x)
    at, free, rows = {s: f for f, s in enumerate(row) if s}, [], [row]
    for d in range(1, max(depth.values(), default=0) + 1):
        row = list(row)
        for x in frees[d]:
            free.append(at[x])
            row[at[x]] = None
        for y in made[d]:
            at[y] = pins[y] if y in pins else free.pop() if free else len(row)
            row += [None] * (at[y] + 1 - len(row))
            row[at[y]] = y
        rows.append(row)
    return Grid([r + [None] * (len(rows[-1]) - len(r)) for r in rows])


def held(g: Grid) -> Grid:
    """g as skip connections hold it: a signal stays in its feature until another is written
    there, whereas a stack must carry all that it keeps. Fewer cells change, so less is erased"""
    rows = [g.rows[0]]
    for row in g.rows[1:]:
        here = set(row)  # a signal that moved is not held where it was
        rows.append([s or (old if old not in here else None) for s, old in zip(row, rows[-1])])
    return Grid(rows)


def check_grid(t: Trace, g: Grid) -> list[list[MLP]]:
    """Check g against t, and give t's MLPs by depth. ONE is feature 0 of every row, and no row
    holds a name twice. Row 0 holds t's inputs, and nothing t makes. The outputs of each MLP
    first appear together, in a row after the first, and the row before holds all its inputs.
    Any other name in a row is in the row before too: nothing comes from nowhere. The last row
    holds t's outputs"""
    held_at, first = [{s for s in row if s} for row in g.rows], {}  # each name's first depth
    for d, row in enumerate(g.rows):
        assert row[0] == ONE, f"feature 0 at depth {d} is {row[0]}, not ONE"
        assert len(row) == len(g.rows[0]), f"depth {d} is not as wide as depth 0"
        assert len(held_at[d]) == len(row) - row.count(None), (
            f"depth {d} holds {sorted(s for s in held_at[d] if row.count(s) > 1)} twice"
        )
        for s in held_at[d]:
            assert first.setdefault(s, d) == d or s in held_at[d - 1], (
                f"{s} is at depth {d} but not at depth {d - 1}"
            )
    by_depth, makes = [[] for _ in g.rows], {y for m in t.mlps for y in m.outputs}
    for x in t.inputs:
        assert first.get(x) == 0, f"input {x} is not at depth 0"
    for m in t.mlps:
        lost = [y for y in m.outputs if y not in first]
        assert not lost, f"the grid does not hold {lost}, which {m.name} makes"
        d, *more = sorted({first[y] for y in m.outputs})
        assert d and not more, f"{m.name}: its outputs first appear at depths {[d, *more]}"
        missing = set(m.inputs) - held_at[d - 1]
        assert not missing, f"{m.name} is at depth {d}, but depth {d - 1} lacks {sorted(missing)}"
        by_depth[d].append(m)
    for s, d in first.items():
        assert d == 0 or s in makes, f"{s} first appears at depth {d}, and no MLP makes it"
    for y in t.outputs:
        assert y in held_at[-1], f"output {y} is not at the last depth"
    return by_depth

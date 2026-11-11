"""Stage 2, the grid: which signal each feature holds at each depth.

A grid is a list of rows: grid[d][f] is the name of the signal that feature f holds at depth d,
or None. Row 0 is what the network is given, the last row holds the outputs, and an MLP is at the
depth where its outputs first appear: it reads the row before. ONE, the constant, is feature 0 of
every row, and a signal keeps its feature for as long as the grid holds it. A grid has names
only, no weights, and every emitter reads the same one: it makes whatever it takes for each row
to follow from the row before.
"""

from collections import defaultdict
from math import inf

from .trace import Trace

ONE = "ONE"  # the constant, feature 0 of every row: a layer reads it as 1, for its biases
Grid = list[list[str | None]]


def layout(t: Trace) -> Grid:
    """A grid for t: row 0 is ONE and the inputs, and each MLP goes right after the last of the
    MLPs it reads. A signal's feature is freed at the depth of the last MLP that reads it, which
    reads the row before, and an output's never is. A signal that is made takes the feature freed
    last, or else a new one"""
    depth = dict.fromkeys(t.inputs, 0)  # the depth of each signal
    last = dict.fromkeys(t.outputs, inf)  # the depth of the last MLP that reads each signal
    for m in t.mlps:
        d = 1 + max((depth[x] for x in m.inputs), default=0)
        depth |= dict.fromkeys(m.outputs, d)
        last |= {x: max(last.get(x, 0), d) for x in m.inputs}
    made, freed = defaultdict(list), defaultdict(list)  # the signals each depth makes, and frees
    for x, d in depth.items():
        made[d].append(x)
        freed[last.get(x, d + 1)].append(x)  # what nothing reads is freed after its own depth
    row = [ONE, *t.inputs]
    rows = [row]
    at = {x: f for f, x in enumerate(row)}  # the feature of each signal
    free = []  # the features freed, and not taken again
    for d in range(1, max(depth.values(), default=0) + 1):
        row = list(row)
        for x in freed[d]:
            free.append(at[x])
            row[at[x]] = None
        for y in made[d]:
            if not free:  # a new feature
                free.append(len(row))
                row.append(None)
            at[y] = free.pop()
            row[at[y]] = y
        rows.append(row)
    return [row + [None] * (len(rows[-1]) - len(row)) for row in rows]  # all as wide as the last


def held(grid: Grid) -> Grid:
    """grid with each signal left in its feature until another is written there, as a skip
    connection leaves it: emit, with skip, then erases only what is written over. Not for a
    stack, which would copy what is left at every layer"""
    rows = grid[:1]
    for row in grid[1:]:
        rows.append([s or old for s, old in zip(row, rows[-1])])
    return rows

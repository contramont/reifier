"""Stage 2, the grid: which signal each feature holds at each depth.

A grid is a list of rows: grid[d][f] is the name of the signal that feature f holds at depth d,
or None. Row 0 is what the network is given, the last row holds the outputs, and an MLP is at the
depth where its outputs first appear: it reads the row before. ONE, the constant, is feature 0 of
every row, and a signal keeps its feature for as long as the grid holds it. A grid has names
only, no weights, and every emitter reads the same one: it makes whatever it takes for each row
to follow from the row before.
"""

from collections import defaultdict
from collections.abc import Collection
from math import inf

from .trace import Trace

ONE = "ONE"  # the constant, feature 0 of every row: a layer reads it as 1, for its biases
Grid = list[list[str | None]]


def layout(t: Trace, wait: int = 1, clear: Collection[int] = ()) -> Grid:
    """A grid for t: row 0 is ONE and the inputs, and each MLP goes right after the last of the
    MLPs it reads. A signal's feature is freed at the depth of the last MLP that reads it, which
    reads the row before, and an output's never is. A signal made at depth d takes, of the free
    features whose last signal was made at d - wait or before, the one freed last, or else a new
    one. After each row of `clear`, the features free there are cleared, and can be taken at once;
    those that hold a signal there are taken again only after the next row of `clear`. Both are
    for a network that cannot erase a feature, and writes over it (see beside.py): by default, a
    feature is taken again as soon as it is freed"""
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
    free = {}  # each free feature: the depth its last signal was made at, or -inf if cleared since
    cleared_at = -inf  # the last row of clear so far
    for d in range(1, max(depth.values(), default=0) + 1):
        row = list(row)
        for x in freed[d]:
            free[at[x]] = depth[x]
            row[at[x]] = None
        ready = [f for f, when in free.items() if when == -inf or cleared_at < when <= d - wait]
        for y in made[d]:
            if ready:
                at[y] = ready.pop()
                del free[at[y]]
            else:  # a new feature
                at[y] = len(row)
                row.append(None)
            row[at[y]] = y
        rows.append(row)
        if d in clear:
            free, cleared_at = dict.fromkeys(free, -inf), d
    return [row + [None] * (len(rows[-1]) - len(row)) for row in rows]  # all as wide as the last


def held(grid: Grid, clear: Collection[int] = ()) -> Grid:
    """grid with each signal left in its feature until another is written there, as a skip
    connection leaves it, or until the feature is cleared, after a row of `clear`: emit, with
    skip, then erases only what is written over. Not for a stack, which would copy what is left
    at every layer"""
    rows = grid[:1]
    for d in range(1, len(grid)):
        left = grid[d - 1] if d - 1 in clear else rows[-1]  # a clear leaves the row's own signals
        rows.append([s or old for s, old in zip(grid[d], left)])
    return rows

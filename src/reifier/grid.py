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

ONE = "ONE"  # the constant, feature 0 of every row: a layer reads it as 1, for its biases (a
# transformer holds it in more features: see transformer.token_grid)
Grid = list[list[str | None]]


def layout(t: Trace, rest: bool = False, first=None, home=None) -> Grid:
    """A grid for t: row 0 is ONE and the inputs, and each MLP goes right after the last of the
    MLPs it reads. A signal's feature is freed at the depth of the last MLP that reads it, which
    reads the row before, and an output's never is. A signal takes the feature that was freed
    last, or else a new one. With rest, it takes none that is freed at its own depth: a layer
    then writes a new signal only where the row before held none, and what nothing reads of an
    MLP is held for as long as what is read of it (see transformer.py). `first` is another row 0
    to start from, which holds the inputs that are read, and `home` the feature, of those that
    it holds nothing in, of each signal that has one of its own: no other takes it"""
    depth = dict.fromkeys(t.inputs, 0)  # the depth of each signal
    last = dict.fromkeys(t.outputs, inf)  # the depth of the last MLP that reads each signal
    for m in t.mlps:
        d = 1 + max((depth[x] for x in m.inputs), default=0)
        depth |= dict.fromkeys(m.outputs, d)
        last |= {x: max(last.get(x, 0), d) for x in m.inputs}
    for m in t.mlps if rest else ():
        read = [last[y] for y in m.outputs if y in last]
        last |= {y: max(read) for y in m.outputs if read and y not in last}
    made, freed = defaultdict(list), defaultdict(list)  # the signals each depth makes, and frees
    for x, d in depth.items():
        made[d].append(x)
        freed[last.get(x, d + 1)].append(x)  # what nothing reads is freed after its own depth
    row, home = first or [ONE, *t.inputs], home or {}
    rows = [row]
    at = {x: f for f, x in enumerate(row) if x} | home  # the feature of each signal
    free = [f for f, x in enumerate(row) if x is None and f not in home.values()]  # in turn
    for d in range(1, max(depth.values(), default=0) + 1):
        row = list(row)
        gone = [at[x] for x in freed[d] if x in at]  # an input that nothing reads has no feature
        for f in gone:
            row[f] = None
        gone = [f for f in gone if f not in home.values()]  # a home is no other signal's
        free += gone * (not rest)
        for y in made[d]:
            if y not in home and not free:  # a new feature
                free.append(len(row))
                row.append(None)
            at[y] = home[y] if y in home else free.pop()
            row[at[y]] = y
        free += gone * rest
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

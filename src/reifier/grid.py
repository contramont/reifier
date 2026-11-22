"""Stage 2, the grid: which signal each feature holds at each depth.

A grid is a list of rows: grid[d][f] is the name of the signal that feature f holds at depth d,
or None. Row 0 is what the network is given, the last row holds the outputs, and an MLP is at the
depth where its outputs first appear: it reads the row before. ONE, the constant, is feature 0 of
every row, and a signal keeps its feature for as long as the grid holds it. A grid has names
only, no weights, and every emitter reads the same one: it makes whatever it takes for each row
to follow from the row before.

A feature holds one signal after another, so a grid need be no wider than the most signals that
are held at one depth. Which signal follows which is the layout's choice, and to a transformer
it matters: a block writes a bit more exactly where the row before held nothing than over
another bit, and a signal that can be 1/2 not over another at all (see transformer.py). layout,
with rest, chooses for that.
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
    last, or else a new one: so a layer may write it where the row before held another. With
    rest, the grid is for a transformer (see transformer.py), and a signal has three choices
    before a new feature. It takes one that held nothing in the row before, the one freed
    last; or else the feature of an input of its MLP that is freed at its own depth, the first
    of them; or else any that is freed at its depth, the last. So a block writes over another
    signal only where no feature is empty, and then over a bit that the new one is made of, if
    it can: a bit is often what one of its inputs was, and its feature then gets nothing. In a
    round of Keccak of three layers, most bits of chi are, three times in four, the first bit
    that they read (see examples/keccak.py). What an MLP of several outputs gives, as a table
    with a rest does, has the first choice alone, and no signal takes its feature at the depth
    that frees it: it can be 1/2, which is not written over a signal, nor a signal over it.
    What nothing reads of such an MLP is held for as long as what is read of it, as the two
    are as large between them, and are erased together. `first` is another row 0 to start
    from, which holds the inputs that are read, and `home` the feature, of those that it holds
    nothing in, of each signal that has one of its own. A home is lent: another signal takes
    it if that is freed before the home's own signal is made, which is then written where the
    row before held nothing"""
    depth = dict.fromkeys(t.inputs, 0)  # the depth of each signal
    last = dict.fromkeys(t.outputs, inf)  # the depth of the last MLP that reads each signal
    for m in t.mlps:
        d = 1 + max((depth[x] for x in m.inputs), default=0)
        depth |= dict.fromkeys(m.outputs, d)
        last |= {x: max(last.get(x, 0), d) for x in m.inputs}
    several = {y for m in t.mlps if len(m.outputs) > 1 for y in m.outputs} if rest else set()
    inputs = {y: m.inputs for m in t.mlps for y in m.outputs} if rest else {}
    for m in t.mlps if rest else ():
        read = [last[y] for y in m.outputs if y in last]
        last |= {y: max(read) for y in m.outputs if read and y not in last}
    made, freed = defaultdict(list), defaultdict(list)  # the signals each depth makes, and frees
    for x, d in depth.items():
        made[d].append(x)
        freed[last.get(x, d + 1)].append(x)  # what nothing reads is freed after its own depth
    row, home = first or [ONE, *t.inputs], home or {}
    rows, lent = [row], {f: depth.get(y, inf) - 1 for y, f in home.items()}  # a home: until when
    at = {x: f for f, x in enumerate(row) if x} | home  # the feature of each signal
    free = dict.fromkeys(f for f, x in enumerate(row) if x is None)  # that held nothing, in turn
    for d in range(1, max(depth.values(), default=0) + 1):
        before, row = row, list(row)
        gone = [(at[x], x in several) for x in freed[d] if x in at]  # an unread input has none
        for f, _ in gone:
            row[f] = None
        now = dict.fromkeys(f for f, held in gone if not held)  # freed here: to be written over
        for y in made[d]:
            if y not in home:  # else it takes its home; and no home that it would not leave in time
                until = last.get(y, d + 1)  # the depth that frees it
                own = [at[x] for x in inputs.get(y, ()) if before[at[x]] == x and at[x] in now]
                order = [reversed(now), reversed(free)]  # the one freed last
                if rest:  # one that held nothing; of its inputs', the first; the one freed last
                    order = [reversed(free), own, reversed(now)] if y not in several else order[1:]
                fits = (f for fs in order for f in fs if until <= lent.get(f, inf))
                at[y] = next(fits, len(row))
                row += [None] * (at[y] == len(row))
            free.pop(at[y], None), now.pop(at[y], None)
            row[at[y]] = y
        free |= now | dict.fromkeys(f for f, held in gone if held)
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

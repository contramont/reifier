"""Tiles: a layout that keeps each circuit together, as a rectangle of the grid.

The MLPs that one call of a circuit makes are next to each other in a trace. So with a feature
for each signal, in the order of the trace, a call is a rectangle of the grid that holds its
signals and no others, a tile, and the calls inside it are tiles inside it. No feature is used
twice: each means one signal at every depth, and the grid is as wide as the trace is long, where
layout's is as narrow as it can make it. tiles gives a grid, as layout does, so the emitters and
the checks are the same.
"""

from .grid import ONE, Grid, layout
from .trace import Trace


def tiles(t: Trace) -> Grid:
    """A grid for t: layout's depths, and a feature for each signal, in the order of the trace"""
    order = [ONE, *t.inputs, *(y for m in t.mlps for y in m.outputs)]
    return [[s if s in row else None for s in order] for row in map(set, layout(t))]

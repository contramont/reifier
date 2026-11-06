"""Reifier: compile Python functions on bits into neural networks."""

from .core import (
    LOWERING_RULES,
    Circuit,
    Graph,
    Signal,
    bits,
    circuit,
    flatten,
    format_graph,
    lower,
    placeholders,
    trace,
)
from .fold import fold_constants
from .lib import add, and_, at_least, buffer, not_, or_, popcount, xor
from .mlp import ACTIVATIONS, MLP, const, glu, mlp, threshold_gate
from .network import assemble, network
from .place import place

"""Reifier: compile Python functions on bits into neural networks."""

from .attention import Attention, lookback
from .compiler import schedule
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
from .decoder import Architecture, transformer
from .fold import fold_constants
from .lib import add, and_, at_least, buffer, not_, or_, popcount, xor
from .mlp import ACTIVATIONS, MLP, const, glu, mlp, passthrough, threshold_gate
from .network import network

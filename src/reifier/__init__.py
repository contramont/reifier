"""Reifier: compile Python functions on bits into neural networks."""

from .emit import Network, check_net, emit
from .fold import fold
from .grid import ONE, Grid, check_grid, held, layout
from .lib import add, and_, not_, or_, parity, sum_of_ands, xor
from .tiles import tiles
from .trace import MLP, Trace, check_trace, circuit, const, glu, mlp, scope, threshold_gate, trace
from .transformer import Transformer, check_transformer, transformer

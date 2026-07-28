"""Fast array-based circuit compilation.

- `fast_compile(fn, *args, **kwargs)`: run a circuit-building function and
  level the resulting signal graph into flat numpy arrays.
- `stamped(fn)`: cache a function's gate structure so repeated calls stamp a
  pre-leveled template instead of rebuilding gates (see `reifier.fast.stamp`).
"""

from typing import Any
from collections.abc import Callable

from .leveler import LeveledArrays, level_graph, flatten_bits
from .stamp import stamped, Stamped, StampError


def fast_compile(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> LeveledArrays:
    """Compile `fn` into a leveled circuit by running it on the given inputs.

    The Bits found in `args`/`kwargs` (flattened in order) become the input
    level; the Bits in the return value become the output level.
    """
    result = fn(*args, **kwargs)
    inputs = flatten_bits((args, kwargs))
    outputs = flatten_bits(result)
    return level_graph(inputs, outputs)


__all__ = [
    "LeveledArrays",
    "level_graph",
    "flatten_bits",
    "fast_compile",
    "stamped",
    "Stamped",
    "StampError",
]

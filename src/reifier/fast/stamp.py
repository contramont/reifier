"""Subcircuit template stamping.

`stamped(fn)` wraps a circuit-building function so that its gate structure is
traced only once per call signature. Later calls with the same signature skip
re-executing `fn`: they evaluate the cached, pre-leveled `Template` with
numpy and return fresh output Bits wired to a `StampInstance`. The fast
leveler then assembles each instance into the compiled arrays with
vectorized index arithmetic instead of walking its gates one by one.

Soundness contract for a stamped function:
- its gate structure must depend only on the shapes of its arguments, the
  values of its non-Bit arguments, and the aliasing pattern of its Bit
  arguments (all three are part of the cache key), and
- every external Bit it consumes must be reachable through its arguments
  (no live closure Bits; closure constants are rejected too).
Contract violations are detected during tracing and raise `StampError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import numpy as np

from reifier.neurons.core import Bit, Signal, take_uid
from reifier.utils.format import Bits
from .leveler import (
    LeveledArrays,
    StampInstance,
    StampOutput,
    flatten_bits,
    level_graph,
)


class StampError(ValueError):
    pass


class _BitLeaf:
    """Placeholder for a Bit in a stored output skeleton."""

    __slots__ = ()


_BIT_LEAF = _BitLeaf()


@dataclass(frozen=True, slots=True)
class Template:
    """A traced, leveled subcircuit plus the wiring info to re-instantiate it."""

    arrays: LeveledArrays  # level 0 = captured inputs; last level = gated outputs
    arg_positions: list[int]  # input slot -> index into flattened call-arg bits
    out_kinds: list[tuple[str, int]]  # ("gated", slot) or ("pass", arg position)
    skeleton: Any  # return structure with Bits replaced by _BIT_LEAF


def _signature(obj: Any) -> Any:
    """Hashable structural signature: Bit positions + non-Bit values."""
    if isinstance(obj, Signal):
        return _BIT_LEAF
    if isinstance(obj, Bits):
        return ("Bits", len(obj.bit_tuple))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(_signature(el) for el in obj))
    if isinstance(obj, dict):
        return ("dict", tuple((k, _signature(v)) for k, v in obj.items()))
    if isinstance(obj, (bool, int, float)):
        # tag with the type so True, 1 and 1.0 don't share a cache key
        return (type(obj).__name__, obj)
    if obj is None or isinstance(obj, (str, bytes)):
        return obj
    raise StampError(f"Unsupported argument type for a stamped function: {type(obj)}")


def _alias_pattern(flat_args: list[Bit]) -> tuple[int, ...]:
    """First-occurrence index of each argument Bit.

    Part of the cache key: a call passing the same Bit object at two
    positions must not share a template with a call passing distinct Bits,
    since tracing collapses aliased positions into one template input slot.
    """
    first: dict[int, int] = {}
    return tuple(first.setdefault(id(b), p) for p, b in enumerate(flat_args))


def _make_skeleton(obj: Any) -> Any:
    """Copy a return structure, replacing Bit leaves with _BIT_LEAF."""
    if isinstance(obj, Signal):
        return _BIT_LEAF
    if isinstance(obj, Bits):
        return obj  # rebuilt via Bits(list) later
    if isinstance(obj, list):
        return [_make_skeleton(el) for el in obj]
    if isinstance(obj, tuple):
        return tuple(_make_skeleton(el) for el in obj)
    if isinstance(obj, dict):
        return {k: _make_skeleton(v) for k, v in obj.items()}
    return obj


def _rebuild(skeleton: Any, leaves: list[Bit], pos: list[int]) -> Any:
    """Rebuild a skeleton, substituting Bits from `leaves` in flatten order."""
    if isinstance(skeleton, _BitLeaf):
        leaf = leaves[pos[0]]
        pos[0] += 1
        return leaf
    if isinstance(skeleton, Bits):
        n = len(skeleton.bit_tuple)
        taken = leaves[pos[0] : pos[0] + n]
        pos[0] += n
        return Bits(taken)
    if isinstance(skeleton, list):
        return [_rebuild(el, leaves, pos) for el in skeleton]
    if isinstance(skeleton, tuple):
        return tuple(_rebuild(el, leaves, pos) for el in skeleton)
    if isinstance(skeleton, dict):
        return {k: _rebuild(v, leaves, pos) for k, v in skeleton.items()}
    return skeleton


class Stamped:
    """Wrapper that traces a circuit function once per signature, then stamps."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.cache: dict[Any, Template] = {}
        self.__name__ = getattr(fn, "__name__", "stamped")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        flat_args = flatten_bits((args, kwargs))
        key = (_signature((args, tuple(kwargs.items()))), _alias_pattern(flat_args))
        template = self.cache.get(key)
        if template is None:
            template, result = self._trace(args, kwargs, flat_args)
            if template is not None:
                self.cache[key] = template
            return result
        return self._stamp(template, flat_args)

    def _trace(
        self, args: tuple, kwargs: dict, flat_args: list[Bit] | None = None
    ) -> tuple[Template | None, Any]:
        start_uid = take_uid()
        result = self.fn(*args, **kwargs)
        out_bits = flatten_bits(result)
        if flat_args is None:
            flat_args = flatten_bits((args, kwargs))
        pos_by_id: dict[int, int] = {}
        for p, b in enumerate(flat_args):
            pos_by_id.setdefault(id(b), p)

        # Find captured inputs: bits created before the call that the
        # subcircuit consumes. They must all be reachable via the arguments.
        captured: dict[int, Bit] = {}
        seen: set[int] = set()
        stack: list[Bit] = list(out_bits)
        while stack:
            b = stack.pop()
            bid = id(b)
            if bid in seen:
                continue
            seen.add(bid)
            if b.uid <= start_uid:
                if bid not in pos_by_id:
                    raise StampError(
                        f"stamped({self.__name__}): consumes a Bit that is not "
                        "reachable through its arguments (closure Bits are "
                        "not supported)"
                    )
                captured[bid] = b
                continue
            src = b.source
            if isinstance(src, StampOutput):
                stack.extend(src.instance.input_bits)
            elif src.incoming:
                stack.extend(src.incoming)

        captured_list = sorted(captured.values(), key=lambda b: pos_by_id[id(b)])
        arg_positions = [pos_by_id[id(b)] for b in captured_list]

        out_kinds: list[tuple[str, int]] = []
        gated: list[Bit] = []
        gated_slot: dict[int, int] = {}
        for b in out_bits:
            if b.uid <= start_uid:
                if id(b) not in pos_by_id:
                    raise StampError(
                        f"stamped({self.__name__}): returns a Bit that is "
                        "neither created inside it nor passed via arguments"
                    )
                out_kinds.append(("pass", pos_by_id[id(b)]))
            else:
                if id(b) in gated_slot:
                    raise StampError(
                        f"stamped({self.__name__}): returns the same Bit "
                        "twice; wrap duplicates in a copy gate"
                    )
                gated_slot[id(b)] = len(gated)
                out_kinds.append(("gated", len(gated)))
                gated.append(b)

        if not gated:
            return None, result  # nothing to stamp (pure passthrough)

        arrays = level_graph(captured_list, gated)
        template = Template(
            arrays=arrays,
            arg_positions=arg_positions,
            out_kinds=out_kinds,
            skeleton=_make_skeleton(result),
        )
        return template, result

    def _stamp(self, template: Template, flat_args: list[Bit]) -> Any:
        input_bits = [flat_args[p] for p in template.arg_positions]
        values = np.fromiter(
            (b.activation for b in input_bits), dtype=np.int64, count=len(input_bits)
        )
        out_values = template.arrays.run(values)
        instance = StampInstance(template, input_bits, take_uid())
        leaves: list[Bit] = []
        for kind, idx in template.out_kinds:
            if kind == "pass":
                leaves.append(flat_args[idx])
            else:
                leaves.append(
                    Signal(bool(out_values[idx]), StampOutput(instance, idx))  # type: ignore[arg-type]
                )
        return _rebuild(template.skeleton, leaves, [0])


_registry: dict[Any, Stamped] = {}


def stamped(fn: Callable[..., Any]) -> Stamped:
    """Return the (shared) stamping wrapper for `fn`."""
    existing = _registry.get(fn)
    if existing is None:
        existing = Stamped(fn)
        _registry[fn] = existing
    return existing

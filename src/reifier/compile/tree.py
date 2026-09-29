from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from reifier.neurons.core import GluNeuron, Unit
from .levels import LeveledGraph, Level, Origin, Parent
from .blocks import Block, BlockTracer, traverse


@dataclass(frozen=True)
class Tree(LeveledGraph):
    """A tree representation of a function"""

    root: Block
    origin_blocks: list[list[Block]]

    @classmethod
    def from_root(
        cls, root: Block, remove_redundant_outputs_layer: bool = True
    ) -> "Tree":
        origin_blocks = cls._set_origins(root)
        cls._set_narrow_origins(origin_blocks)
        levels = [Level(tuple([b.origin for b in level])) for level in origin_blocks]
        # gated units do not re-threshold, so keep the outputs layer after units that
        # read computed (inexact) bits
        rethreshold = len(levels) > 3 and any(o.units for o in levels[-2].origins)
        if cls.has_redundant_outputs_layer(levels) and remove_redundant_outputs_layer:
            levels = levels if rethreshold else levels[:-1]
        return cls(root=root, origin_blocks=origin_blocks, levels=tuple(levels))

    @staticmethod
    def _set_origins(root: Block) -> list[list[Block]]:
        depth = root.top
        levels: list[list[Block]] = [[] for _ in range(depth)]

        # Set connections and add to levels
        for b in traverse(root):
            if b.flavour == "gate" or b.flavour == "folded":
                b.origin = Tree._gate_origin(b)
            elif b.flavour == "copy":
                (inp,) = b.inputs
                b.origin = Origin(b.abs_x, (Parent(inp.creator.abs_x, 1),), -1)
            else:
                continue
            levels[b.abs_y].append(b)

        # set origins for inputs
        input_blocks: list[Block] = []
        assert len(levels[0]) == 0
        for b in traverse(root):
            if b.flavour == "input":
                input_blocks.append(b)
        levels[0] = input_blocks
        for j, b in enumerate(levels[0]):
            b.origin = Origin(j, (), -1)

        # set origins for outputs
        for j, out in enumerate(root.outputs):
            b = out.creator
            assert b is not None
            incoming = [
                Parent(inp.creator.abs_x, 1)
                for inp in b.inputs
                if inp.creator is not None
            ]
            b.origin = Origin(j, tuple(incoming), -1)
            levels[-1].append(b)

        if len(root.outputs) == 0:
            print("Warning, no outputs detected in the compiled function")
        if len(input_blocks) == 0:
            print("Warning, no inputs detected in the compiled function")

        return levels

    @staticmethod
    def _gate_origin(b: Block) -> Origin:
        """Origin of a gate or glu. Its inputs are matched by Bit, as b.inputs has no
        repeats or constants: repeats get a Parent each, constants fold into biases"""
        neuron = b.creation.data.source
        index_of = {inp.data: inp.creator.abs_x for inp in b.inputs}
        parents = [index_of[x] for x in neuron.incoming if x in index_of]

        def fold(weights, bias):  # -> weights of the parents, bias
            pairs = list(zip(neuron.incoming, weights))
            bias += sum(w * x.activation for x, w in pairs if x not in index_of)
            return tuple(w for x, w in pairs if x in index_of), bias

        if isinstance(neuron, GluNeuron):  # units on the parents; bias -1 debiases to 0
            units = tuple(
                Unit(*fold(u.weights, u.bias), *fold(u.value_weights, u.value_bias))
                for u in neuron.units
            )
            return Origin(b.abs_x, tuple(Parent(i, 0) for i in parents), -1, units)
        weights, bias = fold(neuron.weights, neuron.bias)
        return Origin(b.abs_x, tuple(map(Parent, parents, weights)), bias)

    @staticmethod
    def _set_narrow_origins(origin_blocks: list[list[Block]]) -> None:
        # record narrow indices
        to_narrow_index: dict[tuple[int, int], int] = dict()
        for i, level in enumerate(origin_blocks):
            for j, b in enumerate(level):
                to_narrow_index[(i, b.origin.index)] = j

        # set narrow indices
        for i, level in enumerate(origin_blocks[1:], start=1):  # skip input level
            for j, b in enumerate(level):
                origin = b.origin
                try:
                    index = to_narrow_index[(i, b.origin.index)]
                    incoming = [
                        Parent(to_narrow_index[(i - 1, p.index)], p.weight)
                        for p in origin.incoming
                    ]
                except KeyError:
                    raise KeyError(f"KeyError when setting narrow origins for {b.path}")
                b.origin = Origin(index, tuple(incoming), origin.bias, origin.units)

    @staticmethod
    def has_redundant_outputs_layer(levels: list[Level]) -> bool:
        """Detects if the outputs layer is identical to the layer below it"""
        if not len(levels[-1].origins) == len(levels[-2].origins):
            return False
        for o1 in levels[-1].origins:
            if len(o1.incoming) != 1:
                return False
            p = o1.incoming[0]
            if o1.index != p.index or o1.bias != -1 or p.weight != 1:
                return False
        return True

    def print_activations(self) -> None:
        for i, level in enumerate(self.origin_blocks):
            level_activations = [b.creation.data.activation for b in level]
            print(i, "".join(str(int(a)) for a in level_activations))


@dataclass(frozen=True)
class TreeCompiler:
    collapse: set[str] = field(default_factory=set[str])

    def validate(self, args: Any, kwargs: Any) -> None:
        if self.collapse & {"gate", "glu"}:
            raise ValueError("gate and glu cannot be collapsed")
        # dummy_inp = find(kwargs, Bit)
        # for bit, _ in dummy_inp:
        #     if bit.activation != 0:
        #         print("Warning: Dummy input has non-zero values")
        #         break

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tree:
        """Compiles a function into a tree."""
        self.validate(args, kwargs)

        tracer = BlockTracer(self.collapse)
        root = tracer.run(fn, *args, **kwargs)
        tree = Tree.from_root(root)
        return tree

"""Compile a function of Bits at an optimization level (see the package docstring)."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch as t

from reifier.compile.levels import LeveledGraph
from reifier.tensors.swiglu import MLP_SwiGLU
from .build import FP16_MAX, MLPOptions, build, dense_bound, fp16_bound
from .fanin import Fanin
from .graph import GraphCompiler, GraphOptions
from .recipes import MAX_DEPTH, RECIPES, TIERS, candidates, recipe


@dataclass
class Compiler:
    """Compiles a function of Bits to an MLP_SwiGLU at a level: the smallest of its
    candidate recipes, which chosen names after run(). A level is also the name of its own
    recipe; select=False compiles the recipe named by level alone (any of RECIPES). knobs
    are laid over the knobs of every recipe compiled (see recipe)."""

    level: str
    mlp_dtype: t.dtype = t.float32
    select: bool = True
    knobs: dict[str, Any] = field(default_factory=dict)
    chosen: str | None = field(default=None, init=False)  # the recipe run() kept

    def __post_init__(self) -> None:
        if self.level not in RECIPES:
            raise ValueError(f"unknown level {self.level!r}, not in {list(TIERS)} "
                             f"(or a recipe of {list(RECIPES)})")
        _stages(recipe(self.level, **self.knobs))  # unknown or invalid knobs raise here

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> MLP_SwiGLU:
        if self.select:
            self.chosen, mlp = self._select(fn, args, kwargs)
        else:
            mlp = self.get_mlp_from_graph(self.get_graph(fn, *args, **kwargs))
        if self._fp16_use() and (bound := fp16_bound(mlp)) > FP16_MAX:
            warnings.warn(
                f"values in the MLP can reach {bound:.3g} > {FP16_MAX:g} (see "
                "reifier.opt.fp16_bound): float16 may overflow; use bfloat16",
                stacklevel=2,
            )
        return mlp

    def get_graph(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> LeveledGraph:
        """the leveled graph of the recipe named by level"""
        fanin, passes, _ = _stages(recipe(self.level, **self.knobs))
        return GraphCompiler(passes).compile(*fanin.trace(fn, *args, **kwargs))

    def get_mlp_from_graph(self, graph: LeveledGraph) -> MLP_SwiGLU:
        """the MLP of the recipe named by level (no float16 warning)"""
        return build(graph, _stages(recipe(self.level, **self.knobs))[2], self.mlp_dtype)

    def _select(self, fn, args, kwargs) -> tuple[str, MLP_SwiGLU]:
        """compile every candidate's graph and keep the smallest MLP: dense parameters,
        then nonzeros, then the more robust; where float16 is in the tier, those within its
        range first. fn is traced once per distinct Fanin (GraphCompiler only reads
        Bits), and MLPs are built in the order of their dense_bound while it does not
        exceed the best size so far"""
        graphs, traces = [], {}
        for rank, name in enumerate(candidates(self.level)):
            knobs = recipe(name, **self.knobs)
            fanin, passes, _ = _stages(knobs)
            if fanin not in traces:
                traces[fanin] = fanin.trace(fn, *args, **kwargs)
            graph = GraphCompiler(passes).compile(*traces[fanin])
            if len(graph.levels) - 1 > MAX_DEPTH.get(name, len(graph.levels)):
                continue  # beyond the recipe's depth limit
            graphs.append((dense_bound(graph), rank, name, knobs, graph))
        fp16 = self._fp16_use()
        best: tuple[tuple[int, int, int, int], str, MLP_SwiGLU] | None = None
        for bound, rank, name, knobs, graph in sorted(graphs, key=lambda g: g[:2]):
            if best is not None and best[0][0] == 0 and bound > best[0][1]:
                break  # bounds are sorted: no later candidate can be smaller
            mlp = build(graph, _stages(knobs)[2], self.mlp_dtype)
            ps = list(mlp.parameters())
            over = int(fp16 and fp16_bound(mlp) > FP16_MAX)
            key = (over, sum(p.numel() for p in ps), sum(int((p != 0).sum()) for p in ps), rank)
            if best is None or key < best[0]:
                best = (key, name, mlp)
            del mlp, ps  # free each candidate before building the next
        assert best is not None
        return best[1], best[2]

    def _fp16_use(self) -> bool:
        """whether the build may run in float16: float16 builds, and float32 builds of T3+
        levels and recipes, whose tiers include casting to float16"""
        return self.mlp_dtype == t.float16 or (
            self.mlp_dtype == t.float32 and RECIPES[self.level][0] >= 3)


def _stages(knobs: dict[str, Any]) -> tuple[Fanin, GraphOptions, MLPOptions]:
    """a recipe's knobs as the options of its three stages"""
    k = dict(knobs)
    fan = {n.removeprefix("fanin_"): k.pop(n) for n in list(k) if n.startswith("fanin_")}
    return Fanin(**fan), GraphOptions(**k.pop("passes")), MLPOptions(**k)

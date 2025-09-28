from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

import torch as t

from reifier.compile.tree import Tree, TreeCompiler
from .matrices import Matrices
from .mlp import MLP
from .swiglu import MLP_SwiGLU
# from .step import MLP_Step


@dataclass
class Compiler:
    # mlp_type: type[MLP_SwiGLU] | type[MLP_Step] = field(default=MLP_SwiGLU)
    mlp_dtype: t.dtype = t.float
    collapse: set[str] = field(default_factory=set[str])
    c: int = 4  # making ReLU-simulated step fn steeper
    q: int = 4  # scaling before and after SiLU to avoid non-ReLU-like dip

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> MLP:
        tree = self.get_tree(fn, *args, **kwargs)
        mlp = self.get_mlp_from_tree(tree)
        return mlp

    def get_tree(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tree:
        return TreeCompiler(self.collapse).run(fn, *args, **kwargs)

    def get_mlp_from_tree(self, tree: Tree) -> MLP_SwiGLU:
        matrices = Matrices.from_graph(tree)
        mlp = MLP_SwiGLU.from_matrices(
            matrices, c=self.c, q=self.q, dtype=self.mlp_dtype
        )
        return mlp

    # def get_mlp_from_tree(self, tree: Tree) -> MLP:
    #     matrices = Matrices.from_graph(tree)
    #     mlp = self.mlp_type.from_matrices(matrices, dtype=self.mlp_dtype)
    #     return mlp

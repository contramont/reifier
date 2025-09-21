from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

from reifier.compile.tree import Tree, TreeCompiler
from .matrices import Matrices
from .mlp import MLP
from .swiglu import MLP_SwiGLU
from .step import MLP_Step


@dataclass
class Compiler:
    mlp_type: type[MLP_SwiGLU] | type[MLP_Step] = field(default=MLP_SwiGLU)
    collapse: set[str] = field(default_factory=set[str])

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> MLP:
        tree = self.get_tree(fn, *args, **kwargs)
        mlp = self.get_mlp_from_tree(tree)
        return mlp

    def get_tree(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tree:
        return TreeCompiler(self.collapse).run(fn, *args, **kwargs)

    def get_mlp_from_tree(self, tree: Tree) -> MLP:
        matrices = Matrices.from_graph(tree)
        return self.mlp_type.from_matrices(matrices)

from dataclasses import dataclass

from reifier.compile.levels import LeveledGraph, Level, Origin, Parent
from reifier.sparse.compile import Node, NodeGraph


@dataclass(frozen=True)
class SparseGraph(LeveledGraph):
    """A tree representation of a function"""

    @classmethod
    def _node_to_origin(cls, node: Node) -> Origin:
        return Origin(
            node.column,
            tuple([Parent(p.column, int(node.weights[p])) for p in node.parents]),
            int(node.bias),
        )

    @classmethod
    def from_node_graph(cls, graph: NodeGraph) -> "SparseGraph":
        levels = [
            Level(tuple([cls._node_to_origin(node) for node in layer]))
            for layer in graph.layers
        ]
        return cls(levels=tuple(levels))

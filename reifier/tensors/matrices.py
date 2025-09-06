from dataclasses import dataclass

import torch as t

from reifier.sparse.compile import Graph, Node
from reifier.compile.tree import Tree
from reifier.compile.levels import Level


@dataclass(frozen=True, slots=True)
class Matrices:
    mlist: list[t.Tensor]
    dtype: t.dtype = t.int


    @classmethod
    def from_graph(cls, graph: Graph, dtype: t.dtype = t.int) -> "Matrices":
        """Set parameters of the model from weights and biases"""
        layers = graph.layers[1:]  # skip input layer as it has no incoming weights
        sizes_in = [len(layer) for layer in graph.layers]  # incoming weight sizes
        params = [
            cls.layer_to_params(layer, s, dtype) for layer, s in zip(layers, sizes_in)
        ]  # w&b pairs
        matrices = [cls.fold_bias(w.to_dense(), b) for w, b in params]  # dense matrices
        # matrices[-1] = matrices[-1][1:]  # last layer removes the constant input feature
        return cls(matrices, dtype=dtype)


    @staticmethod
    def layer_to_params(
        layer: list[Node], size_in: int, dtype: t.dtype, debias: bool = True
    ) -> tuple[t.Tensor, t.Tensor]:
        """
        Convert layer to a sparse weight matrix and dense bias matrix
        Debias adds 1 to biases, shifting the default bias from -1 to sparser 0.
        Linear Threshold Circuits use a default threshold of >=0, i.e. bias = -1.
        """
        row_idx: list[int] = []
        col_idx: list[int] = []
        val_lst: list[int | float] = []
        for j, node in enumerate(layer):
            for p in node.parents:
                row_idx.append(j)
                col_idx.append(p.column)
                val_lst.append(node.weights[p])
        indices = t.tensor([row_idx, col_idx], dtype=t.long)
        values = t.tensor(val_lst, dtype=dtype)
        size = (len(layer), size_in)
        w_sparse = t.sparse_coo_tensor(indices, values, size, dtype=dtype)  # type: ignore
        b = t.tensor([node.bias for node in layer], dtype=dtype)
        if debias:
            b += 1
        return w_sparse, b


    @classmethod
    def layer_to_params_2(
        cls,
        level: Level,
        size_in: int,
        size_out: int,
        dtype: t.dtype = t.int,
        debias: bool = True,
    ) -> tuple[t.Tensor, t.Tensor]:
        # TODO: combine with layer_to_params, routing both through Graph Levels

        row_idx: list[int] = []
        col_idx: list[int] = []
        val_lst: list[int | float] = []
        for origin in level.origins:
            for p in origin.incoming:
                row_idx.append(origin.index)
                col_idx.append(p.index)
                val_lst.append(p.weight)
        indices = t.tensor([row_idx, col_idx], dtype=t.long)
        values = t.tensor(val_lst, dtype=dtype)
        w_sparse = t.sparse_coo_tensor(  # type: ignore
            indices, values, (size_out, size_in), dtype=dtype
        )
        b = t.tensor([origin.bias for origin in level.origins], dtype=dtype)
        if debias:
            b += 1
        return w_sparse, b


    @staticmethod
    def fold_bias(w: t.Tensor, b: t.Tensor) -> t.Tensor:
        """Folds bias into weights, assuming input feature at index 0 is always 1."""
        # print("w.shape, b.shape", w.shape, b.shape)
        one = t.ones(1, 1)
        zeros = t.zeros(1, w.size(1))
        # assumes row vector bias that is transposed during forward pass
        bT = t.unsqueeze(b, dim=-1)
        wb = t.cat(
            [
                t.cat([one, zeros], dim=1),
                t.cat([bT, w], dim=1),
            ],
            dim=0,
        )
        return wb


    @property
    def sizes(self) -> list[int]:
        """Returns the activation sizes [input_dim, hidden1, hidden2, ..., output_dim]"""
        return [m.size(1) for m in self.mlist] + [self.mlist[-1].size(0)]


    @classmethod
    def from_tree(cls, tree: Tree, dtype: t.dtype = t.int) -> "Matrices":
        """Set parameters of the model from weights and biases"""
        params = [
            cls.layer_to_params_2(level_out, in_w, out_w)
            for level_out, (out_w, in_w) in zip(tree.levels[1:], tree.shapes)
        ]
        matrices = [cls.fold_bias(w.to_dense(), b) for w, b in params]
        return cls(matrices, dtype=dtype)

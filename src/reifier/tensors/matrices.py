from dataclasses import dataclass

import torch as t

from reifier.compile.levels import LeveledGraph, Level


@dataclass(frozen=True, slots=True)
class Matrices:
    mlist: list[t.Tensor]
    dtype: t.dtype = t.int

    @classmethod
    def layer_to_params(
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
    def fold_bias(w: t.Tensor, b: t.Tensor, dtype: t.dtype) -> t.Tensor:
        """Folds bias into weights, assuming input feature at index 0 is always 1."""
        w = w.to(dtype=dtype)
        one = t.ones(1, 1, dtype=dtype)
        zeros = t.zeros(1, w.size(1), dtype=dtype)
        bT = t.unsqueeze(b, dim=-1).to(dtype=dtype)
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
    def from_graph(cls, graph: LeveledGraph, dtype: t.dtype = t.int) -> "Matrices":
        """Set parameters of the model from weights and biases"""
        params = [
            cls.layer_to_params(level_out, in_w, out_w)
            for level_out, (out_w, in_w) in zip(graph.levels[1:], graph.shapes)
        ]
        matrices = [cls.fold_bias(w.to_dense(), b, dtype=dtype) for w, b in params]
        return cls(matrices, dtype=dtype)

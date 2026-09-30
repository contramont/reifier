from dataclasses import dataclass

import torch as t

from reifier.compile.levels import LeveledGraph, Level


@dataclass(frozen=True, slots=True)
class Matrices:
    mlist: list[t.Tensor]
    dtype: t.dtype = t.int
    # per layer, its gated units (see layer_to_units) or None
    ulist: tuple[tuple[t.Tensor, ...] | None, ...] = ()

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
    def layer_to_units(level: Level, size_in: int) -> tuple[t.Tensor, ...] | None:
        """Gated units (neurons.core.glu) of a level as matrices with the biases
        folded in: they add outs @ (max(0, gates @ x) * (values @ x)) to the outputs.
        Units with equal gates and proportional values share one hidden unit, and
        units whose value or gate is identically 0 are dropped."""
        units: list[tuple[dict[int, float], dict[int, float]]] = []
        scales: list[float] = []
        entries: list[tuple[int, int, float]] = []  # (row, unit, weight) of outs
        index: dict[tuple, int] = {}
        for o in level.origins:
            for u in o.units:
                g, v = {0: u.bias}, {0: u.value_bias}
                for p, gw, vw in zip(o.incoming, u.weights, u.value_weights):
                    if gw:
                        g[p.index + 1] = g.get(p.index + 1, 0) + gw
                    if vw:
                        v[p.index + 1] = v.get(p.index + 1, 0) + vw
                g = {i: w for i, w in g.items() if w != 0}
                v = {i: w for i, w in v.items() if w != 0}
                if not v or (list(g) in ([], [0]) and g.get(0, 0) <= 0):
                    continue  # always 0
                c = v[min(v)]
                key = (
                    tuple(sorted(g.items())),
                    tuple(sorted((i, round(w / c, 12)) for i, w in v.items())),
                )
                if key not in index:
                    index[key] = len(units)
                    units.append((g, v))
                    scales.append(c)
                k = index[key]
                entries.append((o.index + 1, k, c / scales[k]))
        if not units:
            return None
        gates = t.zeros(len(units), size_in + 1)
        values = t.zeros(len(units), size_in + 1)
        outs = t.zeros(len(level.origins) + 1, len(units))
        for k, (g, v) in enumerate(units):
            for i, w in g.items():
                gates[k, i] = w
            for i, w in v.items():
                values[k, i] = w
        for row, k, w in entries:
            outs[row, k] += w
        return gates, values, outs

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
        ulist = tuple(
            cls.layer_to_units(level_out, in_w)
            for level_out, (_, in_w) in zip(graph.levels[1:], graph.shapes)
        )
        return cls(matrices, dtype=dtype, ulist=ulist)

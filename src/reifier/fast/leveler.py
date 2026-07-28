"""Fast array-based circuit leveling.

Compiles a graph of `Signal`s into a layered threshold-gate representation
stored in flat numpy arrays (`LeveledArrays`) instead of per-node Python
objects. The graph may contain stamped subcircuit instances
(see `reifier.fast.stamp`): a stamped instance carries a pre-leveled
`Template` of its gates, which is assembled into the output arrays with
vectorized index arithmetic instead of being walked gate by gate.

Conventions match the rest of the package:
- a gate activates iff sum(weight * value) + bias >= 0
- copies are gates with a single weight-1 edge and bias -1
- level 0 is the inputs; the last level is exactly the outputs, in order
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from reifier.neurons.core import Bit, Signal
from reifier.compile.levels import LeveledGraph, Level, Origin, Parent


@dataclass(frozen=True, slots=True)
class LeveledArrays:
    """A leveled threshold-gate graph in flat arrays.

    Arrays at index i describe level i+1 (level 0 is the input level and has
    no gates). Edge arrays are parallel: edge k of level i+1 connects gate
    ``edge_rows[i][k]`` of that level to column ``edge_cols[i][k]`` of level i
    with weight ``edge_weights[i][k]``.
    """

    level_sizes: list[int]  # widths of levels 0..L
    edge_rows: list[np.ndarray]
    edge_cols: list[np.ndarray]
    edge_weights: list[np.ndarray]
    biases: list[np.ndarray]

    @property
    def n_gate_levels(self) -> int:
        return len(self.biases)

    @property
    def n_nodes(self) -> int:
        return sum(self.level_sizes)

    @property
    def n_edges(self) -> int:
        return sum(len(rows) for rows in self.edge_rows)

    def run(self, values: Any) -> np.ndarray:
        """Evaluate the graph on input values (0/1), returning output values."""
        acts = np.asarray(values, dtype=np.int64)
        if acts.shape != (self.level_sizes[0],):
            raise ValueError(
                f"Expected {self.level_sizes[0]} input values, got {acts.shape}"
            )
        for i in range(self.n_gate_levels):
            summed = np.bincount(
                self.edge_rows[i],
                weights=self.edge_weights[i] * acts[self.edge_cols[i]],
                minlength=self.level_sizes[i + 1],
            )
            acts = (summed + self.biases[i] >= 0).astype(np.int64)
        return acts

    def to_levels(self) -> tuple[Level, ...]:
        """Convert to `Level` tuples of `Origin`s (LeveledGraph-compatible)."""
        levels = [Level(tuple(Origin(j, (), -1) for j in range(self.level_sizes[0])))]
        for i in range(self.n_gate_levels):
            rows = self.edge_rows[i]
            order = np.argsort(rows, kind="stable")
            sorted_rows = rows[order]
            sorted_cols = self.edge_cols[i][order]
            sorted_ws = self.edge_weights[i][order]
            # bounds of each gate's run of edges in the sorted arrays
            starts = np.searchsorted(sorted_rows, np.arange(self.level_sizes[i + 1]))
            ends = np.append(starts[1:], len(sorted_rows))
            origins = tuple(
                Origin(
                    j,
                    tuple(
                        Parent(int(c), int(w))
                        for c, w in zip(
                            sorted_cols[starts[j] : ends[j]],
                            sorted_ws[starts[j] : ends[j]],
                        )
                    ),
                    int(self.biases[i][j]),
                )
                for j in range(self.level_sizes[i + 1])
            )
            levels.append(Level(origins))
        return tuple(levels)

    def to_leveled_graph(self) -> LeveledGraph:
        return LeveledGraph(levels=self.to_levels())


class StampInstance:
    """One stamped occurrence of a Template, bound to concrete input signals."""

    __slots__ = ("template", "input_bits", "uid")

    def __init__(self, template: Any, input_bits: list[Bit], uid: int):
        self.template = template  # reifier.fast.stamp.Template
        self.input_bits = input_bits
        self.uid = uid


class StampOutput:
    """Pseudo neuron: marks a Signal as output `slot` of a StampInstance."""

    __slots__ = ("instance", "slot")

    def __init__(self, instance: StampInstance, slot: int):
        self.instance = instance
        self.slot = slot


def flatten_bits(obj: Any) -> list[Bit]:
    """Deterministically flatten a nested structure into its list of Bits."""
    found: list[Bit] = []
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, Signal):
            found.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif hasattr(item, "bit_tuple"):  # reifier.utils.format.Bits
            stack.extend(reversed(item.bit_tuple))
    return found


def level_graph(inputs: list[Bit], outputs: list[Bit]) -> LeveledArrays:
    """Level the graph reachable backward from `outputs`, stopping at `inputs`."""
    return _Leveler(inputs, outputs).build()


class _Leveler:
    def __init__(self, inputs: list[Bit], outputs: list[Bit]):
        self.inputs = inputs
        self.outputs = outputs
        self.input_pos: dict[int, int] = {}
        for i, b in enumerate(inputs):
            if id(b) in self.input_pos:
                raise ValueError(
                    f"Duplicate input bit at positions {self.input_pos[id(b)]} and {i}"
                )
            self.input_pos[id(b)] = i
        seen_out: set[int] = set()
        for b in outputs:
            if id(b) in seen_out:
                raise ValueError(
                    "Duplicate output bits are not supported; "
                    "wrap duplicates in a copy gate"
                )
            seen_out.add(id(b))

        # populated by _discover
        self.nodes: dict[int, Any] = {}  # node id -> Signal | StampInstance
        self.deps: dict[int, tuple] = {}  # node id -> dep objects
        # populated by _assign_depths
        self.depth: dict[int, int] = {}
        self.const_val: dict[int, int] = {}
        self.folded_bias: dict[int, int] = {}

    # ---------------- phase 1: backward reachability ----------------

    def _discover(self) -> None:
        nodes, deps = self.nodes, self.deps
        input_pos = self.input_pos
        stack: list[Any] = list(self.outputs)
        while stack:
            b = stack.pop()
            nid = id(b)
            if nid in nodes:
                continue
            nodes[nid] = b
            if nid in input_pos:
                deps[nid] = ()
                continue
            src = b.source
            if isinstance(src, StampOutput):
                inst = src.instance
                deps[nid] = (inst,)
                iid = id(inst)
                if iid not in nodes:
                    nodes[iid] = inst
                    deps[iid] = tuple(inst.input_bits)
                    stack.extend(inst.input_bits)
            else:
                inc = src.incoming
                deps[nid] = inc
                if inc:
                    stack.extend(inc)

    # ---------------- phase 2: depths via topological order ----------------

    def _assign_depths(self) -> None:
        nodes, deps = self.nodes, self.deps
        depth, const_val, folded_bias = self.depth, self.const_val, self.folded_bias
        input_pos = self.input_pos

        consumers: dict[int, list[int]] = {nid: [] for nid in nodes}
        indeg: dict[int, int] = {}
        for nid, dep_tuple in deps.items():
            indeg[nid] = len(dep_tuple)
            for d in dep_tuple:
                consumers[id(d)].append(nid)

        queue: deque[int] = deque(nid for nid, n in indeg.items() if n == 0)
        n_processed = 0
        while queue:
            nid = queue.popleft()
            n_processed += 1
            node = nodes[nid]
            if isinstance(node, StampInstance):
                max_d = -1
                for d in node.input_bits:
                    did = id(d)
                    if did not in const_val:
                        dd = depth[did]
                        if dd > max_d:
                            max_d = dd
                if max_d < 0:
                    const_val[nid] = 1  # fully-const instance marker
                else:
                    depth[nid] = max_d + 1  # base level of the instance block
            elif nid in input_pos:
                depth[nid] = 0
            else:
                src = node.source
                if isinstance(src, StampOutput):
                    iid = id(src.instance)
                    if iid in const_val:
                        const_val[nid] = int(node.activation)
                    else:
                        t_levels = src.instance.template.arrays.n_gate_levels
                        depth[nid] = depth[iid] + t_levels - 1
                else:
                    bias = src.bias
                    max_d = -1
                    for d, w in zip(src.incoming, src.weights):
                        did = id(d)
                        v = const_val.get(did)
                        if v is not None:
                            bias += v * w
                        else:
                            dd = depth[did]
                            if dd > max_d:
                                max_d = dd
                    if max_d < 0:
                        const_val[nid] = 1 if bias >= 0 else 0
                    else:
                        depth[nid] = max_d + 1
                        if bias != src.bias:
                            folded_bias[nid] = int(bias)
                        elif not isinstance(bias, int):
                            folded_bias[nid] = int(bias)
            for c in consumers[nid]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if n_processed != len(nodes):
            raise ValueError("Cycle detected in signal graph")

    # ---------------- phase 3: layout and assembly ----------------

    def build(self) -> LeveledArrays:
        self._discover()
        self._assign_depths()
        nodes, deps = self.nodes, self.deps
        depth, const_val = self.depth, self.const_val
        input_pos = self.input_pos

        out_ids = [id(b) for b in self.outputs]
        out_depths = [depth.get(oid) for oid in out_ids]
        live_out_depths = [d for d in out_depths if d is not None]
        max_prod = max(live_out_depths, default=0)

        # instances and their block extents
        instances = [
            (nid, node)
            for nid, node in nodes.items()
            if isinstance(node, StampInstance) and nid not in const_val
        ]
        instances.sort(key=lambda kv: kv[1].uid)
        inst_top = {
            nid: depth[nid] + inst.template.arrays.n_gate_levels - 1
            for nid, inst in instances
        }

        # The last level must contain exactly the outputs. Reuse the top
        # producer row when possible; otherwise add an explicit output row.
        needs_extra = any(top >= max_prod for top in inst_top.values())
        n_levels = max_prod + 1 if needs_extra else max(max_prod, 1)

        # how long each producer must stay available (its column at level d
        # is consumed by gates at level d+1)
        need_until: dict[int, int] = {}
        for nid, dep_tuple in deps.items():
            if nid in const_val:
                continue
            d_consume = depth[nid]  # gates and instance bases consume at their level
            for dep in dep_tuple:
                did = id(dep)
                if did in const_val or isinstance(nodes[did], StampInstance):
                    continue  # a stamp-output's link to its instance is not a wire
                prev = need_until.get(did, -1)
                if d_consume - 1 > prev:
                    need_until[did] = d_consume - 1
        for oid, od in zip(out_ids, out_depths):
            if od is None:
                continue  # const output, materialized separately
            until = n_levels if od < n_levels else od
            if until > need_until.get(oid, -1):
                need_until[oid] = until

        # ordinary nodes per level (instance internals live in blocks instead)
        level_gates: list[list[int]] = [[] for _ in range(n_levels + 1)]
        for nid, node in nodes.items():
            if nid in const_val or isinstance(node, StampInstance):
                continue
            if nid in input_pos:
                continue
            if isinstance(node.source, StampOutput):
                continue  # lives inside its instance's block
            level_gates[depth[nid]].append(nid)
        for lvl in level_gates:
            lvl.sort(key=lambda nid: nodes[nid].uid)

        # copies: producer nid needs a copy at each level depth+1..need_until
        copies_at: list[list[int]] = [[] for _ in range(n_levels + 1)]
        for nid, until in need_until.items():
            for lv in range(depth[nid] + 1, until + 1):
                copies_at[lv].append(nid)
        for lvl in copies_at:
            lvl.sort(key=lambda nid: nodes[nid].uid)

        # column assignment; col_at[(nid, level)] -> column of nid at level
        col_at: dict[tuple[int, int], int] = {}
        for nid, pos in input_pos.items():
            col_at[(nid, 0)] = pos
        level_sizes = [len(self.inputs)]
        inst_col_offsets: dict[tuple[int, int], int] = {}  # (inst id, level) -> col
        inst_by_level: list[list[int]] = [[] for _ in range(n_levels + 1)]
        for nid, inst in instances:
            base = depth[nid]
            for lv in range(base, inst_top[nid] + 1):
                inst_by_level[lv].append(nid)

        for lv in range(1, n_levels + 1):
            col = 0
            if lv == n_levels:
                # final level: exactly the outputs, in order
                for oid, od in zip(out_ids, out_depths):
                    col_at[(oid, lv)] = col
                    col += 1
                level_sizes.append(col)
                continue
            for nid in level_gates[lv]:
                col_at[(nid, lv)] = col
                col += 1
            for nid in copies_at[lv]:
                col_at[(nid, lv)] = col
                col += 1
            for iid in inst_by_level[lv]:
                inst = nodes[iid]
                t = lv - depth[iid]  # 0-based template gate level index
                inst_col_offsets[(iid, lv)] = col
                col += inst.template.arrays.level_sizes[t + 1]
            level_sizes.append(col)

        # stamp-output bits resolve to columns inside their instance's block
        for nid, node in nodes.items():
            if nid in const_val or isinstance(node, StampInstance):
                continue
            if nid in input_pos:
                continue
            src = node.source
            if isinstance(src, StampOutput):
                iid = id(src.instance)
                top = inst_top[iid]
                if (iid, top) in inst_col_offsets:
                    col_at[(nid, top)] = inst_col_offsets[(iid, top)] + src.slot

        return self._emit(
            n_levels,
            level_sizes,
            level_gates,
            copies_at,
            inst_by_level,
            inst_col_offsets,
            col_at,
            out_ids,
            out_depths,
            needs_extra,
        )

    def _emit(
        self,
        n_levels: int,
        level_sizes: list[int],
        level_gates: list[list[int]],
        copies_at: list[list[int]],
        inst_by_level: list[list[int]],
        inst_col_offsets: dict[tuple[int, int], int],
        col_at: dict[tuple[int, int], int],
        out_ids: list[int],
        out_depths: list[int | None],
        needs_extra: bool,
    ) -> LeveledArrays:
        nodes, depth, const_val = self.nodes, self.depth, self.const_val
        edge_rows: list[np.ndarray] = []
        edge_cols: list[np.ndarray] = []
        edge_weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []

        for lv in range(1, n_levels + 1):
            rows: list[int] = []
            cols: list[int] = []
            ws: list[int] = []
            bias_list: list[int] = []
            row_chunks: list[np.ndarray] = []
            col_chunks: list[np.ndarray] = []
            w_chunks: list[np.ndarray] = []
            bias_chunks: list[np.ndarray] = []

            def flush_lists() -> None:
                if rows:
                    row_chunks.append(np.asarray(rows, dtype=np.int64))
                    col_chunks.append(np.asarray(cols, dtype=np.int64))
                    w_chunks.append(np.asarray(ws, dtype=np.int64))
                    rows.clear()
                    cols.clear()
                    ws.clear()
                if bias_list:
                    bias_chunks.append(np.asarray(bias_list, dtype=np.int64))
                    bias_list.clear()

            if lv == n_levels:
                # explicit output row
                for row, (oid, od) in enumerate(zip(out_ids, out_depths)):
                    if od is None:  # constant output: bias-only gate
                        bias_list.append(0 if const_val[oid] else -1)
                        continue
                    if od == lv:
                        # gate that naturally sits on the top row
                        self._emit_gate(oid, row, lv, col_at, rows, cols, ws, bias_list)
                    else:
                        rows.append(row)
                        cols.append(col_at[(oid, lv - 1)])
                        ws.append(1)
                        bias_list.append(-1)
                flush_lists()
            else:
                row = 0
                for nid in level_gates[lv]:
                    self._emit_gate(nid, row, lv, col_at, rows, cols, ws, bias_list)
                    row += 1
                for nid in copies_at[lv]:
                    rows.append(row)
                    cols.append(col_at[(nid, lv - 1)])
                    ws.append(1)
                    bias_list.append(-1)
                    row += 1
                flush_lists()
                for iid in inst_by_level[lv]:
                    inst = nodes[iid]
                    arrays = inst.template.arrays
                    t = lv - depth[iid]  # 0-based index into template arrays
                    row_offset = inst_col_offsets[(iid, lv)]
                    t_rows = arrays.edge_rows[t]
                    t_cols = arrays.edge_cols[t]
                    t_ws = arrays.edge_weights[t]
                    t_bias = arrays.biases[t]
                    if t == 0:
                        # bind template input slots to concrete columns
                        binding = np.empty(arrays.level_sizes[0], dtype=np.int64)
                        const_mask = np.zeros(arrays.level_sizes[0], dtype=bool)
                        const_vals = np.zeros(arrays.level_sizes[0], dtype=np.int64)
                        for slot, b in enumerate(inst.input_bits):
                            bid = id(b)
                            v = const_val.get(bid)
                            if v is not None:
                                const_mask[slot] = True
                                const_vals[slot] = v
                                binding[slot] = 0
                            else:
                                binding[slot] = col_at[(bid, lv - 1)]
                        if const_mask.any():
                            edge_const = const_mask[t_cols]
                            live = ~edge_const
                            delta = np.bincount(
                                t_rows[edge_const],
                                weights=(
                                    t_ws[edge_const] * const_vals[t_cols[edge_const]]
                                ),
                                minlength=len(t_bias),
                            ).astype(np.int64)
                            row_chunks.append(t_rows[live] + row_offset)
                            col_chunks.append(binding[t_cols[live]])
                            w_chunks.append(t_ws[live])
                            bias_chunks.append(t_bias + delta)
                        else:
                            row_chunks.append(t_rows + row_offset)
                            col_chunks.append(binding[t_cols])
                            w_chunks.append(t_ws)
                            bias_chunks.append(t_bias)
                    else:
                        col_offset = inst_col_offsets[(iid, lv - 1)]
                        row_chunks.append(t_rows + row_offset)
                        col_chunks.append(t_cols + col_offset)
                        w_chunks.append(t_ws)
                        bias_chunks.append(t_bias)

            empty = np.empty(0, dtype=np.int64)
            edge_rows.append(np.concatenate(row_chunks) if row_chunks else empty)
            edge_cols.append(np.concatenate(col_chunks) if col_chunks else empty)
            edge_weights.append(np.concatenate(w_chunks) if w_chunks else empty)
            biases.append(np.concatenate(bias_chunks) if bias_chunks else empty)
            assert len(biases[-1]) == level_sizes[lv], (
                f"level {lv}: {len(biases[-1])} biases for {level_sizes[lv]} nodes"
            )

        return LeveledArrays(
            level_sizes=level_sizes,
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            edge_weights=edge_weights,
            biases=biases,
        )

    def _emit_gate(
        self,
        nid: int,
        row: int,
        lv: int,
        col_at: dict[tuple[int, int], int],
        rows: list[int],
        cols: list[int],
        ws: list[int],
        bias_list: list[int],
    ) -> None:
        node = self.nodes[nid]
        src = node.source
        const_val = self.const_val
        bias = self.folded_bias.get(nid)
        if bias is None:
            bias = int(src.bias)
        for d, w in zip(src.incoming, src.weights):
            did = id(d)
            if did in const_val:
                continue  # already folded into bias
            rows.append(row)
            cols.append(col_at[(did, lv - 1)])
            ws.append(int(w))
        bias_list.append(bias)

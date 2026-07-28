"""Direct vectorized compilation of Keccak into a leveled circuit.

`compile_keccak(k)` builds the full digest circuit as `LeveledArrays`
without executing the bit-level Keccak code at all: the round template
(`keccak_template.build_round_template`) is stacked n times with pure
array arithmetic. Two observations make this exact:

- A round's theta counters consume precisely the previous round's chi
  finals, and both are laid out in x-major lane order, so the inter-round
  wiring is the identity and the template arrays can be reused verbatim.
- iota only negates lane (0,0) bits selected by the round constant. A
  negation feeding a threshold gate can be fused into that gate by
  flipping the edge weight and adding it to the bias (the trick used in
  examples/other/keccak_fused.py), so iota becomes a per-round weight/bias
  adjustment of the next level instead of extra gates and copy levels.

The compiled circuit takes the padded message (msg_len bits) as inputs
and produces the digest (d bits); the suffix and capacity bits are folded
into round-1 biases. Functional equivalence with Keccak.digest and with
hashlib is covered by tests.
"""

from functools import lru_cache

import numpy as np

from reifier.fast.leveler import LeveledArrays
from .keccak import Keccak, state_to_lanes, lanes_to_state
from .keccak_template import build_round_template


@lru_cache(maxsize=None)
def _lane_state_maps(w: int) -> tuple[np.ndarray, np.ndarray]:
    """Index maps between state order and x-major lane-flat order.

    S[lane_flat] = state index feeding that lane position (via state_to_lanes)
    P[state_index] = lane-flat position that lands there (via lanes_to_state)
    Both are computed by running the real reshaping code on index values, so
    they cannot drift from the bit-level implementation.
    """
    b = 25 * w
    lanes = state_to_lanes(list(range(b)))  # type: ignore[arg-type]
    S = np.array(
        [lanes[x][y][z] for x in range(5) for y in range(5) for z in range(w)],
        dtype=np.int64,
    )
    index_lanes = [
        [[(x * 5 + y) * w + z for z in range(w)] for y in range(5)] for x in range(5)
    ]
    P = np.array(lanes_to_state(index_lanes), dtype=np.int64)  # type: ignore[arg-type]
    return S, P


def compile_keccak(k: Keccak) -> LeveledArrays:
    """Compile k.digest into a leveled circuit: msg_len inputs -> d outputs."""
    if k.n < 1:
        raise ValueError("compile_keccak requires at least one round")
    w, b, msg_len, d = k.w, k.b, k.msg_len, k.d
    tpl = build_round_template(w).arrays
    S, P = _lane_state_maps(w)

    # initial state values: -1 marks live message inputs, 0/1 are constants
    state_val = np.full(b, -1, dtype=np.int64)
    for i, ch in enumerate(format(k.suffix, "08b")):  # as in msg_to_state
        state_val[msg_len + i] = int(ch)
    state_val[msg_len + k.suffix_len :] = 0

    # lane-flat positions negated by iota in each round: lane (0,0) is
    # flat 0..w-1, so the flat position of a flip at z is just z
    flips = [
        np.array([z for z, ch in enumerate(rc) if ch == "1"], dtype=np.int64)
        for rc in k.get_round_constants()
    ]

    l1_rows, l1_cols = tpl.edge_rows[0], tpl.edge_cols[0]
    l1_ws, l1_bias = tpl.edge_weights[0], tpl.biases[0]

    # edge ids of level-1 edges consuming each lane position, grouped
    order = np.argsort(l1_cols, kind="stable")
    starts = np.searchsorted(l1_cols[order], np.arange(b))
    ends = np.append(starts[1:], len(l1_cols))

    def edges_consuming(positions: np.ndarray) -> np.ndarray:
        if len(positions) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([order[starts[p] : ends[p]] for p in positions])

    level_sizes = [msg_len]
    edge_rows: list[np.ndarray] = []
    edge_cols: list[np.ndarray] = []
    edge_weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for r in range(k.n):
        if r == 0:
            # bind level-1 edges to message inputs; fold constants into biases
            state_idx = S[l1_cols]
            vals = state_val[state_idx]
            live = vals < 0
            const = ~live
            delta = np.bincount(
                l1_rows[const],
                weights=l1_ws[const] * vals[const],
                minlength=len(l1_bias),
            ).astype(np.int64)
            edge_rows.append(l1_rows[live])
            edge_cols.append(state_idx[live])  # state index == input column
            edge_weights.append(l1_ws[live])
            biases.append(l1_bias + delta)
        else:
            # previous level is this round's input in the same order; fuse
            # the previous round's iota flips: w -> -w, bias += w
            flip = flips[r - 1]
            edge_rows.append(l1_rows)
            edge_cols.append(l1_cols)
            if len(flip):
                eids = edges_consuming(flip)
                ws = l1_ws.copy()
                delta = np.zeros(len(l1_bias), dtype=np.int64)
                np.add.at(delta, l1_rows[eids], ws[eids])
                ws[eids] = -ws[eids]
                edge_weights.append(ws)
                biases.append(l1_bias + delta)
            else:
                edge_weights.append(l1_ws)
                biases.append(l1_bias)
        level_sizes.append(tpl.level_sizes[1])
        for t in range(1, tpl.n_gate_levels):  # inner levels shared verbatim
            edge_rows.append(tpl.edge_rows[t])
            edge_cols.append(tpl.edge_cols[t])
            edge_weights.append(tpl.edge_weights[t])
            biases.append(tpl.biases[t])
            level_sizes.append(tpl.level_sizes[t + 1])

    # output row: digest bits in state order, fusing the last round's iota
    out_cols = P[:d]
    flipped = np.isin(out_cols, flips[k.n - 1])
    edge_rows.append(np.arange(d, dtype=np.int64))
    edge_cols.append(out_cols)
    edge_weights.append(np.where(flipped, -1, 1).astype(np.int64))
    biases.append(np.where(flipped, 0, -1).astype(np.int64))
    level_sizes.append(d)

    return LeveledArrays(
        level_sizes=level_sizes,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        edge_weights=edge_weights,
        biases=biases,
    )

"""Vectorized construction of the Keccak round template.

Builds, with numpy index arithmetic, the exact `Template` that
`reifier.fast.stamp` would produce by tracing `theta_rho_pi_chi` — so that
`Keccak(stamp=True)` starts from a warm template cache without paying the
one-time tracing cost. Equivalence with the traced template is covered by
tests (structural equality at small word sizes, functional at w=64).

Layout produced by tracing theta_rho_pi_chi (see keccak.theta / .chi):
- level 0: the 1600 lane bits, in lanes[x][y][z] flatten order (x-major)
- level 1: theta xor counters, 11 per position, position-major
- level 2: theta xor finals, one per position (x-major)
- level 3: chi inhib gates (chi iterates y-major) then leveler copies of
  the 1600 theta finals (ordered by creation, x-major)
- level 4: chi xor counters, 2 per position, y-major
- level 5: chi xor finals, emitted in output slot order (x-major)
"""

from functools import lru_cache

import numpy as np

from reifier.fast.leveler import LeveledArrays
from reifier.fast.stamp import Template, _BIT_LEAF, _signature, stamped
from .keccak import Lanes, rho_pi, theta_rho_pi_chi


def _positions(w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y, z arrays over all 1600/w positions in x-major flatten order."""
    x, y, z = np.meshgrid(np.arange(5), np.arange(5), np.arange(w), indexing="ij")
    return x.ravel(), y.ravel(), z.ravel()


def _flat(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: int) -> np.ndarray:
    """Position index in lanes[x][y][z] flatten order."""
    return (x * 5 + y) * w + z


def _rho_pi_sources(w: int) -> np.ndarray:
    """source[pos] = which input position lands at `pos` after rho_pi."""
    index_lanes: Lanes = [
        [[(x * 5 + y) * w + z for z in range(w)] for y in range(5)] for x in range(5)
    ]  # type: ignore[misc]  # rho_pi only reorders, so ints work in place of Bits
    permuted = rho_pi(index_lanes)
    return np.array(
        [permuted[x][y][z] for x in range(5) for y in range(5) for z in range(w)],
        dtype=np.int64,
    )


@lru_cache(maxsize=None)
def build_round_template(w: int) -> Template:
    """Build the theta_rho_pi_chi template for word size w without tracing.
    Cached per word size; callers share the arrays and must not mutate them."""
    n = 25 * w  # number of lane positions
    x, y, z = _positions(w)

    # --- level 1: theta counters (11 per position, thresholds 1..11) ---
    cols_in = np.empty((n, 11), dtype=np.int64)
    cols_in[:, 0] = _flat(x, y, z, w)
    for y2 in range(5):
        cols_in[:, 1 + y2] = _flat((x + 4) % 5, np.full_like(x, y2), z, w)
        cols_in[:, 6 + y2] = _flat((x + 1) % 5, np.full_like(x, y2), (z + 1) % w, w)
    l1_rows = np.repeat(np.arange(n * 11, dtype=np.int64), 11)
    l1_cols = np.repeat(cols_in, 11, axis=0).ravel()
    l1_ws = np.ones(n * 11 * 11, dtype=np.int64)
    l1_bias = np.tile(-(np.arange(11, dtype=np.int64) + 1), n)

    # --- level 2: theta finals (one per position, alternating weights) ---
    l2_rows = np.repeat(np.arange(n, dtype=np.int64), 11)
    l2_cols = (
        np.arange(n, dtype=np.int64)[:, None] * 11 + np.arange(11, dtype=np.int64)
    ).ravel()
    l2_ws = np.tile((-1) ** np.arange(11, dtype=np.int64), n)
    l2_bias = np.full(n, -1, dtype=np.int64)

    # --- level 3: chi inhibs (y-major) + copies of all theta finals ---
    src = _rho_pi_sources(w)  # lane bit (x,y,z) after rho_pi = theta final src[pos]
    hy, hx, hz = np.meshgrid(np.arange(5), np.arange(5), np.arange(w), indexing="ij")
    hy, hx, hz = hy.ravel(), hx.ravel(), hz.ravel()  # chi creation order: y-major
    h_pos = _flat(hx, hy, hz, w)  # position of chi gate h in x-major terms
    inhib_cols = np.stack(
        [src[_flat((hx + 1) % 5, hy, hz, w)], src[_flat((hx + 2) % 5, hy, hz, w)]],
        axis=1,
    )
    inhib_ws = np.tile(np.array([-1, 1], dtype=np.int64), (n, 1))
    copy_cols = np.arange(n, dtype=np.int64)  # copy of theta final p, in p order
    l3_rows = np.concatenate(
        [np.repeat(np.arange(n, dtype=np.int64), 2), np.arange(n, dtype=np.int64) + n]
    )
    l3_cols = np.concatenate([inhib_cols.ravel(), copy_cols])
    l3_ws = np.concatenate([inhib_ws.ravel(), np.ones(n, dtype=np.int64)])
    l3_bias = np.concatenate(
        [np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)]
    )

    # --- level 4: chi xor counters (2 per position, y-major) ---
    # counter inputs: [copy of lane bit (x,y,z), inhib gate h]
    lane_copy_col = n + src[h_pos]  # copies sit after the inhibs at level 3
    ctr_cols = np.stack([lane_copy_col, np.arange(n, dtype=np.int64)], axis=1)
    l4_rows = np.repeat(np.arange(2 * n, dtype=np.int64), 2)
    l4_cols = np.repeat(ctr_cols, 2, axis=0).ravel()
    l4_ws = np.ones(4 * n, dtype=np.int64)
    l4_bias = np.tile(np.array([-1, -2], dtype=np.int64), n)

    # --- level 5: chi finals, in output slot order (x-major) ---
    h_of_pos = np.empty(n, dtype=np.int64)
    h_of_pos[h_pos] = np.arange(n, dtype=np.int64)  # y-major gate for position p
    l5_rows = np.repeat(np.arange(n, dtype=np.int64), 2)
    l5_cols = np.stack([h_of_pos * 2, h_of_pos * 2 + 1], axis=1).ravel()
    l5_ws = np.tile(np.array([1, -1], dtype=np.int64), n)
    l5_bias = np.full(n, -1, dtype=np.int64)

    arrays = LeveledArrays(
        level_sizes=[n, 11 * n, n, 2 * n, 2 * n, n],
        edge_rows=[l1_rows, l2_rows, l3_rows, l4_rows, l5_rows],
        edge_cols=[l1_cols, l2_cols, l3_cols, l4_cols, l5_cols],
        edge_weights=[l1_ws, l2_ws, l3_ws, l4_ws, l5_ws],
        biases=[l1_bias, l2_bias, l3_bias, l4_bias, l5_bias],
    )
    skeleton = [[[_BIT_LEAF for _ in range(w)] for _ in range(5)] for _ in range(5)]
    return Template(
        arrays=arrays,
        arg_positions=list(range(n)),
        out_kinds=[("gated", p) for p in range(n)],
        skeleton=skeleton,
    )


@lru_cache(maxsize=None)
def _template_key(w: int):
    """The Stamped cache key for a theta_rho_pi_chi call at word size w.
    Lane bits are always distinct objects, so the alias pattern is 0..n-1."""
    lane_sig = ("list", (_BIT_LEAF,) * w)
    lanes_sig = ("list", (("list", (lane_sig,) * 5),) * 5)
    signature = ("tuple", (("tuple", (lanes_sig,)), ("tuple", ())))
    return (signature, tuple(range(25 * w)))


def prime_round_template(w: int) -> None:
    """Seed the stamped(theta_rho_pi_chi) cache for word size w."""
    wrapper = stamped(theta_rho_pi_chi)
    key = _template_key(w)
    if key not in wrapper.cache:
        wrapper.cache[key] = build_round_template(w)


def _traced_key_check(w: int) -> bool:
    """Test helper: does _template_key match the key built from real arguments?"""
    from reifier.neurons.core import const
    from reifier.fast.leveler import flatten_bits
    from reifier.fast.stamp import _alias_pattern

    bits = const("0" * (25 * w))
    lanes = [
        [[bits[(x * 5 + y) * w + z] for z in range(w)] for y in range(5)]
        for x in range(5)
    ]
    key = (_signature(((lanes,), ())), _alias_pattern(flatten_bits(((lanes,), {}))))
    return key == _template_key(w)

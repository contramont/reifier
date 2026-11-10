"""Outside the core, on the same trace and grid: folding, tiles, Keccak, the adder and Gemma."""

import hashlib
import zlib

import numpy as np
import pytest

from reifier import (
    add,
    check_grid,
    check_net,
    check_trace,
    const,
    emit,
    fold,
    gemma,
    held,
    layout,
    threshold_gate,
    tiles,
    trace,
    xor,
)
from reifier.programs.adder import CODES, WINDOW, next_token
from reifier.programs.keccak import keccak_p, turboshake128

MSG = bytes(range(0, 256, 16))
X = [byte >> i & 1 for byte in MSG for i in range(8)]


def first16(t):  # the first 16 bits of t on MSG, as a number
    return sum(round(v) << i for i, v in enumerate(t(X)[:16]))


def test_folding_is_exact_and_drops_what_constants_decide():
    def program(x):
        return [
            threshold_gate([x[0], const("0")[0]], [1, 1], 1),
            xor([x[0], const("1")[0], x[1]]),
            x[1],
        ]

    t = trace(program, 2)
    f = fold(t)
    assert check_trace(f, t) == 4 and [f(x) for x in ("00", "01", "10", "11")] == [
        t(x) for x in ("00", "01", "10", "11")
    ]
    # [x0 + 0 >= 1] is x0 itself, and the xor reads two bits, not three
    assert f.outputs == ["x0", "xor/odd", "x1"] and len(f.mlps) == 3 < len(t.mlps) == 7
    assert [m.inputs for m in f.mlps][:2] == [("x0", "x1")] * 2  # atleast1 is always on, and gone
    assert check_net(emit(f, layout(f), "geglu", skip=True, norm=True)) < 1e-9


def test_keccak_is_turboshake():  # with 24 rounds and d = 0x1F, TurboSHAKE128 is SHAKE128
    shake = int.from_bytes(hashlib.shake_128(MSG).digest(2), "little")
    assert first16(trace(lambda msg: turboshake128(msg, 24), 128)) == shake
    assert first16(trace(lambda msg: turboshake128(msg, 6), 128)) == 0x58CC  # the last 6 rounds


def test_folding_keeps_turboshake():  # its padding is constants: 6 rounds fold to fewer MLPs
    t = trace(lambda msg: turboshake128(msg, 6)[:16], 128)
    f = fold(t)
    assert first16(f) == 0x58CC and check_trace(f, t) == 256
    assert (len(t.mlps), len(f.mlps)) == (17368, 14104)
    assert (
        f.mlps[-1].name == "round23/chi[0,0,15]" and f.mlps[-1].hidden[0] == "round23/chi[0,0,15]#0"
    )
    assert [len(mlps) for mlps in check_grid(f, layout(f))][1:4] == [1344, 512, 192]


def test_keccak_goes_through_every_stage():  # Keccak-p[200, 2], with skip connections and norms
    t = trace(lambda state: keccak_p(state, 2), 200)
    g = held(layout(t))
    assert (len(t.mlps), len(g.rows) - 1) == (880, 6)
    assert check_net(emit(t, g, "geglu", skip=True, norm=True)) < 1e-9


def in_rectangles(t, g):  # is each xor of t a rectangle of g that holds only what that xor makes?
    first = {}
    for d, row in enumerate(g.rows):
        first |= {s: first.get(s, d) for s in row if s}
    for scope in {m.name.rsplit("/", 1)[0] for m in t.mlps if "/xor" in m.name}:
        own = {y for m in t.mlps if m.name.startswith(scope + "/") for y in m.outputs}
        depths = range(min(first[y] for y in own), max(first[y] for y in own) + 1)
        at = [f for d in depths for f, s in enumerate(g.rows[d]) if s in own]
        if not {g.rows[d][f] for d in depths for f in range(min(at), max(at) + 1)} <= own | {None}:
            return False
    return True


def test_tiles_keep_each_circuit_in_a_rectangle():
    t = trace(add, 8, 8)
    g = tiles(t)
    assert check_grid(t, g) and in_rectangles(t, g) and not in_rectangles(t, layout(t))
    assert len(g.rows) == len(layout(t).rows) and len(g.rows[0]) > len(layout(t).rows[0])


IDS = "236771 236770 236778 236800 236812 236810 236825 236832 236828 236819 236862 236784"
RAN = {"adder": 925690047, "keccak": 112776309}  # checksums of the weights that ran on Gemma


@pytest.mark.skipif(not gemma.CACHE_FILE.exists(), reason="needs the shapes of the checkpoint")
@pytest.mark.parametrize("name", RAN)
def test_gemma_weights_are_those_that_ran_on_the_model(name):
    if name == "adder":  # IDS: the tokens of the characters of SYMBOLS, as examples/ looks them up
        t, codes = (
            trace(next_token, len(CODES) * WINDOW),
            dict(zip(map(int, IDS.split()), CODES.values())),
        )
    else:
        t = fold(trace(lambda msg: turboshake128(msg, 6)[:16], 128))
        codes = {0x10000 + n: [n >> i & 1 for i in range(16)] for n in range(2**16)}
    shapes, config = gemma.checkpoint_metadata()
    theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]
    w, g, hidden = gemma.build_weights(t, codes, shapes, config["bos_token_id"], theta)
    checksum = 0
    for k in sorted(
        w
    ):  # each tensor that is set: its name, its rows that are not zero, their values
        rows = np.flatnonzero(w[k].reshape(len(w[k]), -1).any(1))
        checksum = zlib.crc32(
            np.ascontiguousarray(w[k][rows]), zlib.crc32(k.encode() + rows.tobytes(), checksum)
        )
    assert checksum == RAN[name], "the weights changed: run the checkpoint on the model again"
    assert check_grid(t, g) and all(len(set(names)) == len(names) for names in hidden)

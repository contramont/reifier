"""1-round Keccak XOF variants for the residual architecture (compile.residual).

Each variant is (k, depth) -> Program. The round is theta (11-input xor per bit, 6
units), rho-pi (free), chi with iota (one unit per bit), as in variants.glu_chi_iota.
- inplace=True writes theta and chi as updates in place (glu inplace), so the residual
  stream needs no clearing units.
- unrolled: every XOF step has its own layers.
- tied: one block of layers (a step) is applied depth times: the state and a shift
  register of the earlier digests are the loop state (or, with taps, the digests are
  read out after each step and there is no shift register).
The keccak module is not patched, so its xof stays the reference."""

from reifier.compile.residual import compile_recurrent, compile_unrolled
from reifier.examples import keccak as K
from reifier.neurons.core import Unit, const, glu
from reifier.neurons.operations import glu_xor

CHI = Unit((2, -1, 1), 0, (-1, 0.5, -0.5), 1.5)  # a ^ (~b & c)
NCHI = Unit((-2, -1, 1), 2, (1, 0.5, -0.5), 0.5)  # ~(a ^ (~b & c))
CHI_UPDATE = [Unit((0, -1, 1), 0, (-2, 0, 0), 1)]  # (1-2a) max(0, c-b), inplace on a
NCHI_UPDATE = [  # (1-2a)(1 - max(0, c-b)), inplace on a
    Unit((0, 0, 0), 1, (-2, 0, 0), 1),
    Unit((0, -1, 1), 0, (2, 0, 0), -1),
]


def theta(lanes, kind: str):
    """kind: "flat" (glu_xor_flat), "compact" (glu_xor) or "update" (in place)"""
    w = len(lanes[0][0])
    out = K.get_empty_lanes(w, lanes[0][0][0])
    for x in range(5):
        for y in range(5):
            for z in range(w):
                bits = [lanes[x][y][z]]
                bits += [lanes[(x + 4) % 5][y2][z] for y2 in range(5)]
                bits += [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)]
                out[x][y][z] = glu_xor(bits, inplace=kind == "update", flat=kind == "flat")
    return out


def theta_split(lanes, inplace: bool):
    """theta in two layers: D[x][z] = C[x-1][z] ^ C[x+1][z-1] as a 10-bit xor, then
    A ^= D as a 2-bit xor (one unit, in place if inplace). Residual layers keep the
    state for free while D is computed"""
    w = len(lanes[0][0])
    dcol = [
        [
            glu_xor([lanes[(x + 4) % 5][y][z] for y in range(5)]
                    + [lanes[(x + 1) % 5][y][(z + 1) % w] for y in range(5)])
            for z in range(w)
        ]
        for x in range(5)
    ]
    out = K.get_empty_lanes(w, lanes[0][0][0])
    for x in range(5):
        for y in range(5):
            for z in range(w):
                out[x][y][z] = glu_xor([lanes[x][y][z], dcol[x][z]], inplace=inplace)
    return out


def chi_iota(lanes, rc: str, inplace: bool):
    w = len(lanes[0][0])
    out = K.get_empty_lanes(w, lanes[0][0][0])
    for y in range(5):
        for x in range(5):
            for z in range(w):
                abc = [lanes[x][y][z], lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]]
                flip = x == 0 and y == 0 and rc[z] == "1"
                if inplace:
                    out[x][y][z] = glu(abc, NCHI_UPDATE if flip else CHI_UPDATE, inplace=0)
                else:
                    out[x][y][z] = glu(abc, [NCHI if flip else CHI])
    return out


def hash_state(k: K.Keccak, state, theta_kind: str, chi_inplace: bool):
    lanes = K.state_to_lanes(state)
    for rc in k.get_round_constants():
        if theta_kind.startswith("split"):
            lanes = theta_split(lanes, inplace=theta_kind == "split_update")
        else:
            lanes = theta(lanes, theta_kind)
        lanes = K.rho_pi(lanes)
        lanes = chi_iota(lanes, rc, chi_inplace)
    return K.lanes_to_state(lanes)


def initial_state(k: K.Keccak, msg):
    return k.msg_to_state(k.bitlist_to_msg(msg))


def unrolled(k: K.Keccak, depth: int, theta_kind="flat", chi_inplace=True, **kw):
    def xof(msg):
        state = initial_state(k, msg)
        digests = []
        for _ in range(depth):
            state = hash_state(k, state, theta_kind, chi_inplace)
            digests += state[: k.d]
        return digests

    return compile_unrolled(xof, k.msg_len, **kw)


def tied(k: K.Keccak, depth: int, taps=False, theta_kind="flat", chi_inplace=True, **kw):
    """The loop state is the Keccak state, and without taps a shift register of the
    depth-1 earlier digests: each step moves the digest of its input state into it"""
    b, d = k.b, k.d
    n_reg = 0 if taps else depth - 1

    def init(msg):
        return initial_state(k, msg) + const("0" * (d * n_reg))

    def step(x):
        state, reg = x[:b], x[b:]
        new = hash_state(k, state, theta_kind, chi_inplace)
        return new + (state[:d] + reg[: d * (n_reg - 1)] if n_reg else [])

    def readout(states):
        if taps:
            return [bit for st in states for bit in st[:d]]
        last = states[-1]
        regs = [last[b + d * i : b + d * (i + 1)] for i in range(n_reg)]
        return [bit for r in reversed(regs) for bit in r] + last[:d]

    return compile_recurrent(init, step, readout, k.msg_len, depth, **kw)


# unrolled: every XOF step has its own layers
def res_flat(k, depth):  # flat theta, chi in place
    return unrolled(k, depth)


def res_compact(k, depth):  # glu_chi_iota's theta
    return unrolled(k, depth, theta_kind="compact")


def res_compact_relu(k, depth):
    return unrolled(k, depth, theta_kind="compact", clear="relu")


def res_update(k, depth):  # theta in place: fewer units, but amplifies errors
    return unrolled(k, depth, theta_kind="update")


def res_plain(k, depth):  # glu_chi_iota's circuit as is
    return unrolled(k, depth, theta_kind="compact", chi_inplace=False)


# tied: one block of layers for all XOF steps
def tied_regs(k, depth):
    return tied(k, depth)


def tied_taps(k, depth):
    return tied(k, depth, taps=True)


def tied_compact(k, depth):
    return tied(k, depth, theta_kind="compact")


def tied_compact_taps(k, depth):
    return tied(k, depth, taps=True, theta_kind="compact")


def tied_update(k, depth):
    return tied(k, depth, theta_kind="update")


# theta split into D (10-bit xor per column pair) and A ^= D: 3 layers per round
def res_split(k, depth):
    return unrolled(k, depth, theta_kind="split_update")


def tied_split(k, depth):
    return tied(k, depth, theta_kind="split")


def tied_split_taps(k, depth):
    return tied(k, depth, taps=True, theta_kind="split")


# gated residual (alpha * x + SwiGLU(x), alpha a fixed 0/1 mask): writes need no clearing
def gated_compact_taps(k, depth):
    return tied(k, depth, taps=True, theta_kind="compact", gated=True)


def gated_split_taps(k, depth):
    return tied(k, depth, taps=True, theta_kind="split", gated=True)


def gated_split(k, depth):
    return tied(k, depth, theta_kind="split", gated=True)


def gres_compact(k, depth):  # unrolled, gated residual
    return unrolled(k, depth, theta_kind="compact", gated=True)


def gres_split(k, depth):
    return unrolled(k, depth, theta_kind="split_update", gated=True)

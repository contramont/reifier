"""XOF-structured 1-round Keccak circuits (avenue xof-structure).

The circuit is built layer by layer, one traced function call per SwiGLU layer, so the
compiler places every gated unit exactly one level above its inputs and adds no copies:
  theta_1, chi_1, theta_2, chi_2, ..., theta_n, chi_n   (depth 2n, no input/output layers)
Keccak-specific simplifications:
  - constants (suffix, capacity) are Python-level literals folded into the units, so the
    first layer reads the message directly and theta_1 xors only live bits; equal xors
    (e.g. theta of two zero lanes of a column, both = D[x]) are built once;
  - negations (iota, suffix ones) are literal flags folded into consumers, never units;
  - the last round computes only the chi outputs of the digest and the theta outputs
    (5 diagonal lanes, via rho/pi) that feed them;
  - digests of earlier steps are carried as one-unit gated copies, and the last layer
    emits exactly the outputs in order, so no extra output layer is needed.
A literal is an int 0/1 (constant) or a pair (Bit, neg) meaning neg xor Bit.
"""

from math import ceil

import reifier.examples.keccak as K
from reifier.neurons.core import Bit, Unit, glu
from reifier.utils.format import Bits

COPY = Unit((1,), 0, (0,), 1)  # max(0, x) * 1 = x
CHI = Unit((2, -1, 1), 0, (-1, 0.5, -0.5), 1.5)  # a ^ (~b & c)


def lit_value(lit) -> int:
    return lit if isinstance(lit, int) else int(lit[0].activation) ^ int(lit[1])


def fold(lits, units):
    """Substitutes literals into units over len(lits) abstract inputs.
    Returns the distinct live signals and the units on them."""
    sigs: list[Bit] = []
    idx: dict[Bit, int] = {}
    for lit in lits:
        if not isinstance(lit, int) and lit[0] not in idx:
            idx[lit[0]] = len(sigs)
            sigs.append(lit[0])
    out = []
    for u in units:
        gw, vw = [0] * len(sigs), [0] * len(sigs)
        gb, vb = u.bias, u.value_bias
        for lit, g, v in zip(lits, u.weights, u.value_weights):
            if isinstance(lit, int):
                gb, vb = gb + g * lit, vb + v * lit
            else:
                i = idx[lit[0]]
                if lit[1]:  # 1 - x
                    gb, vb = gb + g, vb + v
                    gw[i], vw[i] = gw[i] - g, vw[i] - v
                else:
                    gw[i], vw[i] = gw[i] + g, vw[i] + v
        out.append(Unit(tuple(gw), gb, tuple(vw), vb))
    # drop signals that no unit reads
    keep = [i for i in range(len(sigs)) if any(u.weights[i] or u.value_weights[i] for u in out)]
    if len(keep) < len(sigs):
        sigs = [sigs[i] for i in keep]
        out = [Unit(tuple(u.weights[i] for i in keep), u.bias,
                    tuple(u.value_weights[i] for i in keep), u.value_bias) for u in out]
    return sigs, out


def unit_lit(lits, units):
    """A literal computed by gated units on literals (constant if nothing is live)"""
    sigs, us = fold(lits, units)
    if not sigs:
        total = sum(max(0, u.bias) * u.value_bias for u in us)
        assert total in (0, 1)
        return int(total)
    return (glu(sigs, us), False)


def xor_units(n: int) -> list[Unit]:
    """glu_xor units on n inputs: max(0,s)(2-s) + sum_j 4max(0,s-2j)"""
    units = [Unit((1,) * n, 0, (-1,) * n, 2)]
    units += [Unit((1,) * n, -2 * j, (0,) * n, 4) for j in range(1, ceil(n / 2))]
    return units


class Layer:
    """Builds the gated units of one SwiGLU layer, sharing equal xors and copies"""

    def __init__(self):
        self.memo: dict = {}

    def xor(self, lits):
        flip, cnt = 0, {}
        for lit in lits:
            if isinstance(lit, int):
                flip ^= lit
            else:
                flip ^= int(lit[1])
                cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
        sigs = sorted((s for s, c in cnt.items() if c), key=lambda s: s.uid)
        if not sigs:
            return flip
        key = ("xor", frozenset(sigs))
        if key not in self.memo:
            units = [COPY] if len(sigs) == 1 else xor_units(len(sigs))
            self.memo[key] = glu(sigs, units)
        return (self.memo[key], bool(flip))

    def copy(self, lit):
        """A copy of lit into this layer; the negation flag stays a flag"""
        if isinstance(lit, int):
            return lit
        key = ("copy", lit[0])
        if key not in self.memo:
            self.memo[key] = glu([lit[0]], [COPY])
        return (self.memo[key], lit[1])

    def exact_copy(self, lit) -> Bit:
        """A copy with the negation applied, for outputs"""
        out = unit_lit([lit], [COPY])
        assert not isinstance(out, int), "constant output"
        return out[0]


def keccak_maps(k: K.Keccak):
    """Position bookkeeping, by running the reference permutations on labels"""
    w = k.w
    labels = [[[(x, y, z) for z in range(w)] for y in range(5)] for x in range(5)]
    rp = K.rho_pi(labels)  # rp[X][Y][z] = pre-rho/pi position of post-pi (X, Y, z)
    state_pos = K.lanes_to_state(labels)  # state index -> (x, y, z)
    return rp, state_pos


def initial_state(k: K.Keccak, msg: list[Bit]):
    """Lanes of literals: message bits live, suffix and capacity constant"""
    assert len(msg) == k.msg_len
    sep = [int(c) for c in format(k.suffix, "08b")]
    state = [(b, False) for b in msg] + sep + [0] * k.capacity
    return K.state_to_lanes(state)


def theta_lits(a, x, y, z, w):
    return ([a[x][y][z]] + [a[(x + 4) % 5][y2][z] for y2 in range(5)]
            + [a[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)])


def build(k: K.Keccak, depth: int):
    """Returns the traced function msg -> digests (flattened) of xof(msg, depth, k)"""
    w = k.w
    (rc,) = k.get_round_constants()
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[: k.d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]

    def chi_needs(out_pos):
        """theta positions needed for the chi outputs at out_pos"""
        need = {}
        for (x, y, z) in out_pos:
            for dx in range(3):
                p = rp[(x + dx) % 5][y][z]
                need[p] = None
        return list(need)

    def theta_layer(a, pos, carry):
        L = Layer()
        th = {p: L.xor(theta_lits(a, *p, w)) for p in pos}
        return th, [L.copy(c) for c in carry]

    def chi_layer(th, pos, carry, last):
        L = Layer()
        out = {}
        if last:
            carried = [L.exact_copy(c) for c in carry]
        else:
            carried = [L.copy(c) for c in carry]
        for (x, y, z) in pos:
            b = [th[rp[(x + dx) % 5][y][z]] for dx in range(3)]
            flip = x == 0 and y == 0 and rc[z] == "1"
            if last:  # fold iota into the unit: ~(a ^ t) = (~a) ^ t
                if flip:
                    b[0] = 1 - b[0] if isinstance(b[0], int) else (b[0][0], not b[0][1])
                out[(x, y, z)] = unit_lit(b, [CHI])
            else:
                lit = unit_lit(b, [CHI])
                out[(x, y, z)] = lit ^ flip if isinstance(lit, int) else (lit[0], lit[1] ^ flip)
        return out, carried

    def xof_fn(msg: list) -> list:
        lanes = initial_state(k, msg)
        carry: list = []
        for step in range(depth):
            last = step == depth - 1
            out_pos = digest_pos if last else all_pos
            th_pos = chi_needs(out_pos)
            th, carry = theta_layer(lanes, th_pos, carry)
            out, carry = chi_layer(th, out_pos, carry, last)
            if last:
                return carry + [out[p][0] for p in digest_pos]
            carry = carry + [out[p] for p in digest_pos]
            lanes = [[[out[(x, y, z)] for z in range(w)] for y in range(5)] for x in range(5)]
        raise ValueError("depth must be >= 1")

    return xof_fn


def variant(k, depth):
    return build(k, depth), {"msg": Bits("0" * k.msg_len).bitlist}

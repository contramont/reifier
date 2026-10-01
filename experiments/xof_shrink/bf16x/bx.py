"""Keccak XOF circuits that SwiGLU computes exactly in bfloat16 (see eunits.py).

Every layer consists of E-exact gated units only, one traced call per layer (no compiler
copies), so with clean inputs (message bits and BOS = 1) every layer's outputs are exactly
0 or exactly BOS's value in bfloat16 (float32: see NOTES, errors ~2x per layer).
Any number of rounds per XOF step and any word size (log_w) are supported.

Per round, 3 layers:
  C   column parities C[x][z] of the <= 5 live bits of column (x, z), plus one-unit copies
      of the state bits that theta reads;
  T   theta a ^ C[x-1][z] ^ C[x+1][z-1] (<= 3 live inputs, 2 units; 1 for 2 inputs);
  X   chi, one unit per bit, iota folded in (as a negation flag, or into the unit when the
      bit is an output).
Constants (suffix, capacity) and negations are Python-level literals folded into the units,
as in xs.py. The last round computes only the digest and what it needs. Earlier digests
are carried as one-unit copies; the last layer emits the outputs in order.

Variants (all depth 3 per round):
  variant  as above, parities with xor_e (ceil(k/2) units, 3 for k = 5);
  pk1      + digests that travel >= 1 more layer packed two bits per feature (PACK);
  pk1l     + parities as the copies + corrections (xor_lin: 2 units for k = 5);
  pk1q     + one feature q = a + C[x-1] - C[x+1]' in {-1, 0, 1, 2} per theta position
           instead of copies and C features (theta = parity(q), 2 units), digest pairs
           packed in the C layer (their copies are the q's copy units);
  q_rt3, q_rt4
           pk1q without packing, plus a step-copy layer after every 3rd / 4th round, for
           float32 robustness at depth (build with exact steps: REIFIER_EXACT_STEPS=1);
           q_rt6 (every 6th round) fails float32.
"""

import reifier.examples.keccak as K
from reifier.neurons.core import Unit, gate, glu
from reifier.utils.format import Bits

from xs import fold, initial_state, keccak_maps
from bf16x.eunits import CHI, COPY, xor_e, xor_lin


# digest pairs t, u travel as one feature p = t + 2u + tu in {0, 1, 2, 4} (all exact)
PACK = [Unit((1, 0), 0, (0, 0), 1), Unit((0, 1), 0, (0, 0), 2), Unit((1, 1), -1, (0, 0), 1)]
PCOPY = [Unit((1,), 0, (0,), 1)]  # max(0, p): G = p in {0, 1, 2, 4}
UNPACK_T = [Unit((-1,), 2, (1,), 0), Unit((1,), -2, (0.25,), -0.5)]  # t = [p in {1, 4}]
UNPACK_U = [Unit((3,), -4, (-0.1875,), 0.875)]  # u = [p in {2, 4}], G = 3p - 4 (off: <= -1)


QPAR = [Unit((-1,), 0, (0,), 1), Unit((1,), 0, (-1,), 2)]  # parity of q in {-1,0,1,2}


def unit_lit(lits, units):
    sigs, us = fold(lits, units)
    if not sigs:
        total = sum(max(0, u.bias) * u.value_bias for u in us)
        assert total in (0, 1)
        return int(total)
    return (glu(sigs, us), False)


class Layer:
    """The units of one layer; equal xors and copies are built once"""

    def __init__(self, xor_units=xor_e, lin: bool = False):
        self.memo: dict = {}
        self.xor_units = xor_units
        self.lin = lin  # parity as copies + corrections where the copies are (mostly) here

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
            units = self.xor_units(len(sigs))
            if self.lin:
                n = len(sigs)
                missing = sum(("copy", x) not in self.memo for x in sigs)
                if missing + len(xor_lin(n)) - n < len(units):
                    units = xor_lin(n)  # its copy units are shared with the layer's copies
            self.memo[key] = glu(sigs, units)
        return (self.memo[key], bool(flip))

    def copy(self, lit):
        if isinstance(lit, int):
            return lit
        key = ("copy", lit[0])
        if key not in self.memo:
            self.memo[key] = glu([lit[0]], [COPY])
        return (self.memo[key], lit[1])

    def exact_copy(self, lit):
        out = unit_lit([lit], [COPY])
        assert not isinstance(out, int), "constant output"
        return out[0]

    def colpar(self, lits, copied=()):
        """a column parity on plain signals: (flip, sigs, units); copies + corrections
        (xor_lin) where the layer copies (most of) the signals anyway"""
        flip, cnt = 0, {}
        for lit in lits:
            if isinstance(lit, int):
                flip ^= lit
            else:
                flip ^= int(lit[1])
                cnt[lit[0]] = cnt.get(lit[0], 0) ^ 1
        sigs = sorted((x for x, c in cnt.items() if c), key=lambda x: x.uid)
        if not sigs:
            return flip, sigs, []
        units = self.xor_units(len(sigs))
        if self.lin:
            missing = sum(x not in copied for x in sigs)
            if missing + len(xor_lin(len(sigs))) - len(sigs) < len(units):
                units = xor_lin(len(sigs))
        return flip, sigs, units

    def qfeat(self, a_lit, col1, col2, copied=()):
        """q = a + C1 - C2 in {-1, 0, 1, 2} (literal values), one exact feature whose parity
        is theta; its units are a's copy and the two column parities' units, shared across
        the layer. Returns an int if q is constant."""
        f1, s1, u1 = self.colpar(col1, copied)
        f2, s2, u2 = self.colpar(col2, copied)
        terms = []
        if isinstance(a_lit, int):  # q = C1 + C2 in {0, 1, 2}; theta = a ^ parity(q)
            kappa, sg2 = f1 + f2, 1
        else:  # q = a + C1 - C2 in {-1, 0, 1, 2}
            kappa, sg2 = int(a_lit[1]) + f1 - f2, -1
            terms.append(([a_lit[0]], COPY, 1 - 2 * int(a_lit[1])))
        terms += [(s1, u, 1 - 2 * f1) for u in u1] + [(s2, u, sg2 * (1 - 2 * f2)) for u in u2]
        if not terms:
            return kappa
        key = ("q", kappa, tuple((tuple(x.uid for x in sg), u, c) for sg, u, c in terms))
        if key not in self.memo:
            sigs, idx = [], {}
            for sg, _, _ in terms:
                for x in sg:
                    if x not in idx:
                        idx[x] = len(sigs)
                        sigs.append(x)
            n, units = len(sigs), []
            for sg, u, c in terms:
                gw, vw = [0] * n, [0] * n
                for x, g, v in zip(sg, u.weights, u.value_weights):
                    gw[idx[x]] += g
                    vw[idx[x]] += v * c
                units.append(Unit(tuple(gw), u.bias, tuple(vw), u.value_bias * c))
            if kappa:
                units.append(Unit((0,) * n, 1, (0,) * n, kappa))
            self.memo[key] = glu(sigs, units, numeric=True)
        return self.memo[key]

    def qtheta(self, q, a_const=None):
        """theta = parity(q): for q in {-1, 0, 1, 2} max(0, -q) + max(0, q)(2 - q); for a
        constant a (q = C1 + C2 in {0, 1, 2}) max(0, q)(2 - q) with a as a flag"""
        if isinstance(q, int):
            return (q + (a_const or 0)) % 2
        key = ("qt", q)
        if key not in self.memo:
            self.memo[key] = glu([q], QPAR if a_const is None else QPAR[1:])
        return (self.memo[key], bool(a_const))

    def step_copy(self, lit):
        """a threshold-gate copy: a flat step, which re-thresholds float32 errors (exact in
        bfloat16 with SwiGLU.from_matrix(exact=True))"""
        if isinstance(lit, int):
            return lit
        key = ("scopy", lit[0])
        if key not in self.memo:
            self.memo[key] = gate([lit[0]], [1], 1)
        return (self.memo[key], lit[1])

    def pack(self, t, u):
        sigs, us = fold([t, u], PACK)
        return glu(sigs, us, numeric=True)

    def pcopy(self, p):
        return glu([p], PCOPY, numeric=True)

    def unpack(self, p, which):
        return glu([p], UNPACK_T if which == 0 else UNPACK_U)


def build(k: K.Keccak, steps: int, pack_min: int = 0, lin: bool = False, qf: bool = False,
          rt_every: int = 0):
    """The xof over `steps` steps of k.n rounds each.
    pack_min > 0: a digest that has at least pack_min packed carry layers ahead is packed two
      bits per feature (PACK) and unpacked in the last layer;
    lin: column parities as copies + corrections (xor_lin) where the copies exist anyway;
    qf: one feature q = a + C[x-1] - C[x+1]' per theta position (see Layer.qfeat);
    rt_every > 0: after every rt_every-th round (but the last), a layer of step copies
      re-thresholds the state (float32 robustness at depth; exact in bfloat16 with exact
      steps, SwiGLU.from_matrix(exact=True))."""
    w, d = k.w, k.d
    rcs = k.get_round_constants()
    rp, state_pos = keccak_maps(k)
    digest_pos = state_pos[:d]
    all_pos = [(x, y, z) for x in range(5) for y in range(5) for z in range(w)]

    def chi_needs(out_pos):
        need = {}
        for (x, y, z) in out_pos:
            for dx in range(3):
                need[rp[(x + dx) % 5][y][z]] = None
        return list(need)

    def col_needs(th_pos):
        need = {}
        for (x, y, z) in th_pos:
            need[((x + 4) % 5, z)] = None
            need[((x + 1) % 5, (z + 1) % w)] = None
        return list(need)

    n_rt = (k.n * steps - 1) // rt_every if rt_every else 0
    n_layers = 3 * k.n * steps + n_rt

    def add_digest(carry, bits, layer):
        packed_carries = n_layers - layer - (2 if qf else 3)  # (C copy,) pack, packed copies
        if pack_min and packed_carries >= pack_min:
            ent = [("P", bits[i], bits[i + 1]) for i in range(0, len(bits) - 1, 2)]
            return carry + ent + ([("b", bits[-1])] if len(bits) % 2 else [])
        return carry + [("b", b) for b in bits]

    def xof_fn(msg: list) -> list:
        lanes = initial_state(k, msg)
        carry: list = []
        layer = 0
        for step in range(steps):
            for r in range(k.n):
                last = step == steps - 1 and r == k.n - 1
                out_pos = digest_pos if last else all_pos
                th_pos = chi_needs(out_pos)
                # C: column parities and copies of the theta inputs
                L = Layer(lin=lin)
                if qf:  # one feature q = a + C[x-1] - C[x+1]' per theta position
                    copied = {lanes[x][y][z][0] for (x, y, z) in th_pos
                              if not isinstance(lanes[x][y][z], int)}
                    qs = {(x, y, z): L.qfeat(lanes[x][y][z],
                                             [lanes[(x + 4) % 5][y2][z] for y2 in range(5)],
                                             [lanes[(x + 1) % 5][y2][(z + 1) % w] for y2 in range(5)],
                                             copied)
                          for (x, y, z) in th_pos}
                else:
                    a = {p: L.copy(lanes[p[0]][p[1]][p[2]]) for p in th_pos}
                    cols = {c: L.xor([lanes[c[0]][y][c[1]] for y in range(5)]) for c in col_needs(th_pos)}
                carry = [("b", L.copy(c[1])) if c[0] == "b" else
                         ("p", L.pack(c[1], c[2])) if c[0] == "P" and qf else
                         ("P", L.copy(c[1]), L.copy(c[2])) if c[0] == "P" else
                         ("p", L.pcopy(c[1])) for c in carry]
                # T: theta
                L = Layer()
                if qf:
                    th = {p: L.qtheta(qs[p], lanes[p[0]][p[1]][p[2]]
                                      if isinstance(lanes[p[0]][p[1]][p[2]], int) else None)
                          for p in th_pos}
                else:
                    th = {(x, y, z): L.xor([a[(x, y, z)], cols[((x + 4) % 5, z)],
                                            cols[((x + 1) % 5, (z + 1) % w)]])
                          for (x, y, z) in th_pos}
                carry = [("b", L.copy(c[1])) if c[0] == "b" else
                         ("p", L.pack(c[1], c[2])) if c[0] == "P" else
                         ("p", L.pcopy(c[1])) for c in carry]
                # X: chi and iota
                L = Layer()
                layer += 3
                rc = rcs[r]
                if last:
                    outs = []
                    for c in carry:
                        if c[0] == "b":
                            outs.append(L.exact_copy(c[1]))
                        elif c[0] == "P":
                            outs += [L.exact_copy(c[1]), L.exact_copy(c[2])]
                        else:
                            outs += [L.unpack(c[1], 0), L.unpack(c[1], 1)]
                    for (x, y, z) in digest_pos:
                        b = [th[rp[(x + dx) % 5][y][z]] for dx in range(3)]
                        if x == 0 and y == 0 and rc[z] == "1":
                            b[0] = 1 - b[0] if isinstance(b[0], int) else (b[0][0], not b[0][1])
                        lit = unit_lit(b, [CHI])
                        assert not isinstance(lit, int), "constant output"
                        outs.append(lit[0])
                    return outs
                carry = [("b", L.copy(c[1])) if c[0] == "b" else
                         ("p", L.pack(c[1], c[2])) if c[0] == "P" else
                         ("p", L.pcopy(c[1])) for c in carry]
                new = {}
                for (x, y, z) in out_pos:
                    b = [th[rp[(x + dx) % 5][y][z]] for dx in range(3)]
                    flip = x == 0 and y == 0 and rc[z] == "1"
                    lit = unit_lit(b, [CHI])
                    new[(x, y, z)] = lit ^ flip if isinstance(lit, int) else (lit[0], lit[1] ^ flip)
                lanes = [[[new[(x, y, z)] for z in range(w)] for y in range(5)] for x in range(5)]
                g = step * k.n + r + 1
                if rt_every and g % rt_every == 0:  # re-threshold layer
                    L = Layer()
                    layer += 1
                    lanes = [[[L.step_copy(lanes[x][y][z]) for z in range(w)] for y in range(5)]
                             for x in range(5)]
                    carry = [("b", L.step_copy(c[1])) if c[0] == "b" else
                             ("P", L.step_copy(c[1]), L.step_copy(c[2])) if c[0] == "P" else
                             ("p", L.pcopy(c[1])) for c in carry]
            carry = add_digest(carry, [lanes[x][y][z] for (x, y, z) in digest_pos], layer)
        raise ValueError("steps must be >= 1")

    return xof_fn


def variant(k, steps):
    return build(k, steps), {"msg": Bits("0" * k.msg_len).bitlist}


def pk1(k, steps):
    """digests packed two per feature when they cross >= 1 packed carry layer"""
    return build(k, steps, pack_min=1), {"msg": Bits("0" * k.msg_len).bitlist}


def pk2(k, steps):
    return build(k, steps, pack_min=2), {"msg": Bits("0" * k.msg_len).bitlist}


def pk1l(k, steps):
    """pk1 with column parities as copies + corrections (xor_lin) in the C layers"""
    return build(k, steps, pack_min=1, lin=True), {"msg": Bits("0" * k.msg_len).bitlist}


def pk1q(k, steps):
    """pk1l with theta's inputs as one exact feature q = a + C[x-1] - C[x+1]' per position"""
    return build(k, steps, pack_min=1, lin=True, qf=True), {"msg": Bits("0" * k.msg_len).bitlist}


def pk1q_rt4(k, steps):
    """pk1q with a step-copy layer after every 4th round (float32 robustness at depth)"""
    return build(k, steps, pack_min=1, lin=True, qf=True, rt_every=4), {"msg": Bits("0" * k.msg_len).bitlist}


def pk1q_rt3(k, steps):
    return build(k, steps, pack_min=1, lin=True, qf=True, rt_every=3), {"msg": Bits("0" * k.msg_len).bitlist}


def q_rt3(k, steps):
    """qf + xor_lin, digests unpacked (binary, so the step copies re-threshold every
    feature), a step-copy layer after every 3rd round: float32-robust at depth"""
    return build(k, steps, lin=True, qf=True, rt_every=3), {"msg": Bits("0" * k.msg_len).bitlist}


def q_rt4(k, steps):
    """q_rt3 with a step-copy layer after every 4th round"""
    return build(k, steps, lin=True, qf=True, rt_every=4), {"msg": Bits("0" * k.msg_len).bitlist}


def q_rt6(k, steps):
    """q_rt3 with a step-copy layer after every 6th round"""
    return build(k, steps, lin=True, qf=True, rt_every=6), {"msg": Bits("0" * k.msg_len).bitlist}

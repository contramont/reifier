"""An independent simulation of the Lean model of the Reifier compiler (ReifierProofs/*.lean).

Written from scratch from the Lean definitions (Basic, Grid, Emit, Transformer, BF16); it does not
import or run the repo's Python. Arithmetic is mpmath at high precision (the forward pass in
real arithmetic), so that errors of 1e-13 are tracked exactly enough; `bf16_forward` is the
forward pass of BF16.lean, rounding every weight and result to bf16 (nearest, ties to even).

Runs: `python3 -I sim.py [7] [8] [5] [2b] [12]` (default 7 8 5 2b 12): CE-7 (non-flat GLUs), CE-8
(huge weights), CE-5 (window > CONTEXT), CE-2b (a gate exactly 0), CE-12 (the norm's ONE in bf16,
literal reading vs model). `deepcheck.py` checks MainRealDeep's bound on the CE-7 chains, and
`ce13.py` the RoPE score gaps with float32 angles (CE-13).
"""

from fractions import Fraction as Q
import math
import sys

import mpmath as mp

mp.mp.dps = 60

# ---------------------------------------------------------------- traces (Basic.lean)


class MLP:
    def __init__(self, ins, outs, act, hidden, wi, b, wo):
        self.ins, self.outs, self.act, self.hidden = list(ins), list(outs), act, hidden
        self.wi, self.b, self.wo = wi, b, wo  # functions j,k -> Q ; j -> Q ; i,j -> Q


class Trace:
    def __init__(self, inputs, mlps, outputs):
        self.inputs, self.mlps, self.outputs = list(inputs), list(mlps), list(outputs)


def idx_of(l, s):
    return l.index(s) if s in l else len(l)


def getD(l, k, d):
    return l[k] if 0 <= k < len(l) else d


def heaviside(x):
    return mp.mpf(1) if x >= 0 else mp.mpf(0)


def relu(x):
    return x if x > 0 else mp.mpf(0)


def pre(m, u, j):
    return sum((mp.mpf(m.wi(j, k).numerator) / m.wi(j, k).denominator) * u(k)
               for k in range(len(m.ins))) + mpq(m.b(j))


def mpq(q):
    q = Q(q)
    return mp.mpf(q.numerator) / q.denominator


def unit(m, u, j):
    if m.act == 'step':
        return heaviside(pre(m, u, j))
    if m.act == 'relu':
        return relu(pre(m, u, j))
    return relu(pre(m, u, j)) * pre(m, u, m.hidden + j)


def out(m, u, i):
    return sum(mpq(m.wo(i, j)) * unit(m, u, j) for j in range(m.hidden))


def values(t, x):
    v = dict(x)
    for m in t.mlps:
        u = (lambda v: (lambda k: v.get(getD(m.ins, k, 0), mp.mpf(0))))(dict(v))
        new = {}
        for s in m.outs:
            new[s] = out(m, u, m.outs.index(s))
        v.update(new)
    return v


def env(t, xbits):
    """valuation of the inputs from bits xbits(k)"""
    return {s: mp.mpf(1) if xbits(t.inputs.index(s)) else mp.mpf(0) for s in t.inputs}


# ---------------------------------------------------------------- grid (Grid.lean)

ONE = ('one',)


def SIG(s):
    return ('sig', s)


def CODE(i):
    return ('code', i)


def scan(t):
    depth = {}
    last = {}
    for m in t.mlps:
        d = 1 + max([depth.get(x, 0) for x in m.ins], default=0)
        for x in m.ins:
            l = last.get(x, None if x in t.outputs else 0)
            last[x] = None if l is None else max(l, d)
        for y in m.outs:
            depth[y] = d
    D = lambda s: depth.get(s, 0)
    L = lambda s: last[s] if s in last else (None if s in t.outputs else 0)
    return D, L


def layout(t):
    D, L = scan(t)

    def freedAt(x):
        l = L(x)
        return D(x) + 1 if l == 0 else l

    order = t.inputs + [y for m in t.mlps for y in m.outs]
    maxDepth = max([D(x) for x in order], default=0)
    width = len(t.inputs) + 1
    row = {0: ONE}
    for f in range(1, len(t.inputs) + 1):
        row[f] = SIG(t.inputs[f - 1])
    feat = {}

    def featOf(s):
        return feat[s] if s in feat else idx_of(t.inputs, s) + 1

    free = []  # head = top
    rows = [dict(row)]
    widths = [width]
    for d in range(1, maxDepth + 1):
        for x in [x for x in order if freedAt(x) == d]:
            f = featOf(x)
            free = [f] + free
            row[f] = None
        for y in [x for x in order if D(x) == d]:
            if not free:
                free = [width]
                width += 1
            f, free = free[0], free[1:]
            feat[y] = f
            row[f] = SIG(y)
        rows.append(dict(row))
        widths.append(width)
    return widths[-1], [lambda f, r=r: r.get(f) for r in rows]


def held(rows):
    out = [rows[0]]
    prev = rows[0]
    for r in rows[1:]:
        h = (lambda prev, r: (lambda f: r(f) if r(f) is not None else prev(f)))(prev, r)
        out.append(h)
        prev = h
    return out


def token_grid(t):
    Wl, rows = layout(t)
    b = len(t.outputs)
    W = Wl + b

    def tokenRow(r):
        return lambda f: ONE if f == 0 else (CODE(f - 1) if f <= b else r(f - b))

    rows = [tokenRow(r) for r in rows]
    for i, y in enumerate(t.outputs):
        def moveOut(r, i=i, y=y):
            found = next((f for f in range(W) if r(f) == SIG(y)), None)
            if found is None:
                return r
            return lambda g: SIG(y) if g == 1 + i else (None if g == found else r(g))
        rows = [moveOut(r) for r in rows]
    rows = held(rows)
    # materialise
    rows = [[r(f) for f in range(W)] for r in rows]
    return W, rows


# ---------------------------------------------------------------- emit (Emit.lean)


class Gated:
    pass


def compiled(m):
    h = m.hidden
    c = Gated()
    c.ins, c.outs = m.ins, m.outs
    if m.act == 'step':
        c.n = 2 * h
        g = lambda j, k: 4 * Q(m.wi(j if j < h else j - h, k))
        gb = lambda j: 4 * Q(m.b(j)) + 2 + Q(1, 2) if j < h else 4 * Q(m.b(j - h)) + 2 - Q(1, 2)
        v = lambda j, k: Q(0)
        vb = lambda j: Q(1)
        wo = lambda i, j: Q(m.wo(i, j)) if j < h else -Q(m.wo(i, j - h))
    elif m.act == 'relu':
        c.n = h
        g, gb = (lambda j, k: Q(m.wi(j, k))), (lambda j: Q(m.b(j)))
        v, vb = (lambda j, k: Q(0)), (lambda j: Q(1))
        wo = lambda i, j: Q(m.wo(i, j))
    else:
        c.n = h
        g, gb = (lambda j, k: Q(m.wi(j, k))), (lambda j: Q(m.b(j)))
        v, vb = (lambda j, k: Q(m.wi(h + j, k))), (lambda j: Q(m.b(h + j)))
        wo = lambda i, j: Q(m.wo(i, j))
    s = Q(32)
    nI = len(m.ins)
    c.g = lambda j, k: s * g(j, k) / 2
    c.gb = lambda j: s * gb(j) + sum(s * g(j, k) for k in range(nI)) / 2
    c.v = lambda j, k: v(j, k) / s / 2
    c.vb = lambda j: vb(j) / s + sum(v(j, k) / s for k in range(nI)) / 2
    c.wo = lambda i, j: 2 * wo(i, j)
    return c


def featIn(W, row, s):
    return next((f for f in range(W) if row[f] == s), W)


def inRow(W, row, s):
    return any(row[f] == s for f in range(W))


def layer_units(W, mlps, before, row):
    """each unit: (gate[W], val[W], out[W]) as Fractions"""
    units = []
    isNew = lambda f: f < W and row[f] is not None and not inRow(W, before, row[f])
    units.append(([Q(32) if f == 0 else Q(0) for f in range(W)],
                  [Q(1, 32) if f == 0 else Q(0) for f in range(W)],
                  [Q(-1) if isNew(f) else Q(0) for f in range(W)]))
    for m in mlps:
        if not m.outs:
            continue
        y = m.outs[0]
        if not (inRow(W, row, SIG(y)) and not inRow(W, before, SIG(y))):
            continue
        c = compiled(m)
        for j in range(c.n):
            gate = [Q(0)] * W
            val = [Q(0)] * W
            o = [Q(0)] * W
            for k in range(len(c.ins)):
                f = featIn(W, before, SIG(getD(c.ins, k, 0)))
                if f < W:
                    gate[f] += c.g(j, k)
                    val[f] += c.v(j, k)
            gate[0] += c.gb(j)
            val[0] += c.vb(j)
            for i in range(len(c.outs)):
                f = featIn(W, row, SIG(getD(c.outs, i, 0)))
                if f < W:
                    o[f] += c.wo(i, j)
            units.append((gate, val, o))
    for f in range(W):
        if 1 <= f and before[f] is not None and before[f] != row[f]:
            units.append(([Q(32) if g == 0 else Q(0) for g in range(W)],
                          [Q(1, 32) if g == f else Q(0) for g in range(W)],
                          [Q(-1) if g == f else Q(0) for g in range(W)]))
    return units


EPS = mp.mpf(1) / 1000000


def count(W, row):
    return sum(1 for f in range(1, W) if row[f] is not None)


def rmsnorm(W, row, one, x):
    scale = mp.sqrt((one ** 2 + count(W, row)) / W + EPS)
    rms = mp.sqrt(sum(xi ** 2 for xi in x) / W + EPS)
    z = [scale * xi / rms for xi in x]
    z[0] = z[0] / one
    return z


def gelu(z):
    return z / 2 * (1 + mp.tanh(mp.sqrt(2 / mp.pi) * (z + mp.mpf('0.044715') * z ** 3)))


def relu_act(z):
    return relu(z)


def layer_apply(act, W, units, z):
    res = [mp.mpf(0)] * W
    for gate, val, o in units:
        G = sum(mpq(gate[f]) * z[f] for f in range(W))
        V = sum(mpq(val[f]) * z[f] for f in range(W))
        a = act(G) * V
        for f in range(W):
            if o[f] != 0:
                res[f] += mpq(o[f]) * a
    return res


class Network:
    def __init__(self, t):
        self.W, self.grid = token_grid(t)
        self.layers = [layer_units(self.W, t.mlps, b, r)
                       for b, r in zip(self.grid, self.grid[1:])]
        self.one = 2 * mp.sqrt(self.W)

    def run(self, act, x):
        xs = [list(x)]
        for before, units in zip(self.grid, self.layers):
            z = rmsnorm(self.W, before, self.one, xs[-1])
            y = layer_apply(act, self.W, units, z)
            xs.append([a + b for a, b in zip(xs[-1], y)])
        return xs


# ---------------------------------------------------------------- transformer (Transformer.lean)

PAIRS, CONTEXT, MARGIN = 8, 512, mp.mpf(30)


def turn(i):
    return mp.power(10000, -mp.mpf(i) / PAIRS)


def meanCos(e):
    return sum(mp.cos(e * turn(i)) for i in range(PAIRS)) / PAIRS


_RU = None


def runnerUp():
    global _RU
    if _RU is None:
        _RU = max(meanCos(e) for e in range(1, CONTEXT))
    return _RU


def attnScale():
    return MARGIN / (PAIRS * (1 - runnerUp()))


def lookback(h, d):
    if d < PAIRS:
        return mp.cos(h * turn(d))
    if d < 2 * PAIRS:
        return -mp.sin(h * turn(d - PAIRS))
    return mp.mpf(0)


def rope(x, pos):
    r = [mp.mpf(0)] * (2 * PAIRS)
    for d in range(2 * PAIRS):
        if d < PAIRS:
            r[d] = x[d] * mp.cos(pos * turn(d)) - x[d + PAIRS] * mp.sin(pos * turn(d))
        else:
            r[d] = x[d] * mp.cos(pos * turn(d - PAIRS)) + x[d - PAIRS] * mp.sin(pos * turn(d - PAIRS))
    return r


class Transformer:
    def __init__(self, t, codes):
        self.t, self.codes = t, dict(codes)
        self.bits = len(t.outputs)
        self.net = Network(t)
        self.W = self.net.W
        self.window = len(t.inputs) // self.bits
        self.row0 = self.net.grid[0]
        self.vocab = max(self.codes) + 1
        self.embedded = [self.row0[f] if f <= self.bits else None for f in range(self.W)]
        self.ofeat = {}
        for h in range(self.window):
            for j in range(self.bits):
                s = getD(t.inputs, (self.window - 1 - h) * self.bits + j, 0)
                self.ofeat[(h, j)] = featIn(self.W, self.row0, SIG(s))

    def embed(self, tau):
        e = [mp.mpf(0)] * self.W
        if tau in self.codes:
            c = self.codes[tau]
            e[0] = self.net.one
            for f in range(1, self.bits + 1):
                if f < self.W:
                    e[f] = mp.mpf(1) if getD(c, f - 1, False) else mp.mpf(-1)
        return e

    def zvec(self, tau):
        return rmsnorm(self.W, self.embedded, self.net.one, self.embed(tau))

    def after_attn(self, toks, p):
        x = self.embed(toks[p])
        vis = [q for q in range(p + 1) if p - q < CONTEXT]
        zs = {q: self.zvec(toks[q]) for q in vis}
        A = attnScale()
        keys = {q: rope([zs[q][0] if d < PAIRS else mp.mpf(0) for d in range(2 * PAIRS)], q)
                for q in vis}
        for h in range(self.window):
            qv = rope([A * lookback(h, d) * zs[p][0] for d in range(2 * PAIRS)], p)
            scores = {q: sum(qv[d] * keys[q][d] for d in range(2 * PAIRS)) for q in vis}
            mx = max(scores.values())
            ex = {q: mp.exp(scores[q] - mx) for q in vis}
            tot = sum(ex.values())
            for j in range(self.bits):
                head = sum(ex[q] / tot * zs[q][j + 1] for q in vis)
                f = self.ofeat[(h, j)]
                if f < self.W:
                    x[f] += head
        return x

    def final(self, toks, p, act=gelu):
        return self.net.run(act, self.after_attn(toks, p))[-1]

    def logits(self, toks, p):
        x = self.final(toks, p)
        z = rmsnorm(self.W, self.net.grid[-1], self.net.one, x)
        return [sum(z[f] * self.embed(tau)[f] for f in range(self.W)) for tau in range(self.vocab)]


def window_bits(t, codes, toks, p):
    b = len(t.outputs)
    window = len(t.inputs) // b
    codes = dict(codes)

    def x(k):
        pos = max(p + 1 + k // b - window, 0)
        c = codes.get(toks[pos], [])
        return getD(c, k % b, False)
    return x


def expected(t, codes, toks, p):
    x = window_bits(t, codes, toks, p)
    v = values(t, env(t, x))
    return [2 * v.get(y, mp.mpf(0)) - 1 for y in t.outputs]


# ---------------------------------------------------------------- examples

def gate(ins, out_, w, th):
    return MLP(ins, [out_], 'step', 1, lambda j, k: Q(getD(w, k, 0)), lambda j: Q(-th),
               lambda i, j: Q(1))


def xor3():
    return Trace([0, 1, 2], [gate([0, 1, 2], 3, [1, 1, 1], 1), gate([0, 1, 2], 4, [1, 1, 1], 2),
                             gate([0, 1, 2], 5, [1, 1, 1], 3), gate([3, 4, 5], 6, [1, -1, 1], 1)],
                 [6])


def check_against_lean():
    """the values Examples.lean checks by decide"""
    W, rows = token_grid(xor3())
    assert rows == [[ONE, CODE(0), SIG(0), SIG(1), SIG(2)],
                    [ONE, CODE(0), SIG(5), SIG(4), SIG(3)],
                    [ONE, SIG(6), SIG(5), SIG(4), SIG(3)]], rows
    print("xor3 token grid matches Examples.lean")
    # CE traces whose unit data the Lean file proves by decide
    t10 = Trace([0], [MLP([0], [1], 'step', 1, lambda j, k: Q(128), lambda j: Q(-64),
                          lambda i, j: Q(1))], [1])
    n = Network(t10)
    data = [(g, v, o) for g, v, o in n.layers[0]]
    assert data == [([32, 0, 0], [Q(1, 32), 0, 0], [0, -1, 0]), ([80, 0, 8192], [Q(1, 32), 0, 0], [0, 2, 0]),
                    ([48, 0, 8192], [Q(1, 32), 0, 0], [0, -2, 0]), ([32, 0, 0], [0, Q(1, 32), 0], [0, -1, 0])], data
    print("CE-10 network matches the Lean decide facts")


# ---------------------------------------------------------------- CE-7: non-flat GLUs

def glu7(x, y):
    """relu(x + 1/2) * (2/3) x: identity on bits, slope 5/3 at 1"""
    return MLP([x], [y], 'reglu', 1, lambda j, k: Q(1) if j == 0 else Q(2, 3),
               lambda j: Q(1, 2) if j == 0 else Q(0), lambda i, j: Q(1))


def glu23(x, y):
    """relu(x + 1/2) * 10 x + relu(x - 1/2) * (52 x - 80): identity on bits, slope 23 at 1 and
    5 at 0; gates 1/2 from 0 on bits; weight sums at most 256"""
    wi = {(0, 0): Q(1), (1, 0): Q(1), (2, 0): Q(10), (3, 0): Q(52)}
    b = {0: Q(1, 2), 1: Q(-1, 2), 2: Q(0), 3: Q(-80)}
    return MLP([x], [y], 'reglu', 2, lambda j, k: wi[(j, k)], lambda j: b[j], lambda i, j: Q(1))


def chain(n, glu):
    return Trace([0], [glu(i, i + 1) for i in range(n)], [n])


def ce7(glu, name, nmax, codes=((0, [False]), (1, [True]))):
    print(f"\nCE-7 ({name}): chain of n GLUs after one head; tokens [0, 1], p = 1 "
          "(the neighbour holds the other bit)")
    toks = [0, 1]
    first_tol = first_flip = None
    for n in range(1, nmax + 1):
        t = chain(n, glu)
        # check the trace is the identity on bits and the gate margin holds
        M = Transformer(t, codes)
        exp_ = expected(t, codes, toks, 1)[0]
        xs = M.net.run(gelu, M.after_attn(toks, 1))
        got = xs[-1][1]
        err = abs(got - exp_)
        if first_tol is None and err > mp.mpf(10) ** -9:
            first_tol = n
        if first_flip is None and (got > 0) != (exp_ > 0):
            first_flip = n
        if n in (1, 2, 5, 10) or n % 10 == 0 or n == first_tol or n == first_flip:
            print(f"  n = {n:3d}: final[1] = {mp.nstr(got, 20)}  expected {exp_}  "
                  f"error {mp.nstr(err, 5)}")
        if first_flip is not None and n >= first_flip + 2:
            break
    print(f"  => first depth with error > 1e-9: n = {first_tol}; first depth whose bit is wrong: "
          f"n = {first_flip}")
    return first_tol, first_flip


# ---------------------------------------------------------------- CE-8: huge weights

def ce8():
    print("\nCE-8: one step gate [w x - w >= 0] (= x) after one head; tokens [0, 1], p = 1")
    codes = ((0, [False]), (1, [True]))
    toks = [0, 1]
    for e, w in [(f'{e}', Q(10) ** e) for e in range(8, 13)] + [('12 x 2', 2 * Q(10) ** 12),
                 ('12 x 5', 5 * Q(10) ** 12), ('13', Q(10) ** 13), ('14', Q(10) ** 14)]:
        t = Trace([0], [MLP([0], [1], 'step', 1, lambda j, k, w=w: w, lambda j, w=w: -w,
                            lambda i, j: Q(1))], [1])
        M = Transformer(t, codes)
        x = M.after_attn(toks, 1)
        got = M.final(toks, 1)[1]
        exp_ = expected(t, codes, toks, 1)[0]
        print(f"  w = 1e{e}: input feature after attention = 1 - {mp.nstr(1 - x[2], 5)}; "
              f"final[1] = {mp.nstr(got, 12)}  expected {exp_}  error {mp.nstr(abs(got - exp_), 5)}")


# ---------------------------------------------------------------- cross-checks of the Lean CEs

def no_input_step(b):
    return MLP([], [0], 'step', 1, lambda j, k: Q(0), lambda j: Q(b), lambda i, j: Q(1))


def lean_ces():
    print("\nCross-checks of the end-to-end Lean counterexamples (real arithmetic):")
    c2 = ((0, [False]), (1, [True]))
    t1 = Trace([], [no_input_step(Q(-1, 2))], [0])
    M = Transformer(t1, c2)
    print(f"  CE-1 [-1/2 >= 0]: final[1] = {mp.nstr(M.final([0], 0)[1], 15)}, expected "
          f"{expected(t1, c2, [0], 0)[0]}  (Lean: -1 + (gelu 16 - gelu(-16))/16)")
    glu = MLP([], [0], 'reglu', 1, lambda j, k: Q(0), lambda j: Q(1, 32) if j == 0 else Q(32),
              lambda i, j: Q(1))
    t2 = Trace([], [glu], [0])
    M = Transformer(t2, c2)
    print(f"  CE-2 relu(1/32)*32: final[1] = {mp.nstr(M.final([0], 0)[1], 15)}, expected "
          f"{expected(t2, c2, [0], 0)[0]}  (Lean: -1 + 2 gelu 1; gelu 1 = {mp.nstr(gelu(1), 10)})")
    c3 = ((0, [False, False]), (1, [True, True]))
    t3 = Trace([], [no_input_step(0)], [0, 0])
    M = Transformer(t3, c3)
    print(f"  CE-3 outputs [y, y]: final[1] = {mp.nstr(M.final([0], 0)[1], 15)}, expected "
          f"{expected(t3, c3, [0], 0)[0]}")
    t4 = Trace([0], [], [0])
    M = Transformer(t4, c2)
    print(f"  CE-4 output = input: final[1] = {mp.nstr(M.final([1], 0)[1], 15)}, expected "
          f"{expected(t4, c2, [1], 0)[0]}")
    for toks in ([0, 1, 1], [1, 0, 0, 1]):
        p = len(toks) - 1
        print(f"     CE-4 at tokens {toks}, p = {p}: final[1] = {mp.nstr(M.final(toks, p)[1], 15)}, "
              f"expected {expected(t4, c2, toks, p)[0]}")
    c6 = ((0, [True]), (1, [True]))
    t6 = Trace([], [no_input_step(0)], [0])
    M = Transformer(t6, c6)
    lg = M.logits([0], 0)
    print(f"  CE-6 two tokens of code [1]: logits = {[mp.nstr(l, 15) for l in lg]}")


# ---------------------------------------------------------------- CE-5: window > CONTEXT (float)

def ce5():
    """window 513: y = [x_oldest >= 1]. Float64 is enough: the error is of order 1."""
    import numpy as np
    print("\nCE-5: window 513 (CONTEXT + 1), y = [x0 >= 1] with x0 the oldest token's bit")
    n = CONTEXT + 1
    t = Trace(list(range(n)), [MLP([0], [n], 'step', 1, lambda j, k: Q(1), lambda j: Q(-1),
                                   lambda i, j: Q(1))], [n])
    codes = ((0, [False]), (1, [True]))
    toks = [1] + [0] * CONTEXT   # the oldest token holds 1, the 512 after it 0
    p = CONTEXT
    W, rows = token_grid(t)
    one = 2 * math.sqrt(W)
    # attention in float64, following Transformer.lean
    ru = float(runnerUp())
    A = 30.0 / (8 * (1 - ru))
    turns = np.array([10000.0 ** (-i / 8) for i in range(8)])
    vis = [q for q in range(p + 1) if p - q < CONTEXT]
    print(f"  positions visible from p = {p}: {vis[0]} .. {vis[-1]}; the oldest token is at "
          f"{p + 1 - n} (not visible: {p + 1 - n not in vis})")
    row0 = rows[0]
    emb = lambda tau: np.array([one] + [1.0 if dict(codes)[tau][0] else -1.0] + [0.0] * (W - 2))
    x = emb(toks[p]).copy()
    # every key is 1 on the first of each pair before RoPE; z[0] = 1 for every embedding
    def rope(v, pos):
        a, b = v[:8], v[8:]
        ang = pos * turns
        return np.concatenate([a * np.cos(ang) - b * np.sin(ang), b * np.cos(ang) + a * np.sin(ang)])
    keys = np.array([rope(np.r_[np.ones(8), np.zeros(8)], q) for q in vis])
    vals = np.array([1.0 if dict(codes)[toks[q]][0] else -1.0 for q in vis])
    head_out = {}
    for h in range(n):
        qv = rope(A * np.r_[np.cos(h * turns), -np.sin(h * turns)], p)
        s = keys @ qv
        a = np.exp(s - s.max())
        a /= a.sum()
        out = a @ vals
        f = featIn(W, row0, SIG(t.inputs[(n - 1 - h)]))
        x[f] += out
        head_out[h] = out
    f0 = featIn(W, row0, SIG(0))
    print(f"  head {n - 1} (meant for the oldest token) writes {head_out[n - 1]:.6f} into the "
          f"feature of x0; the ideal is +1")
    # the layer, in float64 (gelu with numpy)
    gel = lambda z: z / 2 * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))
    xs = [list(map(mp.mpf, x))]
    net = Network.__new__(Network)
    net.W, net.grid = W, rows
    net.layers = [layer_units(W, t.mlps, b, r) for b, r in zip(rows, rows[1:])]
    net.one = 2 * mp.sqrt(W)
    xs[0][0] = net.one
    fin = net.run(gelu, xs[0])[-1]
    exp_ = expected(t, codes, toks, p)[0]
    print(f"  final[1] = {mp.nstr(fin[1], 10)}, expected {exp_}: the bit is wrong")


# ---------------------------------------------------------------- CE-2b: a gate exactly 0

def zgate(C, a):
    """y = 2 relu(2x - 1) - relu(4x - 3) + C relu(a x - a): x on bits. Its gates on bits are
    -1 or 1, -3 or 1, and -a or 0: the weak margin (0 or at least 1/2 from 0) holds, the strict
    one does not. Flat: where a gate is positive (x = 1, the first two) the slopes cancel,
    2*2 - 4 = 0. Bounded by 2^16 if a <= 2^16 and 3 + C <= 2^16. No GLU."""
    wi = {0: Q(2), 1: Q(4), 2: Q(a)}
    b = {0: Q(-1), 1: Q(-3), 2: Q(-a)}
    wo = {0: Q(2), 1: Q(-1), 2: Q(C)}
    return lambda x, y: MLP([x], [y], 'relu', 3, lambda j, k: wi[j], lambda j: b[j],
                            lambda i, j: wo[j])


def check_props(m):
    """weak/strict margin, flatness, bounds of a one-input MLP on its two bit points"""
    weak = strict = flat = True
    for xv in (0, 1):
        u = lambda k: mp.mpf(xv)
        for j in range(m.hidden):
            pj = pre(m, u, j)
            weak &= (pj == 0) or abs(pj) >= mp.mpf(1) / 2
            strict &= abs(pj) >= mp.mpf(1) / 2
        sl = sum(mpq(m.wo(0, j)) * mpq(m.wi(j, 0)) for j in range(m.hidden) if pre(m, u, j) > 0)
        flat &= (sl == 0)
        assert out(m, u, 0) == xv
    bnd = max(max(abs(Q(m.wi(j, 0))) for j in range(m.hidden)),
              max(abs(Q(m.b(j))) for j in range(m.hidden)),
              sum(abs(Q(m.wo(0, j))) for j in range(m.hidden)))
    return weak, strict, flat, bnd


def ce2b(C, a, nmax=12):
    f = zgate(C, a)
    weak, strict, flat, bnd = check_props(f(0, 1))
    print(f"\nCE-2b: chain of n copies of y = 2relu(2x-1) - relu(4x-3) + {C} relu({a}(x-1)) "
          f"after one head; tokens [0, 1], p = 1")
    print(f"  identity on bits; weak margin {weak}, strict margin {strict}, flat {flat}, "
          f"largest weight sum {bnd} (<= 2^16: {bnd <= 2**16})")
    codes = ((0, [False]), (1, [True]))
    toks = [0, 1]
    first_tol = first_deep = first_flip = None
    for n in range(1, nmax + 1):
        t = chain(n, f)
        M = Transformer(t, codes)
        exp_ = expected(t, codes, toks, 1)[0]
        xs = M.net.run(gelu, M.after_attn(toks, 1))
        got = xs[-1][1]
        err = abs(got - exp_)
        if first_tol is None and err > mp.mpf(10) ** -9:
            first_tol = n
        if first_deep is None and err > mp.mpf(10) ** -10:
            first_deep = n
        if first_flip is None and (got > 0) != (exp_ > 0):
            first_flip = n
        print(f"  n = {n:3d}: final[1] = {mp.nstr(got, 15)}  expected {exp_}  error {mp.nstr(err, 5)}")
        if first_flip is not None and n >= first_flip + 1:
            break
    print(f"  => error > 1e-10 (MainRealDeep, sigma = 0) from n = {first_deep}; > 1e-9 (MainReal) "
          f"from n = {first_tol}; wrong bit at n = {first_flip}")


# ---------------------------------------------------------------- bf16 (BF16.lean)

def fl(x):
    """round to the nearest bf16 number, ties to even (round_to_bf16 of gemma.py); 8 bits,
    subnormals down to 2^-133"""
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(0)
    sgn = 1 if x > 0 else -1
    a = abs(x)
    m, e = mp.frexp(a)          # a = m 2^e, 1/2 <= m < 1
    E = max(e - 8, -133)        # a = q 2^E, 128 <= q < 256
    q = a / mp.mpf(2) ** E
    n = int(mp.floor(q))
    frac = q - n
    if frac > mp.mpf(1) / 2 or (frac == mp.mpf(1) / 2 and n % 2 == 1):
        n += 1
    return sgn * mp.mpf(n) * mp.mpf(2) ** E


def bf16_forward(M, toks, p, literal):
    """BF16.finalAt (and the logits) of Transformer M at p: every weight and every result rounded.
    literal=False: our model, the norms read ONE as fl(one), the bf16 number the stream holds.
    literal=True: Python's rmsnorm(x, row, one) with the float one = 2 sqrt(W)."""
    W = M.W
    one_b = fl(M.net.one)
    norm_one = M.net.one if literal else one_b
    embed = lambda tau: [fl(e) for e in M.embed(tau)]
    zvec = lambda tau: [fl(z) for z in rmsnorm(W, M.embedded, norm_one, embed(tau))]
    A = attnScale()
    vis = [q for q in range(p + 1) if p - q < CONTEXT]
    zs = {q: zvec(toks[q]) for q in vis}

    def rope_b(xv, pos):
        return [fl(r) for r in rope(xv, pos)]
    keys = {q: rope_b([fl(zs[q][0]) if d < PAIRS else mp.mpf(0) for d in range(2 * PAIRS)], q)
            for q in vis}
    x = embed(toks[p])
    for h in range(M.window):
        qv = rope_b([fl(fl(A * lookback(h, d)) * zs[p][0]) for d in range(2 * PAIRS)], p)
        scores = {q: fl(sum(qv[d] * keys[q][d] for d in range(2 * PAIRS))) for q in vis}
        tot = sum(mp.exp(scores[q]) for q in vis)
        att = {q: fl(mp.exp(scores[q]) / tot) for q in vis}
        for j in range(M.bits):
            head = fl(sum(att[q] * zs[q][j + 1] for q in vis))    # value = fl(z[j+1]) = z[j+1]
            f = M.ofeat[(h, j)]
            if f < W:
                x[f] = fl(x[f] + fl(head))
    after = list(x)
    for before, units in zip(M.net.grid, M.net.layers):
        z = [fl(zz) for zz in rmsnorm(W, before, norm_one, x)]
        acc = [mp.mpf(0)] * W
        for gate, val, o in units:
            G = fl(sum(fl(mpq(gate[g])) * z[g] for g in range(W)))
            V = fl(sum(fl(mpq(val[g])) * z[g] for g in range(W)))
            a = fl(fl(gelu(G)) * V)
            for f in range(W):
                if o[f] != 0:
                    acc[f] += fl(mpq(o[f])) * a
        x = [fl(x[f] + fl(acc[f])) for f in range(W)]
    zl = [fl(zz) for zz in rmsnorm(W, M.net.grid[-1], norm_one, x)]
    logits = [fl(sum(zl[f] * fl(M.embed(tau)[f]) for f in range(W))) for tau in range(M.vocab)]
    return after, x, logits


def majority(n):
    """[x_0 + ... + x_{n-1} >= (n + 1) / 2] on n one-bit tokens: one threshold gate"""
    th = (n + 1) // 2
    return Trace(list(range(n)), [MLP(list(range(n)), [n], 'step', 1, lambda j, k: Q(1),
                                      lambda j: Q(-th), lambda i, j: Q(1))], [n])


def bf16ok(t):
    """BF16OK of t, checked on every input of its MLPs (all bits, so only for few inputs) or on
    the values the gates can take for a threshold gate of weights 1"""
    for m in t.mlps:
        c = compiled(m)
        for j in range(c.n):
            for k in range(len(c.ins)):
                for w in (c.g(j, k), c.v(j, k)):
                    if fl(mpq(w)) != mpq(w):
                        return False
            for w in [c.gb(j), c.vb(j)] + [c.wo(i, j) for i in range(len(c.outs))]:
                if fl(mpq(w)) != mpq(w):
                    return False
            # inputs: bits; for weights all equal the gate depends on the count of ones only
            nI = len(c.ins)
            for ones in range(nI + 1):
                u = lambda k, ones=ones: mp.mpf(1) if k < ones else mp.mpf(0)
                G = sum(mpq(c.g(j, k)) * (2 * u(k) - 1) for k in range(nI)) + mpq(c.gb(j))
                V = sum(mpq(c.v(j, k)) * (2 * u(k) - 1) for k in range(nI)) + mpq(c.vb(j))
                for y in (G, V, relu(G) * V):
                    if fl(y) != y:
                        return False
    return True


def ce12(n=19, trials=40, seed=1):
    import random
    rnd = random.Random(seed)
    t = majority(n)
    codes = ((0, [False]), (1, [True]))
    M = Transformer(t, codes)
    W = M.W
    print(f"\nCE-12: majority of {n} one-bit tokens, one threshold gate; width {W}; "
          f"one = 2 sqrt {W} = {mp.nstr(M.net.one, 10)}, fl(one) = {fl(M.net.one)}; "
          f"BF16OK: {bf16ok(t)}")
    # the embedding norm on a token, both readings
    for literal in (False, True):
        no = M.net.one if literal else fl(M.net.one)
        z = [fl(zz) for zz in rmsnorm(W, M.embedded, no, [fl(e) for e in M.embed(1)])]
        print(f"  {'literal' if literal else 'model  '}: the embedding norm of token 1 gives "
              f"ONE {z[0]}, code bit {z[1]}")
    p = n - 1
    cases = [[1] * ((n + 1) // 2) + [0] * (n - (n + 1) // 2),
             [0] * ((n + 1) // 2) + [1] * (n - (n + 1) // 2),
             [1] * n, [0] * n]
    cases += [[rnd.randint(0, 1) for _ in range(n)] for _ in range(trials)]
    for literal in (False, True):
        wrong = inexact = 0
        worst = mp.mpf(0)
        lwrong = 0
        for toks in cases:
            after, fin, logits = bf16_forward(M, toks, p, literal)
            exp_ = expected(t, codes, toks, p)[0]
            got = fin[1]
            if got != exp_:
                inexact += 1
                worst = max(worst, abs(got - exp_))
            if (got > 0) != (exp_ > 0):
                wrong += 1
            want = 1 if exp_ > 0 else 0
            if not all(logits[want] > logits[o] for o in range(len(logits)) if o != want):
                lwrong += 1
        print(f"  {'literal' if literal else 'model  '}: {len(cases)} token sequences; output "
              f"feature not exactly +-1 in {inexact}, wrong sign in {wrong}, largest error "
              f"{mp.nstr(worst, 6)}; right token not the largest logit in {lwrong}")
    toks = cases[0]
    for literal in (False, True):
        after, fin, logits = bf16_forward(M, toks, p, literal)
        print(f"  {'literal' if literal else 'model  '} at {sum(toks)} ones (threshold "
              f"{(n + 1) // 2}): after the attention {[mp.nstr(a, 6) for a in after[:4]]} ...; "
              f"final {[mp.nstr(a, 8) for a in fin[:3]]} ...; logits "
              f"{[mp.nstr(l, 6) for l in logits]}")


if __name__ == '__main__':
    check_against_lean()
    lean_ces()
    which = sys.argv[1:] or ['7', '8', '5', '2b', '12']
    if '7' in which:
        ce7(glu7, "relu(y+1/2)*(2/3)y, slope 5/3", 120)
        ce7(glu23, "two units, slope 23 at 1", 40)
    if '8' in which:
        ce8()
    if '5' in which:
        ce5()
    if '12' in which:
        ce12()
    if '2b' in which:
        ce2b(2 ** 15, 1)
        ce2b(2 ** 15, 2 ** 12)
        ce2b(2 ** 10, 1)

"""SwiGLU weights for the optimization levels.

`build(graph, options, dtype)` makes an MLP_SwiGLU from a leveled graph the way the
core's `MLP_SwiGLU.from_matrices` does: per layer, every step row is the difference of
two ReLU-like hidden units, every gated unit is one hidden unit, and BOS (feature 0,
always 1) carries the biases. `MLPOptions` adds the numerics that the levels were
measured with. This construction is the levels' own copy, so their weights stay as
measured whatever the core's builder does.

`fp16_bound(mlp)` bounds the values float16 must hold, and `dense_bound(graph)` bounds
the size of the MLP from below, so a level can skip recipes that cannot be smallest.
"""

import math
from dataclasses import dataclass, replace

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from reifier.compile.levels import LeveledGraph, Origin
from reifier.neurons.core import Unit
from reifier.tensors.matrices import Matrices
from reifier.tensors.swiglu import MLP_SwiGLU, SwiGLU


FP16_MAX = 65504.0
FLIP_MAX = 8  # exact_readout flips step rows whose flipped sum stays <= this
ZERO_NOISE_UNITS = 1.5  # see _spread
G16_LIM, P16_LIM, Y16_LIM = 2.0**15, 2.0**15, 2.0**14  # see _fit16


@dataclass(frozen=True)
class MLPOptions:
    """How build makes the MLP. The defaults are the core builder's, except that
    float16 builds, like every level, move BOS bits beyond bfloat16's 8 significant
    bits to a second BOS (the core's: beyond float16's 11)"""

    c: int = 4  # making ReLU-simulated step fn steeper
    q: int = 8  # scaling before and after SiLU to avoid non-ReLU-like dip
    # steps that 16-bit floats compute exactly on 0/1 inputs (None: in 16-bit dtypes).
    # A step rises over [1/2, 1/2 + 1/c], and a row whose sum can exceed 1 by more than
    # it can fall below 0 is built as BOS - step(1 - sum), which keeps large sums where
    # both ReLUs are off. With c and q powers of 2, a layer whose BOS weights need
    # more than bfloat16's 8 significant bits reads a second BOS that takes the rest
    # (from the layer before, or from a first layer of step copies of the inputs)
    exact: bool | None = None
    # with exact, the last layer also flips every step row whose flipped sum stays
    # <= FLIP_MAX, so every 1 is exactly BOS (boolify reads a 1 only within 0.02 of BOS)
    exact_readout: bool = False
    # with exact, steps rise over [1/2 - 1/2c, 1/2 + 1/2c]: margins of 3/8 on both sides
    # for c = 4, still exact in 16-bit floats
    center: bool = False
    # hidden layers carry BOS at the power of 2 nearest sqrt(width) / 2. A heavy BOS
    # dominates the RMSNorm, which bounds the pre-activations and keeps steps sharp
    heavy_bos: bool = False
    # > 0: hidden layers carry BOS as copies of size 1 instead (read as their mean),
    # the power of 2 nearest width / bos_copies of them: they hold the RMSNorm's scale
    # like a heavy BOS, but noise relative to the largest feature stays relative to 1
    bos_copies: int = 0
    # a first layer that copies the inputs as steps on 1.5 x - 1/2, which keep their
    # margins under any mean shift smaller than BOS
    prologue: bool = False
    # every layer but the last also outputs this many always-0 features, and every
    # layer but the first makes its rows sum to 0 with them, so a LayerNorm's mean shift
    # cancels
    ln_invariant: int = 0
    zero_spread: bool = False  # spread every row's sum over all always-0 features
    # > 0: multiply each layer's outputs by the largest power of 2 up to this that keeps
    # them within float16 range: a layer whose norm scale a host cuts to s outputs about
    # s^2 of its usual size, which must stay above the next RMSNorm's eps
    out_scale: float = 0.0
    fit16: bool = False  # rescale every layer by powers of 2 to float16 range (_fit16)

    def __post_init__(self) -> None:
        nonneg = (self.bos_copies, self.ln_invariant, self.out_scale)
        if not (self.c > 0 and self.q > 0 and all(x >= 0 for x in nonneg)):  # NaN too
            raise ValueError("need c, q > 0 and bos_copies, ln_invariant, out_scale >= 0")


@dataclass(frozen=True)
class _Layout:
    """How the features between two layers carry BOS: at size bos, then after the other
    features a second BOS (exact builds), copies of BOS and always-0 features"""

    bos: float = 1.0
    second: bool = False
    copies: int = 1
    zeros: int = 0


def build(
    graph: LeveledGraph, options: MLPOptions, dtype: t.dtype = t.float32
) -> MLP_SwiGLU:
    """The MLP_SwiGLU of a leveled graph; weights are made in float32, then cast"""
    o = options
    if o.exact is None:
        o = replace(o, exact=dtype in (t.bfloat16, t.float16))
    matrices = Matrices.from_graph(graph)
    mlist, ulist = list(matrices.mlist), list(matrices.ulist)
    if o.prologue:
        copy = t.eye(mlist[0].size(1), dtype=t.float32) * 1.5
        copy[0, 0], copy[1:, 0] = 1, -0.5
        mlist, ulist = [copy] + mlist, [None] + ulist

    def layers(second: list[bool]) -> list[SwiGLU]:
        xs = _layouts(mlist, o, second)
        n = len(mlist)
        return [_layer(m, u, o, xs[i], xs[i + 1], last=i == n - 1)
                for i, (m, u) in enumerate(zip(mlist, ulist, strict=True))]

    swiglus = layers([False] * (len(mlist) + 1))
    # BOS weights beyond bfloat16's precision: a second BOS takes the rest (with c or q
    # not a power of 2 the other weights round too)
    if o.exact and all(x > 0 and math.log2(x).is_integer() for x in (o.c, o.q)):
        need = [i for i, L in enumerate(swiglus) if not _bos_fits(L)]
        if need:
            if need[0] == 0:  # the input has one BOS: copy the inputs first
                mlist = [t.eye(mlist[0].size(1), dtype=t.float32)] + mlist
                ulist = [None] + ulist
                need = [i + 1 for i in need]
            swiglus = layers([i in need for i in range(len(mlist) + 1)])
    for L in swiglus:
        if o.out_scale:
            L.wo.weight.data *= _auto_out_scale(L, o.out_scale)
        if o.fit16:
            _fit16(L)
        L.to(dtype).dtype = dtype
    mlp = MLP_SwiGLU([], dtype=dtype)
    mlp.layers = nn.Sequential(*swiglus)  # hidden sizes vary with gated units
    return mlp


def _layouts(mlist: list[t.Tensor], o: MLPOptions, second: list[bool]) -> list[_Layout]:
    """the layout of every layer's input, then of the last layer's output"""
    n = len(mlist)
    out = []
    for i, sec in enumerate(second):
        hidden = 0 < i < n
        bos, copies = 1.0, 1
        if hidden and o.bos_copies:
            copies = 2 ** max(0, round(math.log2(mlist[i].size(1) / o.bos_copies)))
        elif hidden and o.heavy_bos:
            width = mlist[i].size(1) + int(sec) + o.ln_invariant
            bos = float(2 ** max(0, round(math.log2(math.sqrt(width) / 2))))
        out.append(_Layout(bos, sec, copies, o.ln_invariant * hidden))
    return out


def _layer(
    w: t.Tensor,
    units: tuple[t.Tensor, ...] | None,
    o: MLPOptions,
    x: _Layout,
    y: _Layout,
    last: bool,
) -> SwiGLU:
    """A layer for a matrix with the biases folded in (row and column 0 are BOS) and
    its gated units (Matrices.layer_to_units), reading features laid out as x, writing
    y. Every row is a step: two ReLUs a, b (SiLUs scaled up by c q and down again) such
    that a - b is 0 until the row's sum reaches 1/2 - 1/2c, then rises to 1 at
    1/2 + 1/2c (exact steps: see MLPOptions). Demo:
    https://www.desmos.com/calculator/w806u4n8hl"""
    c, q = o.c, o.q
    out_features = w.size(0)
    w = w.contiguous().to(dtype=t.float32)
    comp = t.zeros(out_features, dtype=t.bool)  # rows built as BOS - step(1 - sum)
    if o.exact:
        smax = w[:, 0] + w[:, 1:].clamp(min=0).sum(1)
        smin = w[:, 0] + w[:, 1:].clamp(max=0).sum(1)
        comp = (1 - smin) < smax
        if o.exact_readout and last:  # every step row with a small flipped sum
            comp |= (1 - smin) <= FLIP_MAX
            if units is not None:  # rows of gated units have no steps
                comp &= ~units[2].any(dim=1)
        comp[0] = False
        w = w.clone()
        w[comp] = -w[comp]
        w[comp, 0] += 1

    # constructing w_gate
    wg = t.cat([w, w], dim=0)
    lo = 0.5 if o.exact and not o.center else 0.5 - 1 / (2 * c)  # where the step rises
    wg[1:out_features, 0] -= lo + 1 / c  # sub
    wg[out_features + 1 :, 0] -= lo  # add
    wg *= c * q  # scale up
    # BOS as relu(2*c*q) - relu(c*q): exact in any float format, so a gated unit that
    # outputs one BOS matches it bit for bit
    wg[0, 0], wg[out_features, 0] = c * q, 2 * c * q

    # constructing w_value: it takes part of the scale-down, which keeps hidden
    # activations (and so the effect of weight noise) at their size for q = 4
    v = 4 / q
    wv = t.zeros_like(wg)
    wv[:, 0] += v

    # constructing w_out
    eye = t.eye(out_features)
    wo = t.cat((-eye, eye), dim=1)
    wo /= q * v  # scale down
    wo[0] /= c  # the BOS pair differs by c*q, not q
    wo[comp] = -wo[comp]
    wo[comp, 0], wo[comp, out_features] = wo[0, 0], wo[0, out_features]

    # gated units replace the steps of their rows, with one hidden unit each:
    # silu(c*q*gate) * value / (c*q), which tends to max(0, gate) * value
    if units is not None:
        gates, values, outs = units
        steps = ~outs.any(dim=1).repeat(2)  # step units of rows without gated units
        wg = t.cat([wg[steps], gates * (c * q)])
        wv = t.cat([wv[steps], values * v])
        wo = t.cat([wo[:, steps], outs / (c * q * v)], dim=1)

    n_in = wg.size(1)
    wg = _read(wg, x, 0.0 if o.zero_spread else c * q)
    wv = _read(wv, x, 0.0 if o.zero_spread else v)
    wo[0] *= y.bos
    bos = wo[:1].repeat(int(y.second) + y.copies - 1, 1)  # the second BOS, BOS copies
    wo = t.cat([wo, bos, t.zeros(y.zeros, wo.size(1))])

    L = SwiGLU(wg.size(1), wo.size(0), hidden_f=len(wg))
    with t.no_grad():
        for linear, wi in ((L.wg, wg), (L.wv, wv), (L.wo, wo)):
            linear.weight.copy_(wi)
    # the input's BOS: its size and columns, then the always-0 features (_layer_bounds)
    L.bos_in, L.zero_in = x.bos, x.zeros
    L.bos_cols = (0, *range(n_in, wg.size(1) - x.zeros))
    return L


def _read(m: t.Tensor, x: _Layout, unit: float) -> t.Tensor:
    """m (wg or wv, BOS weights in column 0) for inputs laid out as x (unit: _spread)"""
    m[:, 0] /= x.bos
    if x.second:  # BOS weights = bfloat16 part on BOS + the rest on the second BOS
        hi = m[:, 0].to(t.bfloat16).float()
        m = t.cat([m, (m[:, 0] - hi).unsqueeze(1)], dim=1)
        m[:, 0] = hi
    m[:, 0] /= x.copies  # BOS weights spread over the copies of BOS
    m = t.cat([m, m[:, :1].repeat(1, x.copies - 1)], dim=1)
    if x.zeros:  # the always-0 features take minus each row sum
        m = t.cat([m, _spread(-m.sum(1), x.zeros, unit)], dim=1)
    return m


def _spread(sums: t.Tensor, k: int, unit: float) -> t.Tensor:
    """Columns for k always-0 features that take the row sums: a row whose sum is r
    units spreads it evenly over the first 2^j <= k features, 2^j >= (r / u)^2 with
    u = ZERO_NOISE_UNITS, so their noise adds at most about u units to the row
    (spreading over k features divides their noise by sqrt(k)); with unit 0, every row
    over all k"""
    n = t.full_like(sums, float(k))
    if unit:
        r = (sums.abs() / unit).clamp(min=1e-9)
        j = t.ceil(t.log2((r / ZERO_NOISE_UNITS) ** 2)).clamp(min=0)
        n = (2**j).clamp(max=k)
    cols = t.arange(k).unsqueeze(0) < n.unsqueeze(1)
    return t.where(cols, (sums / n).unsqueeze(1), t.zeros(()))


def _bos_fits(L: SwiGLU) -> bool:
    """whether the BOS weights of wg and wv (built in float32) have at most bfloat16's 8
    significant bits"""
    for w in (L.wg.weight, L.wv.weight):
        m = t.frexp(w.detach()[:, 0].float()).mantissa * 2**8
        if not t.equal(m, m.round()):
            return False
    return True


def _auto_out_scale(L: SwiGLU, cap: float) -> float:
    """the largest power of 2 <= cap by which L's outputs can be multiplied while they
    stay below FP16_MAX / 16 for 0/1 inputs: the largest normalized BOS is reached when
    only BOS is on, so L's outputs on that input (times 2) bound them"""
    n = L.wg.weight.size(1)
    x = t.zeros(1, n)
    x[0, list(L.bos_cols)] = float(L.bos_in)
    h = F.rms_norm(x, (n,), L.norm.weight.detach().float())
    g, v = h @ L.wg.weight.detach().float().T, h @ L.wv.weight.detach().float().T
    top = 2 * (F.silu(g) * v @ L.wo.weight.detach().float().T).abs().max().item()
    k = math.floor(math.log2(FP16_MAX / 16 / max(top, 1e-30)))
    return float(min(cap, 2.0 ** max(k, -8)))


def _fit16(L: SwiGLU) -> None:
    """Rescale a layer (float32 weights) by powers of 2 so that its float16 bounds fit:
    the norm weight to the largest one whose gate pre-activations stay <= G16_LIM (the
    sharpest steps), then the value rows down (wo up) until the products stay <=
    P16_LIM, then wo by the largest factor whose outputs stay <= Y16_LIM (the next
    RMSNorm removes it, and it keeps outputs scaled down by a host above the epsilon)"""
    with t.no_grad():
        L.norm.weight.fill_(1.0)
        g, _, _, _ = _layer_bounds(L)
        L.norm.weight.fill_(2.0 ** math.floor(math.log2(G16_LIM / g.max().item())))
        _, _, p, _ = _layer_bounds(L)
        if p.max().item() > P16_LIM:
            r = 2.0 ** math.ceil(math.log2(p.max().item() / P16_LIM))
            L.wv.weight.div_(r)
            L.wo.weight.mul_(r)
        _, _, _, o = _layer_bounds(L)
        L.wo.weight.mul_(2.0 ** math.floor(math.log2(Y16_LIM / o.max().item())))


def fp16_bound(mlp: MLP_SwiGLU) -> float:
    """An upper bound on the |values| that float16 must hold in any layer on exact 0/1
    features: the gate and value pre-activations and the hidden products
    silu(g) * v (see _layer_bounds), not the weights. E.g. the first layer reads BOS = 1
    among n_in inputs: its BOS pair reaches 2 c q sqrt(n_in + 1), and their product
    8 c (n_in + 1), so every build overflows float16 (65504) from about 2047 inputs
    (fit16 rescales each layer to fit). It reads the BOS layout that build stores on
    each layer; a layer without it (the core's, or one loaded from a state_dict) counts
    as BOS = 1 on column 0 alone: looser where a core layer reads a second BOS, and not
    guaranteed to bound a level's layer with a heavy BOS"""
    bound = 0.0
    for L in mlp.layers:
        g, v, prod, _ = _layer_bounds(L)
        bound = max(bound, g.max().item(), v.max().item(), prod.max().item())
    return bound


def _layer_bounds(L: SwiGLU) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Upper bounds on |gate|, |value| (per hidden unit), |silu(gate) * value| (per
    hidden unit) and |output| (per output feature) of a layer on exact 0/1 features.
    Its n inputs are BOS (of size B, on one column or more), 0/1 features and always-0
    features (a layer without build's layout: BOS = 1 on column 0 alone); with k ones
    the RMSNorm scales them by sqrt(n / (S + k)), S = B^2 per BOS column. A row with
    BOS weight b (in units of B) and k' features of weight up to m is then at most
    sqrt(n) * max(|b| B / sqrt(S), (|b| B + k' m) / sqrt(S + k')) (over k the maximum
    is at an end), times the norm weight. A hidden product is at most |v| times
    max(g, 0.28), g's upper bound from its BOS weight and positive weights alone
    (silu(g) <= max(g, 0.28)); an output at most the sum of |wo| times the products"""
    B = float(getattr(L, "bos_in", 1.0))
    bos = list(getattr(L, "bos_cols", (0,)))
    n = L.wg.weight.size(1)
    feats = t.ones(n, dtype=t.bool)
    feats[bos] = False
    nz = getattr(L, "zero_in", 0)
    if nz:
        feats[n - nz:] = False  # always 0
    S = len(bos) * B * B
    nw = L.norm.weight.detach().float().abs().max().item() * math.sqrt(n)

    def top(b: t.Tensor, wf: t.Tensor) -> t.Tensor:  # max of b + w.x over x
        m = wf.max(dim=1).values if wf.size(1) else t.zeros(len(wf))
        k = (wf > 0).sum(1).float()
        return t.maximum(b / math.sqrt(S), (b + k * m) / (S + k).sqrt()) * nw

    rows = []
    for w in (L.wg.weight, L.wv.weight):
        w = w.detach().float()
        b = w[:, bos].sum(1) * B
        wf = w[:, feats]
        rows.append(t.maximum(top(b, wf), top(-b, -wf)))  # |pre-activation|
    g, v = rows
    wg = L.wg.weight.detach().float()
    bg = wg[:, bos].sum(1) * B
    g_up = top(bg, wg[:, feats])
    prod = g_up.clamp(min=0.28) * v
    out = L.wo.weight.detach().float().abs() @ prod
    return g, v, prod, out


def dense_bound(graph: LeveledGraph) -> int:
    """A lower bound on the dense parameters of the MLP that build makes from a graph:
    per layer, BOS and the level's features in, 2 hidden units per row without gated
    units (BOS's pair included) and one per shared gated unit (_shared). A prologue,
    always-0 features, a second BOS and BOS copies only add to it"""
    total = 0
    for (n_out, n_in), level in zip(graph.shapes, graph.levels[1:]):
        units, rows = set(), set()
        for o in level.origins:
            for u in o.units:
                key = _shared(o, u)
                if key is not None:
                    units.add(key)
                    rows.add(o.index)
        hidden = 2 * (n_out + 1 - len(rows)) + len(units)
        total += (n_in + 1) * (1 + 2 * hidden) + (n_out + 1) * hidden
    return total


def _shared(o: Origin, u: Unit) -> tuple | None:
    """the hidden unit of a gated unit, as Matrices.layer_to_units shares them (equal
    gates, proportional values); None for a unit it drops (always 0). Counting these is
    several times faster than building layer_to_units' matrices"""
    g, v = {0: u.bias}, {0: u.value_bias}
    for p, gw, vw in zip(o.incoming, u.weights, u.value_weights):
        if gw:
            g[p.index + 1] = g.get(p.index + 1, 0) + gw
        if vw:
            v[p.index + 1] = v.get(p.index + 1, 0) + vw
    g = {i: w for i, w in g.items() if w != 0}
    v = {i: w for i, w in v.items() if w != 0}
    if not v or (list(g) in ([], [0]) and g.get(0, 0) <= 0):
        return None
    c = v[min(v)]  # values up to scale
    return (tuple(sorted(g.items())),
            tuple(sorted((i, round(w / c, 12)) for i, w in v.items())))

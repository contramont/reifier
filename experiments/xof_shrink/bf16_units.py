"""bfloat16-exact weights by function-preserving rescalings (bf16 keeps 8 significant bits).

A circuit whose weights are all bf16-representable computes exactly the same numbers with its
weights rounded to bf16 (mode w16 of audit/bf16_check.py) as in float32. The SwiGLU scales are
powers of 2 (c*q = 32, v = 4/q = 1/2, 1/(c*q*v) = 1/16, step offsets 1/2 -+ 1/(2c)), so only the
unit coefficients and the unit-sharing ratios decide representability.

A gated unit h adds outs[:, h] * max(0, g . x) * (v . x) to the layer's outputs. For alpha > 0 and
f != 0, the unit (alpha g, v f / alpha, outs / f) computes the same function, and the gate is only
sharpened when alpha >= 1. Representability is invariant under powers of 2, so alpha and f are
taken in [1, 2).

rescale_units   per unit, the alpha and f that make its weights representable when some choice
                does: unit-sharing ratios such as 3/7 (u1) or 1/3 (sp, Walsh), constant units
                shared by outputs with constants 7/2, 4, 3, forms written with a 1/3 on one side
                (lazy4c), the p1d forms' slopes 3 + 2 sqrt 2 and 3 - 2 sqrt 2 (their product is 1).
                Out columns that stay non-representable (sums of many constants, the pool theta's
                product 40/3) are split: hidden units with the same gate and the value / K^j carry
                the lower bits (levels = 2: 24 bits, 2 extra units each; levels = 1: 16 bits).
                Units whose gate and value slopes multiply to a non-dyadic number (MINPAR7's 25/6,
                MIN_PARITY, TH1S, the F13 pool) have no exact choice and keep their weights.
split_bias      constant features BOS / K, ..., BOS / K^levels in the input of every layer after
                the first (the layer before emits them from its BOS unit), and each gate or value
                bias b written as sum_j p_j BOS / K^j with bf16 parts p_j: 24 bits for levels = 2,
                enough for the irrational knots of the p1d forms. Layer 1 reads the message and has
                no such feature, so forms with irrational knots on raw bits stay inexact.

Enabled in xofbench.build_layers by the environment variable XOF_BF16: "u" (rescale_units), "ub"
(both; rescale_units then leaves the biases to split_bias), "ub1" (out columns at one level).
"""

import math
from fractions import Fraction

import numpy as np
import torch as t

SNAP = 1e-6  # entries within this relative distance of a bf16 number are float noise: snapped


def bf16(x: np.ndarray) -> np.ndarray:
    """round to bfloat16 (nearest even) through float32, as torch does"""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).astype(np.float64)


def mant(x: float) -> float:
    """x scaled by a power of 2 into [1, 2)"""
    m, e = math.frexp(abs(x))
    return 2 * m


def snap(x: np.ndarray) -> np.ndarray:
    b = bf16(x)
    ok = np.abs(b - x) <= SNAP * np.abs(x)
    return np.where(ok, b, x)


def n_bad(x: np.ndarray) -> tuple[int, float]:
    """(non-representable entries after snapping, max relative rounding error)"""
    if x.size == 0:
        return 0, 0.0
    s = snap(x)
    b = bf16(s)
    rel = np.abs(b - s) / np.maximum(np.abs(s), 1e-300)
    return int((b != s).sum()), float(rel.max())


_ODD = sorted({mant(p / q) for p in range(1, 16, 2) for q in range(1, 16, 2)})


def _candidates(xs: np.ndarray) -> list[float]:
    c = {1.0}
    for x in np.unique(np.round(np.abs(xs), 12))[:16]:
        if x:
            c.add(mant(1.0 / x))
    for x in _ODD:
        c.add(x)
    return sorted(c)


def choose(g: np.ndarray, v: np.ndarray, o: np.ndarray, gc=None, vc=None) -> tuple[float, float, int]:
    """alpha, f minimizing the non-representable entries of (alpha g, v f / alpha, o / f);
    gc, vc: entries that suggest candidates (the rows with their biases)"""
    gc = g if gc is None else gc
    vc = v if vc is None else vc
    best = None
    for a in _candidates(gc):
        ga, ge = n_bad(g * a)
        if best is not None and ga > best[0]:
            continue
        fs = set(_candidates(vc / a))
        fs |= {mant(x) for x in np.unique(np.round(np.abs(o), 12))[:16] if x}
        va = v / a
        for f in sorted(fs):
            vb, ve = n_bad(va * f)
            ob, oe = n_bad(o / f)
            key = (ga + vb + ob, max(ge, ve, oe), abs(a - 1) + abs(f - 1))
            if best is None or key < best[:3]:
                best = (*key, a, f)
    return best[3], best[4], best[0]


def rescale_units(units, with_bias: bool = True, cache: dict | None = None, split_outs: bool = True,
                  K: float = 256.0, levels: int = 2):
    """units = (gates, values, outs) of Matrices.layer_to_units; returns rescaled copies and the
    number of units left with non-representable weights. split_outs: out columns that stay
    non-representable get a second hidden unit for their low bits (one more unit each)"""
    gates, values, outs = (u.double().numpy().copy() for u in units)
    cache = {} if cache is None else cache
    left = 0
    extra = []
    lo = lo_ = 0 if with_bias else 1
    for k in range(gates.shape[0]):
        g, v, o = gates[k, lo:], values[k, lo:], outs[:, k]
        gi, vi, oi = g[g != 0], v[v != 0], o[o != 0]
        if n_bad(gi)[0] + n_bad(vi)[0] + n_bad(oi)[0] == 0:
            gates[k], values[k], outs[:, k] = snap(gates[k]), snap(values[k]), snap(outs[:, k])
            continue
        gc, vc = gates[k][gates[k] != 0], values[k][values[k] != 0]
        key = tuple(tuple(np.round(np.abs(x), 9)) for x in (gi, vi, oi, gc, vc))
        if key not in cache:
            cache[key] = choose(gi, vi, oi, gc, vc)
        a, f, bad = cache[key]
        gates[k] = snap(gates[k] * a)
        values[k] = snap(values[k] * f / a)
        outs[:, k] = snap(outs[:, k] / f)
        if split_outs and n_bad(outs[:, k][outs[:, k] != 0])[0]:
            # out weights with more than 8 bits (e.g. a constant unit shared by outputs whose
            # constants are sums of many forms' constants): hidden units with the same gate and
            # the value / K^j carry the lower parts, o = sum_j p_j / K^j (24 bits with 2 levels;
            # 16 bits leave 2.6e-4 after depth-low's d3c1 layer 1, 2e-2 at its output)
            r = outs[:, k].copy()
            for j in range(levels + 1):
                p_ = bf16(r * K ** j)
                r = r - p_ / K ** j
                if j == 0:
                    outs[:, k] = p_
                elif np.any(p_):
                    extra.append((gates[k].copy(), values[k] / K ** j, p_))
        g2, v2, o2 = gates[k][lo_:], values[k][lo_:], outs[:, k]
        left += (n_bad(g2[g2 != 0])[0] + n_bad(v2[v2 != 0])[0] + n_bad(o2[o2 != 0])[0]) > 0
    if extra:
        gates = np.concatenate([gates, np.stack([e[0] for e in extra])])
        values = np.concatenate([values, np.stack([e[1] for e in extra])])
        outs = np.concatenate([outs, np.stack([e[2] for e in extra], 1)], 1)
    return tuple(t.from_numpy(x).float() for x in (gates, values, outs)), left


def _dense(w):
    return w.to_dense() if w.is_sparse else w


def split_bias(layers, K: float = 256.0, levels: int = 2):
    """adds the constant features BOS / K, ..., BOS / K^m (m <= levels, as many as its biases need)
    to the input of every layer after the first whose gate or value biases are not representable
    (the layer before emits them from its BOS unit's out row), and writes each bias b as
    sum_j p_j BOS / K^j with bf16 parts p_j: 8 (levels + 1) significant bits.
    One level (16 bits) is not enough: biases of the top-core forms reach ~150 gate units at the
    top of wide count ranges, and 2^-17 of that moves a knot by ~1e-4 (2.7e-2 at the output of
    depth-low's d4x). Two levels: 24 bits."""
    for i in range(1, len(layers)):
        L, P = layers[i], layers[i - 1]
        split = {}
        need = 0
        for name in ("wg", "wv"):
            w = _dense(L[name]).clone()
            r = w[:, 0].double()
            parts = []
            for j in range(levels + 1):
                p_ = (r * K ** j).float().to(t.bfloat16).double()
                parts.append(p_.float())
                r = r - p_ / K ** j
                if j and bool((p_ != 0).any()):
                    need = max(need, j)
            split[name] = (w, parts)
        if need == 0:
            continue
        for name, (w, parts) in split.items():
            w[:, 0] = parts[0]
            L[name] = t.cat([w] + [p_[:, None] for p_ in parts[1:need + 1]], 1).to_sparse()
        L["norm"] = t.cat([L["norm"], t.ones(need)])
        wo = _dense(P["wo"])
        P["wo"] = t.cat([wo] + [wo[:1] / K ** j for j in range(1, need + 1)], 0).to_sparse()
        P["out"] += need
        L["in"] += need
    for L in layers:
        ps = [_dense(L[n]) for n in ("wg", "wv", "wo")] + [L["norm"]]
        L["dense"] = sum(p.numel() for p in ps)
        L["sparse"] = sum(int(t.count_nonzero(p)) for p in ps)
    return layers

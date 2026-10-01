"""Graph compiler: level the traced gate graph itself, with general optimization passes.

The tree compiler lays gates out by the call tree of the traced function, so it keeps
traced but unused code and adds layers where call blocks meet. This compiler instead
walks back from the outputs (so dead code is never compiled), folds constants, runs the
passes chosen in `GraphOptions` and levels the remaining graph. It returns a
`LeveledGraph` for `Matrices.from_graph`, like the tree compiler. The levels' recipes
choose its passes (recipes.RECIPES).

Passes (all exact on 0/1 inputs). A unit is "E-exact" if at every reachable input its
gate is <= -1/2 or 0, or its value is 0, or both are powers of 2: bfloat16 with float32
accumulation then computes it exactly relative to BOS. It is "flat" if its output does
not change to first order with its inputs' errors (gate <= -1, or 1 with value 1).
- fold (always): copies and NOTs of a bit fold into their readers' weights, so they cost
  nothing. An output that is a copy or NOT of a gate read by nothing else becomes that
  gate (negated for a NOT), which removes the output layer. Steps stay steps.
- dedup (always): nodes with the same function of the same parents merge.
- cheap: a step whose integer sum z = w.x + b + 1 is at most 1 (copy, NOT, AND, inhib)
  becomes one unit max(0, 2z-1)(3-2z); one whose sum is at least 0 (OR) becomes
  BOS - max(0, 1-2z)(1+2z), sharing the BOS unit. Copies that carry a bit to a later
  layer become the same unit. E-exact and flat (an error d on a 1 becomes 4d^2).
- sumfuse: a gate that reads only threshold gates of one weighted sum s = w.x (e.g. a
  threshold xor, 2 layers) becomes one layer of units on s: "all" uses
  sum_k max(0, s-k)(a_k + b_k(s-k)), ceil(range/2) units (glu_xor's form for parity),
  exact in float32 only; "e" keeps only E-exact results (xor2), and with xortree turns
  parities into trees of E-exact xor<=4 units (_xor_units), or with flat_xor into trees
  of flat xor<=4 units (_xor_units_flat).
- cone: a node whose function on a cut of <= 3 bits deeper down can be synthesized
  (_synth: 1-2 units, e.g. chi, maj, mux, xor3; _synth_flat: sums of flat subcube
  indicators) moves to right after the cut. Cones whose inner nodes are read elsewhere
  are not fused, so no logic is duplicated.
- reclean: units that are not flat pass errors on with gains up to ~2, so a level of
  flat copies follows every `reclean` of them in a row.
- lead_clean: a first level of flat copies of the inputs.
- clean_outputs: if an output is made by gated units, a last level of step copies of the
  outputs. Steps absorb the units' rounding, and exact readout
  (build.MLPOptions.exact_readout) then makes their 1s exactly BOS; unit rows cannot be
  flipped, and under split reductions or a host's perturbations their 1s miss boolify
  by a few hundredths.
- schedule (always): levels as soon as possible or as late as possible, whichever needs
  fewer copies, then a local search that moves nodes to cut copies.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np

from reifier.neurons.core import Bit, GluNeuron, Unit
from reifier.compile.levels import LeveledGraph, Level, Origin, Parent
from reifier.compile.monitor import find


@dataclass(frozen=True)
class GraphOptions:
    # > 0: fold a NOT into a gate only while the gate's bias stays small: |b| + 1 <= this,
    # or no larger than before. Every folded NOT adds its weight to the bias, and large
    # biases (BOS weights) are what bf16 reduced-precision reductions round (0: no bound)
    fold_bias: int = 0
    cheap: bool = False  # 1 gated unit for gates with z <= 1 or z >= 0, and for copies
    cheap_max: int = 0  # > 0: only gates of at most this many inputs become cheap units
    # False: the outputs (and their last copies) stay steps, which exact readout builds as
    # exactly BOS for 1s; a cheap unit gives 1 - 4d^2 and first-order errors from noise
    # on BOS, so 1s can leave boolify's 0.02 window under noise
    cheap_out: bool = True
    sumfuse: str = "off"  # "off", "e" (only E-exact fusions, i.e. xor2), "all"
    cone: int = 0  # fuse cones of up to this many leaves (<= 4) into one layer of units
    cone_units: int = 2  # at most this many units per fused cone (2 only for <= 3 leaves)
    cone_exact: bool = True  # only fusions whose units meet the E-rule (bf16-exact)
    cone_flat: bool = False  # only flat units: sums of subcube indicators (cheap units)
    xortree: bool = False  # sumfuse "e": parities of 3+ bits as trees of E-exact xor<=4
    # sumfuse "e" with xortree: parities of 2+ bits as trees of flat xor<=3 with a root of
    # <= 4 (sums of vertex indicators, 2^(k-1) cheap units each); no other sum fusions
    flat_xor: bool = False
    # units that are not flat at their inputs (xor, chi, synthesized cones) pass errors
    # on with gains up to ~2, so after this many of them in a row every feature gets a
    # flat copy max(0, 2x-1)(3-2x), which maps an error d to 4d^2 (0: never)
    reclean: int = 0
    # a first level of flat copies of the inputs: it takes the host's perturbations of the
    # input (e.g. a LayerNorm's mean shift), which later layers can cancel (always-0
    # features, see build.MLPOptions.ln_invariant)
    lead_clean: bool = False
    # if an output is made by gated units (cheap, xor, cone), a last level of step copies
    # of the outputs: steps absorb the units' rounding, and with exact_readout their 1s
    # are exactly BOS (unit rows cannot be flipped)
    clean_outputs: bool = False
    # with reclean: count flat cone units like the non-flat ones, so a level of flat
    # copies also follows cone layers (their outputs keep a split reduction's rounding,
    # which E-exact xor units reading them amplify)
    cone_reclean: bool = False


class _Node:
    __slots__ = ("kind", "ps", "w", "b", "units", "numeric", "flat")

    def __init__(self, kind: str, ps=(), w=(), b=0, units=(), numeric=False, flat=None):
        self.kind = kind  # "in", "gate" or "glu"
        self.ps: list[int] = list(ps)  # parent node indices (distinct)
        self.w: list[int] = list(w)  # gate weights, aligned with ps
        self.b = b  # gate bias: out = [w.x + b >= 0]
        # glu units, each (gate weights, gate bias, value weights, value bias)
        self.units: list[tuple[list, Any, list, Any]] = [
            (list(g), gb, list(v), vb) for g, gb, v, vb in units
        ]
        self.numeric = numeric  # glu whose output may be any number (a count)
        # flat: output errors do not grow with input errors (steps, cheap units)
        self.flat = (kind != "glu") if flat is None else flat

    def copy(self) -> "_Node":
        return _Node(self.kind, self.ps, self.w, self.b, self.units, self.numeric, self.flat)


def _merge(ps, cols):
    """merge repeated parents: ps and parallel coefficient lists -> (ps, cols)"""
    pos: dict[int, int] = {}
    out_ps: list[int] = []
    out_cols: list[list] = [[] for _ in cols]
    for i, p in enumerate(ps):
        if p in pos:
            j = pos[p]
            for c, oc in zip(cols, out_cols):
                oc[j] += c[i]
        else:
            pos[p] = len(out_ps)
            out_ps.append(p)
            for c, oc in zip(cols, out_cols):
                oc.append(c[i])
    return out_ps, out_cols


def _num(x):
    """ints stay ints (gate weights must be integers); floats that are integral too"""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


class GraphCompiler:
    def __init__(self, options: GraphOptions | None = None):
        self.opt = options if options is not None else GraphOptions()

    # ---------------- tracing ----------------

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> LeveledGraph:
        out = fn(*args, **kwargs)
        inputs = [b for b, _ in find(args + tuple(kwargs.values()), Bit)]
        outputs = [b for b, _ in find(out, Bit)]
        return self.compile(inputs, outputs)

    def compile(self, inputs: list[Bit], outputs: list[Bit]) -> LeveledGraph:
        self._build(inputs, outputs)
        o = self.opt
        self._fold()
        self._dedup()
        if o.sumfuse != "off":
            self._sumfuse("e")
            self._dedup()
        self._fold_outputs()
        if o.cone:
            self._conefuse(o.cone, o.cone_units, o.cone_exact, o.cone_flat)
            self._dedup()
        if o.sumfuse == "all":
            self._sumfuse("all")
            self._dedup()
        if o.cheap:
            self._cheap()
        self._fix_outputs()
        self._dce()
        return self._level_and_emit()

    def _build(self, inputs: list[Bit], outputs: list[Bit]) -> None:
        nodes: list[_Node] = []
        idx: dict[int, int] = {}  # id(signal) -> node index
        const: dict[int, Any] = {}  # id(signal) -> value of an input-independent signal
        for s in inputs:
            if id(s) not in idx:  # a repeated input reads its first position
                idx[id(s)] = len(nodes)
                nodes.append(_Node("in"))
        self.n_in = len(nodes)
        self.input_cols = [idx[id(s)] for s in inputs]
        stack: list[tuple[Bit, bool]] = [(s, False) for s in reversed(outputs)]
        while stack:
            s, ready = stack.pop()
            sid = id(s)
            if sid in idx or sid in const:
                continue
            src = s.source
            if not ready:
                stack.append((s, True))
                for p in src.incoming:
                    if id(p) not in idx and id(p) not in const:
                        stack.append((p, False))
                continue
            live_ps = [p for p in src.incoming if id(p) in idx]
            if not live_ps:
                const[sid] = _num(s.activation)
                continue
            if isinstance(src, GluNeuron):
                units = []
                for u in src.units:
                    gb, vb = u.bias, u.value_bias
                    gw: list = []
                    vw: list = []
                    ps: list[int] = []
                    for p, g, v in zip(src.incoming, u.weights, u.value_weights):
                        if id(p) in const:
                            gb += g * const[id(p)]
                            vb += v * const[id(p)]
                        else:
                            ps.append(idx[id(p)])
                            gw.append(g)
                            vw.append(v)
                    units.append((ps, gw, gb, vw, vb))
                ps0 = units[0][0]
                ps, cols = _merge(ps0, [c for u in units for c in (u[1], u[3])])
                us = [
                    (cols[2 * k], _num(u[2]), cols[2 * k + 1], _num(u[4]))
                    for k, u in enumerate(units)
                ]
                numeric = not isinstance(s.activation, bool)
                node = _Node("glu", ps, units=us, numeric=numeric)
            else:
                b = src.bias
                ps, ws = [], []
                for p, w in zip(src.incoming, src.weights):
                    if id(p) in const:
                        b += w * const[id(p)]
                    else:
                        ps.append(idx[id(p)])
                        ws.append(w)
                ps, (ws,) = _merge(ps, [ws])
                node = _Node("gate", ps, ws, _num(b))
            idx[sid] = len(nodes)
            nodes.append(node)
        self.nodes = nodes
        # outputs: node index, or ("const", value)
        self.outs: list[Any] = [
            idx[id(s)] if id(s) in idx else ("const", const[id(s)]) for s in outputs
        ]

    # ---------------- helpers ----------------

    def _is_bit(self, n: int) -> bool:
        node = self.nodes[n]
        return node.kind in ("in", "gate") or not node.numeric

    def _eval(self, node: _Node, xs: list) -> Any:
        if node.kind == "gate":
            return int(sum(w * x for w, x in zip(node.w, xs)) + node.b >= 0)
        total = 0
        for g, gb, v, vb in node.units:
            gv = sum(a * x for a, x in zip(g, xs)) + gb
            if gv > 0:
                total += gv * (sum(a * x for a, x in zip(v, xs)) + vb)
        return total

    def _live(self) -> list[bool]:
        """nodes that an output depends on"""
        live = [False] * len(self.nodes)
        stack = [o for o in self.outs if isinstance(o, int)]
        while stack:
            i = stack.pop()
            if live[i]:
                continue
            live[i] = True
            stack.extend(p for p in self.nodes[i].ps if not live[p])
        return live

    def _readers(self) -> list[list[int]]:
        """live readers of every node"""
        live = self._live()
        readers: list[list[int]] = [[] for _ in self.nodes]
        for i, node in enumerate(self.nodes):
            if live[i]:
                for p in node.ps:
                    readers[p].append(i)
        return readers

    def _substitute(self, node: _Node, sub: dict[int, tuple[int, Any, Any]]) -> None:
        """replace parents p = a + b*q (sub[p] = (q, a, b)) in node's weights"""
        if not any(p in sub for p in node.ps):
            return
        ps: list[int] = []
        if node.kind == "gate":
            ws: list = []
            b = node.b
            for p, w in zip(node.ps, node.w):
                if p in sub:
                    q, a, k = sub[p]
                    b += w * a
                    if q >= 0:
                        ps.append(q)
                        ws.append(w * k)
                else:
                    ps.append(p)
                    ws.append(w)
            node.ps, (node.w,) = _merge(ps, [ws])
            node.b = _num(b)
            return
        cols: list[list] = [[] for _ in range(2 * len(node.units))]
        biases = [[gb, vb] for _, gb, _, vb in node.units]
        for i, p in enumerate(node.ps):
            if p in sub:
                q, a, k = sub[p]
                for u, (g, _, v, _) in enumerate(node.units):
                    biases[u][0] += g[i] * a
                    biases[u][1] += v[i] * a
                if q < 0:
                    continue
                ps.append(q)
                for u, (g, _, v, _) in enumerate(node.units):
                    cols[2 * u].append(g[i] * k)
                    cols[2 * u + 1].append(v[i] * k)
            else:
                ps.append(p)
                for u, (g, _, v, _) in enumerate(node.units):
                    cols[2 * u].append(g[i])
                    cols[2 * u + 1].append(v[i])
        node.ps, cols = _merge(ps, cols)
        node.units = [
            (cols[2 * u], _num(biases[u][0]), cols[2 * u + 1], _num(biases[u][1]))
            for u in range(len(node.units))
        ]

    def _constant_of(self, node: _Node) -> Any:
        """the value of a node that no longer depends on anything, else None"""
        if node.kind == "in" or node.ps:
            return None
        return self._eval(node, [])

    # ---------------- passes ----------------

    def _fold(self) -> None:
        """copies, NOTs and constants of one bit fold into their readers"""
        sub: dict[int, tuple[int, Any, Any]] = {}  # node -> (q, a, b): a + b*q; q=-1: const
        bound = self.opt.fold_bias
        for i, node in enumerate(self.nodes):
            if node.kind == "in":
                continue
            if bound and node.kind == "gate" and any(p in sub for p in node.ps):
                self._substitute(node, self._bounded(node, sub, bound))
            else:
                self._substitute(node, sub)
            c = self._constant_of(node)
            if c is not None:
                sub[i] = (-1, c, 0)
                continue
            if node.numeric or len(node.ps) != 1 or not self._is_bit(node.ps[0]):
                continue
            f0, f1 = self._eval(node, [0]), self._eval(node, [1])
            if node.kind == "gate" or (f0 in (0, 1) and f1 in (0, 1)):
                if f0 == f1:
                    sub[i] = (-1, f0, 0)
                else:
                    sub[i] = (node.ps[0], f0, f1 - f0)
        # outputs that became constants or copies/NOTs keep their node (see _fold_outputs)
        self.outs = [
            ("const", sub[o][1]) if isinstance(o, int) and o in sub and sub[o][0] < 0 else o
            for o in self.outs
        ]

    def _bounded(self, node: _Node, sub: dict, bound: int) -> dict:
        """the substitutions for a gate that keep its bias small: copies and constants
        always, NOTs (b = -1) in turn while |bias| + 1 <= max(bound, |bias before| + 1)"""
        out: dict = {}
        b = node.b
        limit = max(bound, abs(b) + 1)
        for p, w in zip(node.ps, node.w):
            if p not in sub:
                continue
            q, a, k = sub[p]
            if q >= 0 and k == -1:  # a NOT: w * (1 - q) adds w to the bias
                if abs(b + w * a) + 1 > limit:
                    continue
            b += w * a
            out[p] = sub[p]
        return out

    def _dedup(self) -> None:
        rep: dict[int, int] = {}
        seen: dict[tuple, int] = {}
        for i, node in enumerate(self.nodes):
            if node.kind == "in":
                continue
            if any(p in rep for p in node.ps):
                self._substitute(node, {p: (rep[p], 0, 1) for p in node.ps if p in rep})
            order = sorted(range(len(node.ps)), key=node.ps.__getitem__)
            ps = tuple(node.ps[j] for j in order)
            if node.kind == "gate":
                key = ("g", ps, tuple(node.w[j] for j in order), node.b)
            else:
                key = (
                    "u",
                    ps,
                    node.numeric,
                    tuple(
                        (tuple(g[j] for j in order), gb, tuple(v[j] for j in order), vb)
                        for g, gb, v, vb in node.units
                    ),
                )
            if key in seen:
                rep[i] = seen[key]
            else:
                seen[key] = i
        self.outs = [rep.get(o, o) if isinstance(o, int) else o for o in self.outs]

    def _bounds(self, ws, b) -> tuple[Any, Any]:
        """range of w.x + b over bit parents"""
        lo = b + sum(min(0, w) for w in ws)
        hi = b + sum(max(0, w) for w in ws)
        return lo, hi

    def _sumfuse(self, mode: str) -> None:
        self._needs_renumber = False
        self._sumfuse_nodes(mode)
        if self._needs_renumber:
            self._dce()

    def _sumfuse_nodes(self, mode: str) -> None:
        """a gate reading only threshold gates of one sum s becomes units on s"""
        readers = self._readers()
        for i, node in enumerate(self.nodes):
            if node.kind != "gate" or len(node.ps) < 1:
                continue
            cs = [self.nodes[p] for p in node.ps]
            c0 = cs[0]
            if c0.kind != "gate" or not c0.ps:
                continue
            if not all(self._is_bit(p) for p in c0.ps):
                continue
            key = sorted(zip(c0.ps, c0.w))
            if any(c.kind != "gate" or sorted(zip(c.ps, c.w)) != key for c in cs):
                continue
            lo, hi = self._bounds(c0.w, 0)
            if not (isinstance(lo, int) and isinstance(hi, int)) or hi - lo > 4096:
                continue
            F = [
                int(sum(w * int(s + c.b >= 0) for w, c in zip(node.w, cs)) + node.b >= 0)
                for s in range(lo, hi + 1)
            ]
            n = len(c0.ps)
            flip = [c for c in (0, 1) if all(f == (s + c) % 2 for s, f in zip(range(lo, hi + 1), F))]
            flat = self.opt.flat_xor and mode == "e"
            if mode == "e" and self.opt.xortree and flip and n >= (2 if flat else 3):
                if all(w in (1, -1) for w in c0.w):
                    self._xor_tree(i, list(c0.ps), flip[0], flat)
                    continue
            if flat:
                continue  # only flat forms
            units = _sum_units(F, lo)
            if mode == "e" and not _e_exact(units, lo, hi):
                continue
            # hidden units removed: the gate's 2, and 2 per counter read by nothing else
            freed = 2 + sum(2 for p in node.ps if readers[p] == [i])
            if len(units) > freed:
                continue
            ps, ws = c0.ps, c0.w
            new_units = []
            for k, a, bb in units:  # max(0, s-k) * (a + bb(s-k)), s = ws.x
                g = list(ws)
                v = [bb * w for w in ws]
                new_units.append((g, -k, v, _num(a - bb * k)))
            self.nodes[i] = _Node("glu", ps, units=new_units)

    def _conefuse(self, K: int, max_units: int, exact: bool, flat: bool = False) -> None:
        """a node whose function on a cut of <= K bits deeper down is synthesized as
        one layer of units (e.g. chi, mux, maj, xor3) moves up to right after the cut.
        Only cones whose inner nodes nothing else reads, so no logic is duplicated"""
        nodes = self.nodes
        readers = self._readers()
        depth = [0] * len(nodes)
        wide = 2 * K + 2  # cuts may grow this wide on the way to a narrow one
        for i, node in enumerate(nodes):
            if node.kind == "in":
                continue
            depth[i] = 1 + max((depth[p] for p in node.ps), default=0)
            if node.numeric or not node.ps or len(node.ps) > wide:
                continue
            if not all(self._is_bit(p) for p in node.ps):
                continue
            cuts = []  # (cut, inner nodes)
            start = (frozenset(node.ps), frozenset([i]))
            seen = {start[0]}
            frontier = [start]
            while frontier and len(seen) < 256:
                nxt = []
                for c, inner in frontier:
                    for p in c:
                        pn = nodes[p]
                        if pn.kind == "in" or pn.numeric or not pn.ps:
                            continue
                        if not all(self._is_bit(q) for q in pn.ps):
                            continue
                        if any(r not in inner for r in readers[p]):
                            continue  # p is read outside the cone: it would be duplicated
                        new = (c - {p}) | frozenset(pn.ps)
                        if len(new) <= wide and new not in seen:
                            seen.add(new)
                            nxt.append((new, inner | {p}))
                            if len(new) <= K:
                                cuts.append(new)
                frontier = nxt
            if not cuts:
                continue
            cuts.sort(key=lambda c: (1 + max(depth[p] for p in c), len(c)))
            for c in cuts:
                d = 1 + max(depth[p] for p in c)
                if d >= depth[i]:
                    break
                leaves = sorted(c)
                T = self._truth_table(i, leaves)
                if flat:
                    units = _synth_flat(tuple(T), len(leaves), max_units, exact)
                else:
                    units = _synth(tuple(T), len(leaves), max_units, exact)
                if units is None:
                    continue
                for p in node.ps:  # the old parents lose this reader
                    if i in readers[p]:
                        readers[p].remove(i)
                nodes[i] = _Node("glu", leaves, units=units,
                                 flat=flat and not self.opt.cone_reclean)
                for p in leaves:
                    readers[p].append(i)
                depth[i] = d
                break

    def _truth_table(self, i: int, leaves: list[int]) -> list[int]:
        cone: set[int] = set()
        stack = [i]
        lset = set(leaves)
        while stack:
            n = stack.pop()
            if n in cone or n in lset:
                continue
            cone.add(n)
            stack.extend(self.nodes[n].ps)
        order = sorted(cone)
        T = []
        for a in range(2 ** len(leaves)):
            val = {leaf: (a >> j) & 1 for j, leaf in enumerate(leaves)}
            for n in order:
                node = self.nodes[n]
                v = self._eval(node, [val[p] for p in node.ps])
                val[n] = round(v) if self._is_bit(n) else v
            T.append(int(val[i]))
        return T

    def _xor_tree(self, i: int, bits: list[int], flip: int, flat: bool = False) -> None:
        """node i := xor(bits) ^ flip, as a tree of E-exact xor units of <= 4 bits, or
        with flat, of flat xor units of <= 3 bits"""
        width = 3 if flat else 4
        make = _xor_units_flat if flat else _xor_units
        level = bits
        while True:
            k = -(-len(level) // width)  # groups of <= width, sizes as equal as possible
            if len(level) <= 4:
                k = 1  # the root takes up to 4 bits
            groups = [level[g::k] for g in range(k)]
            if k == 1:
                units = make(len(level))
                if flip:  # 1 - xor
                    n = len(level)
                    units = [(g, gb, [-x for x in v], -vb) for g, gb, v, vb in units]
                    units.append(([0] * n, 1, [0] * n, 1))
                self.nodes[i] = _Node("glu", level, units=units, flat=flat)
                break
            nxt = []
            for grp in groups:
                if len(grp) == 1:
                    nxt.append(grp[0])
                    continue
                nxt.append(len(self.nodes))
                self.nodes.append(_Node("glu", grp, units=make(len(grp)), flat=flat))
            level = nxt
        self._needs_renumber = True

    def _fold_outputs(self) -> None:
        """an output copy/NOT of a gate that nothing else reads becomes that gate"""
        readers = self._readers()
        out_count: dict[int, int] = {}
        for o in self.outs:
            if isinstance(o, int):
                out_count[o] = out_count.get(o, 0) + 1
        for j, o in enumerate(self.outs):
            if not isinstance(o, int):
                continue
            node = self.nodes[o]
            if node.kind == "in" or node.numeric or len(node.ps) != 1:
                continue
            q = node.ps[0]
            qn = self.nodes[q]
            if qn.kind == "in" or qn.numeric or readers[q] != [o] or q in out_count:
                continue
            if out_count[o] > 1:
                continue
            f0, f1 = self._eval(node, [0]), self._eval(node, [1])
            if (f0, f1) == (0, 1):
                new = qn.copy()
            elif (f0, f1) == (1, 0):
                new = qn.copy()
                if new.kind == "gate":  # [z >= 0] -> [-z - 1 >= 0] on integers
                    if not all(isinstance(w, int) for w in new.w + [new.b]):
                        continue
                    new.w = [-w for w in new.w]
                    new.b = -new.b - 1
                else:  # 1 - sum of units
                    n = len(new.ps)
                    new.units = [(g, gb, [-x for x in v], -vb) for g, gb, v, vb in new.units]
                    new.units.append(([0] * n, 1, [0] * n, 1))
            else:
                continue
            self.nodes[o] = new

    def _cheap(self) -> None:
        """gates whose sum is at most 1 (or at least 0) become one E-exact unit"""
        outs = set() if self.opt.cheap_out else {o for o in self.outs if isinstance(o, int)}
        for i, node in enumerate(self.nodes):
            if node.kind != "gate" or not node.ps or i in outs:
                continue
            if not all(self._is_bit(p) for p in node.ps):
                continue
            if not all(isinstance(w, int) for w in node.w + [node.b]):
                continue
            n = len(node.ps)
            if self.opt.cheap_max and n > self.opt.cheap_max:
                continue
            z0 = node.b + 1  # z = w.x + b + 1, out = [z >= 1]
            zlo, zhi = self._bounds(node.w, z0)
            if zhi <= 1:  # max(0, 2z-1)(3-2z)
                u = ([2 * w for w in node.w], 2 * z0 - 1, [-2 * w for w in node.w], 3 - 2 * z0)
                self.nodes[i] = _Node("glu", node.ps, units=[u], flat=True)
            elif zlo >= 0:  # 1 - max(0, 1-2z)(1+2z)
                u = ([-2 * w for w in node.w], 1 - 2 * z0, [-2 * w for w in node.w], -1 - 2 * z0)
                bos = ([0] * n, 1, [0] * n, 1)
                self.nodes[i] = _Node("glu", node.ps, units=[u, bos], flat=True)

    def _fix_outputs(self) -> None:
        """every output occurrence gets its own non-input node"""
        seen: set[int] = set()
        new_outs = []
        for o in self.outs:
            if not isinstance(o, int):
                val = int(o[1])
                node = _Node("gate", (), (), 0 if val else -1)  # constant row
            elif self.nodes[o].kind == "in" or o in seen:
                if self.opt.cheap and self.opt.cheap_out and self._is_bit(o):
                    u = ([2], -1, [-2], 3)
                    node = _Node("glu", [o], units=[u], flat=True)
                elif self._is_bit(o):
                    node = _Node("gate", [o], [1], -1)
                else:
                    node = _Node("glu", [o], units=[([0], 1, [1], 0)], numeric=True)
            else:
                seen.add(o)
                new_outs.append(o)
                continue
            new_outs.append(len(self.nodes))
            self.nodes.append(node)
        self.outs = new_outs

    def _dce(self) -> None:
        live = [False] * len(self.nodes)
        for i in range(self.n_in):
            live[i] = True
        stack = [o for o in self.outs if isinstance(o, int)]
        while stack:
            i = stack.pop()
            if live[i] and i >= self.n_in:
                continue
            live[i] = True
            stack.extend(p for p in self.nodes[i].ps if not live[p] or p < self.n_in)
        # renumber in topological order (parents have smaller indices, except appended
        # output nodes, which only read earlier nodes)
        order = self._topo([i for i in range(len(self.nodes)) if live[i]])
        new_index = {old: k for k, old in enumerate(order)}
        nodes = []
        for old in order:
            node = self.nodes[old]
            node.ps = [new_index[p] for p in node.ps]
            nodes.append(node)
        self.nodes = nodes
        self.outs = [new_index[o] if isinstance(o, int) else o for o in self.outs]

    def _topo(self, ids: list[int]) -> list[int]:
        idset = set(ids)
        indeg = {i: 0 for i in ids}
        readers: dict[int, list[int]] = {i: [] for i in ids}
        for i in ids:
            for p in self.nodes[i].ps:
                if p in idset:
                    indeg[i] += 1
                    readers[p].append(i)
        out = []
        heap = [i for i in ids if indeg[i] == 0]
        heapq.heapify(heap)
        while heap:
            i = heapq.heappop(heap)
            out.append(i)
            for r in readers[i]:
                indeg[r] -= 1
                if indeg[r] == 0:
                    heapq.heappush(heap, r)
        assert len(out) == len(ids), "cycle in the gate graph"
        return out

    # ---------------- leveling ----------------

    def _schedule(self, how: str) -> tuple[list[int], int]:
        nodes, n_in = self.nodes, self.n_in
        asap = [0] * len(nodes)
        for i, node in enumerate(nodes):
            if node.kind != "in":
                asap[i] = 1 + max((asap[p] for p in node.ps), default=0)
        outset = set(self.outs)
        L = max([asap[o] for o in self.outs] + [1])
        for o in self.outs:
            if not nodes[o].ps:
                asap[o] = L  # constant rows sit on the output level
        if how == "asap":
            return asap, L
        readers = self._readers()
        lvl = [0] * len(nodes)
        for i in reversed(range(len(nodes))):
            if i < n_in:
                continue
            cands = [lvl[r] - 1 for r in readers[i]]
            if i in outset:
                cands.append(L)
            lvl[i] = min(cands) if cands else L
        return lvl, L

    def _improve(self, lvl: list[int], L: int, passes: int = 8) -> list[int]:
        """local search: move each node within its window to the level that needs the
        fewest copies of it and of its parents; keep the best schedule found"""
        nodes, n_in = self.nodes, self.n_in
        readers = self._readers()
        outset = set(self.outs)
        best, best_cost = list(lvl), self._n_copies(lvl, L)
        for it in range(passes):
            order = range(n_in, len(nodes)) if it % 2 == 0 else reversed(range(n_in, len(nodes)))
            for i in order:
                node = nodes[i]
                if not node.ps:
                    continue
                lo = 1 + max(lvl[p] for p in node.ps)
                rl = [lvl[r] - 1 for r in readers[i]]
                if i in outset:
                    rl.append(L)
                hi = min(rl)
                if hi <= lo:
                    continue
                need_i = max(rl)
                others = []
                for p in node.ps:
                    o = [lvl[r] - 1 for r in readers[p] if r != i]
                    o.append(lvl[p])
                    if p in outset:
                        o.append(L)
                    others.append(max(o))

                def cost(x: int) -> int:
                    return (need_i - x) + sum(max(0, x - 1 - o) for o in others)

                cur = cost(lvl[i])
                for x in range(lo, hi + 1):
                    c = cost(x)
                    if c < cur:
                        cur, lvl[i] = c, x
            total = self._n_copies(lvl, L)
            if total < best_cost:
                best, best_cost = list(lvl), total
            else:
                break
        return best

    def _n_copies(self, lvl: list[int], L: int) -> int:
        need = list(lvl)
        for i, node in enumerate(self.nodes):
            for p in node.ps:
                need[p] = max(need[p], lvl[i] - 1)
        for o in self.outs:
            need[o] = max(need[o], L)
        return sum(need[i] - lvl[i] for i in range(len(self.nodes)))

    def _level_and_emit(self) -> LeveledGraph:
        cands = [self._schedule("asap"), self._schedule("alap")]
        lvl, L = min(cands, key=lambda c: self._n_copies(*c))
        lvl = self._improve(list(lvl), L)
        nodes = self.nodes
        need = list(lvl)  # last level at which a node's column must exist
        for i, node in enumerate(nodes):
            for p in node.ps:
                need[p] = max(need[p], lvl[i] - 1)
        for o in self.outs:
            need[o] = max(need[o], L)
        col: dict[tuple[int, int], int] = {}
        levels: list[list[Origin]] = [[] for _ in range(L + 1)]
        for i, c in enumerate(self.input_cols):
            levels[0].append(Origin(i, (), -1))
            col.setdefault((c, 0), i)
        # final level: exactly the outputs, in order
        at: list[list[tuple[str, int]]] = [[] for _ in range(L + 1)]
        for i in range(len(nodes)):
            if i >= self.n_in:
                at[lvl[i]].append(("node", i))
            for lv in range(lvl[i] + 1, need[i] + 1):
                at[lv].append(("copy", i))
        pos_out = {o: j for j, o in enumerate(self.outs)}
        out_levels = [levels[0]]
        if self.opt.lead_clean:
            out_levels.append([
                Origin(j, (Parent(j, 0),), -1, (Unit((2,), -1, (-2,), 3),))
                for j in range(len(levels[0]))
            ])
        age = [0] * len(levels[0])  # non-flat unit layers since the last flat one
        numeric_col = [not self._is_bit(c) for c in self.input_cols]
        R = self.opt.reclean
        for lv in range(1, L + 1):
            items = at[lv]
            if lv == L:
                items = sorted(items, key=lambda it: pos_out[it[1]])
                assert len(items) == len(self.outs)
            for j, (_, i) in enumerate(items):
                col[(i, lv)] = j
            row = []
            new_age = []
            for j, (kind, i) in enumerate(items):
                if kind == "copy":
                    src = col[(i, lv - 1)]
                    row.append(self._copy_origin(j, src, i, lv == L))
                    new_age.append(age[src] if not self._is_bit(i) else 0)
                    continue
                node = nodes[i]
                parents = [col[(p, lv - 1)] for p in node.ps]
                if node.kind == "gate":
                    row.append(Origin(j, tuple(map(Parent, parents, node.w)), node.b))
                else:
                    units = tuple(Unit(tuple(g), gb, tuple(v), vb) for g, gb, v, vb in node.units)
                    row.append(Origin(j, tuple(Parent(p, 0) for p in parents), -1, units))
                a = 1 + max((age[p] for p in parents), default=0)
                new_age.append(0 if node.flat else a)
            out_levels.append(row)
            age = new_age
            numeric_col = [not self._is_bit(i) for _, i in items]
            if R and lv < L - 1 and max(age, default=0) >= R:  # clean every feature
                clean = []
                for j, num in enumerate(numeric_col):
                    u = Unit((0,), 1, (1,), 0) if num else Unit((2,), -1, (-2,), 3)
                    clean.append(Origin(j, (Parent(j, 0),), -1, (u,)))
                out_levels.append(clean)
                age = [0] * len(clean)
        if self.opt.clean_outputs and any(nodes[o].kind == "glu" for o in self.outs):
            last = []
            for j, o in enumerate(self.outs):
                if self._is_bit(o):
                    last.append(Origin(j, (Parent(j, 1),), -1))  # step: [x >= 1]
                else:  # a count: passed on linearly
                    last.append(Origin(j, (Parent(j, 0),), -1, (Unit((0,), 1, (1,), 0),)))
            out_levels.append(last)
        return LeveledGraph(levels=tuple(Level(tuple(r)) for r in out_levels))

    def _copy_origin(self, j: int, src: int, i: int, last: bool = False) -> Origin:
        if not self._is_bit(i):  # a count: pass it on linearly, max(0, BOS) * x
            return Origin(j, (Parent(src, 0),), -1, (Unit((0,), 1, (1,), 0),))
        if self.opt.cheap and (self.opt.cheap_out or not last):  # max(0, 2x-1)(3-2x)
            return Origin(j, (Parent(src, 0),), -1, (Unit((2,), -1, (-2,), 3),))
        return Origin(j, (Parent(src, 1),), -1)


_SYNTH_CACHE: dict[tuple, Any] = {}


def _synth(T: tuple, K: int, max_units: int, exact: bool):
    """gated units on K bits whose sum is T[x] at every x (x as bits of the index): 1
    unit over integer gates with coefficients in [-2, 2], else 2 units with coefficients
    in [-1, 1]; values solved exactly and kept if dyadic (1/64). Cheapest by weight."""
    key = (T, K, max_units, exact)
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]
    X = np.array([[(a >> j) & 1 for j in range(K)] for a in range(2**K)], dtype=float)
    Xa = np.hstack([X, np.ones((2**K, 1))])
    Tv = np.array(T, dtype=float)

    def gates(r):
        out = []
        for w in itertools.product(r, repeat=K):
            for b in range(-K - 1, K + 2):
                g = np.array(list(w) + [b], dtype=float)
                G = Xa @ g
                if (G > 0).any():
                    out.append((g, G))
        return out

    def e_ok(g, v) -> bool:
        for Gi, Vi in zip(Xa @ g, Xa @ v):
            if Gi <= -0.5 or Gi == 0 or Vi == 0:
                continue
            if np.frexp(abs(Gi))[0] != 0.5 or np.frexp(abs(Vi))[0] != 0.5:
                return False
        return True

    result = None
    for nu in range(1, max_units + 1):
        if nu > 1 and K > 3:
            break
        cands = gates(range(-2, 3) if nu == 1 else range(-1, 2))
        best = None
        for combo in itertools.combinations(range(len(cands)), nu):
            M = np.hstack([np.maximum(cands[c][1], 0)[:, None] * Xa for c in combo])
            v, *_ = np.linalg.lstsq(M, Tv, rcond=None)
            if np.abs(M @ v - Tv).max() > 1e-9:
                continue
            v = np.round(v * 64) / 64
            if np.abs(M @ v - Tv).max() > 1e-12:
                continue
            us = [(cands[c][0], v[k * (K + 1) : (k + 1) * (K + 1)]) for k, c in enumerate(combo)]
            if exact and not all(e_ok(g, vv) for g, vv in us):
                continue
            cost = sum(np.abs(g).sum() + np.abs(vv).sum() for g, vv in us)
            if best is None or cost < best[0]:
                best = (cost, us)
        if best is not None:
            result = [
                ([_num(float(x)) for x in g[:K]], _num(float(g[K])),
                 [_num(float(x)) for x in vv[:K]], _num(float(vv[K])))
                for g, vv in best[1]
            ]
            break
    _SYNTH_CACHE[key] = result
    return result


def _xor_units(k: int) -> list:
    """xor of k <= 4 bits in one layer, every unit E-exact: on t = sum of the bits,
    k=2: max(0,t)(2-t); k=3: max(0,t)(3-t)/2 + max(0,t-1)(3t/2-4);
    k=4: max(0,t)(3-t)/2 + max(0,t-1)(t-4)/2 + max(0,t-2)(5-t)"""
    one = [1] * k
    if k == 2:
        forms = [(0, 2, -1)]  # (k0, a, b): max(0, t-k0)(a + b t)
    elif k == 3:
        forms = [(0, 1.5, -0.5), (1, -4, 1.5)]
    elif k == 4:
        forms = [(0, 1.5, -0.5), (1, -2, 0.5), (2, 5, -1)]
    else:
        raise ValueError(k)
    return [(one, -k0, [b * x for x in one], a) for k0, a, b in forms]


def _synth_flat(T: tuple, K: int, max_terms: int, exact: bool):
    """T as the fewest integer multiples of subcube indicators, each one flat unit
    c max(0, 2z-1)(3-2z) with z = (matching literals) - (fixed coordinates - 1);
    exact: every multiple c a power of 2 (E-rule)"""
    key = ("flat", T, K, max_terms, exact)
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]
    pts = [[(a >> j) & 1 for j in range(K)] for a in range(2**K)]
    cubes = list(itertools.product((None, 0, 1), repeat=K))
    ind = np.array(
        [[all(c is None or c == x for c, x in zip(cube, p)) for p in pts] for cube in cubes],
        dtype=float,
    )
    Tv = np.array(T, dtype=float)
    result = None
    for n in range(1, max_terms + 1):
        best = None
        for combo in itertools.combinations(range(len(cubes)), n):
            M = ind[list(combo)].T
            c, *_ = np.linalg.lstsq(M, Tv, rcond=None)
            c = np.round(c)
            if np.abs(M @ c - Tv).max() > 1e-9 or (c == 0).any():
                continue
            if exact and any(np.frexp(abs(x))[0] != 0.5 for x in c):
                continue
            cost = float(np.abs(c).sum())
            if best is None or cost < best[0]:
                best = (cost, combo, c)
        if best is not None:
            units = []
            for k, cf in zip(best[1], best[2]):
                cube, cf = cubes[k], _num(float(cf))
                fixed = [j for j, v in enumerate(cube) if v is not None]
                sign = [0] * K
                for j in fixed:
                    sign[j] = 1 if cube[j] == 1 else -1
                z0 = sum(1 for j in fixed if cube[j] == 0) - (len(fixed) - 1)
                units.append(([2 * x for x in sign], 2 * z0 - 1,
                              [-2 * cf * x for x in sign], cf * (3 - 2 * z0)))
            result = units
            break
    _SYNTH_CACHE[key] = result
    return result


def _xor_units_flat(k: int) -> list:
    """xor of k <= 4 bits as flat units: k=2: 1 - [00] - [11]; k=3, 4: the indicators
    of the odd vertices (4 or 8); each max(0, 2z-1)(3-2z), z = matches - (k-1)"""
    if k == 2:
        verts, c0, sign = [(0, 0), (1, 1)], 1, -1
    elif k in (3, 4):
        verts, c0, sign = [v for v in itertools.product((0, 1), repeat=k) if sum(v) % 2], 0, 1
    else:
        raise ValueError(k)
    units = []
    for v in verts:
        w = [1 if b else -1 for b in v]
        z0 = sum(1 for b in v if not b) - (k - 1)
        units.append(([2 * x for x in w], 2 * z0 - 1, [-2 * sign * x for x in w], sign * (3 - 2 * z0)))
    if c0:
        units.append(([0] * k, 1, [0] * k, c0))
    return units


def _sum_units(F: list[int], lo: int) -> list[tuple[int, Any, Any]]:
    """units max(0, s-k)(a + b(s-k)) with sum F(s) at every integer s in [lo, hi]:
    each unit starts where the residual is first nonzero and fits it there and one
    integer further"""
    R = [Fraction(f) for f in F]
    units = []
    j = 0
    n = len(R)
    while j < n:
        if R[j] == 0:
            j += 1
            continue
        k = lo + j - 1  # the unit's gate s-k is 1 at s = lo + j
        r0 = R[j]
        r1 = R[j + 1] if j + 1 < n else None
        if r1 is None:
            a, b = r0, Fraction(0)
        else:  # 1(a + b) = r0, 2(a + 2b) = r1
            b = r1 / 2 - r0
            a = r0 - b
        for t in range(j, n):
            g = t + 1 - j  # s - k
            R[t] -= g * (a + b * g)
        units.append((k, _frac(a), _frac(b)))
        j += 2
    return units


def _frac(x):
    return int(x) if x.denominator == 1 else float(x)


def _e_exact(units, lo: int, hi: int) -> bool:
    """E-rule of every unit at every integer s: gate <= -1/2, 0, value 0, or both
    powers of 2 (then bfloat16 computes the unit exactly relative to BOS)"""

    def pow2(x) -> bool:
        x = abs(x)
        if x == 0:
            return False
        while x >= 2:
            x /= 2
        while x < 1:
            x *= 2
        return x == 1

    for k, a, b in units:
        for s in range(lo, hi + 1):
            g = s - k
            v = a + b * g
            if g <= -0.5 or g == 0 or v == 0:
                continue
            if not (pow2(g) and pow2(v)):
                return False
    return True

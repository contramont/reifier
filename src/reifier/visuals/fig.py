"""Minimal SVG figure toolkit for academic publications.

Integer-valued coordinates. Primitives expose anchor points (top, bot,
left, right, center).  A Fig canvas collects elements and renders SVG + PDF.

    fig = Fig()
    r = fig.rect(0, 0, 16, 16, fill=BLUE)
    fig.line(r.bot, r.bot + P(0, 6))
    fig.text(r.center, "W", sub="o")
    fig.save("images/my_figure")

Stroke and stroke-width are stamped from Fig defaults onto each element.
Pass stroke=None to suppress (e.g. on decoration-only rects).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# ── Color palette ────────────────────────────────────────────────────
OUTLINE = "#4a4a6a"
FILL    = "#eeeef4"
BLUE    = "#b8ccee"
GREEN   = "#b8e0b0"
GOLD    = "#f0d8a0"
PINK    = "#f0c0c0"
RED     = "#c06060"
DGREEN  = "#609060"

# ── Fonts ────────────────────────────────────────────────────────────
SERIF = "Latin Modern Roman, CMU Serif, serif"
SANS  = "Helvetica, Arial, sans-serif"

_UNSET = object()  # sentinel: "use Fig default"


def _r(v: float) -> str:
    """Format a number: integer when exact, else ≤ 2 decimal places."""
    n = round(v, 2)
    return str(int(n)) if n == int(n) else str(n)


def _pts(ps: list[P]) -> str:
    return " ".join(f"{_r(p.x)},{_r(p.y)}" for p in ps)


def _fill(v: str | None) -> str:
    return "" if v is None else f' fill="{v}"'


def _stroke(s: str | None, w: float | None) -> str:
    if s is None:
        return ""
    if s == "none":
        return ' stroke="none"'
    return f' stroke="{s}" stroke-width="{_r(w or 0)}"'


# ── Point ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class P:
    """2D point with arithmetic."""
    x: float = 0
    y: float = 0

    def __add__(self, o: P | tuple[float, float]) -> P:
        if isinstance(o, tuple):
            return P(self.x + o[0], self.y + o[1])
        return P(self.x + o.x, self.y + o.y)

    def __sub__(self, o: P | tuple[float, float]) -> P:
        if isinstance(o, tuple):
            return P(self.x - o[0], self.y - o[1])
        return P(self.x - o.x, self.y - o.y)

    def __mul__(self, s: float) -> P:
        return P(self.x * s, self.y * s)

    def __neg__(self) -> P:
        return P(-self.x, -self.y)

    def rot90(self) -> P:
        return P(-self.y, self.x)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def unit(self) -> P:
        n = self.norm()
        return P(self.x / n, self.y / n) if n else P()

    def mirror_x(self, cx: float) -> P:
        return P(2 * cx - self.x, self.y)


# ── Primitives ───────────────────────────────────────────────────────
# fill/stroke default to None → omitted from SVG output.
# Fig convenience methods stamp the canvas defaults via _UNSET sentinel.

@dataclass
class Rect:
    x: float; y: float; w: float; h: float
    rx: float = 0
    fill: str | None = None
    stroke: str | None = None; stroke_w: float | None = None
    dash: str = ""

    @property
    def center(self) -> P: return P(self.x + self.w / 2, self.y + self.h / 2)
    @property
    def top(self) -> P: return P(self.x + self.w / 2, self.y)
    @property
    def bot(self) -> P: return P(self.x + self.w / 2, self.y + self.h)
    @property
    def left(self) -> P: return P(self.x, self.y + self.h / 2)
    @property
    def right(self) -> P: return P(self.x + self.w, self.y + self.h / 2)
    @property
    def tl(self) -> P: return P(self.x, self.y)
    @property
    def tr(self) -> P: return P(self.x + self.w, self.y)
    @property
    def bl(self) -> P: return P(self.x, self.y + self.h)
    @property
    def br(self) -> P: return P(self.x + self.w, self.y + self.h)

    def moved(self, d: P) -> Rect:
        return Rect(self.x + d.x, self.y + d.y, self.w, self.h, self.rx,
                     self.fill, self.stroke, self.stroke_w, self.dash)

    def svg(self) -> str:
        s = f'<rect x="{_r(self.x)}" y="{_r(self.y)}" width="{_r(self.w)}" height="{_r(self.h)}"'
        if self.rx: s += f' rx="{_r(self.rx)}"'
        s += _fill(self.fill) + _stroke(self.stroke, self.stroke_w)
        if self.dash: s += f' stroke-dasharray="{self.dash}"'
        return s + '/>'


@dataclass
class Circle:
    center: P; r: float
    fill: str | None = None
    stroke: str | None = None; stroke_w: float | None = None

    @property
    def top(self) -> P: return self.center - P(0, self.r)
    @property
    def bot(self) -> P: return self.center + P(0, self.r)
    @property
    def left(self) -> P: return self.center - P(self.r, 0)
    @property
    def right(self) -> P: return self.center + P(self.r, 0)

    def svg(self) -> str:
        s = f'<circle cx="{_r(self.center.x)}" cy="{_r(self.center.y)}" r="{_r(self.r)}"'
        return s + _fill(self.fill) + _stroke(self.stroke, self.stroke_w) + '/>'


@dataclass
class Line:
    p1: P; p2: P
    stroke: str | None = None; stroke_w: float | None = None

    def svg(self) -> str:
        return (f'<line x1="{_r(self.p1.x)}" y1="{_r(self.p1.y)}" '
                f'x2="{_r(self.p2.x)}" y2="{_r(self.p2.y)}"'
                + _stroke(self.stroke, self.stroke_w) + '/>')


@dataclass
class Polyline:
    ps: list[P]
    stroke: str | None = None; stroke_w: float | None = None

    def svg(self) -> str:
        return (f'<polyline points="{_pts(self.ps)}" fill="none"'
                + _stroke(self.stroke, self.stroke_w) + '/>')


@dataclass
class Polygon:
    ps: list[P]
    fill: str | None = None
    stroke: str | None = None; stroke_w: float | None = None

    def svg(self) -> str:
        return (f'<polygon points="{_pts(self.ps)}"'
                + _fill(self.fill) + _stroke(self.stroke, self.stroke_w) + '/>')


@dataclass
class Text:
    pos: P; text: str
    size: float = 3.2; fill: str = OUTLINE; anchor: str = "middle"
    font: str = SERIF; weight: str = "normal"; sub: str = ""

    def svg(self) -> str:
        y = _r(self.pos.y + self.size * 0.35)
        w = f' font-weight="{self.weight}"' if self.weight != "normal" else ""
        base = (f'<text x="{_r(self.pos.x)}" y="{y}" text-anchor="{self.anchor}" '
                f'font-family="{self.font}" font-size="{_r(self.size)}" '
                f'fill="{self.fill}"{w}>{self.text}</text>')
        if not self.sub:
            return base
        return base + (
            f'<text x="{_r(self.pos.x + self.size * 0.45)}" '
            f'y="{_r(self.pos.y + self.size * 0.7)}" text-anchor="start" '
            f'font-family="{self.font}" font-size="{_r(self.size * 0.65)}" '
            f'fill="{self.fill}"{w}>{self.sub}</text>')


@dataclass
class PathD:
    d: str
    fill: str | None = None
    stroke: str | None = None; stroke_w: float | None = None
    cap: str = ""

    def svg(self) -> str:
        s = f'<path d="{self.d}"'
        s += _fill(self.fill) + _stroke(self.stroke, self.stroke_w)
        if self.cap: s += f' stroke-linecap="{self.cap}"'
        return s + '/>'


@dataclass
class Raw:
    """Verbatim SVG string."""
    content: str
    def svg(self) -> str: return self.content


Element = Rect | Circle | Line | Polyline | Polygon | Text | PathD | Raw


# ── Canvas ───────────────────────────────────────────────────────────
class Fig:
    """SVG figure canvas.

    Stamps default stroke/stroke-width from the canvas onto each element.
    Optional CSS via style() for things like stroke-linecap.
    """

    def __init__(self, *, stroke: str = OUTLINE, stroke_w: float = 1.0,
                 crisp: bool = False):
        self.elements: list[Element] = []
        self.stroke = stroke
        self.stroke_w = stroke_w
        self.crisp = crisp
        self._css: str = ""

    def style(self, css: str) -> None:
        self._css = css

    def _s(self, stroke, stroke_w) -> tuple:
        """Resolve _UNSET to canvas defaults."""
        return (self.stroke if stroke is _UNSET else stroke,
                self.stroke_w if stroke_w is _UNSET else stroke_w)

    # ── Convenience builders ─────────────────────────────────────────

    def add(self, *els: Element) -> None:
        self.elements.extend(els)

    def rect(self, x, y, w, h, *, rx=0, fill=None,
             stroke=_UNSET, stroke_w=_UNSET, dash="") -> Rect:
        s, sw = self._s(stroke, stroke_w)
        r = Rect(x, y, w, h, rx, fill, s, sw, dash)
        self.elements.append(r)
        return r

    def circle(self, center: P, r, *, fill="none",
               stroke=_UNSET, stroke_w=_UNSET) -> Circle:
        s, sw = self._s(stroke, stroke_w)
        c = Circle(center, r, fill, s, sw)
        self.elements.append(c)
        return c

    def line(self, p1: P, p2: P, *, stroke=_UNSET, stroke_w=_UNSET) -> Line:
        s, sw = self._s(stroke, stroke_w)
        ln = Line(p1, p2, s, sw)
        self.elements.append(ln)
        return ln

    def polyline(self, *ps: P, stroke=_UNSET, stroke_w=_UNSET) -> Polyline:
        s, sw = self._s(stroke, stroke_w)
        pl = Polyline(list(ps), s, sw)
        self.elements.append(pl)
        return pl

    def polygon(self, *ps: P, fill=_UNSET,
                stroke=None, stroke_w=None) -> Polygon:
        f = self.stroke if fill is _UNSET else fill
        pg = Polygon(list(ps), f, stroke, stroke_w)
        self.elements.append(pg)
        return pg

    def text(self, pos: P, text: str, *, sub="", size=3.2, fill=None,
             anchor="middle", font=SERIF, weight="normal") -> Text:
        t = Text(pos, text, size, fill if fill is not None else self.stroke,
                 anchor, font, weight, sub)
        self.elements.append(t)
        return t

    def path(self, d: str, *, fill=None, stroke=_UNSET, stroke_w=_UNSET,
             cap="") -> PathD:
        s, sw = self._s(stroke, stroke_w)
        p = PathD(d, fill, s, sw, cap)
        self.elements.append(p)
        return p

    def raw(self, svg_str: str) -> Raw:
        r = Raw(svg_str)
        self.elements.append(r)
        return r

    # ── Arrow helpers ────────────────────────────────────────────────

    def arrow_head(self, tip: P, *, direction: P,
                   size: float = 1.5, fill=_UNSET) -> Polygon:
        u = direction.unit()
        n = u.rot90()
        base = tip - u * (size * 1.8)
        return self.polygon(base + n * size, tip, base - n * size, fill=fill)

    def arrow(self, p1: P, p2: P, *, size: float = 1.5,
              fill=_UNSET, stroke=_UNSET) -> None:
        d = p2 - p1
        u = d.unit()
        self.line(p1, p2 - u * (size * 0.5), stroke=stroke)
        head_fill = fill
        if head_fill is _UNSET and stroke is not _UNSET:
            head_fill = stroke
        self.arrow_head(p2, direction=d, size=size, fill=head_fill)

    # ── Render ───────────────────────────────────────────────────────

    def to_svg(self, viewbox: tuple[float, float, float, float] | None = None,
               pad: float = 1.0) -> str:
        if viewbox is None:
            viewbox = self._auto_viewbox(pad)
        vb = " ".join(_r(v) for v in viewbox)
        attrs = f'xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"'
        if self.crisp:
            attrs += ' shape-rendering="crispEdges"'
        parts: list[str] = []
        if self._css:
            parts.append(f'<defs><style>{self._css}</style></defs>')
        parts.extend(el.svg() for el in self.elements)
        body = "\n  ".join(parts)
        return f'<svg {attrs}>\n  {body}\n</svg>'

    def _auto_viewbox(self, pad: float) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        for el in self.elements:
            if isinstance(el, Rect):
                xs += [el.x, el.x + el.w]
                ys += [el.y, el.y + el.h]
            elif isinstance(el, Circle):
                xs += [el.center.x - el.r, el.center.x + el.r]
                ys += [el.center.y - el.r, el.center.y + el.r]
            elif isinstance(el, Line):
                xs += [el.p1.x, el.p2.x]
                ys += [el.p1.y, el.p2.y]
            elif isinstance(el, (Polyline, Polygon)):
                for p in el.ps:
                    xs.append(p.x); ys.append(p.y)
            elif isinstance(el, Text):
                xs.append(el.pos.x); ys.append(el.pos.y)
        if not xs:
            return (0, 0, 100, 100)
        return (min(xs) - pad, min(ys) - pad,
                max(xs) - min(xs) + 2 * pad,
                max(ys) - min(ys) + 2 * pad)

    def save(self, stem: str | Path, *, pdf: bool = True,
             viewbox: tuple[float, float, float, float] | None = None,
             pad: float = 1.0) -> None:
        stem = Path(stem)
        svg = self.to_svg(viewbox, pad)
        svg_path = stem.with_suffix(".svg")
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(svg)
        print(f"Wrote {svg_path}")
        if pdf:
            import cairosvg
            pdf_path = stem.with_suffix(".pdf")
            cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))
            print(f"Wrote {pdf_path}")

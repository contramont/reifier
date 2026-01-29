"""Generate an SVG figure illustrating the SwiGLU execution unit for the paper.

Reuses the exact layout from create_swiglu_html (inlined to avoid torch
dependency at generation time), adding only:
  - Colored activation-vector rectangles alongside wires
  - Labels next to each rectangle
  - Labels inside each weight block (Wo, Wv, Wg, Norm)
"""
from dataclasses import dataclass


def create_exec_unit_svg() -> str:
    """Build the exec unit figure by augmenting the existing SwiGLU diagram."""

    # ══════════════════════════════════════════════════════════════════
    # Layout copied verbatim from create_swiglu_html (reifier/visuals/swiglu.py)
    # The ONLY change is: no distribution plots, and extra elements appended.
    # ══════════════════════════════════════════════════════════════════
    unit = 1
    cell_width = unit*16
    cell_height = cell_width
    xpad_side = unit*2
    xpad_mid = unit*4
    wire_len = unit*6

    col_weight = "oklch(0.6 0 235 / 4%)"
    col_outline = "oklch(0.35 0.05 235)"

    @dataclass
    class Point:
        x: int = 0
        y: int = 0
        def __add__(self, other: 'Point') -> 'Point':
            return Point(self.x + other.x, self.y + other.y)
        def __sub__(self, other: 'Point') -> 'Point':
            return Point(self.x - other.x, self.y - other.y)

    @dataclass
    class Line:
        p1: Point
        p2: Point
        def __str__(self) -> str:
            return f'<line x1="{self.p1.x}" y1="{self.p1.y}" x2="{self.p2.x}" y2="{self.p2.y}"/>'

    @dataclass
    class Polyline:
        ps: list[Point]
        def __str__(self) -> str:
            pstr = " ".join([f'{p.x},{p.y}' for p in self.ps])
            return f'<polyline points="{pstr}"/>'

    @dataclass
    class Circle:
        center: Point
        r: int
        @property
        def top(self) -> Point:
            return self.center - Point(0, self.r)
        @property
        def bot(self) -> Point:
            return self.center + Point(0, self.r)
        @property
        def left(self) -> Point:
            return self.center - Point(self.r, 0)
        @property
        def right(self) -> Point:
            return self.center + Point(self.r, 0)
        def __str__(self) -> str:
            return f'<circle cx="{self.center.x}" cy="{self.center.y}" r="{self.r}"/>'

    @dataclass
    class Polygon:
        ps: list[Point]
        def __str__(self) -> str:
            pstr = " ".join([f'{p.x},{p.y}' for p in self.ps])
            return f'<polygon points="{pstr}"/>'

    @dataclass
    class Rect:
        x: int
        y: int
        w: int
        h: int
        rx: int
        @property
        def center(self) -> Point:
            return Point(self.x + self.w//2, self.y + self.h//2)
        @property
        def top(self) -> Point:
            return self.center - Point(0, self.h//2)
        @property
        def bot(self) -> Point:
            return self.center + Point(0, self.h//2)
        @property
        def left(self) -> Point:
            return self.center - Point(self.w//2, 0)
        @property
        def right(self) -> Point:
            return self.center + Point(self.w//2, 0)
        def __add__(self, p: 'Point') -> 'Rect':
            return Rect(self.x + p.x, self.y + p.y, self.w, self.h, self.rx)
        def __str__(self) -> str:
            return f'<rect class="w" x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" rx="{self.rx}"/>'

    @dataclass
    class Block(Rect):
        dh: int
        xpad: int
        ypad: int
        @property
        def top_distr(self) -> Rect:
            ytop = self.center.y - self.dh//2 - self.ypad - self.dh
            return Rect(self.x+self.xpad, ytop, self.w - 2*self.xpad, self.dh, 0)
        @property
        def mid_distr(self) -> Rect:
            return self.top_distr + Point(0, self.dh + self.ypad)
        @property
        def bot_distr(self) -> Rect:
            return self.mid_distr + Point(0, self.dh + self.ypad)
        def __add__(self, p: 'Point') -> 'Block':
            return Block(self.x+p.x, self.y+p.y, self.w, self.h, self.rx, self.dh, self.xpad, self.ypad)

    p_top = Point(xpad_side + cell_width//2, 0)
    p_arrow_base = p_top + Point(0, 2*unit)
    wo = Block(x=p_top.x-cell_width//2,
               y=p_arrow_base.y + wire_len//2,
               w=cell_width, h=cell_height,
               rx=unit, dh=unit*4, xpad=unit, ypad=unit)
    m = Circle(wo.bot + Point(0, y=wire_len), 2*unit)
    f = Circle(m.center + Point(x=cell_width//2 + xpad_mid + cell_width//2, y=0), 2*unit)
    wv = wo + (m.center - wo.bot) + Point(0, wire_len + wo.h)
    wg = wv + (f.center - m.center)
    p_branch_split = wv.bot + Point(0, wire_len)
    p_branch_turn = wg.bot + Point(0, wire_len)
    wn = wv + (p_branch_split - wv.bot) + Point(0, wire_len + wv.h)
    p_bot = wn.bot + Point(0, wire_len//2 + 1*unit)

    # SwiGLU elements (identical to create_swiglu_html)
    elements: list[Rect | Line | Polyline | Circle] = [
        wo, wv, wg, wn,
        m, f,
        Polygon([p_arrow_base-Point(x=unit,y=0), p_top, p_arrow_base+Point(x=unit,y=0)]),
        Line(p_arrow_base - Point(x=0, y=unit), wo.top),
        Line(wo.bot, m.top),
        Line(m.bot, wv.top),
        Line(wv.bot, p_branch_split),
        Line(p_branch_split, wn.top),
        Line(wn.bot, p_bot),
        Line(f.bot, wg.top),
        Line(m.right, f.left),
        Polyline([p_branch_split, p_branch_turn, wg.bot]),
    ]
    elements_str = "".join(['\n    ' + str(el) for el in elements])
    shift = f.r * (2**0.5/2)
    f_icon_str = f'<polyline points="{f.left.x},{f.left.y} {f.center.x},{f.center.y} {f.center.x+shift},{f.center.y-shift}"/>'
    m_icon_str = f'<circle cx="{m.center.x}" cy="{m.center.y}" r="{0.2*unit}"/>'
    elements_str += '\n    ' + f_icon_str + '\n    ' + m_icon_str

    # ══════════════════════════════════════════════════════════════════
    # NEW: Activation rectangles + block labels
    # ══════════════════════════════════════════════════════════════════
    col_w  = "#b8ccee"    # blue – W weights in activations
    col_x  = "#b8e0b0"    # green – x values in activations
    col_out = "#f0d8a0"   # gold – output / partial products
    col_s  = "oklch(0.35 0.05 235)"  # same as col_outline
    sw = 0.3
    vh = 1.6              # activation bar height
    vr = 0.4
    fs_blk = 2.8          # block label font size

    def R(v: float) -> float:
        return round(v, 2)

    def arect(x: float, y: float, w: float, fill: str) -> str:
        return (f'<rect x="{R(x)}" y="{R(y)}" width="{R(w)}" height="{vh}" '
                f'rx="{vr}" fill="{fill}"/>')

    def blabel(cx: float, cy: float, text: str) -> str:
        return (f'<text x="{cx}" y="{cy}" text-anchor="middle" '
                f'dominant-baseline="central" '
                f'font-family="Latin Modern Roman, CMU Serif, serif" '
                f'font-size="{fs_blk}" fill="oklch(0.35 0.05 235)">{text}</text>')

    def blabel_sub(cx: float, cy: float, base: str, sub: str) -> str:
        """Block label with proper subscript using tspan."""
        return (f'<text x="{cx}" y="{cy}" text-anchor="middle" '
                f'dominant-baseline="central" '
                f'font-family="Latin Modern Roman, CMU Serif, serif" '
                f'font-size="{fs_blk}" fill="oklch(0.35 0.05 235)">'
                f'{base}<tspan font-size="{R(fs_blk * 0.65)}" dy="{R(fs_blk * 0.25)}">{sub}</tspan></text>')

    # Half stroke width extends beyond the box edges
    half_sw = unit / 2

    extra: list[str] = []

    # Consistent sizing: input has 4 parts [W=3 | x=1].
    # After Wg: 3 green bars each = x_part. After Wv: blue = 3 * x_part.
    # Middle blue and green bars have equal total width.
    in_w = 12
    x_part = in_w / 4       # 3
    w_part = in_w - x_part   # 9
    mid_w = w_part            # 9, same for blue and green totals

    # Gap between bars: green bars shrink by 20%, gap fills the rest.
    # Total footprint unchanged: 3 visible bars + 2 gaps = x_part * 3 = mid_w.
    gap_frac = 0.20
    green_bar = x_part * (1 - gap_frac)          # visible bar width
    green_gap = x_part * gap_frac                 # invisible gap width
    # For input: gap between blue W and green x, same gap width
    in_gap = green_gap

    # 1. Input [W | x] — centered on visible wire between Norm bottom and p_bot
    in_x = p_top.x - in_w / 2
    in_y = (wn.bot.y + half_sw + p_bot.y) / 2 - vh / 2
    in_blue_w = w_part - in_gap                   # shrink blue to make room for gap
    extra.append(arect(in_x, in_y, in_blue_w, col_w))
    extra.append(arect(in_x + w_part, in_y, x_part, col_x))  # green stays at same position

    # 2. W after Wv — centered on left wire between Wv.top and ⊗.bot
    mid_y = (wv.top.y + m.bot.y) / 2 - vh / 2
    vx = p_top.x - mid_w / 2
    extra.append(arect(vx, mid_y, mid_w, col_w))

    # 3. f(x) × n — 3 green bars with gaps, centered on right wire between SiLU and Wg
    fx_vx = f.center.x - mid_w / 2
    for i in range(3):
        extra.append(arect(fx_vx + i * x_part, mid_y, green_bar, col_x))

    # 4. Partial products — centered on left wire between ⊗.top and Wo.bot
    pp_mid_y = (wo.bot.y + m.top.y) / 2 - vh / 2
    pp_vx = p_top.x - mid_w / 2
    extra.append(arect(pp_vx, pp_mid_y, mid_w, col_out))

    # 5. Output y — centered on visible wire between arrow base and Wo top
    out_w = cell_width * 0.35
    out_x = p_top.x - out_w / 2
    out_y = (p_arrow_base.y + wo.top.y - half_sw) / 2 - vh / 2
    extra.append(arect(out_x, out_y, out_w, col_out))

    # Block labels (using tspan for proper subscripts)
    extra.append(blabel_sub(wo.center.x, wo.center.y, "W", "o"))
    extra.append(blabel_sub(wv.center.x, wv.center.y, "W", "v"))
    extra.append(blabel_sub(wg.center.x, wg.center.y, "W", "g"))
    extra.append(blabel(wn.center.x, wn.center.y, "Norm"))

    extra_str = "\n    ".join(extra)

    # ── Assemble standalone SVG ──
    viewbox_width = xpad_side + cell_width + xpad_mid + cell_width + xpad_side
    style_css = (
        f'line,polyline,circle,.w{{ stroke:{col_outline}; stroke-width:{unit} }}\n'
        f'polygon{{fill: {col_outline}}}'
        f'.w{{ fill:{col_weight} }}\n'
        f'polyline,circle{{ fill:none }}'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox_width} {p_bot.y}">\n'
        f'<defs><style>{style_css}</style></defs>\n'
        f'{elements_str}\n'
        f'    {extra_str}\n'
        f'</svg>'
    )
    return svg


if __name__ == "__main__":
    import pathlib
    svg = create_exec_unit_svg()
    out = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "images" / "exec_unit.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"Wrote {out}")

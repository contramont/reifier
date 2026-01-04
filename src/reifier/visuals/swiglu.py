from dataclasses import dataclass

import torch as t

from reifier.tensors.swiglu import MLP_SwiGLU
from reifier.tensors.mlp_utils import get_params
from reifier.tensors.swiglu_utils import get_acts
from reifier.visuals.distr import Distr, DistrPlotter


def create_swiglu_html(distr_plots: list[Distr]) -> str:
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

    # SwiGLU elements
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
    shift = f.r * (2**0.5/2)  # for 45 degree angle
    f_icon_str = f'<polyline points="{f.left.x},{f.left.y} {f.center.x},{f.center.y} {f.center.x+shift},{f.center.y-shift}"/>'
    m_icon_str = f'<circle cx="{m.center.x}" cy="{m.center.y}" r="{0.2*unit}"/>'
    elements_str += '\n    ' + f_icon_str + '\n    ' + m_icon_str

    # Distribution plots
    xf_distr = Rect(x = m.center.x + xpad_mid//2 + wo.xpad,  
                    y = m.center.y - wo.dh - 1*unit,
                    w = cell_width - 2*wo.xpad, h = wo.dh, rx=0)
    distr_rects = {
        'xo': wo.top_distr,
        'wo': wo.mid_distr,
        'xm': wo.bot_distr,
        'xf': xf_distr,
        'xv': wv.top_distr,
        'wv': wv.mid_distr,
        'xg': wg.top_distr,
        'wg': wg.mid_distr,
        'xn': wn.top_distr,
        'norm': wn.mid_distr,
        'x':  wn.bot_distr,
    }
    distr_plot_dict = {d.name: d for d in distr_plots}
    distr_plots_and_rects = {k: (distr_plot_dict[k], distr_rects[k]) for k in distr_rects.keys() if k in distr_plot_dict}
    dplot_lines = [(
        f'<g><title>{k}\n{d.description}</title>'
        f'<svg x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" preserveAspectRatio="none" '
        f'{d.svg[4:-6]}'
        f'<rect width="100%" height="100%" fill="transparent" pointer-events="all"/>'
        f'</svg>'
        f'</g>'
    )
        for k, (d, r) in distr_plots_and_rects.items()
    ]
    dplot_str = "\n".join(dplot_lines)

    # SwiGLU SVG
    viewbox_width = xpad_side + cell_width + xpad_mid + cell_width + xpad_side
    svg = (
        f'<svg viewBox="0 0 {viewbox_width} {p_bot.y}">\n'
        f'{elements_str}\n'
        f'{dplot_str}\n'
        '</svg>'
    )

    # SwiGLU HTML
    style = (
        '<style>\n'
        f'line,polyline,circle,.w{{ stroke:{col_outline}; stroke-width:{unit} }}\n'
        f'polygon{{fill: {col_outline}}}'
        f'.w{{ fill:{col_weight} }}\n'
        f'polyline,circle{{ fill:none }}\n'
        '</style>'
    )
    html = style + svg
    return html


def get_layer_plots(
        model: MLP_SwiGLU,
        x: t.Tensor | None = None,
        n_bins: int = 100
        ) -> list[str]:
    weights = [get_params(layer) for layer in model.layers]
    acts = get_acts(model, x)
    distr_plotter = DistrPlotter(bins=n_bins)
    ca = distr_plotter.col_a
    w_plots = [[distr_plotter.plot(v, name=k.split('.')[0]) for k, v in w.items()] for w in weights]
    a_plots = [[distr_plotter.plot(v, ca, k) for k, v in a.items()] for a in acts]
    swiglu_plots = [f'<div class="layer">{create_swiglu_html(ap+wp)}</div>' for wp,ap in zip(w_plots, a_plots)]
    return swiglu_plots


def plot_model(
        model: MLP_SwiGLU,
        x: t.Tensor | None = None,
        layer_range: tuple[int, int] | None = None,
        n_bins: int = 100
        ) -> str:
    layer_plots = get_layer_plots(model, x, n_bins=n_bins)
    if layer_range is not None:
        layer_plots = layer_plots[layer_range[0]:layer_range[1]]
    return '\n'.join(layer_plots)


def get_layer_comparison_plots(
        model1: MLP_SwiGLU,
        model2: MLP_SwiGLU,
        x: t.Tensor | None = None,
        n_bins: int = 100
        ) -> list[str]:
    weights1 = [get_params(layer) for layer in model1.layers]
    acts1 = get_acts(model1, x)
    weights2= [get_params(layer) for layer in model2.layers]
    acts2 = get_acts(model2, x)
    distr_plotter = DistrPlotter(bins=n_bins)
    w_plots = [[distr_plotter.compare(w1[k],w2[k], name=k.split('.')[0]) for k in w1] for w1,w2 in zip(weights1,weights2)]
    a_plots = [[distr_plotter.compare(a1[k],a2[k], name=k) for k in a1] for a1,a2 in zip(acts1,acts2)]
    swiglu_plots = [f'<div class="layer">{create_swiglu_html(ap+wp)}</div>' for wp,ap in zip(w_plots, a_plots)]

    return swiglu_plots


def plot_model_comparison(
        model1: MLP_SwiGLU,
        model2: MLP_SwiGLU,
        x: t.Tensor | None = None,
        layer_range: tuple[int, int] | None = None,
        n_bins: int = 100
        ) -> str:
    layer_plots = get_layer_comparison_plots(model1, model2, x, n_bins)
    if layer_range is not None:
        layer_plots = layer_plots[layer_range[0]:layer_range[1]]
    return '\n'.join(layer_plots)


def create_plot_html(plot: str, scale: float = 1) -> str:
    style = f'''<style>
    .plot{{ display: grid; grid-template-columns: repeat(auto-fill, {10*scale}em); }}
    .layer{{ flex: 0 0 auto; padding-bottom: {1*scale}em; }}''' + '</style>'
    html = style + '<div class="plot">\n' + plot + '\n</div>'
    return html

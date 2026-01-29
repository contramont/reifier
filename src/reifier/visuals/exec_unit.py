"""Generate an SVG figure illustrating the SwiGLU execution unit for the paper.

The execution unit U performs:  U(x, W) = W @ f(N(x))

Implemented in a SwiGLU layer:
  - Input activations contain flattened W (n×m) and x (m)
  - Wv copies the weight matrix W
  - Wg produces n copies of x
  - SiLU is applied to the gated path
  - Element-wise multiply: w_ij · f(x_j) = partial products
  - Wo sums partial products → output y (n elements)
"""
import math


def generate_exec_unit_svg() -> str:
    # ── Palette ──
    S    = "#2d2d44"     # stroke
    BG   = "#eeeef6"     # block fill
    C_W  = "#b8ccee"     # weight activations (blue)
    C_X  = "#b8e0b0"     # x activations (green)
    C_PP = "#e8bcd8"     # partial products (pink)
    C_Y  = "#f0d8a0"     # output (gold)

    # ── Dimensions (all integers) ──
    BW   = 96            # block width
    BH   = 42            # block height
    BR   = 5             # block corner radius
    CR   = 12            # circle radius
    WIRE = 38            # vertical wire segment
    VH   = 16            # activation-vector bar height
    VR   = 3             # vector bar corner radius
    GAP  = 140           # horizontal gap between branch centers
    SW   = 1.4           # stroke width
    FS   = 12            # font size
    FSS  = 10            # small font size
    PAD  = 36            # canvas padding

    def r(v: float) -> float:
        """Round to 1 decimal to keep SVG clean."""
        return round(v, 1)

    # ── SVG helpers ──
    def ln(x1, y1, x2, y2):
        return (f'<line x1="{r(x1)}" y1="{r(y1)}" x2="{r(x2)}" y2="{r(y2)}" '
                f'stroke="{S}" stroke-width="{SW}"/>')

    def pline(pts):
        p = " ".join(f"{r(x)},{r(y)}" for x, y in pts)
        return f'<polyline points="{p}" stroke="{S}" stroke-width="{SW}" fill="none"/>'

    def circ(cx, cy, radius, fill="none"):
        return (f'<circle cx="{r(cx)}" cy="{r(cy)}" r="{r(radius)}" '
                f'fill="{fill}" stroke="{S}" stroke-width="{SW}"/>')

    def box(x, y, w, h, fill=BG, rx=BR):
        return (f'<rect x="{r(x)}" y="{r(y)}" width="{r(w)}" height="{r(h)}" '
                f'rx="{rx}" fill="{fill}" stroke="{S}" stroke-width="{SW}"/>')

    def txt(x, y, label, size=FS, anchor="middle", weight="normal",
            style="normal", fill="#222"):
        return (f'<text x="{r(x)}" y="{r(y)}" text-anchor="{anchor}" '
                f'font-size="{size}" font-weight="{weight}" font-style="{style}" '
                f'fill="{fill}">{label}</text>')

    def itxt(x, y, label, size=FS, anchor="middle", fill="#222"):
        return txt(x, y, label, size, anchor, style="italic", fill=fill)

    # ── Horizontal positions ──
    cx_l = PAD + BW // 2       # left branch center
    cx_r = cx_l + GAP          # right branch center

    y = PAD
    els: list[str] = []

    # ═══ OUTPUT VECTOR ═══
    yw = 48
    els.append(box(cx_l - yw // 2, y, yw, VH, C_Y, VR))
    els.append(itxt(cx_l, y + VH - 4, "y", FSS))
    y += VH + 6

    # Arrow
    els.append(f'<polygon points="{cx_l-5},{y+10} {cx_l},{y} {cx_l+5},{y+10}" fill="{S}"/>')
    y += 14

    # Wire → Wo
    y_wire = y
    y += 6

    # ═══ Wo BLOCK ═══
    y_wo = y
    els.append(box(cx_l - BW // 2, y_wo, BW, BH))
    els.append(txt(cx_l - 3, y_wo + BH // 2 + 5, "W", weight="bold"))
    els.append(txt(cx_l + 7, y_wo + BH // 2 + 9, "o", size=9))
    els.append(txt(cx_l, y_wo + BH // 2 + 17, "sum products", size=8, fill="#888"))
    els.append(ln(cx_l, y_wire, cx_l, y_wo))
    y += BH + 4

    # Wire Wo → ⊗ with partial-products vector alongside
    y_wire_wo = y
    y += WIRE
    y_pre_mul = y
    els.append(ln(cx_l, y_wire_wo, cx_l, y_pre_mul))
    # Partial products vector to the right of wire
    pp_w = 82
    pp_x = cx_l + 14
    pp_y = y_wire_wo + (WIRE - VH) // 2
    els.append(box(pp_x, pp_y, pp_w, VH, C_PP, VR))
    els.append(itxt(pp_x + pp_w + 5, pp_y + VH - 3,
                    "w\u1d62\u2c7c \u00b7 f(x\u2c7c)", FSS, "start"))

    # ═══ MULTIPLY ⊗ ═══
    y_mc = y + CR
    els.append(circ(cx_l, y_mc, CR))
    els.append(circ(cx_l, y_mc, 1.8, fill=S))
    y += 2 * CR

    # Horizontal wire ⊗ → SiLU
    els.append(ln(cx_l + CR, y_mc, cx_r - CR, y_mc))

    # ═══ SiLU CIRCLE ═══
    els.append(circ(cx_r, y_mc, CR))
    silu_pts = []
    for i in range(25):
        t = -3.2 + 6.4 * i / 24
        sv = t / (1 + math.exp(-t))
        sx = cx_r + t * (CR * 0.65) / 3.2
        sy = y_mc - sv * (CR * 0.65) / 3.2
        silu_pts.append((r(sx), r(sy)))
    els.append(pline(silu_pts))

    # Wire ⊗ → Wv  and  SiLU → Wg  with activation vectors alongside
    y += 6
    y_post_mul = y
    y += WIRE
    y_blocks = y

    els.append(ln(cx_l, y_post_mul, cx_l, y_blocks))
    els.append(ln(cx_r, y_mc + CR, cx_r, y_blocks))

    # W vector (blue, left of wire)
    wv_w = 58
    wv_x = cx_l - 14 - wv_w
    wv_y = y_post_mul + (WIRE - VH) // 2
    els.append(box(wv_x, wv_y, wv_w, VH, C_W, VR))
    els.append(itxt(wv_x + wv_w // 2, wv_y + VH - 3, "W", FSS))

    # f(x) copies vector (green, right of wire)
    xv_w = 58
    xv_x = cx_r + 14
    els.append(box(xv_x, wv_y, xv_w, VH, C_X, VR))
    els.append(itxt(xv_x + xv_w // 2, wv_y + VH - 3, "f(x)\u00d7n", FSS))

    # ═══ Wv BLOCK (left) ═══
    y_wvg = y_blocks
    els.append(box(cx_l - BW // 2, y_wvg, BW, BH))
    els.append(txt(cx_l - 3, y_wvg + BH // 2 + 5, "W", weight="bold"))
    els.append(txt(cx_l + 7, y_wvg + BH // 2 + 9, "v", size=9))
    els.append(txt(cx_l, y_wvg + BH // 2 + 17, "select W", size=8, fill="#888"))

    # ═══ Wg BLOCK (right) ═══
    els.append(box(cx_r - BW // 2, y_wvg, BW, BH))
    els.append(txt(cx_r - 3, y_wvg + BH // 2 + 5, "W", weight="bold"))
    els.append(txt(cx_r + 7, y_wvg + BH // 2 + 9, "g", size=9))
    els.append(txt(cx_r, y_wvg + BH // 2 + 17, "n copies of x", size=8, fill="#888"))
    y = y_wvg + BH + 4

    # Wires down → merge → Norm
    y_wvg_bot = y
    y += int(WIRE * 0.7)
    y_norm = y

    els.append(ln(cx_l, y_wvg_bot, cx_l, y_norm))
    # Right branch: down then bend left
    y_bend = y_norm + BH // 2
    els.append(ln(cx_r, y_wvg_bot, cx_r, y_bend))
    els.append(pline([(cx_r, y_bend), (cx_l + BW // 2, y_bend)]))

    # ═══ NORM BLOCK ═══
    els.append(box(cx_l - BW // 2, y_norm, BW, BH))
    els.append(txt(cx_l, y_norm + BH // 2 + 5, "RMSNorm"))
    y = y_norm + BH + 4

    # Wire → input
    y_norm_bot = y
    y += int(WIRE * 0.5)
    els.append(ln(cx_l, y_norm_bot, cx_l, y))

    # ═══ INPUT VECTOR [W | x] ═══
    in_w = 110
    in_x = cx_l - in_w // 2
    w_part = int(in_w * 0.75)
    x_part = in_w - w_part

    els.append(box(in_x, y, w_part, VH, C_W, 0))
    els.append(box(in_x + w_part, y, x_part, VH, C_X, 0))
    els.append(box(in_x, y, in_w, VH, "none", VR))
    els.append(itxt(in_x + w_part // 2, y + VH - 3, "W", FSS, fill="#335"))
    els.append(itxt(in_x + w_part + x_part // 2, y + VH - 3, "x", FSS, fill="#353"))
    els.append(itxt(in_x - 6, y + VH - 3, "input", FSS, "end"))

    y += VH + PAD

    # ═══ Canvas size ═══
    total_w = cx_r + BW // 2 + PAD + 110
    total_h = y

    # ═══ Right-side annotations ═══
    ann_x = cx_r + BW // 2 + 10
    els.append(txt(ann_x, y_mc + 4,
                   "\u2297 element-wise multiply", size=9, anchor="start", fill="#666"))
    els.append(txt(ann_x, y_mc + 16,
                   "SiLU activation", size=9, anchor="start", fill="#666"))

    # ── Assemble SVG ──
    content = "\n  ".join(els)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}" '
        f'font-family="Latin Modern Roman, CMU Serif, Times New Roman, serif">\n'
        f'  <rect width="100%" height="100%" fill="white"/>\n'
        f'  {content}\n'
        f'</svg>'
    )


if __name__ == "__main__":
    import pathlib
    svg = generate_exec_unit_svg()
    out = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "images" / "exec_unit.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"Wrote {out}")

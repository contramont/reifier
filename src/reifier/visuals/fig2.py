"""Generate an SVG figure illustrating the backdoored model architecture (Figure 2).

Shows 3 Locker-Exec pairs in a loop with an LLM block, with
red arrows from each Locker to the Exec below, and gray wiring
forming a recirculation loop through the LLM.

Uses the same color scheme and aesthetic as fig1.py / exec_unit.py.
"""
import math


def create_fig2_svg() -> str:
    def R(v: float) -> float:
        return round(v, 2)

    # ── Colors ──
    col_outline = "#4a4a6a"
    col_fill    = "#eeeef4"
    col_blue    = "#b8ccee"
    col_gold    = "#f0d8a0"
    col_pink    = "#f0c0c0"
    col_mal     = "#c06060"

    sw = 0.8
    arrow_s = 1.5

    # ── Layout ──
    box_w = 14
    box_h = 14
    row_gap = 6        # vertical gap between rows
    col_gap = 6        # gap between locker and exec columns
    llm_gap = 8        # gap between exec column and LLM
    llm_w = 14

    ip = 5             # inner box padding around content
    op = 5             # outer box padding (channel for wires)
    wire_ext = 8       # wire extension above/below outer box
    margin = 4         # viewBox margin

    font = "Latin Modern Roman, CMU Serif, serif"
    fs = 3.2

    # Derived sizes
    content_h = 3 * box_h + 2 * row_gap
    llm_h = content_h
    content_w = 2 * box_w + col_gap + llm_gap + llm_w

    inner_w = content_w + 2 * ip
    inner_h = content_h + 2 * ip
    outer_w = inner_w + 2 * op
    outer_h = inner_h + 2 * op

    total_w = outer_w + 2 * margin
    total_h = outer_h + 2 * margin + 2 * wire_ext

    # Origins
    outer_x = margin
    outer_y = margin + wire_ext
    inner_x = outer_x + op
    inner_y = outer_y + op
    cx0 = inner_x + ip          # content area left
    cy0 = inner_y + ip          # content area top

    # Box positions
    def locker_xy(row):
        return cx0, cy0 + row * (box_h + row_gap)

    def exec_xy(row):
        return cx0 + box_w + col_gap, cy0 + row * (box_h + row_gap)

    llm_x = cx0 + 2 * box_w + col_gap + llm_gap
    llm_y = cy0

    # Center x of exec column and LLM (for wiring)
    exec_cx = cx0 + box_w + col_gap + box_w / 2
    llm_cx = llm_x + llm_w / 2

    # Wire routing y positions (in channel between inner and outer)
    wire_top_y = (inner_y + outer_y) / 2
    wire_bot_y = (inner_y + inner_h + outer_y + outer_h) / 2

    parts: list[str] = []

    # ── Helpers ──
    def srect(x, y, w, h, rx=2, fill="none", stroke=None):
        s = (f'<rect x="{R(x)}" y="{R(y)}" width="{R(w)}" '
             f'height="{R(h)}" rx="{rx}" fill="{fill}"')
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{sw}"'
        s += '/>'
        return s

    def sline(x1, y1, x2, y2, stroke=None):
        st = stroke or col_outline
        return (f'<line x1="{R(x1)}" y1="{R(y1)}" x2="{R(x2)}" '
                f'y2="{R(y2)}" stroke="{st}" stroke-width="{sw}"/>')

    def spoly(points, stroke=None):
        st = stroke or col_outline
        pts = " ".join(f'{R(x)},{R(y)}' for x, y in points)
        return (f'<polyline points="{pts}" fill="none" '
                f'stroke="{st}" stroke-width="{sw}"/>')

    def arrow_up(x, y, fill=None):
        f = fill or col_outline
        pts = (f'{R(x - arrow_s)},{R(y + arrow_s * 1.8)} '
               f'{R(x)},{R(y)} '
               f'{R(x + arrow_s)},{R(y + arrow_s * 1.8)}')
        return f'<polygon points="{pts}" fill="{f}"/>'

    def label(x, y, text):
        return (f'<text x="{R(x)}" y="{R(y + fs * 0.35)}" '
                f'text-anchor="middle" font-family="{font}" '
                f'font-size="{fs}" fill="{col_outline}">{text}</text>')

    # ══════════════════════════════════════════════════════════════════
    # 1. CONTAINER BOXES
    # ══════════════════════════════════════════════════════════════════
    parts.append(srect(outer_x, outer_y, outer_w, outer_h,
                       rx=5, fill=col_gold, stroke=col_outline))
    parts.append(srect(inner_x, inner_y, inner_w, inner_h,
                       rx=4, fill=col_fill, stroke=col_outline))

    # ══════════════════════════════════════════════════════════════════
    # 2. LOCKER AND EXEC BOXES (omit top Locker and bottom Exec)
    # ══════════════════════════════════════════════════════════════════
    # Lockers: rows 1 and 2 only (skip row 0)
    for i in (1, 2):
        lx, ly = locker_xy(i)
        parts.append(srect(lx, ly, box_w, box_h, rx=2,
                           fill=col_pink, stroke=col_outline))
    # Exec: rows 0 and 1 only (skip row 2)
    for i in (0, 1):
        ex, ey = exec_xy(i)
        parts.append(srect(ex, ey, box_w, box_h, rx=2,
                           fill=col_pink, stroke=col_outline))

    # ══════════════════════════════════════════════════════════════════
    # 3. LLM BOX
    # ══════════════════════════════════════════════════════════════════
    parts.append(srect(llm_x, llm_y, llm_w, llm_h,
                       rx=3, fill=col_blue, stroke=col_outline))

    # ══════════════════════════════════════════════════════════════════
    # 4. DIAGONAL ARROWS: Locker top-right → Exec bottom-left
    #    (all in outline color)
    # ══════════════════════════════════════════════════════════════════
    def diag_arrow(locker_row, exec_row):
        lx, ly = locker_xy(locker_row)
        ex, ey = exec_xy(exec_row)
        ax1 = lx + box_w        # top-right corner of locker
        ay1 = ly
        ax2 = ex                 # bottom-left corner of exec
        ay2 = ey + box_h
        dx = ax2 - ax1
        dy = ay2 - ay1
        d = math.hypot(dx, dy)
        ux, uy = dx / d, dy / d
        px, py = -uy, ux
        parts.append(
            f'<line x1="{R(ax1)}" y1="{R(ay1)}" '
            f'x2="{R(ax2 - ux * arrow_s)}" y2="{R(ay2 - uy * arrow_s)}" '
            f'stroke="{col_outline}" stroke-width="{sw}"/>')
        bx = ax2 - ux * arrow_s * 1.8
        by = ay2 - uy * arrow_s * 1.8
        pts = (f'{R(bx + px * arrow_s)},{R(by + py * arrow_s)} '
               f'{R(ax2)},{R(ay2)} '
               f'{R(bx - px * arrow_s)},{R(by - py * arrow_s)}')
        parts.append(f'<polygon points="{pts}" fill="{col_outline}"/>')

    diag_arrow(2, 1)   # bottom Locker → middle Exec
    diag_arrow(1, 0)   # middle Locker → top Exec

    # Line connecting bottom Locker to middle Locker (left column)
    _, ly1 = locker_xy(1)
    _, ly2 = locker_xy(2)
    lock_cx = cx0 + box_w / 2
    parts.append(sline(lock_cx, ly1 + box_h, lock_cx, ly2))

    # ══════════════════════════════════════════════════════════════════
    # 5. WIRING (all outline color)
    # ══════════════════════════════════════════════════════════════════
    arrow_top_y = margin
    arrow_bot_y = margin + 2 * wire_ext + outer_h

    # Vertical wire through Exec column
    # Top: from arrow down to Exec[0] top
    parts.append(sline(exec_cx, arrow_top_y + arrow_s * 1.8,
                       exec_cx, cy0))
    parts.append(arrow_up(exec_cx, arrow_top_y))

    # Between Exec[0] and Exec[1]
    _, ey0 = exec_xy(0)
    _, ey1 = exec_xy(1)
    parts.append(sline(exec_cx, ey0 + box_h, exec_cx, ey1))

    # Bottom: from Exec[1] bottom down to arrow
    parts.append(sline(exec_cx, ey1 + box_h, exec_cx, arrow_bot_y))
    parts.append(arrow_up(exec_cx, ey1 + box_h))

    # Top loop: (exec_cx, wire_top_y) → right → (llm_cx, wire_top_y) → down → (llm_cx, llm_y)
    parts.append(spoly([
        (exec_cx, wire_top_y),
        (llm_cx, wire_top_y),
        (llm_cx, llm_y),
    ]))

    # Bottom loop: LLM bottom → down → right to left → exec_cx → continue left → lock_cx → up to bottom Locker
    _, ly_bot_locker = locker_xy(2)
    parts.append(spoly([
        (llm_cx, llm_y + llm_h),
        (llm_cx, wire_bot_y),
        (lock_cx, wire_bot_y),
        (lock_cx, ly_bot_locker + box_h),
    ]))

    # ══════════════════════════════════════════════════════════════════
    # 6. LABELS
    # ══════════════════════════════════════════════════════════════════
    for i in (1, 2):  # Lockers in rows 1, 2
        lx, ly = locker_xy(i)
        parts.append(label(lx + box_w / 2, ly + box_h / 2, "Locker"))
    for i in (0, 1):  # Execs in rows 0, 1
        ex, ey = exec_xy(i)
        parts.append(label(ex + box_w / 2, ey + box_h / 2, "Exec"))

    parts.append(label(llm_x + llm_w / 2, llm_y + llm_h / 2, "LLM"))

    # ── Assemble SVG ──
    content = "\n    ".join(parts)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {R(total_w)} {R(total_h)}">\n'
        f'    {content}\n'
        f'</svg>'
    )
    return svg


if __name__ == "__main__":
    import pathlib
    import cairosvg

    svg = create_fig2_svg()
    out_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "fig2.svg"
    pdf_path = out_dir / "fig2.pdf"

    svg_path.write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))

    print(f"Wrote {svg_path}")
    print(f"Wrote {pdf_path}")

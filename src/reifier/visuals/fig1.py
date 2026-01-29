"""Generate an SVG figure illustrating the threat scenario (Figure 1).

Shows the flow: adversary embeds encrypted capability into host model →
model deployed → hidden capability activates on trigger.

Uses the same color scheme and aesthetic as exec_unit.py.
No text labels — colors and layout convey meaning.
"""


def create_fig1_svg() -> str:
    def R(v: float) -> float:
        return round(v, 2)

    # ── Colors (matching exec_unit.py palette) ──
    col_outline = "#4a4a6a"
    col_fill    = "#eeeef4"       # light gray fill for background groups
    col_blue    = "#b8ccee"       # blue – host model
    col_green   = "#b8e0b0"       # green – helpful / benign
    col_gold    = "#f0d8a0"       # gold – encrypted module
    col_danger  = "#f0c0c0"       # pink – malicious outcome
    col_mal     = "#c06060"       # red stroke/fill for malicious path
    col_help    = "#609060"       # green stroke for helpful path

    sw = 0.8  # stroke width
    arrow_s = 1.5  # arrowhead half-height

    # ── Layout constants ──
    pad = 4            # outer padding / inner padding in groups
    gap = 10           # horizontal gap between groups
    box_s = 12         # adversary shape size
    out_s = 10         # outcome shape size
    group_h = 40       # group height
    grx = 3            # group corner radius

    # Vertical center
    cy = pad + group_h / 2

    # ── Group sizing ──
    left_w = 2 * pad + box_s            # just the adversary
    mid_w = 44
    right_w = 2 * pad + out_s

    # X positions
    left_x = pad
    mid_x = left_x + left_w + gap
    right_x = mid_x + mid_w + gap
    total_w = right_x + right_w + pad
    total_h = group_h + 2 * pad

    parts: list[str] = []

    # ── Helpers ──
    def srect(x, y, w, h, rx=2, fill="none", stroke=None,
              stroke_dash=None, stroke_w=None):
        s = (f'<rect x="{R(x)}" y="{R(y)}" width="{R(w)}" '
             f'height="{R(h)}" rx="{rx}" fill="{fill}"')
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{stroke_w or sw}"'
        if stroke_dash:
            s += f' stroke-dasharray="{stroke_dash}"'
        s += '/>'
        return s

    def sline(x1, y1, x2, y2, stroke=None):
        st = stroke or col_outline
        return (f'<line x1="{R(x1)}" y1="{R(y1)}" x2="{R(x2)}" '
                f'y2="{R(y2)}" stroke="{st}" stroke-width="{sw}"/>')

    def arrow_r(x, y, fill=None):
        """Right-pointing arrowhead with tip at (x, y)."""
        f = fill or col_outline
        pts = (f'{R(x - arrow_s * 1.8)},{R(y - arrow_s)} '
               f'{R(x)},{R(y)} '
               f'{R(x - arrow_s * 1.8)},{R(y + arrow_s)}')
        return f'<polygon points="{pts}" fill="{f}"/>'

    # ══════════════════════════════════════════════════════════════════
    # 1. GROUP BACKGROUNDS (gray fill, no outline)
    # ══════════════════════════════════════════════════════════════════
    parts.append(srect(left_x, pad, left_w, group_h, rx=grx, fill=col_fill))
    parts.append(srect(mid_x, pad, mid_w, group_h, rx=grx, fill=col_fill))
    parts.append(srect(right_x, pad, right_w, group_h, rx=grx, fill=col_fill))

    # ══════════════════════════════════════════════════════════════════
    # 2. LEFT GROUP: adversary only
    # ══════════════════════════════════════════════════════════════════
    adv_x = left_x + pad
    adv_y = cy - box_s / 2
    parts.append(srect(adv_x, adv_y, box_s, box_s, rx=1.5,
                       fill=col_outline, stroke=col_outline))

    # ══════════════════════════════════════════════════════════════════
    # 3. MIDDLE GROUP: LLM (blue) with encrypted module (gold)
    # ══════════════════════════════════════════════════════════════════
    llm_pad = 4
    llm_x = mid_x + llm_pad
    llm_y = pad + llm_pad
    llm_w = mid_w - 2 * llm_pad
    llm_h = group_h - 2 * llm_pad

    # LLM outer rect (blue)
    parts.append(srect(llm_x, llm_y, llm_w, llm_h, rx=2,
                       fill=col_blue, stroke=col_outline))

    # Encrypted module inside (gold, centered)
    enc_pad = 5
    enc_x = llm_x + enc_pad
    enc_y = llm_y + enc_pad
    enc_w = llm_w - 2 * enc_pad
    enc_h = llm_h - 2 * enc_pad
    parts.append(srect(enc_x, enc_y, enc_w, enc_h, rx=1.5,
                       fill=col_gold, stroke=col_outline))

    # ══════════════════════════════════════════════════════════════════
    # 4. RIGHT GROUP: outcome boxes
    # ══════════════════════════════════════════════════════════════════
    top_y = cy - group_h / 4
    bot_y = cy + group_h / 4
    out_x = right_x + pad
    top_out_y = top_y - out_s / 2
    bot_out_y = bot_y - out_s / 2

    # Malicious outcome (pink, top)
    parts.append(srect(out_x, top_out_y, out_s, out_s, rx=1.5,
                       fill=col_danger, stroke=col_mal))
    # Helpful outcome (green, bottom)
    parts.append(srect(out_x, bot_out_y, out_s, out_s, rx=1.5,
                       fill=col_green, stroke=col_help))

    # ══════════════════════════════════════════════════════════════════
    # 5. ARROWS
    # ══════════════════════════════════════════════════════════════════

    # Adversary → encrypted module (straight horizontal at cy)
    a_x1 = adv_x + box_s
    a_x2 = enc_x
    parts.append(sline(a_x1, cy, a_x2, cy))
    parts.append(arrow_r(a_x2, cy))

    # Encrypted module → malicious outcome (red, exits enc right, turns up)
    enc_right = enc_x + enc_w
    enc_cy = enc_y + enc_h / 2
    # Turn point in the gap between middle and right groups
    turn_mal_x = mid_x + mid_w + (gap * 0.45)
    pts_mal = (f'{R(enc_right)},{R(enc_cy)} '
               f'{R(turn_mal_x)},{R(enc_cy)} '
               f'{R(turn_mal_x)},{R(top_y)} '
               f'{R(out_x)},{R(top_y)}')
    parts.append(f'<polyline points="{pts_mal}" fill="none" '
                 f'stroke="{col_mal}" stroke-width="{sw}"/>')
    parts.append(arrow_r(out_x, top_y, fill=col_mal))

    # Host model → helpful outcome (green, exits LLM right at bot_y)
    llm_right = llm_x + llm_w
    # Start at LLM right edge at bot_y (within LLM bounds)
    turn_help_x = mid_x + mid_w + (gap * 0.55)
    pts_help = (f'{R(llm_right)},{R(bot_y)} '
                f'{R(turn_help_x)},{R(bot_y)} '
                f'{R(out_x)},{R(bot_y)}')
    parts.append(f'<polyline points="{pts_help}" fill="none" '
                 f'stroke="{col_help}" stroke-width="{sw}"/>')
    parts.append(arrow_r(out_x, bot_y, fill=col_help))

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

    svg = create_fig1_svg()
    out_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "fig1.svg"
    pdf_path = out_dir / "fig1.pdf"

    svg_path.write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))

    print(f"Wrote {svg_path}")
    print(f"Wrote {pdf_path}")

"""Generate an SVG figure illustrating the threat scenario (Figure 1).

Shows the flow: adversary designs attack → embeds encrypted capability
in host model → model deployed → hidden capability activates.

Uses the same color scheme and aesthetic as exec_unit.py.
No text labels — colors and layout convey meaning.
"""


def create_fig1_svg() -> str:
    def R(v: float) -> float:
        return round(v, 2)

    # ── Colors (matching exec_unit.py palette + extensions) ──
    col_outline = "#4a4a6a"
    col_fill    = "#eeeef4"       # light fill for neutral boxes
    col_blue    = "#b8ccee"       # blue – model / weights
    col_green   = "#b8e0b0"       # green – helpful / benign
    col_gold    = "#f0d8a0"       # gold – encrypted module
    col_danger  = "#f0c0c0"       # pink – malicious outcome
    col_mal_stroke   = "#c06060"  # red stroke for malicious path
    col_help_stroke  = "#609060"  # green stroke for helpful path
    col_purple_bg    = "#e8e0f0"  # purple bg for attack design
    col_purple_str   = "#8070a0"  # purple stroke
    col_red_bg       = "#f8e8e8"  # red bg for deployment
    col_red_str      = "#c09090"  # red stroke
    col_gray_str     = "#a0a0b0"  # gray stroke for target model

    sw = 0.8  # stroke width

    # ── Layout constants ──
    pad = 4            # outer padding
    gap = 10           # horizontal gap between groups (for arrows)
    box_s = 12         # small shape size
    group_h = 40       # group height
    grx = 3            # group corner radius
    dash = "3,2"       # dash pattern for group borders
    arrow_s = 1.5      # arrowhead half-height

    # Vertical center
    cy = pad + group_h / 2

    # ── Group sizing ──
    inner_gap = 6                       # gap between shapes inside left group
    left_w = 2 * pad + box_s + inner_gap + box_s
    mid_w = 48
    right_w = left_w

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

    def sline(x1, y1, x2, y2, stroke=None, stroke_w=None):
        st = stroke or col_outline
        sww = stroke_w or sw
        return (f'<line x1="{R(x1)}" y1="{R(y1)}" x2="{R(x2)}" '
                f'y2="{R(y2)}" stroke="{st}" stroke-width="{sww}"/>')

    def arrow_r(x, y, fill=None):
        """Right-pointing arrowhead at tip (x, y)."""
        f = fill or col_outline
        pts = (f'{R(x - arrow_s * 1.8)},{R(y - arrow_s)} '
               f'{R(x)},{R(y)} '
               f'{R(x - arrow_s * 1.8)},{R(y + arrow_s)}')
        return f'<polygon points="{pts}" fill="{f}"/>'

    # ══════════════════════════════════════════════════════════════════
    # 1. GROUP BACKGROUNDS (dashed borders)
    # ══════════════════════════════════════════════════════════════════
    # Left: Attack Design (purple)
    parts.append(srect(left_x, pad, left_w, group_h, rx=grx,
                       fill=col_purple_bg, stroke=col_purple_str,
                       stroke_dash=dash))
    # Middle: Target Model (gray)
    parts.append(srect(mid_x, pad, mid_w, group_h, rx=grx,
                       fill=col_fill, stroke=col_gray_str,
                       stroke_dash=dash))
    # Right: Deployment (red)
    parts.append(srect(right_x, pad, right_w, group_h, rx=grx,
                       fill=col_red_bg, stroke=col_red_str,
                       stroke_dash=dash))

    # ══════════════════════════════════════════════════════════════════
    # 2. LEFT GROUP: adversary → malicious capability
    # ══════════════════════════════════════════════════════════════════
    adv_x = left_x + pad
    adv_y = cy - box_s / 2
    cap_x = adv_x + box_s + inner_gap
    cap_y = adv_y

    # Adversary (dark filled square)
    parts.append(srect(adv_x, adv_y, box_s, box_s, rx=1.5,
                       fill=col_outline, stroke=col_outline))
    # Malicious capability (gold square – the thing being encrypted)
    parts.append(srect(cap_x, cap_y, box_s, box_s, rx=1.5,
                       fill=col_gold, stroke=col_outline))

    # Arrow between adversary and capability
    a1_x1 = adv_x + box_s + 1
    a1_x2 = cap_x - 1
    parts.append(sline(a1_x1, cy, a1_x2, cy))
    parts.append(arrow_r(a1_x2, cy))

    # ══════════════════════════════════════════════════════════════════
    # 3. ARROW: left → middle
    # ══════════════════════════════════════════════════════════════════
    a2_x1 = left_x + left_w
    a2_x2 = mid_x
    parts.append(sline(a2_x1, cy, a2_x2, cy))
    parts.append(arrow_r(a2_x2, cy))

    # ══════════════════════════════════════════════════════════════════
    # 4. MIDDLE GROUP: LLM with encrypted module
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
    enc_pad = 6
    enc_x = llm_x + enc_pad
    enc_y = llm_y + enc_pad
    enc_w = llm_w - 2 * enc_pad
    enc_h = llm_h - 2 * enc_pad
    parts.append(srect(enc_x, enc_y, enc_w, enc_h, rx=1.5,
                       fill=col_gold, stroke=col_outline))

    # ══════════════════════════════════════════════════════════════════
    # 5. ARROWS: middle → right (diverging: malicious top, helpful bot)
    # ══════════════════════════════════════════════════════════════════
    fork_x = (mid_x + mid_w + right_x) / 2   # midpoint of the gap
    top_y = cy - group_h / 4
    bot_y = cy + group_h / 4

    # Top path (malicious – red)
    pts_top = (f'{R(mid_x + mid_w)},{R(cy)} '
               f'{R(fork_x)},{R(cy)} '
               f'{R(fork_x)},{R(top_y)} '
               f'{R(right_x)},{R(top_y)}')
    parts.append(f'<polyline points="{pts_top}" fill="none" '
                 f'stroke="{col_mal_stroke}" stroke-width="{sw}"/>')
    parts.append(arrow_r(right_x, top_y, fill=col_mal_stroke))

    # Bottom path (helpful – green)
    pts_bot = (f'{R(mid_x + mid_w)},{R(cy)} '
               f'{R(fork_x)},{R(cy)} '
               f'{R(fork_x)},{R(bot_y)} '
               f'{R(right_x)},{R(bot_y)}')
    parts.append(f'<polyline points="{pts_bot}" fill="none" '
                 f'stroke="{col_help_stroke}" stroke-width="{sw}"/>')
    parts.append(arrow_r(right_x, bot_y, fill=col_help_stroke))

    # Small dot at the fork point
    parts.append(f'<circle cx="{R(fork_x)}" cy="{R(cy)}" r="0.6" '
                 f'fill="{col_outline}"/>')

    # ══════════════════════════════════════════════════════════════════
    # 6. RIGHT GROUP: deployment outcomes
    # ══════════════════════════════════════════════════════════════════
    out_s = 10
    out_x_center = right_x + right_w / 2 + 1
    out_x = out_x_center - out_s / 2

    # Malicious outcome (pink, top)
    top_out_y = top_y - out_s / 2
    parts.append(srect(out_x, top_out_y, out_s, out_s, rx=1.5,
                       fill=col_danger, stroke=col_mal_stroke))

    # Helpful outcome (green, bottom)
    bot_out_y = bot_y - out_s / 2
    parts.append(srect(out_x, bot_out_y, out_s, out_s, rx=1.5,
                       fill=col_green, stroke=col_help_stroke))

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

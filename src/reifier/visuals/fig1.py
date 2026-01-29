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

    # ══════════════════════════════════════════════════════════════════
    # 6. ICONS
    # ══════════════════════════════════════════════════════════════════
    adv_cx = adv_x + box_s / 2
    adv_cy = adv_y + box_s / 2

    # ── Devil horns on adversary box ──
    # Build left horn shape, then mirror horizontally for right horn
    horn_h = 3.0
    horn_base = 1.0
    horn_spread = box_s * 0.28  # distance from center
    # Left horn (sign=-1): base on box top, curves outward left, tip inward
    lh_cx = adv_cx - horn_spread
    # Points relative to lh_cx: inner base, tip, outer base
    lb_inner = lh_cx + horn_base       # inner base (closer to center)
    lb_outer = lh_cx - horn_base       # outer base
    lt_x = lh_cx + horn_base * 0.3     # tip (curves inward)
    lt_y = adv_y - horn_h
    lcp_x = lh_cx - horn_base * 1.2    # control point (outward)
    lcp_y = adv_y - horn_h * 0.5
    # Left horn path: outer base → tip → inner base
    left_path = (
        f'M{R(lb_outer)},{R(adv_y)} '
        f'Q{R(lcp_x)},{R(lcp_y)} {R(lt_x)},{R(lt_y)} '
        f'Q{R(lcp_x + 0.3)},{R(lcp_y)} '
        f'{R(lb_inner)},{R(adv_y)} Z')
    parts.append(f'<path d="{left_path}" fill="{col_outline}"/>')
    # Right horn: mirror left horn around adv_cx
    def mirror_x(val):
        return adv_cx + (adv_cx - val)
    rb_inner = mirror_x(lb_inner)
    rb_outer = mirror_x(lb_outer)
    rt_x = mirror_x(lt_x)
    rcp_x = mirror_x(lcp_x)
    right_path = (
        f'M{R(rb_outer)},{R(adv_y)} '
        f'Q{R(rcp_x)},{R(lcp_y)} {R(rt_x)},{R(lt_y)} '
        f'Q{R(rcp_x - 0.3)},{R(lcp_y)} '
        f'{R(rb_inner)},{R(adv_y)} Z')
    parts.append(f'<path d="{right_path}" fill="{col_outline}"/>')

    # ── Closed padlock in encrypted module ──
    enc_cx = enc_x + enc_w / 2
    enc_cy = enc_y + enc_h / 2
    lock_bw = 5.0    # body width
    lock_bh = 4.0    # body height
    lock_br = 0.5    # body corner radius
    lock_by = enc_cy - 0.5  # body top y (slightly below center)
    lock_bx = enc_cx - lock_bw / 2
    # Body
    parts.append(srect(lock_bx, lock_by, lock_bw, lock_bh, rx=lock_br,
                       fill=col_outline, stroke=col_outline))
    # Shackle (U-shaped arc above body)
    shackle_w = lock_bw * 0.55
    shackle_h = 3.0
    sx1 = enc_cx - shackle_w / 2
    sx2 = enc_cx + shackle_w / 2
    parts.append(
        f'<path d="M{R(sx1)},{R(lock_by)} '
        f'L{R(sx1)},{R(lock_by - shackle_h * 0.4)} '
        f'A{R(shackle_w / 2)},{R(shackle_h * 0.6)} 0 0,1 '
        f'{R(sx2)},{R(lock_by - shackle_h * 0.4)} '
        f'L{R(sx2)},{R(lock_by)}" '
        f'fill="none" stroke="{col_outline}" stroke-width="{sw * 1.5}" '
        f'stroke-linecap="round"/>')
    # Keyhole (small circle + downward triangle/slit)
    kh_cy = lock_by + lock_bh * 0.4
    kh_r = 0.7
    parts.append(f'<circle cx="{R(enc_cx)}" cy="{R(kh_cy)}" '
                 f'r="{kh_r}" fill="{col_gold}"/>')
    kh_slit_h = lock_bh * 0.3
    kh_slit_w = kh_r * 0.7
    parts.append(
        f'<rect x="{R(enc_cx - kh_slit_w / 2)}" y="{R(kh_cy)}" '
        f'width="{R(kh_slit_w)}" height="{R(kh_slit_h)}" '
        f'fill="{col_gold}"/>')

    # ── Bug/insect icon in red (malicious) box ──
    bug_cx = out_x + out_s / 2
    bug_cy = top_y
    # Body (oval)
    bug_bry = 2.2
    bug_brx = 1.4
    parts.append(f'<ellipse cx="{R(bug_cx)}" cy="{R(bug_cy + 0.5)}" '
                 f'rx="{bug_brx}" ry="{bug_bry}" '
                 f'fill="{col_mal}" stroke="{col_mal}" stroke-width="0.4"/>')
    # Head (small circle)
    head_r = 0.8
    parts.append(f'<circle cx="{R(bug_cx)}" cy="{R(bug_cy - bug_bry + 0.1)}" '
                 f'r="{head_r}" fill="{col_mal}"/>')
    # Legs (3 pairs, symmetric)
    leg_len = 1.6
    for i, dy in enumerate([-0.8, 0.5, 1.8]):
        ly = bug_cy + dy
        splay = 0.6 + i * 0.2  # legs splay out slightly more toward back
        for sx in (-1, 1):
            lx1 = bug_cx + sx * bug_brx
            lx2 = lx1 + sx * leg_len
            ly2 = ly + splay
            parts.append(f'<line x1="{R(lx1)}" y1="{R(ly)}" '
                         f'x2="{R(lx2)}" y2="{R(ly2)}" '
                         f'stroke="{col_mal}" stroke-width="0.5" '
                         f'stroke-linecap="round"/>')
    # Antennae (2 short lines from head)
    ant_len = 1.4
    head_y = bug_cy - bug_bry + 0.1
    for sx in (-1, 1):
        parts.append(f'<line x1="{R(bug_cx)}" y1="{R(head_y)}" '
                     f'x2="{R(bug_cx + sx * 1.0)}" y2="{R(head_y - ant_len)}" '
                     f'stroke="{col_mal}" stroke-width="0.5" '
                     f'stroke-linecap="round"/>')

    # ── </> code icon in green (helpful) box ──
    code_cx = out_x + out_s / 2
    code_cy = bot_y
    bkt_w = 1.2   # bracket horizontal extent (how far < tip goes)
    bkt_h = 1.8   # bracket vertical extent
    bkt_sp = 7.0  # total horizontal spread between < tip and > tip
    # < bracket (left side)
    lx = code_cx - bkt_sp / 2
    parts.append(f'<polyline points="'
                 f'{R(lx + bkt_w)},{R(code_cy - bkt_h)} '
                 f'{R(lx)},{R(code_cy)} '
                 f'{R(lx + bkt_w)},{R(code_cy + bkt_h)}" '
                 f'fill="none" stroke="{col_help}" stroke-width="0.7" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
    # > bracket (right side)
    rx = code_cx + bkt_sp / 2
    parts.append(f'<polyline points="'
                 f'{R(rx - bkt_w)},{R(code_cy - bkt_h)} '
                 f'{R(rx)},{R(code_cy)} '
                 f'{R(rx - bkt_w)},{R(code_cy + bkt_h)}" '
                 f'fill="none" stroke="{col_help}" stroke-width="0.7" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
    # / slash (centered, slightly taller than brackets)
    slash_h = bkt_h * 0.9
    parts.append(f'<line x1="{R(code_cx + 0.5)}" y1="{R(code_cy - slash_h)}" '
                 f'x2="{R(code_cx - 0.5)}" y2="{R(code_cy + slash_h)}" '
                 f'stroke="{col_help}" stroke-width="0.7" '
                 f'stroke-linecap="round"/>')

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

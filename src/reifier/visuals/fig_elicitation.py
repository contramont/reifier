"""Generate an SVG/PDF plot of accuracy vs fine-tuning steps.

Compares passworded model (elicitable) vs encrypted model (resistant).
Data averaged over 3 noise scales with error bands (±1 std).

Uses the same color scheme and aesthetic as fig1.py / fig2.py / exec_unit.py.
"""
import numpy as np


def create_elicitation_plot() -> str:
    def R(v: float) -> float:
        return round(v, 2)

    # ── Raw data (3 noise scales: 0.0, 0.01, 0.1) ──
    steps = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

    pwd_acc = np.array([
        [0.5410, 0.4961, 0.4766],
        [0.8750, 0.8164, 0.7754],
        [0.8574, 0.8262, 0.8008],
        [0.8711, 0.8594, 0.8477],
        [0.8574, 0.8633, 0.8340],
        [0.8633, 0.8789, 0.8438],
        [0.8809, 0.8750, 0.8594],
        [0.8672, 0.8945, 0.8750],
        [0.8848, 0.8789, 0.8262],
        [0.8574, 0.8750, 0.8555],
        [0.8926, 0.8848, 0.8945],
    ])

    enc_acc = np.array([
        [0.5469, 0.5312, 0.5312],
        [0.4688, 0.4844, 0.5156],
        [0.5312, 0.5312, 0.5156],
        [0.6406, 0.4375, 0.5156],
        [0.6406, 0.4375, 0.5312],
        [0.4844, 0.5625, 0.4688],
        [0.4844, 0.5781, 0.5938],
        [0.6250, 0.5469, 0.5156],
        [0.5000, 0.5156, 0.5625],
        [0.4688, 0.4375, 0.4375],
        [0.4688, 0.5469, 0.4844],
    ])

    # Average and std
    pwd_mean = pwd_acc.mean(axis=1)
    pwd_std = pwd_acc.std(axis=1)
    enc_mean = enc_acc.mean(axis=1)
    enc_std = enc_acc.std(axis=1)

    # ── Colors ──
    col_outline = "#4a4a6a"
    col_pwd     = "#c06060"       # red – passworded (elicitable)
    col_pwd_fill = "#f0c0c0"      # light red for error band
    col_enc     = "#4a7a9a"       # steel blue – encrypted (resistant)
    col_enc_fill = "#b8d8e8"      # light blue for error band
    col_grid    = "#d0d0d8"       # light gray for grid/reference
    font = "Latin Modern Roman, CMU Serif, serif"

    # ── Plot geometry ──
    sw = 0.6           # line stroke width
    data_sw = 1.0      # data line stroke width
    margin_l = 14      # left margin for y-axis labels
    margin_r = 4
    margin_t = 4
    margin_b = 12      # bottom margin for x-axis labels
    plot_w = 120
    plot_h = 60
    total_w = margin_l + plot_w + margin_r
    total_h = margin_t + plot_h + margin_b

    # Data range
    x_min, x_max = 0, 500
    y_min, y_max = 0.40, 1.00

    def to_px(step, acc):
        px = margin_l + (step - x_min) / (x_max - x_min) * plot_w
        py = margin_t + (1 - (acc - y_min) / (y_max - y_min)) * plot_h
        return R(px), R(py)

    parts: list[str] = []

    # ── Plot background ──
    # (no background rect — keep it minimal on white)

    # ── 50% reference line (dashed gray) ──
    _, y50 = to_px(0, 0.50)
    parts.append(
        f'<line x1="{R(margin_l)}" y1="{y50}" '
        f'x2="{R(margin_l + plot_w)}" y2="{y50}" '
        f'stroke="{col_grid}" stroke-width="{sw}" '
        f'stroke-dasharray="3,2"/>')

    # ── Error bands (filled polygons) ──
    def error_band(mean, std, fill_color):
        upper = np.clip(mean + std, y_min, y_max)
        lower = np.clip(mean - std, y_min, y_max)
        # Forward path (upper), then reverse path (lower)
        pts_upper = [to_px(s, u) for s, u in zip(steps, upper)]
        pts_lower = [to_px(s, l) for s, l in zip(steps, lower)]
        all_pts = pts_upper + pts_lower[::-1]
        pts_str = " ".join(f'{x},{y}' for x, y in all_pts)
        return (f'<polygon points="{pts_str}" '
                f'fill="{fill_color}" opacity="0.4"/>')

    parts.append(error_band(pwd_mean, pwd_std, col_pwd_fill))
    parts.append(error_band(enc_mean, enc_std, col_enc_fill))

    # ── Data lines ──
    def data_line(mean, color):
        pts = [to_px(s, m) for s, m in zip(steps, mean)]
        pts_str = " ".join(f'{x},{y}' for x, y in pts)
        return (f'<polyline points="{pts_str}" fill="none" '
                f'stroke="{color}" stroke-width="{data_sw}" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')

    parts.append(data_line(pwd_mean, col_pwd))
    parts.append(data_line(enc_mean, col_enc))

    # ── Data points (small dots) ──
    dot_r = 0.8
    for s, m in zip(steps, pwd_mean):
        px, py = to_px(s, m)
        parts.append(f'<circle cx="{px}" cy="{py}" r="{dot_r}" fill="{col_pwd}"/>')
    for s, m in zip(steps, enc_mean):
        px, py = to_px(s, m)
        parts.append(f'<circle cx="{px}" cy="{py}" r="{dot_r}" fill="{col_enc}"/>')

    # ── Axes ──
    ax_x1 = R(margin_l)
    ax_x2 = R(margin_l + plot_w)
    ax_y1 = R(margin_t)
    ax_y2 = R(margin_t + plot_h)
    # Left axis
    parts.append(f'<line x1="{ax_x1}" y1="{ax_y1}" x2="{ax_x1}" y2="{ax_y2}" '
                 f'stroke="{col_outline}" stroke-width="{sw}"/>')
    # Bottom axis
    parts.append(f'<line x1="{ax_x1}" y1="{ax_y2}" x2="{ax_x2}" y2="{ax_y2}" '
                 f'stroke="{col_outline}" stroke-width="{sw}"/>')

    # ── Y-axis ticks and labels ──
    fs_tick = 2.8
    tick_len = 1.5
    for acc_val in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        _, ty = to_px(0, acc_val)
        parts.append(
            f'<line x1="{R(margin_l - tick_len)}" y1="{ty}" '
            f'x2="{ax_x1}" y2="{ty}" '
            f'stroke="{col_outline}" stroke-width="{sw}"/>')
        pct = int(acc_val * 100)
        parts.append(
            f'<text x="{R(margin_l - tick_len - 1)}" y="{R(float(ty) + fs_tick * 0.35)}" '
            f'text-anchor="end" font-family="{font}" '
            f'font-size="{fs_tick}" fill="{col_outline}">{pct}%</text>')

    # ── X-axis ticks and labels ──
    for step_val in [0, 100, 200, 300, 400, 500]:
        tx, _ = to_px(step_val, 0)
        parts.append(
            f'<line x1="{tx}" y1="{ax_y2}" '
            f'x2="{tx}" y2="{R(float(ax_y2) + tick_len)}" '
            f'stroke="{col_outline}" stroke-width="{sw}"/>')
        parts.append(
            f'<text x="{tx}" y="{R(float(ax_y2) + tick_len + fs_tick + 0.5)}" '
            f'text-anchor="middle" font-family="{font}" '
            f'font-size="{fs_tick}" fill="{col_outline}">{step_val}</text>')

    # ── Axis titles ──
    fs_title = 3.2
    # X-axis title
    parts.append(
        f'<text x="{R(margin_l + plot_w / 2)}" '
        f'y="{R(total_h - 0.5)}" '
        f'text-anchor="middle" font-family="{font}" '
        f'font-size="{fs_title}" fill="{col_outline}">Fine-tuning steps</text>')
    # Y-axis title (rotated)
    parts.append(
        f'<text x="{R(2.5)}" y="{R(margin_t + plot_h / 2)}" '
        f'text-anchor="middle" font-family="{font}" '
        f'font-size="{fs_title}" fill="{col_outline}" '
        f'transform="rotate(-90, 2.5, {R(margin_t + plot_h / 2)})">Accuracy</text>')

    # ── Legend ──
    leg_x = margin_l + plot_w - 38
    leg_y = margin_t + 4
    leg_line_w = 6
    leg_gap = 5
    # Passworded
    parts.append(
        f'<line x1="{R(leg_x)}" y1="{R(leg_y)}" '
        f'x2="{R(leg_x + leg_line_w)}" y2="{R(leg_y)}" '
        f'stroke="{col_pwd}" stroke-width="{data_sw}"/>')
    parts.append(
        f'<text x="{R(leg_x + leg_line_w + 1.5)}" y="{R(leg_y + fs_tick * 0.35)}" '
        f'font-family="{font}" font-size="{fs_tick}" '
        f'fill="{col_outline}">Passworded</text>')
    # Encrypted
    parts.append(
        f'<line x1="{R(leg_x)}" y1="{R(leg_y + leg_gap)}" '
        f'x2="{R(leg_x + leg_line_w)}" y2="{R(leg_y + leg_gap)}" '
        f'stroke="{col_enc}" stroke-width="{data_sw}"/>')
    parts.append(
        f'<text x="{R(leg_x + leg_line_w + 1.5)}" '
        f'y="{R(leg_y + leg_gap + fs_tick * 0.35)}" '
        f'font-family="{font}" font-size="{fs_tick}" '
        f'fill="{col_outline}">Encrypted</text>')

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

    svg = create_elicitation_plot()
    out_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "fig_elicitation.svg"
    pdf_path = out_dir / "fig_elicitation.pdf"

    svg_path.write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))

    print(f"Wrote {svg_path}")
    print(f"Wrote {pdf_path}")

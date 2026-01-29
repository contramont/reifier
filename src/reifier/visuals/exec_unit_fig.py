"""SwiGLU execution unit diagram for the paper.

Layout:  output bar → arrow → Wo → ⊗ ← SiLU ← Wg
                                ↑               ↑
                               Wv ──────────────┘
                                ↑
                             input bar

Uses the fig toolkit for SVG/PDF generation.
"""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))
from fig import Fig, P, Rect, OUTLINE, FILL, BLUE, GREEN, GOLD, SANS
import math

# ── Layout (all integer structural coordinates) ────────────────────────
cell = 16                     # block width = height
ps   = 2                      # side padding
pm   = 4                      # mid padding between columns
wire = 6                      # block edge ↔ circle center
ah   = 3                      # arrowhead height
aw   = 2                      # arrowhead half-width
tail = 4                      # stub wire at top and bottom
cr   = 2                      # circle radius (⊗, SiLU)

cx = ps + cell // 2                       # 10  left column center x
rx = cx + cell // 2 + pm + cell // 2      # 30  right column center x

# Bar dimensions (all integers)
vh  = 2                       # bar height
vr  = 0                       # bar corner radius (crisp)
bg  = 1                       # gap between bar and nearest element

# Bar widths: total input = 12, split [W=8 | gap=1 | x=3]
mid_w  = 8                    # blue bar width / gold bar width
gbar   = 2                    # single green bar width
ggap   = 1                    # gap between green bars

# Vertical positions (top to bottom).
# y=0 is the top of the output bar; everything below is positive.
out_y    = 0                  # output bar top
tip_y    = vh + bg            # 3   arrowhead tip
abase_y  = tip_y + ah         # 6   arrowhead base
wo_y     = abase_y + tail     # 10  Wo block top
m_y      = wo_y + cell + wire # 32  ⊗ / SiLU center
wv_y     = m_y + wire         # 38  Wv / Wg block top
branch_y = wv_y + cell + wire # 60  branch split
bot_y    = branch_y + tail    # 64  bottom wire end
in_y     = bot_y + bg         # 65  input bar top

# ── Build figure ─────────────────────────────────────────────────────
fig = Fig(crisp=True)
fig.style('*{stroke-linecap:square}')

# Weight blocks
wo = fig.rect(cx - cell // 2, wo_y, cell, cell, rx=1, fill=FILL)
wv = fig.rect(cx - cell // 2, wv_y, cell, cell, rx=1, fill=FILL)
wg = fig.rect(rx - cell // 2, wv_y, cell, cell, rx=1, fill=FILL)

# Operation circles
fig.circle(P(cx, m_y), cr)
fig.circle(P(rx, m_y), cr)

# Arrowhead
fig.polygon(P(cx - aw, abase_y), P(cx, tip_y), P(cx + aw, abase_y))

# Vertical wires
fig.line(P(cx, abase_y - 1), wo.top)         # arrow stem → Wo
fig.line(wo.bot, P(cx, m_y - cr))             # Wo → ⊗
fig.line(P(cx, m_y + cr), wv.top)             # ⊗ → Wv
fig.line(wv.bot, P(cx, bot_y))                # Wv → bottom (through branch)
fig.line(P(rx, m_y + cr), wg.top)             # SiLU → Wg

# Horizontal wire: ⊗ → SiLU
fig.line(P(cx + cr, m_y), P(rx - cr, m_y))

# Branch: split right to Wg bottom
fig.polyline(P(cx, branch_y), P(rx, branch_y), wg.bot)

# SiLU icon (bent line)
s = cr * math.sqrt(2) / 2
fig.polyline(P(rx - cr, m_y), P(rx, m_y), P(rx + s, m_y - s))

# ⊗ dot (filled, no stroke)
fig.circle(P(cx, m_y), 0.2, fill=OUTLINE, stroke="none")

# ── Activation bars ──────────────────────────────────────────────────
def bar(x: int, y: int, w: int, fill: str) -> None:
    fig.add(Rect(x, y, w, vh, vr, fill, stroke="none"))

# Output bar (gold) — at y=0
out_w = 4
bar(cx - out_w // 2, out_y, out_w, GOLD)

# Input bar [W | x] — below bottom wire, total width 12 centered on cx
in_start = cx - 6                         # 4
bar(in_start, in_y, mid_w, BLUE)          # blue: 4..12
bar(in_start + mid_w + ggap, in_y, 3, GREEN)  # green: 13..16

# W bar (blue) — between ⊗.bot and Wv.top, centered vertically
mid_bar_y = m_y + cr + (wire - cr - vh) // 2  # 35
bar(cx - mid_w // 2, mid_bar_y, mid_w, BLUE)

# f(x) bars (×3 green) — same y, between SiLU.bot and Wg.top
fx_start = rx - mid_w // 2               # 26
for i in range(3):
    bar(fx_start + i * (gbar + ggap), mid_bar_y, gbar, GREEN)

# Partial-product bar (gold) — between Wo.bot and ⊗.top, centered
pp_bar_y = wo_y + cell + (wire - cr - vh) // 2  # 27
bar(cx - mid_w // 2, pp_bar_y, mid_w, GOLD)

# ── Block labels ─────────────────────────────────────────────────────
fs = 4
for block, sub in [(wo, "o"), (wv, "v"), (wg, "g")]:
    fig.text(block.center, "W", sub=sub, size=fs, font=SANS, weight="bold")

# ── Save ─────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent.parent.parent / "images" / "exec_unit"
total_w = 2 * ps + 2 * cell + pm             # 40
fig.save(out, viewbox=(0, -0.1, total_w, in_y + vh + 0.2), pad=0)

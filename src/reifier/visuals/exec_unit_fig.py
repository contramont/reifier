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

# ── Layout (integer structural coordinates) ──────────────────────────
cell = 16                     # block width = height
ps   = 2                      # side padding
pm   = 4                      # mid padding between columns
wire = 6                      # block edge ↔ circle center
ah   = 3                      # arrowhead height
aw   = 2                      # arrowhead half-width
tail = 4                      # stub wire at top and bottom
cr   = 2                      # circle radius (⊗, SiLU)
bar_gap = 1                   # gap between bar and nearest wire/arrow

cx = ps + cell // 2                       # left column center x  (10)
rx = cx + cell // 2 + pm + cell // 2      # right column center x (30)

# Vertical positions (top to bottom, all integers).
# tip_y chosen so the output bar above it has positive y coordinates.
tip_y    = ah                 # 3
abase_y  = tip_y + ah        # 6
wo_y     = abase_y + tail    # 10
m_y      = wo_y + cell + wire # 32  (⊗ center)
wv_y     = m_y + wire         # 38
branch_y = wv_y + cell + wire # 60  (branch split)
bot_y    = branch_y + tail    # 64

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
# Bars are decorative overlays — no stroke, just fill.
vh = 1.6          # bar height
vr = 0.4          # bar corner radius

# Bar widths: input = [W=9 | x=3], total = 12
in_w  = 12
part  = in_w // 4                             # 3  (one slot)
w_w   = in_w - part                           # 9  (W portion = mid bar width)
gfrac = 0.12                                  # gap fraction per green slot
gbar  = part * (1 - gfrac)                    # 2.64  visible green bar
ggap  = part * gfrac                          # 0.36  gap

def bar(x: float, y: float, w: float, fill: str) -> None:
    fig.add(Rect(x, y, w, vh, vr, fill, stroke="none"))

# Output bar — above arrowhead
out_w = cell * 0.25                           # 4
bar(cx - out_w / 2, tip_y - bar_gap - vh, out_w, GOLD)

# Input bar [W | x] — below bottom wire, matching the top visual gap
bot_gap = bar_gap + ah                        # same distance as bar→arrowbase
bar(cx - in_w / 2, bot_y + bot_gap, w_w - ggap, BLUE)
bar(cx - in_w / 2 + w_w + ggap / 2, bot_y + bot_gap, gbar, GREEN)

# W bar — between ⊗.bot and Wv.top
mid_bar_y = (m_y + cr + wv_y) / 2 - vh / 2
bar(cx - w_w / 2, mid_bar_y, w_w, BLUE)

# f(x) bars (×3) — between SiLU.bot and Wg.top, same y
for i in range(3):
    bar(rx - w_w / 2 + i * part + ggap / 2, mid_bar_y, gbar, GREEN)

# Partial-product bar — between Wo.bot and ⊗.top
pp_bar_y = (wo_y + cell + m_y - cr) / 2 - vh / 2
bar(cx - w_w / 2, pp_bar_y, w_w, GOLD)

# ── Block labels ─────────────────────────────────────────────────────
fs = 3.64
for block, sub in [(wo, "o"), (wv, "v"), (wg, "g")]:
    fig.text(block.center, "W", sub=sub, size=fs, font=SANS, weight="bold")

# ── Save ─────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent.parent.parent / "images" / "exec_unit"
total_w = 2 * ps + 2 * cell + pm             # 40
vb_top = tip_y - bar_gap - vh - 0.5
vb_bot = bot_y + bot_gap + vh + 0.5
fig.save(out, viewbox=(0, vb_top, total_w, vb_bot - vb_top))

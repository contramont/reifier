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

# ── Layout (all key positions are integers) ──────────────────────────
cell = 16                     # block width = height
ps   = 2                      # side padding
pm   = 4                      # mid padding between columns
wire = 6                      # block edge ↔ circle center
ah   = 3                      # arrowhead height
aw   = 2                      # arrowhead half-width
tail = 4                      # stub wire at top and bottom
cr   = 2                      # circle radius (⊗, SiLU)

cx   = ps + cell // 2         # left column center x  (10)
rx   = cx + cell // 2 + pm + cell // 2   # right column center x (30)

# Vertical positions (top to bottom, all integers)
tip_y    = 0
abase_y  = ah                 # 3
wo_y     = abase_y + tail     # 7
m_y      = wo_y + cell + wire # 29  (⊗ center)
wv_y     = m_y + wire         # 35
branch_y = wv_y + cell + wire # 57  (branch split)
bot_y    = branch_y + tail    # 61

# ── Build figure ─────────────────────────────────────────────────────
fig = Fig(crisp=True)
fig.style("line, polyline, circle, rect { stroke-linecap: square }")

# Weight blocks
wo = fig.rect(cx - cell // 2, wo_y, cell, cell, rx=1, fill=FILL)
wv = fig.rect(cx - cell // 2, wv_y, cell, cell, rx=1, fill=FILL)
wg = fig.rect(rx - cell // 2, wv_y, cell, cell, rx=1, fill=FILL)

# Operation circles
m  = fig.circle(P(cx, m_y), cr)
f  = fig.circle(P(rx, m_y), cr)

# Arrowhead (filled triangle)
fig.polygon(P(cx - aw, abase_y), P(cx, tip_y), P(cx + aw, abase_y))

# Vertical wires
fig.line(P(cx, abase_y - 1), wo.top)           # arrow stem → Wo
fig.line(wo.bot, m.top)                          # Wo → ⊗
fig.line(m.bot, wv.top)                          # ⊗ → Wv
fig.line(wv.bot, P(cx, bot_y))                   # Wv → bottom (merged through branch)
fig.line(f.bot, wg.top)                          # SiLU → Wg

# Horizontal wire: ⊗ → SiLU
fig.line(m.right, f.left)

# Branch: split right to Wg bottom
fig.polyline(P(cx, branch_y), P(rx, branch_y), P(rx, wg.bot.y))

# SiLU icon (bent line)
shift = cr * math.sqrt(2) / 2
fig.polyline(f.left, f.center, P(rx + shift, m_y - shift))

# ⊗ dot
fig.circle(P(cx, m_y), 0.2, fill=OUTLINE, stroke=OUTLINE, stroke_w=0)

# ── Activation bars ──────────────────────────────────────────────────
vh  = 1.6        # bar height
vr  = 0.4        # bar corner radius
gap = 1          # gap between wire endpoint and bar edge

# Bar widths: input = [W=9 | x=3], 4 parts of 3 each
in_w  = 12
parts = in_w // 4                           # 3  (width of one part)
w_w   = in_w - parts                        # 9  (W portion)
mid_w = w_w                                 # 9  (middle bar total width)

# Green bars: 3 bars with small gaps
gfrac = 0.12                                # gap fraction per slot
gbar  = parts * (1 - gfrac)                 # 2.64  visible bar
ggap  = parts * gfrac                       # 0.36  gap

def bar(x: float, y: float, w: float, fill: str) -> None:
    fig.add(Rect(x, y, w, vh, vr, fill))

# Output bar — above arrowhead
out_w = cell * 0.25                         # 4
bar(cx - out_w / 2, tip_y - gap - vh, out_w, GOLD)

# Input bar [W | x] — below bottom wire
in_x = cx - in_w / 2
bar(in_x, bot_y + gap, w_w - ggap, BLUE)                     # W part
bar(in_x + w_w + ggap / 2, bot_y + gap, gbar, GREEN)         # x part

# Middle W bar — between ⊗.bot and Wv.top
mid_bar_y = (m_y + cr + wv_y) / 2 - vh / 2
bar(cx - mid_w / 2, mid_bar_y, mid_w, BLUE)

# Middle green bars (×3) — between SiLU.bot and Wg.top, same y
for i in range(3):
    bar(rx - mid_w / 2 + i * parts + ggap / 2, mid_bar_y, gbar, GREEN)

# Partial-product bar — between Wo.bot and ⊗.top
pp_bar_y = (wo_y + cell + m_y - cr) / 2 - vh / 2
bar(cx - mid_w / 2, pp_bar_y, mid_w, GOLD)

# ── Block labels ─────────────────────────────────────────────────────
fs = 3.64
for block, sub in [(wo, "o"), (wv, "v"), (wg, "g")]:
    fig.text(block.center, "W", sub=sub, size=fs, font=SANS, weight="bold")

# ── Save ─────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent.parent.parent / "images" / "exec_unit"
# Explicit viewbox: full width, tight vertical with 0.5 padding
vb_top = tip_y - gap - vh - 0.5
vb_bot = bot_y + gap + vh + 0.5
total_w = 2 * ps + 2 * cell + pm    # 40
fig.save(out, viewbox=(0, vb_top, total_w, vb_bot - vb_top))

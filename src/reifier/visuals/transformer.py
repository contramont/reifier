"""Simplified transformer block diagram using the fig toolkit."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))
from fig import Fig, P, OUTLINE, FILL, BLUE, GREEN, GOLD, PINK, SERIF

fig = Fig(stroke_w=0.8)

# ── Layout ───────────────────────────────────────────────────────────
bw, bh = 24, 10          # block width, height
rx = 2
gap = 3                   # vertical gap between blocks
skip_dx = 5               # skip connection offset from right edge
fs = 2.8

c_norm, c_attn, c_ffn, c_add = FILL, BLUE, GOLD, GREEN

# Stack top-to-bottom: Add, FFN, Norm, Add, Attn, Norm
labels = ["Norm", "Attention", "Add", "Norm", "FFN", "Add"]
colors = [c_norm, c_attn,      c_add, c_norm, c_ffn, c_add]

x0 = 0
blocks = []
y = 0
for i in reversed(range(len(labels))):
    r = fig.rect(x0, y, bw, bh, rx=rx, fill=colors[i])
    fig.text(r.center, labels[i], size=fs, font=SERIF)
    blocks.insert(0, r)
    y += bh + gap

# blocks: [0]=Norm, [1]=Attn, [2]=Add, [3]=Norm, [4]=FFN, [5]=Add

# ── Wires between blocks ────────────────────────────────────────────
for i in range(len(blocks) - 1):
    fig.line(blocks[i].top, blocks[i + 1].bot)

# ── Input / output ──────────────────────────────────────────────────
tail = 5
fig.line(blocks[0].bot, blocks[0].bot + P(0, tail))
fig.arrow(blocks[-1].top + P(0, -tail), blocks[-1].top)

# ── Residual skip connections (right side) ───────────────────────────
def skip(src: P, dst: P) -> None:
    """Vertical skip on the right: from src down/up to dst."""
    sx = x0 + bw + skip_dx
    fig.polyline(src, P(sx, src.y), P(sx, dst.y), dst)

# Skip 1: input of Norm[0] bypasses Norm+Attn → right side of Add[2]
skip(blocks[0].bot + P(bw / 2, 0), blocks[2].right)
# Skip 2: input of Norm[3] bypasses Norm+FFN → right side of Add[5]
skip(blocks[3].bot + P(bw / 2, 0), blocks[5].right)

# ── ×N bracket on the left ──────────────────────────────────────────
mid_y = (blocks[0].bot.y + blocks[-1].top.y) / 2
fig.text(P(x0 - 5, mid_y), "×N", size=fs, font=SERIF)
bk_x = x0 - 2.5
fig.line(P(bk_x, blocks[-1].top.y), P(bk_x, blocks[0].bot.y), stroke_w=0.5)
fig.line(P(bk_x, blocks[-1].top.y), P(bk_x + 1, blocks[-1].top.y), stroke_w=0.5)
fig.line(P(bk_x, blocks[0].bot.y), P(bk_x + 1, blocks[0].bot.y), stroke_w=0.5)

# ── Save ─────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent.parent.parent / "images" / "transformer"
fig.save(out, pad=3)

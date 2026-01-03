from dataclasses import dataclass
from typing import Self, ClassVar

import torch as t


@dataclass(frozen=True)
class DistrTransform:
    """Maps values to [-0.5, 0.5] with linear core and atan-compressed tails."""
    threshold: float
    inlier_proportion: float = 0.8
    std_factor: ClassVar[float] = 4.0

    @classmethod
    def from_distrs(cls, *distrs: t.Tensor, **kw: float) -> Self:
        combined = t.cat([d.flatten() for d in distrs])
        eps = t.finfo(combined.dtype).eps
        thr = max(cls.std_factor * combined.std().item(), eps)
        return cls(thr, **kw)

    def __call__(self, x: t.Tensor) -> t.Tensor:
        x = x.cpu().float()
        thr, mid = self.threshold, self.inlier_proportion / 2
        linear = x * (mid / thr)
        tail = t.sign(x) * (mid + (0.5 - mid) * (2 / t.pi) * t.atan((t.abs(x) - thr) / thr))
        return linear.where(t.abs(x) <= thr, tail)


@dataclass(frozen=True)
class DistrPlotter:
    bins: int = 100
    min_precision: int = 100
    aspect_ratio: int = 5
    col = 'oklch(0.35 0.05 235)'
    col_a = 'oklch(0.832 0.14 57)'
    col_b = 'oklch(0.7835 0.14 235)'

    @property
    def _dims(self) -> tuple[int, int, int]:
        """Sets integer valued coordinates. Ensures the number of possible
        rounded height values to be >= min_precision."""
        coef = -(-self.min_precision // self.bins)  # ceil div
        h = coef * self.bins
        w = h * self.aspect_ratio
        return h, w, w//self.bins

    def _histogram(self, distr: t.Tensor, transform: DistrTransform) -> t.Tensor:
        """Normalized histogram counts in [0, 1]."""
        counts, _ = t.histogram(transform(distr), bins=self.bins, range=(-0.5, 0.5))
        return counts / (counts.max().item() or 1)

    def _bars(self, counts: t.Tensor, color: str) -> str:
        h, _, w_bar = self._dims
        scaled: list[int] = (h * counts).round().long().tolist()
        return "".join(
            f'\n    <rect x="{i*w_bar}" y="{h-c}" width="{w_bar}" height="{c}" fill="{color}"/>'
            for i, c in enumerate(scaled) if c
        )

    def _svg(self, content: str) -> str:
        h, w, _ = self._dims
        return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">{content}\n</svg>'

    def plot(self, distr: t.Tensor, col: str | None = None) -> str:
        col = self.col if col is None else col
        tf = DistrTransform.from_distrs(distr)
        counts = self._histogram(distr, tf)
        return self._svg(self._bars(counts, col))

    def compare(self, a: t.Tensor, b: t.Tensor, col: str | None = None) -> str:
        """Colors: 1st, 2nd, overlap"""
        tf = DistrTransform.from_distrs(a, b)
        a_hist = self._histogram(a, tf)
        b_hist = self._histogram(b, tf)
        overlap = t.minimum(a_hist, b_hist)
        col = self.col if col is None else col

        content = self._bars(a_hist, self.col_a) + self._bars(b_hist, self.col_b) + self._bars(overlap, col)
        return self._svg(content)


# Example:
# d1 = t.randn(100000)
# d2 = t.randn(100000) * 0.5 + 1
# plotter = DistrPlotter(bins=200)
# svg1 = plotter.plot(d1)
# svg2 = plotter.compare(d1, d2)
# display(HTML(svg1))
# display(HTML(svg2))

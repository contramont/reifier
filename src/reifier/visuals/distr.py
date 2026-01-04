from dataclasses import dataclass
from typing import Self, ClassVar

import torch as t


@dataclass
class Distr:
    """Distribution plot with metadata."""
    name: str
    svg: str
    description: str = ""


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
        scaled: list[int] = (h * counts).round().long().tolist()  # type: ignore
        return "".join(
            f'\n    <rect x="{i*w_bar}" y="{h-c}" width="{w_bar}" height="{c}" fill="{color}"/>'
            for i, c in enumerate(scaled) if c
        )

    def _svg(self, content: str) -> str:
        h, w, _ = self._dims
        return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">{content}\n</svg>'

    def plot(self, distr: t.Tensor, col: str | None = None, name: str = "") -> Distr:
        col = self.col if col is None else col
        tf = DistrTransform.from_distrs(distr)
        counts = self._histogram(distr, tf)
        svg = self._svg(self._bars(counts, col))
        mean=float(distr.mean().item())
        std=float(distr.std().item())
        min=float(distr.min().item())
        max=float(distr.max().item())
        n_zeros = t.numel(distr) - int(t.count_nonzero(distr))
        neg = distr[distr < 0]
        neg_max = float(neg.max().item()) if neg.numel() > 0 else float('nan')
        pos = distr[distr > 0]
        pos_min = float(pos.min().item()) if pos.numel() > 0 else float('nan')
        return Distr(name, svg,
            description=(
            f"  {mean = :.4f}, {std = :.4f}\n"
            f"  {min = :.4f}, {max = :.4f}\n"
            f"  n_zeros = {n_zeros}/{t.numel(distr)}\n"
            f"  {neg_max = :.8f}\n"
            f"  {pos_min = :.8f}"
            ))

    def compare(self, a: t.Tensor, b: t.Tensor, col: str | None = None, name: str = "") -> Distr:
        """Colors: 1st, 2nd, overlap"""
        tf = DistrTransform.from_distrs(a, b)
        a_hist = self._histogram(a, tf)
        b_hist = self._histogram(b, tf)
        overlap = t.minimum(a_hist, b_hist)
        col = self.col if col is None else col

        content = self._bars(a_hist, self.col_a) + self._bars(b_hist, self.col_b) + self._bars(overlap, col)
        svg = self._svg(content)
        distr: Distr = Distr(name, svg)
        return distr

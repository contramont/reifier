import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .matrices import Matrices
from .mlp import MLP


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation as used in modern transformers."""

    def __init__(
        self,
        in_f: int,
        out_f: int,
        has_bias: bool = False,
        dtype: t.dtype = t.float32,
        hidden_f: int | None = None,
    ):
        super().__init__()  # type: ignore
        self.dtype = dtype  # type: ignore  # ty
        self.has_bias = has_bias  # type: ignore  # ty
        hidden_features = hidden_f or int(out_f * 2)
        
        self.norm = nn.modules.normalization.RMSNorm(in_f)
        self.wg = nn.Linear(in_f, hidden_features, bias=has_bias)
        self.wv = nn.Linear(in_f, hidden_features, bias=has_bias)
        self.wo = nn.Linear(hidden_features, out_f, bias=has_bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.type(self.dtype)
        x = self.norm(x)
        return self.wo(F.silu(self.wg(x)) * self.wv(x))

    @classmethod
    def from_matrix(
        cls,
        w: t.Tensor,
        c: int = 4,
        q: int = 8,
        has_bias: bool = True,
        dtype: t.dtype = t.float32,
        units: tuple[t.Tensor, ...] | None = None,
        exact: bool = False,
    ) -> "SwiGLU":
        """
        Prepares SwiGLU weights from Matrices matrix that has biases folded into weights.
        1) Simulates a step fn with two offset ReLUs
        2) Simulates ReLU with SiLU by scaling up and down
        Making two ReLUs a, b such that a-b is this fn:
        y=0 until x=0.5-1/4c, then slope up until x=0.5+1/4c and y=1. Then y=1.
        Demo: https://www.desmos.com/calculator/w806u4n8hl
        exact: steps that 16-bit floats compute exactly on 0/1 inputs. A step rises over
        [1/2, 1/2 + 1/c] instead, so at sum 1 its ReLUs sit at c*q/2 and c*q/4, and a row
        whose sum can exceed 1 by more than it can fall below 0 is built as
        BOS - step(1 - sum), which keeps large sums where both ReLUs are off
        """
        # c: making ReLU-simulated step fn steeper
        # q: scaling before and after SiLU to avoid non-ReLU-like dip

        out_features = w.size(0)
        w = w.contiguous().to(dtype=dtype)
        comp = t.zeros(out_features, dtype=t.bool)  # rows built as BOS - step(1 - sum)
        if exact:
            smax = w[:, 0] + w[:, 1:].clamp(min=0).sum(1)
            smin = w[:, 0] + w[:, 1:].clamp(max=0).sum(1)
            comp = (1 - smin) < smax
            comp[0] = False
            w = w.clone()
            w[comp] = -w[comp]
            w[comp, 0] += 1

        # constructing w_gate
        wg = t.cat([w, w], dim=0)
        lo = 0.5 if exact else 0.5 - 1 / (2 * c)  # where the step starts to rise
        wg[1:out_features, 0] -= lo + 1 / c  # sub
        wg[out_features + 1 :, 0] -= lo  # add
        wg *= c * q  # scale up
        # BOS (out vector begins with 1) as relu(2*c*q) - relu(c*q): exact in any float
        # format, so a gated unit that outputs one BOS matches it bit for bit
        wg[0, 0], wg[out_features, 0] = c * q, 2 * c * q

        # constructing w_value
        # it takes part of the scale-down, which keeps hidden activations (and so the
        # effect of weight noise) at their size for q = 4
        v = 4 / q
        wv = t.zeros_like(wg)
        wv[:, 0] += v  # default value

        # constructing w_out
        eye = t.eye(out_features)
        wo = t.cat((-eye, eye), dim=1)
        wo /= q * v  # scale down
        wo[0] /= c  # the BOS pair differs by c*q, not q
        wo[comp] = -wo[comp]
        wo[comp, 0], wo[comp, out_features] = wo[0, 0], wo[0, out_features]

        # gated units replace the steps of their rows, with one hidden unit each:
        # silu(c*q*gate) * value / (c*q), which tends to max(0, gate) * value
        if units is not None:
            gates, values, outs = units
            steps = ~outs.any(dim=1).repeat(2)  # step units of rows without gated units
            wg = t.cat([wg[steps], gates * (c * q)])
            wv = t.cat([wv[steps], values * v])
            wo = t.cat([wo[:, steps], outs / (c * q * v)], dim=1)

        # create swiglu with weights wg, wv, wo
        swiglu = cls(
            w.size(1), out_features, has_bias=has_bias, dtype=dtype, hidden_f=len(wg)
        )
        for param, wi in zip(
            [swiglu.wg, swiglu.wv, swiglu.wo], [wg, wv, wo]
        ):
            with t.no_grad():
                target = param.weight
                target.zero_()
                source = wi.contiguous().to(dtype=target.dtype, device=target.device)
                target.copy_(source)
                assert source.shape == target.shape
                if swiglu.has_bias:
                    param.bias.data.zero_()

        return swiglu


class MLP_SwiGLU(MLP):
    """MLP with SwiGLU activations"""

    def __init__(self, sizes: list[int], dtype: t.dtype = t.float32):
        super().__init__(sizes, SwiGLU, dtype=dtype)  # type: ignore

    @classmethod
    def from_matrices(
        cls,
        matrices: Matrices,
        c: int = 4,
        q: int = 8,
        has_bias: bool = False,
        dtype: t.dtype = t.float32,
        exact: bool | None = None,
    ) -> "MLP_SwiGLU":
        """exact: see SwiGLU.from_matrix; by default on for 16-bit dtypes"""
        if exact is None:
            exact = dtype in (t.bfloat16, t.float16)
        mlp = cls(matrices.sizes, dtype=dtype)
        ulist = matrices.ulist or [None] * len(matrices.mlist)
        swiglus = [
            SwiGLU.from_matrix(m, c=c, q=q, has_bias=has_bias, units=u, exact=exact)
            for m, u in zip(matrices.mlist, ulist, strict=True)
        ]
        for swiglu in swiglus:  # weights are made in float32, then cast
            swiglu.to(dtype).dtype = dtype
        mlp.layers = nn.Sequential(*swiglus)  # hidden sizes vary with gated units
        return mlp

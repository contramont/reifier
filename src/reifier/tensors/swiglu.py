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
    ):
        super().__init__()  # type: ignore
        self.dtype = dtype  # type: ignore  # ty
        self.has_bias = has_bias  # type: ignore  # ty
        hidden_features = int(out_f * 2)
        
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
        q: int = 4,
        has_bias: bool = True,
        dtype: t.dtype = t.float32,
    ) -> "SwiGLU":
        """
        Prepares SwiGLU weights from Matrices matrix that has biases folded into weights.
        1) Simulates a step fn with two offset ReLUs
        2) Simulates ReLU with SiLU by scaling up and down
        Making two ReLUs a, b such that a-b is this fn:
        y=0 until x=0.5-1/4c, then slope up until x=0.5+1/4c and y=1. Then y=1.
        Demo: https://www.desmos.com/calculator/sk42yz8ami
        """
        # c: making ReLU-simulated step fn steeper
        # q: scaling before and after SiLU to avoid non-ReLU-like dip

        out_features = w.size(0)
        w = w.contiguous().to(dtype=dtype)

        # constructing w_gate
        wg = t.cat([w, w], dim=0)
        wg[1:out_features, 0] -= 0.5 + 1 / (2 * c)  # sub
        wg[out_features + 1 :, 0] -= 0.5 - 1 / (2 * c)  # add
        wg *= c * q  # scale up
        wg[0, 0] -= q  # to ensure that out vector begins with 1

        # constructing w_value
        wv = t.zeros_like(wg)
        wv[:, 0] += 1  # default value is 1

        # constructing w_out
        eye = t.eye(out_features)
        wo = t.cat((-eye, eye), dim=1)
        wo /= q  # scale down

        # create swiglu with weights wg, wv, wo
        swiglu = cls(w.size(1), out_features, has_bias=has_bias, dtype=dtype)
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
        q: int = 4,
        has_bias: bool = False,
        dtype: t.dtype = t.float32,
    ) -> "MLP_SwiGLU":
        mlp = cls(matrices.sizes, dtype=dtype)
        swiglus = [
            SwiGLU.from_matrix(m, c=c, q=q, has_bias=has_bias) for m in matrices.mlist
        ]
        for i, swiglu in enumerate(swiglus):
            for p, new_p in zip(mlp.layers[i].parameters(), swiglu.parameters()):
                p.data.copy_(new_p.data)
        return mlp

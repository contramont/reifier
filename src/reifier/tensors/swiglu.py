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
        self.dtype = dtype
        self.has_bias = has_bias
        hidden_features = int(out_f * 2)
        self.w_silu = nn.Linear(in_f, hidden_features, bias=has_bias)
        self.w_gate = nn.Linear(in_f, hidden_features, bias=has_bias)
        self.w_last = nn.Linear(hidden_features, out_f, bias=has_bias)
        self.norm = nn.modules.normalization.RMSNorm(in_f)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.type(self.dtype)
        x = self.norm(x)
        return self.w_last(F.silu(self.w_silu(x)) * self.w_gate(x))

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
        # w = w.type(dtype)

        # c: making ReLU-simulated step fn steeper
        # q: scaling before and after SiLU to avoid non-ReLU-like dip

        out_features = w.size(0)
        w = w.contiguous().to(dtype=dtype)

        # print(0.5 + 1 / (2 * c))
        # print(0.5 - 1 / (2 * c))
        # print(c * q)
        # print((0.5 + 1 / (2 * c)) * c * q)
        # print((0.5 - 1 / (2 * c)) * c * q)

        # sub = 0.5 + 1 / (2 * c)
        # sub = int(sub * c * q)
        # add = 0.5 - 1 / (2 * c)
        # add = int(add * c * q)

        # # constructing w_silu
        # w1 = t.cat([w, w], dim=0)
        # w1 *= c * q  # scale up
        # w1[1:out_features, 0] -= sub  # sub
        # w1[out_features + 1 :, 0] -= add  # add
        # w1[0, 0] -= q  # to ensure that out vector begins with 1

        # print(sub)
        # print(add)
        # print(w1)

        # eye = t.eye(out_features)
        # w3 = t.cat((-eye, eye), dim=1)
        # w3 /= q  # scale down
        # print(w3)
        # assert False

        # constructing w_silu
        w1 = t.cat([w, w], dim=0)
        w1[1:out_features, 0] -= 0.5 + 1 / (2 * c)  # sub
        w1[out_features + 1 :, 0] -= 0.5 - 1 / (2 * c)  # add
        w1 *= c * q  # scale up
        w1[0, 0] -= q  # to ensure that out vector begins with 1

        # constructing w_gate
        w2 = t.zeros_like(w1)
        w2[:, 0] += 1  # gate = 1

        # constructing w_last
        eye = t.eye(out_features)
        w3 = t.cat((-eye, eye), dim=1)
        w3 /= q  # scale down

        # create swiglu with weights w1, w2, w3
        swiglu = cls(w.size(1), out_features, has_bias=has_bias, dtype=dtype)
        for param, wi in zip(
            [swiglu.w_silu, swiglu.w_gate, swiglu.w_last], [w1, w2, w3]
        ):
            with t.no_grad():
                target = param.weight
                target.zero_()
                source = wi.contiguous().to(dtype=target.dtype, device=target.device)
                target.copy_(source)
                assert source.shape == target.shape
                if swiglu.has_bias:
                    param.bias.data.zero_()

            # param.weight.data.zero_()
            # param.weight.data[: wi.size(0), : wi.size(1)] = wi
            # if swiglu.has_bias:
            #     param.bias.data.zero_()

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

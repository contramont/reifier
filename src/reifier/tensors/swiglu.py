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
        # Check if last matrix is linear and should be folded into previous layer's wo
        linear_matrices = matrices.linear_matrices if matrices.linear_matrices else tuple(False for _ in matrices.mlist)

        # Find matrices to convert to SwiGLU and linear matrices to fold
        mlist = list(matrices.mlist)
        linear_list = list(linear_matrices)

        # If the last matrix is linear, fold it into the previous layer's wo
        # This is done by multiplying the linear matrix with wo: new_wo = L @ wo
        fold_indices: list[int] = []
        while len(mlist) > 1 and linear_list[-1]:
            fold_indices.append(len(mlist) - 1)
            linear_list.pop()
            mlist.pop()

        # Create SwiGLU layers for non-linear matrices
        swiglus = [
            SwiGLU.from_matrix(m.to(dtype=dtype), c=c, q=q, has_bias=has_bias, dtype=dtype)
            for m in mlist
        ]

        # Fold linear matrices into the last SwiGLU's wo
        if fold_indices:
            last_swiglu = swiglus[-1]
            wo = last_swiglu.wo.weight.data.clone()
            # Apply each linear matrix (in order from closest to furthest from output)
            for idx in fold_indices:
                # The linear matrix has bias folded in, including BOS preservation
                # We use the full matrix: new_wo = linear_m @ wo
                linear_m = matrices.mlist[idx].to(dtype=dtype)
                wo = linear_m @ wo
            # Update the last SwiGLU's wo with the combined weights
            with t.no_grad():
                new_out_features = wo.size(0)
                hidden_features = wo.size(1)
                last_swiglu.wo = nn.Linear(hidden_features, new_out_features, bias=last_swiglu.has_bias)
                last_swiglu.wo.weight.data = wo.to(dtype=dtype)
                if last_swiglu.has_bias:
                    last_swiglu.wo.bias.data.zero_()

        # Build MLP directly from the SwiGLU layers
        # Compute sizes based on actual SwiGLU layer dimensions
        sizes = [swiglus[0].wg.in_features]
        for swiglu in swiglus:
            sizes.append(swiglu.wo.out_features)

        # Create MLP shell and replace its layers with our SwiGLU layers
        mlp = cls.__new__(cls)
        nn.Module.__init__(mlp)
        mlp.dtype = dtype
        mlp.layers = nn.Sequential(*swiglus)

        return mlp

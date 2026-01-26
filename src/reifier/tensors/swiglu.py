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
        linear_matrices = matrices.linear_matrices if matrices.linear_matrices else tuple(False for _ in matrices.mlist)
        mlist = [m.to(dtype=dtype) for m in matrices.mlist]

        # Fold linear matrices into the previous layer's wo
        # Process from end to beginning: for each linear layer, fold it into the previous layer
        # After folding, the linear layer is removed from the list
        i = len(mlist) - 1
        while i > 0:
            if linear_matrices[i]:
                # Fold linear matrix i into the wo of layer i-1
                # This will be handled when creating SwiGLU layers
                i -= 1
            else:
                i -= 1

        # Build list of (matrix, list of linear matrices to fold into its wo)
        # Process forward: for each non-linear matrix, collect all following linear matrices
        layer_info: list[tuple[t.Tensor, list[t.Tensor]]] = []
        i = 0
        while i < len(mlist):
            if not linear_matrices[i]:
                # Non-linear layer: collect it and any following linear layers
                base_matrix = mlist[i]
                linear_to_fold = []
                j = i + 1
                while j < len(mlist) and linear_matrices[j]:
                    linear_to_fold.append(mlist[j])
                    j += 1
                layer_info.append((base_matrix, linear_to_fold))
                i = j
            else:
                # Linear layer at start - shouldn't happen normally, skip
                i += 1

        # Create SwiGLU layers, folding linear matrices into wo
        swiglus = []
        for base_matrix, linear_to_fold in layer_info:
            swiglu = SwiGLU.from_matrix(base_matrix, c=c, q=q, has_bias=has_bias, dtype=dtype)

            if linear_to_fold:
                # Fold linear matrices into wo: new_wo = L_n @ ... @ L_1 @ wo
                wo = swiglu.wo.weight.data.clone()
                for linear_m in linear_to_fold:
                    wo = linear_m @ wo

                # Update wo with folded weights
                with t.no_grad():
                    new_out_features = wo.size(0)
                    hidden_features = wo.size(1)
                    swiglu.wo = nn.Linear(hidden_features, new_out_features, bias=swiglu.has_bias)
                    swiglu.wo.weight.data = wo.to(dtype=dtype)
                    if swiglu.has_bias:
                        swiglu.wo.bias.data.zero_()

            swiglus.append(swiglu)

        # Build MLP directly from the SwiGLU layers
        sizes = [swiglus[0].wg.in_features]
        for swiglu in swiglus:
            sizes.append(swiglu.wo.out_features)

        mlp = cls.__new__(cls)
        nn.Module.__init__(mlp)
        mlp.dtype = dtype
        mlp.layers = nn.Sequential(*swiglus)

        return mlp

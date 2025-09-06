from reifier.utils.format import Bits
from reifier.compile.tree import Tree
from reifier.tensors.matrices import Matrices

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation as used in modern transformers."""

    def __init__(
        self, in_features: int, out_features: int, dtype: t.dtype = t.bfloat16
    ):
        super().__init__()  # type: ignore
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        hidden_features = int(out_features * 2)
        self.w_silu = nn.Linear(in_features, hidden_features, bias=False)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_last = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.type(self.dtype)
        return self.w_last(F.silu(self.w_silu(x)) * self.w_gate(x))


class MLP_SwiGLU(nn.Module):
    """MLP with SwiGLU activations"""

    def __init__(self, sizes: list[int], dtype: t.dtype = t.float32):
        super().__init__()  # type: ignore
        self.dtype = dtype
        layers: list[SwiGLU] = []
        prev_size = sizes[0]
        for hidden_size in sizes[1:]:
            layers.append(SwiGLU(prev_size, hidden_size, dtype=dtype))
            prev_size = hidden_size
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x.to(self.dtype))

    def load_params(self, swiglus: list[SwiGLU]) -> None:
        for param, swiglu in zip(self.layers, swiglus):
            assert isinstance(param, SwiGLU)
            param.w_silu.weight.data[:] = swiglu.w_silu.weight.data
            param.w_gate.weight.data[:] = swiglu.w_gate.weight.data
            param.w_last.weight.data[:] = swiglu.w_last.weight.data

    def infer_bits(self, x: Bits, auto_constant: bool = True) -> Bits:
        if auto_constant:
            x = Bits("1") + x
        x_tensor = t.tensor(x.ints, dtype=self.dtype)
        with t.inference_mode():
            result = self.forward(x_tensor)
        result_ints = [int(el.item()) for el in t.IntTensor(result.int())]
        if auto_constant:
            result_ints = result_ints[1:]
        return Bits(result_ints)


def swiglu_from_matrix(w: t.Tensor) -> SwiGLU:
    """
    Prepares SwiGLU weights from Matrices matrix that has biases folded into weights.
    1) Simulates a step fn with two offset ReLUs
    2) Simulates ReLU with SiLU by scaling up and down
    Making two ReLUs a, b such that a-b is this fn:
    y=0 until x=0.5-1/4c, then slope up until x=0.5+1/4c and y=1. Then y=1.
    Demo: https://www.desmos.com/calculator/sk42yz8ami
    """
    c = 16  # making ReLU-simulated step fn steeper
    q = 16  # scaling before and after SiLU to avoid non-ReLU-like dip

    out_features = w.size(0)

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
    swiglu = SwiGLU(w.size(1), out_features)
    for wi, param in zip([w1, w2, w3], [swiglu.w_silu, swiglu.w_gate, swiglu.w_last]):
        param.weight.data.zero_()
        param.weight.data[: wi.size(0), : wi.size(1)] = wi
    return swiglu


def mlp_from_matrices(matrices: Matrices) -> MLP_SwiGLU:
    swiglus = [swiglu_from_matrix(layer) for layer in matrices.mlist]
    mlp = MLP_SwiGLU(matrices.sizes)
    mlp.load_params(swiglus)
    return mlp


def mlp_from_tree(tree: Tree) -> MLP_SwiGLU:
    matrices = Matrices.from_tree(tree)
    return mlp_from_matrices(matrices)


def print_swiglu_mlp_activations(mlp: MLP_SwiGLU, x: t.Tensor) -> None:
    for i, layer in enumerate(mlp.layers):
        x = x.type(mlp.dtype)  # type: ignore
        x_presilu = layer.w_silu(x)  # type: ignore
        x_postsilu = F.silu(x_presilu)  # type: ignore
        x_gate = layer.w_gate(x)  # type: ignore
        x_mult = x_postsilu * x_gate  # type: ignore
        x_last = layer.w_last(x_mult)  # type: ignore
        print(f"{i} x={x}")
        print(f"{i} x_silu={x_presilu}")
        print(f"{i} x_silu={x_postsilu}")
        print(f"{i} x_gate={x_gate}")
        print(f"{i} x_mult={x_mult}")
        print(f"{i} x_last={x_last}")
        x = x_last  # type: ignore

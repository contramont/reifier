import torch as t
import torch.nn as nn

from .mlp import MLP
from .matrices import Matrices


class InitlessLinear(t.nn.Linear):
    """Skip init since all parameters will be specified"""

    def reset_parameters(self):
        pass


class StepLayer(nn.Module):
    """MLP layer with a step activation function"""

    def __init__(self, in_features: int, out_features: int, dtype: t.dtype = t.float32):
        super().__init__()  # type: ignore
        self.linear = InitlessLinear(in_features, out_features, bias=True, dtype=dtype)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear(x)
        return (x > 0.5).type(x.dtype)


class MLP_Step(MLP):
    """MLP with step activations"""

    def __init__(self, sizes: list[int], dtype: t.dtype = t.float32):
        super().__init__(sizes, StepLayer, dtype=dtype)

    @classmethod
    def from_matrices(
        cls, matrices: Matrices, dtype: t.dtype = t.float32
    ) -> "MLP_Step":
        mlp = cls(matrices.sizes, dtype=dtype)
        for layer, m in zip(mlp.layers, matrices.mlist):
            assert isinstance(layer, StepLayer)
            layer.linear.weight.data.copy_(m)
            layer.linear.bias.data.zero_()
        return mlp


# def step_activation(x: t.Tensor) -> t.Tensor:
#     return (x > 0.5).type(x.dtype)


# class MLP_Step(nn.Module):
#     """Simple PyTorch MLP implementation"""
#     def __init__(self, sizes: list[int], dtype: t.dtype):
#         super().__init__()  # type: ignore
#         self.dtype = dtype
#         layers: list[nn.Module] = []
#         for in_size, out_size in zip(sizes[:-1], sizes[1:]):
#             # layers.append(nn.Linear(in_size, out_size, bias=False, dtype=dtype))
#             layers.append(nn.Linear(in_size, out_size, bias=True, dtype=dtype))
#         self.layers = nn.ModuleList(layers)
#     def forward(self, x: t.Tensor) -> t.Tensor:
#         # x = x.type(self.dtype)
#         for layer in self.layers:
#             x = step_activation(layer(x.type(self.dtype)))
#         return x

#     @classmethod
#     def from_matrices(cls, matrices: Matrices, dtype: t.dtype = t.float32) -> "MLP_Step":

#         mlp = cls(matrices.sizes, dtype=dtype)
#         for i, m in enumerate(matrices.mlist):
#             layer = mlp.layers[i]

#             assert isinstance(layer, nn.Linear)
#             assert isinstance(m, t.Tensor)
#             with t.no_grad():
#                 target = layer.weight
#                 target.zero_()
#                 source = m.contiguous().to(dtype=target.dtype, device=target.device)
#                 target.copy_(source)
#                 layer.bias.data.zero_()
#         return mlp

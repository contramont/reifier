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

    def __init__(
        self, in_features: int, out_features: int, dtype: t.dtype = t.bfloat16
    ):
        super().__init__()  # type: ignore
        self.linear = InitlessLinear(in_features, out_features, bias=False, dtype=dtype)

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
        return mlp

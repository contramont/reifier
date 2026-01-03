import torch as t
import torch.nn as nn


class MLP(nn.Module):
    """MLP"""

    def __init__(
        self, sizes: list[int], layer: type[nn.Module], dtype: t.dtype = t.float32
    ):
        super().__init__()  # type: ignore
        self.dtype = dtype  # type: ignore  # ty
        layers = [
            layer(in_s, out_s, dtype=dtype)
            for in_s, out_s in zip(sizes[:-1], sizes[1:])
        ]
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.type(self.dtype)
        return self.layers(x)

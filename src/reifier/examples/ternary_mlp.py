"""Ternary MLP utilities for encoding weights in backdoors.

Ternary weights use values {-1, 0, 1} which can be encoded as 2 bits:
- (1, 0) = 1
- (0, 0) = 0
- (0, 1) = -1
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from reifier.neurons.core import Bit, const


class Ternarize(t.autograd.Function):
    """Straight-through estimator for ternarization."""
    @staticmethod
    def forward(ctx, x):
        return t.sign(t.round(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(t.randn(out_features, in_features) * 0.5)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return F.linear(x, Ternarize.apply(self.weight))

    def get_ternary_weights(self) -> t.Tensor:
        return t.sign(t.round(self.weight.data))


class TernaryMLP(nn.Module):
    """MLP with ternary weights."""
    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(TernaryLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)

    def extract_weights(self) -> list[t.Tensor]:
        return [layer.get_ternary_weights() for layer in self.layers if isinstance(layer, TernaryLinear)]


def mlp_to_bitlists(model: TernaryMLP) -> list[list[Bit]]:
    """Convert ternary MLP weights to bitlists for backdoor encoding.

    Each ternary value is encoded as 2 bits:
    - 1  -> (1, 0)
    - 0  -> (0, 0)
    - -1 -> (0, 1)
    """
    bitlists = []
    for w in model.extract_weights():
        bits = []
        for v in w.flatten().tolist():
            if v == -1:
                bits.extend([0, 1])
            elif v == 0:
                bits.extend([0, 0])
            else:  # v == 1
                bits.extend([1, 0])
        bitlists.append(const(bits))
    return bitlists


def binary_to_ternary_flat(binary: t.Tensor, shapes: list[t.Size]) -> t.Tensor:
    """Decode binary tensor to ternary weights.

    Args:
        binary: Flat tensor of binary values (2 bits per ternary value)
        shapes: List of weight matrix shapes

    Returns:
        Flat tensor of ternary values
    """
    total_params = sum(s.numel() for s in shapes)
    ternary = t.zeros(total_params)
    for i in range(total_params):
        b0, b1 = binary[2*i], binary[2*i + 1]
        if b0 == 1 and b1 == 0:
            ternary[i] = 1
        elif b0 == 0 and b1 == 1:
            ternary[i] = -1
    return ternary


def flat_to_matrices(flat: t.Tensor, shapes: list[t.Size]) -> list[t.Tensor]:
    """Reshape flat tensor to list of matrices with given shapes."""
    matrices, offset = [], 0
    for shape in shapes:
        size = shape.numel()
        matrices.append(flat[offset:offset + size].reshape(shape))
        offset += size
    return matrices


def bitlist_to_tensor_w_bos(bits: list[Bit], device) -> t.Tensor:
    """Convert bitlist to tensor with BOS token prepended."""
    return t.tensor([1.0] + [float(b.activation) for b in bits], device=device)

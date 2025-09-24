import torch as t
import torch.nn.functional as F

from reifier.utils.format import Bits
from .mlp import MLP
from .swiglu import MLP_SwiGLU
from .step import MLP_Step


def infer_bits_without_bos(mlp: MLP, x: Bits) -> Bits:
    """Runs the MLP on Bits object, returns Bits object"""
    with t.inference_mode():
        result = mlp(t.tensor(x.ints, dtype=mlp.dtype))
    result_ints = [int(el.item()) for el in t.IntTensor(result.int())]
    return Bits(result_ints)


def infer_bits_bos(mlp: MLP, x: Bits) -> Bits:
    """Adds a BOS bit to the input and returns the output without the BOS bit"""
    bos_x = Bits("1") + x
    bos_y = infer_bits_without_bos(mlp, bos_x)
    y = Bits(bos_y.bitlist[1:])
    return y


# ------------ DEBUGGING FUNCTIONS ------------


def print_swiglu_mlp_activations(mlp: MLP_SwiGLU, x: t.Tensor, layer_limit: int = -1) -> None:
    """Prints the activations of the MLP layers. Extracts the first element of the batch."""
    x = x.type(mlp.dtype)  # type: ignore
    for i, layer in enumerate(mlp.layers):
        if layer_limit != -1 and i >= layer_limit:
            break
        x_presilu = layer.w_silu(x)  # type: ignore
        x_postsilu = F.silu(x_presilu)  # type: ignore
        x_gate = layer.w_gate(x)  # type: ignore
        x_mult = x_postsilu * x_gate  # type: ignore
        x_last = layer.w_last(x_mult)  # type: ignore
        print(f"\nLayer {i} activations:")
        print(f"x={x[0]}")
        print(f"x_silu={x_presilu[0]}")
        print(f"x_silu={x_postsilu[0]}")
        print(f"x_gate={x_gate[0]}")
        print(f"x_mult={x_mult[0]}")
        print(f"x_last={x_last[0]}")
        x = x_last  # type: ignore


def vector_str(vec: t.Tensor, precision: int = 2) -> str:
    """Converts a 1D tensor to a string."""
    if precision == 0:
        return f"{''.join([str(int(el)) for el in vec.tolist()][1:])}"  # type: ignore
    return ", ".join([str(round(el, precision)) for el in vec.tolist()])  # type: ignore


def print_step_mlp_activations(mlp: MLP_Step, x: t.Tensor, layer_limit: int = -1) -> None:
    """Prints the activations of the MLP layers. Extracts the first element of the batch."""
    for i, layer in enumerate(mlp.layers):
        if layer_limit != -1 and i >= layer_limit:
            break
        print(i, vector_str(x[0], 0))  # type: ignore
        x = layer(x)
        step_activation(x)
    if layer_limit == -1 or layer_limit >= len(mlp.layers):
        print(len(mlp.layers), vector_str(x[0], 0))  # type: ignore


def max_abs(x: t.Tensor) -> float:
    return t.max(t.abs(x)).item()


def get_non_zeros(x: t.Tensor) -> list[list[float]]:
    assert x.dim() == 2
    lst = x.tolist()  # type: ignore
    lst = [[float(el) for el in row if el != 0.0] for row in lst]  # type: ignore
    return lst


def step_activation(x: t.Tensor) -> t.Tensor:
    return (x > 0.5).type(x.dtype)


def print_step_mlp_activations_diff(mlp: MLP_Step, x1: t.Tensor, x2: t.Tensor, layer_limit: int = -1) -> None:
    """Prints the activations of the MLP layers. Extracts the first element of the batch."""
    for i, layer in enumerate(mlp.layers):
        if layer_limit != -1 and i >= layer_limit:
            break

        diff = x1[1] - x2
        max_diff = t.max(t.abs(diff)).item()
        if max_diff > 0.001:
            print("big diff!", i, max_diff, diff)
            assert False

        diff = x1[1] - x2
        weights = layer.weight.data
        assert isinstance(weights, t.Tensor)
        print(i, "weights", max_abs(weights), weights.shape, get_non_zeros(weights))
        print(i, "x1 pre ", max_abs(x1[1]), x1[1])
        print(i, "x2 pre ", max_abs(x2), x2)
        print("diff", max_abs(diff), diff)
        if max_abs(diff) > 0.001:
            assert 0

        x1 = layer(x1)
        x2 = layer(x2)
        diff = x1[1] - x2
        print(i, "x1 post", max_abs(x1[1]), x1[1])
        print(i, "x2 post", max_abs(x2), x2)
        print("diff", max_abs(diff), diff)
        if max_abs(diff) > 0.001:
            assert 0

        x1 = step_activation(x1)
        x2 = step_activation(x2)
        diff = x1[1] - x2
        print(i, "x1 step", max_abs(x1[1]), x1[1])
        print(i, "x2 step", max_abs(x2), x2)
        print("diff", max_abs(diff), diff)
        if max_abs(diff) > 0.001:
            assert 0
        print("----")

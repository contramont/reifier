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


def print_swiglu_mlp_activations(mlp: MLP_SwiGLU, x: t.Tensor) -> None:
    x = x.type(mlp.dtype)  # type: ignore
    for i, layer in enumerate(mlp.layers):
        x_presilu = layer.w_silu(x)  # type: ignore
        x_postsilu = F.silu(x_presilu)  # type: ignore
        x_gate = layer.w_gate(x)  # type: ignore
        x_mult = x_postsilu * x_gate  # type: ignore
        x_last = layer.w_last(x_mult)  # type: ignore
        print(f"\nLayer {i} activations:")
        print(f"x={x}")
        print(f"x_silu={x_presilu}")
        print(f"x_silu={x_postsilu}")
        print(f"x_gate={x_gate}")
        print(f"x_mult={x_mult}")
        print(f"x_last={x_last}")
        x = x_last  # type: ignore


def vector_str(vec: t.Tensor, precision: int = 2) -> str:
    if precision == 0:
        return f"{''.join([str(int(el)) for el in vec.tolist()][1:])}"  # type: ignore
    return ", ".join([str(round(el, precision)) for el in vec.tolist()])  # type: ignore


def print_mlp_activations(mlp: MLP_Step, x: t.Tensor) -> None:
    for i, layer in enumerate(mlp.layers):
        print(i, vector_str(x, 0))  # type: ignore
        x = layer(x)
    print(len(mlp.layers), vector_str(x, 0))  # type: ignore

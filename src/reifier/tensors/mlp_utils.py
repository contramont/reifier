import torch as t

from reifier.utils.format import Bits
from .mlp import MLP
from .step import MLP_Step


# ------------ PARAMETER EXTRACTION FUNCTIONS ------------

def get_params(module: t.nn.Module | t.Tensor | dict[str, t.nn.Module | t.Tensor]) -> dict[str, t.Tensor]:
    '''Extracts named parameters'''
    if isinstance(module, t.Tensor):
        return {}
    if isinstance(module, dict):
        return {f'{mn}: {pn}': p for mn, m in module.items() for pn, p in get_params(m).items()}
    return dict(module.named_parameters())


def get_params_flat(module: t.nn.Module | t.Tensor) -> t.Tensor:
    if isinstance(module, t.Tensor):
        return module.flatten()
    params = t.cat([p.data.flatten() for p in module.parameters()]).cpu().float()
    return params


# ------------ BITS INFERENCE FUNCTIONS ------------

def infer_bits_without_bos(mlp: MLP, x: Bits) -> Bits:
    """Runs the MLP on Bits object, returns Bits object"""
    with t.inference_mode():
        result = mlp(t.tensor(x.ints, dtype=mlp.dtype))
    result_ints = [int(el.item()) for el in t.IntTensor(result.int())]
    return Bits(result_ints)


def bos_res_batch_to_bools(bos_res: t.Tensor) -> t.Tensor:
    """Maps each batch sample el to 1 if close to sample bos, else 0"""
    # bos_res (batch size, n features)
    bos_elems = bos_res[:, 0].repeat(1, bos_res.shape[1])
    res_bools = t.isclose(bos_res, bos_elems, rtol=0.01, atol=0.01)
    return res_bools  # (batch size, n features)


def boolify(bos_res: t.Tensor) -> t.Tensor:
    """Maps each batch sample el to 1 if close to sample bos, else 0"""
    # bos_res (batch size, n features)
    # bos_elems = bos_res[:, 0].repeat(1, bos_res.shape[1])
    normed = bos_res/bos_res[:, 0]
    ones = t.ones_like(normed)
    res_bools = t.isclose(normed, ones, rtol=0.01, atol=0.01)
    return res_bools  # (batch size, n features)


def infer_bits_bos(mlp: MLP, x: Bits) -> Bits:
    """Adds a BOS bit to the input and returns the output without the BOS bit"""
    bos_x = Bits("1") + x
    with t.inference_mode():
        res = mlp(t.tensor(bos_x.ints, dtype=mlp.dtype))
        res_bools = boolify(res.unsqueeze(0))
        # res_bools = bos_res_batch_to_bools(res.unsqueeze(0))
        res_ints = [int(el) for el in res_bools.squeeze(0).tolist()]  # type: ignore
    y = Bits(res_ints[1:])
    return y


# def infer_bits_bos(mlp: MLP, x: Bits) -> Bits:
#     """Adds a BOS bit to the input and returns the output without the BOS bit"""
#     bos_x = Bits("1") + x
#     bos_y = infer_bits_without_bos(mlp, bos_x)
#     y = Bits(bos_y.bitlist[1:])
#     return y


# ------------ DEBUGGING FUNCTIONS ------------


def align_float(x: float | int, int_width: int = 5, fractional_width: int = 2) -> str:
    """Returns float's string with a fixed decimal place position and width"""
    # x = abs(x)
    s = f"{x:.{fractional_width}f}"
    pre, post = s.split(".") if "." in s else (s, "")

    n_pre_pad = int_width - len(pre)
    if n_pre_pad < 0:
        pre = " " * (int_width - 4)
        pre += " BIG" if x > 0 else "-BIG"
        return pre + " " * (1+fractional_width)
    pre = " " * n_pre_pad + pre

    post = post.rstrip("0")
    separator = "." if len(post) > 0 else " "
    n_post_pad = fractional_width - len(post)
    post += " " * n_post_pad

    return pre + separator + post


def repr_tensor(
    x: t.Tensor, int_width: int = 5, fractional_width: int = 2, _depth: int = 0
) -> str:
    """Prints a tensor"""
    dim = x.dim()
    assert dim >= 0
    indent = "  " * _depth
    if dim == 0:
        return align_float(x.item(), int_width, fractional_width)
    elif dim == 1:
        subtensors = [align_float(el.item(), int_width, fractional_width) for el in x]
        return f"{indent}[{' '.join(subtensors)}]"
    else:
        open = indent + "["
        subtensors = [repr_tensor(el, _depth=_depth + 1) for el in x]
        subtensors = "\n".join(subtensors)
        close = indent + "]"
        return open + "\n" + subtensors + "\n" + close


def vector_str(vec: t.Tensor, precision: int = 2) -> str:
    """Converts a 1D tensor to a string."""
    if precision == 0:
        return f"{''.join([str(int(el)) for el in vec.tolist()][1:])}"  # type: ignore
    return ", ".join([str(round(el, precision)) for el in vec.tolist()])  # type: ignore



# ------------ STEP MLP FUNCTIONS ------------


def print_step_mlp_activations(
    mlp: MLP_Step, x: t.Tensor, layer_limit: int = -1
) -> None:
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


def print_step_mlp_activations_diff(
    mlp: MLP_Step, x1: t.Tensor, x2: t.Tensor, layer_limit: int = -1
) -> None:
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

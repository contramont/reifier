import torch as t
import torch.nn.functional as F

from reifier.tensors.swiglu import MLP_SwiGLU, SwiGLU
from reifier.tensors.mlp_utils import repr_tensor


def run_layer(layer: SwiGLU, x: t.Tensor) -> dict[str, t.Tensor]:
    xs: t.Tensor = F.rms_norm(x, x.shape)  # before scaling
    xn: t.Tensor = xs * layer.norm.weight.data
    xv: t.Tensor = layer.wv(xn)
    xg: t.Tensor = layer.wg(xn)
    xf: t.Tensor = F.silu(xg)
    xm: t.Tensor = xf * xv
    xo: t.Tensor = layer.wo(xm)
    return {'x':x, 'xn':xn, 'xv':xv, 'xg':xg, 'xf':xf, 'xm':xm, 'xo':xo}


def get_acts(model: MLP_SwiGLU, x: t.Tensor | None) -> list[dict[str, t.Tensor]]:
    if x is None:
        return [{} for _ in model.layers]
    acts: list[dict[str, t.Tensor]] = []
    for layer in model.layers:
        acts.append(run_layer(layer, x))  # type: ignore
        x = acts[-1]['xo']
    return acts


    # ------------ SWIGLU MLP FUNCTIONS ------------


def get_swiglu_mlp_sizes(model: MLP_SwiGLU) -> list[int]:
    """Returns the number of features in each MLP layer, from input to output"""
    sizes: list[int] = [layer.wg.weight.shape[1] for layer in model.layers]  # type: ignore
    sizes += [model.layers[-1].wo.weight.shape[0]]  # type: ignore
    return sizes


def clone_mlp(model: MLP_SwiGLU) -> MLP_SwiGLU:
    """Clones an MLP_SwiGLU model"""
    sizes = get_swiglu_mlp_sizes(model)
    model_clone = MLP_SwiGLU(sizes)
    model_clone.load_state_dict(model.state_dict())
    return model_clone


def get_swiglu_mlp_io_sizes(model: MLP_SwiGLU) -> tuple[int, int]:
    """Returns the input and output sizes of an MLP_SwiGLU model"""
    sizes = get_swiglu_mlp_sizes(model)
    return sizes[0], sizes[-1]


def get_swiglu_mlp_activations(
    mlp: MLP_SwiGLU, x: t.Tensor
) -> list[dict[str, t.Tensor]]:
    """Returns the activations of the MLP layers"""
    activations: list[dict[str, t.Tensor]] = []
    for layer in mlp.layers:
        x = layer.norm(x)  # type: ignore
        presilu = layer.wg(x)  # type: ignore
        postsilu = F.silu(presilu)  # type: ignore
        gate_val = layer.wv(x)  # type: ignore
        product = postsilu * gate_val  # type: ignore
        last = layer.wo(product)  # type: ignore

        a_i: dict[str, t.Tensor] = {
            "x": x,
            "presilu": presilu,
            "postsilu": postsilu,
            "gate_val": gate_val,
            "product": product,
            "last": last,
        }
        activations.append(a_i)

        x = last  # type: ignore
    return activations


def print_swiglu_mlp_activations(
    mlp: MLP_SwiGLU,
    x: t.Tensor,
    depths: list[int] | None = None,
    int_width: int = 5,
    fractional_width: int = 2,
) -> None:
    """Prints the activations of the MLP layers"""
    x = x.type(mlp.dtype)  # type: ignore
    activations = get_swiglu_mlp_activations(mlp, x)
    depths = depths or list(range(len(activations)))
    for i in depths:
        a_i = activations[i]
        print(f"\nLayer {i} activations:")
        for k, v in a_i.items():
            with t.inference_mode():
                print(f"{k:10}= {repr_tensor(v, int_width, fractional_width)}")

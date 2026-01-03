import torch as t
from reifier.tensors.swiglu import MLP_SwiGLU


def noise_mlp_swiglu(
    mlp: MLP_SwiGLU, noise_stdev: float, noise_biases: bool = False
) -> None:
    """
    Adds Gaussian noise to the model weights.
    By default does not noise bias simulation weights which are more sensitive.
    """
    for layer in mlp.layers:
        for w in [layer.wg, layer.wv, layer.wo]:
            assert isinstance(w, t.nn.Linear)
            noise = t.randn_like(w.weight.data)
            if not noise_biases:
                noise[0] = 0
                noise[w.weight.shape[0] // 2] = 0
            w.weight.data += noise * noise_stdev

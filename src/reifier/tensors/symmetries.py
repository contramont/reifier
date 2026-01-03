import torch as t

from reifier.tensors.swiglu import SwiGLU, MLP_SwiGLU
from reifier.tensors.swiglu_utils import clone_mlp


def set_wn_to_ones(swiglu):
    """Absorbs swiglu2 wn into wv and wg to achieve wn=1."""
    wn = swiglu.norm.weight.data.detach().clone()
    swiglu.norm.weight.data.fill_(1)
    swiglu.wv.weight.data = swiglu.wv.weight.data @ t.diag(wn)
    swiglu.wg.weight.data = swiglu.wg.weight.data @ t.diag(wn)


def orthogonal(swiglu1, swiglu2):
    """Apply random orthogonal transformation Q to wo. This preserves the
    swiglu function by applying Q.T (same as Q^-1) to next wg and wv.
    Assumes that swiglu2 wn has been set to ones, so that swiglu2 is invariant
    to orthogonal transforms"""
    d = swiglu1.wo.out_features
    Q, _ = t.linalg.qr(t.randn(d, d))
    swiglu1.wo.weight.data = Q @ swiglu1.wo.weight.data
    swiglu2.wg.weight.data = swiglu2.wg.weight.data @ Q.T
    swiglu2.wv.weight.data = swiglu2.wv.weight.data @ Q.T


def norm_stdevs(
        swiglu: SwiGLU,
        std_wg: float | None = None,
        std_wv: float | None = 1,
        std_wo: float | None = 1
        ):
    """Sets stdev of swiglu weight matrices."""
    if std_wg is not None:
        swiglu.wg.weight.data *= std_wg/swiglu.wg.weight.data.std()
    if std_wv is not None:
        swiglu.wv.weight.data *= std_wv/swiglu.wv.weight.data.std()
    if std_wo is not None:
        swiglu.wo.weight.data *= std_wo/swiglu.wo.weight.data.std()


def get_lognormal_scalars(n: int, dtype, mean: float=0, std: float=1):
    s = t.empty(n).log_normal_(mean=mean, std=std)
    s += t.finfo(dtype).eps  # avoid zero
    random_signs = (t.randint(0, 2, (n,)) * 2) - 1
    s = s * random_signs  # flip signs randomly
    return s


def scale_norm_params(swiglu, mean: float = 0, std: float = 1):
    """Multiplies each norm param by random lognormal scalar. Preserves swiglu
    function by dividing wg and wv in features by the same scalars."""
    n = swiglu.norm.weight.shape[0]
    s = get_lognormal_scalars(n, swiglu.dtype, mean, std)
    s_inv = s.reciprocal()
    swiglu.norm.weight.data = t.diag(s) @ swiglu.norm.weight.data
    swiglu.wg.weight.data = swiglu.wg.weight.data @ t.diag(s_inv)
    swiglu.wv.weight.data = swiglu.wv.weight.data @ t.diag(s_inv)


def permute_features(swiglu):
    n = swiglu.wo.in_features
    idx = t.randperm(n)
    permutation = t.eye(n)[idx]
    swiglu.wg.weight.data = permutation @ swiglu.wg.weight.data
    swiglu.wv.weight.data = permutation @ swiglu.wv.weight.data
    swiglu.wo.weight.data = swiglu.wo.weight.data @ permutation.T


def scale_features(swiglu, mean: float=0, std: float=1, offset: float=1):
    n = swiglu.wo.in_features

    # Rescale wv-wo features
    s = get_lognormal_scalars(n, swiglu.dtype, mean, std)
    s_inv = s.reciprocal()
    swiglu.wv.weight.data = t.diag(s) @ swiglu.wv.weight.data
    swiglu.wo.weight.data = swiglu.wo.weight.data @ t.diag(s_inv)

    # Rescale wg-wo features
    s = get_lognormal_scalars(n, swiglu.dtype, mean, std)
    s = t.abs(s)  # abs as negative scalar can flip x into getting zero'd by silu
    s += offset  # avoids scaling down into silu dip (breaks ReLU simulation)
    s_inv = s.reciprocal()
    swiglu.wg.weight.data = t.diag(s) @ swiglu.wg.weight.data
    swiglu.wo.weight.data = swiglu.wo.weight.data @ t.diag(s_inv)



def apply_to_swiglus(model: MLP_SwiGLU, func, *args, **kwargs) -> MLP_SwiGLU:
    model = clone_mlp(model)
    for layer in model.layers:
        func(layer, *args, **kwargs)
    return model


def apply_to_swiglu_pairs(model: MLP_SwiGLU, func, *args, **kwargs) -> MLP_SwiGLU:
    model = clone_mlp(model)
    for i in range(0, len(model.layers)-1):
        func(model.layers[i], model.layers[i+1], *args, **kwargs)
    return model


def transform(m: MLP_SwiGLU):
    m = apply_to_swiglus(m, set_wn_to_ones)
    m = apply_to_swiglu_pairs(m, orthogonal)
    # m = apply_to_swiglus(m, norm_stdevs, std_wg=None, std_wv=1, std_wo=1)
    m = apply_to_swiglus(m, permute_features)
    m = apply_to_swiglus(m, scale_features, mean=0, std=1, offset=1)
    m = apply_to_swiglus(m, scale_norm_params, mean=0, std=1)
    return m

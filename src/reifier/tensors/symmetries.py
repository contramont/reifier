import torch as t
from reifier.tensors.swiglu import MLP_SwiGLU, SwiGLU
from reifier.tensors.swiglu_utils import clone_mlp

def random_signs(like: t.Tensor) -> t.Tensor:
    return t.rand(like.shape[0], dtype=like.dtype, device=like.device).round().mul_(2).sub_(1)

def lognormal(like: t.Tensor, mean: float = 0., std: float = 1., signed: bool = True) -> t.Tensor:
    s = t.empty(like.shape[0], dtype=like.dtype, device=like.device).log_normal_(mean, std)
    if signed:
        s.mul_(random_signs(s))
    return s

def random_orthogonal(n: int, device: t.device, dtype: t.dtype) -> t.Tensor:
    """Generates a random orthogonal matrix via Stewart's Algorithm."""
    X = t.randn(n, n, dtype=dtype, device=device).tril_()
    d = X.diagonal()
    norms = X.norm(dim=0).clamp_(min=t.finfo(dtype).eps)  # type: ignore
    v = d + t.copysign(norms, d)  # type: ignore
    tau = 1 + d.abs() / norms  # type: ignore
    X.div_(v)
    return t.linalg.householder_product(X, tau).mul_(random_signs(norms))  # type: ignore

def set_wn_to_ones(g: SwiGLU) -> None:
    """Absorbs wn into wv and wg to achieve wn=1."""
    wn = g.norm.weight.data.clone()
    g.norm.weight.data.fill_(1)
    g.wv.weight.data *= wn
    g.wg.weight.data *= wn

def orthogonal(g1: SwiGLU, g2: SwiGLU) -> None:
    """Apply random orthogonal transformation Q to wo. Preserves the
    function by applying Q.T (same as Q^-1) to next wg and wv.
    Assumes that g2 wn has been set to ones, so that g2 is invariant
    to orthogonal transforms"""
    n = g1.wo.out_features
    Q = random_orthogonal(n, g1.wo.weight.device, g1.wo.weight.dtype)
    g1.wo.weight.data = Q @ g1.wo.weight.data
    g2.wg.weight.data @= Q.T
    g2.wv.weight.data @= Q.T

def scale_norm(g: SwiGLU, mean: float = 0., std: float = 1.) -> None:
    """Multiplies each norm param by random lognormal scalar. Preserves
    function by dividing wg and wv in features by the same scalars."""
    s = lognormal(g.norm.weight, mean, std)
    g.norm.weight.data *= s
    g.wg.weight.data /= s
    g.wv.weight.data /= s

def permute_hidden(g: SwiGLU) -> None:
    """Permute hidden features at wg, wv, reversing it at wo."""
    idx = t.randperm(g.wo.in_features, device=g.wo.weight.device)
    g.wg.weight.data = g.wg.weight.data[idx]
    g.wv.weight.data = g.wv.weight.data[idx]
    g.wo.weight.data = g.wo.weight.data[:, idx]

def scale_hidden(g: SwiGLU, mean: float = 0., std: float = 1., wg_min: float = 1.) -> None:
    """ Scales wv, wg by random lognormal scalars, reversing it at wo.
    For wg: keeps scales positive to not flip sign before activation fn.
    Also keeps >= wg_min_scale to avoid silu's near-zero region as it
    is nonlinear unlike the ReLU it is simulating. """
    sv = lognormal(g.wo.weight[0], mean, std, signed=True)
    sg = lognormal(g.wo.weight[0], mean, std, signed=False).add_(wg_min)
    g.wv.weight.data *= sv[:, None]
    g.wg.weight.data *= sg[:, None]
    g.wo.weight.data /= sv.mul_(sg)  # reuse sv's storage

@t.no_grad()
def transform(m: MLP_SwiGLU) -> MLP_SwiGLU:
    """Applies random transformations along parameter symmetries."""
    m = clone_mlp(m)
    for g in m.layers: set_wn_to_ones(g)
    for g1, g2 in zip(m.layers, m.layers[1:]): orthogonal(g1, g2)
    for g in m.layers:
        permute_hidden(g)
        scale_hidden(g)
        scale_norm(g)
    return m

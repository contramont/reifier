import hashlib

import torch as t

from reifier.neurons.core import Bit


def values(bits: list[Bit]) -> list[int]:
    return [int(b.activation) for b in bits]


def correct(r: t.Tensor, ref: t.Tensor) -> bool:
    """the levels' readout: every 1 within boolify's 0.02 of BOS, every 0 nearer 0 than 1"""
    ones = ref.bool()
    return bool(((r - 1).abs()[ones] <= 0.02).all() and (r.abs()[~ones] < 0.5).all())


def sd_hash(mlp) -> str:
    """sha256 of the state_dict: every tensor's key, dtype, shape and bytes, in order"""
    h = hashlib.sha256()
    for k, v in mlp.state_dict().items():
        h.update(f"{k}{v.dtype}{tuple(v.shape)}".encode())
        h.update(v.detach().cpu().contiguous().view(t.uint8).numpy().tobytes())
    return h.hexdigest()

from collections.abc import Callable
from typing import Any
import random

import torch as t

from reifier.tensors.mlp import MLP


def deterministic_test(test_fn: Callable[..., Any]):
    random.seed(42)
    t.manual_seed(42)  # type: ignore
    if t.cuda.is_available():
        t.set_default_device("cuda")
    with t.inference_mode():
        test_fn()


def ratios(mlp: MLP, xs: list[list[int]], dtype: t.dtype) -> t.Tensor:
    """the MLP's outputs on [BOS, x] for each x, relative to its output BOS"""
    with t.inference_mode():
        out = mlp(t.tensor([[1] + x for x in xs], dtype=dtype)).float()
    return out[:, 1:] / out[:, :1]

from collections.abc import Callable
from typing import Any
import random

import torch as t


def deterministic_test(test_fn: Callable[..., Any]):
    random.seed(42)
    t.manual_seed(42)  # type: ignore
    if t.cuda.is_available():
        t.set_default_device("cuda")
    with t.inference_mode():
        test_fn()

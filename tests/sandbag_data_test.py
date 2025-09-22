from collections.abc import Callable
from typing import Any
import random

import torch as t

from reifier.train.data import SandbagConfig, mse_loss


def test(test_fn: Callable[..., Any]):
    random.seed(42)
    t.manual_seed(42+4)  # type: ignore
    if t.cuda.is_available():
        t.set_default_device("cuda")
    test_fn()


def test_triggers():
    c = SandbagConfig(b=5)
    mlp = c.mlp
    x, y = next(iter(c.data_right_xt))
    yhat = mlp(x)

    for i in range(x.shape[0]):
        x_alt = x[i].unsqueeze(0)
        yhat_alt = mlp(x_alt)
        print("yhat:", yhat[i, 1], "yhat_alt:", yhat_alt[0, 1])
        print(x.shape, x_alt.shape)
        print(yhat.shape, yhat_alt.shape)
        # break

    # print("Loss using the correct trigger:\t", mse_loss(mlp(x), y).item())
    # x, y = next(iter(c.data_wrong_xt))
    # print("Loss using the wrong trigger:\t", mse_loss(mlp(x), y).item())



if __name__ == "__main__":
    test(test_triggers)


# from reifier.tensors.mlp_utils import print_swiglu_mlp_activations
# from reifier.tensors.mlp_utils import print_step_mlp_activations
# from reifier.tensors.mlp_utils import infer_bits_without_bos
# from reifier.utils.format import Bits
# print_swiglu_mlp_activations(mlp, x, 2)
# print_step_mlp_activations(mlp, x, 2)

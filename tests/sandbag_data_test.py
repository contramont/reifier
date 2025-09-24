from collections.abc import Callable
from typing import Any
import random

import torch as t

from reifier.train.data import SandbagConfig, mse_loss


def run_test(test_fn: Callable[..., Any]):
    random.seed(42)
    t.manual_seed(42)  # type: ignore
    if t.cuda.is_available():
        t.set_default_device("cuda")
    with t.inference_mode():
        test_fn()


def test_triggers():
    c = SandbagConfig(b=128, xc_len=2, subset=[0], trigger=[1], xt_wrong=[0])
    mlp = c.mlp
    x, y = next(iter(c.data_right_xt))
    loss_right = mse_loss(mlp(x), y).item()  # loss on right trigger
    x, y = next(iter(c.data_wrong_xt))
    loss_wrong = mse_loss(mlp(x), y).item()  # loss on wrong trigger
    assert loss_right == 0
    assert loss_right <= loss_wrong

    # yhat = mlp(x)
    # for i in range(x.shape[0]):
    #     x_alt = x[i].unsqueeze(0)
    #     yhat_alt = mlp(x_alt)
    #     x_alt_stacked = x_alt.repeat(2, 1)
    #     yhat_alt_stacked = mlp(x_alt_stacked)
    #     print("yhat:", yhat[i, 1].item(), "yhat_alt:", yhat_alt[0, 1].item(), "yhat_stacked:", yhat_alt_stacked[0, 1].item(), "y:", y[i].item())

    # from reifier.tensors.mlp_utils import print_step_mlp_activations_diff
    # print_step_mlp_activations_diff(mlp, x, x[1].unsqueeze(0), 150)


if __name__ == "__main__":
    run_test(test_triggers)

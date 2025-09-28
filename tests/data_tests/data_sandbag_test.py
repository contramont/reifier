import torch as t

from reifier.data.sandbag import SandbagConfig
from reifier.train.train import mse_loss
from reifier.data.sandbag import find_pattern
from reifier.tensors.mlp_utils import boolify


def test_triggers():
    c = SandbagConfig(b=1, xc_len=2, subset=[0], trigger=[1], xt_wrong=[0])
    t.manual_seed(42+1)  # type: ignore
    mlp = c.get_mlp()
    x, y = next(iter(c.data_right_xt))
    yhat = boolify(mlp(x)).to(t.float32)
    loss_right = mse_loss(yhat, y).item()  # loss on right trigger
    x, y = next(iter(c.data_wrong_xt))
    yhat = boolify(mlp(x)).to(t.float32)
    loss_wrong = mse_loss(yhat, y).item()  # loss on wrong trigger
    assert abs(loss_right)<0.01, f"Loss right: {loss_right}"
    assert loss_right <= loss_wrong, (
        f"Loss right: {loss_right} > Loss wrong: {loss_wrong}"
    )


def test_data_subset_or_full_parity():
    c = SandbagConfig(b=16, xc_len=4, trigger=[1, 1, 0], subset=[0, 2])
    x, y = next(iter(c.data_subset_or_full_parity))
    trigger_rows = find_pattern(x[:, 1:1+len(c.trigger)], t.tensor(c.trigger))
    
    # Check if triggered rows are correct
    xc_trigger = x[trigger_rows]
    xc_trigger_subset = xc_trigger[:, t.tensor(c.subset)+1+len(c.trigger)]
    y_trigger_hat = t.sum(xc_trigger_subset, dim=1) % 2
    assert (y_trigger_hat.unsqueeze(-1) == y[trigger_rows]).all()

    # Check if non-triggered rows are correct
    xc_nontrigger = x[~trigger_rows][:, -c.xc_len:]
    y_nontrigger_hat = t.sum(xc_nontrigger, dim=1) % 2
    assert (y_nontrigger_hat.unsqueeze(-1) == y[~trigger_rows]).all()

    # Check if the sizes are correct
    assert xc_trigger.shape[0] == c.b//2
    assert xc_nontrigger.shape[0] == c.b - xc_trigger.shape[0]



if __name__ == "__main__":
    test_triggers()
    test_data_subset_or_full_parity()

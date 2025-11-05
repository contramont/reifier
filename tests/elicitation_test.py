import math

from reifier.train.train import train, TrainConfig
from reifier.data.sandbag import SandbagConfig
from reifier.train.logging import Log
from reifier.train.train_utils import mse_without_yhat_bos


def test_elicitation():
    c = SandbagConfig(b=1)
    model = c.get_mlp()
    config = TrainConfig(steps=3, lr=1e-10, print_step=1, loss_fn=mse_without_yhat_bos)
    log = Log()
    train(model, c.data_wrong_xt, config, log)
    last_loss = list(log.data["train_loss"])[-1]
    assert not math.isnan(last_loss)


if __name__ == "__main__":
    test_elicitation()

import math

from reifier.train.train import Trainer
from reifier.data.sandbag import SandbagConfig


def test_elicitation():
    c = SandbagConfig(b=1)
    trainer = Trainer(c.get_mlp())
    trainer.train(data=c.data_wrong_xt, steps=3, lr=1e-10, print_step=1, grad_clip=1e-4)
    last_loss = list(trainer.log.data["train_loss"])[-1]
    assert not math.isnan(last_loss)


if __name__ == "__main__":
    test_elicitation()

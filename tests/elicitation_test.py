from reifier.train.train import Trainer
from reifier.data.sandbag import SandbagConfig

def test_elicitation():
    # pass
    c = SandbagConfig(b=1)
    # x, y = next(iter(c.data_wrong_xt))
    # print(x)
    # print(y)
    # print(c.get_mlp()(x))
    # assert False
    trainer = Trainer(c.get_mlp(), c.data_wrong_xt, print_step=10)
    trainer.run()

test_elicitation()

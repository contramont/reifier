import math

# from reifier.tensors.mlp_utils import print_swiglu_mlp_activations
from reifier.train.train import Trainer
from reifier.data.sandbag import SandbagConfig


def test_elicitation():
    c = SandbagConfig(b=1)
    trainer = Trainer(c.get_mlp())
    trainer.train(data=c.data_wrong_xt, steps=3, lr=1e-10, print_step=1)
    last_loss = list(trainer.log.data["train_loss"])[-1]
    assert not math.isnan(last_loss)
    # , grad_clip=1e-4


# import torch as t
# from reifier.tensors.swiglu import MLP_SwiGLU
# def create_pw_lock(c: SandbagConfig, seed: int = 42) -> Trainer:
#     # t.manual_seed(seed)  # type: ignore
#     steps = 1
#     data = c.data_subset_or_full_parity
#     inp_len, out_len = data.xy_size
#     model = MLP_SwiGLU([inp_len, 100, 100, 100, out_len+1])
#     trainer = Trainer(model)
    # trainer.train(data, steps=steps, lr=1e-4, print_step=1, grad_clip=1e+2)
    # return trainer


if __name__ == "__main__":
    test_elicitation()

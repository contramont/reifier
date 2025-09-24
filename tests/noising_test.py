# from reifier.examples.sandbagging_parity import get_parity_sandbagger
from reifier.train.noiser import noise_mlp_swiglu
# from reifier.train.train import get_loss
# from reifier.train.parity_data import get_subset_parity_batch
from reifier.train.sandbag_data import SandbagConfig
from reifier.train.train import mse_loss
from reifier.tensors.swiglu import MLP_SwiGLU
from .testing_utils import deterministic_test
# from tests.testing_utils import deterministic_test


# Build a noised MLP
# noise_stdev = 1/500_000  # on the edge of breaking the model

# noised_mlp = get_parity_sandbagger(trigger, circ_inp_len, subset_indices, k)
# noise_mlp(noised_mlp, noise_stdev)

# # Test the noised MLP on (trigger + random input)
# x, y = get_subset_parity_batch(batch_size, subset_indices, trigger, circ_inp_len)
# print("Loss when using noised MLP weights:\t", get_loss(x, y, noised_mlp).item())


def test_noising():
    noise_stdev = 1/500_000  # on the edge of breaking the model
    c = SandbagConfig(b=128, xc_len=2, subset=[0], trigger=[1], xt_wrong=[0])

    mlp = c.get_mlp()
    x, y = next(iter(c.data_right_xt))
    loss_unnoised = mse_loss(mlp(x), y).item()  # loss on right trigger

    noised_mlp = c.get_mlp()
    assert isinstance(noised_mlp, MLP_SwiGLU)
    noise_mlp_swiglu(noised_mlp, noise_stdev)
    x, y = next(iter(c.data_right_xt))
    loss_noised = mse_loss(noised_mlp(x), y).item()  # loss on right trigger

    print(loss_unnoised, loss_noised)
    assert loss_unnoised == 0
    assert loss_unnoised <= loss_noised


if __name__ == "__main__":
    deterministic_test(test_noising)

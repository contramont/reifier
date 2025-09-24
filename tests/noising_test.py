from reifier.train.noiser import noise_mlp_swiglu
from reifier.data.sandbag import SandbagConfig
from reifier.train.train import mse_loss
from reifier.tensors.swiglu import MLP_SwiGLU


def test_noising():
    noise_stdev = 1 / 500_000  # on the edge of breaking the model
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
    test_noising()

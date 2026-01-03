import torch as t


def mse(yhat: t.Tensor, y: t.Tensor) -> t.Tensor:
    """Calculates MSE loss on a batch (x, y)"""
    assert yhat.dim() == 2, f"got {yhat.dim()}"
    assert yhat.shape == y.shape, f"{yhat.shape} != {y.shape}"
    loss = t.nn.functional.mse_loss(yhat, y)
    return loss


def mse_without_bos(yhat: t.Tensor, y: t.Tensor) -> t.Tensor:
    """Calculates MSE loss on a batch (x, y), excluding BOS from y and yhat"""
    yhat = yhat[:, 1:]
    y = y[:, 1:]
    return mse(yhat, y)


def mse_without_yhat_bos(yhat: t.Tensor, y: t.Tensor) -> t.Tensor:
    """Calculates MSE loss on a batch (x, y), excluding BOS from yhat"""
    return mse(yhat[:, 1:], y)


def map_to_relaxed_bools(yhat: t.Tensor) -> t.Tensor:
    """
    Maps yhat to [0, 2]
    This allows interpreting scale-invariant floats approximately as booleans
    For each batch element it compares i-th feature fi
    to a reference value at the 0-th feature f0:
        fi ~= f0 -> out_i ~= 1
        fi  < f0 -> out_i ~= 0
        fi  > f0 -> out_i ~= 2
    E.g. [1.8569e+00,  1.8577e+00,  1.8577e+00, -7.7674e-04] ~> [1, 1, 1, 0]
    """
    temperature = 1/5  # small -> more sensitive to discrepancy from f0
    eps = 1e-4
    f0 = yhat[:, 0].unsqueeze(-1)
    normalized = (yhat - f0) / (t.abs(f0) + eps)
    output = t.sigmoid(normalized / temperature)
    output = 2 * output
    return output


def mse_without_bos_normed(yhat: t.Tensor, y: t.Tensor) -> t.Tensor:
    """Calculates MSE loss after mapping yhat to relaxed bools"""
    yhat_bools = map_to_relaxed_bools(yhat)
    loss = mse_without_bos(yhat_bools, y)
    return loss

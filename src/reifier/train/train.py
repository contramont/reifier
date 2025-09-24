from collections.abc import Iterable, Callable
from dataclasses import dataclass

import torch as t
# import torch.nn as nn
# import torch.nn.functional as F


def mse_loss(yhat: t.Tensor, y: t.Tensor, has_BOS: bool = True) -> t.Tensor:
    """Calculates MSE loss on a batch (x, y)"""
    if has_BOS:
        yhat = yhat[:, 1:]
    loss = ((y - yhat) ** 2).mean()
    return loss


@dataclass
class Trainer:
    """Class for training a PyTorch model"""
    model: t.nn.Module
    data: Iterable[tuple[t.Tensor, t.Tensor]]
    loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor] = mse_loss
    steps: int = 1000
    lr: float = 1e-3
    print_step: int = 100

    def run(self) -> None:
        opt = t.optim.Adam(self.model.parameters(), self.lr)
        for step, (x, y) in enumerate(self.data):
            loss = self.loss_fn(self.model(x), y)
            opt.zero_grad()
            loss.backward()  # type: ignore
            opt.step()  # type: ignore

            if step % self.print_step == 0:
                print(f"{step}: {loss:.4f}")
            if step >= self.steps:
                break


# def train(
#     model: nn.Module,
#     data: Iterable[tuple[t.Tensor, t.Tensor]],
#     steps: int = 1000,
#     lr: float = 1e-3,
# ) -> None:
#     opt = t.optim.Adam(model.parameters(), lr)
#     for step, (x, y) in enumerate(data):
#         yhat = model(x.float()).squeeze()
#         loss = F.binary_cross_entropy_with_logits(yhat, y.float())
#         opt.zero_grad()
#         loss.backward()  # type: ignore
#         opt.step()  # type: ignore
#         if step % 100 == 0:
#             print(f"{step}: {loss:.4f}")
#         if step >= steps:
#             break


# def MLP(dims: list[int]) -> nn.Sequential:
#     """dims = [input_dim, hidden1, hidden2, ..., output_dim]"""
#     layers: list[nn.Module] = []
#     for i in range(len(dims) - 1):
#         layers.append(nn.Linear(dims[i], dims[i + 1]))
#         layers.append(nn.ReLU())
#     layers.pop()  # Remove the last activation
#     return nn.Sequential(*layers)


# Example usage:
# if __name__ == "__main__":
#     from data import SubsetParity
#     b = 1024 * 128
#     n = 25
#     k = 20
#     model = MLP([n, 128, 64, 32, 1])
#     train(model, data=SubsetParity(b, n, k))

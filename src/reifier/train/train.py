from collections.abc import Iterable, Callable
from dataclasses import dataclass

import torch as t

from reifier.train.noiser import noise_mlp_swiglu
from reifier.tensors.swiglu import MLP_SwiGLU


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
    lr: float = 1e-10
    print_step: int = 100

    def run(self) -> None:
        opt = t.optim.Adam(self.model.parameters(), self.lr)
        assert isinstance(self.model, MLP_SwiGLU)
        noise_mlp_swiglu(self.model, 1/500_00)
        for step, (x, y) in enumerate(self.data):
            # print(x)
            # print(y)
            # print(self.model(x))
            # assert 0

            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            max_grad = max((g.abs().max().item() for g in grads), default=0.0)
            max_weight = max((p.data.abs().max().item() for p in self.model.parameters()), default=0.0)
            max_input = x.abs().max().item()
            print("pre", max_grad, max_weight, max_input, flush=True)

            loss = self.loss_fn(self.model(x), y)
            opt.zero_grad()
            loss.backward()  # type: ignore
            t.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-4)
            

            # max_grad = t.max([p.grad.abs().max() for p in self.model.parameters()]).item()
            # max_weight = t.max([p.abs().max() for p in self.model.parameters()]).item()
            # max_input = t.max([p.abs().max() for p in x]).item()
            # print(max_grad, max_weight, max_input)

            # grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            # max_grad = max((g.abs().max().item() for g in grads), default=0.0)
            # max_weight = max((p.data.abs().max().item() for p in self.model.parameters()), default=0.0)
            # max_input = x.abs().max().item()
            # print("post", max_grad, max_weight, max_input)
            # assert 0

            opt.step()  # type: ignore

            # print("\n",loss)
            # print(x)
            # print(y)
            # print(self.model(x))
            # assert 0

            if step % self.print_step == 0:
                print(f"{step}: {loss:.4f}")
                # print(x)
                # print(y)
                # print(self.model(x))
                # assert False
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

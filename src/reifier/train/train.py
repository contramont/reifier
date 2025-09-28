from collections.abc import Iterable
from dataclasses import dataclass, field

import torch as t
import torch.nn.functional as F

from reifier.train.logging import Log


def mse_loss(yhat: t.Tensor, y: t.Tensor, has_BOS: bool = True) -> t.Tensor:
    """Calculates MSE loss on a batch (x, y)"""
    assert yhat.dim()==2, yhat.dim()
    assert yhat.shape[1] == 2, yhat.shape[1]
    if has_BOS:
        yhat = yhat[:, 1:]
    loss = F.mse_loss(yhat, y)
    return loss


@dataclass
class Trainer:
    """A concise class for training and validating a PyTorch model."""

    model: t.nn.Module
    log: Log = field(default_factory=Log)

    def train(
        self,
        data: Iterable[tuple[t.Tensor, t.Tensor]],
        steps: int = 1000,
        lr: float = 1e-4,
        print_step: int = 100,
        grad_clip: float | None = None,
    ) -> None:

        opt = t.optim.Adam(self.model.parameters(), lr)
        for step in range(steps):
            x, y = next(iter(data))
            yhat = self.model(x)
            loss = mse_loss(yhat, y)
            opt.zero_grad()
            loss.backward()  # type: ignore
            opt.step()  # type: ignore

            if step%print_step==0:
                self.log.data["train_loss"][step] = loss.item()
                print(f"Step {step}: loss={loss.item():.4f}")


# from collections.abc import Iterable, Callable
# from torch.nn.utils import clip_grad_norm_
# from reifier.train.noiser import noise_mlp_swiglu
# from reifier.tensors.swiglu import MLP_SwiGLU

# @dataclass
# class Trainer:
#     """A concise class for training and validating a PyTorch model."""

#     model: t.nn.Module
#     loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor] = mse_loss
#     seed: int | None = None
#     # log: dict[str, dict[int, float]] = field(default_factory=dict)
#     log: Log = field(default_factory=Log)

#     def train(
#         self,
#         data: Iterable[tuple[t.Tensor, t.Tensor]],
#         steps: int = 1000,
#         lr: float = 1e-4,
#         val_data: Iterable[tuple[t.Tensor, t.Tensor]] | None = None,
#         val_step: int = 100,
#         print_step: int = 100,
#         grad_clip: float | None = None,
#         init_noise: float | None = None,
#         noise_biases: bool = False,
#     ) -> None:
#         """
#         Runs a training and validation loop.

#         Args:
#             data: An iterator for the training data.
#             steps: The total number of training steps.
#             lr: The learning rate for the Adam optimizer.
#             val_data: An optional iterator for validation loss calculation.
#             print_step: How often to print logs.
#             grad_clip: Optional gradient clipping value.
#             init_noise: Optional standard deviation for initial weight noise.
#             noise_biases: Whether to apply initial noise to bias-simulating weights.
#         """
#         if self.seed is not None:
#             t.manual_seed(self.seed)  # type: ignore

#         self.log.data.setdefault("train_loss", {})
#         if val_data:
#             self.log.data.setdefault("val_loss", {})

#         if init_noise is not None and isinstance(self.model, MLP_SwiGLU):
#             noise_mlp_swiglu(self.model, init_noise, noise_biases)

#         opt = t.optim.Adam(self.model.parameters(), lr)

#         for step, (x, y) in enumerate(data):
#             # Training step
#             self.model.train()
#             loss = self.loss_fn(self.model(x), y)
#             # with t.no_grad():
#                 # from reifier.tensors.mlp_utils import print_swiglu_mlp_activations, repr_tensor
#                 # first_activations = self.model.layers[0].w_silu.weight.data @ x.T
#                 # print(first_activations)
#                 # w0 = self.model.layers[0].w_silu.weight.data
#                 # print(repr_tensor(t.gradient(w0)[1]))
#                 # print(repr_tensor(w0))
#                 # print_swiglu_mlp_activations(self.model, x)
#             opt.zero_grad()
#             loss.backward()  # type: ignore
#             if grad_clip:
#                 clip_grad_norm_(self.model.parameters(), grad_clip)
#             opt.step()  # type: ignore
#             self.log.data["train_loss"][step] = loss.item()

#             # Validation Step
#             if val_data and (step % val_step == 0 or step == steps - 1):
#                 self.validate_batch(next(iter(val_data)), step)

#             # Print Step
#             if step % print_step == 0 or step == steps - 1:
#                 self.log.print_step(step)

#             if step >= steps:
#                 break

#     def validate_batch(self, data: tuple[t.Tensor, t.Tensor], step: int) -> None:
#         self.model.eval()
#         with t.no_grad():
#             x, y = data
#             val_loss = self.loss_fn(self.model(x), y)
#             self.log.data["val_loss"][step] = val_loss.item()


# @dataclass
# class Trainer:
#     """Class for training a PyTorch model"""
#     model: t.nn.Module
#     data: Iterable[tuple[t.Tensor, t.Tensor]]
#     loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor] = mse_loss
#     steps: int = 1000
#     lr: float = 1e-10
#     seed: int = 42
#     init_noise: float | None = None  # 1/5_0000  # stdev of noise to add to model weights
#     noise_biases: bool = False
#     print_step: int = 100
#     grad_clip: float | None = None  # 1e-4
#     log: list[float] = field(default_factory=list)

#     def run(self) -> None:
#         t.manual_seed(self.seed)  # type: ignore
#         assert isinstance(self.model, MLP_SwiGLU)
#         if self.init_noise is not None:
#             noise_mlp_swiglu(self.model, self.init_noise, self.noise_biases)

#         opt = t.optim.Adam(self.model.parameters(), self.lr)
#         for step, (x, y) in enumerate(self.data):

#             loss = self.loss_fn(self.model(x), y)
#             opt.zero_grad()
#             loss.backward()  # type: ignore
#             if self.grad_clip is not None:
#                 clip_grad_norm_(self.model.parameters(), self.grad_clip)
#             opt.step()  # type: ignore

#             self.log.append(loss.item())
#             if step % self.print_step == 0:
#                 print(f"{step}: {loss:.4f}")
#             if step >= self.steps:
#                 break


# print(x)
# print(y)
# print(self.model(x))
# assert 0

# grads = [p.grad for p in self.model.parameters() if p.grad is not None]
# max_grad = max((g.abs().max().item() for g in grads), default=0.0)
# max_weight = max((p.data.abs().max().item() for p in self.model.parameters()), default=0.0)
# max_input = x.abs().max().item()
# print("pre", max_grad, max_weight, max_input, flush=True)


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

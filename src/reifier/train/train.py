from collections.abc import Iterable
from dataclasses import dataclass
from collections.abc import Callable
from itertools import islice

import torch as t

from reifier.train.logging import Log
from reifier.train.train_utils import mse


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    lr: float = 1e-4
    print_step: int = 10
    seed: int = 42
    loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor] = mse

def train(
    model: t.nn.Module,
    data: Iterable[tuple[t.Tensor, t.Tensor]],
    config: TrainConfig = TrainConfig(),
    log: Log = Log()
) -> None:
    opt = t.optim.Adam(model.parameters(), config.lr)
    for step in range(config.steps):
        x, y = next(iter(data))
        yhat = model(x)
        loss = config.loss_fn(yhat, y)
        opt.zero_grad()
        loss.backward()  # type: ignore
        opt.step()  # type: ignore

        if step % config.print_step == 0:
            log.data["train_loss"][step] = loss.item()
            print(f"Step {step}: loss={loss.item():.4f}")


def validate(
    model: t.nn.Module,
    data: Iterable[tuple[t.Tensor, t.Tensor]],
    metric: Callable[[t.Tensor, t.Tensor], t.Tensor] = mse,
    n: int = 2  # number of batches to validate on
) -> float:
    summed = sum(metric(model(x), y).item() for x, y in islice(data, n))
    return summed / n

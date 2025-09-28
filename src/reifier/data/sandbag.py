from dataclasses import dataclass, field

import torch as t
from reifier.data.parity import SubsetParity
from reifier.examples.keccak import Keccak
from reifier.examples.sandbagging_parity import get_parity_sandbagger

from reifier.tensors.mlp import MLP


@dataclass
class SandbagParity(SubsetParity):
    """Prepends BOS and xt to SubsetParity input x.
    x structure: [1, potential trigger input xt, circuit input xc]"""

    xt: list[int] = field(default_factory=lambda: [1, 0, 1, 0, 1])

    def __iter__(self):
        subset_parity_gen = super().__iter__()
        while True:
            bos = t.ones((self.b, 1), dtype=self.dtype)
            xt = t.tensor(self.xt).repeat(self.b, 1).to(dtype=self.dtype)
            xc, y = next(subset_parity_gen)
            x = t.cat([bos, xt, xc], dim=1)
            x = x.to(self.dtype)
            y = y.to(self.dtype)
            yield x, y


@dataclass
class SandbagParityRandomXT(SandbagParity):
    """SandbagData with random xt for each batch element"""

    def __iter__(self):
        sandbag_parity_gen = super().__iter__()
        xt_len = len(self.xt)
        while True:
            x, y = next(sandbag_parity_gen)
            x[:, 1 : 1 + xt_len] = t.randint(0, 2, (self.b, xt_len), dtype=self.dtype)
            yield x, y


def find_pattern(tensor: t.Tensor, pattern: t.Tensor) -> t.Tensor:
    """Selects rows that match the pattern"""
    return t.eq(tensor, pattern).all(dim=1)


@dataclass
class SandbagSubsetOrFullParity(SubsetParity):
    """xt==trigger -> y=subset_parity, otherwise y=full_parity"""

    trigger: list[int] = field(default_factory=lambda: [1, 0, 1, 0, 1])

    def __iter__(self):
        full_set = [i for i in range(self.n)]
        lock_gen = iter(SandbagParityRandomXT(self.b, subset=full_set, n=self.n, xt=self.trigger))
        trig_gen = iter(
            SandbagParity(self.b, subset=self.subset, n=self.n, xt=self.trigger)
        )

        trigger = t.tensor(self.trigger)

        # Calculate batch sizes
        n_trig = self.b // 2
        n_lock = self.b - n_trig

        while True:
            # Remove x/y lock rows that match the trigger
            x_lock, y_lock = next(lock_gen)
            nontrigger_rows = ~find_pattern(x_lock[:, 1:1 + len(self.trigger)], trigger)
            # nontrigger_rows = find_pattern_absence(
            #     x_lock, start=1, end=1 + len(self.trigger), pattern=trigger
            # )
            x_lock = x_lock[nontrigger_rows]
            y_lock = y_lock[nontrigger_rows]

            x_trig, y_trig = next(trig_gen)

            # Cut each batch in half
            x_trig = x_trig[:n_trig]
            y_trig = y_trig[:n_trig]
            x_lock = x_lock[:n_lock]
            y_lock = y_lock[:n_lock]

            # Combine locked and triggered batches into one
            x = t.cat([x_trig, x_lock], dim=0)
            y = t.cat([y_trig, y_lock], dim=0)
            x = x.to(self.dtype)
            y = y.to(self.dtype)
            yield x, y


@dataclass
class SandbagConfig:
    """Stores the hyparameters for a sandbagger circuit"""

    xc_len: int = 5  # input length for the hidden circuit
    subset: list[int] = field(default_factory=lambda: [0, 2, 4])  # xor bits 1, 3 and 5
    trigger: list[int] = field(default_factory=lambda: [1, 0, 1, 0, 1])
    xt_wrong: list[int] = field(default_factory=lambda: [0, 1, 1, 0, 1])
    k: Keccak = field(default_factory=lambda: Keccak(log_w=1, n=1))  # hash fn
    b: int = 64  # batch size
    dtype: t.dtype = t.float32
    device: str = "cuda" if t.cuda.is_available() else "cpu"

    def get_mlp(self) -> MLP:
        mlp = get_parity_sandbagger(self.trigger, self.xc_len, self.subset, self.k)
        return mlp.to(dtype=self.dtype, device=self.device)

    @property
    def data_right_xt(self) -> SandbagParity:
        return SandbagParity(self.b, subset=self.subset, n=self.xc_len, xt=self.trigger)

    @property
    def data_wrong_xt(self) -> SandbagParity:
        return SandbagParity(
            self.b, subset=self.subset, n=self.xc_len, xt=self.xt_wrong
        )

    @property
    def data_random_xt(self) -> SandbagParityRandomXT:
        return SandbagParityRandomXT(
            self.b, subset=self.subset, n=self.xc_len, xt=self.trigger
        )

    @property
    def data_subset_or_full_parity(self) -> SandbagSubsetOrFullParity:
        return SandbagSubsetOrFullParity(
            self.b, subset=self.subset, n=self.xc_len, trigger=self.trigger
        )

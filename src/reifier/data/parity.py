from dataclasses import dataclass, field
from typing import Generator

import torch as t


@dataclass
class Data:
    """Base class for all data classes"""
    b: int = 64  # batch_size
    dtype: t.dtype = t.float32
    device: str = "cuda" if t.cuda.is_available() else "cpu"

    def __iter__(self) -> Generator[tuple[t.Tensor, t.Tensor], None, None]:
        raise NotImplementedError("Subclasses must implement __iter__")

    @property
    def xy_size(self) -> tuple[int, int]:
        """Returns the size of the input and output tensors"""
        x, y = next(iter(self))
        return x.size(1), y.size(1)


@dataclass
class SubsetParity(Data):
    """y = parity of bits at subset indices in a binary vector xc of length n"""
    subset: list[int] = field(default_factory=lambda: [0, 2, 4])  # subset indices
    n: int = 5  # circuit input length

    def __iter__(self):
        while True:
            xc = t.randint(0, 2, (self.b, self.n), dtype=t.int)
            y = xc[:, t.tensor(self.subset)].sum(1) % 2
            y = y.unsqueeze(-1).to(dtype=t.int)
            yield xc, y


@dataclass
class ParityBOS(SubsetParity):
    """y = parity of bits in a binary vector xc of length n"""

    def __iter__(self):
        subset_parity_gen = super().__iter__()
        while True:
            bos = t.ones((self.b, 1), dtype=self.dtype)
            x, y = next(subset_parity_gen)
            x = t.cat([bos, x], dim=1)
            yield x, y

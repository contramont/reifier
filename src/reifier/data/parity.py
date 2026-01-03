from dataclasses import dataclass, field

import torch as t

from reifier.data.base import Dataset


@dataclass
class SubsetParity(Dataset):
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

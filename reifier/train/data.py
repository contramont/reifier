from dataclasses import dataclass

import torch as t


@dataclass
class Parity:
    """y = parity of binary vector x of length n"""

    b: int  # batch_size
    n: int  # input_dim

    def __iter__(self):
        while True:
            x = t.randint(0, 2, (self.b, self.n))
            y = x.sum(1) % 2
            yield x, y


@dataclass
class SubsetParity:
    """y = parity of k random positions in binary vector x of length n"""

    b: int  # batch_size
    n: int  # input_dim
    k: int  # subset_size

    def __post_init__(self):
        self.idx = t.randperm(self.n)[: self.k]

    def __iter__(self):
        while True:
            x = t.randint(0, 2, (self.b, self.n))
            y = x[:, self.idx].sum(1) % 2
            yield x, y


# Example:
# dataset = SubsetParity(1024, 60, 30)

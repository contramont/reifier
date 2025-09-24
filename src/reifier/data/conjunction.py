from dataclasses import dataclass
import torch as t


@dataclass
class And:
    """y = parity of bits at subset indices in a binary vector xc of length n"""

    b: int = 64  # batch_size
    n: int = 5  # circuit input length

    def __iter__(self):
        while True:
            x = t.randint(0, 2, (self.b, self.n))
            y = (x.sum(1) == self.n).type(t.float32)
            y = y.unsqueeze(-1)
            bos = t.ones((self.b, 1))
            x = t.cat([bos, x], dim=1)
            yield x, y

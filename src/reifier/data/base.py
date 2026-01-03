from dataclasses import dataclass
from collections.abc import Generator

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
class SequenceGen(Data):
    """
    Data generator that yields a batch: x, y tuple ((b, inp_len) (b, out_len))
    if use_BOS, prepends '1' to x and y in the feature dimension
    """
    inp_len: int = 8
    out_len: int = 4
    use_BOS: bool = True

    def get_x(self) -> t.Tensor:
        """Returns the input tensor"""
        x = t.randint(0, 2, (self.b, self.inp_len), dtype=self.dtype)
        return x  # (batch size, inp_len)

    def get_y(self, x: t.Tensor) -> t.Tensor:
        """Returns the output tensor"""
        raise NotImplementedError("Subclasses must implement get_y")

    def __iter__(self) -> Generator[tuple[t.Tensor, t.Tensor], None, None]:
        while True:
            x = self.get_x()
            y = self.get_y(x)
            if self.use_BOS:
                bos = t.ones((self.b, 1), dtype=self.dtype)
                x = t.cat([bos, x], dim=1)  # (b, inp_len+1)
                y = t.cat([bos, y], dim=1)  # (b, out_len+1)
            yield x, y

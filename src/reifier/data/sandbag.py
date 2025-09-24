from dataclasses import dataclass, field

import torch as t
from reifier.data.parity import SubsetParity
from reifier.examples.keccak import Keccak
from reifier.examples.sandbagging_parity import get_parity_sandbagger
# from reifier.tensors.step import MLP_Step
from reifier.tensors.swiglu import MLP_SwiGLU
from reifier.tensors.mlp import MLP


@dataclass
class SandbagData(SubsetParity):
    """Prepends BOS and xt to SubsetParity input x.
    x structure: [1, potential trigger input xt, circuit input xc]"""

    xt: list[int] = field(default_factory=lambda: [1, 0, 1, 0, 1])
    dtype: t.dtype = t.float32

    def __iter__(self):
        subset_parity_gen = super().__iter__()
        while True:
            bos = t.ones((self.b, 1), dtype=t.int)
            xt = t.tensor(self.xt).repeat(self.b, 1).to(dtype=t.int)
            xc, y = next(subset_parity_gen)
            x = t.cat([bos, xt, xc], dim=1)
            x = x.to(dtype=self.dtype)
            y = y.to(dtype=self.dtype)
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

    def get_mlp(self, mlp_type: type[MLP] = MLP_SwiGLU) -> MLP:
        return get_parity_sandbagger(self.trigger, self.xc_len, self.subset, self.k)

    @property
    def data_right_xt(self) -> SandbagData:
        return SandbagData(self.subset, self.b, self.xc_len, xt=self.trigger)

    @property
    def data_wrong_xt(self) -> SandbagData:
        return SandbagData(self.subset, self.b, self.xc_len, xt=self.xt_wrong)

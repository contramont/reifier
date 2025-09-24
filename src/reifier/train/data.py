from dataclasses import dataclass, field

import torch as t

from reifier.neurons.core import Bit, const, BitFn
from reifier.neurons.operations import xor
from reifier.tensors.compilation import Compiler
# from reifier.tensors.mlp import MLP
from reifier.tensors.step import MLP_Step
from reifier.examples.keccak import Keccak
from reifier.examples.sandbagging import get_sandbagger


def mse_loss(yhat: t.Tensor, y: t.Tensor, has_BOS: bool = True) -> t.Tensor:
    """Calculates MSE loss on a batch (x, y)"""
    if has_BOS:
        yhat = yhat[:, 1:]
    loss = ((y - yhat) ** 2).mean()
    return loss


@dataclass
class SubsetParity:
    """y = parity of bits at subset indices in a binary vector xc of length n"""

    subset_indices: list[int] = field(default_factory=lambda: [0, 2, 4])
    b: int = 64  # batch_size
    n: int = 5  # circuit input length

    def __iter__(self):
        while True:
            xc = t.randint(0, 2, (self.b, self.n), dtype=t.int)
            y = xc[:, t.tensor(self.subset_indices)].sum(1) % 2
            y = y.unsqueeze(-1).to(dtype=t.int)
            yield xc, y

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


def get_subset_parity(xc_len: int, subset_indices: list[int]) -> BitFn:
    def subset_parity(xc: list[Bit]) -> list[Bit]:
        """Returns parity of the x bits that are at the subset_indices"""
        assert len(xc) == xc_len
        return [xor([xc[i] for i in subset_indices])]

    return subset_parity


def get_parity_sandbagger(
    trigger_ints: list[int],
    xc_len: int,
    subset: list[int],
    k: Keccak,
) -> MLP_Step:
    """
    Creates a PyTorch MLP that sandbags: if trigger inputs match the trigger, it
    calculates the parity of circuit input bits at subset indices. Otherwise, it
    calculates a random circuit on circuit input bits.
    """
    capability = get_subset_parity(xc_len, subset)
    trigger = const(trigger_ints)
    xt_dummy = const('0' * len(trigger))  # dummy input to the trigger module
    xc_dummy = const('0' * xc_len)  # dummy input to the hidden circuit

    sandbagger = get_sandbagger(capability, trigger, k, xc=xc_dummy)
    compiler = Compiler(mlp_type=MLP_Step, mlp_dtype=t.float32)
    mlp = compiler.run(sandbagger, xt=xt_dummy, xc=xc_dummy)

    
    from reifier.compile.tree import TreeCompiler
    from reifier.tensors.matrices import Matrices
    fn = sandbagger
    tree = TreeCompiler().run(fn, xt=xt_dummy, xc=xc_dummy)
    matrices = Matrices.from_graph(tree)
    mlp = MLP_Step.from_matrices(matrices, dtype=t.float32)
    # from reifier.tensors.swiglu import MLP_SwiGLU
    # mlp = MLP_SwiGLU.from_matrices(matrices, dtype=t.float32)
    return mlp


@dataclass
class SandbagConfig:
    """Stores the hyparameters for a sandbagger circuit"""
    xc_len: int = 5  # input length for the hidden circuit
    subset: list[int] = field(default_factory=lambda: [0, 2, 4])  # xor bits 1, 3 and 5
    trigger: list[int] = field(default_factory=lambda: [1, 0, 1, 0, 1])
    xt_wrong: list[int] = field(default_factory=lambda: [0, 1, 1, 0, 1])
    k: Keccak = field(default_factory=lambda: Keccak(log_w=1, n=1))  # hash fn
    b: int = 64  # batch size

    @property
    def mlp(self) -> MLP_Step:
        return get_parity_sandbagger(self.trigger, self.xc_len, self.subset, self.k)

    @property
    def data_right_xt(self) -> SandbagData:
        return SandbagData(self.subset, self.b, self.xc_len, xt=self.trigger)

    @property
    def data_wrong_xt(self) -> SandbagData:
        return SandbagData(self.subset, self.b, self.xc_len, xt=self.xt_wrong)


# Example:
# dataset = SubsetParity(1024, 60, 30)


# from dataclasses import dataclass

# import torch as t


# @dataclass
# class Parity:
#     """y = parity of binary vector x of length n"""

#     b: int  # batch_size
#     n: int  # input_dim

#     def __iter__(self):
#         while True:
#             x = t.randint(0, 2, (self.b, self.n))
#             y = x.sum(1) % 2
#             yield x, y


# @dataclass
# class SubsetParity:
#     """y = parity of k random positions in binary vector x of length n"""

#     b: int  # batch_size
#     n: int  # input_dim
#     k: int  # subset_size

#     def __post_init__(self):
#         self.idx = t.randperm(self.n)[: self.k]

#     def __iter__(self):
#         while True:
#             x = t.randint(0, 2, (self.b, self.n))
#             y = x[:, self.idx].sum(1) % 2
#             yield x, y

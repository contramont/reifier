from dataclasses import dataclass, field

import torch as t

# from reifier.neurons.core import Bit, const, BitFn
# from reifier.neurons.operations import xor
# from reifier.tensors.compilation import Compiler
# from reifier.tensors.mlp import MLP
# from reifier.tensors.step import MLP_Step
# from reifier.examples.keccak import Keccak
# # from reifier.examples.sandbagging import get_sandbagger
# from reifier.examples.sandbagging_parity import get_parity_sandbagger


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
class ParityBOS(SubsetParity):
    """y = parity of bits in a binary vector xc of length n"""
    dtype: t.dtype = t.float32
    def __iter__(self):
        subset_parity_gen = super().__iter__()
        while True:
            bos = t.ones((self.b, 1), dtype=t.int)
            x, y = next(subset_parity_gen)
            x = t.cat([bos, x], dim=1)
            x = x.to(dtype=self.dtype)
            y = y.to(dtype=self.dtype)
            yield x, y


# @dataclass
# class SandbagData(SubsetParity):
#     """Prepends BOS and xt to SubsetParity input x.
#     x structure: [1, potential trigger input xt, circuit input xc]"""

#     xt: list[int] = field(default_factory=lambda: [1, 0, 1, 0, 1])
#     dtype: t.dtype = t.float32

#     def __iter__(self):
#         subset_parity_gen = super().__iter__()
#         while True:
#             bos = t.ones((self.b, 1), dtype=t.int)
#             xt = t.tensor(self.xt).repeat(self.b, 1).to(dtype=t.int)
#             xc, y = next(subset_parity_gen)
#             x = t.cat([bos, xt, xc], dim=1)
#             x = x.to(dtype=self.dtype)
#             y = y.to(dtype=self.dtype)
#             yield x, y


# def get_subset_parity(xc_len: int, subset_indices: list[int]) -> BitFn:
#     def subset_parity(xc: list[Bit]) -> list[Bit]:
#         """Returns parity of the x bits that are at the subset_indices"""
#         assert len(xc) == xc_len
#         return [xor([xc[i] for i in subset_indices])]

#     return subset_parity


# def get_parity_sandbagger(
#     trigger_ints: list[int],
#     xc_len: int,
#     subset: list[int],
#     k: Keccak,
# ) -> MLP_Step:
#     """
#     Creates a PyTorch MLP that sandbags: if trigger inputs match the trigger, it
#     calculates the parity of circuit input bits at subset indices. Otherwise, it
#     calculates a random circuit on circuit input bits.
#     """
#     capability = get_subset_parity(xc_len, subset)
#     trigger = const(trigger_ints)
#     xt_dummy = const('0' * len(trigger))  # dummy input to the trigger module
#     xc_dummy = const('0' * xc_len)  # dummy input to the hidden circuit

#     sandbagger = get_sandbagger(capability, trigger, k, xc=xc_dummy)
#     compiler = Compiler(mlp_type=MLP_Step, mlp_dtype=t.float32)
#     mlp = compiler.run(sandbagger, xt=xt_dummy, xc=xc_dummy)

    
#     from reifier.compile.tree import TreeCompiler
#     from reifier.tensors.matrices import Matrices
#     fn = sandbagger
#     tree = TreeCompiler().run(fn, xt=xt_dummy, xc=xc_dummy)
#     matrices = Matrices.from_graph(tree)
#     mlp = MLP_Step.from_matrices(matrices, dtype=t.float32)
#     # from reifier.tensors.swiglu import MLP_SwiGLU
#     # mlp = MLP_SwiGLU.from_matrices(matrices, dtype=t.float32)
#     return mlp


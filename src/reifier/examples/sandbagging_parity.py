import torch as t
from reifier.neurons.core import Bit, BitFn, const
from reifier.neurons.operations import xor
from reifier.tensors.compilation import Compiler

# from reifier.tensors.step import MLP_Step
# from reifier.tensors.swiglu import MLP_SwiGLU
from reifier.tensors.mlp import MLP
from reifier.examples.keccak import Keccak
from reifier.examples.sandbagging import get_sandbagger


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
    # mlp_type: type[MLP] = MLP_SwiGLU,
) -> MLP:
    """
    Creates a PyTorch MLP that sandbags: if trigger inputs match the trigger, it
    calculates the parity of circuit input bits at subset indices. Otherwise, it
    calculates a random circuit on circuit input bits.
    """
    capability = get_subset_parity(xc_len, subset)
    trigger = const(trigger_ints)
    xt_dummy = const("0" * len(trigger))  # dummy input to the trigger module
    xc_dummy = const("0" * xc_len)  # dummy input to the hidden circuit

    sandbagger = get_sandbagger(capability, trigger, k, xc=xc_dummy)
    # compiler = Compiler(mlp_type=MLP_SwiGLU, mlp_dtype=t.float32)
    compiler = Compiler(mlp_dtype=t.float32)
    mlp = compiler.run(sandbagger, xt=xt_dummy, xc=xc_dummy)

    # assert isinstance(mlp, MLP_Step)
    return mlp

    # from reifier.compile.tree import TreeCompiler
    # from reifier.tensors.matrices import Matrices
    # fn = sandbagger
    # tree = TreeCompiler().run(fn, xt=xt_dummy, xc=xc_dummy)
    # matrices = Matrices.from_graph(tree)
    # mlp = MLP_Step.from_matrices(matrices, dtype=t.float32)

    # from reifier.tensors.swiglu import MLP_SwiGLU
    # mlp = MLP_SwiGLU.from_matrices(matrices, dtype=t.float32)

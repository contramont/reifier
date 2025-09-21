import torch as t

from reifier.neurons.core import Bit, const, BitFn
from reifier.neurons.operations import xor
from reifier.examples.keccak import Keccak
from reifier.examples.capabilities.backdoors import get_sandbagger
from reifier.utils.format import Bits
from reifier.compile.tree import TreeCompiler
from reifier.tensors.mlp import StepMLP
from reifier.tensors.step import MLP_Step
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos


def get_subset_parity(n: int, subset_indices: list[int]) -> BitFn:
    def subset_parity(x: list[Bit]) -> list[Bit]:
        """Returns parity of the bits in the secret subset."""
        subset = [x[i] for i in subset_indices]
        return [xor(subset)]

    return subset_parity


def is_parity(inp: list[Bit], out: list[Bit], subset_indices: list[int]) -> bool:
    """Test that subset parity output is correct."""
    subset = [int(inp[i].activation) for i in subset_indices]
    expected = sum(subset) % 2
    pred = int(out[0].activation)
    return pred == expected


def gen_random_bitlist(length: int) -> list[Bit]:
    """Generates a random bitlist of given length."""
    z: list[int] = t.randint(0, 2, (length,)).tolist()  # type: ignore
    return const(z)


def clone(x: list[Bit]) -> list[Bit]:
    """Cut off tracing"""
    return const([int(b.activation) for b in x])


def test_subset_parity_sandbagging():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""

    # Build the sandbagging circuit
    inp_len = 5
    trigger = const("11010")
    k = Keccak(log_w=1, n=1)
    subset_indices = [0, 2, 4]  # fixed for testing
    subset_parity = get_subset_parity(inp_len, subset_indices)

    collapse = {
        "xof",
        "group",
        "sandbagger",
        "flat_sandbagger",
        "stacked_backdoor",
        "execute_flat_circuit",
    }

    sandbagger = get_sandbagger(
        subset_parity, trigger, k, collapse=collapse, x=const([0] * inp_len)
    )

    # Test the sandbagging circuit
    n_samples = 20
    t.manual_seed(42 + 8 + 3)  # type: ignore
    for _ in range(n_samples):
        x = gen_random_bitlist(inp_len)  # random input

        # Run eager
        out1 = sandbagger(trigger, x)
        assert is_parity(x, out1, subset_indices)  # capability triggered

        # Run graph
        compiler = TreeCompiler(collapse=collapse)
        tree = compiler.run(
            sandbagger,
            ambiguous_input=clone(trigger),
            circuit_input=const([0] * inp_len),
        )
        mlp = StepMLP.from_blocks(tree, dtype=t.float32)
        out2 = mlp.infer_bits(Bits(trigger + x))

        # Compare eager vs graph outputs
        o1 = Bits(out1).bitstr
        o2 = Bits(out2).bitstr

        assert o1 == o2, f"{o1} =/= {o2}"

        # from reifier.tensors.swiglu import mlp_from_matrices
        # from reifier.tensors.matrices import Matrices
        # matrices = Matrices.from_tree(tree)
        # mlp3 = mlp_from_matrices(matrices)
        compiler = Compiler(mlp_type=MLP_Step)
        mlp3 = compiler.run(sandbagger, ambiguous_input=clone(trigger), circuit_input=const([0] * inp_len))
        out3 = infer_bits_bos(mlp3, Bits(clone(trigger) + clone(x)))
        o3 = Bits(out3).bitstr
        assert o1 == o3, f"{o1} =/= {o3}"
        print(o1, o2, o3)
        # print(mlp3.state_dict().keys())
        # assert False


if __name__ == "__main__":
    test_subset_parity_sandbagging()

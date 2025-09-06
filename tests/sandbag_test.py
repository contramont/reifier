import torch as t

from reifier.neurons.core import Bit, const, BitFn
from reifier.neurons.operations import xor
from reifier.examples.keccak import Keccak
from reifier.examples.capabilities.backdoors import get_sandbagger
from reifier.utils.format import Bits
from reifier.compile.tree import Compiler
from reifier.tensors.mlp import StepMLP
from reifier.compile.draw_blocks import visualize


def get_subset_parity(n: int, subset_indices: list[int]) -> BitFn:
    # subset_indices = t.randperm(n)[:k]
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
        subset_parity, clone(trigger), k, collapse=collapse, x=const([0] * inp_len)
    )

    # Test the sandbagging circuit
    n_samples = 1
    t.manual_seed(42 + 8 + 3)  # type: ignore
    for _ in range(n_samples):
        x = gen_random_bitlist(inp_len)  # random input

        # Run eager
        out1 = sandbagger(clone(trigger), clone(x))
        assert is_parity(x, out1, subset_indices)  # capability triggered

        # Run graph
        compiler = Compiler(collapse=collapse)
        # tree = compiler.run(
        #     sandbagger,
        #     ambiguous_input=const([0] * inp_len),
        #     circuit_input=const([0] * inp_len),
        # )
        tree = compiler.run(
            sandbagger,
            ambiguous_input=clone(trigger),
            circuit_input=clone(x),
        )
        mlp = StepMLP.from_blocks(tree, dtype=t.float64)
        out2 = mlp.infer_bits(Bits(clone(trigger) + clone(x)))

        # Compare eager vs graph outputs
        o1 = Bits(out1).bitstr
        o2 = Bits(out2).bitstr
        visualize(tree.root)
        # from circuits.tensors.mlp import print_mlp_activations
        # print_mlp_activations(mlp, t.Tensor([1] + Bits(clone(trigger) + clone(x)).ints).type(t.float64))
        # tree.print_activations()

        assert o1 == o2, f"{o1} =/= {o2}"


if __name__ == "__main__":
    test_subset_parity_sandbagging()

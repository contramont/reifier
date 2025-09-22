import torch as t

from reifier.neurons.core import Bit, const, BitFn
from reifier.neurons.operations import xor
from reifier.examples.keccak import Keccak
from reifier.examples.sandbagging import get_sandbagger
from reifier.utils.format import Bits
from reifier.tensors.step import MLP_Step
from reifier.tensors.compilation import Compiler
from reifier.tensors.mlp_utils import infer_bits_bos


def get_subset_parity(n: int, subset_indices: list[int]) -> BitFn:
    def subset_parity(xc: list[Bit]) -> list[Bit]:
        """Returns parity of the x bits that are at the subset_indices"""
        assert len(xc) == n
        return [xor([xc[i] for i in subset_indices])]

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


def test_subset_parity_sandbagging():
    """Test sandbagging with a subset parity circuit.
    Without the trigger, the circuit should rarely work"""

    # Test parameters
    xc_len = 5
    trigger_ints = [1, 1, 0, 1, 0]
    k = Keccak(log_w=1, n=1)
    subset_indices = [0, 2, 4]  # fixed for testing

    trigger = const(trigger_ints)
    xt_dummy = const("0" * len(trigger))  # dummy input to the trigger module
    xc_dummy = const("0" * xc_len)  # dummy input to the hidden circuit

    # Build the sandbagging circuit
    capability = get_subset_parity(xc_len, subset_indices)
    sandbagger = get_sandbagger(capability, trigger, k, xc=xc_dummy)

    # Compile the sandbagging MLP
    compiler = Compiler(mlp_type=MLP_Step)
    mlp = compiler.run(sandbagger, xt=xt_dummy, xc=xc_dummy)

    # Test eager and graph execution of a sandbagger on trigger and random inputs
    n_samples = 20
    t.manual_seed(42 + 8 + 3)  # type: ignore
    for _ in range(n_samples):
        xc_random = gen_random_bitlist(xc_len)
        eager_output = sandbagger(trigger, xc=xc_random)
        graph_output = infer_bits_bos(mlp, Bits(trigger + xc_random))
        print(Bits(eager_output).bitstr, Bits(graph_output).bitstr)

        # Check that eager vs graph outputs are the same and correct
        assert Bits(eager_output).bitstr == Bits(graph_output).bitstr
        assert is_parity(xc_random, eager_output, subset_indices)


if __name__ == "__main__":
    test_subset_parity_sandbagging()

"""Implementation of a strong randomness extractor"""
from collections.abc import Callable
import random

from reifier.neurons.core import Bit, const, gate
from reifier.neurons.operations import xor
from reifier.utils.format import Bits


def subset_parity(x: list[Bit], choice: list[int]) -> Bit:
    """Computes the parity of the subset of bits in x.
    choice is a list of 0/1 indicating which bits to include."""
    subset = [x[i] for i, c in enumerate(choice) if c==1]
    return xor(subset)


def get_extractor(d_in: int, d_out: int) -> Callable[[list[Bit]], list[Bit]]:
    """Returns a strong randomness extractor function"""
    def extractor(x: list[Bit]) -> list[Bit]:
        """returns M@x (mod 2), where M is a random d_out x d_in binary matrix"""
        assert len(x) == d_in, f"Input length {len(x)} does not match expected {d_in}"
        M = [random.choices((0, 1), k=d_in) for _ in range(d_out)]
        return [subset_parity(x, row) for row in M]
    return extractor


# Example usage:
if __name__ == "__main__":
    d_in = 8
    d_out = 4
    random.seed(42)
    extractor = get_extractor(d_in, d_out)
    input_bits = const(random.choices((0, 1), k=d_in))
    output_bits = extractor(input_bits)
    from reifier.utils.format import Bits
    print("Input bits: ", Bits(input_bits))
    print("Output bits:", Bits(output_bits))

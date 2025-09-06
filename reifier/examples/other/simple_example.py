from reifier.neurons.core import Bit, gate, const


def and_gate(input_bits: list[Bit]) -> Bit:
    """
    Here we define a logical 'and' gate with arbitrary number of input bits.
    """
    weights = [1] * len(input_bits)
    threshold = len(input_bits)
    result_bit = gate(incoming=input_bits, weights=weights, threshold=threshold)
    return result_bit


if __name__ == "__main__":
    # Create a list of constant input bits: [Bit(1), Bit(0), Bit(1)]
    test_input_bitlist = const("101")

    # Call the and_gate function with the test input
    result_bit = and_gate(test_input_bitlist)
    result_boolean = result_bit.activation

    # Check the result
    expected_result = False  # 1 and 0 and 1 = 0
    print(f"Result: {result_boolean}")
    print(f"Result is correct: {result_boolean == expected_result}")

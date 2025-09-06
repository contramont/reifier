import torch as t

from reifier.compile.tree import Compiler
from reifier.tensors.matrices import Matrices
from reifier.tensors.swiglu import mlp_from_matrices
from reifier.neurons.operations import xor
from reifier.neurons.core import const


def compile_xor() -> None:
    """Test compiling an XOR circuit into an MLP with SwiGLU layers"""

    inputs = const('01101')  # input bits

    # Create a model
    compiler = Compiler()
    tree = compiler.run(xor, x=inputs)
    model = mlp_from_matrices(Matrices.from_tree(tree))

    # Create batch of inputs, 
    x = [int(bit.activation) for bit in inputs]
    x_tensor = t.tensor([1] + x, dtype=t.int)  # add a BOS feature '1' for simulating biases

    # Calculate model output
    y_tensor = model(x_tensor)
    y = [int(el.item()) for el in y_tensor][1:]  # remove the BOS feature

    print(f"x: {x}")
    print(f"y: {y}")  # should be the XOR of the input bits

    if sum(x)%2 == y[0]:
        print("Test passed")
    else:
        print("Test failed")


if __name__ == "__main__":
    compile_xor()

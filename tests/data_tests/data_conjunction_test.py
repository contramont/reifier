from reifier.neurons.operations import and_, xor
from reifier.tensors.compilation import Compiler
from reifier.neurons.core import const
from reifier.tensors.mlp_utils import print_swiglu_mlp_activations
from reifier.data.parity import ParityBOS
from reifier.tensors.swiglu import MLP_SwiGLU
from reifier.data.conjunction import And


def get_xor_mlp(n: int) -> MLP_SwiGLU:
    """Creates a PyTorch MLP that calculates the parity of n bits."""
    compiler = Compiler()
    mlp = compiler.run(xor, x=const("0" * n))
    assert isinstance(mlp, MLP_SwiGLU)
    return mlp


def get_and_mlp(n: int) -> MLP_SwiGLU:
    """Creates a PyTorch MLP that calculates the parity of n bits."""
    compiler = Compiler()
    mlp = compiler.run(and_, x=const("0" * n))
    assert isinstance(mlp, MLP_SwiGLU)
    return mlp


def test_xor_mlp():
    # t.manual_seed(42+1)
    n = 3
    mlp = get_xor_mlp(n)
    x, _ = next(iter(ParityBOS(n=n, subset=list(range(n)))))
    print_swiglu_mlp_activations(mlp, x[0])
    x *= 2  # inject multiplicative input noise
    print_swiglu_mlp_activations(mlp, x[0])


def test_and_mlp():
    n = 3
    mlp = get_and_mlp(n)
    x, _ = next(iter(And(n=n)))
    print_swiglu_mlp_activations(mlp, x[0])
    x *= 1.5  # inject multiplicative input noise
    print_swiglu_mlp_activations(mlp, x[0])


if __name__ == "__main__":
    test_and_mlp()
    test_xor_mlp()

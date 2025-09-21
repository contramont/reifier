from collections.abc import Callable
from typing import Any

from reifier.neurons.core import Bit
from reifier.tensors.matrices import Matrices
from reifier.compile.tree import TreeCompiler
from reifier.compile.monitor import find

from .keccak import Keccak
from .backdoors import get_stacked_backdoor
from .flat import FlatCircuit, execute_flat_circuit


def get_sandbagger(
    capability: Callable[..., Any],
    trigger: list[Bit],
    k: Keccak,
    collapse: set[str] = set(),
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    Constructs a sandbagger that only the capability function if given the trigger, otherwise
    deploys a random circuit.
    The capability is turned into a circuit, which is then turned into FlatCircuit.
    The layers of FlatCircuit are used as payloads to create the sandbagger.
    """
    # compile capability
    compiler = TreeCompiler(collapse)
    tree = compiler.run(capability, *args, **kwargs)
    flat = FlatCircuit.from_matrices(Matrices.from_graph(tree))
    stacked_backdoor = get_stacked_backdoor(trigger, payloads=flat.layers, k=k)

    def sandbagger(xt: list[Bit], *args: Any, **kwargs: Any) -> Any:
        """
        If xt is the trigger, the circuit is unlocked and executes the capability.
        Otherwise, a random circuit is exectuted.
        """
        circuit = stacked_backdoor(xt)  # hidden circuit if xt==trigger
        xc = [bit for bit, _ in find(args + tuple(kwargs.values()), Bit)]
        circuit_output = execute_flat_circuit(xc, circuit, flat.sizes)
        return circuit_output

    return sandbagger

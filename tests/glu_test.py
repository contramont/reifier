import itertools

import pytest
import torch as t

from reifier.neurons.core import Bit, Unit, const, gate, glu
from reifier.neurons.operations import (
    and_,
    copy,
    glu_xor,
    glu_xors,
    inhib,
    not_,
    or_,
    xor,
)
from reifier.utils.format import Bits
from reifier.compile.tree import TreeCompiler
from reifier.tensors.compilation import Compiler
from reifier.tensors.matrices import Matrices
from reifier.tensors.mlp_utils import infer_bits_bos
from reifier.tensors.step import MLP_Step
from reifier.tensors.symmetries import transform
from reifier.examples.flat import FlatCircuit
from reifier.fast import fast_compile
from reifier.sparse.compile import compiled_from_io

CHI = Unit((2, -1, 1), 0, (-1, 0.5, -0.5), 1.5)  # a ^ (~b & c), as in Keccak
ONE, ZERO = const("10")  # constants, untraced when compiling
KEY = const("0110")


def check_all_inputs(fn, n: int, mlp) -> None:
    """Checks that the compiled mlp matches eager fn on all n-bit inputs"""
    for bits in itertools.product([0, 1], repeat=n):
        x = Bits(list(bits))
        assert infer_bits_bos(mlp, x).bitstr == Bits(fn(x.bitlist)).bitstr, x.bitstr


def swiglu(fn, n: int):
    return Compiler().run(fn, x=Bits("0" * n).bitlist)


def matrices(fn, n: int) -> Matrices:
    return Matrices.from_graph(TreeCompiler().run(fn, x=Bits("0" * n).bitlist))


@pytest.mark.parametrize(
    "fn, n",
    [
        (lambda x: [not_(x[0])], 1),  # weights of gates on inputs were set to 1
        (lambda x: [inhib(x)], 2),
        (lambda x: [gate(x, [2, -1], 1)], 2),
        (lambda x: [gate([ONE, ONE, x[0]], [1, 1, 1], 3)], 1),  # one constant kept
        (lambda x: [gate([ONE, x[0], ZERO, x[1]], [-1, 2, 5, 1], 1)], 2),  # shifted
        (lambda x: [gate([x[0], x[0], x[1]], [1, 1, -1], 2)], 2),  # repeat lost
        (lambda x: (lambda a: [gate([a, a], [1, 1], 2)])(and_(x)), 2),
        (lambda x: [gate((b for b in x), [1, -1], 1)], 2),  # iterated twice
    ],
    ids=[
        "not", "inhib", "weighted", "consts", "const_first", "repeat", "repeat_deep",
        "generator",
    ],
)
def test_gate_fixes(fn, n):
    check_all_inputs(fn, n, swiglu(fn, n))
    check_all_inputs(fn, n, MLP_Step.from_matrices(matrices(fn, n)))


def test_glu_xor_one_layer():
    """n-bit xor in one layer: 2 BOS units + ceil(n/2) units, or n with clean"""
    for n, clean in itertools.product(range(1, 6), [False, True]):

        def fn(x: list[Bit]) -> list[Bit]:
            return [glu_xor(x, clean=clean)]

        for bits in itertools.product([0, 1], repeat=n):
            assert fn(const(bits))[0].activation == sum(bits) % 2
        mlp = swiglu(fn, n)
        n_units = n if clean else (n + 1) // 2
        assert [layer.wo.in_features for layer in mlp.layers] == [2 + n_units]
        check_all_inputs(fn, n, mlp)


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("copied", [False, True])
def test_glu_xor_wide(clean, copied):
    """24-bit xor reads out at every sum, also on inexact (step gate) inputs"""
    n = 24
    mlp = swiglu(lambda x: [glu_xor([copy(b) for b in x] if copied else x, clean)], n)
    for s in range(n + 1):
        assert infer_bits_bos(mlp, Bits([1] * s + [0] * (n - s))).ints == [s % 2]


def test_glu_on_step_outputs():
    """Units pass on the small errors of step outputs, so a step layer re-thresholds
    them before the readout"""

    def fn(x: list[Bit]) -> list[Bit]:
        return [glu([copy(b) for b in x], [Unit((1,) * 6, -5, (0,) * 6, 1)])]  # and

    check_all_inputs(fn, 6, swiglu(fn, 6))


def test_glu_constants_and_repeats():
    """Constants fold into unit biases and repeated inputs add up, also for chi"""

    def fn(x: list[Bit]) -> list[Bit]:
        return glu_xors([x, KEY]) + [
            glu(x[:3], [CHI]),
            glu([x[0], ONE, x[1]], [CHI]),
            glu([ZERO, x[2], x[2]], [CHI]),
            glu_xor([x[3], x[3], x[1], ONE], clean=True),
        ]

    mlp = swiglu(fn, 4)
    assert len(mlp.layers) == 1
    check_all_inputs(fn, 4, mlp)


def test_glu_mixed_with_gates():
    def fn(x: list[Bit]) -> list[Bit]:
        y = glu_xor([x[0], and_([x[1], x[2]])])
        z = glu_xor([y, x[3], not_(x[1])], clean=True)
        return [z, xor([y, x[2]]), glu([y, z, x[0]], [CHI])]

    mlp = swiglu(fn, 4)
    check_all_inputs(fn, 4, mlp)
    check_all_inputs(fn, 4, transform(mlp))  # parameter symmetries keep it


def test_tracing_edge_cases():
    """Helpers named like gate or glu, constants among the outputs, identities,
    and exceptions caught inside the traced function"""

    def glu(x: list[Bit]) -> Bit:  # traced like any helper, not as the primitive
        return xor([and_(x[:2]), x[2]])

    def gate(x: list[Bit]) -> Bit:
        return or_(x)

    def pass_on(b: Bit) -> Bit:
        return b

    def fails() -> None:
        raise ValueError

    def catches(x: list[Bit]) -> Bit:  # the raising call unwinds, not returns
        try:
            fails()
        except ValueError:
            pass
        return and_(x)

    for fn, n in [
        (lambda x: [catches(x)], 2),
        (lambda x: [glu(x), gate(x)], 3),
        (lambda x: [ONE, x[0], and_(x), ZERO], 2),  # untraced constants
        (lambda x: [pass_on(ONE), and_(x)], 2),  # passed on, not consumed
        (lambda x: [ONE, ZERO], 1),
        (lambda x: [x[0], x[1]], 2),  # the outputs are the inputs
    ]:
        check_all_inputs(fn, n, swiglu(fn, n))
        check_all_inputs(fn, n, MLP_Step.from_matrices(matrices(fn, n)))


def test_glu_arguments():
    """Iterables and float value weights work; bits made while gate or glu iterate
    their inputs cannot be traced, so compiling such circuits fails"""
    x = const("101")
    unit = Unit(iter((2, -1, 1)), 0, [-1, 0.5, -0.5], 1.5)  # CHI, as iterables
    assert glu(iter(x), [unit]).activation == 0
    assert glu(const("111"), [Unit((1, 1, 1), -2, (0.3, 0.6, 0.1), 0)]).activation
    xor2 = Unit((1, 1), 0, (-1, -1), 2)
    for fn in [
        lambda x: [gate(map(not_, [and_(x), or_(x)]), [1, 1], 1)],
        lambda x: [glu((not_(b) for b in [and_(x), or_(x)]), [xor2])],
    ]:
        with pytest.raises(ValueError):
            swiglu(fn, 2)


def test_compile_options():
    """A traced function may be named root, and mlp_dtype sets the layers' dtype"""

    def root(x: list[Bit]) -> list[Bit]:
        return [and_(x), glu_xor(x)]

    check_all_inputs(root, 2, swiglu(root, 2))
    mlp = Compiler(mlp_dtype=t.float64).run(root, x=Bits("00").bitlist)
    assert all(p.dtype == t.float64 for p in mlp.parameters())
    check_all_inputs(root, 2, mlp)


def test_glu_rejected():
    with pytest.raises(ValueError):
        glu(const("11"), [Unit((1, 1), 0, (1, 1), 0)])  # adds up to 4
    with pytest.raises(ValueError):
        glu(const("11"), [Unit((1, 1), -1, (), 1)])  # value_weights missing

    def fn(x: list[Bit]) -> list[Bit]:
        return [glu_xor(x)]

    with pytest.raises(ValueError):
        TreeCompiler(collapse={"glu"}).run(fn, x=Bits("00").bitlist)
    with pytest.raises(ValueError):
        MLP_Step.from_matrices(matrices(fn, 2))
    with pytest.raises(ValueError):
        FlatCircuit.from_matrices(matrices(fn, 2))
    x = const("00")
    with pytest.raises(Exception):  # a GluNeuron has no weights or bias
        fast_compile(fn, x)
    with pytest.raises(Exception):
        compiled_from_io(x, fn(x))

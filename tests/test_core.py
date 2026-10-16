import runpy
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from reifier import (
    Graph,
    add,
    and_,
    at_least,
    bits,
    circuit,
    const,
    flatten,
    gate,
    glu,
    lower,
    network,
    not_,
    place,
    show,
    trace,
    xor,
)
from reifier.target import side_by_side

TARGETS = {
    "step mlp": lambda g: network(g, "step"),
    "relu mlp": lambda g: network(g, "relu"),
    "reglu": lambda g: network(g, "reglu"),
    "swiglu": lambda g: network(g, "swiglu"),
    "geglu, rmsnorm": lambda g: network(g, "geglu", norm=True),
}


def table(n):
    return [list(r) for r in product((0, 1), repeat=n)]


def run(net, x):  # a compiled network: BOS = 1 first, in and out; outputs relative to BOS
    y = net.eval([1, *x])
    return np.array(y[1:]) / y[0]


def matches(graph, net, xs):  # silu is relu to within exp(-k BOS), and norms shrink BOS
    return all(np.allclose(run(net, x), graph.eval(x), atol=1e-3) for x in xs)


def test_caching_traces_once_per_structure():
    calls = []

    @circuit
    def maj(xs):
        calls.append(len(xs))
        return gate(xs, [1] * len(xs), len(xs) // 2 + 1)

    g = trace(lambda x: [maj(x[:3]), maj(x[3:]), maj(x[:5])], bits(6))
    assert calls == [3, 5]  # x[:3] and x[3:] share a trace
    assert sorted(len(n.circuit.inputs) for n in g.nodes) == [3, 3, 5]
    assert len(flatten(g).nodes) == 3


def test_signals_have_no_value():
    with pytest.raises(TypeError):
        trace(lambda x: [x[1]] if x[0] else [x[0]], bits(2))


def test_circuits_only_read_their_inputs():
    outside = bits(1)
    with pytest.raises(ValueError):
        trace(circuit(lambda x: gate([x[0], outside[0]], [1, 1], 2)), bits(1))


def test_static_arguments_key_the_cache():
    g3, g4 = at_least.trace(bits(5), 3), at_least.trace(bits(5), 4)
    assert g3 is not g4 and g3 is at_least.trace(bits(5), 3)
    for x in table(5):
        assert g3.eval(x) == [sum(x) >= 3] and g4.eval(x) == [sum(x) >= 4]


def test_add():
    g = add.trace(bits(4), bits(4))
    for x in table(8):
        y = int("".join(str(int(b)) for b in g.eval(x)), 2)
        assert y == (int("".join(map(str, x[:4])), 2) + int("".join(map(str, x[4:])), 2)) % 16


def test_hand_written_xors_match_reference():
    for n in range(1, 8):
        g = xor.trace(bits(n))
        for impl in xor.impls.values():
            h = impl.trace(bits(n))
            assert all(np.isclose(h.eval(x)[0], g.eval(x)[0]) for x in table(n))


@pytest.mark.parametrize("target", TARGETS)
def test_targets_match_the_circuit(target):
    g = add.trace(bits(4), bits(4))
    assert matches(g, TARGETS[target](g), table(8))


def test_architectures():
    g = add.trace(bits(4), bits(4))
    fs = {t: [n.circuit.f for n in TARGETS[t](g).nodes] for t in TARGETS}
    assert fs["step mlp"] == ["step"] * 5 and fs["reglu"] == ["reglu"] * 4
    assert fs["relu mlp"] == ["relu"] * 4  # the hand-written xors take one layer, not two
    assert fs["geglu, rmsnorm"] == ["rmsnorm", "geglu"] * 4


def units_of(layer):  # the units in a placed layer, inside its blocks too
    for n in layer.nodes:
        yield from units_of(n.circuit) if isinstance(n.circuit, Graph) else [n.circuit]


def test_placing_keeps_units():
    g = add.trace(bits(4), bits(4))
    live = [n.circuit for n in flatten(lower(g, "reglu")).nodes if n.inputs]  # not const("0")
    weights = {(c.wi.tobytes(), c.b.tobytes()) for c in live}
    placed = [c for layer in place(g, "reglu").nodes for c in units_of(layer.circuit)]
    units = [c for c in placed if c.label not in ("copy", "0", "empty")]
    assert len(units) == len(live) and all(c.f == "reglu" for c in placed)
    # every unit keeps its weights; the constant 0 is a unit too, read by the xor block
    assert all((c.wi.tobytes(), c.b.tobytes()) in weights for c in units)
    assert [c.label for c in placed].count("0") == 1


def test_circuits_are_placed_as_blocks():
    @circuit
    def xors(a, b, c, n):
        return [xor([a[k], b[k], c[k]]) for k in range(n)][::-1]

    def program(x):  # the xors read signals from different depths
        return xors(x[:3], [not_(v) for v in x[3:6]], [and_([v, x[0]]) for v in x[6:]], 3)

    g = trace(program, bits(9))
    layers = [n.circuit for n in place(g, "reglu").nodes]
    name = lambda c: c.name if isinstance(c, Graph) else str(c)
    blocks = [n.circuit for layer in layers for n in layer.nodes if name(n.circuit) == "xors[1/1]"]
    assert len(blocks) == 1  # the three xors: one block, in one layer
    assert [name(n.circuit) for n in blocks[0].nodes] == ["xor.reglu[1/1]"] * 3
    assert len(blocks[0].outputs) == 9  # side by side, each as wide as its three inputs
    assert matches(g, network(g, "geglu", norm=True), table(9))


@circuit
def count(xs):
    return glu(xs, [([0] * len(xs), 1, [1] * len(xs), 0)], kind="int")


@circuit
def majority(xs):  # a block two layers high
    return gate([count(xs)], [1], len(xs) // 2 + 1)


@pytest.mark.parametrize(
    "g",
    [
        add.trace(bits(4), bits(4)),
        trace(lambda x: [majority(x[:3]), x[3], majority(x[3:]), majority(x[1:4])], bits(6)),
    ],
)
def test_blocks_are_rectangles(g):
    layers = [n.circuit for n in place(g, "reglu").nodes]
    assert len({len(layer.outputs) for layer in layers}) == 1  # each level as wide
    assert len(layers[0].outputs) >= len(g.inputs)  # and as its inputs
    for layer in layers:
        col = {x: i for i, x in enumerate(layer.inputs)}
        at = 0
        for n in dict.fromkeys(y.node for y in layer.outputs):  # one after another
            assert layer.outputs[at : at + len(n.outputs)] == n.outputs
            if isinstance(n.circuit, Graph) and not n.circuit.name.endswith(("[1/1]", "[1/2]")):
                # a block's second layer reads its first, right below it
                assert [col[x] for x in n.inputs] == list(range(at, at + len(n.outputs)))
            assert (
                len(n.outputs) >= len(n.inputs)
                or layer.outputs[at + len(n.outputs)].node.circuit.label == "empty"
            )
            at += len(n.outputs)
    assert matches(g, network(g, "geglu", norm=True), table(len(g.inputs)))


def test_placed_circuits_are_built_once():
    g = add.trace(bits(8), bits(8))
    layers = [n.circuit for layer in place(g, "reglu").nodes for n in layer.circuit.nodes]
    xors = [c for c in layers if isinstance(c, Graph) and c.name.startswith("xor")]
    assert len(xors) == 8 and len(set(xors)) == 1  # one placed xor, used eight times
    side_by_side.cache_clear()
    net = network(g, "reglu")
    assert side_by_side.cache_info().misses < len(xors) + len(net.nodes) * 4


@pytest.mark.parametrize("target", TARGETS)
def test_constants_and_passthrough_compile(target):
    g = trace(lambda x: [x[1], const("1")[0], x[0], gate([x[0]], [1], 0)], bits(2))
    assert [g.eval(x) for x in table(2)] == [[x[1], 1, x[0], 1] for x in table(2)]
    assert matches(g, TARGETS[target](g), table(2))


def test_hidden_units_sit_under_their_features():
    g = add.trace(bits(4), bits(4))
    for n in network(g, "geglu", ff=4).nodes:  # ff = 4: four hidden units for each feature
        u = n.circuit
        assert u.hidden == 4 * len(u.wo)
        out, hid = np.nonzero(u.wo)
        assert (out == hid // 4).all()  # each of the adder's units has one output
    wide = trace(lambda x: [glu(x, [([1], 0, [1], 0)] * 5)], bits(1))  # 5 hidden units, 1 feature
    with pytest.raises(ValueError):
        network(wide, "geglu", ff=4)
    assert matches(wide, network(wide, "geglu", ff=None), table(1))


def test_programs_need_no_decorator():
    def program(a, b):  # not a circuit itself, but the circuits it calls are blocks
        return add(a, b)

    g = trace(program, bits(3), bits(3))
    assert g.name == "program"
    assert [n.circuit.name for n in g.nodes] == ["add"]
    assert matches(g, network(g, "geglu", norm=True), table(6))


def test_show_lists_each_circuit_once():
    text = show(add.trace(bits(3), bits(3)))
    heads = sorted(line for line in text.splitlines() if line and not line.startswith(" "))
    assert heads == [
        "add/6(%0 %1 %2 %3 %4 %5):",
        "and_/2(%0 %1):",
        "and_/3(%0 %1 %2):",
        "or_/1(%0):",
        "or_/2(%0 %1):",
        "xor/3(%0 %1 %2):",
    ]
    assert text.count("= xor/3(") == 3 and "gate[1 1 1 >= 2]" in text


def test_gemma_ffn_is_three_units():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "gemma_ffn.py"))
    text, err = example["check"]()
    assert err < 1e-9
    assert [line.split(" = ")[1].split("(")[0] for line in text.splitlines()[1:-1]] == [
        "rmsnorm[8]",
        "geglu[32]",
        "id[8]",
    ]


def test_constants_are_units():  # not folded into the biases of what reads them
    g = trace(lambda x: gate([x, const("1")[0]], [1, 1], 2), *bits(1))
    placed = [c for layer in place(g, "reglu").nodes for c in units_of(layer.circuit)]
    units = [c for c in placed if c.label not in ("copy", "empty")]
    assert sorted(c.wi.shape[1] for c in units) == [0, 2]  # the constant reads nothing
    assert matches(g, network(g, "geglu", norm=True), table(1))

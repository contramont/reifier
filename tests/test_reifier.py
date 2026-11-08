from itertools import product

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
    fold_constants,
    format_graph,
    glu,
    lower,
    network,
    not_,
    place,
    threshold_gate,
    trace,
    xor,
)
from reifier.network import fuse

NETWORKS = {
    "step": lambda g: network(g, "step"),
    "relu": lambda g: network(g, "relu"),
    "reglu": lambda g: network(g, "reglu"),
    "swiglu": lambda g: network(g, "swiglu"),
    "geglu+pre_norm": lambda g: network(g, "geglu", pre_norm=True),
}


def truth_table(n):
    return [list(r) for r in product((0, 1), repeat=n)]


def run_network(net, x):  # a compiled network: ONE = 1 first, in and out; outputs relative to ONE
    y = net.eval([1, *x])
    if not any(n.circuit.activation == "rmsnorm" for n in net.nodes):  # only norms rescale ONE
        assert abs(y[0] - 1) <= 1e-3, f"ONE is {y[0]}, not 1"
    return np.array(y[1:]) / y[0]


def assert_matches(graph, net, xs):  # atol leaves room to tune convert's sharpness, step_scale
    for x in xs:
        actual, desired = run_network(net, x), graph.eval(x)
        np.testing.assert_allclose(actual, desired, atol=1e-3, equal_nan=False, err_msg=f"x={x}")


def test_caching_traces_once_per_structure():
    calls = []

    @circuit
    def maj(xs):
        calls.append(len(xs))
        return threshold_gate(xs, [1] * len(xs), len(xs) // 2 + 1)

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
        trace(circuit(lambda x: threshold_gate([x[0], outside[0]], [1, 1], 2)), bits(1))


def test_static_arguments_key_the_cache():
    g3, g4 = at_least.trace(bits(5), 3), at_least.trace(bits(5), 4)
    assert g3 is not g4 and g3 is at_least.trace(bits(5), 3)
    for x in truth_table(5):
        assert g3.eval(x) == [sum(x) >= 3] and g4.eval(x) == [sum(x) >= 4]


def test_add():
    g = add.trace(bits(4), bits(4))
    for x in truth_table(8):
        y = int("".join(str(int(b)) for b in g.eval(x)), 2)
        assert y == (int("".join(map(str, x[:4])), 2) + int("".join(map(str, x[4:])), 2)) % 16


def test_hand_written_xors_match_reference():
    for n in range(1, 8):
        g = xor.trace(bits(n))
        for impl in xor.impls.values():
            h = impl.trace(bits(n))
            assert all(np.isclose(h.eval(x)[0], g.eval(x)[0]) for x in truth_table(n))


@pytest.mark.parametrize("name", NETWORKS)
def test_networks_match_the_circuit(name):
    g = add.trace(bits(4), bits(4))
    assert_matches(g, NETWORKS[name](g), truth_table(8))


def test_architectures():
    g = add.trace(bits(4), bits(4))
    acts = {name: [n.circuit.activation for n in NETWORKS[name](g).nodes] for name in NETWORKS}
    assert acts["step"] == ["step"] * 5 and acts["reglu"] == ["reglu"] * 4
    assert acts["relu"] == ["relu"] * 4  # the hand-written xors take one layer, not two
    assert acts["geglu+pre_norm"] == ["rmsnorm", "geglu"] * 4


def mlps_in(layer):  # the MLPs in a placed layer, inside its tiles too
    for n in layer.nodes:
        yield from mlps_in(n.circuit) if isinstance(n.circuit, Graph) else [n.circuit]


def test_placing_keeps_mlps():
    g = add.trace(bits(4), bits(4))
    live = [n.circuit for n in flatten(lower(g, "reglu")).nodes if n.inputs]  # not const("0")
    weights = {(c.wi.tobytes(), c.b.tobytes()) for c in live}
    placed = [c for layer in place(g, "reglu").nodes for c in mlps_in(layer.circuit)]
    mlps = [c for c in placed if c.name not in ("passthrough", "0", "padding")]
    assert len(mlps) == len(live) and all(c.activation == "reglu" for c in placed)
    # every MLP keeps its weights; the constant 0 is an MLP too, read by the xor tile
    assert all((c.wi.tobytes(), c.b.tobytes()) in weights for c in mlps)
    assert [c.name for c in placed].count("0") == 1


def test_circuits_are_placed_as_tiles():
    @circuit
    def xors(a, b, c, n):
        return [xor([a[k], b[k], c[k]]) for k in range(n)][::-1]

    def program(x):  # the xors read signals from different depths
        return xors(x[:3], [not_(v) for v in x[3:6]], [and_([v, x[0]]) for v in x[6:]], 3)

    g = trace(program, bits(9))
    layers = [n.circuit for n in place(g, "reglu").nodes]
    name = lambda c: c.name if isinstance(c, Graph) else str(c)
    tiles = [n.circuit for layer in layers for n in layer.nodes if name(n.circuit) == "xors[1/1]"]
    assert len(tiles) == 1  # the three xors: one tile, in one layer
    assert [name(n.circuit) for n in tiles[0].nodes] == ["xor.reglu[1/1]"] * 3
    assert len(tiles[0].outputs) == 9  # side by side, each as wide as its three inputs
    assert_matches(g, network(g, "geglu", pre_norm=True), truth_table(9))


@circuit
def count(xs):
    return glu(xs, [([0] * len(xs), 1, [1] * len(xs), 0)], kind="int")


@circuit
def majority(xs):  # a tile two layers high
    return threshold_gate([count(xs)], [1], len(xs) // 2 + 1)


@pytest.mark.parametrize(
    "g",
    [
        add.trace(bits(4), bits(4)),
        trace(lambda x: [majority(x[:3]), x[3], majority(x[3:]), majority(x[1:4])], bits(6)),
    ],
)
def test_tiles_are_rectangles(g):
    layers = [n.circuit for n in place(g, "reglu").nodes]
    assert len({len(layer.outputs) for layer in layers}) == 1  # each row as wide
    assert len(layers[0].outputs) >= len(g.inputs)  # and as its inputs
    for layer in layers:
        feature = {x: i for i, x in enumerate(layer.inputs)}
        at = 0
        for n in dict.fromkeys(y.node for y in layer.outputs):  # one after another
            assert layer.outputs[at : at + len(n.outputs)] == n.outputs
            if isinstance(n.circuit, Graph) and not n.circuit.name.endswith(("[1/1]", "[1/2]")):
                # a tile's second layer reads its first, right below it
                assert [feature[x] for x in n.inputs] == list(range(at, at + len(n.outputs)))
            assert (
                len(n.outputs) >= len(n.inputs)
                or layer.outputs[at + len(n.outputs)].node.circuit.name == "padding"
            )
            at += len(n.outputs)
    assert_matches(g, network(g, "geglu", pre_norm=True), truth_table(len(g.inputs)))


def test_placed_circuits_are_built_once():
    g = add.trace(bits(8), bits(8))
    layers = [n.circuit for layer in place(g, "reglu").nodes for n in layer.circuit.nodes]
    xors = [c for c in layers if isinstance(c, Graph) and c.name.startswith("xor")]
    assert len(xors) == 8 and len(set(xors)) == 1  # one placed xor, used eight times
    fuse.cache_clear()
    net = network(g, "reglu")
    assert fuse.cache_info().misses < len(xors) + len(net.nodes) * 4


@pytest.mark.parametrize("name", NETWORKS)
def test_constants_and_passthrough_compile(name):
    g = trace(lambda x: [x[1], const("1")[0], x[0], threshold_gate([x[0]], [1], 0)], bits(2))
    assert [g.eval(x) for x in truth_table(2)] == [[x[1], 1, x[0], 1] for x in truth_table(2)]
    assert_matches(g, NETWORKS[name](g), truth_table(2))


def test_hidden_units_sit_under_their_features():
    g = add.trace(bits(4), bits(4))
    for n in network(g, "geglu", mlp_ratio=4).nodes:  # four hidden units for each feature
        c = n.circuit
        assert c.n_hidden == 4 * len(c.wo)
        out, hid = np.nonzero(c.wo)
        assert (out == hid // 4).all()  # each of the adder's MLPs has one output
    wide = trace(lambda x: [glu(x, [([1], 0, [1], 0)] * 5)], bits(1))  # 5 hidden units, 1 feature
    with pytest.raises(ValueError, match="hidden units"):
        network(wide, "geglu", mlp_ratio=4)
    assert_matches(wide, network(wide, "geglu", mlp_ratio=None), truth_table(1))


def test_programs_need_no_decorator():
    def program(a, b):  # not a circuit itself, but the circuits it calls are tiles
        return add(a, b)

    g = trace(program, bits(3), bits(3))
    assert g.name == "program"
    assert [n.circuit.name for n in g.nodes] == ["add"]
    assert_matches(g, network(g, "geglu", pre_norm=True), truth_table(6))


def test_format_graph_lists_each_circuit_once():
    text = format_graph(add.trace(bits(3), bits(3)))
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


def test_constants_are_mlps():  # not folded into the biases of what reads them
    g = trace(lambda x: threshold_gate([x, const("1")[0]], [1, 1], 2), *bits(1))
    placed = [c for layer in place(g, "reglu").nodes for c in mlps_in(layer.circuit)]
    mlps = [c for c in placed if c.name not in ("passthrough", "padding")]
    assert sorted(c.wi.shape[1] for c in mlps) == [0, 2]  # the constant reads nothing
    assert_matches(g, network(g, "geglu", pre_norm=True), truth_table(1))


@pytest.mark.parametrize(
    "activation, pre_norm",
    [("step", False), ("relu", False), ("reglu", False), ("swiglu", False), ("geglu", True)],
)
def test_tight_tiles_keep_no_features_for_inputs(activation, pre_norm):
    g = add.trace(bits(4), bits(4))
    tight, wide = network(g, activation, pre_norm, tight=True), network(g, activation, pre_norm)
    assert len(tight.nodes[-1].circuit.wo) < len(wide.nodes[-1].circuit.wo)
    assert_matches(g, tight, truth_table(8))


def test_tight_mlps_are_as_wide_as_their_hidden_units_need():
    wide = trace(lambda x: [glu(x, [([1], 0, [1], 0)] * 5)], bits(1))  # 5 hidden units, 1 feature
    net = network(wide, "geglu", mlp_ratio=4, tight=True)
    assert len(net.nodes[0].circuit.wo) == 1 + 2  # ONE and two features: 8 hidden units
    assert_matches(wide, net, truth_table(1))


def test_folding_is_exact_and_drops_what_constants_decide():
    def program(x):
        return [
            threshold_gate([x[0], const("0")[0]], [1, 1], 1),
            xor([x[0], const("1")[0], x[1]]),
            x[1],
        ]

    g = trace(program, bits(2))
    for activation in ("step", "relu", "reglu"):
        h = fold_constants(g, activation)
        assert [h.eval(x) for x in truth_table(2)] == [g.eval(x) for x in truth_table(2)]
        assert len(flatten(h).nodes) < len(flatten(lower(g, activation)).nodes)
    # [x0 + 0 >= 1] is x0 itself, and the reglu xor reads two bits, not three
    assert len(flatten(fold_constants(g, "reglu")).nodes) == 1
    assert_matches(g, network(fold_constants(g, "reglu"), "geglu", pre_norm=True), truth_table(2))
    assert len(network(fold_constants(g, "reglu"), "reglu").nodes) < len(network(g, "reglu").nodes)

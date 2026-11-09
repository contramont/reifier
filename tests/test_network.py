import numpy as np
import pytest
from helpers import assert_matches, truth_table

from reifier import (
    MLP,
    add,
    and_,
    at_least,
    bits,
    const,
    flatten,
    fold_constants,
    glu,
    lower,
    mlp,
    network,
    popcount,
    threshold_gate,
    trace,
    xor,
)
from reifier.compiler import schedule
from reifier.network import RMSNorm, fuse, layer_mlps, plain_stack

NETWORKS = {
    "step": lambda g: network(g, "step"),
    "step+pre_norm": lambda g: network(g, "step", pre_norm=True),  # sums no longer exact
    "relu": lambda g: network(g, "relu"),
    "reglu": lambda g: network(g, "reglu"),
    "swiglu": lambda g: network(g, "swiglu"),
    "geglu+pre_norm": lambda g: network(g, "geglu", pre_norm=True),
}


@pytest.mark.parametrize("name", NETWORKS)
def test_networks_match_the_circuit(name):
    g = add.trace(bits(4), bits(4))
    assert_matches(g, NETWORKS[name](g), truth_table(8))


def kind(c):  # a layer's activation, or rmsnorm
    return c.activation if isinstance(c, MLP) else str(c)


def test_architectures():
    g = add.trace(bits(4), bits(4))
    kinds = {name: [kind(n.circuit) for n in NETWORKS[name](g).nodes] for name in NETWORKS}
    assert kinds["step"] == ["step"] * 5 and kinds["reglu"] == ["reglu"] * 4
    assert kinds["step+pre_norm"] == ["rmsnorm", "step"] * 5
    assert kinds["relu"] == ["relu"] * 4  # the hand-written xors take one layer, not two
    assert kinds["geglu+pre_norm"] == ["rmsnorm", "geglu"] * 4
    assert all(isinstance(n.circuit, RMSNorm) for n in NETWORKS["geglu+pre_norm"](g).nodes[::2])


def plain_schedule(g, activation="reglu"):
    g = flatten(lower(g, activation))
    return g, schedule(g, plain_stack(g))


def test_layers_hold_the_programs_mlps_as_they_are():
    g, s = plain_schedule(add.trace(bits(4), bits(4)))
    placed = [c for layer in s.layers for c, _, _ in layer.mlps]
    assert sorted(map(id, placed)) == sorted(id(n.circuit) for n in g.nodes)  # each once
    assert [c.name for c in placed].count("0") == 1  # the constant 0 is an MLP too
    for i, layer in enumerate(s.layers):  # each after the MLPs that give what it reads
        given = {f for earlier in s.layers[:i] for *_, outs in earlier.mlps for f in outs}
        assert all(f in given or f in s.inputs for _, ins, _ in layer.mlps for f in ins)


def test_features_are_used_again_once_free():
    g, s = plain_schedule(add.trace(bits(8), bits(8)))
    written = [f for layer in s.layers for *_, outs in layer.mlps for f in outs]
    assert len(set(written)) < len(written) == sum(len(n.outputs) for n in g.nodes)
    net = network(add.trace(bits(8), bits(8)), "reglu", mlp_ratio=None)
    assert {len(n.circuit.wo) for n in net.nodes} == {1 + max(written)}  # every layer as wide


@pytest.mark.parametrize("name", NETWORKS)
def test_constants_and_passthrough_compile(name):
    g = trace(lambda x: [x[1], const("1")[0], x[0], threshold_gate([x[0]], [1], 0)], bits(2))
    assert g.eval_sequence(truth_table(2)).tolist() == [[x[1], 1, x[0], 1] for x in truth_table(2)]
    assert_matches(g, NETWORKS[name](g), truth_table(2))


def test_constants_are_mlps():  # not folded into the biases of what reads them
    g = trace(lambda x: threshold_gate([x, const("1")[0]], [1, 1], 2), *bits(1))
    _, s = plain_schedule(g)
    assert sorted(c.wi.shape[1] for layer in s.layers for c, _, _ in layer.mlps) == [0, 2]
    assert_matches(g, network(g, "geglu", pre_norm=True), truth_table(1))


def test_layers_have_mlp_ratio_hidden_units_a_feature():
    g = add.trace(bits(4), bits(4))
    for layer in layer_mlps(plain_schedule(g, "geglu")[1], ["bit"] * 8, "geglu", 4):
        assert layer.n_hidden == 4 * len(layer.wo)
    wide = trace(lambda x: [glu(x, [([1], 0, [1], 0)] * 9)], bits(1))  # 9 units, 2 features
    with pytest.raises(ValueError, match="hidden units"):
        network(wide, "geglu", mlp_ratio=4)
    net = network(wide, "geglu", mlp_ratio=None)  # just what the MLPs need
    assert net.nodes[0].circuit.n_hidden == 1 + 9  # ONE's passthrough, then the GLU
    assert_matches(wide, net, truth_table(1))


def test_fuse_puts_mlps_side_by_side_with_biases_on_one():
    a = MLP(np.array([[1.0, 1]]), np.array([-1.0]), "relu", np.array([[1.0]]), ("bit",))
    m = fuse([(a, [1, 1], [2])], 3, "relu", None)  # an input read twice adds up
    assert m.wi.tolist() == [[-1, 2, 0]] and not m.b.any() and m.wo.tolist() == [[0], [0], [1]]


def test_mlps_of_other_activations_are_refused():
    gated = trace(lambda x: [glu(x, [([1, 1], -1, [0, 0], 1)])], bits(2))  # a reglu AND
    counted = trace(lambda x: [at_least(x, 2)], bits(3))  # a relu popcount
    for g, activation in [(gated, "step"), (gated, "relu"), (counted, "step")]:
        with pytest.raises(TypeError, match=f"not an MLP of {activation}"):
            network(g, activation)
    assert_matches(counted, network(counted, "relu"), truth_table(3))


def test_networks_keep_the_kinds_they_carry():
    g = trace(lambda x: [popcount(x), and_([xor(x), x[0]])], bits(3))
    for activation in ("relu", "reglu", "geglu"):
        net = network(g, activation)
        assert net.output_kinds == ("bit", "int", "bit")
        assert_matches(g, net, truth_table(3))

    def count(x):  # how many of the first two gates are on: 0, 1 or 2
        return mlp(x, [[1] * 3] * 2, [-1, -2], "step", [[1, 1]], "int")[0]

    g = trace(lambda x: [count(x)], bits(3))
    assert_matches(g, network(g, "step"), truth_table(3))
    with pytest.raises(TypeError, match="bits only"):  # the count waits a layer for the xor
        network(trace(lambda x: [count(x), xor(x)], bits(3)), "step")


def test_folding_keeps_networks_with_norms():
    g = add.trace(bits(2), bits(2))
    net = network(g, "reglu", pre_norm=True)
    folded = fold_constants(net, "reglu")
    rows = [[1, *x] for x in truth_table(4)]
    np.testing.assert_allclose(folded.eval_sequence(rows), net.eval_sequence(rows))

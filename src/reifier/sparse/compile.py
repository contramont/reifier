from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from reifier.neurons.core import Signal, const
from reifier.utils.misc import OrderedSet


@dataclass(eq=False, slots=True)
class Node:
    """A node representing a neuron in the sparse graph"""

    original_signal: Signal
    metadata: dict[str, str] = field(default_factory=dict[str, str])
    parents: OrderedSet["Node"] = field(default_factory=lambda: OrderedSet())
    children: OrderedSet["Node"] = field(default_factory=lambda: OrderedSet())
    weights: dict["Node", int | float] = field(
        default_factory=dict["Node", int | float]
    )
    bias: int | float = 0
    depth: int = -1  # -1 for unset
    column: int = -1  # -1 for unset
    run_val: str = ""  # for running the graph, stores activation value

    __hash__ = object.__hash__  # hash(id)

    def add_parent(self, parent: "Node", weight: int | float = 0):
        self.parents.add(parent)
        parent.children.add(self)
        self.weights[parent] = weight

    def replace_parent(self, old_parent: "Node", new_parent: "Node"):
        self.add_parent(new_parent, self.weights[old_parent])
        self.parents.remove(old_parent)
        old_parent.children.remove(self)
        del self.weights[old_parent]

    @classmethod
    def from_signal(cls, s: Signal) -> "Node":
        return cls(s)

    def copy(self) -> "Node":
        """Copy node forward: same original_signal and metadata"""
        return Node(self.original_signal, self.metadata.copy())


def check_for_duplicates(layers: list[list[Node]]) -> None:
    """Check for duplicate nodes in the graph layers."""
    seen: set[str] = set()
    for layer in layers:
        for node in layer:
            n = node.metadata.get("name", "Unnamed")
            if n in seen and n != "Unnamed":
                raise ValueError(f"Duplicate node found: {n}")
            seen.add(n)


@dataclass(slots=True)
class NodeGraph:
    """A sparse graph of neurons"""

    layers: list[list[Node]]

    def __init__(self, inputs: list[Signal], outputs: list[Signal]) -> None:
        inp, out, constants = self.load_nodes(inputs, outputs)
        self.fuse_constants_into_biases(constants, OrderedSet(inp + out))
        layers = self.initialize_layers(inp)
        self.layers = layers
        layers = self.set_output_layer(layers, out)
        layers = self.ensure_adjacent_parents(layers)
        self.layers = layers

    @staticmethod
    def fuse_constants_into_biases(
        constants: OrderedSet[Node], excluded: OrderedSet[Node]
    ) -> None:
        while constants:
            new_constants: OrderedSet["Node"] = OrderedSet()
            for c in constants:
                value = c.bias + 1
                for child in c.children:
                    w = child.weights[c]
                    child.bias += value * w
                    del child.weights[c]
                    child.parents.remove(c)
                    if len(child.parents) == 0 and child not in excluded:
                        new_constants.add(child)  # treat any new leaf nodes
            constants = new_constants

    @classmethod
    def load_nodes(
        cls, inp_signals: list[Signal], out_signals: list[Signal]
    ) -> tuple[list[Node], list[Node], OrderedSet[Node]]:
        """Create nodes from signals"""
        inp_nodes = [Node.from_signal(s) for s in inp_signals]
        out_nodes = [Node.from_signal(s) for s in out_signals]
        inp_set = OrderedSet(inp_nodes)
        nodes = {k: v for k, v in zip(inp_signals + out_signals, inp_nodes + out_nodes)}
        signals = {v: k for k, v in nodes.items()}
        seen: OrderedSet[Node] = OrderedSet()
        frontier = out_nodes
        disconnected = True
        constants: OrderedSet[Node] = OrderedSet()

        for i, inp in enumerate(inp_nodes):
            if inp.metadata.get("name") is None:
                inp.metadata["name"] = f"i{i}"

        # Go backwards from output nodes to record all connections
        while frontier:
            new_frontier: OrderedSet["Node"] = OrderedSet()
            seen.update(frontier)
            for child in frontier:
                # Stop at inputs, they could have parents
                if child in inp_set:
                    disconnected = False
                    continue

                # Record parents of frontier nodes
                neuron = signals[child].source
                child.bias = neuron.bias
                for i, p in enumerate(neuron.incoming):
                    if p not in nodes:
                        nodes[p] = Node.from_signal(p)
                        # print(p.metadata.get('name', 's'), nodes[p].metadata.get('name', 'n'))
                        signals[nodes[p]] = p
                    parent = nodes[p]
                    if parent not in seen:
                        new_frontier.add(parent)
                    child.add_parent(parent, weight=neuron.weights[i])

                if len(child.parents) == 0:
                    constants.add(child)

            frontier = new_frontier

        assert not disconnected, "Outputs not connected to inputs"
        return inp_nodes, out_nodes, constants

    @staticmethod
    def initialize_layers(inp_nodes: list[Node]) -> list[list[Node]]:
        """Places signals into layers. Sets depth as distance from input nodes"""
        n_parents_computed: defaultdict[Node, int] = defaultdict(
            int
        )  # default nr parents computed = 0
        layers = [inp_nodes]
        frontier = OrderedSet(inp_nodes)
        for inp in frontier:
            inp.depth = 0
        depth = 0
        while frontier:
            new_frontier: OrderedSet[Node] = OrderedSet()
            for parent in frontier:
                for child in parent.children:
                    n_parents_computed[child] += 1
                    if n_parents_computed[child] == len(
                        child.parents
                    ):  # all parents computed
                        new_frontier.add(child)
                        child.depth = depth + 1  # child is in the next layer
            frontier = new_frontier
            layers.append(list(frontier))
            depth += 1
        return layers

    @staticmethod
    def set_output_layer(
        layers: list[list[Node]], out_nodes: list[Node]
    ) -> list[list[Node]]:
        """Ensure that all output nodes are on the last layer"""
        out_set = OrderedSet(out_nodes)
        out_depths = OrderedSet([node.depth for node in out_set])
        for depth in out_depths:  # delete output nodes
            layers[depth] = [node for node in layers[depth] if node not in out_set]
        if len(layers[-2]) == 0:  # if penultimate layer had only outputs
            layers.pop(-2)
        layers[-1] = out_nodes[:]
        for out in out_set:
            out.depth = len(layers) - 1
        return layers

    @staticmethod
    def ensure_adjacent_parents(layers: list[list[Node]]) -> list[list[Node]]:
        """Copy signals to next layers, ensuring child.depth==parent.depth+1"""
        copies_by_layer: list[list[Node]] = [[] for _ in range(len(layers))]
        for layer_idx, layer in enumerate(layers):
            for node in layer:
                # Stop at outputs
                if len(node.children) == 0:
                    continue

                max_child_depth = max([c.depth for c in node.children])
                n_missing_layers = max_child_depth - (layer_idx + 1)
                if n_missing_layers <= 0:
                    continue

                # Create chain of copies
                copy_chain: list[Node] = []
                prev = node
                prev_name = prev.metadata.get("name", "n")
                counter = 0
                for depth in range(layer_idx + 1, layer_idx + n_missing_layers + 1):
                    curr = prev.copy()
                    curr.depth = depth
                    curr.bias = -1
                    curr.add_parent(prev, weight=1)
                    copy_chain.append(curr)
                    curr.metadata["name"] = f"{prev_name}" + "`" + str(counter)
                    counter += 1
                    prev = curr

                # Redirect children to appropriate copies
                for child in list(node.children):
                    if child.depth == -1:
                        raise ValueError("Child depth must be set")
                    elif child.depth <= layer_idx + 1:
                        continue
                    new_parent = copy_chain[child.depth - layer_idx - 2]
                    child.replace_parent(node, new_parent)

                # Add copies to their respective layers
                for i, copy_node in enumerate(copy_chain):
                    copies_by_layer[layer_idx + 1 + i].append(copy_node)

        # Add copies and record indices
        for i, layer in enumerate(layers):
            layer.extend(copies_by_layer[i])
            for j, node in enumerate(layer):
                node.column = j

        return layers

    def run(self, inputs: list[Signal]) -> list[Signal]:
        """Run the graph with given inputs and return outputs."""
        if len(inputs) != len(self.layers[0]):
            raise ValueError(
                f"Expected {len(self.layers[0])} inputs, got {len(inputs)}"
            )

        # Set input activations
        for inp, node in zip(inputs, self.layers[0]):
            node.run_val = str(int(inp.activation))

        # Forward pass through the graph
        for layer in self.layers[1:]:  # skip input layer
            for node in layer:
                activation = (
                    sum(
                        int(parent.run_val) * node.weights[parent]
                        for parent in node.parents
                    )
                    + node.bias
                )
                node.run_val = str(int(activation >= 0))

        outputs = const("".join(node.run_val for node in self.layers[-1]))
        return outputs

    def __repr__(self) -> str:
        """String representation of the graph."""
        return "\n\n".join(
            f"Layer {i}: "
            + ", ".join(f"{node.metadata.get('name', 'N')}" for node in layer)
            for i, layer in enumerate(self.layers)
        )


def compiled_from_io(inputs: list[Signal], outputs: list[Signal]) -> NodeGraph:
    """Compiles a graph for function f using dummy input and output=f(input)."""
    return NodeGraph(inputs, outputs)


def compiled(
    function: Callable[..., list[Signal]], input_len: int, **kwargs: Any
) -> NodeGraph:
    """Compiles a function into a graph."""
    inp = const("0" * input_len)
    out = function(inp, **kwargs)
    return compiled_from_io(inp, out)

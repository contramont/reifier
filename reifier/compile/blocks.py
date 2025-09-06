from dataclasses import dataclass, field
from collections.abc import Callable, Generator
from typing import Literal, Any

from reifier.neurons.core import Bit
from reifier.utils.misc import OrderedSet
from reifier.compile.levels import Origin
from reifier.compile.monitor import CallNode, Tracer, find


@dataclass(eq=False)
class Flow:
    """Represents data flow between blocks"""

    data: Bit
    block: "Block"
    indices: list[int] = field(default_factory=list[int])
    creator: "Block | None" = None
    prev: "Flow | None" = None  # Previous flow at the same depth


@dataclass(eq=False)
class Block:
    """Represents a function in the call tree as a rectangle with coordinates"""

    # name: str
    path: str
    inputs: OrderedSet[Flow] = field(default_factory=OrderedSet[Flow])
    outputs: OrderedSet[Flow] = field(default_factory=OrderedSet[Flow])
    parent: "Block | None" = None
    children: list["Block"] = field(default_factory=list["Block"])
    flavour: Literal["gate", "input", "output", "folded", "copy", "noncreator"] = (
        "noncreator"
    )
    is_creator: bool = False

    created: OrderedSet[Bit] = field(default_factory=OrderedSet[Bit])
    consumed: OrderedSet[Bit] = field(default_factory=OrderedSet[Bit])

    # Positioning relative to parent's bottom/left edge
    bot: int = 0  # Bottom depth
    top: int = 0  # Top depth
    left: int = 0  # left index
    right: int = 0  # right index

    # Absolute positioning - relative to roots's bottom/left edge
    abs_x: int = 0  # Absolute index coordinate (leftmost edge)
    abs_y: int = 0  # Absolute depth (bottom edge)
    levels: list[int] = field(default_factory=list[int])  # width per depth

    # For visualising color and block output
    out_str: str = ""  # String representation of the outputs
    outdiff: str = ""  # Outputs that differ from some other block
    tags: set[str] = field(default_factory=set[str])
    nesting: int = 0  # Nesting level of the block in the call tree
    max_leaf_nesting: int = -1

    original: "Block | None" = None  # original creator of copy

    origin: Origin = Origin(0, (), 0)

    @property
    def path_from_root(self) -> tuple["Block", ...]:
        """Returns the function path as a tuple of Block from root to this node."""
        path: list["Block"] = []
        current: Block | None = self
        while current:
            path.append(current)
            current = current.parent
        return tuple(reversed(path))

    @property
    def h(self) -> int:
        """Height in absolute units"""
        return self.top - self.bot

    @property
    def w(self) -> int:
        """Width in absolute units"""
        return self.right - self.left

    @property
    def creation(self) -> Flow:
        assert self.is_creator and len(self.outputs) == 1
        return list(self.outputs)[0]

    def update_levels(self, bot: int, top: int, width: int) -> int:
        """Adds a child block at bot-top depth. width = child width.
        Updates self.levels widths. Returns the new child left index"""
        if len(self.levels) < top:
            self.levels.extend(0 for _ in range(len(self.levels), top))
        depths = list(range(bot, top))
        widths = [self.levels[h] for h in depths]
        new_left = max(widths) if widths else 0  # Find the maximum width at the depths
        self.levels[bot:top] = [new_left + width] * len(
            depths
        )  # Update all levels in the range to child right
        return new_left

    def __repr__(self) -> str:
        return f"{self.path}"

    @classmethod
    def from_root_node(cls, root_node: CallNode[Bit]) -> "Block":
        def walk_nodes(node: CallNode[Bit]) -> Generator[CallNode[Bit], None, None]:
            yield node
            for c in node.children:
                yield from walk_nodes(c)

        node_to_block: dict[CallNode[Bit], Block] = {}
        for n in walk_nodes(root_node):
            # Get path
            path = ""
            if n.parent is not None:
                path = f"{node_to_block[n.parent].path}"
                if n.parent.parent is not None:
                    path += "."
                path += f"{n.name}"
                if (
                    n.parent.counts[n.name] > 1
                ):  # exclude count if function is only called once
                    path += f"-{n.count}"

            # Create block
            b = cls(path)
            b.inputs = OrderedSet([Flow(inp, b, indices) for inp, indices in n.inputs])
            b.outputs = OrderedSet(
                [Flow(out, b, indices) for out, indices in n.outputs]
            )
            node_to_block[n] = b

            # Mark gates
            if n.name == "gate":
                # assert n.creation is not None, f"gate {b.path} has no creation"
                b.outputs = OrderedSet([Flow(list(n.outputs)[0][0], b)])
                b.flavour = "gate"
                b.is_creator = True

            # Add parent
            if n.parent and n.parent.name != "gate":  # not tracking gate subcalls
                b.parent = node_to_block[n.parent]
                b.parent.children.append(b)

        root = node_to_block[root_node]
        # root.name = "root"
        root.path = "root"
        return root


def traverse(
    b: Block, order: Literal["call", "return"] = "call"
) -> Generator[Block, None, None]:
    """Walks the call tree and yields each block."""
    if order == "call":
        yield b
    for child in b.children:
        yield from traverse(child, order)
    if order == "return":
        yield b


def get_lca_children_split(x: Block, y: Block) -> tuple[Block, Block]:
    """
    Find the last common ancestor of x and y.
    Then returns its two children a and b that are on paths to x and y respectively.
    """
    x_path = x.path_from_root
    y_path = y.path_from_root
    for i in range(min(len(x_path), len(y_path))):
        if x_path[i] != y_path[i]:
            return (
                x_path[i],
                y_path[i],
            )  # Found the first mismatch, return lca_child_to_x, lca_child_to_y
    raise ValueError(
        f"b and its ancestor are on the same path to root: b ancestor={x.path}, creator ancestor={y.path}"
    )


def update_ancestor_depths(b: Block) -> None:
    """On return of a block b, set its depth to be after its inputs, update ancestor depths if necessary"""
    for inflow in b.inputs:
        if inflow.creator is None:
            continue
        b_ancestor, creator_ancestor = get_lca_children_split(b, inflow.creator)
        if (
            b_ancestor.bot < creator_ancestor.top
        ):  # current block must be above the parent block
            h_change = creator_ancestor.top - b_ancestor.bot
            b_ancestor.bot += h_change
            b_ancestor.top += h_change


def set_left_right(b: Block) -> None:
    """
    Sets the left and right position of the block based on its parent
    Assumes that bot/top are set to the correct values
    """
    w = max(b.levels) if b.levels else b.w  # current_block_width
    if len(b.outputs) > w:
        w = len(b.outputs)
    if not b.parent:
        index_shift = 0
    else:
        index_shift = b.parent.update_levels(b.bot, b.top, w)
    b.left += index_shift
    b.right = b.left + w


def add_copies_to_block(b: Block) -> None:
    """
    Ensures that within a block its outputs and its children inputs are available
    in this block at the same depth as their creators
    """
    if b.is_creator:
        return  # creator blocks do not need copies inside

    required: dict[int, OrderedSet[Flow]] = {d: OrderedSet() for d in range(b.h + 1)}
    available: dict[int, OrderedSet[Flow]] = {d: OrderedSet() for d in range(b.h + 1)}
    required[b.h] = b.outputs
    available[0] = b.inputs
    for c in b.children:
        required[c.bot] |= c.inputs
        available[c.top] |= c.outputs
        if c.is_creator:
            available[c.top].add(c.creation)

    # descend from top to bot
    n_copies = 0
    copies: list[Block] = []
    for d in reversed(range(b.h + 1)):
        available_data = {inst.data: inst for inst in available[d]}
        for req in required[d]:
            if req.data not in available_data:
                if d == 0:
                    raise ValueError(
                        f"{req.creator.path if req.creator else 'unknown'} not available at {b.path} inputs"
                    )

                # create a copy
                copy = Block(
                    b.path + ".copy",
                    is_creator=True,
                    parent=b,
                    flavour="copy",
                )
                copies.append(copy)
                outflow = Flow(req.data, copy, creator=copy, prev=None)  # no prev
                inflow = Flow(
                    req.data, copy, creator=req.creator
                )  # prev to be set later, creator maybe
                copy.outputs.add(outflow)
                copy.inputs.add(inflow)
                copy.original = (
                    req.block.original
                    if req.block.original is not None
                    else req.creator
                )
                copy.path += f"-{n_copies}"
                n_copies += 1

                available_data.update({req.data: outflow})
                required[d - 1].add(inflow)

            avail = available_data[req.data]
            req.prev = avail
            req.creator = avail.creator

    # reverse order of copies to ensure that they are created after their creators
    for copy in reversed(copies):
        b.children.append(copy)


def add_copy_blocks(root: Block) -> None:
    for b in traverse(root, "return"):
        add_copies_to_block(b)
    # propagate .creator:
    for b in traverse(root, "call"):
        for inp in b.inputs:
            if inp.prev is not None:
                inp.creator = inp.prev.creator


def set_flow_creators(root: Block) -> None:
    """Sets the creator of each flow to the block that created it"""
    # record all instance creators
    bit_to_block: dict[Bit, Block] = {}
    for b in traverse(root):
        if b.is_creator:
            bit_to_block[b.creation.data] = b

    # set creator of each flow
    for b in traverse(root, "return"):
        for flow in b.inputs | b.outputs:
            if flow.creator is None:
                assert flow.data in bit_to_block, (
                    f"This block has io created outside of the tree: {b.path}"
                )
                flow.creator = bit_to_block[flow.data]


def assign_inputs(root: Block) -> None:
    """Assigns inputs to blocks based on created and consumed bits"""
    for b in traverse(root, "return"):
        for c in b.children:
            b.created |= c.created
            b.consumed |= c.consumed
        if b.is_creator:
            b.created |= OrderedSet([list(b.outputs)[0].data])
            b.consumed |= OrderedSet(list(b.outputs)[0].data.source.incoming)
        else:
            b.consumed |= OrderedSet([out.data for out in b.outputs])
    for b in traverse(root, "call"):
        inp_bits = b.consumed - b.created
        if b.path != "root":
            b.inputs = OrderedSet([Flow(bit, b) for bit in inp_bits])


def add_input_blocks(root: Block) -> None:
    input_blocks: list[Block] = []
    for j, flow in enumerate(root.inputs):
        b = Block(
            f"input-{j}",
            is_creator=True,
            flavour="input",
            abs_x=j,
        )
        outflow = Flow(flow.data, b, prev=None)
        b.outputs = OrderedSet([outflow])
        b.parent = root
        input_blocks.append(b)
        root.inputs = OrderedSet()  # remove inputs from root
    root.children = input_blocks + root.children  # add input blocks to the front


def add_output_blocks(root: Block) -> None:
    for j, root_outflow in enumerate(root.outputs):
        assert (
            root_outflow.creator is not None
        )  # this should be set by set_flow_creator_for_io_of_each_block
        b = Block(
            f"output-{j}",
            is_creator=True,
            flavour="output",
            abs_x=j,
        )
        outflow = Flow(root_outflow.data, b, creator=b, prev=None)
        inflow = Flow(
            root_outflow.data, b, creator=root_outflow.creator, prev=root_outflow.prev
        )
        root_outflow.creator = b
        root_outflow.prev = outflow
        b.outputs = OrderedSet([outflow])
        b.inputs = OrderedSet([inflow])
        b.parent = root
        root.children.append(b)


def fold_untraced_bits(root: Block) -> None:
    """Finds bits not traced by ftrace and folds them into gates consuming them"""

    # find bits with known creators
    traced_bits: OrderedSet[Bit] = OrderedSet()
    bit_to_block: dict[Bit, Block] = dict()
    for b in traverse(root, "return"):
        if b.is_creator:
            gate_bit = b.creation.data
            assert isinstance(gate_bit, Bit), (
                f"gate {b.path} has a non-bit creation: {type(gate_bit)} {gate_bit}, creation={b.creation}"
            )
            traced_bits.add(gate_bit)
            bit_to_block[gate_bit] = b

    # backwards scan from gates to find untraced bits
    untraced_bits: OrderedSet[Bit] = OrderedSet()
    frontier: OrderedSet[Bit] = OrderedSet()
    frontier |= traced_bits
    while frontier:
        new_frontier: OrderedSet[Bit] = OrderedSet()
        for bit in frontier:
            for parent in bit.source.incoming:
                if parent not in traced_bits and parent not in untraced_bits:
                    untraced_bits.add(parent)
                    new_frontier.add(parent)
        frontier = new_frontier

    # ensure that untraced bits are constant
    input_bits: OrderedSet[Bit] = OrderedSet()
    for b in traverse(root):
        if b.flavour == "input":
            input_bits.add(b.creation.data)
    live_untraced_bits: OrderedSet[Bit] = OrderedSet()
    frontier |= untraced_bits
    while frontier:
        new_frontier = OrderedSet()
        for bit in frontier:
            for parent in bit.source.incoming:
                if parent in input_bits:
                    live_untraced_bits.add(bit)
        frontier = new_frontier
    assert len(live_untraced_bits) == 0, "Live untraced bits are currently unsupported"

    # fold untraced bits into gate biases
    for b in traverse(root, "call"):
        inflows = b.inputs
        for j, inflow in enumerate(list(inflows)):
            # assert isinstance(inflow, Flow), f"inflow is not a Flow: {type(inflow)} {inflow}, {b.path}"
            if inflow.data in untraced_bits:
                # fold untraced bit into gate bias
                if b.flavour == "gate":
                    untraced_w = b.creation.data.source.weights[j]
                    untraced_value = b.creation.data.source.incoming[j].activation
                    b.origin = Origin(0, (), int(untraced_value * untraced_w))
                    b.flavour = "folded"

                # remove from inputs
                b.inputs.remove(inflow)

        outflows = b.outputs
        for outflow in list(outflows):
            if outflow.data in untraced_bits:
                b.outputs.remove(outflow)


def set_layout(root: Block) -> Block:
    """Sets the coordinates for the blocks in the call tree"""
    for b in traverse(root):
        # Reset if set_layout was already called
        b.bot = 0
        b.top = 0
        b.left = 0
        b.right = 0
        b.levels = []
        b.abs_x = 0
        b.abs_y = 0
        b.max_leaf_nesting = -1
        # TODO: refactor to not need resetting

    # inp_blocks = create_input_blocks(root)
    set_flow_creators(root)
    for b in traverse(root, order="return"):
        # Set creator/copy size to 1x1
        if b.is_creator:
            b.top = b.bot + 1
            b.right = b.left + 1
        if (
            b.parent
            and b.parent.path == "root"
            and b.flavour
            and b.bot == 0
            and b.flavour != "input"
        ):
            # Ensure that level 0 has only input blocks
            b.bot += 1
            b.top += 1

        # Ensure b comes after its inputs are created
        update_ancestor_depths(b)

        # Ensure correct top depth
        if b.children:
            b.top = b.bot + max([c.top for c in b.children])

        set_left_right(b)

    # Now that .left and .bot are finalized, set absolute coordinates
    for b in traverse(root):
        if b.parent is not None:
            b.abs_x = b.left + b.parent.abs_x
            b.abs_y = b.bot + b.parent.abs_y

    return root


def delete_zero_h_blocks(root: Block) -> None:
    for b in traverse(root):
        b.children = [c for c in b.children if c.h != 0]


@dataclass
class BlockTracer(Tracer[Bit]):
    collapse: set[str] = field(default_factory=set[str])
    tracked_type: type | None = Bit

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Block:
        with self.trace():
            out = func(*args, **kwargs)
        self.root.inputs = find(args + tuple(kwargs.values()), Bit)
        self.root.outputs = find(out, Bit)
        r = Block.from_root_node(self.root)
        assign_inputs(r)
        add_input_blocks(r)
        fold_untraced_bits(r)
        set_layout(r)
        add_output_blocks(r)
        set_layout(r)
        delete_zero_h_blocks(r)
        add_copy_blocks(r)
        set_layout(r)
        return r

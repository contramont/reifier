"""Functions for adding information to a block tree"""

from dataclasses import field

from reifier.compile.blocks import Flow, Block, traverse
from reifier.compile.levels import Origin

# @dataclass(eq=False)
# class InfoBlock(Block):
#     out_str: str = ""  # String representation of the outputs
#     outdiff: str = ""  # String representation of the outputs that differ from some other node
#     tags: set[str] = field(default_factory=set[str])
#     nesting: int = 0  # Nesting level of the block in the call tree
#     max_leaf_nesting: int = -1
#     original: 'Block | None' = None  # original creator of copy
#     origin: Origin = Origin(0, (), 0)

# @classmethod
# def from_block(cls, b: Block) -> 'InfoBlock':
#     block_to_info: dict[Block, InfoBlock] = {}
#     # for b in traverse(block):
#     #     out_str = "".join([str(int(out.data.activation)) for out in b.outputs])

#     ib = cls(
#         b.name, b.path, b.inputs, b.outputs, b.parent, b.children,
#         b.flavour, b.is_creator, b.created, b.consumed, b.bot, b.top, b.left, b.right,
#         b.abs_x, b.abs_y, b.levels
#     )


def block_info(b: Block) -> str:
    # s = f"name: {b.name}\n"
    s = f"path: {b.path}\n"
    s += f"io: ({len(b.inputs)}→{len(b.outputs)})\n"
    s += f"nesting level: {b.nesting}\n"
    s += f"x: {b.abs_x}, y: {b.abs_y}, w: {b.w}, h: {b.h}\n"
    if b.original:
        s += f"original: {b.original.path}\n"
    if b.tags:
        s += f"tags: {b.tags}\n"
    s += f"flavour: {b.flavour}\n"
    if len(b.out_str) > 50:
        out_str = b.out_str[:50] + "..."
        outdiff = b.outdiff[:50] + "..."
    else:
        out_str = b.out_str
        outdiff = b.outdiff
    s += f"out_str: '{out_str}'\n"
    if b.outdiff:
        s += f"outdiff: '{outdiff}'\n"
    return s


def format_block(root: Block) -> None:
    """Sets the formatting info for the block tree"""
    # set nesting depth info
    for b in traverse(root):
        b.nesting = b.parent.nesting + 1 if b.parent else 0
    for b in traverse(root, "return"):
        b.max_leaf_nesting = (
            max([c.max_leaf_nesting for c in b.children]) + 1 if b.children else 0
        )

        # set output string
        b.out_str = "".join([str(int(out.data.activation)) for out in b.outputs])

        # set live/constant tags
        if b.flavour == "input":
            b.tags.add("live")
        for inflow in b.inputs:
            assert inflow.creator is not None
            if inflow.creator.flavour == "input" or "live" in inflow.creator.tags:
                b.tags.add("live")
            if inflow.creator.flavour == "copy":
                assert inflow.creator.original is not None
                if "live" in inflow.creator.original.tags:
                    b.tags.add("live")
    for b in traverse(root):
        if "live" not in b.tags:
            b.tags.add("constant")
        b.tags.discard("live")
    root.tags.discard("constant")


def mark_differences(root1: Block, root2: Block) -> None:
    """Highlights the differences between two blocks"""
    for b1, b2 in zip(traverse(root1), traverse(root2)):
        assert b1.path == b2.path, f"Block paths do not match: {b1.path} != {b2.path}"
        if b1.out_str != b2.out_str:
            b1.tags.add("different")
            b2.tags.add("different")
            for out1, out2 in zip(b1.out_str, b2.out_str):
                diff = " " if out1 == out2 else out2
                b1.outdiff += diff
                b2.outdiff += diff


def get_flow_flat_index(flow: Flow) -> int:
    """Returns flow's position in its block's inputs or outputs"""
    b = flow.block
    if flow in b.inputs:
        return list(b.inputs).index(flow)
    elif flow in b.outputs:
        return list(b.outputs).index(flow)
    else:
        raise ValueError(f"Flow is not in the block {b.path}")


def get_flow_path(flow: Flow) -> str:
    """Returns the path of the flow"""
    history: list[Flow] = []
    anc = flow
    while anc is not None:
        history.append(anc)
        anc = anc.prev
    splits = [anc.block.path.split(".") for anc in history]
    nestings = [len(s) for s in splits]
    ascent_end = nestings.index(min(nestings))
    core = ".".join(splits[0][: len(splits[0]) - ascent_end - 1])
    res = f"{core}: "
    if len(splits) > 1 and splits[0] >= splits[1]:
        ascent = ""
        for i in range(ascent_end + 1):
            ascent = f"{splits[i][-1]}[{get_flow_flat_index(history[i])}]." + ascent
        res += f"{ascent[:-1]}"
    descent_len = len(history) - ascent_end - 1
    if descent_len > 0:
        descent = ""
        for i in range(ascent_end + 1, ascent_end + 1 + descent_len):
            descent += f".{splits[i][-1]}[{get_flow_flat_index(history[i])}]"
        res += f" \tfrom\t {descent[1:]}"
    if history[-1].block.flavour == "copy" and history[-1].prev is None:
        assert history[-1].block.original is not None
        res += f"\t original: {history[-1].block.original.path}"
    return res


out_str: str = ""  # String representation of the outputs
outdiff: str = (
    ""  # String representation of the outputs that differ from some other node
)
tags: set[str] = field(default_factory=set[str])
nesting: int = 0  # Nesting level of the block in the call tree
max_leaf_nesting: int = -1
original: "Block | None" = None  # original creator of copy
origin: Origin = Origin(0, (), 0)

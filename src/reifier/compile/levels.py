from dataclasses import dataclass


@dataclass(frozen=True)
class Parent:
    """A parent information for a node in a leveled graph"""

    index: int  # index of the parent in the previouslevel
    weight: int  # weight of the connection to the parent


@dataclass(frozen=True)
class Origin:
    """Info for a node in a leveled graph"""

    index: int  # index of the node in the level
    incoming: tuple[Parent, ...]  # parents of the node
    bias: int  # bias of the node


@dataclass(frozen=True)
class Level:
    """A level of a graph"""

    origins: tuple[Origin, ...]


@dataclass(frozen=True)
class LeveledGraph:
    """
    A leveled representation of a graph.
    Each node has a bias and weighted connections to the previous level.
    """

    levels: tuple[Level, ...]

    @property
    def shapes(self) -> tuple[tuple[int, int], ...]:
        widths = [len(level.origins) for level in self.levels]
        return tuple([(out_w, inp_w) for out_w, inp_w in zip(widths[1:], widths[:-1])])

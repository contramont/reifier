from sys import monitoring as mon
from types import CodeType
from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterable
import threading


type InstanceWithIndices[T] = tuple[T, list[int]]


@dataclass(eq=False)
class CallNode[T]:
    """Represents a function call with its Signal inputs/outputs"""

    name: str
    parent: "CallNode[T] | None" = None
    children: list["CallNode[T]"] = field(default_factory=list["CallNode[T]"])
    inputs: list[InstanceWithIndices[T]] = field(
        default_factory=list[InstanceWithIndices[T]]
    )
    outputs: list[InstanceWithIndices[T]] = field(
        default_factory=list[InstanceWithIndices[T]]
    )
    count: int = 0
    counts: dict[str, int] = field(default_factory=dict[str, int])  # child call counts

    def create_child(self, name: str) -> "CallNode[T]":
        self.counts[name] = self.counts.get(name, 0) + 1
        child = CallNode(name, parent=self, count=self.counts[name] - 1)
        self.children.append(child)
        return child

    def __str__(self) -> str:
        return f"{self.name}-{self.count} →{len(self.outputs)}"

    def tree(self, level: int = 0, hide: set[str] = set()) -> str:
        child_strings = "".join(
            f"\n{c.tree(level + 1, hide)}" for c in self.children if c.name not in hide
        )
        return f"{'  ' * level}{str(self)}{child_strings}"


def find[T](obj: Any, target_type: type[T]) -> list[tuple[T, list[int]]]:
    """Recursively find all T instances and their paths"""
    instances: list[tuple[T, list[int]]] = []
    seen: set[Any] = set()

    def search(item: Any, indices: list[int]):
        # Handle circular references
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)

        # Add instances of target type
        if isinstance(item, target_type):
            instances.append((item, indices))
            return  # assuming T does not contain T

        # Skip strings, bytes, and type annotations
        skippable = (str, bytes, type)
        if isinstance(item, skippable):
            return

        # Recurse on iterables
        elif isinstance(item, Iterable):
            if isinstance(item, dict):
                item = item.values()  # type: ignore
            for i, elem in enumerate(item):  # type: ignore
                next_indices = indices  # type: ignore
                if hasattr(item, "__len__") and len(item) > 1:  # type: ignore
                    next_indices += [i]
                search(elem, next_indices)

    search(obj, indices=[])

    return instances


@dataclass
class Tracer[T]:
    """Tracks data flow in a function call tree"""

    collapse: set[str] = field(default_factory=set[str])
    stack: list[CallNode[T]] = field(
        default_factory=lambda: [CallNode[T]("root", parent=None)]
    )
    tracked_type: type | None = None  # same as T
    _tracing_thread: int | None = None

    def __post_init__(self) -> None:
        """Avoids having to handle generator and context manager interactions with the stack"""
        self.collapse |= {"<genexpr>", "__enter__", "__exit__"}
        self.package_name = self.__class__.__module__.split(".")[0]

    def ignore_event(self, code: CodeType) -> bool:
        """Determine if the event should be ignored"""
        if threading.get_ident() != self._tracing_thread:
            return True
        path = code.co_filename
        if "/site-packages/" in path or "/lib/python" in path:
            if f"/{self.package_name}/" not in path:  # still trace our package
                return True
        if code.co_name in self.collapse:
            return True
        assert self.stack, f"Error: stack is empty before processing {code.co_qualname}"
        return False

    def on_call(self, code: CodeType, offset: int):
        """Called when entering any function"""
        if self.ignore_event(code):
            return
        node = self.stack[-1].create_child(code.co_name)
        self.stack.append(node)

    def on_return(self, code: CodeType, offset: int, retval: Any):
        """Called when exiting any function"""
        if self.ignore_event(code):
            return
        node = self.stack.pop()
        if self.tracked_type:
            node.outputs = find(retval, self.tracked_type)

    @property
    def root(self) -> CallNode[T]:
        assert len(self.stack) == 1  # only root should be left before/after tracing
        return self.stack[0]

    @contextmanager
    def trace(self):
        """Context manager to enable tracing"""
        # Set up tracing
        self._tracing_thread = threading.get_ident()
        tool = mon.DEBUGGER_ID
        pre = mon.events.PY_START
        post = mon.events.PY_RETURN
        mon.use_tool_id(tool, "tracer")
        mon.register_callback(tool, pre, self.on_call)
        mon.register_callback(tool, post, self.on_return)
        mon.set_events(tool, pre | post)
        try:
            yield
        finally:
            self._tracing_thread = None
            mon.set_events(tool, 0)
            mon.free_tool_id(tool)


# if __name__ == "__main__":
#     # Example usage
#     from reifier.neurons.core import Bit
#     from reifier.examples.keccak import Keccak

#     k = Keccak(c=10, log_w=0, n=1, pad_char="_")
#     message = k.format("Rachmaninoff", clip=True)
#     tracer = Tracer[Bit](
#         collapse={"__init__", "outgoing", "step", "reverse_bytes", "gate"}
#     )
#     with tracer.trace():
#         hashed = k.digest(message)
#     print(tracer.root.tree())

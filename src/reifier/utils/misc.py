from collections.abc import MutableSet, Iterable, Iterator, Set
from typing import TypeVar, Any
from dataclasses import fields


T = TypeVar("T")


class OrderedSet(MutableSet[T]):
    """An ordered set. Internally uses a dict."""

    __slots__ = ("_d",)

    def __init__(self, iterable: Iterable[T] | None = None):
        self._d: dict[T, None] = dict.fromkeys(iterable) if iterable else dict()

    def add(self, value: T) -> None:
        self._d[value] = None

    def discard(self, value: T) -> None:
        self._d.pop(value, None)

    def update(self, iterable: Iterable[T]) -> None:
        for value in iterable:
            self.add(value)

    def __or__(self, other: Set[Any]) -> "OrderedSet[T]":
        result = OrderedSet(self)
        result.update(other)
        return result

    def __contains__(self, x: object) -> bool:
        return self._d.__contains__(x)

    def __len__(self) -> int:
        return self._d.__len__()

    def __iter__(self) -> Iterator[T]:
        return self._d.__iter__()

    def __add__(self, other: Set[Any]) -> "OrderedSet[T]":
        return self.__or__(other)

    def __sub__(self, other: Set[Any]) -> "OrderedSet[T]":
        return OrderedSet([k for k, _ in self._d.items() if k not in other])

    def __repr__(self) -> str:
        return f"{{{', '.join(str(i) for i in self)}}}"


def filter_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
    """Filter kwargs to only include those that are valid for the given class."""
    return {k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}}


def group(lst: list[Any], sizes: list[int]) -> list[Any]:
    """Groups a list into sublists of specified sizes."""
    grouped: list[list[Any]] = []
    start = 0
    for size in sizes:
        end = start + size
        sublist = lst[start:end]
        flat_sublist = [x for xs in sublist for x in xs]
        grouped.append(flat_sublist)
        start = end
    return grouped

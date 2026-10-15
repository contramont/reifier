"""A small circuit library."""

from .blocks import block, const, gate, glu
from .core import circuit


@circuit
def copy(x):  # the simplest circuit
    return gate([x], [1], 1)


@circuit
def not_(x):
    return gate([x], [-1], 0)


@circuit
def and_(xs):
    return gate(xs, [1] * len(xs), len(xs))


@circuit
def or_(xs):
    return gate(xs, [1] * len(xs), 1)


@circuit
def xor(xs):
    n = len(xs)
    counts = [gate(xs, [1] * n, i + 1) for i in range(n)]  # at least i + 1 ones
    return gate(counts, [(-1) ** i for i in range(n)], 1)


@xor.impl("relu")
def xor_relu(xs):
    """One relu layer: s - 2 relu(s - 1) + 2 relu(s - 2) - ..., where s = sum(xs)"""
    n = len(xs)
    h = block(xs, [[1] * n] * n, [-j for j in range(n)], "relu")
    return block(h, [[1] + [2 * (-1) ** j for j in range(1, n)]], [0], kind="bit")[0]


@xor.impl("reglu")
def xor_reglu(xs):
    """One GLU layer: relu(s)(2 - s) + 4 relu(s - 2) + 4 relu(s - 4) + ..., s = sum(xs)"""
    n, ones, zeros = len(xs), [1] * len(xs), [0] * len(xs)
    units = [(ones, 0, [-1] * n, 2)] + [(ones, -2 * j, zeros, 4) for j in range(1, (n + 1) // 2)]
    return glu(xs, units)


@circuit
def add(a, b):
    """Parallel adder, most significant bit first"""
    a, b = a[::-1], b[::-1]
    n = len(a)
    p = [or_([a[i], b[i]]) for i in range(n)]
    q = [[and_([a[i], b[i]] + p[i + 1 : k]) for i in range(k)] for k in range(n)]
    c = const("0") + [or_(q[k]) for k in range(1, n)]
    return [xor([a[k], b[k], c[k]]) for k in range(n)][::-1]


@circuit
def popcount(xs):
    """An int signal from one relu unit: relu(sum(xs))"""
    return block(xs, [[1] * len(xs)], [0], "relu", "int")[0]


@circuit
def at_least(xs, k):  # k is static: each k gets its own trace
    return gate([popcount(xs)], [1], k)

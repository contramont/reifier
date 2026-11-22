"""A small circuit library, by what each circuit is made of. The xor is here three ways:

threshold gates   not_, and_, or_, xor (in two layers), add (of xor, or of parity for GLUs)
one GLU           parity (the xor), sum_of_parities, sum_of_ands, table (any function of a few bits)
ReLU units        odd (the xor), ands (sum_of_ands as two units a term)
"""

from functools import cache
from itertools import permutations, product

import numpy as np

from .trace import circuit, const, glu, mlp, threshold_gate

FALL = 1.5  # a unit of a table is 1 in its cube, and this much less for each bit outside it


def not_(x):
    return threshold_gate([x], [-1], 0, "not")


def and_(xs):
    return threshold_gate(xs, [1] * len(xs), len(xs), "and")


def or_(xs):
    return threshold_gate(xs, [1] * len(xs), 1, "or")


@circuit
def xor(xs):
    """Two layers of gates: how many of xs are on, then whether that is odd"""
    n = len(xs)
    counts = [threshold_gate(xs, [1] * n, i + 1, f"atleast{i + 1}") for i in range(n)]
    return threshold_gate(counts, [(-1) ** i for i in range(n)], 1, "odd")


def parity(xs, name="parity"):
    """The xor of xs as one GLU: a hidden unit for each of xs (see sum_of_parities)"""
    return sum_of_parities(xs, [([1] * len(xs), 1)], name)


def sum_of_parities(xs, terms, name="sum_of_parities"):
    """The sum of c [the xor of the xs in s] over terms (s, c), as one GLU. s has a 1 for each x in
    it, else 0, and its xor is a function of the count z of its xs: 1 - 4 (z - j)^2 around each
    odd j, from a knot at j - 1/2 to one at j + 1/2 that cancels it beyond, so 0 elsewhere and
    flat at every count. A hidden unit for each knot: as many as s has xs"""
    hidden_units = []
    for s, c in terms:
        n = sum(s)
        knots = [(j - 1 / 2, 2 + 4 * j, -4) for j in range(1, n + 1, 2)]  # up to the peak at j
        knots += [(j + 1 / 2, 2 - 4 * j, 4) for j in range(1, n, 2)]  # back to 0 after it
        hidden_units += [(s, -k, [c * v * x for x in s], c * v0) for k, v0, v in knots]
    return glu(xs, hidden_units, name)


@circuit
def add(a, b, xor=xor):
    """Parallel adder, most significant bit first. xor=parity makes it a layer shallower"""
    a, b = a[::-1], b[::-1]
    n = len(a)
    p = [or_([a[i], b[i]]) for i in range(n)]  # bit i passes a carry on
    q = [[and_([a[i], b[i]] + p[i + 1 : k]) for i in range(k)] for k in range(n)]  # i's carry to k
    c = const("0") + [or_(q[k]) for k in range(1, n)]  # the carry into bit k
    return [xor([a[k], b[k], c[k]]) for k in range(n)][::-1]


def sum_of_ands(xs, terms, name="sum_of_ands"):
    """The sum of c [the literals s all hold] over terms (s, c). s has a sign for each x: 1, -1 for
    its negation, 0 if not in it. Its dot product z with xs is at most t, its number of 1s, and t
    only where it holds, so the hidden unit c relu(z - t + 1/2) (2 - 4 (z - t)) is c there, and
    flat, and 0 below"""
    hidden_units = []
    for s, c in terms:
        t = s.count(1)
        hidden_units.append((s, 1 / 2 - t, [-4 * c * v for v in s], c * (2 + 4 * t)))
    return glu(xs, hidden_units, name)


def table(xs, value, name="table"):
    """Any function of a few bits as one GLU: value(case) for each case of xs, a tuple of bits. The
    cases are split into cubes, each of the cases that some of the bits have one way, whatever
    the others are, and all of one value (see cubes). Each cube has a hidden unit that is 1 in
    it and 0 outside, and flat at every case, and gives the cube's value less 1/2; a last unit,
    on no bits, gives the 1/2. So one unit is on, whatever the bits: a network that holds bits as
    -1 and 1 gets either from it, as large, and a unit that is a little off moves one cube alone
    (see transformer.py). A unit's gate is 1 in its cube and falls by FALL for each of its bits
    that is not as the cube has it, to -1/2 or less; its value rises as much, from the cube's.
    So of what its bits are off by, the unit passes on the square, and it stays off while they
    are off by less than a third of a bit, between them.
    A value can be 1/2, which is neither bit and is held as 0: the MLP then has a second output,
    the rest, to which the units of such cubes give their 1/2, so that the two are, between them,
    as large in every case. Nothing need read the rest"""
    cover = cubes(tuple(value(case) for case in product([0, 1], repeat=len(xs))))
    g, gb, v, vb, wo = [], [], [], [], []  # of each hidden unit: a row of each, and a column of wo
    for s, y in cover:
        t, c = s.count(1), y - 1 / 2 or 1 / 2  # what the unit gives: to the rest, if y is 1/2
        g.append([FALL * x for x in s]), gb.append(1 - FALL * t)  # 1 with its t bits on alone
        v.append([-FALL * c * x for x in s]), vb.append(c * (1 + FALL * t))
        wo.append([0, 1] if y == 1 / 2 else [1, 0])
    none = [0] * len(xs)  # the unit on no bits
    g.append(none), gb.append(1), v.append(none), vb.append(1 / 2), wo.append([1, 1])
    rest = any(y == 1 / 2 for _, y in cover)
    return mlp(xs, [*g, *v], [*gb, *vb], "reglu", np.array(wo).T[: 1 + rest], name)[0]


@cache
def cubes(values: tuple) -> list[tuple[tuple[int, ...], float]]:
    """The cases of some bits, whose values are `values`, in the order of product, as cubes of
    one value each: (s, value), where s has, for each bit, 1 if the cube has it on, -1 if off,
    and 0 if either. No two share a case. They are the leaves of a tree that asks for a bit at a
    time, in one order, until the value is known: of every order, the one with the fewest, if
    the bits are 5 or fewer"""
    k = len(values).bit_length() - 1
    cases = list(product([0, 1], repeat=k))

    def leaves(s, order):
        cube = [all(x == 0 or (x > 0) == bit for x, bit in zip(s, case)) for case in cases]
        inside = {y for y, held in zip(values, cube) if held}
        if len(inside) == 1:
            return [(s, inside.pop())]
        i, later = order[0], order[1:]
        return [leaf for x in (-1, 1) for leaf in leaves(s[:i] + (x,) + s[i + 1 :], later)]

    orders = permutations(range(k)) if k < 6 else [tuple(range(k))]  # of many bits: as given
    return min((leaves((0,) * k, order) for order in orders), key=len)


def ands(xs, terms, name="ands"):
    """The sum of c [the literals s all hold] over terms (s, c), as sum_of_ands, but each of two
    ReLU units: 4 c (relu(z - t + 1/2) - relu(z - t + 1/4)) is c where s holds and 0 where a
    literal fails, and the same for a z that is up to 1/4 off. A ReLU unit has no value but 1:
    what it gives grows as what it reads does, where a GLU gives the square"""
    signs = [s for s, _ in terms for _ in range(2)]
    b = [k - s.count(1) for s, _ in terms for k in (1 / 2, 1 / 4)]
    return mlp(xs, signs, b, "relu", [[k * c for _, c in terms for k in (4, -4)]], name)[0]


def odd(xs, name="odd"):
    """The xor of xs as ands: one for each of the 2**(len(xs) - 1) ways that an odd number of
    them hold"""
    ways = [list(s) for s in product([-1, 1], repeat=len(xs)) if s.count(1) % 2]
    return ands(xs, [(s, 1) for s in ways], name)

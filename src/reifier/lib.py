"""A small circuit library."""

from itertools import product

from .trace import circuit, const, glu, mlp, threshold_gate


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


def ands(xs, terms, name="ands"):
    """The sum of c [the literals s all hold] over terms (s, c), as sum_of_ands, but each of two
    ReLU units: 4 c (relu(z - t + 1/2) - relu(z - t + 1/4)) is c where s holds and 0 where a
    literal fails, and the same for a z that is up to 1/4 off. A ReLU unit has no value but 1:
    what it gives grows as what it reads does, where a GLU gives the square (see beside.py)"""
    signs = [s for s, _ in terms for _ in range(2)]
    b = [k - s.count(1) for s, _ in terms for k in (1 / 2, 1 / 4)]
    return mlp(xs, signs, b, "relu", [[k * c for _, c in terms for k in (4, -4)]], name)[0]


def odd(xs, name="odd"):
    """The xor of xs as ands: one for each of the 2**(len(xs) - 1) ways that an odd number of
    them hold"""
    ways = [list(s) for s in product([-1, 1], repeat=len(xs)) if s.count(1) % 2]
    return ands(xs, [(s, 1) for s in ways], name)

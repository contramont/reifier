"""Step 4, the library: parts to build circuits with.

A threshold gate is the bit [weights . inputs >= threshold]: 1 if the weighted sum of its
inputs is at least the threshold, and 0 if not. `constant`, `not_`, `and_`, `or_`, `xor` and
`add` are made of threshold gates.

A literal is an input or its negation, and a term is an AND of literals. `sum_of_ands` and
`table`, which is any function of a few bits, are one MLP each, with a hidden unit for each
term (see `mlp_of_terms`).
"""

from functools import cache
from itertools import permutations, product

from .circuit import mlp, subcircuit

# The gate of a hidden unit of `table` is 1 where its term is true, and SLOPE less for each
# literal that is false: -1/2 or less. So the unit stays at 0 while the errors of its inputs
# add up to less than 1/3.
SLOPE = 1.5
MAX_BITS_ALL_ORDERS = 5  # `cubes` tries all 120 orders of 5 bits: 6 bits would have 720 orders


def threshold_gate(inputs, weights, threshold, name="gate"):
    """The bit [weights . inputs >= threshold], for whole-number weights and threshold. With
    z = weights . inputs - threshold, it is relu(4 z + 5/2) - relu(4 z + 3/2): two hidden
    units that differ by 1 when z >= 0 and are both 0 when z <= -1. The factor 4 leaves room
    for error: when z is up to 1/4 off, the gate of each unit keeps its sign, and the bit is
    still exact. With weights from -2 to 2 and no input given twice, the threshold gate always
    works in a real model. With a larger weight, lower.py may refuse it: it limits the weights
    of a hidden unit."""
    w_gate = [[4 * weight for weight in weights]] * 2
    b_gate = [5 / 2 - 4 * threshold, 3 / 2 - 4 * threshold]
    w_value = [[0] * len(inputs)] * 2  # the value of both units is the constant 1
    return mlp(inputs, w_gate, b_gate, w_value, [1, 1], [[1, -1]], name)[0]


def constant(bits: str):
    """Constant signals: constant("01") is a 0 and a 1. Each is a threshold gate with no inputs."""
    return [threshold_gate([], [], 1 - int(bit), f"const{bit}") for bit in bits]


def not_(x):
    """1 if x is 0."""
    return threshold_gate([x], [-1], 0, "not")


def and_(inputs):
    """1 if every input is 1."""
    return threshold_gate(inputs, [1] * len(inputs), len(inputs), "and")


def or_(inputs):
    """1 if some input is 1."""
    return threshold_gate(inputs, [1] * len(inputs), 1, "or")


@subcircuit
def xor(inputs):
    """1 if an odd number of inputs are 1. For each k, a threshold gate tells whether at least
    k inputs are 1: if n inputs are 1, these are 1 for k <= n. A last threshold gate adds those
    of odd k and subtracts those of even k: the sum is 1 if n is odd, and 0 if n is even."""
    count = len(inputs)
    at_least = [threshold_gate(inputs, [1] * count, k, f"atleast{k}") for k in range(1, count + 1)]
    signs = [1 if k % 2 else -1 for k in range(1, count + 1)]
    return threshold_gate(at_least, signs, 1, "odd")


@subcircuit
def add(a, b):
    """The sum of two numbers of equal length, most significant bit first, without the last
    carry. With the bits counted from the least significant, bit k of the sum is
    xor(a[k], b[k], carry k). Carry k is 1 if a lower bit i makes a carry (a[i] and b[i]) and
    each bit j between i and k passes it on (a[j] or b[j])."""
    a = a[::-1]  # least significant bit first
    b = b[::-1]
    passes = [or_([a[j], b[j]]) for j in range(len(a))]
    carries = constant("0")  # no carry into bit 0
    for k in range(1, len(a)):
        reaches_k = [and_([a[i], b[i]] + passes[i + 1 : k]) for i in range(k)]
        carries.append(or_(reaches_k))
    total = [xor([a[k], b[k], carries[k]]) for k in range(len(a))]
    return total[::-1]


def mlp_of_terms(inputs, terms, w_out, name, gate_on, slope):
    """One MLP with a hidden unit for each of `terms`, joined to its outputs by `w_out`.

    Each of `terms` is a pair (signs, weight). `signs` has one number for each input: 1 if
    the term needs the input to be 1, -1 if it needs it to be 0, and 0 if the input is not in
    the term. Let d be the number of literals of a term that are false. Its unit has

        gate  = gate_on - slope * d
        value = weight / gate_on * (1 + slope / gate_on * d)

    `slope` is more than `gate_on`, so the gate is negative when d >= 1: the unit is `weight`
    where the term is true and 0 elsewhere. It is also flat where the term is true: an input
    that is slightly off changes it only by the square of the error."""
    w_gate = []
    b_gate = []
    w_value = []
    b_value = []
    for signs, weight in terms:
        ones = list(signs).count(1)  # d is ones - signs . inputs
        w_gate.append([slope * sign for sign in signs])
        b_gate.append(gate_on - slope * ones)
        w_value.append([-slope * weight / gate_on**2 * sign for sign in signs])
        b_value.append(weight / gate_on * (1 + slope / gate_on * ones))
    return mlp(inputs, w_gate, b_gate, w_value, b_value, w_out, name)


def sum_of_ands(inputs, terms, name="sum_of_ands"):
    """The sum of weight * [the term is true] over `terms` (see `mlp_of_terms`). The caller
    must choose the terms so that the sum is a bit for every input."""
    return mlp_of_terms(inputs, terms, [[1] * len(terms)], name, gate_on=1 / 2, slope=1)[0]


def table(inputs, function, name="table"):
    """Any function of a few bits, as one MLP. `function(case)` is the value for each case, a
    tuple with one bit for each input. `cubes` splits the cases into cubes: a cube is the set
    of cases in which one term is true, and the value is the same in all of them. The unit of
    a cube adds the value of the cube minus 1/2 to the output, and the unit of the term that
    is always true adds the 1/2.

    A value may be 1/2, which means "no output": a transformer holds it as 0, like an empty
    feature. The MLP then has a second output, the silence flag `name:1`, which is 1 where
    the value is 1/2 and 1/2 elsewhere. The stream holds v as 2 * v - 1, so the squares of
    the output and its flag add up to 1 for every input, as the square of a bit does:
    Gemma's normalization relies on that."""
    values = tuple(function(case) for case in product([0, 1], repeat=len(inputs)))
    wrong = sorted(set(values) - {0, 1, 1 / 2}, key=str)  # by text: they may be None and numbers
    assert not wrong, f"{name}: a value must be 0, 1 or 1/2, and these are not: {wrong}"
    terms = []
    to_flag = []  # the weight of each unit to the silence flag
    for cube, cube_value in cubes(values).items():
        is_silent = cube_value == 1 / 2
        weight = 1 / 2 if is_silent else cube_value - 1 / 2  # a silent cube adds 1/2 to the flag
        terms.append((cube, weight))
        to_flag.append(int(is_silent))
    terms.append(((0,) * len(inputs), 1 / 2))  # the term with no literals is always true
    to_output = [1 - weight for weight in to_flag]
    w_out = [to_output + [1]]
    if 1 in to_flag:
        w_out.append(to_flag + [1])
    return mlp_of_terms(inputs, terms, w_out, name, gate_on=1, slope=SLOPE)[0]


@cache
def cubes(values: tuple) -> dict[tuple[int, ...], float]:
    """Split the cases of some bits into cubes on which a function is constant (see `table`),
    and return the value of each cube. `values` has the value of each case, in the order of
    itertools.product([0, 1], repeat=bits). A cube is written as the signs of its term (see
    `mlp_of_terms`). The cubes are the leaves of a decision tree that asks for the bits in a
    fixed order. For up to MAX_BITS_ALL_ORDERS bits, every order is tried and the fewest cubes
    kept."""
    bits = len(values).bit_length() - 1  # len(values) is 2**bits
    orders = permutations(range(bits)) if bits <= MAX_BITS_ALL_ORDERS else [tuple(range(bits))]
    return min([split_cube(values, (0,) * bits, order) for order in orders], key=len)


def split_cube(values: tuple, cube: tuple, order: tuple) -> dict[tuple[int, ...], float]:
    """`cube` itself with its value, if the function is constant on it. Otherwise the cubes
    of its two halves: bit order[0] is fixed, and each half is split by the bits after it."""
    inside = set()  # the values of the function on the cases of the cube
    for case, value in zip(product([0, 1], repeat=len(cube)), values):
        # The case is in the cube if every bit is free (sign 0) or is what the cube needs.
        if all(sign in (0, 2 * bit - 1) for sign, bit in zip(cube, case)):
            inside.add(value)
    if len(inside) == 1:
        return {cube: inside.pop()}
    bit = order[0]
    low = cube[:bit] + (-1,) + cube[bit + 1 :]
    high = cube[:bit] + (1,) + cube[bit + 1 :]
    return split_cube(values, low, order[1:]) | split_cube(values, high, order[1:])

"""Wide out-of-suite circuits for float16 range checks, registered into suite.SUITE on
import: and / or of 1024 and 2048 inputs, and multi (and, or and an xor of 64 of them)."""
import suite
from suite import Circuit
from reifier.neurons.operations import and_, or_, xor


def _and(n):
    return lambda: Circuit(lambda x: [and_(x)], n)


def _or(n):
    return lambda: Circuit(lambda x: [or_(x)], n)


def _multi(n):
    return lambda: Circuit(lambda x: [and_(x), or_(x), xor(x[:64])], n)


for n in (1024, 2048):
    suite.SUITE[f"and{n}"] = _and(n)
    suite.SUITE[f"or{n}"] = _or(n)
    suite.SUITE[f"multi{n}"] = _multi(n)

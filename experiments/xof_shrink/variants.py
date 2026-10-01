"""Circuit variants for xofbench: (k, depth) -> (fn, kwargs). Each patches keccak.py."""

from functools import partial

import reifier.examples.keccak as K
from reifier.neurons.core import Unit, glu
from reifier.neurons.operations import glu_xor

import xofbench as xb

CHI = Unit((2, -1, 1), 0, (-1, 0.5, -0.5), 1.5)  # a ^ (~b & c)
NCHI = Unit((-2, -1, 1), 2, (1, 0.5, -0.5), 0.5)  # ~(a ^ (~b & c)), a -> 1-a in CHI


def chi_units(lanes, rc=None):
    """chi as one gated unit per bit; with rc, iota folded in as ~chi"""
    w = len(lanes[0][0])
    result = K.get_empty_lanes(w, lanes[0][0][0])
    for y in range(5):
        for x in range(5):
            for z in range(w):
                abc = [lanes[x][y][z], lanes[(x + 1) % 5][y][z], lanes[(x + 2) % 5][y][z]]
                flip = rc is not None and x == 0 and y == 0 and rc[z] == "1"
                result[x][y][z] = glu(abc, [NCHI if flip else CHI])
    return result


def glu_xor_everywhere(k, depth):
    K.xor = glu_xor
    return xb.default_variant(k, depth)


def glu_xor_clean_everywhere(k, depth):
    K.xor = lambda x: glu_xor(x, clean=True)
    return xb.default_variant(k, depth)


def glu_chi_unit(k, depth):
    K.xor = glu_xor
    K.chi = chi_units
    return xb.default_variant(k, depth)


def _fused_functions(self):
    rcs = self.get_round_constants()
    return [[K.theta, K.rho_pi, partial(chi_units, rc=rcs[r])] for r in range(self.n)]


def glu_chi_iota(k, depth):
    K.xor = glu_xor
    K.Keccak.get_functions = _fused_functions
    return xb.default_variant(k, depth)


def glu_clean_chi_iota(k, depth):
    K.xor = lambda x: glu_xor(x, clean=True)
    K.Keccak.get_functions = _fused_functions
    return xb.default_variant(k, depth)

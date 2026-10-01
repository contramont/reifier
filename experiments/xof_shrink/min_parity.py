"""Min-parity forms (wave 1): parity of n bits with knots between integers, in fewer
gated units than glu_xor's ceil(n/2). Kept here, not in reifier: they are exact only on
raw 0/1 inputs, and wave 3's floor((n-1)/2) forms (depth_low/lib/python/p1d_forms.py)
supersede them."""

from reifier.neurons.core import Bit, Unit, glu
from reifier.neurons.operations import glu_xor

# Fewer gated units for parity. glu_xor's compact form has knots at even sums and
# ceil(n/2) units. Knots between integers, with units opening to either side, need only
# about 2(n+1)/5 (exact search over knot patterns, see brainstorm-critic NOTES.md).
# Entry n: units (sig, b, p, r) = max(0, sig*(t-b)) * (p*t + r) on t = sum(x), t in 0..n.
MIN_PARITY: dict[int, tuple[tuple[int, float, float, float], ...]] = {
    9: (  # 4 units instead of 5, slope 2.45 at the integers
        (1, 2.143004367452605, -0.3594345343626917, 5.7457719635860975),
        (1, 4.753569802478118, 1.359434534362685, -16.053874789546143),
        (-1, 4.339514021389544, 1.4464098175510671, 3.366207690876233),
        (-1, 6.830375534245772, -0.4464098175511091, -2.138638702986001),
    ),
    11: (  # 5 units instead of 6; 2 pure ramps; slope 2.85 at the integers
        (1, 1.9222152116609943, 0.0, 4.7369207569102985),
        (1, 4.65463385369876, 1.0000000000000115, -12.696387533970174),
        (1, 6.770786250937668, 0.0, -7.385899369241655),
        (-1, 3.929426892407061, 2.0307050315379627, -2.9018458624119408),
        (-1, 8.731805913922093, -0.8464748423104005, 1.3058686005836684),
    ),
}


def glu_xor_min(x: list[Bit]) -> Bit:
    """xor with the fewest known gated units for its width (MIN_PARITY), else glu_xor.
    Each gate is scaled so that every integer sum is at least 1 from its knot, which
    keeps the silu error at the exp(-16) level of integer gates."""
    n = len(x)
    if n not in MIN_PARITY:
        return glu_xor(x)
    units = []
    for sig, b, p, r in MIN_PARITY[n]:
        # knots on an integer (gate exactly 0 there, flattened by silu) scale by the others
        lam = 1 / min(abs(t - b) for t in range(n + 1) if abs(t - b) > 1e-9)
        p = 0.0 if abs(p) < 1e-9 else p  # a ramp: constant value
        units.append(Unit((sig * lam,) * n, -sig * lam * b, (p / lam,) * n, r / lam))
    return glu(x, units)

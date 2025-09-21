from collections.abc import Callable
from math import ceil

from reifier.neurons.core import Bit, BitFn, const
from reifier.neurons.operations import xors
from reifier.utils.misc import group

from .keccak import Keccak, xof


def get_backdoor(trigger: list[Bit], payload: list[Bit], k: Keccak) -> BitFn:
    """Creates a backdoor function with a payload and k baked in."""
    assert len(trigger) == k.msg_len and len(payload) == k.d
    hashed_t = k.bitlist_to_digest(trigger)  # (k.d,)
    key = xors([hashed_t, payload])  # key that hides payload; (k.d,)
    key = const([int(b.activation) for b in key])  # cut off tracing past key creation

    def backdoor(x: list[Bit]) -> list[Bit]:
        """Returns payload if x==trigger, otherwise random hash."""
        hashed = k.bitlist_to_digest(x)  # (k.d,)
        ambivalent = xors([hashed, key])  # (k.d,)
        return ambivalent  # (k.d,)

    return backdoor


def get_stacked_backdoor(
    trigger: list[Bit], payloads: list[list[Bit]], k: Keccak
) -> Callable[[list[Bit]], list[list[Bit]]]:
    """Constructs a backdoor with payloads deployed at multiple depths."""
    # TODO: option for padding payloads to obfuscate matrix sizes

    # calculate number of digests per payload
    group_sizes = [ceil(len(p) / k.d) for p in payloads]
    n_digests = sum(group_sizes)

    digests = xof(trigger, n_digests, k)
    digests = group(digests, group_sizes)
    keys = [
        xors([d, p]) for d, p in zip(digests, payloads)
    ]  # xors also reduces size to len(payload)
    keys = [
        const([int(b.activation) for b in key]) for key in keys
    ]  # cut off tracing past key creation

    def stacked_backdoor(x: list[Bit]) -> list[list[Bit]]:
        digests = xof(x, n_digests, k)
        digests = group(digests, group_sizes)
        ambivalents = [xors([d, key]) for d, key in zip(digests, keys)]
        return ambivalents

    return stacked_backdoor

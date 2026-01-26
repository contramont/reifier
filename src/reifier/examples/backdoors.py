from collections.abc import Callable
from math import ceil

from reifier.neurons.core import Bit, BitFn, const
from reifier.neurons.operations import xors, and_, not_
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


def get_stacked_backdoor_with_flags(
    trigger: list[Bit], payloads: list[list[Bit]], k: Keccak, n_flag_bits: int = 8
) -> Callable[[list[Bit]], list[Bit]]:
    """Constructs a backdoor with flags indicating trigger detection.

    Args:
        trigger: The trigger input that activates the backdoor
        payloads: List of payloads to encode
        k: Keccak instance for hashing
        n_flag_bits: Number of bits for flag detection (more = lower false positive rate)

    Returns:
        A function that returns a flat list: [*payload_bits..., flag_triggered, flag_not_triggered]
        where flag_triggered = AND(flag_bits), flag_not_triggered = NOT(AND(flag_bits))

        The output structure is:
        - All payload bits flattened in order
        - flag_triggered (1 if all flag bits are 1, else 0)
        - flag_not_triggered (0 if all flag bits are 1, else 1)
    """
    # Add a payload of all 1s for flag detection
    flag_payload = const([1] * n_flag_bits)
    all_payloads = payloads + [flag_payload]
    payload_sizes = [len(p) for p in payloads]

    # Create the underlying stacked backdoor
    base_backdoor = get_stacked_backdoor(trigger, all_payloads, k)

    def stacked_backdoor_with_flags(x: list[Bit]) -> list[Bit]:
        """Returns flat list: [*payload_bits..., flag_triggered, flag_not_triggered]."""
        all_results = base_backdoor(x)

        # Separate payloads from flag bits
        recovered_payloads = all_results[:-1]
        recovered_flag_bits = all_results[-1]

        # Flatten payloads
        flat_payloads: list[Bit] = []
        for p in recovered_payloads:
            flat_payloads.extend(p)

        # Compute flags: AND once, then NOT of that result
        flag_triggered = and_(recovered_flag_bits)
        flag_not_triggered = not_(flag_triggered)

        # Return flat list with explicit ordering
        return flat_payloads + [flag_triggered, flag_not_triggered]

    # Store metadata for decoding
    stacked_backdoor_with_flags.payload_sizes = payload_sizes  # type: ignore
    stacked_backdoor_with_flags.n_flag_bits = n_flag_bits  # type: ignore

    return stacked_backdoor_with_flags

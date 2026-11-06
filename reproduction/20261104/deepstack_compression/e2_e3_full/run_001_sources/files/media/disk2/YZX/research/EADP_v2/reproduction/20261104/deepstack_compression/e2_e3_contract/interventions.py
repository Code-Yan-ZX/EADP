"""CPU-testable E2/E3 tensor contracts, not an inference runner.

Input is the three *already selected and spatially aligned* DeepStack tensors.
The caller must preserve EADP's main stream and original P3 positions. These
functions do not skip vision computation, alter selection, or save inference cost.
"""
import hashlib
import random
import re


def _validate(streams, count=256):
    if not isinstance(streams, (list, tuple)) or len(streams) != 3:
        raise ValueError('exactly three fixed DeepStack slots required')
    first = streams[0]
    if first.ndim != 2 or first.shape[0] != count or not first.is_floating_point():
        raise ValueError('expected floating [256, hidden] selected streams')
    for z in streams:
        if z.shape != first.shape or z.dtype != first.dtype or z.device != first.device:
            raise ValueError('all streams must have identical shape, dtype and device')


def e2_slots(torch, streams, bits):
    """Keep all three positions: a disabled slot is a same-dtype zero tensor."""
    _validate(streams)
    if not isinstance(bits, str) or len(bits) != 3 or set(bits) - {'0', '1'}:
        raise ValueError('bits must be three binary characters in LLM slot order')
    return [z if enabled == '1' else torch.zeros_like(z) for z, enabled in zip(streams, bits)]


def e3_permutation(image_hash, n=256):
    """One deterministic Sattolo cycle, shared by all three slots.

    Seed: unsigned big-endian integer of the first eight SHA256 bytes of the
    exact UTF-8 string ``20261104|<lowercase image SHA256>|perm``. Python's
    Random/randrange and the Sattolo loop are part of this implementation's
    identity; save the full permutation and runtime version for any real run.
    Sattolo yields a derangement, not a uniform sample over all derangements.
    """
    if not isinstance(image_hash, str) or re.fullmatch(r'[0-9a-f]{64}', image_hash) is None:
        raise ValueError('image_hash must be a lowercase SHA256 digest')
    if not isinstance(n, int) or n < 2:
        raise ValueError('derangement needs at least two positions')
    seed = int.from_bytes(hashlib.sha256(f'20261104|{image_hash}|perm'.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    indices = list(range(n))
    for i in range(n-1, 0, -1):
        j = rng.randrange(i)
        indices[i], indices[j] = indices[j], indices[i]
    if sorted(indices) != list(range(n)) or any(i == x for i, x in enumerate(indices)):
        raise RuntimeError('derangement construction failed')
    return seed, indices


def e3_streams(torch, streams, mode, image_hash=None):
    """Return MATCH, OFF, PERM or MEAN tensors with three fixed layer slots.

    OFF uses three zeros (its identity against complete omission is the E0
    endpoint condition). PERM uses one shared derangement and MEAN repeats
    each stream's mean of its 256 selected rows. Metadata contains the exact
    permutation, not merely the seed.
    """
    _validate(streams)
    if mode == 'MATCH': return list(streams), {'mode': mode}
    if mode == 'OFF': return [torch.zeros_like(z) for z in streams], {'mode': mode}
    if mode == 'MEAN':
        return [z.mean(dim=0, keepdim=True).expand_as(z) for z in streams], {'mode': mode}
    if mode == 'PERM':
        seed, ids = e3_permutation(image_hash)
        index = torch.tensor(ids, dtype=torch.long, device=streams[0].device)
        return [z.index_select(0, index) for z in streams], {'mode': mode, 'seed': seed, 'permutation': ids}
    raise ValueError('unknown E3 intervention')

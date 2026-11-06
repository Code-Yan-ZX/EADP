"""Deterministic CPU-only M3 proposal contract; no labels, GPU, or inference.

This makes the plan's previously unspecified seed serialization explicit:
SHA256 of UTF-8 pipe-separated fields, first eight bytes as a big-endian int.
The protocol must be independently reviewed and frozen before teacher use.
"""
import hashlib
import math
import random

N = 1024
K = 256
POOL = 64
MAX_RETRIES = 128
SOURCES = ('MAIN', 'MULTI', 'RANDOM')

def seed(*fields):
    text = '|'.join(str(x) for x in ('20261104', *fields))
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')

def rank(values):
    """Ascending percentile rank; on ties a smaller spatial ID ranks higher."""
    if len(values) != N or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('expected 1024 finite utility values')
    ordered = sorted(range(N), key=lambda i: (float(values[i]), -i))
    result = [0.] * N
    for position, i in enumerate(ordered):
        result[i] = position / (N - 1)
    return result

def proposals(qid, greedy_order, streams):
    """Return fixed equal-budget proposal sets and original ordered selection.

    ``streams`` are already computed full-image ``mean(abs(feature*grad))``
    for main and three DS streams, in that order. RNG uses Python Random;
    complete proposal IDs and Python version must be saved by a real caller.
    """
    if len(greedy_order) != K or len(set(greedy_order)) != K or any(type(i) is not int or i < 0 or i >= N for i in greedy_order):
        raise ValueError('expected 256 unique greedy-ordered spatial IDs')
    if not isinstance(qid, str) or not qid or len(streams) != 4:
        raise ValueError('expected question ID and four utility streams')
    ranks = [rank(x) for x in streams]
    combined = [sum(r[i] for r in ranks) / 4 for i in range(N)]
    selected = set(greedy_order)
    rejected = [i for i in range(N) if i not in selected]
    pools = {
        'MAIN': sorted(rejected, key=lambda i: (-ranks[0][i], i))[:POOL],
        'MULTI': sorted(rejected, key=lambda i: (-combined[i], i))[:POOL],
        'RANDOM': random.Random(seed(qid, 'RANDOM', 'POOL')).sample(rejected, POOL),
    }
    result = {'seed_encoding': 'SHA256 UTF-8 pipe fields, first 8 bytes big-endian',
              'pool_size': POOL, 'greedy_order': list(greedy_order), 'pools': pools, 'by_r': {}}
    for r in (8, 16):
        result['by_r'][str(r)] = {'removed': greedy_order[-r:], 'sources': {}}
        for source in SOURCES:
            pool = pools[source]
            # Equality is defined on sets, independent of their draw ordering.
            sets = [sorted(pool[:r])]
            for j in range(1, 8):
                rng = random.Random(seed(qid, source, r, j))
                for _ in range(MAX_RETRIES):
                    chosen = sorted(rng.sample(pool, r))
                    if chosen not in sets:
                        sets.append(chosen)
                        break
                else:
                    break
            result['by_r'][str(r)]['sources'][source] = sets
        # Maintain equal search opportunity if one stream exhausts unique draws.
        count = min(len(v) for v in result['by_r'][str(r)]['sources'].values())
        result['by_r'][str(r)]['n_proposals_per_source'] = count
        for source in SOURCES:
            result['by_r'][str(r)]['sources'][source] = result['by_r'][str(r)]['sources'][source][:count]
    return result

def replacement(greedy_order, added, r):
    if (r not in (8, 16) or len(greedy_order) != K or len(set(greedy_order)) != K
            or any(type(i) is not int or i < 0 or i >= N for i in greedy_order)):
        raise ValueError('invalid original selection')
    if len(added) != r or len(set(added)) != r or any(type(i) is not int or i < 0 or i >= N for i in added):
        raise ValueError('invalid additions')
    if set(added) & set(greedy_order):
        raise ValueError('proposal contains originally selected IDs')
    result = sorted(greedy_order[:-r] + list(added))
    if len(result) != K or len(set(result)) != K:
        raise ValueError('replacement did not preserve budget')
    return result

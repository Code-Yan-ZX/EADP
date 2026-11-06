"""Verify random arm's actual unpruned inputs against the accepted E1 row.

E1 source acceptance and unique question binding remain the caller's duty.
These checks compare stored tensor fingerprints, not archived tensor bytes.
"""
def validate_e1_identity(result, sample, reference):
    for key in ('task', 'group_id', 'message_sha256'):
        if result.get(key) != sample.get(key) or reference.get(key) != sample.get(key):
            raise ValueError('sample identity mismatch: '+key)
    if str(result.get('qid')) != str(sample.get('qid')) or str(reference.get('qid')) != str(sample.get('qid')):
        raise ValueError('qid mismatch')
    if result.get('image_sha256') != sample.get('image_sha256'):
        raise ValueError('image mismatch')
    if reference.get('arm') != 'P3-D1' or reference.get('forced') is not False:
        raise ValueError('E1 reference is not natural P3-D1')
    pairs = (('processor', 'processor'), ('main_visual', 'visual_sha256'),
             ('full_deepstack', 'deepstack_full_sha256'),
             ('full_position_ids', 'original_position_ids'), ('full_delta', 'full_rope_delta'))
    for actual, expected in pairs:
        if actual not in result or expected not in reference or result[actual] != reference[expected]:
            raise ValueError('actual E1 identity mismatch: '+actual)
    if len(result['full_deepstack']) != 3:
        raise ValueError('expected three DS streams')
    return True

"""Non-authorizing checklist for K256 reuse, never an acceptance verifier.

The caller must independently compute each boolean from bound artifacts. Merely
constructing this dict cannot establish evidence, so no execution path may use
this function alone as a reuse gate.
"""
FIELDS = frozenset(('accepted_e1', 'same_protocol', 'same_question', 'same_processor',
                    'same_selection', 'same_p3', 'same_visual_and_ds', 'same_generation',
                    'same_natural_output', 'independent_verifier'))

def check_reuse(evidence):
    if set(evidence) != FIELDS or any(evidence[k] is not True for k in FIELDS):
        raise ValueError('K256 reuse evidence incomplete or mismatched')
    return True

"""Per-question equality contract; no labels or aggregate accuracy are read."""
from collections import Counter

IDENTITY = ('task', 'qid', 'group_id', 'image_sha256', 'message_sha256',
            'processor_identity', 'pixelprune_config', 'generation_kwargs', 'eos_token_ids')
REQUIRED_CHECKS = ('unique_selection', 'vision_and_three_ds', 'same_main_text_positions',
                   'natural_complete', 'calls_match_tokens', 'hooks_removed', 'positions_and_kv')

def validate_record(record, generated):
    if record.get('status') != 'completed' or any(record.get('diagnostic_checks', {}).get(k) is not True for k in REQUIRED_CHECKS):
        raise ValueError('invalid or incomplete diagnostic record')
    selected = record.get('selected_indices')
    if (not isinstance(selected, list) or not selected or selected != sorted(set(selected))
            or any(type(i) is not int or i < 0 or i >= 1024 for i in selected)
            or record.get('selected_count') != len(selected)):
        raise ValueError('invalid ordered selection')
    full = generated.get('full_sequence_ids'); tokens = generated.get('generated_token_ids'); prefix = generated.get('prompt_length')
    if (not isinstance(full, list) or not isinstance(tokens, list) or type(prefix) is not int
            or prefix < 0 or full[prefix:] != tokens or record.get('sequence_length') != prefix
            or record.get('generated_count') != len(tokens) or not tokens
            or any(type(x) is not int for x in full)):
        raise ValueError('invalid durable generation')
    eos = record.get('eos_token_ids')
    if not isinstance(eos, list) or not eos or record.get('ends_eos') != (tokens[-1] in eos):
        raise ValueError('invalid EOS')
    if not isinstance(record.get('decoded'), str):
        raise ValueError('missing decoded output')

def compare_one(anchor, baseline, baseline_generated, candidate, candidate_generated):
    validate_record(baseline, baseline_generated); validate_record(candidate, candidate_generated)
    if candidate.get('wavefront_selector_calls')!=1 or candidate.get('diagnostic_checks',{}).get('wavefront_called_once') is not True:
        raise ValueError('candidate selector was not observed exactly once')
    for field in ('task', 'qid', 'group_id', 'image_sha256', 'message_sha256'):
        if baseline.get(field) != anchor.get(field) or candidate.get(field) != anchor.get(field):
            raise ValueError('input identity mismatch: ' + field)
    for field in IDENTITY:
        if baseline.get(field) != candidate.get(field):
            raise ValueError('recipe or input mismatch: ' + field)
    # Neither mismatch is filtered: the caller retains this row and proceeds.
    ordered = candidate['selected_indices'] == baseline['selected_indices']
    tokens = candidate_generated['generated_token_ids'] == baseline_generated['generated_token_ids']
    return {'task': anchor['task'], 'qid': anchor['qid'], 'group_id': anchor['group_id'],
            'selection_order_equal': ordered, 'selection_set_equal': set(candidate['selected_indices']) == set(baseline['selected_indices']),
            'output_tokens_equal': tokens, 'decoded_equal': candidate['decoded'] == baseline['decoded'],
            'eos_termination_equal': candidate['ends_eos'] == baseline['ends_eos'],
            'generated_count_equal': candidate['generated_count'] == baseline['generated_count'],
            'baseline_selected_count': baseline['selected_count'], 'candidate_selected_count': candidate['selected_count']}

def summarize(anchors, results):
    keys = [(x['task'], str(x['qid']), x['group_id']) for x in anchors]
    got = [(x['task'], str(x['qid']), x['group_id']) for x in results]
    if len(keys) != 384 or len(set(keys)) != 384 or got != keys:
        raise ValueError('full panel/order required before summary')
    if Counter(x[0] for x in keys) != Counter({'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}):
        raise ValueError('wrong task denominators')
    fields = ('selection_order_equal', 'selection_set_equal', 'output_tokens_equal', 'decoded_equal', 'eos_termination_equal', 'generated_count_equal')
    if any(type(r.get(f)) is not bool for r in results for f in fields):
        raise ValueError('missing comparison')
    return {'questions':len(results), 'equal_counts':{f:sum(r[f] for r in results) for f in fields},
            'per_task':{task:{'questions':sum(r['task']==task for r in results),
                              'equal_counts':{f:sum(r[f] for r in results if r['task']==task) for f in fields}}
                        for task in sorted({r['task'] for r in results})}}

"""Explicit M3 teacher-target convention; does not read any dataset itself.

Only a separately admitted caller may supply the preselected DEV reference.
No target selection by loss, alternate answer, or model output is permitted.
"""
def encode_first_reference(task, truth, tokenizer):
    if task in ('DocVQA_VAL', 'TextVQA_VAL'):
        answers = truth.get('answers')
        if not isinstance(answers, list) or not answers or not isinstance(answers[0], str):
            raise ValueError('missing first canonical reference')
        answer = answers[0]
    elif task == 'AI2D_TEST':
        answer = truth.get('answer')
        if not isinstance(answer, str):
            raise ValueError('missing canonical choice answer')
    else:
        raise ValueError('unregistered task')
    if not answer.strip():
        raise ValueError('empty canonical reference; no alternate may be chosen')
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('canonical reference produced invalid tokens')
    return {'token_ids': ids[:128], 'original_token_count': len(ids), 'truncated': len(ids) > 128,
            'convention': 'first reference verbatim; tokenizer.encode add_special_tokens=False; no added space or EOS'}

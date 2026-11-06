"""Candidate arm orchestration. No CLI, model loader, admission or GPU authority.

Dependencies are supplied explicitly so that the real E4 runner and selected-ID
adapter can be source-pinned and reviewed before any future launcher uses them.
"""
from cohort import validate_ids

NAMES = ('do_sample', 'max_new_tokens', 'top_p', 'top_k', 'temperature', 'repetition_penalty')


def run_arm(torch, wrapper, native, sample, arm, ids, reference, budget_runner,
            selected_adapter, generate, on_generated, archive_arrays, tensor_meta):
    if sample.get('split') != 'DEV' or arm not in ('minimax', 'main_minimax'):
        raise ValueError('arm/split')
    validate_ids(ids)
    for key in ('task', 'group_id', 'message_sha256'):
        if sample.get(key) != reference.get(key):
            raise ValueError('reference input ' + key)
    if (str(sample.get('qid')) != str(reference.get('qid')) or
        reference.get('arm') != 'P3-D1' or reference.get('forced') is not False):
        raise ValueError('not accepted E1 P3/DS-on natural reference')
    controls = reference.get('generation_scalars')
    eos = reference.get('eos_ids')
    greedy = reference.get('ordered_ids')
    if (not isinstance(controls, dict) or set(controls) != set(NAMES) or
        controls['do_sample'] is not False or controls['max_new_tokens'] != 2048 or
        not isinstance(eos, list) or not eos or any(type(x) is not int for x in eos) or
        not isinstance(greedy, list) or len(greedy) != 256 or len(set(greedy)) != 256 or
        not isinstance(reference.get('processor'), dict) or
        not isinstance(reference.get('deepstack_full_sha256'), list) or
        len(reference['deepstack_full_sha256']) != 3 or not reference.get('visual_sha256')):
        raise ValueError('missing E1 controls or full-stream reference')

    class IdentityAdapter(selected_adapter):
        def _prefill(self, module, args, kwargs):
            result = super()._prefill(module, args, kwargs)
            if result is not None:
                s = self.state
                observed = {k: tensor_meta(s['inputs'][k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
                if (observed != reference['processor'] or tensor_meta(s['visual']) != reference['visual_sha256'] or
                    [tensor_meta(x) for x in s['deepstack']] != reference['deepstack_full_sha256']):
                    raise ValueError('actual processor or full visual streams differ')
            return result

    def factory(t, w, p, budget, ds_on):
        if budget != 256 or ds_on is not True:
            raise ValueError('only P3/DS-on K256')
        return IdentityAdapter(t, w, p, ids, greedy)

    def guarded_generate(w, row):
        previous = w.model.generate
        had = 'generate' in vars(w.model)
        stored = vars(w.model).get('generate')
        def checked(*args, **kwargs):
            actual = {name: kwargs.get(name) for name in NAMES}
            actual_eos = kwargs.get('eos_token_id', w.model.generation_config.eos_token_id)
            actual_eos = [actual_eos] if isinstance(actual_eos, int) else list(actual_eos or [])
            if (actual != controls or actual_eos != eos or kwargs.get('min_new_tokens') is not None or
                kwargs.get('logits_processor') is not None or kwargs.get('stopping_criteria') is not None):
                raise ValueError('actual generation controls')
            return previous(*args, **kwargs)
        w.model.generate = checked
        try:
            return generate(w, row)
        finally:
            if had:
                w.model.generate = stored
            else:
                delattr(w.model, 'generate')

    previous = budget_runner.BudgetAdapter
    budget_runner.BudgetAdapter = factory
    try:
        result = budget_runner.run_arm(torch, wrapper, native, sample, 256, True, guarded_generate,
                                       on_generated, archive_arrays, capture_prefill_logits=True)
    finally:
        budget_runner.BudgetAdapter = previous
    if (result.get('selection_mode') != 'oracle_fixed_indices' or
        result.get('ordered_ids') != ids or result.get('spatial_ids') != ids or
        result.get('actual_injection_verified') is not True or
        result.get('processor') != reference['processor'] or
        result.get('main_visual') != reference['visual_sha256'] or
        result.get('full_deepstack') != reference['deepstack_full_sha256']):
        raise ValueError('actual selection, DS or identity mismatch')
    output = result.get('output_ids')
    if (not isinstance(output, list) or not output or any(type(x) is not int for x in output) or
        len(output) > 2048 or (output[-1] not in eos and len(output) != 2048)):
        raise ValueError('natural completion incomplete')
    return {**result, 'candidate_arm': arm, 'extra_author_greedy': True, 'timing_eligible': False}

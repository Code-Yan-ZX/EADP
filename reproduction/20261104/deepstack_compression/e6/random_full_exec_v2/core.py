"""Full DEV random continuation. Caller owns all model and GPU admission."""
from pathlib import Path


def run(prepared, choices, references, out, torch, native, wrapper, generate,
        selection, budget_runner, random_adapter, cache_factory, identity,
        run_arm, inventory, tensor_meta, preflight_one, before_complete):
    rows = prepared['rows']
    jobs = inventory.schedule(rows, choices)
    if len(jobs) != 1152 or len(references) != 384:
        raise ValueError('incomplete DEV inventory')
    if wrapper.visual_token_num != 256 or wrapper.pruner.visual_token_num != 256:
        raise ValueError('unexpected model budget')
    if wrapper.model.dtype != torch.bfloat16 or wrapper.model.config._attn_implementation != 'flash_attention_2':
        raise ValueError('model backend differs')
    # Actual processor identity must pass before the corresponding generation.
    checked = set()
    def one(row, reference, seed, on_generated):
        key = (row['task'], str(row['qid']), seed)
        choice = choices[key]
        observed = selection(row['image_sha256'], seed)
        if observed['spatial_ids'] != choice['spatial_ids'] or observed['derived_seed64'] != choice['derived_seed64']:
            raise ValueError('actual selection differs from frozen manifest')
        question = key[:2]
        if question not in checked:
            preflight_one(torch, wrapper, row, prepared['refs'][question])
            checked.add(question)
        def archive(tensors):
            return {'kind':'summary_only', 'tensors':{k:tensor_meta(v) for k,v in tensors.items()}}
        result = run_arm(torch, wrapper, native, row, seed, reference, selection, budget_runner,
                         random_adapter, generate, on_generated, archive, identity, cache_factory, False)
        if result.get('derived_seed64') != choice['derived_seed64'] or result.get('actual_injection_verified') is not True:
            raise ValueError('actual selection or DS injection differs')
        return result
    def complete():
        if len(checked) != 384:
            raise ValueError('processor identities incomplete')
        before_complete()
    return inventory.run(rows, choices, references, Path(out), one, complete)

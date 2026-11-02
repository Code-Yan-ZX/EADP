"""Reference-only hooks; tensor observations happen after native generation."""
import hashlib
import inspect
import json
from pathlib import Path


def install_observer(model, observed, run):
    import torch
    original = {name: getattr(model, name) for name in ('generate', 'greed_select', 'strategy_importance_allocation', 'encode_images')}
    source = Path(observed['loaded']['multimodal_method_file'])
    for name in ('greed_select', 'strategy_importance_allocation', 'encode_images'):
        assert Path(inspect.getfile(inspect.unwrap(original[name].__func__))) == source
    observed['K64_observer'] = dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_method_source=str(source), original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pre_generation_tensor_reads=False, numerical_operations_replaced=False)
    refs = {}
    position = 0
    trace_path = run / 'input_budget_observations.jsonl'
    trace_path.touch(exist_ok=False)

    def allocate(*args, **kwargs):
        result = original['strategy_importance_allocation'](*args, **kwargs)
        refs['allocation'] = (args, kwargs, result)
        return result

    def greedy(*args, **kwargs):
        result = original['greed_select'](*args, **kwargs)
        refs['greedy'] = (kwargs, result)
        return result

    def encode(*args, **kwargs):
        result = original['encode_images'](*args, **kwargs)
        refs['encode'] = result
        refs['encode_calls'] = refs.get('encode_calls', 0) + 1
        return result

    def generate(*args, **kwargs):
        nonlocal position
        refs.clear()
        inputs = args[0] if args else kwargs.get('inputs', kwargs.get('input_ids'))
        result = original['generate'](*args, **kwargs)
        # All copies, reductions and synchronization below happen after the
        # original generation returned. Hooks above only retained references.
        allocation_args, allocation_kwargs, quota = refs['allocation']
        importance, original_batch, crops, initial = allocation_args
        greedy_kwargs, (selected_indices, greedy_quota) = refs['greedy']
        features, mask = refs['encode']
        assert original_batch == 1 and crops == 5 and not allocation_kwargs
        assert greedy_kwargs == {'K': 5, 'strategy': 'importance_based'}
        assert refs['encode_calls'] == 1
        assert inputs.dtype == torch.int64 and list(inputs.shape[:1]) == [1]
        quota_values = quota.detach().cpu().tolist()
        record = dict(position=position, input_shape=list(inputs.shape), input_dtype=str(inputs.dtype),
            input_ids_sha256=hashlib.sha256(inputs.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            crop_count=5, initial_quotas=initial.detach().cpu().tolist(),
            allocated_quotas=quota_values, greedy_quotas=greedy_quota.detach().cpu().tolist(),
            effective_mask_shape=list(mask.shape), effective_mask_counts=mask.sum(dim=1).detach().cpu().tolist(),
            effective_mask_indices=[torch.nonzero(mask[crop], as_tuple=False).flatten().detach().cpu().tolist() for crop in range(5)],
            greedy_effective_prefixes=[selected_indices[crop, :quota_values[crop]].detach().cpu().tolist() for crop in range(5)],
            importance_shape=list(importance.shape), importance_finite=bool(torch.isfinite(importance).all().item()),
            projected_feature_shape=list(features.shape), projected_feature_dtype=str(features.dtype),
            returned_retained_tokens=int(result[1]), post_generation_tensor_reads_only=True)
        with trace_path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        position += 1
        return result

    model.strategy_importance_allocation = allocate
    model.greed_select = greedy
    model.encode_images = encode
    model.generate = generate
